"""Leakage-safe ML dataset and future excess-return labels."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from app.quant.factors import FactorRegistry
from app.quant.models import ModelSpec, ResearchPanelSpec
from app.quant.panel import ResearchPanelBuilder
from app.tickflow.repository import KlineRepository


@dataclass
class MLDataset:
    frame: pl.DataFrame
    feature_columns: list[str]
    calendar: list
    fingerprint: str
    input_file_fingerprint: str
    warnings: list[str]


class MLDatasetBuilder:
    def __init__(
        self,
        repo: KlineRepository,
        data_dir: Path,
        registry: FactorRegistry,
        *,
        max_rows: int = 10_000_000,
    ) -> None:
        self.repo = repo
        self.data_dir = data_dir
        self.registry = registry
        self.panels = ResearchPanelBuilder(repo, data_dir, max_rows=max_rows)

    def build(
        self,
        spec: ModelSpec,
        *,
        cancelled: threading.Event | None = None,
        require_complete_features: bool = True,
    ) -> MLDataset:
        self._check_cancelled(cancelled)
        definitions = [self.registry.get(name) for name in spec.features]
        for definition in definitions:
            expected = spec.feature_versions.get(definition.id)
            if expected and definition.version != expected:
                raise ValueError(
                    f"因子 {definition.id} 版本不匹配: 请求 {expected}, 当前 {definition.version}"
                )
        warmup = max([120, *(factor.warmup for factor in definitions)])
        fields = sorted({
            "open", "close", *spec.features,
            *(name for factor in definitions for name in factor.inputs),
            *(name for factor in definitions for name in factor.preprocess.neutralize),
        })
        panel_spec = ResearchPanelSpec(
            asset_type=spec.asset_type,
            frequency="1d",
            symbols=spec.symbols,
            start=spec.start - timedelta(days=warmup * 2),
            end=spec.end,
            fields=fields,
            warmup=warmup,
        )
        panel = self.panels.build(panel_spec, keep_warmup=True)
        self._check_cancelled(cancelled)
        if panel.is_empty():
            raise ValueError("训练区间没有可用日行情")
        required = {"symbol", "date", "open", "close"}
        if not required.issubset(panel.columns):
            raise ValueError(f"研究面板缺少标签字段: {sorted(required - set(panel.columns))}")
        for definition in definitions:
            self._check_cancelled(cancelled)
            panel = self.registry.evaluate(panel, definition)
            panel = self._preprocess(panel, definition.id, definition)
        panel = panel.filter((pl.col("date") >= spec.start) & (pl.col("date") <= spec.end))
        labeled, calendar = self._add_target(panel, spec)
        self._check_cancelled(cancelled)
        valid = labeled.filter(pl.col("target").is_not_null())
        if require_complete_features:
            valid = valid.filter(
                pl.all_horizontal([pl.col(name).is_not_null() for name in spec.features])
            )
        if valid.is_empty():
            raise ValueError("特征和标签过滤后没有有效样本")
        valid = valid.with_columns(
            (1.0 / pl.len().over("date")).cast(pl.Float64).alias("sample_weight")
        )
        warnings: list[str] = []
        if not spec.universe_filters.get("point_in_time_constituents", False):
            warnings.append("缺少历史成分股/退市证券口径, 结果可能存在幸存者偏差")
        fingerprint = self._fingerprint(spec, definitions, valid)
        input_file_fingerprint = self.input_file_fingerprint(spec.asset_type)
        columns = ["symbol", "date", *spec.features, "target", "forward_return", "benchmark_return", "sample_weight"]
        return MLDataset(
            valid.select(columns).sort(["date", "symbol"]), spec.features, calendar,
            fingerprint, input_file_fingerprint, warnings,
        )

    def build_latest_features(self, spec: ModelSpec) -> pl.DataFrame:
        definitions = [self.registry.get(name) for name in spec.features]
        warmup = max([120, *(factor.warmup for factor in definitions)])
        fields = sorted({
            "open", "close", *spec.features,
            *(name for factor in definitions for name in factor.inputs),
            *(name for factor in definitions for name in factor.preprocess.neutralize),
        })
        panel = self.panels.build(ResearchPanelSpec(
            asset_type=spec.asset_type, frequency="1d", symbols=spec.symbols,
            start=spec.end - timedelta(days=warmup * 2), end=spec.end,
            fields=fields, warmup=warmup,
        ), keep_warmup=True)
        for definition in definitions:
            panel = self.registry.evaluate(panel, definition)
            panel = self._preprocess(panel, definition.id, definition)
        if panel.is_empty():
            return panel
        latest = panel["date"].max()
        return panel.filter(pl.col("date") == latest).select(["symbol", "date", *spec.features])

    @staticmethod
    def _check_cancelled(cancelled: threading.Event | None) -> None:
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("训练已取消")

    def _add_target(self, panel: pl.DataFrame, spec: ModelSpec) -> tuple[pl.DataFrame, list]:
        benchmark = pl.DataFrame()
        if spec.target.benchmark_mode == "index":
            benchmark = self.repo.get_index_daily(
                spec.target.benchmark_symbol or "000300.SH", spec.start,
                spec.end + timedelta(days=spec.target.horizon * 3 + 10),
                columns=["date", "open", "close"],
            ).sort("date")
            if benchmark.is_empty():
                raise ValueError(f"基准 {spec.target.benchmark_symbol} 没有可用历史数据")
            calendar = benchmark["date"].to_list()
        else:
            calendar = sorted(panel["date"].unique().to_list())
        horizon = spec.target.horizon
        mapping = pl.DataFrame({
            "date": calendar[: max(0, len(calendar) - horizon)],
            "_entry_date": calendar[1: len(calendar) - horizon + 1],
            "_exit_date": calendar[horizon:],
        })
        base = panel.join(mapping, on="date", how="inner")
        prices = panel.select(["symbol", "date", "open", "close"])
        entry = prices.select([
            "symbol", pl.col("date").alias("_entry_date"), pl.col("open").alias("_entry_open")
        ])
        exit_prices = prices.select([
            "symbol", pl.col("date").alias("_exit_date"), pl.col("close").alias("_exit_close")
        ])
        base = base.join(entry, on=["symbol", "_entry_date"], how="left").join(
            exit_prices, on=["symbol", "_exit_date"], how="left"
        ).with_columns(
            (pl.col("_exit_close") / pl.col("_entry_open") - 1).alias("forward_return")
        )
        if spec.target.benchmark_mode == "index":
            bench_entry = benchmark.select([
                pl.col("date").alias("_entry_date"), pl.col("open").alias("_bench_entry")
            ])
            bench_exit = benchmark.select([
                pl.col("date").alias("_exit_date"), pl.col("close").alias("_bench_exit")
            ])
            base = base.join(bench_entry, on="_entry_date", how="left").join(
                bench_exit, on="_exit_date", how="left"
            ).with_columns(
                (pl.col("_bench_exit") / pl.col("_bench_entry") - 1).alias("benchmark_return")
            )
        else:
            base = base.with_columns(
                pl.col("forward_return").mean().over("date").alias("benchmark_return")
            )
        return base.with_columns(
            (pl.col("forward_return") - pl.col("benchmark_return")).alias("target")
        ), calendar

    def _preprocess(self, panel: pl.DataFrame, column: str, definition) -> pl.DataFrame:
        settings = definition.preprocess
        value = pl.col(column).cast(pl.Float64, strict=False) * definition.direction
        panel = panel.with_columns(value.alias(column))
        if settings.winsorize_mad is not None:
            median_col = f"__{column}_median"
            mad_col = f"__{column}_mad"
            panel = panel.with_columns(pl.col(column).median().over("date").alias(median_col))
            panel = panel.with_columns(
                (pl.col(column) - pl.col(median_col)).abs().median().over("date").alias(mad_col)
            )
            width = pl.col(mad_col) * settings.winsorize_mad
            panel = panel.with_columns(
                pl.when(pl.col(column) < pl.col(median_col) - width).then(pl.col(median_col) - width)
                .when(pl.col(column) > pl.col(median_col) + width).then(pl.col(median_col) + width)
                .otherwise(pl.col(column)).alias(column)
            ).drop([median_col, mad_col])
        if settings.neutralize:
            panel = self._neutralize(panel, column, settings.neutralize)
        if settings.normalize == "rank":
            panel = panel.with_columns(
                (pl.col(column).rank(method="average").over("date") / pl.len().over("date")).alias(column)
            )
        elif settings.normalize == "zscore":
            panel = panel.with_columns(
                ((pl.col(column) - pl.col(column).mean().over("date"))
                 / pl.col(column).std().over("date")).alias(column)
            )
        return panel

    @staticmethod
    def _neutralize(panel: pl.DataFrame, column: str, controls: list[str]) -> pl.DataFrame:
        missing = [name for name in controls if name not in panel.columns]
        if missing:
            raise ValueError(f"中性化字段不存在: {missing}")
        pieces: list[pl.DataFrame] = []
        for group in panel.partition_by("date", maintain_order=True):
            y = group[column].to_numpy().astype(float)
            encoded: list[np.ndarray] = []
            control_valid = np.ones(len(group), dtype=bool)
            for name in controls:
                series = group[name]
                if series.dtype.is_numeric():
                    values = series.cast(pl.Float64, strict=False).to_numpy()
                    encoded.append(values)
                    control_valid &= np.isfinite(values)
                else:
                    values = series.cast(pl.Utf8, strict=False).to_numpy()
                    control_valid &= np.array([value is not None for value in values])
                    categories = sorted({value for value in values if value is not None})
                    encoded.extend([
                        np.array([float(value == category) for value in values])
                        for category in categories[1:]
                    ])
            xs = np.column_stack(encoded) if encoded else np.empty((len(group), 0))
            valid = np.isfinite(y) & control_valid & np.all(np.isfinite(xs), axis=1)
            residual = np.full(len(group), np.nan)
            if valid.sum() > xs.shape[1] + 1:
                design = np.column_stack([np.ones(valid.sum()), xs[valid]])
                residual[valid] = y[valid] - design @ np.linalg.lstsq(design, y[valid], rcond=None)[0]
            pieces.append(group.with_columns(pl.Series(column, residual)))
        return pl.concat(pieces) if pieces else panel

    def _fingerprint(self, spec: ModelSpec, definitions: list, frame: pl.DataFrame) -> str:
        payload = {
            "spec": spec.model_dump(mode="json"),
            "factors": {item.id: item.version for item in definitions},
            "input_files": self.input_file_fingerprint(spec.asset_type),
            "rows": frame.height,
            "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def input_file_fingerprint(self, asset_type: str) -> str:
        inputs: list[tuple[str, int, int]] = []
        base = "kline_etf_enriched" if asset_type == "etf" else "kline_daily_enriched"
        for path in sorted((self.data_dir / base).rglob("*.parquet")):
            stat = path.stat()
            inputs.append((str(path.relative_to(self.data_dir)), stat.st_size, stat.st_mtime_ns))
        return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
