"""Leakage-safe ML dataset and future excess-return labels."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from app.quant.factor_cache import FactorValueCache
from app.quant.factors import FactorRegistry
from app.quant.models import FactorDefinition, ModelSpec, ResearchPanelSpec
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


@dataclass
class MLSearchDataset:
    base_frame: pl.DataFrame
    definitions: dict[str, FactorDefinition]
    model_spec: ModelSpec
    calendar: list
    fingerprint: str
    input_file_fingerprint: str
    warnings: list[str]
    cache_events: list[dict]


class MLDatasetBuilder:
    def __init__(
        self,
        repo: KlineRepository,
        data_dir: Path,
        registry: FactorRegistry,
        *,
        max_rows: int = 10_000_000,
        factor_cache_max_bytes: int = 10 * 1024**3,
    ) -> None:
        self.repo = repo
        self.data_dir = data_dir
        self.registry = registry
        self.panels = ResearchPanelBuilder(repo, data_dir, max_rows=max_rows)
        self.factor_cache = FactorValueCache(
            data_dir, registry, max_bytes=factor_cache_max_bytes
        )

    def build(
        self,
        spec: ModelSpec,
        *,
        cancelled: threading.Event | None = None,
        require_complete_features: bool = True,
    ) -> MLDataset:
        self._check_cancelled(cancelled)
        definitions = [self.registry.get(name) for name in spec.features]
        unavailable = [
            f"{item.id}: {item.blocked_reason or item.compute_status}"
            for item in definitions
            if not item.enabled or item.compute_status != "ready"
        ]
        if unavailable:
            raise ValueError("存在不可用因子: " + "; ".join(unavailable))
        for definition in definitions:
            expected = spec.feature_versions.get(definition.id)
            if expected and definition.version != expected:
                raise ValueError(
                    f"因子 {definition.id} 版本不匹配: 请求 {expected}, 当前 {definition.version}"
                )
        warmup = max([120, *(factor.warmup for factor in definitions)])
        min_history_days = int(spec.universe_filters.get("min_history_days", 0) or 0)
        min_median_amount = float(
            spec.universe_filters.get("min_median_amount_20d", 0.0) or 0.0
        )
        warmup = max(warmup, min_history_days, 20 if min_median_amount > 0 else 0)
        direct_feature_fields = [
            factor.id for factor in definitions if factor.authoring_type == "builtin"
        ]
        fields = sorted({
            "open", "close", *direct_feature_fields,
            *(name for factor in definitions for name in factor.inputs),
            *(name for factor in definitions for name in factor.preprocess.neutralize),
            *(["amount"] if min_median_amount > 0 else []),
        })
        panel_spec = ResearchPanelSpec(
            asset_type=spec.asset_type,
            frequency="1d",
            symbols=spec.symbols,
            start=spec.start,
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
        eligible_panel = self._apply_universe_filters(panel, spec)
        for definition in definitions:
            self._check_cancelled(cancelled)
            source = (
                eligible_panel
                if self._has_cross_section_operator(definition)
                else panel
            )
            values = self.registry.evaluate(source, definition).select(
                ["symbol", "date", definition.id]
            )
            if definition.id in eligible_panel.columns:
                eligible_panel = eligible_panel.drop(definition.id)
            eligible_panel = eligible_panel.join(
                values, on=["symbol", "date"], how="left"
            )
        panel = eligible_panel.filter(
            (pl.col("date") >= spec.start) & (pl.col("date") <= spec.end)
        )
        for definition in definitions:
            panel = self._preprocess(panel, definition.id, definition)
        labeled, calendar = self._add_target(panel, spec)
        self._check_cancelled(cancelled)
        valid = labeled.filter(pl.all_horizontal([
            pl.col(name).cast(pl.Float64, strict=False).is_finite()
            for name in ["target", "forward_return", "benchmark_return"]
        ]))
        if require_complete_features:
            valid = valid.filter(
                pl.all_horizontal([
                    pl.col(name).cast(pl.Float64, strict=False).is_finite()
                    for name in spec.features
                ])
            )
        if valid.is_empty():
            raise ValueError("特征和标签过滤后没有有效样本")
        valid = valid.with_columns(
            (1.0 / pl.len().over("date")).cast(pl.Float64).alias("sample_weight")
        )
        warnings = self._dataset_warnings(spec)
        fingerprint = self._fingerprint(spec, definitions, valid)
        input_file_fingerprint = self.input_file_fingerprint(spec.asset_type)
        columns = ["symbol", "date", *spec.features, "target", "forward_return", "benchmark_return", "sample_weight"]
        return MLDataset(
            valid.select(columns).sort(["date", "symbol"]), spec.features, calendar,
            fingerprint, input_file_fingerprint, warnings,
        )

    def prepare_search(
        self,
        spec: ModelSpec,
        definitions: list[FactorDefinition],
        *,
        cancelled: threading.Event | None = None,
        on_factor=None,
        on_stage=None,
    ) -> MLSearchDataset:
        """Build labels once and persist raw factor values one column at a time."""
        self._check_cancelled(cancelled)
        unavailable = [
            f"{item.id}: {item.blocked_reason or item.compute_status}"
            for item in definitions
            if not item.enabled or item.compute_status != "ready"
        ]
        if unavailable:
            raise ValueError("存在不可用因子: " + "; ".join(unavailable))
        min_history_days = int(spec.universe_filters.get("min_history_days", 0) or 0)
        min_median_amount = float(
            spec.universe_filters.get("min_median_amount_20d", 0.0) or 0.0
        )
        warmup = max(
            [120, min_history_days, 20 if min_median_amount > 0 else 0,
             *(item.warmup for item in definitions)]
        )
        direct_fields = [
            item.id for item in definitions if item.authoring_type == "builtin"
        ]
        fields = sorted({
            "open", "close", *direct_fields,
            *(name for item in definitions for name in item.inputs),
            *(["amount"] if min_median_amount > 0 else []),
        })
        if on_stage is not None:
            on_stage("正在读取训练区间日行情")
        panel = self.panels.build(ResearchPanelSpec(
            asset_type=spec.asset_type,
            frequency="1d",
            symbols=spec.symbols,
            start=spec.start,
            end=spec.end,
            fields=fields,
            warmup=warmup,
        ), keep_warmup=True)
        if panel.is_empty():
            raise ValueError("训练区间没有可用日行情")
        required = {"symbol", "date", "open", "close"}
        if not required.issubset(panel.columns):
            raise ValueError(
                f"研究面板缺少标签字段: {sorted(required - set(panel.columns))}"
            )

        if on_stage is not None:
            on_stage(f"行情面板完成 · {panel.height:,} 行")
        eligible_panel = self._apply_universe_filters(panel, spec)
        source_state = self.factor_cache.source_state(
            spec.asset_type,
            panel["date"].min(),
            panel["date"].max(),
        )
        cache_events: list[dict] = []
        for index, definition in enumerate(definitions):
            self._check_cancelled(cancelled)
            source = (
                eligible_panel
                if self._has_cross_section_operator(definition)
                else panel
            )
            try:
                _, event = self.factor_cache.get_or_compute(
                    spec,
                    definition,
                    source,
                    load_values=False,
                    source_state=source_state,
                )
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                raise RuntimeError(
                    f"因子计算失败: {definition.name} ({definition.id}) · "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            cache_events.append(event)
            if on_factor is not None:
                on_factor(index + 1, len(definitions), event)

        if on_stage is not None:
            on_stage("因子缓存完成, 正在生成训练标签")
        eligible = eligible_panel.filter(
            (pl.col("date") >= spec.start) & (pl.col("date") <= spec.end)
        )
        labeled, calendar = self._add_target(eligible, spec)
        valid = labeled.filter(pl.all_horizontal([
            pl.col(name).cast(pl.Float64, strict=False).is_finite()
            for name in ["target", "forward_return", "benchmark_return"]
        ]))
        if valid.is_empty():
            raise ValueError("ETF/股票池过滤和标签计算后没有有效样本")
        valid = valid.with_columns(
            (1.0 / pl.len().over("date")).cast(pl.Float64).alias("sample_weight")
        )
        return MLSearchDataset(
            base_frame=valid.select([
                "symbol", "date", "target", "forward_return",
                "benchmark_return", "sample_weight",
            ]).sort(["date", "symbol"]),
            definitions={item.id: item for item in definitions},
            model_spec=spec,
            calendar=calendar,
            fingerprint=self._fingerprint(spec, definitions, valid),
            input_file_fingerprint=self.input_file_fingerprint(spec.asset_type),
            warnings=self._dataset_warnings(spec),
            cache_events=cache_events,
        )

    def attach_search_features(
        self,
        dataset: MLSearchDataset,
        factor_ids: list[str],
        frame: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        result = frame if frame is not None else dataset.base_frame
        if result.is_empty() or not factor_ids:
            return result
        start = result["date"].min()
        end = result["date"].max()
        for factor_id in factor_ids:
            definition = dataset.definitions[factor_id]
            values = self.factor_cache.load(
                dataset.model_spec, definition, start=start, end=end
            ).rename({"value": factor_id})
            result = result.join(values, on=["symbol", "date"], how="left")
            result = self._preprocess(result, factor_id, definition)
        return result

    def build_latest_features(self, spec: ModelSpec) -> pl.DataFrame:
        definitions = [self.registry.get(name) for name in spec.features]
        unavailable = [
            f"{item.id}: {item.blocked_reason or item.compute_status}"
            for item in definitions
            if not item.enabled or item.compute_status != "ready"
        ]
        if unavailable:
            raise ValueError("存在不可用因子: " + "; ".join(unavailable))
        min_history_days = int(spec.universe_filters.get("min_history_days", 0) or 0)
        min_median_amount = float(
            spec.universe_filters.get("min_median_amount_20d", 0.0) or 0.0
        )
        warmup = max(
            [120, min_history_days, 20 if min_median_amount > 0 else 0,
             *(factor.warmup for factor in definitions)]
        )
        direct_feature_fields = [
            factor.id for factor in definitions if factor.authoring_type == "builtin"
        ]
        fields = sorted({
            "open", "close", *direct_feature_fields,
            *(name for factor in definitions for name in factor.inputs),
            *(name for factor in definitions for name in factor.preprocess.neutralize),
            *(["amount"] if min_median_amount > 0 else []),
        })
        panel = self.panels.build(ResearchPanelSpec(
            asset_type=spec.asset_type, frequency="1d", symbols=spec.symbols,
            start=spec.end, end=spec.end,
            fields=fields, warmup=warmup,
        ), keep_warmup=True)
        eligible_panel = self._apply_universe_filters(panel, spec)
        for definition in definitions:
            source = (
                eligible_panel
                if self._has_cross_section_operator(definition)
                else panel
            )
            values = self.registry.evaluate(source, definition).select(
                ["symbol", "date", definition.id]
            )
            if definition.id in eligible_panel.columns:
                eligible_panel = eligible_panel.drop(definition.id)
            eligible_panel = eligible_panel.join(
                values, on=["symbol", "date"], how="left"
            )
        panel = eligible_panel
        for definition in definitions:
            panel = self._preprocess(panel, definition.id, definition)
        if panel.is_empty():
            return panel
        latest = panel["date"].max()
        return panel.filter(pl.col("date") == latest).select(["symbol", "date", *spec.features])

    @staticmethod
    def _check_cancelled(cancelled: threading.Event | None) -> None:
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("训练已取消")

    @staticmethod
    def _has_cross_section_operator(definition: FactorDefinition) -> bool:
        if set(definition.operators) & {"Rank", "ZScore", "Scale"}:
            return True

        def visit(node: Any) -> bool:
            if isinstance(node, dict):
                if node.get("op") == "rank":
                    return True
                return any(visit(value) for value in node.values())
            if isinstance(node, list):
                return any(visit(value) for value in node)
            return False

        return visit(definition.expression)

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
        numeric = (
            pl.col(column).cast(pl.Float64, strict=False) * definition.direction
        )
        value = pl.when(numeric.is_finite()).then(numeric).otherwise(None)
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
        return panel.with_columns(
            pl.when(pl.col(column).is_finite())
            .then(pl.col(column))
            .otherwise(None)
            .alias(column)
        )

    @staticmethod
    def _apply_universe_filters(panel: pl.DataFrame, spec: ModelSpec) -> pl.DataFrame:
        min_history_days = int(spec.universe_filters.get("min_history_days", 0) or 0)
        min_median_amount = float(
            spec.universe_filters.get("min_median_amount_20d", 0.0) or 0.0
        )
        if min_history_days <= 0 and min_median_amount <= 0:
            return panel
        ordered = panel.sort(["symbol", "date"])
        predicates: list[pl.Expr] = []
        helper_columns: list[str] = []
        if min_history_days > 0:
            helper_columns.append("__history_days")
            ordered = ordered.with_columns(
                pl.col("date").cum_count().over("symbol").alias("__history_days")
            )
            predicates.append(pl.col("__history_days") >= min_history_days)
        if min_median_amount > 0:
            if "amount" not in ordered.columns:
                raise ValueError("ETF 流动性过滤需要 amount 成交额字段")
            helper_columns.append("__median_amount_20d")
            ordered = ordered.with_columns(
                pl.col("amount")
                .cast(pl.Float64, strict=False)
                .rolling_median(window_size=20, min_samples=20)
                .over("symbol")
                .alias("__median_amount_20d")
            )
            predicates.append(pl.col("__median_amount_20d") >= min_median_amount)
        return ordered.filter(pl.all_horizontal(predicates)).drop(helper_columns)

    @staticmethod
    def _dataset_warnings(spec: ModelSpec) -> list[str]:
        warnings: list[str] = []
        if not spec.universe_filters.get("point_in_time_constituents", False):
            warnings.append("缺少历史成分股/退市证券口径, 结果可能存在幸存者偏差")
        if spec.asset_type == "etf" and spec.symbols is None:
            warnings.extend([
                "ETF 全市场包含股票、债券、商品及跨境等不同资产类别",
                "缺少历史清盘 ETF 完整口径, 结果可能存在幸存者偏差",
            ])
        return warnings

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
