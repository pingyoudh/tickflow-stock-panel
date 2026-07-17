"""Point-in-time research panel construction for daily and minute data."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from app.backtest.engine import BacktestEngine
from app.quant.models import ResearchPanelSpec
from app.services.ext_data import ExtConfigStore
from app.tickflow.repository import KlineRepository

_MINUTE_INTERVALS = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60}


class ResearchPanelBuilder:
    """Build bounded panels while preserving market-time semantics."""

    def __init__(
        self, repo: KlineRepository, data_dir: Path, *, max_rows: int = 10_000_000
    ) -> None:
        self.repo = repo
        self.data_dir = data_dir
        self.max_rows = max_rows
        self.engine = BacktestEngine(repo)

    def estimate(self, spec: ResearchPanelSpec) -> dict[str, Any]:
        weekdays = sum(
            1 for i in range((spec.end - spec.start).days + 1)
            if (spec.start + timedelta(days=i)).weekday() < 5
        )
        if spec.symbols:
            symbol_count = len(set(spec.symbols))
        else:
            instruments = self.repo.get_instruments_asset(spec.asset_type)
            symbol_count = instruments["symbol"].n_unique() if "symbol" in instruments.columns else 0
        bars_per_day = 1 if spec.frequency == "1d" else 240 // _MINUTE_INTERVALS[spec.frequency]
        estimated_rows = int(weekdays * symbol_count * bars_per_day)
        missing: list[str] = []
        if symbol_count == 0:
            missing.append(f"{spec.asset_type} 标的列表为空")
        if spec.frequency != "1d":
            base = "kline_etf_minute" if spec.asset_type == "etf" else "kline_minute"
            if not any((self.data_dir / base).rglob("*.parquet")):
                missing.append("分钟行情数据不存在")
        max_rows = spec.max_rows or self.max_rows
        return {
            "estimated_rows": estimated_rows,
            "max_rows": max_rows,
            "allowed": estimated_rows <= max_rows and not missing,
            "symbol_count": symbol_count,
            "estimated_trading_days": weekdays,
            "missing_data": missing,
        }

    def build(self, spec: ResearchPanelSpec, *, keep_warmup: bool = False) -> pl.DataFrame:
        estimate = self.estimate(spec)
        max_rows = int(estimate["max_rows"])
        if estimate["estimated_rows"] > max_rows:
            raise ValueError(
                f"预计 {estimate['estimated_rows']:,} 行, 超过本地限制 {max_rows:,} 行"
            )
        panel = self._build_daily(spec) if spec.frequency == "1d" else self._build_minute(spec)
        if panel.height > max_rows:
            raise ValueError(f"实际面板 {panel.height:,} 行, 超过本地限制 {max_rows:,} 行")
        if spec.ext_datasets:
            panel = self._join_extensions(panel, spec)
        if not keep_warmup:
            time_col = "date" if "date" in panel.columns else "datetime"
            value = pl.col(time_col) if time_col == "date" else pl.col(time_col).dt.date()
            panel = panel.filter((value >= spec.start) & (value <= spec.end))
        if spec.fields:
            keys = [c for c in ["symbol", "date", "datetime"] if c in panel.columns]
            selected = [c for c in [*keys, *spec.fields] if c in panel.columns]
            panel = panel.select(list(dict.fromkeys(selected)))
        return panel

    def _build_daily(self, spec: ResearchPanelSpec) -> pl.DataFrame:
        warmup_start = spec.start - timedelta(days=max(10, spec.warmup * 2))
        if not spec.fields:
            return self.engine.load_panel(
                spec.symbols, warmup_start, spec.end, columns=None, asset_type=spec.asset_type
            ).sort(["symbol", "date"])

        # Historical enriched partitions may contain only adjusted OHLCV. Read a
        # narrow raw basis, then calculate only requested indicators that are absent.
        basis = ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]
        columns = list(dict.fromkeys(["symbol", "date", *basis, *spec.fields]))
        panel = self.engine.load_panel(
            spec.symbols, warmup_start, spec.end, columns=columns, asset_type=spec.asset_type
        ).sort(["symbol", "date"])
        missing = set(spec.fields) - set(panel.columns)
        if missing:
            from app.indicators.pipeline import compute_indicators

            panel = compute_indicators(panel, needed=missing)
        return panel

    def _build_minute(self, spec: ResearchPanelSpec) -> pl.DataFrame:
        panel = self.repo.get_minute_range(
            spec.symbols or [], spec.start, spec.end, asset_type=spec.asset_type
        )
        if panel.is_empty() or spec.frequency == "1m":
            return panel
        interval = _MINUTE_INTERVALS[spec.frequency]
        panel = panel.with_columns(
            pl.col("datetime").dt.date().alias("date"),
            pl.when(pl.col("datetime").dt.hour() < 12)
            .then(pl.lit("am"))
            .otherwise(pl.lit("pm"))
            .alias("_session"),
            (
                pl.col("datetime").dt.hour().cast(pl.Int32) * 60
                + pl.col("datetime").dt.minute().cast(pl.Int32)
                - pl.when(pl.col("datetime").dt.hour() < 12).then(570).otherwise(780)
            ).floordiv(interval).alias("_bucket"),
        )
        aggregations = [
            pl.col("datetime").last().alias("datetime"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
        ]
        if "volume" in panel.columns:
            aggregations.append(pl.col("volume").sum().alias("volume"))
        if "amount" in panel.columns:
            aggregations.append(pl.col("amount").sum().alias("amount"))
        return (
            panel.sort(["symbol", "datetime"])
            .group_by(["symbol", "date", "_session", "_bucket"], maintain_order=True)
            .agg(aggregations)
            .drop(["_session", "_bucket"])
            .sort(["symbol", "datetime"])
        )

    def _join_extensions(self, panel: pl.DataFrame, spec: ResearchPanelSpec) -> pl.DataFrame:
        store = ExtConfigStore(self.data_dir)
        result = panel
        for dataset_id in spec.ext_datasets:
            config = store.get(dataset_id)
            if config is None:
                raise ValueError(f"扩展数据集不存在: {dataset_id}")
            if config.mode == "snapshot":
                if spec.start != spec.end or spec.end != date.today():
                    raise ValueError(
                        f"扩展数据集 {dataset_id} 是覆盖式快照, 不能用于历史研究或训练"
                    )
                path = self.data_dir / "ext_data" / dataset_id / "part.parquet"
                if path.exists():
                    result = result.join(pl.read_parquet(path), on="symbol", how="left", suffix=f"_{dataset_id}")
                continue
            files = list((self.data_dir / "ext_data" / dataset_id / "timeseries").rglob("*.parquet"))
            if not files:
                continue
            ext = pl.scan_parquet(files, hive_partitioning=True).collect()
            if "date" not in ext.columns:
                raise ValueError(f"时序扩展数据集 {dataset_id} 缺少 date 分区字段")
            ext = ext.with_columns(pl.col("date").cast(pl.Date)).sort(["symbol", "date"])
            if "date" not in result.columns and "datetime" in result.columns:
                result = result.with_columns(pl.col("datetime").dt.date().alias("date"))
            result = result.sort(["symbol", "date"]).join_asof(
                ext, on="date", by="symbol", strategy="backward", suffix=f"_{dataset_id}", check_sortedness=False
            )
        return result
