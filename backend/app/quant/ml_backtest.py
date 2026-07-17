"""Leakage-safe target-weight backtests over persisted OOS predictions."""
from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from app.indicators.pipeline import compute_indicators, compute_limit_signals
from app.quant.experiments import ExperimentStore
from app.quant.model_registry import ModelRegistry
from app.quant.models import MLBacktestSpec
from app.tickflow.repository import KlineRepository

ProgressCallback = Callable[[float, str], None]


class MLBacktestService:
    def __init__(
        self,
        repo: KlineRepository,
        data_dir: Path,
        models: ModelRegistry,
        experiments: ExperimentStore,
    ) -> None:
        self.repo = repo
        self.data_dir = data_dir
        self.models = models
        self.experiments = experiments

    def find_training_run(self, version: str):
        metadata = self.models.get(version)
        source_run_id = metadata.get("source_run_id")
        if source_run_id:
            try:
                source = self.experiments.get(source_run_id)
                if source.kind == "ml_training" and source.status == "completed":
                    return source
            except ValueError:
                pass
        for manifest in self.experiments.list():
            if (
                manifest.kind == "ml_training"
                and manifest.status == "completed"
                and manifest.result.get("model_version") == version
            ):
                return manifest
        raise ValueError("模型没有可关联的 OOS 训练实验, 不能进行无泄漏回测")

    def list_runs(self, version: str) -> list:
        return [
            item for item in self.experiments.list()
            if item.kind == "ml_backtest" and item.spec.get("model_version") == version
        ]

    def run(
        self,
        spec: MLBacktestSpec,
        run_dir: Path,
        progress: ProgressCallback,
        cancelled: threading.Event,
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )
        metadata = self.models.get(spec.model_version)
        training_run = self.find_training_run(spec.model_version)
        oos_path = self.experiments.root / training_run.run_id / "oos_predictions.parquet"
        if not oos_path.exists():
            raise ValueError("训练实验缺少 oos_predictions.parquet")
        progress(0.05, "正在读取严格样本外预测")
        oos = pl.read_parquet(oos_path)
        self._validate_oos(oos, training_run.result.get("folds", []))
        oos = (
            oos.sort(["date", "symbol", "fold"])
            .unique(["date", "symbol"], keep="last")
            .sort(["date", "symbol"])
        )
        self._check_cancelled(cancelled)

        symbols = oos["symbol"].unique().to_list()
        signal_start = oos["date"].min()
        signal_end = oos["date"].max()
        progress(0.12, "正在加载 OOS 区间可交易行情")
        panel = self._load_panel(
            metadata["spec"]["asset_type"], symbols, signal_start, signal_end + timedelta(days=14)
        )
        if panel.is_empty():
            raise ValueError("OOS 区间没有可交易行情")
        self._check_cancelled(cancelled)

        horizon = int(metadata["spec"]["target"]["horizon"])
        rebalance_days = spec.rebalance_days or horizon
        targets = self._build_targets(oos, panel, spec, rebalance_days)
        if targets.is_empty():
            raise ValueError("OOS 预测无法生成有效调仓目标")
        targets.write_parquet(run_dir / "target_weights.parquet")
        progress(0.22, "正在执行 T+1 目标权重撮合")
        result = self._simulate(panel, targets, spec, metadata, progress, cancelled)
        self._write_artifacts(run_dir, result)
        result["model_version"] = spec.model_version
        result["source_run_id"] = training_run.run_id
        result["oos_only"] = True
        result["rebalance_days"] = rebalance_days
        result["warnings"] = list(dict.fromkeys([
            *metadata.get("training", {}).get("warnings", []),
            *result.get("warnings", []),
        ]))
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        progress(1.0, "OOS 组合回测完成")
        return result

    @staticmethod
    def _validate_oos(oos: pl.DataFrame, folds: list[dict[str, Any]]) -> None:
        required = {
            "symbol", "date", "fold", "prediction", "rank", "target", "forward_return"
        }
        if not required.issubset(oos.columns):
            raise ValueError(f"OOS 文件缺少列: {sorted(required - set(oos.columns))}")
        if oos.is_empty() or not folds:
            raise ValueError("OOS 文件或 Walk-forward 折为空")
        bounds = {int(item["index"]): (date.fromisoformat(item["test_start"]),
                                       date.fromisoformat(item["test_end"])) for item in folds}
        for group in oos.partition_by("fold"):
            fold = int(group["fold"][0])
            if fold not in bounds:
                raise ValueError(f"OOS 文件包含未知测试折: {fold}")
            start, end = bounds[fold]
            if group["date"].min() < start or group["date"].max() > end:
                raise ValueError(f"第 {fold} 折包含测试区间之外的预测, 已拒绝回测")

    def _load_panel(
        self, asset_type: str, symbols: list[str], start: date, end: date
    ) -> pl.DataFrame:
        columns = [
            "symbol", "date", "open", "high", "low", "close", "volume", "amount",
            "raw_close", "raw_high",
        ]
        if asset_type == "stock":
            panel = self.repo.get_daily_batch(symbols, start, end, columns=columns)
        else:
            frames = [
                self.repo.get_daily_asset(asset_type, symbol, start, end, columns=columns)
                for symbol in symbols
            ]
            panel = pl.concat([item for item in frames if not item.is_empty()], how="diagonal_relaxed") \
                if any(not item.is_empty() for item in frames) else pl.DataFrame()
        if panel.is_empty():
            return panel
        panel = panel.sort(["symbol", "date"])
        instruments = self.repo.get_instruments_asset(asset_type)
        if asset_type == "stock" and {"raw_close", "raw_high"}.issubset(panel.columns):
            panel = compute_indicators(panel, needed={"change_pct", "vol_ratio_5d"})
            panel = compute_limit_signals(panel, instruments)
        elif not instruments.is_empty() and "name" in instruments.columns:
            panel = panel.join(
                instruments.select(["symbol", "name"]).unique("symbol"), on="symbol", how="left"
            )
        for name in ("signal_limit_up", "signal_limit_down"):
            if name not in panel.columns:
                panel = panel.with_columns(pl.lit(False).alias(name))
        if "name" not in panel.columns:
            panel = panel.with_columns(pl.lit("").alias("name"))
        return panel.sort(["date", "symbol"])

    @staticmethod
    def _build_targets(
        oos: pl.DataFrame, panel: pl.DataFrame, spec: MLBacktestSpec, rebalance_days: int
    ) -> pl.DataFrame:
        calendar = sorted(panel["date"].unique().to_list())
        oos_start, oos_end = oos["date"].min(), oos["date"].max()
        signal_calendar = [day for day in calendar if oos_start <= day <= oos_end]
        prediction_dates = set(oos["date"].unique().to_list())
        rows: list[dict[str, Any]] = []
        for offset in range(0, len(signal_calendar), rebalance_days):
            signal_date = signal_calendar[offset]
            if offset + 1 >= len(signal_calendar) and signal_date == calendar[-1]:
                continue
            calendar_index = calendar.index(signal_date)
            if calendar_index + 1 >= len(calendar):
                continue
            execution_date = calendar[calendar_index + 1]
            if signal_date not in prediction_dates:
                rows.append({
                    "signal_date": signal_date, "execution_date": execution_date,
                    "symbol": "__CASH__", "weight": 0.0, "score": 0.0,
                })
                continue
            ranked = oos.filter(pl.col("date") == signal_date).sort("rank", descending=True).head(spec.top_n)
            if ranked.is_empty():
                continue
            if spec.weighting == "score":
                scores = ranked["prediction"].to_numpy().astype(float)
                scores = scores - np.min(scores) + 1e-9
                weights = scores / scores.sum()
            else:
                weights = np.full(ranked.height, 1.0 / ranked.height)
            for index, row in enumerate(ranked.iter_rows(named=True)):
                rows.append({
                    "signal_date": signal_date, "execution_date": execution_date,
                    "symbol": row["symbol"], "weight": float(weights[index]),
                    "score": float(row["prediction"]),
                })
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def _simulate(
        self,
        panel: pl.DataFrame,
        targets: pl.DataFrame,
        spec: MLBacktestSpec,
        metadata: dict[str, Any],
        progress: ProgressCallback,
        cancelled: threading.Event,
    ) -> dict[str, Any]:
        rows_by_date: dict[date, dict[str, dict[str, Any]]] = {}
        for group in panel.partition_by("date", maintain_order=True):
            rows_by_date[group["date"][0]] = {
                row["symbol"]: row for row in group.iter_rows(named=True)
            }
        targets_by_date: dict[date, dict[str, float]] = {}
        scores_by_date: dict[date, dict[str, float]] = {}
        target_rows = targets.filter(pl.col("symbol") != "__CASH__")
        for group in target_rows.partition_by("execution_date", maintain_order=True):
            day = group["execution_date"][0]
            targets_by_date[day] = dict(zip(group["symbol"], group["weight"], strict=True))
            scores_by_date[day] = dict(zip(group["symbol"], group["score"], strict=True))
        for day in targets.filter(pl.col("symbol") == "__CASH__")["execution_date"].to_list():
            targets_by_date[day] = {}
            scores_by_date[day] = {}

        dates = sorted(rows_by_date)
        first_execution = min(targets["execution_date"])
        dates = [day for day in dates if day >= first_execution]
        cash = float(spec.initial_capital)
        positions: dict[str, dict[str, Any]] = {}
        last_close: dict[str, float] = {}
        buy_rate = spec.commission_pct + spec.slippage_bps / 10_000
        sell_rate = spec.commission_pct + spec.stamp_tax_pct + spec.slippage_bps / 10_000
        equity_rows: list[dict[str, Any]] = []
        holding_rows: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        total_cost = 0.0
        total_turnover = 0.0

        for day_index, day in enumerate(dates):
            self._check_cancelled(cancelled)
            market = rows_by_date[day]
            for symbol, row in market.items():
                value = self._number(row.get("close"))
                if value > 0:
                    last_close[symbol] = value
            if day in targets_by_date:
                desired = targets_by_date[day]
                scores = scores_by_date[day]
                open_equity = cash + sum(
                    position["shares"] * self._trade_price(market.get(symbol), last_close.get(symbol, 0.0), "open")
                    for symbol, position in positions.items()
                )
                traded_value = 0.0
                for symbol in list(positions):
                    position = positions[symbol]
                    row = market.get(symbol)
                    price = self._trade_price(row, last_close.get(symbol, 0.0), "open")
                    target_value = open_equity * desired.get(symbol, 0.0)
                    current_value = position["shares"] * price
                    if current_value <= target_value + price * 100:
                        continue
                    shares = position["shares"] if target_value <= 0 else (
                        math.floor((current_value - target_value) / price / 100) * 100
                    )
                    if shares <= 0 or not self._tradable(row, "sell"):
                        continue
                    gross = shares * price
                    cost = gross * sell_rate
                    cash += gross - cost
                    total_cost += cost
                    traded_value += gross
                    pnl = (price - position["avg_price"]) * shares - cost
                    trades.append({
                        "symbol": symbol, "name": position["name"], "side": "sell",
                        "date": day, "price": price, "shares": shares,
                        "gross_value": gross, "cost": cost, "pnl": pnl,
                        "signal_date": position.get("last_signal_date"),
                    })
                    position["shares"] -= shares
                    if position["shares"] <= 0:
                        positions.pop(symbol)
                for symbol, weight in sorted(desired.items(), key=lambda item: item[1], reverse=True):
                    row = market.get(symbol)
                    price = self._trade_price(row, 0.0, "open")
                    if price <= 0 or not self._tradable(row, "buy"):
                        continue
                    position = positions.get(symbol)
                    current_value = (position["shares"] * price) if position else 0.0
                    target_value = open_equity * weight
                    shares = math.floor(max(0.0, target_value - current_value) / price / 100) * 100
                    affordable = math.floor(cash / (price * (1 + buy_rate)) / 100) * 100
                    shares = min(shares, affordable)
                    if shares <= 0:
                        continue
                    gross = shares * price
                    cost = gross * buy_rate
                    cash -= gross + cost
                    total_cost += cost
                    traded_value += gross
                    if position:
                        old_value = position["avg_price"] * position["shares"]
                        position["shares"] += shares
                        position["avg_price"] = (old_value + gross) / position["shares"]
                        position["last_signal_date"] = day
                    else:
                        positions[symbol] = {
                            "shares": shares, "avg_price": price,
                            "entry_date": day, "name": str((row or {}).get("name") or ""),
                            "last_signal_date": day,
                        }
                    trades.append({
                        "symbol": symbol, "name": str((row or {}).get("name") or ""),
                        "side": "buy", "date": day, "price": price, "shares": shares,
                        "gross_value": gross, "cost": cost, "pnl": None,
                        "score": scores.get(symbol),
                    })
                total_turnover += traded_value / max(open_equity, 1.0)

            close_equity = cash + sum(
                position["shares"] * last_close.get(symbol, position["avg_price"])
                for symbol, position in positions.items()
            )
            equity_rows.append({"date": day, "value": close_equity, "cash": cash})
            for symbol, position in positions.items():
                market_value = position["shares"] * last_close.get(symbol, position["avg_price"])
                holding_rows.append({
                    "date": day, "symbol": symbol, "name": position["name"],
                    "shares": position["shares"], "market_value": market_value,
                    "weight": market_value / max(close_equity, 1.0),
                })
            if day_index % 20 == 0:
                progress(0.22 + 0.58 * (day_index + 1) / max(1, len(dates)),
                         f"正在撮合第 {day_index + 1}/{len(dates)} 个交易日")

        equity = pl.DataFrame(equity_rows)
        index_curve = self._index_benchmark(metadata, dates, spec.initial_capital)
        universe_curve = self._universe_benchmark(panel, dates, spec.initial_capital)
        equity = equity.join(index_curve, on="date", how="left").join(
            universe_curve, on="date", how="left"
        ).with_columns(
            pl.col("index_benchmark").forward_fill(), pl.col("universe_benchmark").forward_fill()
        )
        values = equity["value"].to_numpy().astype(float)
        peaks = np.maximum.accumulate(values)
        equity = equity.with_columns(pl.Series("drawdown", values / peaks - 1))
        metrics = self._metrics(equity, spec.initial_capital, total_cost, total_turnover, trades)
        return {
            "metrics": metrics,
            "equity_curve": equity.to_dicts(),
            "holdings": holding_rows,
            "trades": trades,
            "warnings": [],
        }

    def _index_benchmark(
        self, metadata: dict[str, Any], dates: list[date], initial_capital: float
    ) -> pl.DataFrame:
        symbol = metadata["spec"]["target"].get("benchmark_symbol")
        if not symbol or not dates:
            return pl.DataFrame({"date": dates, "index_benchmark": [initial_capital] * len(dates)})
        frame = self.repo.get_index_daily(symbol, dates[0], dates[-1], columns=["date", "close"])
        if frame.is_empty():
            return pl.DataFrame({"date": dates, "index_benchmark": [initial_capital] * len(dates)})
        first = float(frame["close"][0])
        return frame.select("date", (pl.col("close") / first * initial_capital).alias("index_benchmark"))

    @staticmethod
    def _universe_benchmark(
        panel: pl.DataFrame, dates: list[date], initial_capital: float
    ) -> pl.DataFrame:
        frame = panel.filter(pl.col("date").is_in(dates)).sort(["symbol", "date"]).with_columns(
            pl.col("close").pct_change().over("symbol").alias("_return")
        ).group_by("date").agg(pl.col("_return").mean().fill_null(0).alias("_return")).sort("date")
        values = initial_capital * np.cumprod(1 + frame["_return"].to_numpy())
        return frame.select("date").with_columns(pl.Series("universe_benchmark", values))

    @staticmethod
    def _metrics(
        equity: pl.DataFrame,
        initial_capital: float,
        total_cost: float,
        total_turnover: float,
        trades: list[dict[str, Any]],
    ) -> dict[str, Any]:
        values = equity["value"].to_numpy().astype(float)
        returns = np.diff(values) / values[:-1] if len(values) > 1 else np.array([])
        years = max(len(values) / 252, 1 / 252)
        total_return = float(values[-1] / initial_capital - 1) if len(values) else 0.0
        annual_return = float(
            max(values[-1] / initial_capital, 1e-12) ** (1 / years) - 1
        ) if len(values) else 0.0
        volatility = float(np.std(returns, ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)) \
            if len(returns) > 1 and np.std(returns, ddof=1) > 1e-12 else 0.0
        downside = returns[returns < 0]
        sortino = float(np.mean(returns) / np.std(downside, ddof=1) * np.sqrt(252)) \
            if len(downside) > 1 and np.std(downside, ddof=1) > 1e-12 else 0.0
        max_drawdown = float(equity["drawdown"].min()) if len(equity) else 0.0
        calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < -1e-12 else 0.0
        index_end = float(equity["index_benchmark"].drop_nulls()[-1]) if equity["index_benchmark"].drop_nulls().len() else initial_capital
        universe_end = float(equity["universe_benchmark"].drop_nulls()[-1]) if equity["universe_benchmark"].drop_nulls().len() else initial_capital
        sell_trades = [item for item in trades if item["side"] == "sell"]
        monthly = equity.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month")).group_by("month").agg(
            ((pl.col("value").last() / pl.col("value").first()) - 1).alias("return")
        ).sort("month")
        return {
            "total_return": total_return, "annual_return": annual_return,
            "volatility": volatility, "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": max_drawdown, "calmar": calmar,
            "index_total_return": index_end / initial_capital - 1,
            "universe_total_return": universe_end / initial_capital - 1,
            "excess_vs_index": float(total_return - (index_end / initial_capital - 1)),
            "excess_vs_universe": float(total_return - (universe_end / initial_capital - 1)),
            "win_rate": float(np.mean([item["pnl"] > 0 for item in sell_trades])) if sell_trades else 0.0,
            "trade_count": len(trades), "total_cost": total_cost,
            "average_turnover": total_turnover / max(1, len(equity)),
            "oos_trading_days": len(equity), "monthly_returns": monthly.to_dicts(),
        }

    @staticmethod
    def _write_artifacts(run_dir: Path, result: dict[str, Any]) -> None:
        equity = pl.DataFrame(result["equity_curve"])
        equity.write_parquet(run_dir / "equity.parquet")
        if result["holdings"]:
            pl.DataFrame(result["holdings"]).write_parquet(run_dir / "holdings.parquet")
        if result["trades"]:
            pl.DataFrame(result["trades"]).write_parquet(run_dir / "trades.parquet")
        (run_dir / "metrics.json").write_text(
            json.dumps(result["metrics"], ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    @staticmethod
    def _tradable(row: dict[str, Any] | None, side: str) -> bool:
        if not row:
            return False
        prices = [MLBacktestService._number(row.get(name)) for name in ("open", "high", "low", "close")]
        if any(value <= 0 for value in prices):
            return False
        volume = MLBacktestService._number(row.get("volume"))
        if volume <= 0 and max(prices) - min(prices) <= max(prices[-1] * 1e-4, 0.01):
            return False
        one_price = max(prices) - min(prices) <= max(prices[-1] * 1e-4, 0.01)
        if side == "buy" and one_price and bool(row.get("signal_limit_up")):
            return False
        return not (
            side == "sell" and one_price and bool(row.get("signal_limit_down"))
        )

    @staticmethod
    def _trade_price(row: dict[str, Any] | None, fallback: float, column: str) -> float:
        if row:
            value = MLBacktestService._number(row.get(column))
            if value > 0:
                return value
        return fallback

    @staticmethod
    def _number(value: Any) -> float:
        try:
            number = float(value)
            return number if np.isfinite(number) else 0.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _check_cancelled(cancelled: threading.Event) -> None:
        if cancelled.is_set():
            raise InterruptedError("回测已取消")
