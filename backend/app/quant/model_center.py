"""Model-center aggregation and explainable research-grade diagnostics."""
from __future__ import annotations

from typing import Any

from app.quant.experiments import ExperimentStore
from app.quant.ml_backtest import MLBacktestService
from app.quant.model_registry import ModelRegistry
from app.quant.predictions import PredictionService


class ModelCenterService:
    def __init__(
        self,
        models: ModelRegistry,
        experiments: ExperimentStore,
        backtests: MLBacktestService,
        predictions: PredictionService,
    ) -> None:
        self.models = models
        self.experiments = experiments
        self.backtests = backtests
        self.predictions = predictions

    def list(self) -> list[dict[str, Any]]:
        return [self.summary(item) for item in self.models.list()]

    def detail(self, version: str) -> dict[str, Any]:
        metadata = self.models.get(version)
        try:
            training_run = self.backtests.find_training_run(version)
        except ValueError:
            training_run = None
        backtest_runs = self.backtests.list_runs(version)
        completed = [item for item in backtest_runs if item.status == "completed"]
        latest_backtest = completed[0] if completed else None
        prediction_dates = self.predictions.dates(version)
        diagnostic = self._diagnostic(metadata, training_run, latest_backtest)
        return {
            **metadata,
            "training_run": training_run.model_dump(mode="json") if training_run else None,
            "backtests": [item.model_dump(mode="json") for item in backtest_runs],
            "latest_backtest": latest_backtest.model_dump(mode="json") if latest_backtest else None,
            "prediction_dates": prediction_dates,
            "latest_prediction": prediction_dates[0] if prediction_dates else None,
            "diagnostic": diagnostic,
        }

    def summary(self, metadata: dict[str, Any]) -> dict[str, Any]:
        version = metadata["version"]
        try:
            training_run = self.backtests.find_training_run(version)
        except ValueError:
            training_run = None
        completed = [
            item for item in self.backtests.list_runs(version) if item.status == "completed"
        ]
        latest_backtest = completed[0] if completed else None
        prediction_dates = self.predictions.dates(version)
        diagnostic = self._diagnostic(metadata, training_run, latest_backtest)
        metrics = latest_backtest.result.get("metrics", {}) if latest_backtest else {}
        return {
            **metadata,
            "diagnostic": diagnostic,
            "latest_backtest": {
                "run_id": latest_backtest.run_id, "metrics": metrics,
                "created_at": latest_backtest.created_at,
            } if latest_backtest else None,
            "latest_prediction": prediction_dates[0] if prediction_dates else None,
        }

    @staticmethod
    def _diagnostic(metadata, training_run, latest_backtest) -> dict[str, Any]:
        statistical = metadata.get("metrics", {})
        folds = training_run.result.get("folds", []) if training_run else []
        daily_ic = training_run.result.get("metrics", {}).get("daily_ic", []) if training_run else []
        warnings = list(dict.fromkeys([
            *metadata.get("training", {}).get("warnings", []),
            *(latest_backtest.result.get("warnings", []) if latest_backtest else []),
        ]))
        coverage = float(statistical.get("coverage") or 0.0)
        oos_days = len(daily_ic)
        data_green = len(folds) >= 2 and oos_days >= 252 and coverage >= 0.9
        data_status = "green" if data_green else "yellow" if folds and coverage >= 0.8 else "red"

        rank_ic = float(statistical.get("rank_ic") or 0.0)
        icir = float(statistical.get("icir") or 0.0)
        positive_rate = float(statistical.get("ic_positive_rate") or 0.0)
        stat_green = rank_ic >= 0.03 and icir >= 0.5 and positive_rate >= 0.55
        stat_yellow = rank_ic >= 0.01 and icir >= 0.2 and positive_rate >= 0.5
        stat_status = "green" if stat_green else "yellow" if stat_yellow else "red"

        fold_ics = [float(item.get("metrics", {}).get("rank_ic") or 0.0) for item in folds]
        annual = [float(value) for value in statistical.get("annual_stability", {}).values()]
        fold_positive = sum(value > 0 for value in fold_ics) / max(1, len(fold_ics))
        annual_positive = sum(value > 0 for value in annual) / max(1, len(annual))
        recent_ok = not fold_ics or fold_ics[-1] >= 0
        stability_green = fold_positive >= 0.7 and annual_positive >= 0.7 and recent_ok
        stability_yellow = fold_positive >= 0.5 and recent_ok
        stability_status = "green" if stability_green else "yellow" if stability_yellow else "red"

        economic = latest_backtest.result.get("metrics", {}) if latest_backtest else {}
        sharpe = float(economic.get("sharpe") or 0.0)
        max_drawdown = abs(float(economic.get("max_drawdown") or 0.0))
        excess_index = float(economic.get("excess_vs_index") or 0.0)
        excess_universe = float(economic.get("excess_vs_universe") or 0.0)
        econ_green = (
            latest_backtest is not None and sharpe >= 0.8 and max_drawdown <= 0.2
            and excess_index > 0 and excess_universe > 0
        )
        econ_yellow = (
            latest_backtest is not None and sharpe >= 0.4 and max_drawdown <= 0.3
            and (excess_index > 0 or excess_universe > 0)
        )
        economic_status = "green" if econ_green else "yellow" if econ_yellow else "red"

        if len(folds) < 2 or oos_days < 252 or latest_backtest is None:
            grade = "unverified"
        elif (
            not latest_backtest.result.get("oos_only", False)
            or (rank_ic <= 0 and excess_index <= 0 and excess_universe <= 0)
        ):
            grade = "invalid"
        elif "red" in {stat_status, stability_status, economic_status}:
            grade = "weak"
        else:
            grade = "candidate"
        has_survivorship_warning = any("幸存者偏差" in item for item in warnings)
        if (
            grade == "candidate" and data_green and stat_green and stability_green and econ_green
            and oos_days >= 504 and not has_survivorship_warning
        ):
            grade = "robust"
        return {
            "grade": grade,
            "dimensions": {
                "data": {
                    "status": data_status, "folds": len(folds), "oos_days": oos_days,
                    "coverage": coverage,
                    "reason": f"{len(folds)} 折 / {oos_days} 个 OOS 交易日 / 覆盖率 {coverage:.1%}",
                },
                "statistics": {
                    "status": stat_status, "rank_ic": rank_ic, "icir": icir,
                    "positive_rate": positive_rate,
                    "reason": f"Rank IC {rank_ic:.4f} / ICIR {icir:.2f} / 正值率 {positive_rate:.1%}",
                },
                "stability": {
                    "status": stability_status, "fold_positive_rate": fold_positive,
                    "annual_positive_rate": annual_positive,
                    "reason": f"正 IC 折 {fold_positive:.0%} / 正 IC 年度 {annual_positive:.0%}",
                },
                "economics": {
                    "status": economic_status, "sharpe": sharpe,
                    "max_drawdown": max_drawdown, "excess_vs_index": excess_index,
                    "excess_vs_universe": excess_universe,
                    "reason": "待完成 OOS 组合回测" if latest_backtest is None else (
                        f"Sharpe {sharpe:.2f} / 回撤 {max_drawdown:.1%} / "
                        f"指数超额 {excess_index:.1%} / 股票池超额 {excess_universe:.1%}"
                    ),
                },
            },
            "warnings": warnings,
            "publish_warning": grade in {"unverified", "weak", "invalid"},
        }
