"""Purged walk-forward model training and OOS-only evaluation."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from app.quant.adapters import FitResult, get_adapter, model_artifact_name, write_model_metadata
from app.quant.dataset import MLDatasetBuilder
from app.quant.metrics import evaluate_oos
from app.quant.model_registry import ModelRegistry
from app.quant.models import ModelSpec
from app.quant.splits import assert_no_label_overlap, generate_purged_folds

ProgressCallback = Callable[[float, str], None]


class MLTrainer:
    def __init__(self, datasets: MLDatasetBuilder, models: ModelRegistry) -> None:
        self.datasets = datasets
        self.models = models

    def run(self, spec: ModelSpec, run_dir: Path, progress: ProgressCallback,
            cancelled: threading.Event) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        progress(0.04, "正在构建时点正确的特征与标签")
        dataset = self.datasets.build(spec, cancelled=cancelled)
        self._check_cancelled(cancelled)
        folds = generate_purged_folds(dataset.calendar, spec.walk_forward, spec.target.horizon)
        for fold in folds:
            assert_no_label_overlap(fold, dataset.calendar, spec.target.horizon)
        (run_dir / "splits.json").write_text(
            json.dumps([fold.summary() for fold in folds], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        adapter = get_adapter(spec.algorithm)
        predictions: list[pl.DataFrame] = []
        fold_results: list[dict[str, Any]] = []
        importance: dict[str, dict[str, list[float]]] = {
            name: {"gain": [], "split": []} for name in spec.features
        }
        all_warnings = list(dataset.warnings)
        selected_params = dict(spec.params)
        devices: set[str] = set()
        versions: set[str] = set()
        elapsed = 0.0
        for index, fold in enumerate(folds):
            if cancelled.is_set():
                raise InterruptedError("训练已取消")
            train = dataset.frame.filter(pl.col("date").is_in(fold.train_dates))
            validation = dataset.frame.filter(pl.col("date").is_in(fold.validation_dates))
            test = dataset.frame.filter(pl.col("date").is_in(fold.test_dates))
            params, baseline_ic, tuning = self._select_params(
                adapter, spec, train, validation, cancelled
            )
            selected_params = params
            fitted = self._fit(adapter, spec, train, validation, params, cancelled)
            self._check_cancelled(cancelled)
            prediction = adapter.predict(fitted.model, test.select(spec.features).to_numpy())
            test_out = test.select(["symbol", "date", "target", "forward_return", "benchmark_return"]).with_columns(
                pl.Series("prediction", prediction), pl.lit(index).alias("fold")
            ).with_columns(
                (pl.col("prediction").rank(method="average").over("date") / pl.len().over("date")).alias("rank")
            )
            fold_metrics = evaluate_oos(test_out)
            predictions.append(test_out)
            raw_importance = adapter.feature_importance(fitted.model, spec.features)
            for name in spec.features:
                importance[name]["gain"].append(raw_importance[name]["gain"])
                importance[name]["split"].append(raw_importance[name]["split"])
            devices.add(fitted.actual_device)
            versions.add(fitted.library_version)
            elapsed += fitted.elapsed_seconds
            all_warnings.extend(fitted.warnings)
            fold_results.append({
                **fold.summary(), "metrics": {k: v for k, v in fold_metrics.items() if k != "daily_ic"},
                "baseline_validation_rank_ic": baseline_ic, "tuning": tuning,
                "params": params, "actual_device": fitted.actual_device,
                "library_version": fitted.library_version, "training_seconds": fitted.elapsed_seconds,
                "best_iteration": fitted.best_iteration, "warnings": fitted.warnings,
            })
            progress(0.08 + 0.72 * (index + 1) / len(folds), f"完成 Walk-forward 第 {index + 1}/{len(folds)} 折")
        oos = pl.concat(predictions).sort(["date", "symbol"])
        oos.write_parquet(run_dir / "oos_predictions.parquet")
        metrics = evaluate_oos(oos)
        (run_dir / "fold_metrics.json").write_text(
            json.dumps(fold_results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        progress(0.84, "正在用最新可用标签重训候选发布模型")
        self._check_cancelled(cancelled)
        final_fit = self._fit_final(adapter, spec, dataset.frame, selected_params, cancelled)
        self._check_cancelled(cancelled)
        source_model = run_dir / model_artifact_name(spec.algorithm)
        adapter.save(final_fit.model, source_model)
        averaged_importance = {
            name: {
                kind: float(np.mean(values)) if values else 0.0
                for kind, values in values_by_kind.items()
            }
            for name, values_by_kind in importance.items()
        }
        schema = {"features": spec.features, "dtypes": {name: str(dataset.frame.schema[name]) for name in spec.features}}
        training = {
            "actual_devices": sorted(devices | {final_fit.actual_device}),
            "library_versions": sorted(versions | {final_fit.library_version}),
            "training_seconds": elapsed + final_fit.elapsed_seconds,
            "final_params": selected_params, "feature_importance_gain": averaged_importance,
            "reference_prediction_quantiles": np.quantile(
                oos["prediction"].to_numpy(), np.linspace(0, 1, 11)
            ).tolist(),
            "warnings": list(dict.fromkeys([*all_warnings, *final_fit.warnings])),
        }
        write_model_metadata(run_dir / "model_metadata.json", {"schema": schema, "training": training})
        registered = self.models.register(
            spec=spec.model_dump(mode="json"), source_model=source_model, schema=schema,
            metrics={k: v for k, v in metrics.items() if k != "daily_ic"},
            data_fingerprint=dataset.fingerprint, training=training,
            source_run_id=run_dir.name,
        )
        result = {
            "model_version": registered["version"], "model_status": registered["status"],
            "data_fingerprint": dataset.fingerprint, "rows": dataset.frame.height,
            "input_file_fingerprint": dataset.input_file_fingerprint,
            "date_range": [str(dataset.frame["date"].min()), str(dataset.frame["date"].max())],
            "folds": fold_results, "metrics": metrics, "feature_importance": averaged_importance,
            "training": training, "warnings": training["warnings"],
        }
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        progress(1.0, "训练完成, 模型等待手动发布")
        return result

    def _select_params(self, adapter, spec: ModelSpec, train: pl.DataFrame,
                       validation: pl.DataFrame, cancelled: threading.Event,
                       ) -> tuple[dict[str, Any], float | None, dict[str, Any]]:
        baseline = self._fit(adapter, spec, train, validation, spec.params, cancelled)
        validation_out = validation.select(["symbol", "date", "target"]).with_columns(
            pl.Series("prediction", adapter.predict(baseline.model, validation.select(spec.features).to_numpy()))
        )
        baseline_ic = evaluate_oos(validation_out)["rank_ic"]
        if not spec.tuning.enabled:
            return dict(spec.params), baseline_ic, {"enabled": False, "trials": 0}
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("已启用 Optuna 调优, 但 ml 可选依赖尚未安装") from exc

        def objective(trial) -> float:
            self._check_cancelled(cancelled)
            if spec.algorithm == "lightgbm":
                trial_params = {
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 3.0, log=True),
                }
            elif spec.algorithm == "xgboost":
                trial_params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                    "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 3.0, log=True),
                }
            else:
                trial_params = {
                    "alpha": trial.suggest_float("alpha", 1e-6, 0.1, log=True),
                    "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                }
            fitted = self._fit(
                adapter, spec, train, validation, {**spec.params, **trial_params}, cancelled
            )
            predicted = validation.select(["symbol", "date", "target"]).with_columns(
                pl.Series("prediction", adapter.predict(fitted.model, validation.select(spec.features).to_numpy()))
            )
            return float(evaluate_oos(predicted)["rank_ic"] or -1.0)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=spec.seed))
        study.optimize(objective, n_trials=spec.tuning.max_trials, show_progress_bar=False)
        if baseline_ic is not None and baseline_ic >= study.best_value:
            return dict(spec.params), baseline_ic, {
                "enabled": True, "trials": len(study.trials), "selected": "baseline",
                "best_validation_rank_ic": baseline_ic,
            }
        return {**spec.params, **study.best_params}, baseline_ic, {
            "enabled": True, "trials": len(study.trials), "selected": "tuned",
            "best_validation_rank_ic": study.best_value,
        }

    @staticmethod
    def _fit(adapter, spec: ModelSpec, train: pl.DataFrame,
             validation: pl.DataFrame, params: dict[str, Any],
             cancelled: threading.Event | None = None) -> FitResult:
        lower, upper = train["target"].quantile(0.01), train["target"].quantile(0.99)
        y_train = train["target"].clip(lower, upper).to_numpy()
        return adapter.fit(
            train.select(spec.features).to_numpy(), y_train,
            validation.select(spec.features).to_numpy(), validation["target"].to_numpy(),
            train["sample_weight"].to_numpy(), params, spec.device, spec.seed, cancelled,
        )

    def _fit_final(self, adapter, spec: ModelSpec, frame: pl.DataFrame,
                   params: dict[str, Any], cancelled: threading.Event) -> FitResult:
        dates = sorted(frame["date"].unique().to_list())
        validation_days = min(spec.walk_forward.validation_days, max(20, len(dates) // 5))
        validation_start = dates[-validation_days]
        train = frame.filter(pl.col("date") < validation_start)
        validation = frame.filter(pl.col("date") >= validation_start)
        return self._fit(adapter, spec, train, validation, params, cancelled)

    @staticmethod
    def _check_cancelled(cancelled: threading.Event) -> None:
        if cancelled.is_set():
            raise InterruptedError("训练已取消")
