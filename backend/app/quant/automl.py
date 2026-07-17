"""Controlled multi-factor AutoML with nested, leakage-safe model selection."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from app.quant.adapters import get_adapter, model_artifact_name, write_model_metadata
from app.quant.dataset import MLDatasetBuilder
from app.quant.factors import FactorRegistry
from app.quant.metrics import evaluate_oos
from app.quant.model_registry import ModelRegistry
from app.quant.models import MLSearchSpec, ModelSpec, ResearchPanelSpec
from app.quant.splits import assert_no_label_overlap, generate_purged_folds

ProgressCallback = Callable[[float, str], None]

_BUDGET_TRIALS = {
    "quick": {"elastic_net": 2, "lightgbm": 3, "xgboost": 3},
    "standard": {"elastic_net": 12, "lightgbm": 30, "xgboost": 30},
    "overnight": {"elastic_net": 30, "lightgbm": 75, "xgboost": 75},
}
_SUBSET_SIZES = (8, 12, 16, 20, 24, 30)


@dataclass(frozen=True)
class InnerFold:
    index: int
    train_dates: list[date]
    validation_dates: list[date]


class AutoMLSearchEngine:
    def __init__(
        self,
        datasets: MLDatasetBuilder,
        factors: FactorRegistry,
        models: ModelRegistry,
        experiments: Any,
    ) -> None:
        self.datasets = datasets
        self.factors = factors
        self.models = models
        self.experiments = experiments

    def estimate(self, spec: MLSearchSpec) -> dict[str, Any]:
        definitions = self._resolve_factors(spec)
        fields = sorted({
            "open", "close", *(item.id for item in definitions),
            *(name for item in definitions for name in item.inputs),
        })
        panel = self.datasets.panels.estimate(ResearchPanelSpec(
            asset_type=spec.asset_type, frequency="1d", symbols=spec.symbols,
            start=spec.start, end=spec.end, fields=fields,
            warmup=max([120, *(item.warmup for item in definitions)]),
        ))
        trading_days = max(0, round((spec.end - spec.start).days * 252 / 365.25))
        wf = spec.walk_forward
        required = wf.train_days + wf.validation_days + wf.test_days + spec.target.horizon * 2
        outer_folds = max(0, 1 + (trading_days - required) // wf.step_days)
        trials = sum(self._trial_counts(spec).values())
        fits = (outer_folds + 1) * (trials * spec.inner_folds + 1)
        hours = {"quick": 0.5, "standard": 3.0, "overnight": 10.0}[spec.budget]
        hours *= max(0.5, min(2.0, len(definitions) / 40))
        warnings = [*panel.get("warnings", []), *panel.get("missing_data", [])]
        if not panel.get("allowed", True):
            warnings.append(panel.get("reason") or "预计数据量超过本地限制")
        if outer_folds < 2:
            warnings.append("预计少于 2 个外层测试折, 结果最高为待验证")
        return {
            **panel,
            "factor_count": len(definitions),
            "outer_folds": outer_folds,
            "search_trials_per_window": trials,
            "estimated_model_fits": fits,
            "estimated_hours": round(hours, 1),
            "warnings": list(dict.fromkeys(warnings)),
        }

    def run(
        self,
        spec: MLSearchSpec,
        run_dir: Path,
        progress: ProgressCallback,
        cancelled: threading.Event,
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        definitions, preliminary_rejections = self._resolve_factor_pool(spec)
        feature_versions = {item.id: item.version for item in definitions}
        base_spec = self._model_spec(
            spec, "lightgbm", list(feature_versions), feature_versions, model_id=spec.id
        )
        progress(0.03, "正在构建多因子研究面板")
        dataset = self.datasets.build(
            base_spec, cancelled=cancelled, require_complete_features=False
        )
        self._check_cancelled(cancelled)
        folds = generate_purged_folds(
            dataset.calendar, spec.walk_forward, spec.target.horizon
        )
        for fold in folds:
            assert_no_label_overlap(fold, dataset.calendar, spec.target.horizon)

        predictions: list[pl.DataFrame] = []
        fold_results: list[dict[str, Any]] = []
        selection_counts: Counter[str] = Counter()
        algorithm_counts: Counter[str] = Counter()
        all_trials: list[dict[str, Any]] = []
        all_warnings = list(dataset.warnings)
        total_test_rows = 0
        latest_quality: list[dict[str, Any]] = list(preliminary_rejections)
        latest_clusters: list[dict[str, Any]] = []

        for index, fold in enumerate(folds):
            self._check_cancelled(cancelled)
            train = dataset.frame.filter(pl.col("date").is_in(fold.train_dates))
            validation = dataset.frame.filter(pl.col("date").is_in(fold.validation_dates))
            test = dataset.frame.filter(pl.col("date").is_in(fold.test_dates))
            total_test_rows += test.height
            def fold_progress(
                fraction: float, message: str, *, outer_index: int = index
            ) -> None:
                progress(
                    0.06 + 0.68 * (outer_index + fraction) / len(folds),
                    f"外层第 {outer_index + 1}/{len(folds)} 折 · {message}",
                )

            window = self._search_window(
                train, spec, list(feature_versions), cancelled,
                fold_progress,
            )
            winner = window["winner"]
            fitted = self._fit(
                winner["algorithm"], winner["features"], winner["params"], spec,
                train, validation, cancelled,
            )
            valid_test = test.drop_nulls(winner["features"])
            if valid_test.is_empty():
                raise ValueError(f"外层第 {index + 1} 折冠军在测试集没有完整特征")
            values = get_adapter(winner["algorithm"]).predict(
                fitted.model, valid_test.select(winner["features"]).to_numpy()
            )
            test_out = valid_test.select([
                "symbol", "date", "target", "forward_return", "benchmark_return"
            ]).with_columns(
                pl.Series("prediction", values), pl.lit(index).alias("fold")
            ).with_columns(
                (pl.col("prediction").rank(method="average").over("date")
                 / pl.len().over("date")).alias("rank")
            )
            fold_metrics = evaluate_oos(test_out)
            predictions.append(test_out)
            selection_counts.update(winner["features"])
            algorithm_counts.update([winner["algorithm"]])
            fold_results.append({
                **fold.summary(),
                "algorithm": winner["algorithm"], "features": winner["features"],
                "params": winner["params"], "selection_score": winner["score"],
                "metrics": {key: value for key, value in fold_metrics.items() if key != "daily_ic"},
                "actual_device": fitted.actual_device,
                "training_seconds": fitted.elapsed_seconds,
            })
            all_trials.extend([
                {**trial, "outer_fold": index} for trial in window["trials"]
            ])
            latest_quality = [*preliminary_rejections, *window["quality"]]
            latest_clusters = window["clusters"]
            all_warnings.extend(window["warnings"])

        oos = pl.concat(predictions).sort(["date", "symbol"])
        oos.write_parquet(run_dir / "oos_predictions.parquet")
        metrics = evaluate_oos(oos)
        metrics["coverage"] = oos.height / max(1, total_test_rows)
        selection_backtest = self._selection_backtest(oos, spec)
        oos_signature = hashlib.sha256(
            json.dumps(sorted(set(str(value) for value in oos["date"].to_list()))).encode()
        ).hexdigest()[:16]
        if any(
            item.kind == "ml_search" and item.status == "completed"
            and item.result.get("oos_signature") == oos_signature
            for item in self.experiments.list()
        ):
            all_warnings.append("同一 OOS 区间已用于其他智能搜索, 存在研究选择偏差")
        if len(folds) < 2 or len(metrics.get("daily_ic", [])) < 252:
            all_warnings.append("OOS 折数或交易日不足, 模型最高为待验证")

        progress(0.78, "正在最新训练窗口重新选择发布候选")
        labeled_dates = sorted(dataset.frame["date"].unique().to_list())
        latest_dates = labeled_dates[-min(len(labeled_dates), spec.walk_forward.train_days):]
        latest_frame = dataset.frame.filter(pl.col("date").is_in(latest_dates))
        final_window = self._search_window(
            latest_frame, spec, list(feature_versions), cancelled,
            lambda fraction, message: progress(0.78 + 0.15 * fraction, f"最新窗口 · {message}"),
        )
        champion = final_window["winner"]
        all_trials.extend([{**trial, "outer_fold": "final"} for trial in final_window["trials"]])
        latest_quality = [*preliminary_rejections, *final_window["quality"]]
        latest_clusters = final_window["clusters"]
        all_warnings.extend(final_window["warnings"])

        validation_days = min(
            spec.walk_forward.validation_days, max(20, len(labeled_dates) // 5)
        )
        final_train = dataset.frame.filter(pl.col("date") < labeled_dates[-validation_days])
        final_validation = dataset.frame.filter(pl.col("date") >= labeled_dates[-validation_days])
        final_fit = self._fit(
            champion["algorithm"], champion["features"], champion["params"], spec,
            final_train, final_validation, cancelled,
        )
        source_model = run_dir / model_artifact_name(champion["algorithm"])
        adapter = get_adapter(champion["algorithm"])
        adapter.save(final_fit.model, source_model)
        importance = adapter.feature_importance(final_fit.model, champion["features"])
        champion_versions = {
            name: feature_versions[name] for name in champion["features"]
        }
        champion_spec = self._model_spec(
            spec, champion["algorithm"], champion["features"], champion_versions,
            model_id=spec.id,
        )
        schema = {
            "features": champion["features"],
            "dtypes": {
                name: str(dataset.frame.schema[name]) for name in champion["features"]
            },
        }
        training = {
            "actual_devices": sorted({
                item["actual_device"] for item in fold_results
            } | {final_fit.actual_device}),
            "library_versions": [final_fit.library_version],
            "training_seconds": sum(item["training_seconds"] for item in fold_results)
            + final_fit.elapsed_seconds,
            "final_params": champion["params"],
            "feature_importance_gain": importance,
            "feature_selection_frequency": {
                name: selection_counts[name] / max(1, len(folds)) for name in feature_versions
            },
            "algorithm_frequency": dict(algorithm_counts),
            "reference_prediction_quantiles": np.quantile(
                oos["prediction"].to_numpy(), np.linspace(0, 1, 11)
            ).tolist(),
            "warnings": list(dict.fromkeys([*all_warnings, *final_fit.warnings])),
            "automl": True,
        }
        write_model_metadata(run_dir / "model_metadata.json", {
            "schema": schema, "training": training
        })
        registered = self.models.register(
            spec=champion_spec.model_dump(mode="json"), source_model=source_model,
            schema=schema, metrics={key: value for key, value in metrics.items() if key != "daily_ic"},
            data_fingerprint=dataset.fingerprint, training=training,
            source_run_id=run_dir.name,
        )

        self._write_artifacts(
            run_dir, latest_quality, latest_clusters, all_trials, fold_results,
            selection_backtest,
        )
        result = {
            "model_version": registered["version"], "model_status": registered["status"],
            "champion": {
                "algorithm": champion["algorithm"], "features": champion["features"],
                "feature_versions": champion_versions, "params": champion["params"],
                "selection_score": champion["score"],
            },
            "data_fingerprint": dataset.fingerprint,
            "input_file_fingerprint": dataset.input_file_fingerprint,
            "rows": dataset.frame.height,
            "date_range": [str(dataset.frame["date"].min()), str(dataset.frame["date"].max())],
            "factor_funnel": {
                "submitted": len(spec.factor_pool),
                "quality_passed": sum(item["status"] == "accepted" for item in latest_quality),
                "shortlisted": len(final_window["shortlist"]),
                "selected": len(champion["features"]),
            },
            "factor_quality": latest_quality, "correlation_clusters": latest_clusters,
            "feature_selection_frequency": training["feature_selection_frequency"],
            "algorithm_frequency": dict(algorithm_counts),
            "candidate_leaderboard": sorted(
                final_window["trials"], key=lambda item: item.get("score", -999), reverse=True
            )[:20],
            "folds": fold_results, "metrics": metrics,
            "selection_backtest": selection_backtest,
            "feature_importance": importance, "training": training,
            "oos_signature": oos_signature, "warnings": training["warnings"],
        }
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        progress(1.0, "智能训练完成, 冠军模型等待手动发布")
        return result

    def _search_window(
        self,
        frame: pl.DataFrame,
        spec: MLSearchSpec,
        factors: list[str],
        cancelled: threading.Event,
        progress: ProgressCallback,
    ) -> dict[str, Any]:
        required_order = [item.id for item in spec.required_factors]
        required = set(required_order)
        quality = self._factor_quality(frame, factors, required)
        accepted = [item["factor_id"] for item in quality if item["status"] == "accepted"]
        if len(accepted) < spec.min_features:
            raise ValueError(
                f"质量筛选后只有 {len(accepted)} 个因子, 少于最小值 {spec.min_features}"
            )
        progress(0.08, f"质量筛选保留 {len(accepted)}/{len(factors)} 个因子")
        clusters, representatives = self._correlation_clusters(
            frame, accepted, quality, required
        )
        progress(0.14, f"相关性聚类得到 {len(clusters)} 个因子组")
        permutation = self._permutation_importance(frame, representatives, spec.seed)
        quality_rank = sorted(
            representatives,
            key=lambda name: next(item["selection_rank"] for item in quality if item["factor_id"] == name),
            reverse=True,
        )
        importance_rank = sorted(
            representatives, key=lambda name: permutation.get(name, 0.0), reverse=True
        )
        combined: list[str] = []
        for name in [*required_order, *quality_rank[:60], *importance_rank[:60]]:
            if name in representatives and name not in combined:
                combined.append(name)
        shortlist = combined[:spec.shortlist_limit]
        if len(shortlist) < spec.min_features:
            raise ValueError("相关性去重后因子不足, 请降低 min_features 或增加因子池")
        progress(0.20, f"形成 {len(shortlist)} 个因子短名单")
        folds = self._inner_folds(
            sorted(frame["date"].unique().to_list()), spec.inner_folds,
            spec.inner_validation_days, spec.target.horizon,
        )
        trials: list[dict[str, Any]] = []
        trial_specs = self._trial_specs(spec, shortlist)
        for index, trial in enumerate(trial_specs):
            self._check_cancelled(cancelled)
            try:
                result = self._evaluate_trial(frame, spec, folds, trial, cancelled)
            except InterruptedError:
                raise
            except Exception as exc:
                result = {**trial, "status": "failed", "error": str(exc), "score": -999.0}
            trials.append(result)
            progress(
                0.20 + 0.78 * (index + 1) / max(1, len(trial_specs)),
                f"完成候选 {index + 1}/{len(trial_specs)}",
            )
        completed = [item for item in trials if item["status"] == "completed"]
        if not completed:
            errors = [item.get("error", "未知错误") for item in trials[:3]]
            raise ValueError(f"所有搜索候选均失败: {errors}")
        winner = self._select_winner(completed)
        return {
            "winner": winner, "trials": trials, "quality": quality,
            "clusters": clusters, "shortlist": shortlist, "warnings": [],
        }

    def _evaluate_trial(
        self,
        frame: pl.DataFrame,
        spec: MLSearchSpec,
        folds: list[InnerFold],
        trial: dict[str, Any],
        cancelled: threading.Event,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        outputs: list[pl.DataFrame] = []
        fold_ics: list[float] = []
        coverages: list[float] = []
        devices: set[str] = set()
        warnings: list[str] = []
        for fold in folds:
            train = frame.filter(pl.col("date").is_in(fold.train_dates))
            validation = frame.filter(pl.col("date").is_in(fold.validation_dates))
            valid_train = train.drop_nulls(trial["features"])
            valid_validation = validation.drop_nulls(trial["features"])
            coverage = valid_validation.height / max(1, validation.height)
            if coverage < 0.9:
                raise ValueError(f"内层第 {fold.index + 1} 折覆盖率 {coverage:.1%} 低于 90%")
            if valid_train.is_empty() or valid_validation.is_empty():
                raise ValueError("内层折没有完整训练或验证样本")
            fitted = self._fit(
                trial["algorithm"], trial["features"], trial["params"], spec,
                valid_train, valid_validation, cancelled,
            )
            prediction = get_adapter(trial["algorithm"]).predict(
                fitted.model, valid_validation.select(trial["features"]).to_numpy()
            )
            output = valid_validation.select([
                "symbol", "date", "target", "forward_return", "benchmark_return"
            ]).with_columns(pl.Series("prediction", prediction))
            metrics = evaluate_oos(output)
            outputs.append(output)
            fold_ics.append(float(metrics.get("rank_ic") or 0.0))
            coverages.append(coverage)
            devices.add(fitted.actual_device)
            warnings.extend(fitted.warnings)
        combined = pl.concat(outputs).sort(["date", "symbol"])
        metrics = evaluate_oos(combined)
        economic = self._selection_backtest(combined, spec)
        score = self._composite_score(
            metrics, economic, float(np.mean(coverages)), fold_ics,
            len(trial["features"]),
        )
        return {
            **trial, "status": "completed", "score": score,
            "metrics": {key: value for key, value in metrics.items() if key != "daily_ic"},
            "economic": economic, "fold_positive_rate": float(np.mean(np.array(fold_ics) > 0)),
            "coverage": float(np.mean(coverages)), "actual_devices": sorted(devices),
            "training_seconds": time.perf_counter() - started,
            "warnings": list(dict.fromkeys(warnings)),
        }

    @staticmethod
    def _factor_quality(
        frame: pl.DataFrame, factors: list[str], required: set[str]
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for name in factors:
            values = frame[name].cast(pl.Float64, strict=False).to_numpy()
            finite = np.isfinite(values)
            coverage = float(np.mean(finite)) if len(values) else 0.0
            valid = values[finite]
            unique = len(np.unique(valid)) if len(valid) else 0
            standard_deviation = float(np.std(valid)) if len(valid) else 0.0
            median = float(np.median(valid)) if len(valid) else 0.0
            mad = float(np.median(np.abs(valid - median))) if len(valid) else 0.0
            extreme_rate = float(np.mean(np.abs(valid - median) > 20 * mad)) \
                if mad > 1e-12 else float(np.mean(np.abs(valid - median) > 1e-12))
            reason = None
            if coverage < 0.9:
                reason = f"覆盖率 {coverage:.1%} 低于 90%"
            elif unique <= 1 or standard_deviation <= 1e-12:
                reason = "近零方差"
            elif extreme_rate > 0.05:
                reason = f"极端值比例 {extreme_rate:.1%} 过高"
            metrics: dict[str, Any] = {}
            if reason is None:
                evaluation = frame.select([
                    "symbol", "date", "target", pl.col(name).alias("prediction")
                ]).drop_nulls(["target", "prediction"])
                try:
                    metrics = evaluate_oos(evaluation)
                except ValueError:
                    reason = "有效截面不足"
            if name in required and reason:
                raise ValueError(f"必选因子 {name} 未通过质量检查: {reason}")
            rank_ic = abs(float(metrics.get("rank_ic") or 0.0))
            icir = min(abs(float(metrics.get("icir") or 0.0)), 2.0)
            positive = abs(float(metrics.get("ic_positive_rate") or 0.5) - 0.5) * 2
            selection_rank = rank_ic * 0.55 + icir * 0.15 + positive * 0.15 + coverage * 0.15
            result.append({
                "factor_id": name, "status": "accepted" if reason is None else "rejected",
                "reason": reason, "coverage": coverage, "unique_values": unique,
                "standard_deviation": standard_deviation,
                "extreme_rate": extreme_rate, "rank_ic": metrics.get("rank_ic"),
                "icir": metrics.get("icir"),
                "ic_positive_rate": metrics.get("ic_positive_rate"),
                "top_bottom_return": metrics.get("top_bottom_return"),
                "annual_stability": metrics.get("annual_stability", {}),
                "selection_rank": selection_rank,
            })
        return result

    @staticmethod
    def _correlation_clusters(
        frame: pl.DataFrame,
        factors: list[str],
        quality: list[dict[str, Any]],
        required: set[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if len(factors) == 1:
            return [{"id": 0, "factors": factors, "representatives": factors}], factors
        dates = sorted(frame["date"].unique().to_list())
        if len(dates) > 126:
            indices = np.linspace(0, len(dates) - 1, 126, dtype=int)
            dates = [dates[index] for index in indices]
        matrices: list[np.ndarray] = []
        for day in dates:
            values = frame.filter(pl.col("date") == day).select(factors).to_numpy().astype(float)
            if len(values) < 3:
                continue
            ranked = np.empty_like(values)
            for column in range(values.shape[1]):
                data = values[:, column]
                finite = np.isfinite(data)
                fill = float(np.nanmedian(data[finite])) if finite.any() else 0.0
                data = np.where(finite, data, fill)
                order = np.argsort(data, kind="mergesort")
                ranks = np.empty(len(data), dtype=float)
                ranks[order] = np.arange(len(data), dtype=float)
                ranked[:, column] = ranks
            matrix = np.corrcoef(ranked, rowvar=False)
            if matrix.shape == (len(factors), len(factors)):
                matrices.append(matrix)
        median = np.nanmedian(np.stack(matrices), axis=0) if matrices else np.eye(len(factors))
        parent = list(range(len(factors)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for left in range(len(factors)):
            for right in range(left + 1, len(factors)):
                if abs(float(median[left, right])) >= 0.9:
                    union(left, right)
        grouped: dict[int, list[str]] = {}
        for index, name in enumerate(factors):
            grouped.setdefault(find(index), []).append(name)
        ranks = {item["factor_id"]: item["selection_rank"] for item in quality}
        clusters: list[dict[str, Any]] = []
        representatives: list[str] = []
        for cluster_id, names in enumerate(grouped.values()):
            required_names = [name for name in names if name in required]
            chosen = required_names or [max(names, key=lambda name: ranks.get(name, 0.0))]
            representatives.extend(chosen)
            clusters.append({
                "id": cluster_id, "factors": names, "representatives": chosen,
                "max_abs_correlation": max(
                    [abs(float(median[factors.index(a), factors.index(b)]))
                     for index, a in enumerate(names) for b in names[index + 1:]] or [0.0]
                ),
            })
        return clusters, representatives

    @staticmethod
    def _permutation_importance(
        frame: pl.DataFrame, factors: list[str], seed: int
    ) -> dict[str, float]:
        if not factors:
            return {}
        try:
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.inspection import permutation_importance
        except ImportError:
            return {name: 0.0 for name in factors}
        dates = sorted(frame["date"].unique().to_list())
        validation_days = min(63, max(20, len(dates) // 5))
        train = frame.filter(pl.col("date") < dates[-validation_days])
        validation = frame.filter(pl.col("date") >= dates[-validation_days])
        rng = np.random.default_rng(seed)

        def sample(data: pl.DataFrame, limit: int) -> pl.DataFrame:
            if data.height <= limit:
                return data
            return data[rng.choice(data.height, limit, replace=False)]

        train = sample(train, 50_000)
        validation = sample(validation, 20_000)
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(train.select(factors).to_numpy())
        x_validation = imputer.transform(validation.select(factors).to_numpy())
        model = ExtraTreesRegressor(
            n_estimators=64, max_depth=4, min_samples_leaf=20,
            random_state=seed, n_jobs=max(1, (os.cpu_count() or 4) - 2),
        )
        model.fit(x_train, train["target"].to_numpy(), sample_weight=train["sample_weight"].to_numpy())
        measured = permutation_importance(
            model, x_validation, validation["target"].to_numpy(),
            scoring="neg_mean_absolute_error", n_repeats=1, random_state=seed,
        )
        return {
            name: float(max(0.0, measured.importances_mean[index]))
            for index, name in enumerate(factors)
        }

    @staticmethod
    def _inner_folds(
        dates: list[date], count: int, validation_days: int, horizon: int
    ) -> list[InnerFold]:
        ordered = sorted(set(dates))
        validation_start = len(ordered) - count * validation_days
        if validation_start - horizon < 252:
            raise ValueError("外层训练窗口不足以生成至少 2 个内层验证折")
        folds: list[InnerFold] = []
        for index in range(count):
            start = validation_start + index * validation_days
            train_end = start - horizon
            folds.append(InnerFold(
                index=index, train_dates=ordered[:train_end],
                validation_dates=ordered[start:start + validation_days],
            ))
        return folds

    def _trial_specs(self, spec: MLSearchSpec, shortlist: list[str]) -> list[dict[str, Any]]:
        rng = np.random.default_rng(spec.seed)
        sizes = [
            size for size in _SUBSET_SIZES
            if spec.min_features <= size <= min(spec.max_features, len(shortlist))
        ]
        if not sizes:
            sizes = [min(spec.max_features, len(shortlist))]
        required = [item.id for item in spec.required_factors]
        optional = [name for name in shortlist if name not in required]
        result: list[dict[str, Any]] = []
        for algorithm, count in self._trial_counts(spec).items():
            for index in range(count):
                size = max(sizes[index % len(sizes)], len(required))
                remaining = max(0, size - len(required))
                if index < len(sizes):
                    selected_optional = optional[:remaining]
                else:
                    selected_optional = list(rng.choice(
                        optional, size=min(remaining, len(optional)), replace=False
                    ))
                selected = [*required, *selected_optional]
                params: dict[str, Any]
                if algorithm == "elastic_net":
                    params = {
                        "alpha": 0.001 if index == 0 else float(10 ** rng.uniform(-6, -1)),
                        "l1_ratio": 0.5 if index == 0 else float(rng.uniform(0, 1)),
                    }
                elif algorithm == "lightgbm":
                    params = {
                        "n_estimators": 800,
                        "num_leaves": 31 if index == 0 else int(rng.choice([15, 31, 47, 63])),
                        "min_child_samples": 40 if index == 0 else int(rng.integers(20, 101)),
                        "reg_alpha": 0.1 if index == 0 else float(10 ** rng.uniform(-4, 0.3)),
                        "reg_lambda": 1.0 if index == 0 else float(10 ** rng.uniform(-4, 0.5)),
                    }
                else:
                    params = {
                        "n_estimators": 800,
                        "max_depth": 6 if index == 0 else int(rng.integers(3, 10)),
                        "min_child_weight": 5.0 if index == 0 else float(rng.uniform(1, 12)),
                        "reg_alpha": 0.1 if index == 0 else float(10 ** rng.uniform(-4, 0.3)),
                        "reg_lambda": 1.0 if index == 0 else float(10 ** rng.uniform(-4, 0.5)),
                    }
                result.append({
                    "trial": len(result), "algorithm": algorithm,
                    "features": selected, "params": params,
                })
        return result

    def _trial_counts(self, spec: MLSearchSpec) -> dict[str, int]:
        base = _BUDGET_TRIALS[spec.budget]
        return {name: base[name] for name in dict.fromkeys(spec.algorithms)}

    @staticmethod
    def _select_winner(completed: list[dict[str, Any]]) -> dict[str, Any]:
        best = max(item["score"] for item in completed)
        near = [item for item in completed if item["score"] >= best - 0.01]
        preference = {"elastic_net": 0, "lightgbm": 1, "xgboost": 2}
        return min(near, key=lambda item: (
            len(item["features"]), preference[item["algorithm"]],
            item["training_seconds"], -item["score"],
        ))

    @staticmethod
    def _composite_score(
        metrics: dict[str, Any], economic: dict[str, Any], coverage: float,
        fold_ics: list[float], factor_count: int,
    ) -> float:
        def clip(value: float) -> float:
            return float(np.clip(value, -1.0, 1.0))
        rank_ic = float(metrics.get("rank_ic") or 0.0)
        icir = float(metrics.get("icir") or 0.0)
        positive = float(metrics.get("ic_positive_rate") or 0.0)
        net_excess = min(
            float(economic.get("annual_excess_vs_index") or 0.0),
            float(economic.get("annual_excess_vs_universe") or 0.0),
        )
        sharpe = float(economic.get("sharpe") or 0.0)
        fold_positive = float(np.mean(np.array(fold_ics) > 0)) if fold_ics else 0.0
        turnover_efficiency = 1 - float(np.clip(economic.get("turnover", 1.0), 0, 1))
        complexity_penalty = 0.03 * float(np.clip((factor_count - 8) / 22, 0, 1))
        return (
            0.30 * clip(rank_ic / 0.03)
            + 0.15 * clip(icir / 0.5)
            + 0.10 * clip((positive - 0.5) / 0.05)
            + 0.15 * clip(net_excess / 0.10)
            + 0.10 * clip(sharpe / 0.8)
            + 0.10 * fold_positive
            + 0.05 * float(np.clip(coverage / 0.9, 0, 1))
            + 0.05 * turnover_efficiency
            - complexity_penalty
        )

    @staticmethod
    def _selection_backtest(frame: pl.DataFrame, spec: MLSearchSpec) -> dict[str, Any]:
        dates = sorted(frame["date"].unique().to_list())[::spec.target.horizon]
        strategy: list[float] = []
        index_returns: list[float] = []
        universe_returns: list[float] = []
        turnovers: list[float] = []
        previous: set[str] | None = None
        buy_rate = spec.costs.commission_pct + spec.costs.slippage_bps / 10_000
        sell_rate = buy_rate + spec.costs.stamp_tax_pct
        for day in dates:
            group = frame.filter(pl.col("date") == day).sort("prediction", descending=True)
            if group.is_empty():
                continue
            top = group.head(spec.costs.top_n)
            current = set(top["symbol"].to_list())
            turnover = 1.0 if previous is None else 1 - len(current & previous) / max(1, len(current))
            cost = buy_rate if previous is None else turnover * (buy_rate + sell_rate)
            strategy.append(float(top["forward_return"].mean()) - cost)
            index_returns.append(float(group["benchmark_return"].mean()))
            universe_returns.append(float(group["forward_return"].mean()))
            turnovers.append(turnover)
            previous = current
        if not strategy:
            return {
                "annual_return": 0.0, "annual_excess_vs_index": 0.0,
                "annual_excess_vs_universe": 0.0, "sharpe": 0.0, "turnover": 1.0,
            }
        values = np.asarray(strategy, dtype=float)
        periods = 252 / spec.target.horizon

        def annualized(items: list[float]) -> float:
            product = float(np.prod(1 + np.asarray(items, dtype=float)))
            return max(product, 1e-12) ** (periods / max(1, len(items))) - 1

        annual = annualized(strategy)
        index_annual = annualized(index_returns)
        universe_annual = annualized(universe_returns)
        sharpe = float(np.mean(values) / np.std(values, ddof=1) * np.sqrt(periods)) \
            if len(values) > 1 and np.std(values, ddof=1) > 1e-12 else 0.0
        return {
            "periods": len(values), "annual_return": annual,
            "annual_excess_vs_index": annual - index_annual,
            "annual_excess_vs_universe": annual - universe_annual,
            "sharpe": sharpe, "turnover": float(np.mean(turnovers)),
        }

    @staticmethod
    def _fit(
        algorithm: str, features: list[str], params: dict[str, Any],
        spec: MLSearchSpec, train: pl.DataFrame, validation: pl.DataFrame,
        cancelled: threading.Event,
    ):
        train = train.drop_nulls(features)
        validation = validation.drop_nulls(features)
        if train.is_empty() or validation.is_empty():
            raise ValueError("完整特征训练或验证样本为空")
        lower, upper = train["target"].quantile(0.01), train["target"].quantile(0.99)
        return get_adapter(algorithm).fit(
            train.select(features).to_numpy(), train["target"].clip(lower, upper).to_numpy(),
            validation.select(features).to_numpy(), validation["target"].to_numpy(),
            train["sample_weight"].to_numpy(), params, spec.device, spec.seed, cancelled,
        )

    def _resolve_factors(self, spec: MLSearchSpec):
        return self._resolve_factor_pool(spec)[0]

    def _resolve_factor_pool(self, spec: MLSearchSpec):
        excluded = {item.id for item in spec.excluded_factors}
        required = {item.id for item in spec.required_factors}
        result = []
        rejected: list[dict[str, Any]] = []
        for reference in spec.factor_pool:
            factor = self.factors.get_version(reference.id, reference.version)
            if reference.id in excluded:
                rejected.append({
                    "factor_id": reference.id, "status": "rejected",
                    "reason": "用户排除", "coverage": None, "selection_rank": 0.0,
                })
                continue
            reason = None
            if spec.asset_type not in factor.asset_types:
                reason = f"不支持资产类型 {spec.asset_type}"
            elif factor.frequency != "1d":
                reason = "智能训练第一阶段只支持日频因子"
            elif factor.authoring_type == "model":
                reason = "模型因子暂不允许进入 AutoML, 避免堆叠泄漏"
            elif not factor.point_in_time:
                reason = "因子包含非时点正确的历史快照数据"
            elif factor.authoring_type == "python" and not factor.trusted:
                reason = "Python 因子尚未确认信任"
            if reason:
                if reference.id in required:
                    raise ValueError(f"必选因子 {reference.id} 不可用: {reason}")
                rejected.append({
                    "factor_id": reference.id, "status": "rejected",
                    "reason": reason, "coverage": None, "selection_rank": 0.0,
                })
                continue
            result.append(factor)
        if len(result) < spec.min_features:
            raise ValueError(
                f"可用因子只有 {len(result)} 个, 少于最小值 {spec.min_features}"
            )
        return result, rejected

    @staticmethod
    def _model_spec(
        spec: MLSearchSpec, algorithm: str, features: list[str],
        versions: dict[str, str], *, model_id: str,
    ) -> ModelSpec:
        return ModelSpec(
            id=model_id, name=spec.name, algorithm=algorithm,
            asset_type=spec.asset_type, symbols=spec.symbols,
            features=features, feature_versions=versions,
            start=spec.start, end=spec.end, target=spec.target,
            walk_forward=spec.walk_forward, device=spec.device,
            params={}, seed=spec.seed, universe_filters=spec.universe_filters,
        )

    @staticmethod
    def _write_artifacts(
        run_dir: Path, quality: list[dict[str, Any]], clusters: list[dict[str, Any]],
        trials: list[dict[str, Any]], folds: list[dict[str, Any]],
        backtest: dict[str, Any],
    ) -> None:
        if quality:
            columns = {
                "factor_id": None, "status": None, "reason": None,
                "coverage": None, "unique_values": None, "standard_deviation": None,
                "extreme_rate": None,
                "rank_ic": None, "icir": None, "ic_positive_rate": None,
                "top_bottom_return": None, "annual_stability": {},
                "selection_rank": 0.0,
            }
            normalized = [{**columns, **item} for item in quality]
            pl.DataFrame(normalized, infer_schema_length=None).write_parquet(
                run_dir / "factor_quality.parquet"
            )
        for name, value in {
            "correlation_clusters.json": clusters,
            "search_trials.json": trials,
            "fold_selections.json": folds,
            "selection_backtest.json": backtest,
        }.items():
            (run_dir / name).write_text(
                json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )

    @staticmethod
    def _check_cancelled(cancelled: threading.Event) -> None:
        if cancelled.is_set():
            raise InterruptedError("智能训练已取消")
