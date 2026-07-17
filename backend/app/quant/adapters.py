"""Optional ML adapters with recorded device fallback."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class FitResult:
    model: Any
    actual_device: str
    library_version: str
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)
    best_iteration: int | None = None


class ModelAdapter(Protocol):
    algorithm: str

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray,
            y_validation: np.ndarray, sample_weight: np.ndarray, params: dict[str, Any],
            device: str, seed: int, cancelled: threading.Event | None = None) -> FitResult: ...
    def predict(self, model: Any, values: np.ndarray) -> np.ndarray: ...
    def save(self, model: Any, path: Path) -> None: ...
    def load(self, path: Path) -> Any: ...
    def feature_importance(self, model: Any, feature_names: list[str]) -> dict[str, dict[str, float]]: ...


def _cpu_threads() -> int:
    return max(1, (os.cpu_count() or 4) - 2)


def _nvidia_info() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        name, memory, driver = [part.strip() for part in proc.stdout.splitlines()[0].split(",")]
        return {"available": True, "name": name, "memory_mb": int(memory), "driver": driver}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def ml_capabilities() -> dict[str, Any]:
    gpu = _nvidia_info()
    result: dict[str, Any] = {"gpu": gpu, "cpu_threads": _cpu_threads(), "algorithms": {}}
    for package in ("lightgbm", "xgboost", "sklearn", "optuna", "joblib"):
        available = importlib.util.find_spec(package) is not None
        version = None
        if available:
            try:
                module = __import__(package)
                version = getattr(module, "__version__", None)
            except Exception:
                available = False
        item = {"installed": available, "version": version}
        if package == "lightgbm":
            item.update({"gpu_backend": "opencl", "gpu_candidate": available and gpu["available"]})
            result["algorithms"][package] = item
        elif package == "xgboost":
            item.update({"gpu_backend": "cuda", "gpu_candidate": available and gpu["available"]})
            result["algorithms"][package] = item
        else:
            result[package] = item
    sklearn_item = result.get("sklearn", {})
    result["algorithms"]["elastic_net"] = {
        "installed": bool(sklearn_item.get("installed")),
        "version": sklearn_item.get("version"),
        "gpu_backend": "cpu",
        "gpu_candidate": False,
    }
    return result


class LightGBMAdapter:
    algorithm = "lightgbm"

    @staticmethod
    def _module():
        try:
            import lightgbm as lgb
            return lgb
        except ImportError as exc:
            raise RuntimeError("LightGBM 未安装, 请使用 `uv sync --extra ml` 安装机器学习依赖") from exc

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray,
            y_validation: np.ndarray, sample_weight: np.ndarray, params: dict[str, Any],
            device: str, seed: int, cancelled: threading.Event | None = None) -> FitResult:
        lgb = self._module()
        baseline = {
            "objective": "regression_l1", "n_estimators": 2000, "learning_rate": 0.03,
            "num_leaves": 31, "max_depth": -1, "min_child_samples": 40,
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "random_state": seed, "n_jobs": _cpu_threads(), "verbosity": -1,
        }
        baseline.update(params)
        candidates = ["gpu", "cpu"] if device in {"auto", "gpu"} else ["cpu"]
        warnings: list[str] = []
        started = time.perf_counter()
        callbacks = [lgb.early_stopping(100, verbose=False)]
        if cancelled is not None:
            def check_cancelled(_environment) -> None:
                if cancelled.is_set():
                    raise InterruptedError("训练已取消")

            check_cancelled.order = 0
            check_cancelled.before_iteration = True
            callbacks.append(check_cancelled)
        for actual in candidates:
            current = {**baseline, "device_type": actual}
            model = lgb.LGBMRegressor(**current)
            try:
                model.fit(
                    x_train, y_train, sample_weight=sample_weight,
                    eval_set=[(x_validation, y_validation)], eval_metric="l2",
                    callbacks=callbacks,
                )
                return FitResult(
                    model=model, actual_device="opencl" if actual == "gpu" else "cpu",
                    library_version=lgb.__version__, elapsed_seconds=time.perf_counter() - started,
                    warnings=warnings, best_iteration=getattr(model, "best_iteration_", None),
                )
            except InterruptedError:
                raise
            except Exception as exc:
                if actual == "cpu":
                    raise
                warnings.append(f"LightGBM OpenCL GPU 不可用, 已回退 CPU: {exc}")
        raise RuntimeError("LightGBM 训练未产生模型")

    def predict(self, model: Any, values: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(values), dtype=float)

    def save(self, model: Any, path: Path) -> None:
        booster = getattr(model, "booster_", model)
        booster.save_model(str(path))

    def load(self, path: Path) -> Any:
        return self._module().Booster(model_file=str(path))

    def feature_importance(self, model: Any, feature_names: list[str]) -> dict[str, dict[str, float]]:
        booster = getattr(model, "booster_", model)
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        return {name: {"gain": float(gain[i]), "split": float(split[i])} for i, name in enumerate(feature_names)}


class XGBoostAdapter:
    algorithm = "xgboost"

    @staticmethod
    def _module():
        try:
            import xgboost as xgb
            return xgb
        except ImportError as exc:
            raise RuntimeError("XGBoost 未安装, 请使用 `uv sync --extra ml` 安装机器学习依赖") from exc

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray,
            y_validation: np.ndarray, sample_weight: np.ndarray, params: dict[str, Any],
            device: str, seed: int, cancelled: threading.Event | None = None) -> FitResult:
        xgb = self._module()
        baseline = {
            "objective": "reg:squarederror", "tree_method": "hist", "n_estimators": 2000,
            "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 5,
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "random_state": seed, "n_jobs": _cpu_threads(), "early_stopping_rounds": 100,
        }
        baseline.update(params)
        if cancelled is not None:
            class CancelTraining(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log) -> bool:
                    if cancelled.is_set():
                        raise InterruptedError("训练已取消")
                    return False

            baseline["callbacks"] = [*baseline.get("callbacks", []), CancelTraining()]
        candidates = ["cuda", "cpu"] if device in {"auto", "gpu"} else ["cpu"]
        warnings: list[str] = []
        started = time.perf_counter()
        for actual in candidates:
            model = xgb.XGBRegressor(**{**baseline, "device": actual})
            try:
                model.fit(
                    x_train, y_train, sample_weight=sample_weight,
                    eval_set=[(x_validation, y_validation)], verbose=False,
                )
                return FitResult(
                    model=model, actual_device=actual, library_version=xgb.__version__,
                    elapsed_seconds=time.perf_counter() - started, warnings=warnings,
                    best_iteration=getattr(model, "best_iteration", None),
                )
            except InterruptedError:
                raise
            except Exception as exc:
                if actual == "cpu":
                    raise
                warnings.append(f"XGBoost CUDA 不可用或显存不足, 已回退 CPU: {exc}")
        raise RuntimeError("XGBoost 训练未产生模型")

    def predict(self, model: Any, values: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(values), dtype=float)

    def save(self, model: Any, path: Path) -> None:
        model.save_model(str(path))

    def load(self, path: Path) -> Any:
        model = self._module().XGBRegressor()
        model.load_model(str(path))
        return model

    def feature_importance(self, model: Any, feature_names: list[str]) -> dict[str, dict[str, float]]:
        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        split = booster.get_score(importance_type="weight")
        return {
            name: {"gain": float(gain.get(f"f{i}", gain.get(name, 0.0))),
                   "split": float(split.get(f"f{i}", split.get(name, 0.0)))}
            for i, name in enumerate(feature_names)
        }


class ElasticNetAdapter:
    algorithm = "elastic_net"

    @staticmethod
    def _modules():
        try:
            import joblib
            import sklearn
            from sklearn.linear_model import ElasticNet
            from sklearn.preprocessing import StandardScaler
            return joblib, sklearn, ElasticNet, StandardScaler
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn/joblib 未安装, 请使用 `uv sync --extra ml` 安装机器学习依赖"
            ) from exc

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray,
            y_validation: np.ndarray, sample_weight: np.ndarray, params: dict[str, Any],
            device: str, seed: int, cancelled: threading.Event | None = None) -> FitResult:
        joblib, sklearn, elastic_net, scaler_type = self._modules()
        del joblib, x_validation, y_validation
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("训练已取消")
        baseline = {
            "alpha": 0.001, "l1_ratio": 0.5, "max_iter": 5000,
            "tol": 1e-4, "selection": "cyclic", "random_state": seed,
        }
        baseline.update(params)
        started = time.perf_counter()
        scaler = scaler_type()
        transformed = scaler.fit_transform(x_train)
        model = elastic_net(**baseline)
        model.fit(transformed, y_train, sample_weight=sample_weight)
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("训练已取消")
        warnings = ["ElasticNet 仅使用 CPU"] if device == "gpu" else []
        return FitResult(
            model={"scaler": scaler, "model": model}, actual_device="cpu",
            library_version=sklearn.__version__, elapsed_seconds=time.perf_counter() - started,
            warnings=warnings,
        )

    def predict(self, model: Any, values: np.ndarray) -> np.ndarray:
        transformed = model["scaler"].transform(values)
        return np.asarray(model["model"].predict(transformed), dtype=float)

    def save(self, model: Any, path: Path) -> None:
        joblib, *_ = self._modules()
        joblib.dump(model, path)

    def load(self, path: Path) -> Any:
        joblib, *_ = self._modules()
        return joblib.load(path)

    def feature_importance(self, model: Any, feature_names: list[str]) -> dict[str, dict[str, float]]:
        coefficients = np.asarray(model["model"].coef_, dtype=float)
        return {
            name: {"gain": float(abs(coefficients[index])), "split": 0.0}
            for index, name in enumerate(feature_names)
        }


def model_artifact_name(algorithm: str) -> str:
    return {
        "elastic_net": "model.joblib",
        "lightgbm": "model.txt",
        "xgboost": "model.json",
    }.get(algorithm, "model.bin")


def get_adapter(algorithm: str) -> ModelAdapter:
    if algorithm == "elastic_net":
        return ElasticNetAdapter()
    if algorithm == "lightgbm":
        return LightGBMAdapter()
    if algorithm == "xgboost":
        return XGBoostAdapter()
    raise ValueError(f"不支持的算法: {algorithm}")


def write_model_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
