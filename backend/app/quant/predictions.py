"""Post-close predictions for explicitly published immutable models."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from app.quant.adapters import get_adapter
from app.quant.dataset import MLDatasetBuilder
from app.quant.model_registry import ModelRegistry
from app.quant.models import ModelSpec


class PredictionService:
    def __init__(self, data_dir: Path, models: ModelRegistry, datasets: MLDatasetBuilder) -> None:
        self.root = data_dir / "user_data" / "quant" / "predictions"
        self.models = models
        self.datasets = datasets

    def generate(self, version: str, as_of: date | None = None) -> dict[str, Any]:
        metadata = self.models.get(version)
        if metadata["status"] != "published":
            raise ValueError("只有手动发布的模型才能生成盘后预测")
        spec = ModelSpec.model_validate(metadata["spec"])
        target_date = as_of or date.today()
        if target_date <= spec.start:
            raise ValueError("预测日期必须晚于训练开始日期")
        spec = spec.model_copy(update={"end": target_date})
        frame = self.datasets.build_latest_features(spec)
        if frame.is_empty():
            raise ValueError("预测日期没有可用特征")
        schema_features = metadata["schema"]["features"]
        if schema_features != spec.features or any(name not in frame.columns for name in schema_features):
            raise ValueError("模型特征 schema 与当前面板不一致")
        expected_dtypes = metadata["schema"].get("dtypes", {})
        mismatched = [
            name for name in schema_features
            if expected_dtypes.get(name) and str(frame.schema[name]) != expected_dtypes[name]
        ]
        if mismatched:
            raise ValueError(f"模型特征 dtype 漂移: {mismatched}")
        coverage_expr = pl.sum_horizontal([
            pl.col(name).is_not_null().cast(pl.Float64) for name in schema_features
        ]) / len(schema_features)
        frame = frame.with_columns(coverage_expr.alias("feature_coverage"))
        valid = frame.filter(pl.col("feature_coverage") >= 1.0)
        if valid.is_empty():
            raise ValueError("特征覆盖率不足, 没有生成预测; 不会复用旧分数")
        adapter = get_adapter(metadata["algorithm"])
        model = adapter.load(self.models.model_path(version))
        prediction = adapter.predict(model, valid.select(schema_features).to_numpy())
        output = valid.select(["symbol", "date", "feature_coverage"]).with_columns(
            pl.lit(version).alias("model_version"), pl.Series("prediction", prediction)
        ).with_columns(
            (pl.col("prediction").rank(method="average") / pl.len()).alias("rank")
        ).select(["symbol", "date", "model_version", "prediction", "rank", "feature_coverage"])
        output_date = output["date"][0]
        directory = self.root / version / f"date={output_date}"
        directory.mkdir(parents=True, exist_ok=True)
        output.write_parquet(directory / "part.parquet")
        psi = self._psi(prediction, metadata.get("training", {}).get("reference_prediction_quantiles"))
        warnings = ["预测分布 PSI 超过 0.25"] if psi is not None and psi > 0.25 else []
        return {
            "model_version": version, "date": str(output_date), "rows": output.height,
            "universe_rows": frame.height, "coverage": output.height / max(1, frame.height),
            "prediction_min": float(np.min(prediction)), "prediction_max": float(np.max(prediction)),
            "prediction_mean": float(np.mean(prediction)), "psi": psi, "warnings": warnings,
        }

    def list(self, version: str | None = None, target_date: date | None = None) -> pl.DataFrame:
        root = self.root / version if version else self.root
        files = list(root.rglob("*.parquet")) if root.exists() else []
        if not files:
            return pl.DataFrame()
        frame = pl.read_parquet(files)
        if target_date is not None:
            frame = frame.filter(pl.col("date") == target_date)
        return frame.sort(["date", "rank"], descending=[True, True])

    def dates(self, version: str) -> list[dict[str, Any]]:
        frame = self.list(version)
        if frame.is_empty():
            return []
        quantiles = self.models.get(version).get("training", {}).get(
            "reference_prediction_quantiles"
        )
        result = []
        for group in frame.partition_by("date", maintain_order=True):
            values = group["prediction"].to_numpy()
            psi = self._psi(values, quantiles)
            result.append({
                "date": group["date"][0], "rows": group.height,
                "coverage": float(group["feature_coverage"].mean()),
                "prediction_min": float(np.min(values)),
                "prediction_max": float(np.max(values)),
                "prediction_mean": float(np.mean(values)), "psi": psi,
                "warnings": ["预测分布 PSI 超过 0.25"] if psi is not None and psi > 0.25 else [],
            })
        return sorted(result, key=lambda item: item["date"], reverse=True)

    def query(
        self,
        version: str,
        target_date: date | None = None,
        search: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        frame = self.list(version, target_date)
        if frame.is_empty():
            return {"predictions": [], "total": 0, "date": None, "summary": None}
        selected_date = target_date or frame["date"].max()
        frame = frame.filter(pl.col("date") == selected_date)
        metadata = self.models.get(version)
        instruments = self.datasets.repo.get_instruments_asset(metadata["spec"]["asset_type"])
        if not instruments.is_empty() and "name" in instruments.columns:
            frame = frame.join(
                instruments.select(["symbol", "name"]).unique("symbol"),
                on="symbol", how="left",
            )
        elif "name" not in frame.columns:
            frame = frame.with_columns(pl.lit("").alias("name"))
        if search.strip():
            needle = search.strip().upper()
            frame = frame.filter(
                pl.col("symbol").str.to_uppercase().str.contains(needle, literal=True)
                | pl.col("name").fill_null("").str.to_uppercase().str.contains(needle, literal=True)
            )
        total = frame.height
        page = frame.sort("rank", descending=True).slice(offset, limit)
        predictions = page.to_dicts()
        summary = {
            "date": str(selected_date),
            "rows": total,
            "coverage": float(frame["feature_coverage"].mean()) if total else 0.0,
            "prediction_min": float(frame["prediction"].min()) if total else None,
            "prediction_max": float(frame["prediction"].max()) if total else None,
            "prediction_mean": float(frame["prediction"].mean()) if total else None,
        }
        matching_date = next(
            (item for item in self.dates(version) if item["date"] == selected_date), None
        )
        if matching_date:
            summary.update({"psi": matching_date["psi"], "warnings": matching_date["warnings"]})
        return {
            "predictions": predictions, "total": total, "date": str(selected_date),
            "summary": summary,
        }

    @staticmethod
    def _psi(values: np.ndarray, quantiles: list[float] | None) -> float | None:
        if not quantiles or len(quantiles) < 3:
            return None
        edges = np.asarray(quantiles, dtype=float)
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(values, bins=edges)
        actual = np.maximum(counts / max(1, counts.sum()), 1e-6)
        expected = np.full(len(counts), 1 / len(counts))
        return float(np.sum((actual - expected) * np.log(actual / expected)))
