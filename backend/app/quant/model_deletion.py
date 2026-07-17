"""Dependency-aware permanent deletion for immutable model versions."""
from __future__ import annotations

import hashlib
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import polars as pl

from app.quant.experiments import ExperimentStore
from app.quant.model_registry import ModelRegistry
from app.quant.strategy_store import QuantStrategyStore

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = {"queued", "running", "cancelling"}


class ModelDeletionConflict(ValueError):
    """The model cannot be deleted until a lifecycle dependency is resolved."""


class ModelDeletionService:
    def __init__(
        self,
        data_dir: Path,
        models: ModelRegistry,
        experiments: ExperimentStore,
        strategies: QuantStrategyStore,
    ) -> None:
        self.quant_root = data_dir / "user_data" / "quant"
        self.models = models
        self.experiments = experiments
        self.strategies = strategies
        self.predictions_root = self.quant_root / "predictions"
        self.transaction_root = self.quant_root / ".deleting"

    def impact(self, version: str) -> dict[str, Any]:
        metadata = self.models.get(version)
        source_run_id = metadata.get("source_run_id")
        related_experiments = []
        active_blockers = []
        for manifest in self.experiments.list():
            if not self._experiment_references_model(manifest, version, source_run_id):
                continue
            item = {
                "run_id": manifest.run_id,
                "kind": manifest.kind,
                "status": manifest.status,
                "name": manifest.spec.get("name") or manifest.kind,
            }
            related_experiments.append(item)
            if manifest.status in _ACTIVE_STATUSES:
                active_blockers.append(item)

        factor_id = self._model_factor_id(metadata)
        dependent_strategies = []
        for strategy in self.strategies.list():
            if any(
                reference.factor_version == version or reference.factor_id == factor_id
                for reference in strategy.factors
            ):
                dependent_strategies.append({"id": strategy.id, "name": strategy.name})

        prediction_path = self.predictions_root / version
        prediction_files = list(prediction_path.rglob("*.parquet")) if prediction_path.exists() else []
        prediction_rows = 0
        if prediction_files:
            prediction_rows = int(
                pl.scan_parquet(prediction_files)
                .select(pl.len().alias("rows"))
                .collect()["rows"][0]
            )
        targets = self._target_paths(
            version, related_experiments, dependent_strategies
        )
        return {
            "model_version": version,
            "model_name": metadata.get("name", version),
            "status": metadata.get("status"),
            "source_run_id": source_run_id,
            "model_factor_id": factor_id,
            "experiments": related_experiments,
            "strategies": dependent_strategies,
            "prediction_files": len(prediction_files),
            "prediction_rows": prediction_rows,
            "total_bytes": sum(self._path_size(path) for path in targets),
            "active_blockers": active_blockers,
            "can_delete": metadata.get("status") != "published" and not active_blockers,
        }

    def delete(
        self,
        version: str,
        *,
        confirm_version: str,
        cascade: bool,
    ) -> dict[str, Any]:
        if confirm_version != version:
            raise ValueError("确认模型版本不匹配")
        if not cascade:
            raise ValueError("永久删除必须显式设置 cascade=true")
        impact = self.impact(version)
        if impact["status"] == "published":
            raise ModelDeletionConflict("已发布模型必须先归档再删除")
        if impact["status"] not in {"validated", "archived"}:
            raise ModelDeletionConflict("只有已验证或已归档模型可以删除")
        if impact["active_blockers"]:
            ids = ", ".join(item["run_id"] for item in impact["active_blockers"])
            raise ModelDeletionConflict(f"存在运行中的关联实验, 请先取消: {ids}")

        targets = self._target_paths(
            version, impact["experiments"], impact["strategies"]
        )
        transaction = self.transaction_root / uuid.uuid4().hex
        payload_root = transaction / "payload"
        moved: list[tuple[Path, Path]] = []
        try:
            for source in targets:
                if not source.exists():
                    continue
                relative = source.relative_to(self.quant_root)
                staged = payload_root / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                source.replace(staged)
                moved.append((source, staged))
        except Exception:
            for source, staged in reversed(moved):
                try:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    staged.replace(source)
                except Exception:
                    logger.exception(
                        "model deletion rollback failed: source=%s staged=%s",
                        source,
                        staged,
                    )
            shutil.rmtree(transaction, ignore_errors=True)
            raise

        try:
            shutil.rmtree(transaction)
        except Exception:
            logger.exception(
                "model deletion committed but transaction cleanup is pending: %s",
                transaction,
            )
        return {
            "deleted": True,
            "model_version": version,
            "experiments_deleted": len(impact["experiments"]),
            "strategies_deleted": len(impact["strategies"]),
            "prediction_files_deleted": impact["prediction_files"],
            "bytes_deleted": impact["total_bytes"],
        }

    def _target_paths(
        self,
        version: str,
        experiments: list[dict[str, Any]],
        strategies: list[dict[str, Any]],
    ) -> list[Path]:
        candidates = [
            self.models.root / version,
            self.predictions_root / version,
            *(self.experiments.root / item["run_id"] for item in experiments),
            *(self.strategies.root / f"{item['id']}.json" for item in strategies),
        ]
        result: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            resolved = path.resolve(strict=False)
            if resolved in seen:
                continue
            if self.quant_root.resolve(strict=False) not in resolved.parents:
                raise ValueError(f"删除目标不在量化数据目录: {path}")
            seen.add(resolved)
            result.append(path)
        return result

    @staticmethod
    def _experiment_references_model(manifest, version: str, source_run_id: str | None) -> bool:
        if source_run_id and manifest.run_id == source_run_id:
            return True
        return (
            manifest.spec.get("model_version") == version
            or manifest.result.get("model_version") == version
        )

    @staticmethod
    def _model_factor_id(metadata: dict[str, Any]) -> str:
        version = metadata["version"]
        suffix = hashlib.sha256(version.encode()).hexdigest()[:10]
        return f"ml_{metadata['model_id'][:40]}_{suffix}"

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        if not path.exists():
            return 0
        return sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file()
        )
