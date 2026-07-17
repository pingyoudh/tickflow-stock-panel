"""Immutable local model artifacts and explicit lifecycle transitions."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app.quant.adapters import model_artifact_name


class ModelRegistry:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "user_data" / "quant" / "models"
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, *, spec: dict[str, Any], source_model: Path, schema: dict[str, Any],
                 metrics: dict[str, Any], data_fingerprint: str, training: dict[str, Any],
                 source_run_id: str | None = None) -> dict[str, Any]:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        digest = hashlib.sha256(
            json.dumps({"spec": spec, "fingerprint": data_fingerprint, "stamp": stamp}, sort_keys=True).encode()
        ).hexdigest()[:8]
        version = f"{spec['id']}-{stamp}-{digest}"
        target = self.root / version
        target.mkdir(parents=True, exist_ok=False)
        model_name = model_artifact_name(spec["algorithm"])
        shutil.copy2(source_model, target / model_name)
        payload = {
            "version": version, "model_id": spec["id"], "name": spec["name"],
            "algorithm": spec["algorithm"], "status": "validated", "created_at": datetime.now().isoformat(),
            "published_at": None, "archived_at": None, "model_file": model_name,
            "spec": spec, "schema": schema, "metrics": metrics,
            "data_fingerprint": data_fingerprint, "training": training,
            "source_run_id": source_run_id,
        }
        self._write(target / "metadata.json", payload)
        return payload

    def list(self) -> list[dict[str, Any]]:
        result = []
        for path in self.root.glob("*/metadata.json"):
            try:
                result.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return sorted(result, key=lambda item: item.get("created_at", ""), reverse=True)

    def get(self, version: str) -> dict[str, Any]:
        path = self.root / version / "metadata.json"
        if not path.exists():
            raise ValueError(f"模型版本不存在: {version}")
        return json.loads(path.read_text(encoding="utf-8"))

    def publish(self, version: str) -> dict[str, Any]:
        payload = self.get(version)
        if payload["status"] == "archived":
            raise ValueError("归档模型不能发布")
        payload["status"] = "published"
        payload["published_at"] = datetime.now().isoformat()
        self._write(self.root / version / "metadata.json", payload)
        return payload

    def archive(self, version: str) -> dict[str, Any]:
        payload = self.get(version)
        payload["status"] = "archived"
        payload["archived_at"] = datetime.now().isoformat()
        self._write(self.root / version / "metadata.json", payload)
        return payload

    def model_path(self, version: str) -> Path:
        payload = self.get(version)
        return self.root / version / payload["model_file"]

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temp.replace(path)
