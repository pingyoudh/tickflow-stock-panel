"""Persistent model research specifications."""
from __future__ import annotations

from pathlib import Path

from app.quant.models import ModelSpec


class ModelSpecStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "user_data" / "quant" / "specs"
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[ModelSpec]:
        result = []
        for path in sorted(self.root.glob("*.json")):
            try:
                result.append(ModelSpec.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return result

    def upsert(self, spec: ModelSpec) -> ModelSpec:
        (self.root / f"{spec.id}.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        return spec

    def get(self, spec_id: str) -> ModelSpec:
        path = self.root / f"{spec_id}.json"
        if not path.exists():
            raise ValueError(f"模型规格不存在: {spec_id}")
        return ModelSpec.model_validate_json(path.read_text(encoding="utf-8"))

    def delete(self, spec_id: str) -> bool:
        path = self.root / f"{spec_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True
