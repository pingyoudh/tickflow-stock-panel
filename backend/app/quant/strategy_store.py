"""Version-locked multi-factor strategy specifications."""
from __future__ import annotations

from pathlib import Path

from app.quant.factors import FactorRegistry
from app.quant.models import QuantStrategySpec


class QuantStrategyStore:
    def __init__(self, data_dir: Path, factors: FactorRegistry) -> None:
        self.root = data_dir / "user_data" / "quant" / "strategies"
        self.root.mkdir(parents=True, exist_ok=True)
        self.factors = factors

    def list(self) -> list[QuantStrategySpec]:
        result = []
        for path in sorted(self.root.glob("*.json")):
            try:
                result.append(QuantStrategySpec.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return result

    def get(self, strategy_id: str) -> QuantStrategySpec:
        path = self.root / f"{strategy_id}.json"
        if not path.exists():
            raise ValueError(f"量化策略不存在: {strategy_id}")
        return QuantStrategySpec.model_validate_json(path.read_text(encoding="utf-8"))

    def upsert(self, spec: QuantStrategySpec) -> QuantStrategySpec:
        for reference in spec.factors:
            factor = self.factors.get(reference.factor_id)
            if factor.version != reference.factor_version:
                raise ValueError(
                    f"因子 {factor.id} 版本不匹配: 策略={reference.factor_version}, 当前={factor.version}"
                )
            if spec.asset_type not in factor.asset_types:
                raise ValueError(f"因子 {factor.id} 不支持资产类型 {spec.asset_type}")
        (self.root / f"{spec.id}.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        return spec

    def delete(self, strategy_id: str) -> bool:
        path = self.root / f"{strategy_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True
