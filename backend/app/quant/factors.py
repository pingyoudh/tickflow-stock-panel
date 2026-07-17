"""Factor registry and safe declarative expression compiler."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, ClassVar

import polars as pl

from app.quant.models import FactorDefinition

BUILTIN_FACTORS: list[dict[str, str]] = [
    {"id": "momentum_5d", "name": "5日动量", "group": "动量", "description": "5日涨跌幅"},
    {"id": "momentum_10d", "name": "10日动量", "group": "动量", "description": "10日涨跌幅"},
    {"id": "momentum_20d", "name": "20日动量", "group": "动量", "description": "20日涨跌幅"},
    {"id": "momentum_30d", "name": "30日动量", "group": "动量", "description": "30日涨跌幅"},
    {"id": "momentum_60d", "name": "60日动量", "group": "动量", "description": "60日涨跌幅"},
    {"id": "rsi_6", "name": "RSI(6)", "group": "超买超卖", "description": "6日相对强弱指标"},
    {"id": "rsi_14", "name": "RSI(14)", "group": "超买超卖", "description": "14日相对强弱指标"},
    {"id": "rsi_24", "name": "RSI(24)", "group": "超买超卖", "description": "24日相对强弱指标"},
    {"id": "annual_vol_20d", "name": "20日波动率", "group": "波动率", "description": "20日年化波动率"},
    {"id": "atr_14", "name": "ATR(14)", "group": "波动率", "description": "14日平均真实波幅"},
    {"id": "vol_ratio_5d", "name": "量比(5日)", "group": "量价", "description": "成交量与5日均量之比"},
    {"id": "turnover_rate", "name": "换手率", "group": "量价", "description": "当日换手率"},
    {"id": "macd_hist", "name": "MACD柱", "group": "趋势", "description": "MACD柱状图"},
    {"id": "kdj_k", "name": "KDJ-K", "group": "趋势", "description": "KDJ K值"},
    {"id": "change_pct", "name": "日涨跌幅", "group": "基础", "description": "当日涨跌幅"},
    {"id": "amplitude", "name": "日振幅", "group": "基础", "description": "当日振幅"},
]


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def builtin_factor_columns() -> list[dict[str, str]]:
    return [
        {"id": item["id"], "label": item["name"], "group": item["group"], "desc": item["description"]}
        for item in BUILTIN_FACTORS
    ]


class FactorExpressionCompiler:
    """Compile a JSON AST into Polars expressions without eval or free-form code."""

    _BINARY: ClassVar[dict[str, Any]] = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "gt": lambda a, b: a > b,
        "gte": lambda a, b: a >= b,
        "lt": lambda a, b: a < b,
        "lte": lambda a, b: a <= b,
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
    }

    @classmethod
    def compile(cls, node: dict[str, Any], available: set[str]) -> pl.Expr:
        if not isinstance(node, dict):
            raise ValueError("expression 节点必须是对象")
        op = node.get("op")
        if op == "field":
            name = str(node.get("name", ""))
            if name not in available:
                raise ValueError(f"未知因子字段: {name}")
            return pl.col(name)
        if op == "literal":
            value = node.get("value")
            if not isinstance(value, (int, float, bool)):
                raise ValueError("literal 只允许数值或布尔值")
            return pl.lit(value)
        if op in cls._BINARY:
            return cls._BINARY[op](
                cls.compile(node.get("left"), available),
                cls.compile(node.get("right"), available),
            )
        value = cls.compile(node.get("value"), available)
        if op == "neg":
            return -value
        if op == "abs":
            return value.abs()
        if op == "log":
            return value.log()
        if op in {"delay", "diff", "return", "rolling_mean", "rolling_std", "rolling_min", "rolling_max"}:
            window = int(node.get("window", 1))
            if window < 1 or window > 5000:
                raise ValueError("window 必须在 1~5000 之间")
            if op == "delay":
                return value.shift(window).over("symbol")
            if op == "diff":
                return value.diff(window).over("symbol")
            if op == "return":
                return value.pct_change(window).over("symbol")
            method = {
                "rolling_mean": value.rolling_mean,
                "rolling_std": value.rolling_std,
                "rolling_min": value.rolling_min,
                "rolling_max": value.rolling_max,
            }[op]
            return method(window_size=window, min_samples=window).over("symbol")
        if op == "corr":
            other = cls.compile(node.get("other"), available)
            window = int(node.get("window", 20))
            return pl.rolling_corr(value, other, window_size=window, min_samples=window).over("symbol")
        if op == "rank":
            return value.rank(method="average").over("date") / pl.len().over("date")
        raise ValueError(f"不支持的 expression op: {op}")


class FactorRegistry:
    def __init__(self, data_dir: Path) -> None:
        self.base = data_dir / "user_data" / "quant" / "factors"
        self.declarative_dir = self.base / "declarative"
        self.code_dir = self.base / "code"
        self.trust_path = self.base / "trusted_code.json"
        self.declarative_dir.mkdir(parents=True, exist_ok=True)
        self.code_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[FactorDefinition]:
        factors = [self._builtin(item) for item in BUILTIN_FACTORS]
        for path in sorted(self.declarative_dir.glob("*.json")):
            try:
                factors.append(FactorDefinition.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        factors.extend(self._list_code())
        factors.extend(self._list_models())
        return factors

    def get(self, factor_id: str) -> FactorDefinition:
        for factor in self.list():
            if factor.id == factor_id:
                return factor
        raise ValueError(f"未知因子: {factor_id}")

    def get_version(self, factor_id: str, version: str) -> FactorDefinition:
        factor = self.get(factor_id)
        if factor.version != version:
            raise ValueError(
                f"因子 {factor_id} 版本不匹配: 请求 {version}, 当前 {factor.version}"
            )
        return factor

    def upsert(self, definition: FactorDefinition) -> FactorDefinition:
        if definition.authoring_type != "declarative":
            raise ValueError("该接口只保存声明式因子")
        FactorExpressionCompiler.compile(definition.expression or {}, set(definition.inputs))
        payload = definition.model_dump(mode="json")
        payload["version"] = _digest({k: v for k, v in payload.items() if k != "version"})
        saved = FactorDefinition.model_validate(payload)
        path = self.declarative_dir / f"{saved.id}.json"
        path.write_text(saved.model_dump_json(indent=2), encoding="utf-8")
        return saved

    def delete(self, factor_id: str) -> bool:
        path = self.declarative_dir / f"{factor_id}.json"
        if not path.exists():
            return False
        path.unlink()
        return True

    def save_code(self, source: str) -> FactorDefinition:
        """Validate metadata without executing code, then persist it disabled."""
        tree = ast.parse(source)
        meta: dict[str, Any] | None = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "FACTOR_META" for target in node.targets
            ):
                meta = ast.literal_eval(node.value)
                break
        if not isinstance(meta, dict) or not meta.get("id"):
            raise ValueError("Python 因子必须声明可静态解析的 FACTOR_META")
        factor_id = str(meta["id"])
        version = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        FactorDefinition.model_validate({
            **meta, "id": factor_id, "version": version,
            "authoring_type": "python", "trusted": False,
        })
        (self.code_dir / f"{factor_id}.py").write_text(source, encoding="utf-8")
        trust = self._trust_map()
        trust.pop(factor_id, None)
        self.trust_path.write_text(json.dumps(trust, indent=2), encoding="utf-8")
        return self.get(factor_id)

    def set_code_trust(self, factor_id: str, trusted: bool) -> FactorDefinition:
        factor = self.get(factor_id)
        if factor.authoring_type != "python":
            raise ValueError("只有 Python 因子需要信任确认")
        trust = self._trust_map()
        if trusted:
            trust[factor_id] = factor.version
        else:
            trust.pop(factor_id, None)
        self.trust_path.parent.mkdir(parents=True, exist_ok=True)
        self.trust_path.write_text(json.dumps(trust, indent=2), encoding="utf-8")
        return self.get(factor_id)

    def evaluate(self, df: pl.DataFrame, factor: FactorDefinition) -> pl.DataFrame:
        if factor.authoring_type == "builtin":
            if factor.id not in df.columns:
                raise ValueError(f"数据面板缺少内置因子列: {factor.id}")
            return df
        if factor.authoring_type == "declarative":
            expr = FactorExpressionCompiler.compile(factor.expression or {}, set(df.columns))
            return df.with_columns(expr.cast(pl.Float64, strict=False).alias(factor.id))
        if factor.authoring_type == "python":
            if not factor.trusted:
                raise ValueError(f"Python 因子 {factor.id} 尚未确认信任")
            path = self.code_dir / f"{factor.id}.py"
            spec = importlib.util.spec_from_file_location(f"quant_factor_{factor.id}", path)
            if spec is None or spec.loader is None:
                raise ValueError(f"无法加载 Python 因子: {factor.id}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = module.compute(df.lazy(), factor.params)
            out = result.collect() if isinstance(result, pl.LazyFrame) else result
            if not isinstance(out, pl.DataFrame) or "value" not in out.columns:
                raise ValueError("Python 因子 compute 必须返回含 value 列的 LazyFrame/DataFrame")
            keys = [c for c in ["symbol", "date", "datetime"] if c in out.columns and c in df.columns]
            if "symbol" not in keys or not ({"date", "datetime"} & set(keys)):
                raise ValueError("Python 因子输出必须包含 symbol 和 date/datetime 键")
            if out.select(keys).n_unique() != out.height:
                raise ValueError("Python 因子输出键必须唯一")
            return df.join(out.select([*keys, pl.col("value").cast(pl.Float64).alias(factor.id)]), on=keys, how="left")
        if factor.authoring_type == "model":
            version = str(factor.params.get("model_version", ""))
            files = list((self.base.parent / "predictions" / version).rglob("*.parquet"))
            if not files:
                raise ValueError(f"模型因子 {factor.id} 尚无已保存预测")
            predictions = pl.read_parquet(files).select([
                "symbol", "date", pl.col("rank").cast(pl.Float64).alias(factor.id)
            ])
            return df.join(predictions, on=["symbol", "date"], how="left")
        raise ValueError(f"因子类型不可计算: {factor.authoring_type}")

    def _builtin(self, item: dict[str, str]) -> FactorDefinition:
        return FactorDefinition(
            id=item["id"], name=item["name"], description=item["description"],
            family=item["group"],
            version=_digest(item), authoring_type="builtin", readonly=True,
            inputs=[item["id"]], expression=None, trusted=True,
        )

    def _trust_map(self) -> dict[str, str]:
        if not self.trust_path.exists():
            return {}
        try:
            return json.loads(self.trust_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _list_code(self) -> list[FactorDefinition]:
        trust = self._trust_map()
        result: list[FactorDefinition] = []
        for path in sorted(self.code_dir.glob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source)
                meta: dict[str, Any] | None = None
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FACTOR_META" for t in node.targets):
                        meta = ast.literal_eval(node.value)
                        break
                if not isinstance(meta, dict):
                    continue
                version = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
                payload = {**meta, "id": meta.get("id", path.stem), "version": version,
                           "authoring_type": "python", "trusted": trust.get(path.stem) == version}
                result.append(FactorDefinition.model_validate(payload))
            except Exception:
                continue
        return result

    def _list_models(self) -> list[FactorDefinition]:
        result: list[FactorDefinition] = []
        for path in (self.base.parent / "models").glob("*/metadata.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "published":
                    continue
                version = payload["version"]
                suffix = hashlib.sha256(version.encode()).hexdigest()[:10]
                result.append(FactorDefinition(
                    id=f"ml_{payload['model_id'][:40]}_{suffix}",
                    name=f"{payload['name']} 发布预测排名",
                    description=f"只读模型因子, 精确版本 {version}", version=version,
                    authoring_type="model", readonly=True, trusted=True,
                    inputs=payload.get("schema", {}).get("features", []),
                    params={"model_version": version}, expression=None,
                    asset_types=[payload["spec"]["asset_type"]],
                ))
            except Exception:
                continue
        return result
