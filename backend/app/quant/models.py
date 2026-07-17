"""Public data contracts for the quant research domain."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

AssetType = Literal["stock", "etf"]
Frequency = Literal["1d", "1m", "5m", "15m", "30m", "60m"]
MLAlgorithm = Literal["elastic_net", "lightgbm", "xgboost"]


class FactorPreprocess(BaseModel):
    drop_missing: bool = True
    winsorize_mad: float | None = Field(default=3.0, gt=0)
    normalize: Literal["zscore", "rank", "none"] = "zscore"
    neutralize: list[str] = Field(default_factory=list)


class FactorDefinition(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    family: str = Field(default="other", min_length=1, max_length=32)
    version: str = ""
    frequency: Frequency = "1d"
    asset_types: list[AssetType] = Field(default_factory=lambda: ["stock", "etf"])
    inputs: list[str] = Field(default_factory=list)
    warmup: int = Field(default=0, ge=0, le=5000)
    direction: Literal[1, -1] = 1
    authoring_type: Literal["builtin", "declarative", "python", "model"] = "declarative"
    params: dict[str, Any] = Field(default_factory=dict)
    preprocess: FactorPreprocess = Field(default_factory=FactorPreprocess)
    point_in_time: bool = True
    expression: dict[str, Any] | None = None
    trusted: bool = False
    readonly: bool = False

    @model_validator(mode="after")
    def validate_expression(self) -> FactorDefinition:
        if self.authoring_type == "declarative" and not self.expression:
            raise ValueError("声明式因子必须提供 expression")
        return self


class ResearchPanelSpec(BaseModel):
    asset_type: AssetType = "stock"
    frequency: Frequency = "1d"
    symbols: list[str] | None = None
    start: date
    end: date
    fields: list[str] = Field(default_factory=list)
    warmup: int = Field(default=0, ge=0, le=5000)
    ext_datasets: list[str] = Field(default_factory=list)
    max_rows: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> ResearchPanelSpec:
        if self.start > self.end:
            raise ValueError("start 不能晚于 end")
        if self.frequency != "1d" and not self.symbols:
            raise ValueError("分钟研究必须显式指定股票池")
        return self


class TargetSpec(BaseModel):
    horizon: Literal[1, 5, 10, 20] = 5
    benchmark_mode: Literal["index", "cross_section_mean"] = "index"
    benchmark_symbol: str | None = "000300.SH"

    @model_validator(mode="after")
    def validate_benchmark(self) -> TargetSpec:
        if self.benchmark_mode == "index" and not self.benchmark_symbol:
            raise ValueError("指数超额收益必须指定 benchmark_symbol")
        return self


class WalkForwardSpec(BaseModel):
    train_days: int = Field(default=756, ge=60)
    validation_days: int = Field(default=126, ge=20)
    test_days: int = Field(default=126, ge=20)
    step_days: int = Field(default=126, ge=1)


class TuningSpec(BaseModel):
    enabled: bool = False
    max_trials: int = Field(default=20, ge=1, le=100)


class FactorRef(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    version: str = Field(min_length=1, max_length=128)


class ModelSpec(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    name: str = Field(min_length=1, max_length=100)
    algorithm: MLAlgorithm = "lightgbm"
    asset_type: AssetType = "stock"
    symbols: list[str] | None = None
    features: list[str] = Field(min_length=1)
    feature_versions: dict[str, str] = Field(default_factory=dict)
    start: date
    end: date
    target: TargetSpec = Field(default_factory=TargetSpec)
    walk_forward: WalkForwardSpec = Field(default_factory=WalkForwardSpec)
    tuning: TuningSpec = Field(default_factory=TuningSpec)
    device: Literal["auto", "cpu", "gpu"] = "auto"
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    universe_filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_spec(self) -> ModelSpec:
        if self.start >= self.end:
            raise ValueError("训练开始日期必须早于结束日期")
        if self.asset_type == "etf" and "target" not in self.model_fields_set:
            raise ValueError("ETF 训练必须显式选择基准或股票池截面平均收益")
        if self.asset_type == "etf" and self.target.benchmark_mode == "index" and not self.target.benchmark_symbol:
            raise ValueError("ETF 指数超额标签必须显式指定基准")
        if self.feature_versions and not set(self.feature_versions).issubset(self.features):
            raise ValueError("feature_versions 只能包含 features 中的因子")
        return self


class MLSearchCostSpec(BaseModel):
    top_n: int = Field(default=10, ge=1, le=100)
    commission_pct: float = Field(default=0.0002, ge=0, le=0.02)
    stamp_tax_pct: float = Field(default=0.0005, ge=0, le=0.02)
    slippage_bps: float = Field(default=5.0, ge=0, le=500)


class MLSearchSpec(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    name: str = Field(min_length=1, max_length=100)
    asset_type: AssetType = "stock"
    symbols: list[str] | None = None
    start: date
    end: date
    target: TargetSpec = Field(default_factory=TargetSpec)
    factor_pool: list[FactorRef] = Field(min_length=1)
    required_factors: list[FactorRef] = Field(default_factory=list)
    excluded_factors: list[FactorRef] = Field(default_factory=list)
    algorithms: list[MLAlgorithm] = Field(
        default_factory=lambda: ["elastic_net", "lightgbm", "xgboost"], min_length=1
    )
    budget: Literal["quick", "standard", "overnight"] = "standard"
    min_features: int = Field(default=8, ge=1, le=30)
    max_features: int = Field(default=30, ge=1, le=30)
    shortlist_limit: int = Field(default=80, ge=8, le=200)
    inner_folds: int = Field(default=3, ge=2, le=5)
    inner_validation_days: int = Field(default=63, ge=20, le=126)
    walk_forward: WalkForwardSpec = Field(default_factory=WalkForwardSpec)
    costs: MLSearchCostSpec = Field(default_factory=MLSearchCostSpec)
    device: Literal["auto", "cpu", "gpu"] = "auto"
    seed: int = 42
    universe_filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_search(self) -> MLSearchSpec:
        if self.start >= self.end:
            raise ValueError("搜索开始日期必须早于结束日期")
        if self.min_features > self.max_features:
            raise ValueError("min_features 不能大于 max_features")
        pool = {item.id: item.version for item in self.factor_pool}
        if len(pool) != len(self.factor_pool):
            raise ValueError("factor_pool 中的因子 ID 不能重复")
        required = {item.id: item.version for item in self.required_factors}
        excluded = {item.id: item.version for item in self.excluded_factors}
        if not set(required).issubset(pool):
            raise ValueError("必选因子必须包含在 factor_pool 中")
        if set(required) & set(excluded):
            raise ValueError("同一因子不能同时必选和排除")
        if any(pool.get(name) != version for name, version in {**required, **excluded}.items()):
            raise ValueError("必选/排除因子版本必须与 factor_pool 一致")
        if len(required) > self.max_features:
            raise ValueError("必选因子数量不能超过 max_features")
        if len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("algorithms 不能重复")
        if self.asset_type == "etf" and "target" not in self.model_fields_set:
            raise ValueError("ETF 智能训练必须显式选择基准或截面平均收益")
        return self


class MLBacktestSpec(BaseModel):
    model_version: str = Field(min_length=1)
    top_n: int = Field(default=10, ge=1, le=100)
    rebalance_days: int | None = Field(default=None, ge=1, le=252)
    weighting: Literal["equal", "score"] = "equal"
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    commission_pct: float = Field(default=0.0002, ge=0, le=0.02)
    stamp_tax_pct: float = Field(default=0.0005, ge=0, le=0.02)
    slippage_bps: float = Field(default=5.0, ge=0, le=500)


class StrategyFactorSpec(BaseModel):
    factor_id: str
    factor_version: str
    weight: float = Field(default=1.0, ge=-100, le=100)


class QuantStrategySpec(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    name: str = Field(min_length=1, max_length=100)
    asset_type: AssetType = "stock"
    symbols: list[str] | None = None
    factors: list[StrategyFactorSpec] = Field(min_length=1)
    candidate_mode: Literal["threshold", "top_n"] = "top_n"
    score_threshold: float | None = None
    top_n: int = Field(default=10, ge=1, le=500)
    rebalance: Literal["daily", "weekly", "monthly"] = "weekly"
    entry_rule: Literal["next_open"] = "next_open"
    exit_rule: Literal["rebalance", "score_below_threshold"] = "rebalance"

    @model_validator(mode="after")
    def validate_candidate_rule(self) -> QuantStrategySpec:
        if self.candidate_mode == "threshold" and self.score_threshold is None:
            raise ValueError("阈值候选模式必须设置 score_threshold")
        if sum(abs(item.weight) for item in self.factors) <= 1e-12:
            raise ValueError("多因子权重不能全部为 0")
        return self


class FoldDefinition(BaseModel):
    index: int
    train_dates: list[date]
    validation_dates: list[date]
    test_dates: list[date]

    def summary(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_start": str(self.train_dates[0]),
            "train_end": str(self.train_dates[-1]),
            "validation_start": str(self.validation_dates[0]),
            "validation_end": str(self.validation_dates[-1]),
            "test_start": str(self.test_dates[0]),
            "test_end": str(self.test_dates[-1]),
        }


class ExperimentManifest(BaseModel):
    run_id: str
    kind: Literal["ml_training", "ml_search", "ml_backtest", "factor", "strategy", "portfolio"] = "ml_training"
    status: Literal["queued", "running", "cancelling", "completed", "failed", "cancelled"] = "queued"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    progress: float = 0.0
    message: str = ""
    spec: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
