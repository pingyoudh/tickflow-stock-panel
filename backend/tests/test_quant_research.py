from __future__ import annotations

import json
import math
import threading
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from app.backtest.factor import FACTOR_COLUMNS
from app.quant.adapters import ElasticNetAdapter, LightGBMAdapter, XGBoostAdapter
from app.quant.automl import AutoMLSearchEngine
from app.quant.dataset import MLDataset, MLDatasetBuilder, MLSearchDataset
from app.quant.experiments import ExperimentManager, ExperimentStore
from app.quant.factor_cache import FactorValueCache
from app.quant.factors import FactorRegistry
from app.quant.ml_backtest import MLBacktestService
from app.quant.metrics import evaluate_oos
from app.quant.model_center import ModelCenterService
from app.quant.model_deletion import ModelDeletionConflict, ModelDeletionService
from app.quant.model_registry import ModelRegistry
from app.quant.models import (
    FactorDefinition,
    FactorRef,
    MLBacktestSpec,
    MLSearchSpec,
    ModelSpec,
    QuantStrategySpec,
    ResearchPanelSpec,
    StrategyFactorSpec,
    TargetSpec,
    WalkForwardSpec,
)
from app.quant.panel import ResearchPanelBuilder
from app.quant.portfolio import PortfolioOptimizer
from app.quant.splits import assert_no_label_overlap, generate_purged_folds
from app.quant.standard_expression import (
    evaluate_standard_expression,
    import_standard_expression_library,
    parse_standard_expression,
    standard_expression_asset_types,
)
from app.quant.strategy_store import QuantStrategyStore


def test_declarative_factor_dsl_hash_is_stable_and_rejects_unknown_ops(tmp_path: Path):
    registry = FactorRegistry(tmp_path)
    definition = FactorDefinition(
        id="close_delay",
        name="收盘延迟",
        inputs=["close"],
        expression={"op": "delay", "value": {"op": "field", "name": "close"}, "window": 1},
    )
    first = registry.upsert(definition)
    second = registry.upsert(definition)
    assert first.version == second.version
    changed = registry.upsert(definition.model_copy(update={
        "expression": {"op": "delay", "value": {"op": "field", "name": "close"}, "window": 2}
    }))
    assert changed.version != first.version
    with pytest.raises(ValueError, match="不支持"):
        registry.upsert(definition.model_copy(update={"expression": {"op": "eval", "value": {"op": "field", "name": "close"}}}))


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_standard_expression_importer_names_and_marks_blocked(tmp_path: Path):
    root = tmp_path / "std"
    _write_csv(
        root / "候选因子库.csv",
        [
            {
                "序号": "1",
                "算子表达式": "TsMean($close, 5)",
                "类型": "技术-均线",
                "来源": "unit",
                "因子解释": "计算每只股票过去5日收盘价简单均值，刻画5日价格中枢。",
            },
            {
                "序号": "2",
                "算子表达式": "Div($market_value, $np)",
                "类型": "基本面-推导",
                "来源": "unit",
                "因子解释": "市值除以净利润。",
            },
        ],
        ["序号", "算子表达式", "类型", "来源", "因子解释"],
    )
    _write_csv(
        root / "因子筛选" / "因子筛选" / "机器学习因子库.csv",
        [{
            "来源": "候选因子库",
            "原序号": "1",
            "类型": "技术-均线",
            "算子表达式": "TsMean($close, 5)",
            "因子解释": "计算每只股票过去5日收盘价简单均值，刻画5日价格中枢。",
            "入库状态": "入库",
        }],
        ["来源", "原序号", "类型", "算子表达式", "因子解释", "入库状态"],
    )

    stats = import_standard_expression_library(root, dry_run=True)
    factors = {item.source_expression: item for item in stats["factors"]}

    assert stats["unique_expressions"] == 2
    assert factors["TsMean($close, 5)"].name == "5日收盘均线"
    assert factors["TsMean($close, 5)"].enabled is True
    assert factors["TsMean($close, 5)"].compute_status == "ready"
    assert factors["Div($market_value, $np)"].enabled is False
    assert factors["Div($market_value, $np)"].compute_status == "blocked"
    assert "missing_field" in factors["Div($market_value, $np)"].blocked_reason


def test_standard_expression_parser_and_evaluator_representative_ops():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(30)]
    frame = pl.DataFrame({
        "symbol": ["A"] * len(days),
        "date": days,
        "open": [9.0 + i for i in range(30)],
        "high": [10.0 + i for i in range(30)],
        "low": [8.0 + i for i in range(30)],
        "close": [9.5 + i for i in range(30)],
        "volume": [1000.0 + i * 10 for i in range(30)],
        "amount": [10000.0 + i * 100 for i in range(30)],
        "turnover_rate": [1.0 + i * 0.01 for i in range(30)],
    })
    expressions = [
        "Div(TsMean($close, 3), Add(TsStd($close, 3), 1e-6))",
        "Rank(TsPctChange($close, 5))",
        "IfElse(Greater($close, Ref($close, 1)), $volume, Neg($volume))",
        "RSI($close, 6)",
        "ATR($high, $low, $close, 14)",
        "KDJ_K($high, $low, $close, 9)",
    ]
    out = frame
    for index, expression in enumerate(expressions):
        ast = parse_standard_expression(expression)
        out = evaluate_standard_expression(out, ast, output_name=f"f{index}")
        assert out[f"f{index}"].drop_nulls().len() > 0


def test_standard_expression_tswma_handles_null_input():
    days = [date(2025, 1, 1) + timedelta(days=index) for index in range(6)]
    frame = pl.DataFrame({
        "symbol": ["A"] * len(days),
        "date": days,
        "close": [None, 2.0, 3.0, 4.0, 5.0, 6.0],
    })

    out = evaluate_standard_expression(
        frame,
        parse_standard_expression("TsWMA($close, 3)"),
        output_name="factor",
    )

    finite = out["factor"].drop_nulls().drop_nans()
    assert finite.len() == 3
    assert finite[0] == pytest.approx((2.0 + 3.0 * 2 + 4.0 * 3) / 6)


def test_standard_expression_registry_import_is_idempotent_and_state_guard(tmp_path: Path):
    root = tmp_path / "std"
    _write_csv(
        root / "候选因子库.csv",
        [{
            "序号": "1",
            "算子表达式": "TsMean($close, 5)",
            "类型": "技术-均线",
            "来源": "unit",
            "因子解释": "计算每只股票过去5日收盘价简单均值，刻画5日价格中枢。",
        }],
        ["序号", "算子表达式", "类型", "来源", "因子解释"],
    )
    registry = FactorRegistry(tmp_path)
    first = registry.import_standard_expression(root, dry_run=False)
    second = registry.import_standard_expression(root, dry_run=False)
    factors = [item for item in registry.list() if item.origin == "standard_expression"]

    assert first["imported"] == second["imported"] == 1
    assert len(factors) == 1
    disabled = registry.update_state(factors[0].id, enabled=False)
    assert disabled.enabled is False
    enabled = registry.update_state(factors[0].id, enabled=True)
    assert enabled.enabled is True


def test_python_factor_requires_hash_locked_trust(tmp_path: Path):
    registry = FactorRegistry(tmp_path)
    source = """FACTOR_META = {\"id\": \"trusted_local\", \"name\": \"可信本地\", \"inputs\": [\"close\"]}\n\ndef compute(frame, params):\n    return frame.select(\"symbol\", \"date\", frame.collect_schema() and __import__('polars').col(\"close\").alias(\"value\"))\n"""
    saved = registry.save_code(source)
    assert saved.trusted is False
    trusted = registry.set_code_trust(saved.id, True)
    assert trusted.trusted is True
    changed = registry.save_code(source.replace('alias(\"value\")', 'mul(2).alias(\"value\")'))
    assert changed.version != trusted.version
    assert changed.trusted is False


class _BenchmarkRepo:
    def __init__(self, benchmark: pl.DataFrame):
        self.benchmark = benchmark

    def get_index_daily(self, symbol, start, end, columns=None):
        frame = self.benchmark.filter((pl.col("date") >= start) & (pl.col("date") <= end))
        return frame.select(columns) if columns else frame


def test_label_uses_global_t_plus_one_open_and_nth_day_close():
    days = [date(2025, 1, 2) + timedelta(days=i) for i in range(8)]
    benchmark = pl.DataFrame({
        "date": days, "open": [100.0 + i for i in range(8)],
        "close": [100.5 + i for i in range(8)],
    })
    panel = pl.DataFrame({
        "symbol": ["A"] * 8 + ["B"] * 7,
        "date": [*days, days[0], *days[2:]],
        "open": [10.0 + i for i in range(8)] + [20.0 + i for i in range(7)],
        "close": [10.5 + i for i in range(8)] + [20.5 + i for i in range(7)],
        "momentum_20d": [0.1] * 15,
    })
    builder = object.__new__(MLDatasetBuilder)
    builder.repo = _BenchmarkRepo(benchmark)
    spec = ModelSpec(
        id="test_model", name="测试", features=["momentum_20d"], start=days[0], end=days[-1],
        target=TargetSpec(horizon=5, benchmark_symbol="000300.SH"),
    )
    labeled, calendar = builder._add_target(panel, spec)
    row = labeled.filter((pl.col("symbol") == "A") & (pl.col("date") == days[0])).row(0, named=True)
    assert row["forward_return"] == pytest.approx(15.5 / 11.0 - 1)
    assert row["benchmark_return"] == pytest.approx(105.5 / 101.0 - 1)
    assert row["target"] == pytest.approx(row["forward_return"] - row["benchmark_return"])
    suspended = labeled.filter((pl.col("symbol") == "B") & (pl.col("date") == days[0])).row(0, named=True)
    assert suspended["forward_return"] is None
    assert calendar == days


def test_purged_walk_forward_has_horizon_sized_gaps():
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(160)]
    spec = WalkForwardSpec(train_days=60, validation_days=20, test_days=20, step_days=20)
    folds = generate_purged_folds(days, spec, horizon=5)
    assert len(folds) == 3
    for fold in folds:
        assert_no_label_overlap(fold, days, 5)
        positions = {day: i for i, day in enumerate(days)}
        assert positions[fold.validation_dates[0]] - positions[fold.train_dates[-1]] == 6
        assert positions[fold.test_dates[0]] - positions[fold.validation_dates[-1]] == 6


class _MinuteRepo:
    def __init__(self, frame: pl.DataFrame):
        self.frame = frame

    def get_minute_range(self, symbols, start, end, asset_type="stock"):
        return self.frame


def test_panel_limit_uses_local_default_and_allows_request_override(tmp_path: Path):
    builder = ResearchPanelBuilder(_MinuteRepo(pl.DataFrame()), tmp_path, max_rows=10)
    base = ResearchPanelSpec(
        asset_type="stock", frequency="1d", symbols=["A", "B", "C"],
        start=date(2025, 1, 2), end=date(2025, 1, 6),
    )
    estimate = builder.estimate(base)
    assert estimate["estimated_rows"] == 9
    assert estimate["max_rows"] == 10
    assert estimate["allowed"] is True

    override = base.model_copy(update={"max_rows": 8})
    estimate = builder.estimate(override)
    assert estimate["max_rows"] == 8
    assert estimate["allowed"] is False


def test_daily_panel_projects_requested_columns(tmp_path: Path):
    captured = {}

    class Engine:
        def load_panel(self, symbols, start, end, columns, asset_type):
            captured["columns"] = columns
            return pl.DataFrame({
                "symbol": ["A"], "date": [date(2025, 1, 2)],
                "open": [10.0], "momentum_20d": [0.1],
            })

    builder = object.__new__(ResearchPanelBuilder)
    builder.engine = Engine()
    spec = ResearchPanelSpec(
        start=date(2025, 1, 2), end=date(2025, 1, 2),
        fields=["open", "momentum_20d"],
    )
    builder._build_daily(spec)
    assert captured["columns"] == [
        "symbol", "date", "open", "high", "low", "close", "volume", "amount",
        "turnover_rate", "momentum_20d",
    ]


def test_daily_panel_computes_requested_indicator_missing_from_parquet():
    days = [date(2025, 1, 1) + timedelta(days=index) for index in range(20)]

    class Engine:
        def load_panel(self, symbols, start, end, columns, asset_type):
            return pl.DataFrame({
                "symbol": ["A"] * len(days), "date": days,
                "open": list(range(1, 21)), "high": list(range(2, 22)),
                "low": list(range(0, 20)), "close": list(range(1, 21)),
                "volume": [100] * len(days), "amount": [1000] * len(days),
            })

    builder = object.__new__(ResearchPanelBuilder)
    builder.engine = Engine()
    result = builder._build_daily(ResearchPanelSpec(
        start=days[0], end=days[-1], fields=["rsi_14"],
    ))
    assert "rsi_14" in result.columns
    assert result["rsi_14"].drop_nulls().len() > 0


def test_minute_resampling_does_not_cross_lunch_or_day_boundary(tmp_path: Path):
    frame = pl.DataFrame({
        "symbol": ["A"] * 6,
        "datetime": [
            datetime(2025, 1, 2, 9, 31), datetime(2025, 1, 2, 10, 0),
            datetime(2025, 1, 2, 11, 29), datetime(2025, 1, 2, 13, 1),
            datetime(2025, 1, 2, 13, 40), datetime(2025, 1, 3, 9, 31),
        ],
        "open": [1, 2, 3, 10, 11, 20], "high": [2, 3, 4, 11, 12, 21],
        "low": [0, 1, 2, 9, 10, 19], "close": [1.5, 2.5, 3.5, 10.5, 11.5, 20.5],
        "volume": [1] * 6, "amount": [1] * 6,
    })
    builder = ResearchPanelBuilder(_MinuteRepo(frame), tmp_path)
    spec = ResearchPanelSpec(
        asset_type="stock", frequency="60m", symbols=["A"],
        start=date(2025, 1, 2), end=date(2025, 1, 3), max_rows=100,
    )
    result = builder.build(spec)
    first_pm = result.filter(pl.col("datetime").dt.hour() == 13).row(0, named=True)
    assert first_pm["open"] == 10
    assert first_pm["close"] == 11.5
    assert result.filter(pl.col("date") == date(2025, 1, 3))["open"].to_list() == [20]


def test_snapshot_extension_is_rejected_for_history(tmp_path: Path):
    config_dir = tmp_path / "ext_data" / "latest_only"
    config_dir.mkdir(parents=True)
    (config_dir / "config.json").write_text(json.dumps({
        "id": "latest_only", "label": "最新快照", "mode": "snapshot", "fields": []
    }, ensure_ascii=False), encoding="utf-8")
    builder = object.__new__(ResearchPanelBuilder)
    builder.data_dir = tmp_path
    panel = pl.DataFrame({"symbol": ["A"], "date": [date(2025, 1, 2)]})
    spec = ResearchPanelSpec(
        start=date(2025, 1, 2), end=date(2025, 1, 3), ext_datasets=["latest_only"]
    )
    with pytest.raises(ValueError, match="覆盖式快照"):
        builder._join_extensions(panel, spec)


class _FakeLGBRegressor:
    def __init__(self, **params):
        self.params = params
        self.best_iteration_ = 3
        self.booster_ = self

    def fit(self, x, y, **kwargs):
        if self.params["device_type"] == "gpu":
            raise RuntimeError("OpenCL device not found")
        return self

    def predict(self, values):
        return np.full(len(values), 0.25)

    def feature_importance(self, importance_type):
        return np.array([1.0])


def test_lightgbm_gpu_failure_records_cpu_fallback(monkeypatch):
    module = SimpleNamespace(
        __version__="test", LGBMRegressor=_FakeLGBRegressor,
        early_stopping=lambda *args, **kwargs: object(),
    )
    adapter = LightGBMAdapter()
    monkeypatch.setattr(adapter, "_module", lambda: module)
    result = adapter.fit(
        np.ones((5, 1)), np.ones(5), np.ones((2, 1)), np.ones(2), np.ones(5), {}, "auto", 42
    )
    assert result.actual_device == "cpu"
    assert "OpenCL" in result.warnings[0]
    assert adapter.predict(result.model, np.ones((2, 1))).tolist() == [0.25, 0.25]


class _FakeXGBRegressor:
    def __init__(self, **params):
        self.params = params
        self.best_iteration = 4

    def fit(self, x, y, **kwargs):
        if self.params["device"] == "cuda":
            raise RuntimeError("CUDA out of memory")
        return self

    def predict(self, values):
        return np.zeros(len(values))


def test_xgboost_cuda_failure_records_cpu_fallback(monkeypatch):
    module = SimpleNamespace(__version__="test", XGBRegressor=_FakeXGBRegressor)
    adapter = XGBoostAdapter()
    monkeypatch.setattr(adapter, "_module", lambda: module)
    result = adapter.fit(
        np.ones((5, 1)), np.ones(5), np.ones((2, 1)), np.ones(2), np.ones(5), {}, "gpu", 7
    )
    assert result.actual_device == "cpu"
    assert "显存" in result.warnings[0]


def test_elastic_net_serialization_and_feature_importance(tmp_path: Path):
    rng = np.random.default_rng(11)
    values = rng.normal(size=(160, 3))
    target = values[:, 0] * 0.4 - values[:, 1] * 0.2
    adapter = ElasticNetAdapter()
    fitted = adapter.fit(
        values[:120], target[:120], values[120:], target[120:],
        np.ones(120), {"alpha": 1e-5, "l1_ratio": 0.5}, "gpu", 42,
    )
    path = tmp_path / "model.joblib"
    adapter.save(fitted.model, path)
    restored = adapter.load(path)
    assert fitted.actual_device == "cpu"
    assert "CPU" in fitted.warnings[0]
    assert np.allclose(
        adapter.predict(fitted.model, values[120:]),
        adapter.predict(restored, values[120:]),
    )
    importance = adapter.feature_importance(restored, ["strong", "inverse", "noise"])
    assert importance["strong"]["gain"] > importance["noise"]["gain"]


def test_automl_quality_clusters_budget_and_score(tmp_path: Path):
    rng = np.random.default_rng(5)
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(80)]
    symbols = [f"S{index:02d}" for index in range(20)]
    rows = []
    for day in days:
        strong = rng.normal(size=len(symbols))
        for index, symbol in enumerate(symbols):
            rows.append({
                "date": day, "symbol": symbol, "target": strong[index] + rng.normal(0, 0.05),
                "strong": strong[index], "duplicate": strong[index],
                "noise": rng.normal(), "near_zero": rng.normal() * 1e-14,
                "spiky": 100.0 if index < 2 else 0.0,
                "sample_weight": 1 / len(symbols),
                "forward_return": strong[index] * 0.01, "benchmark_return": 0.0,
            })
    frame = pl.DataFrame(rows)
    quality = AutoMLSearchEngine._factor_quality(
        frame, ["strong", "duplicate", "noise", "near_zero", "spiky"], set()
    )
    by_name = {item["factor_id"]: item for item in quality}
    assert by_name["strong"]["status"] == "accepted"
    assert by_name["noise"]["status"] == "accepted"
    assert by_name["near_zero"]["reason"] == "近零方差"
    assert "极端值比例" in by_name["spiky"]["reason"]
    assert abs(by_name["strong"]["rank_ic"]) > abs(by_name["noise"]["rank_ic"])
    clusters, representatives = AutoMLSearchEngine._correlation_clusters(
        frame, ["strong", "duplicate", "noise"], quality, set()
    )
    assert any(set(item["factors"]) == {"strong", "duplicate"} for item in clusters)
    assert len({"strong", "duplicate"} & set(representatives)) == 1

    registry = FactorRegistry(tmp_path)
    refs = [
        FactorRef(id=name, version=registry.get(name).version)
        for name in ["momentum_5d", "momentum_10d", "momentum_20d",
                     "rsi_6", "rsi_14", "rsi_24", "annual_vol_20d", "atr_14"]
    ]
    spec = MLSearchSpec(
        id="budget_search", name="预算测试", start=date(2020, 1, 1),
        end=date(2025, 1, 1), factor_pool=refs,
    )
    engine = object.__new__(AutoMLSearchEngine)
    assert sum(engine._trial_counts(spec).values()) == 72
    pool = [FactorRef(id=f"factor_{index}", version="v1") for index in range(12)]
    subset_spec = MLSearchSpec(
        id="subset_search", name="子集搜索", start=date(2020, 1, 1),
        end=date(2025, 1, 1), factor_pool=pool,
        required_factors=[pool[0]], algorithms=["elastic_net"],
        budget="standard", min_features=8, max_features=8,
    )
    trials = engine._trial_specs(subset_spec, [item.id for item in pool])
    assert engine._trial_counts(subset_spec) == {"elastic_net": 12}
    assert all("factor_0" in item["features"] for item in trials)
    assert len({tuple(item["features"]) for item in trials}) > 1
    retained = AutoMLSearchEngine._retain_candidates([
        {
            "trial": index, "algorithm": algorithm,
            "baseline": baseline, "score": score,
        }
        for index, (algorithm, baseline, score) in enumerate([
            ("elastic_net", True, 0.1), ("elastic_net", False, 0.9),
            ("lightgbm", True, 0.2), ("lightgbm", False, 0.8),
            ("xgboost", True, 0.3), ("xgboost", False, 0.7),
        ])
    ], 4)
    assert {item["algorithm"] for item in retained} == {
        "elastic_net", "lightgbm", "xgboost",
    }
    good = AutoMLSearchEngine._composite_score(
        {"rank_ic": 0.03, "icir": 0.5, "ic_positive_rate": 0.55},
        {"annual_excess_vs_index": 0.1, "annual_excess_vs_universe": 0.1,
         "sharpe": 0.8, "turnover": 0.2},
        0.95, [0.02, 0.03, 0.01], 8,
    )
    weak = AutoMLSearchEngine._composite_score(
        {"rank_ic": -0.01, "icir": -0.1, "ic_positive_rate": 0.4},
        {"annual_excess_vs_index": -0.1, "annual_excess_vs_universe": -0.1,
         "sharpe": -0.2, "turnover": 0.9},
        0.9, [-0.02, -0.01], 30,
    )
    assert good > weak


def test_automl_rejects_non_finite_factor_without_numpy_warnings():
    frame = pl.DataFrame({
        "date": [date(2024, 1, 1)] * 4,
        "symbol": ["A", "B", "C", "D"],
        "target": [0.01, 0.02, -0.01, 0.0],
        "invalid": [float("nan"), float("inf"), float("-inf"), None],
    })

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        quality = AutoMLSearchEngine._factor_quality(
            frame, ["invalid"], set()
        )

    assert quality[0]["status"] == "rejected"
    assert quality[0]["coverage"] == 0.0
    assert "覆盖率" in quality[0]["reason"]


def test_oos_metrics_ignore_non_finite_rows_and_constant_sections():
    frame = pl.DataFrame({
        "date": [date(2024, 1, 1)] * 4 + [date(2024, 1, 2)] * 4,
        "symbol": ["A", "B", "C", "D"] * 2,
        "target": [0.01, 0.02, 0.03, float("inf"), 0.1, 0.1, 0.1, 0.1],
        "prediction": [0.1, 0.2, 0.3, 0.4, 1.0, 1.0, 1.0, float("nan")],
    })

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = evaluate_oos(frame)

    assert metrics["coverage"] == pytest.approx(6 / 8)
    assert metrics["rank_ic"] == pytest.approx(1.0)
    assert np.isfinite(metrics["rmse"])
    assert len(metrics["daily_ic"]) == 1


def test_factor_preprocess_converts_non_finite_values_to_null():
    builder = object.__new__(MLDatasetBuilder)
    definition = FactorDefinition(
        id="finite_guard",
        name="有限值保护",
        authoring_type="builtin",
        direction=1,
        preprocess={"winsorize_mad": None, "normalize": "none"},
    )
    frame = pl.DataFrame({
        "symbol": ["A", "B", "C", "D"],
        "date": [date(2024, 1, 1)] * 4,
        "finite_guard": [1.0, float("nan"), float("inf"), float("-inf")],
    })

    result = builder._preprocess(frame, definition.id, definition)

    assert result[definition.id].to_list() == [1.0, None, None, None]


def test_automl_rejects_non_point_in_time_required_factor_and_writes_quality(tmp_path: Path):
    registry = FactorRegistry(tmp_path)
    unsafe = registry.upsert(FactorDefinition(
        id="snapshot_factor", name="快照因子", family="fundamental",
        inputs=["close"], expression={"op": "field", "name": "close"},
        point_in_time=False,
    ))
    reference = FactorRef(id=unsafe.id, version=unsafe.version)
    spec = MLSearchSpec(
        id="unsafe_search", name="时点检查", start=date(2020, 1, 1),
        end=date(2025, 1, 1), factor_pool=[reference], required_factors=[reference],
        min_features=1,
    )
    engine = object.__new__(AutoMLSearchEngine)
    engine.factors = registry
    with pytest.raises(ValueError, match="非时点正确"):
        engine._resolve_factor_pool(spec)

    run_dir = tmp_path / "artifacts"
    run_dir.mkdir()
    engine._write_artifacts(
        run_dir,
        [{"factor_id": "excluded", "status": "rejected", "reason": "用户排除"},
         {"factor_id": "accepted", "status": "accepted", "coverage": 1.0,
          "annual_stability": {"2024": 0.1}}],
        [], [], [], {},
    )
    saved = pl.read_parquet(run_dir / "factor_quality.parquet")
    assert saved.height == 2
    assert "selection_rank" in saved.columns


def test_automl_end_to_end_uses_only_fold_champion_oos(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(17)
    calendar = [date(2023, 1, 1) + timedelta(days=index) for index in range(380)]
    symbols = [f"S{index}" for index in range(5)]
    rows = []
    for day_index, day in enumerate(calendar):
        for symbol_index, symbol in enumerate(symbols):
            feature = symbol_index / 5 + math.sin(day_index / 20) * 0.05
            target = feature * 0.02 + rng.normal(0, 0.0005)
            rows.append({
                "symbol": symbol, "date": day, "momentum_20d": feature,
                "target": target, "forward_return": target + 0.001,
                "benchmark_return": 0.001, "sample_weight": 0.2,
            })
    dataset = MLDataset(
        frame=pl.DataFrame(rows), feature_columns=["momentum_20d"],
        calendar=calendar, fingerprint="synthetic", input_file_fingerprint="input",
        warnings=[],
    )

    class Datasets:
        def __init__(self):
                self.factor_cache = SimpleNamespace(
                    inspect=lambda spec, definitions: {
                    "factor_hits": len(definitions), "factor_misses": 0,
                    "hit_ratio": 1.0, "bytes_present": 0, "max_bytes": 1,
                    "used_bytes": 0, "used_ratio": 0.0, "entries": 1,
                    "active_entries": 0, "oldest_accessed_at": None,
                    },
                    status=lambda: {
                        "max_bytes": 1, "used_bytes": 0, "used_ratio": 0.0,
                        "entries": 1, "active_entries": 0,
                        "oldest_accessed_at": None,
                    },
                    evict=lambda: {"entries_removed": 0, "used_bytes": 0},
                )

        def prepare_search(self, spec, definitions, **kwargs):
            event = kwargs.get("on_factor")
            if event:
                event(1, 1, {"hit": True})
            return MLSearchDataset(
                base_frame=dataset.frame.drop("momentum_20d"),
                definitions={item.id: item for item in definitions},
                model_spec=spec, calendar=dataset.calendar,
                fingerprint=dataset.fingerprint,
                input_file_fingerprint=dataset.input_file_fingerprint,
                warnings=dataset.warnings,
                cache_events=[{"factor_id": "momentum_20d", "hit": True}],
            )

        def attach_search_features(self, search_dataset, factor_ids, frame=None):
            result = frame if frame is not None else search_dataset.base_frame
            values = dataset.frame.select(["symbol", "date", *factor_ids])
            return result.join(values, on=["symbol", "date"], how="left")

    factors = FactorRegistry(tmp_path)
    model_registry = ModelRegistry(tmp_path)
    experiments = ExperimentStore(tmp_path)
    engine = AutoMLSearchEngine(Datasets(), factors, model_registry, experiments)
    factor = factors.get("momentum_20d")
    spec = MLSearchSpec(
        id="synthetic_search", name="合成搜索", start=calendar[0], end=calendar[-1],
        factor_pool=[FactorRef(id=factor.id, version=factor.version)],
        algorithms=["elastic_net"], budget="quick", min_features=1, max_features=1,
        shortlist_limit=8, inner_folds=2, inner_validation_days=20,
        walk_forward=WalkForwardSpec(
            train_days=300, validation_days=20, test_days=20, step_days=20,
        ),
        target=TargetSpec(horizon=1), device="cpu",
    )
    searched_through = []
    original_search = engine._search_window

    def observe_search(search_dataset, frame, *args, **kwargs):
        searched_through.append(frame["date"].max())
        return original_search(search_dataset, frame, *args, **kwargs)

    monkeypatch.setattr(engine, "_search_window", observe_search)
    result = engine.run(
        spec, tmp_path / "run", lambda value, message: None, threading.Event()
    )
    oos = pl.read_parquet(tmp_path / "run" / "oos_predictions.parquet")
    assert result["champion"]["algorithm"] == "elastic_net"
    assert result["champion"]["features"] == ["momentum_20d"]
    assert set(oos["fold"].unique().to_list()) == {0, 1}
    assert searched_through[:2] == [
        date.fromisoformat(item["train_end"]) for item in result["folds"]
    ]
    assert oos["date"].min() == date.fromisoformat(result["folds"][0]["test_start"])
    assert oos["date"].max() == date.fromisoformat(result["folds"][-1]["test_end"])
    assert model_registry.get(result["model_version"])["source_run_id"] == "run"


def test_running_experiment_moves_through_cancelling_to_cancelled(tmp_path: Path):
    started = threading.Event()

    class Trainer:
        def run(self, spec, run_dir, progress, cancelled):
            started.set()
            cancelled.wait(timeout=2)
            raise InterruptedError("训练已取消")

    store = ExperimentStore(tmp_path)
    manager = ExperimentManager(store, Trainer())
    spec = ModelSpec(
        id="cancel_model", name="取消测试", features=["momentum_20d"],
        start=date(2020, 1, 1), end=date(2025, 1, 1),
    )
    manifest = manager.submit_ml(spec)
    assert started.wait(timeout=1)
    assert manager.cancel(manifest.run_id).status == "cancelling"
    for _ in range(100):
        current = store.get(manifest.run_id)
        if current.status == "cancelled":
            break
        time.sleep(0.01)
    assert current.status == "cancelled"
    manager.executor.shutdown(wait=True)


def test_background_engine_panic_is_recorded_as_failed(tmp_path: Path):
    class EnginePanic(BaseException):
        pass

    class Searcher:
        datasets = SimpleNamespace(
            factor_cache=SimpleNamespace(
                evict=lambda: {"entries_removed": 0, "used_bytes": 0}
            )
        )

        @staticmethod
        def run(spec, run_dir, progress, cancelled):
            raise EnginePanic("weighted rolling failed")

    store = ExperimentStore(tmp_path)
    manager = ExperimentManager(store, SimpleNamespace(), searcher=Searcher())
    spec = MLSearchSpec(
        id="panic_search",
        name="底层异常测试",
        start=date(2020, 1, 1),
        end=date(2025, 1, 1),
        factor_pool=[FactorRef(id="momentum_20d", version="v1")],
        min_features=1,
        max_features=1,
    )
    manifest = manager.submit_search(spec)
    for _ in range(100):
        current = store.get(manifest.run_id)
        if current.status == "failed":
            break
        time.sleep(0.01)

    assert current.status == "failed"
    assert "EnginePanic" in (current.error or "")
    assert "weighted rolling failed" in (current.error or "")
    manager.executor.shutdown(wait=True)


def test_cancel_marks_orphaned_active_manifest_cancelled(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    manager = ExperimentManager(store, SimpleNamespace())
    manifest = store.create("ml_training", {})
    manifest.status = "running"
    manifest.message = "旧任务仍显示运行中"
    store.save(manifest)

    cancelled = manager.cancel(manifest.run_id)

    assert cancelled.status == "cancelled"
    assert "后台任务已结束" in cancelled.message
    manager.executor.shutdown(wait=True)


def test_ml_backtest_targets_use_next_open_and_cash_on_prediction_gap():
    days = [date(2025, 1, 2) + timedelta(days=index) for index in range(6)]
    panel = pl.DataFrame({
        "symbol": [symbol for day in days for symbol in ["A", "B"]],
        "date": [day for day in days for _ in range(2)],
    })
    oos = pl.DataFrame({
        "symbol": ["A", "B", "A", "B"],
        "date": [days[0], days[0], days[4], days[4]],
        "prediction": [0.2, 0.1, 0.3, 0.05], "rank": [1.0, 0.5, 1.0, 0.5],
    })
    targets = MLBacktestService._build_targets(
        oos, panel, MLBacktestSpec(model_version="v", top_n=1), rebalance_days=2
    )
    first = targets.filter(pl.col("execution_date") == days[1]).row(0, named=True)
    assert first["symbol"] == "A"
    assert first["signal_date"] == days[0]
    gap = targets.filter(pl.col("execution_date") == days[3]).row(0, named=True)
    assert gap["symbol"] == "__CASH__"


def test_ml_backtest_rejects_prediction_outside_test_fold():
    oos = pl.DataFrame({
        "symbol": ["A"], "date": [date(2025, 1, 5)], "fold": [0],
        "prediction": [0.1], "rank": [1.0], "target": [0.2], "forward_return": [0.2],
    })
    folds = [{"index": 0, "test_start": "2025-01-01", "test_end": "2025-01-04"}]
    with pytest.raises(ValueError, match="测试区间之外"):
        MLBacktestService._validate_oos(oos, folds)


def test_ml_backtest_accepts_automl_search_as_oos_source(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    source_run = store.create("ml_search", {"name": "智能训练"})
    source_run.status = "completed"
    source_run.result = {"model_version": "search-model-version"}
    store.save(source_run)
    models = SimpleNamespace(get=lambda version: {
        "version": version, "source_run_id": source_run.run_id,
    })
    service = MLBacktestService(
        SimpleNamespace(), tmp_path, models, store
    )
    assert service.find_training_run("search-model-version").run_id == source_run.run_id


def test_ml_backtest_simulates_lots_costs_and_target_cash():
    days = [date(2025, 1, 2) + timedelta(days=index) for index in range(4)]
    panel = pl.DataFrame({
        "symbol": ["A"] * 4, "date": days,
        "open": [10.0, 10.0, 11.0, 11.0], "high": [10.5, 10.5, 11.5, 11.5],
        "low": [9.5, 9.5, 10.5, 10.5], "close": [10.0, 10.0, 11.0, 11.0],
        "volume": [1000] * 4, "name": ["测试股"] * 4,
        "signal_limit_up": [False] * 4, "signal_limit_down": [False] * 4,
    })
    targets = pl.DataFrame({
        "signal_date": [days[0], days[2]], "execution_date": [days[1], days[3]],
        "symbol": ["A", "__CASH__"], "weight": [1.0, 0.0], "score": [0.2, 0.0],
    })

    class Repo:
        def get_index_daily(self, symbol, start, end, columns=None):
            return pl.DataFrame({"date": days, "close": [100.0] * 4})

    service = object.__new__(MLBacktestService)
    service.repo = Repo()
    spec = MLBacktestSpec(model_version="v", top_n=1)
    result = service._simulate(
        panel, targets, spec,
        {"spec": {"target": {"benchmark_symbol": "000300.SH"}}},
        lambda value, message: None, threading.Event(),
    )
    buys = [item for item in result["trades"] if item["side"] == "buy"]
    sells = [item for item in result["trades"] if item["side"] == "sell"]
    assert buys[0]["date"] == days[1]
    assert buys[0]["shares"] % 100 == 0
    assert sells[0]["date"] == days[3]
    assert result["metrics"]["total_cost"] > 0
    assert 0 < result["metrics"]["total_return"] < 0.1


def test_model_diagnostic_is_weak_when_net_excess_is_negative():
    metadata = {
        "metrics": {
            "rank_ic": 0.05, "icir": 0.6, "ic_positive_rate": 0.7,
            "coverage": 1.0, "annual_stability": {"2025": 0.05, "2026": 0.02},
        },
        "training": {"warnings": []},
    }
    training = SimpleNamespace(result={
        "folds": [
            {"metrics": {"rank_ic": 0.06}}, {"metrics": {"rank_ic": 0.02}},
        ],
        "metrics": {"daily_ic": [{"date": str(index), "ic": 0.05} for index in range(260)]},
    })
    backtest = SimpleNamespace(result={
        "oos_only": True, "warnings": [],
        "metrics": {
            "sharpe": 0.42, "max_drawdown": -0.22,
            "excess_vs_index": -0.1, "excess_vs_universe": -0.2,
        },
    })
    diagnostic = ModelCenterService._diagnostic(metadata, training, backtest)
    assert diagnostic["grade"] == "weak"
    assert diagnostic["dimensions"]["statistics"]["status"] == "green"
    assert diagnostic["dimensions"]["economics"]["status"] == "red"


def test_model_registry_versions_are_immutable_and_publish_is_explicit(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("model", encoding="utf-8")
    registry = ModelRegistry(tmp_path)
    spec = ModelSpec(
        id="stable_model", name="稳定模型", features=["momentum_20d"],
        start=date(2020, 1, 1), end=date(2025, 1, 1),
    ).model_dump(mode="json")
    first = registry.register(
        spec=spec, source_model=source, schema={"features": ["momentum_20d"]},
        metrics={"rank_ic": 0.02}, data_fingerprint="abc", training={},
    )
    second = registry.register(
        spec=spec, source_model=source, schema={"features": ["momentum_20d"]},
        metrics={"rank_ic": 0.02}, data_fingerprint="abc", training={},
    )
    assert first["version"] != second["version"]
    assert registry.get(first["version"])["status"] == "validated"
    assert registry.publish(first["version"])["status"] == "published"
    assert registry.archive(first["version"])["status"] == "archived"
    with pytest.raises(ValueError, match="归档"):
        registry.publish(first["version"])


def test_model_deletion_cascades_after_archive(tmp_path: Path):
    factors = FactorRegistry(tmp_path)
    models = ModelRegistry(tmp_path)
    experiments = ExperimentStore(tmp_path)
    strategies = QuantStrategyStore(tmp_path, factors)
    source_run = experiments.create("ml_training", {"name": "源训练"})
    source_run.status = "completed"
    experiments.save(source_run)
    source = tmp_path / "source.txt"
    source.write_text("model", encoding="utf-8")
    spec = ModelSpec(
        id="delete_model", name="待删除模型", features=["momentum_20d"],
        start=date(2020, 1, 1), end=date(2025, 1, 1),
    ).model_dump(mode="json")
    model = models.register(
        spec=spec, source_model=source, schema={"features": ["momentum_20d"]},
        metrics={}, data_fingerprint="abc", training={},
        source_run_id=source_run.run_id,
    )
    source_run.result = {"model_version": model["version"]}
    experiments.save(source_run)
    models.publish(model["version"])
    model_factor = next(
        item for item in factors.list()
        if item.authoring_type == "model" and item.version == model["version"]
    )
    strategies.upsert(QuantStrategySpec(
        id="dependent_strategy", name="依赖策略",
        factors=[StrategyFactorSpec(
            factor_id=model_factor.id, factor_version=model_factor.version, weight=1
        )],
    ))
    backtest = experiments.create(
        "ml_backtest", {"model_version": model["version"]}
    )
    backtest.status = "completed"
    experiments.save(backtest)
    prediction = (
        tmp_path / "user_data" / "quant" / "predictions" / model["version"]
        / "date=2025-01-02" / "part.parquet"
    )
    prediction.parent.mkdir(parents=True)
    pl.DataFrame({"value": [1]}).write_parquet(prediction)
    deletion = ModelDeletionService(tmp_path, models, experiments, strategies)

    with pytest.raises(ModelDeletionConflict, match="先归档"):
        deletion.delete(
            model["version"], confirm_version=model["version"], cascade=True
        )
    models.archive(model["version"])
    with pytest.raises(ValueError, match="版本不匹配"):
        deletion.delete(
            model["version"], confirm_version="wrong-version", cascade=True
        )
    with pytest.raises(ValueError, match="cascade=true"):
        deletion.delete(
            model["version"], confirm_version=model["version"], cascade=False
        )
    impact = deletion.impact(model["version"])
    assert {item["run_id"] for item in impact["experiments"]} == {
        source_run.run_id, backtest.run_id
    }
    assert impact["strategies"] == [{"id": "dependent_strategy", "name": "依赖策略"}]
    assert impact["prediction_files"] == 1
    assert impact["prediction_rows"] == 1

    result = deletion.delete(
        model["version"], confirm_version=model["version"], cascade=True
    )

    assert result["experiments_deleted"] == 2
    assert not (models.root / model["version"]).exists()
    assert not prediction.exists()
    assert strategies.list() == []
    with pytest.raises(ValueError, match="实验不存在"):
        experiments.get(source_run.run_id)


def test_model_deletion_blocks_active_related_experiment(tmp_path: Path):
    factors = FactorRegistry(tmp_path)
    models = ModelRegistry(tmp_path)
    experiments = ExperimentStore(tmp_path)
    strategies = QuantStrategyStore(tmp_path, factors)
    source = tmp_path / "source.txt"
    source.write_text("model", encoding="utf-8")
    spec = ModelSpec(
        id="busy_model", name="忙碌模型", features=["momentum_20d"],
        start=date(2020, 1, 1), end=date(2025, 1, 1),
    ).model_dump(mode="json")
    model = models.register(
        spec=spec, source_model=source, schema={"features": ["momentum_20d"]},
        metrics={}, data_fingerprint="abc", training={},
    )
    active = experiments.create(
        "ml_backtest", {"model_version": model["version"]}
    )
    deletion = ModelDeletionService(tmp_path, models, experiments, strategies)

    with pytest.raises(ModelDeletionConflict, match="先取消"):
        deletion.delete(
            model["version"], confirm_version=model["version"], cascade=True
        )


def test_model_deletion_rolls_back_when_staging_fails(
    tmp_path: Path, monkeypatch,
):
    factors = FactorRegistry(tmp_path)
    models = ModelRegistry(tmp_path)
    experiments = ExperimentStore(tmp_path)
    strategies = QuantStrategyStore(tmp_path, factors)
    source = tmp_path / "source.txt"
    source.write_text("model", encoding="utf-8")
    spec = ModelSpec(
        id="rollback_model", name="回滚模型", features=["momentum_20d"],
        start=date(2020, 1, 1), end=date(2025, 1, 1),
    ).model_dump(mode="json")
    model = models.register(
        spec=spec, source_model=source, schema={"features": ["momentum_20d"]},
        metrics={}, data_fingerprint="abc", training={},
    )
    prediction_root = (
        tmp_path / "user_data" / "quant" / "predictions" / model["version"]
    )
    prediction_root.mkdir(parents=True)
    (prediction_root / "marker.txt").write_text("prediction", encoding="utf-8")
    deletion = ModelDeletionService(tmp_path, models, experiments, strategies)
    original_replace = Path.replace

    def fail_prediction_move(path: Path, target: Path):
        if path == prediction_root:
            raise PermissionError("injected staging failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_prediction_move)
    with pytest.raises(PermissionError, match="injected"):
        deletion.delete(
            model["version"], confirm_version=model["version"], cascade=True
        )

    assert models.get(model["version"])["name"] == "回滚模型"
    assert (prediction_root / "marker.txt").exists()


def test_strategy_locks_exact_factor_version(tmp_path: Path):
    factors = FactorRegistry(tmp_path)
    factor = factors.get("momentum_20d")
    store = QuantStrategyStore(tmp_path, factors)
    spec = QuantStrategySpec(
        id="locked_strategy", name="锁定策略",
        factors=[StrategyFactorSpec(factor_id=factor.id, factor_version=factor.version, weight=1)],
    )
    assert store.upsert(spec).factors[0].factor_version == factor.version
    with pytest.raises(ValueError, match="版本不匹配"):
        store.upsert(spec.model_copy(update={
            "factors": [StrategyFactorSpec(factor_id=factor.id, factor_version="old", weight=1)]
        }))


@pytest.mark.parametrize("algorithm", ["lightgbm", "xgboost"])
def test_real_ml_serialization_preserves_cpu_predictions(tmp_path: Path, algorithm: str):
    pytest.importorskip(algorithm)
    rng = np.random.default_rng(7)
    values = rng.normal(size=(120, 3))
    target = values[:, 0] * 0.2 - values[:, 1] * 0.1
    adapter = LightGBMAdapter() if algorithm == "lightgbm" else XGBoostAdapter()
    fitted = adapter.fit(
        values[:90], target[:90], values[90:], target[90:], np.ones(90),
        {"n_estimators": 30, "early_stopping_rounds": 5} if algorithm == "xgboost" else {"n_estimators": 30},
        "cpu", 42,
    )
    path = tmp_path / ("model.txt" if algorithm == "lightgbm" else "model.json")
    adapter.save(fitted.model, path)
    restored = adapter.load(path)
    assert np.allclose(adapter.predict(fitted.model, values[90:]), adapter.predict(restored, values[90:]))


def test_etf_model_defaults_to_cross_section_target_and_liquidity_filter():
    spec = ModelSpec(
        id="etf_model", name="ETF模型", asset_type="etf",
        features=["momentum_20d"], start=date(2020, 1, 1),
        end=date(2025, 1, 1),
    )
    assert spec.target.benchmark_mode == "cross_section_mean"
    assert spec.target.benchmark_symbol is None
    assert spec.universe_filters == {
        "min_history_days": 120,
        "min_median_amount_20d": 10_000_000.0,
    }

    custom = spec.model_copy(update={"symbols": ["510300.SH"]})
    assert custom.symbols == ["510300.SH"]


def test_etf_factor_compatibility_keeps_stock_only_fields_out():
    assert standard_expression_asset_types({"close", "volume", "returns"}) == [
        "stock", "etf",
    ]
    assert standard_expression_asset_types({"close", "turnover_rate"}) == ["stock"]


def test_etf_all_market_filter_uses_only_past_history_and_liquidity():
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(130)]
    panel = pl.DataFrame({
        "symbol": [symbol for day in days for symbol in ["LIQUID", "ILLIQUID"]],
        "date": [day for day in days for _ in range(2)],
        "amount": [
            20_000_000.0 if symbol == "LIQUID" else 1_000_000.0
            for _day in days for symbol in ["LIQUID", "ILLIQUID"]
        ],
    })
    spec = ModelSpec(
        id="etf_filter", name="ETF过滤", asset_type="etf",
        features=["momentum_20d"], start=days[0], end=days[-1],
    )
    filtered = MLDatasetBuilder._apply_universe_filters(panel, spec)
    assert filtered["symbol"].unique().to_list() == ["LIQUID"]
    assert filtered["date"].min() == days[119]
    assert filtered["date"].max() == days[-1]


def test_factor_cache_isolated_by_universe_and_invalidated_by_partition(
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    partition = data_dir / "kline_daily_enriched" / "date=2024-01-03"
    partition.mkdir(parents=True)
    pl.DataFrame({
        "symbol": ["A", "B"],
        "date": [date(2024, 1, 3)] * 2,
        "close": [10.0, 20.0],
    }).write_parquet(partition / "part.parquet")
    registry = FactorRegistry(data_dir)
    cache = FactorValueCache(data_dir, registry, max_bytes=10_000_000)
    definition = registry.upsert(FactorDefinition(
        id="close_return_5d", name="5日收盘收益", inputs=["close"], warmup=5,
        expression={
            "op": "return",
            "value": {"op": "field", "name": "close"},
            "window": 5,
        },
    ))
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(8)]
    panel = pl.DataFrame({
        "symbol": [symbol for day in days for symbol in ["A", "B"]],
        "date": [day for day in days for _ in ["A", "B"]],
        "close": [
            float(index + (10 if symbol == "B" else 1))
            for index, day in enumerate(days)
            for symbol in ["A", "B"]
        ],
    })
    spec = ModelSpec(
        id="cache_model", name="缓存模型", symbols=["A", "B"],
        features=[definition.id], start=days[0], end=days[-1],
    )
    first, first_event = cache.get_or_compute(spec, definition, panel)
    second, second_event = cache.get_or_compute(spec, definition, panel)
    assert first.equals(second, null_equal=True)
    assert first_event["hit"] is False
    assert second_event["hit"] is True
    assert cache.inspect(spec, [definition])["factor_hits"] == 1
    directory = cache._factor_dir(spec, definition)
    with cache._active_path(directory):
        assert cache.clear()["entries_removed"] == 0
        assert directory.exists()

    other_spec = spec.model_copy(update={"symbols": ["A"]})
    assert cache.universe_fingerprint(spec) != cache.universe_fingerprint(other_spec)
    _, isolated_event = cache.get_or_compute(other_spec, definition, panel)
    assert isolated_event["hit"] is False

    pl.DataFrame({
        "symbol": ["A", "B"],
        "date": [date(2024, 1, 3)] * 2,
        "close": [11.0, 21.0],
    }).write_parquet(partition / "part.parquet")
    _, invalidated_event = cache.get_or_compute(spec, definition, panel)
    assert invalidated_event["hit"] is False
    assert invalidated_event["recompute_start"] == "2024-01-03"


def test_factor_cache_metadata_hit_can_skip_loading_values(
    tmp_path: Path, monkeypatch,
):
    data_dir = tmp_path / "data"
    partition = data_dir / "kline_daily_enriched" / "date=2024-01-03"
    partition.mkdir(parents=True)
    pl.DataFrame({
        "symbol": ["A"],
        "date": [date(2024, 1, 3)],
        "close": [10.0],
    }).write_parquet(partition / "part.parquet")
    registry = FactorRegistry(data_dir)
    cache = FactorValueCache(data_dir, registry)
    definition = registry.upsert(FactorDefinition(
        id="close_return_5d",
        name="5日收盘收益",
        inputs=["close"],
        warmup=5,
        expression={
            "op": "return",
            "value": {"op": "field", "name": "close"},
            "window": 5,
        },
    ))
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(30)]
    panel = pl.DataFrame({
        "symbol": ["A"] * len(days),
        "date": days,
        "close": [float(index + 1) for index in range(len(days))],
    })
    spec = ModelSpec(
        id="cache_metadata",
        name="缓存元数据命中",
        features=[definition.id],
        start=days[0],
        end=days[-1],
    )
    cache.get_or_compute(spec, definition, panel)
    monkeypatch.setattr(
        cache,
        "_read_values",
        lambda path: (_ for _ in ()).throw(AssertionError("不应读取因子值文件")),
    )

    values, event = cache.get_or_compute(
        spec, definition, panel, load_values=False
    )

    assert values.is_empty()
    assert event["hit"] is True
    assert event["rows"] == len(days)


def test_portfolio_failure_falls_back_with_warning():
    symbols = [f"S{i}" for i in range(10)]
    scores = np.arange(10, dtype=float)
    returns = np.random.default_rng(42).normal(0, 0.01, size=(60, 10))
    result = PortfolioOptimizer().optimize(
        symbols, scores, returns, "min_tracking_error", max_positions=10, max_weight=0.2
    )
    assert result.success is False
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert max(result.weights.values()) <= 0.2 + 1e-9
    assert "回退" in result.warnings[0]


def test_legacy_factor_api_shape_is_preserved():
    assert {"id", "label", "group", "desc"}.issubset(FACTOR_COLUMNS[0])
    assert {item["id"] for item in FACTOR_COLUMNS} >= {"momentum_20d", "rsi_14", "turnover_rate"}
