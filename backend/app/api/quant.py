"""Quant research, ML experiment, model and prediction APIs."""
from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.quant.adapters import ml_capabilities
from app.quant.automl import AutoMLSearchEngine
from app.quant.dataset import MLDatasetBuilder
from app.quant.experiments import ExperimentManager, ExperimentStore
from app.quant.factors import FactorRegistry
from app.quant.ml_backtest import MLBacktestService
from app.quant.model_center import ModelCenterService
from app.quant.model_deletion import ModelDeletionConflict, ModelDeletionService
from app.quant.model_registry import ModelRegistry
from app.quant.models import (
    FactorDefinition,
    MLBacktestSpec,
    MLSearchSpec,
    ModelSpec,
    QuantStrategySpec,
    ResearchPanelSpec,
)
from app.quant.panel import ResearchPanelBuilder
from app.quant.portfolio import PortfolioOptimizer
from app.quant.predictions import PredictionService
from app.quant.spec_store import ModelSpecStore
from app.quant.strategy_store import QuantStrategyStore
from app.quant.trainer import MLTrainer

router = APIRouter(prefix="/api/quant", tags=["quant"])


class CodeFactorPayload(BaseModel):
    source: str


class TrustPayload(BaseModel):
    trusted: bool


class FactorImportPayload(BaseModel):
    root: str | None = None
    dry_run: bool = True


class FactorStatePayload(BaseModel):
    enabled: bool | None = None
    tags: list[str] | None = None


class TrainPayload(BaseModel):
    spec_id: str | None = None
    spec: ModelSpec | None = None


class ExperimentCreatePayload(BaseModel):
    kind: str
    spec: dict[str, Any]


class PortfolioPayload(BaseModel):
    model_version: str
    objective: str = "score_weight"
    max_positions: int = 10
    max_weight: float = 0.2
    industry_cap: float = 0.3
    turnover_cap: float = 0.5
    benchmark_symbol: str | None = None


class ModelDeletePayload(BaseModel):
    confirm_version: str
    cascade: bool = False


def _services(request: Request) -> dict[str, Any]:
    cached = getattr(request.app.state, "quant_services", None)
    if cached is not None:
        return cached
    factors = FactorRegistry(settings.data_dir)
    models = ModelRegistry(settings.data_dir)
    datasets = MLDatasetBuilder(
        request.app.state.repo,
        settings.data_dir,
        factors,
        max_rows=settings.quant_max_panel_rows,
        factor_cache_max_bytes=settings.quant_factor_cache_max_bytes,
    )
    experiments = ExperimentStore(settings.data_dir)
    searcher = AutoMLSearchEngine(datasets, factors, models, experiments)
    predictions = PredictionService(settings.data_dir, models, datasets)
    backtests = MLBacktestService(
        request.app.state.repo, settings.data_dir, models, experiments
    )
    center = ModelCenterService(models, experiments, backtests, predictions)
    strategies = QuantStrategyStore(settings.data_dir, factors)
    cached = {
        "factors": factors,
        "models": models,
        "datasets": datasets,
        "panels": ResearchPanelBuilder(
            request.app.state.repo,
            settings.data_dir,
            max_rows=settings.quant_max_panel_rows,
        ),
        "specs": ModelSpecStore(settings.data_dir),
        "experiments": experiments,
        "manager": ExperimentManager(
            experiments, MLTrainer(datasets, models), backtests, searcher
        ),
        "searcher": searcher,
        "predictions": predictions,
        "backtests": backtests,
        "center": center,
        "strategies": strategies,
        "model_deletion": ModelDeletionService(
            settings.data_dir, models, experiments, strategies
        ),
    }
    request.app.state.quant_services = cached
    return cached


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _experiment_response(manifest, services: dict[str, Any]) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json")
    stored = manifest.result.get("input_file_fingerprint")
    asset_type = manifest.spec.get("asset_type")
    if stored and asset_type:
        payload["input_changed"] = stored != services["datasets"].input_file_fingerprint(asset_type)
    else:
        payload["input_changed"] = False
    return payload


@router.get("/factors")
def list_factors(request: Request):
    return {"factors": [item.model_dump(mode="json") for item in _services(request)["factors"].list()]}


@router.post("/factors")
def save_factor(definition: FactorDefinition, request: Request):
    try:
        return _services(request)["factors"].upsert(definition)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.delete("/factors/{factor_id}")
def delete_factor(factor_id: str, request: Request):
    return {"deleted": _services(request)["factors"].delete(factor_id)}


@router.post("/factors/code")
def save_code_factor(payload: CodeFactorPayload, request: Request):
    try:
        return _services(request)["factors"].save_code(payload.source)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.post("/factors/import/standard-expression")
def import_standard_expression_factors(payload: FactorImportPayload, request: Request):
    try:
        return _services(request)["factors"].import_standard_expression(
            payload.root, dry_run=payload.dry_run
        )
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.patch("/factors/{factor_id}/state")
def update_factor_state(factor_id: str, payload: FactorStatePayload, request: Request):
    try:
        return _services(request)["factors"].update_state(
            factor_id, enabled=payload.enabled, tags=payload.tags
        )
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.post("/factors/{factor_id}/trust")
def trust_code_factor(factor_id: str, payload: TrustPayload, request: Request):
    try:
        return _services(request)["factors"].set_code_trust(factor_id, payload.trusted)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.post("/panel/estimate")
def estimate_panel(spec: ResearchPanelSpec, request: Request):
    return _services(request)["panels"].estimate(spec)


@router.get("/strategies")
def list_quant_strategies(request: Request):
    return {"strategies": [
        item.model_dump(mode="json") for item in _services(request)["strategies"].list()
    ]}


@router.post("/strategies")
def save_quant_strategy(spec: QuantStrategySpec, request: Request):
    try:
        return _services(request)["strategies"].upsert(spec)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.delete("/strategies/{strategy_id}")
def delete_quant_strategy(strategy_id: str, request: Request):
    return {"deleted": _services(request)["strategies"].delete(strategy_id)}


@router.get("/ml/capabilities")
def capabilities():
    return ml_capabilities()


@router.get("/ml/factor-cache")
def factor_cache_status(request: Request):
    return _services(request)["datasets"].factor_cache.status()


@router.delete("/ml/factor-cache")
def clear_factor_cache(request: Request):
    try:
        return _services(request)["datasets"].factor_cache.clear()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/ml/specs")
def list_specs(request: Request):
    return {"specs": [item.model_dump(mode="json") for item in _services(request)["specs"].list()]}


@router.post("/ml/specs")
def save_spec(spec: ModelSpec, request: Request):
    return _services(request)["specs"].upsert(spec)


@router.delete("/ml/specs/{spec_id}")
def delete_spec(spec_id: str, request: Request):
    return {"deleted": _services(request)["specs"].delete(spec_id)}


@router.post("/ml/train")
def train(payload: TrainPayload, request: Request):
    try:
        spec = payload.spec or _services(request)["specs"].get(payload.spec_id or "")
        return _services(request)["manager"].submit_ml(spec)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.post("/ml/search/estimate")
def estimate_ml_search(spec: MLSearchSpec, request: Request):
    try:
        return _services(request)["searcher"].estimate(spec)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.post("/ml/searches")
def create_ml_search(spec: MLSearchSpec, request: Request):
    try:
        return _services(request)["manager"].submit_search(spec)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.get("/ml/searches")
def list_ml_searches(request: Request):
    services = _services(request)
    return {"searches": [
        _experiment_response(item, services)
        for item in services["experiments"].list() if item.kind == "ml_search"
    ]}


@router.get("/ml/searches/{run_id}")
def get_ml_search(run_id: str, request: Request):
    try:
        services = _services(request)
        manifest = services["experiments"].get(run_id)
        if manifest.kind != "ml_search":
            raise ValueError("该实验不是智能训练")
        return _experiment_response(manifest, services)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/ml/models")
def list_models(request: Request):
    return {"models": _services(request)["center"].list()}


@router.get("/ml/models/{version}")
def get_model(version: str, request: Request):
    try:
        return _services(request)["center"].detail(version)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/ml/models/{version}/publish")
def publish_model(version: str, request: Request):
    try:
        return _services(request)["models"].publish(version)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.post("/ml/models/{version}/archive")
def archive_model(version: str, request: Request):
    try:
        return _services(request)["models"].archive(version)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.get("/ml/models/{version}/deletion-impact")
def model_deletion_impact(version: str, request: Request):
    try:
        return _services(request)["model_deletion"].impact(version)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/ml/models/{version}")
def delete_model(version: str, payload: ModelDeletePayload, request: Request):
    try:
        return _services(request)["model_deletion"].delete(
            version,
            confirm_version=payload.confirm_version,
            cascade=payload.cascade,
        )
    except ModelDeletionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.post("/ml/models/{version}/predictions")
def generate_predictions(version: str, request: Request, as_of: date | None = None):
    try:
        return _services(request)["predictions"].generate(version, as_of)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.get("/ml/models/{version}/predictions")
def query_model_predictions(
    version: str,
    request: Request,
    target_date: date | None = None,
    search: str = "",
    limit: int = 200,
    offset: int = 0,
):
    try:
        bounded_limit = min(max(limit, 1), 10_000)
        return _services(request)["predictions"].query(
            version, target_date, search, bounded_limit, max(offset, 0)
        )
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.get("/ml/models/{version}/prediction-dates")
def model_prediction_dates(version: str, request: Request):
    try:
        return {"dates": _services(request)["predictions"].dates(version)}
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.post("/ml/models/{version}/backtests")
def create_model_backtest(version: str, spec: MLBacktestSpec, request: Request):
    if spec.model_version != version:
        raise HTTPException(status_code=400, detail="路径模型版本与回测配置不一致")
    try:
        return _services(request)["manager"].submit_backtest(spec)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.get("/ml/models/{version}/backtests")
def list_model_backtests(version: str, request: Request):
    try:
        return {"backtests": [
            item.model_dump(mode="json")
            for item in _services(request)["backtests"].list_runs(version)
        ]}
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.get("/ml/predictions")
def list_predictions(request: Request, version: str | None = None, target_date: date | None = None):
    frame = _services(request)["predictions"].list(version, target_date)
    return {"predictions": frame.to_dicts()}


@router.post("/portfolio/optimize")
def optimize_portfolio(payload: PortfolioPayload, request: Request):
    try:
        services = _services(request)
        metadata = services["models"].get(payload.model_version)
        predictions = services["predictions"].list(payload.model_version)
        if predictions.is_empty():
            raise ValueError("该模型尚无发布后预测")
        latest = predictions["date"].max()
        predictions = predictions.filter(predictions["date"] == latest).sort("rank", descending=True)
        if predictions.height < max(2, int(np.ceil(1 / payload.max_weight))):
            raise ValueError("有效预测标的不足以满足单标的权重上限")
        symbols = predictions["symbol"].to_list()
        scores = predictions["prediction"].to_numpy()
        end = latest
        start = end - timedelta(days=120)
        series: dict[str, dict[date, float]] = {}
        for symbol in symbols:
            history = request.app.state.repo.get_daily_asset(
                metadata["spec"]["asset_type"], symbol, start, end, columns=["date", "close"]
            ).sort("date")
            if history.height > 1:
                history = history.with_columns(history["close"].pct_change().alias("return")).drop_nulls("return")
                series[symbol] = dict(zip(history["date"].to_list(), history["return"].to_list(), strict=True))
        common_dates = sorted(set.intersection(*(set(values) for values in series.values()))) if series else []
        if len(common_dates) < 20:
            raise ValueError("候选标的共同收益历史不足 20 日")
        active_symbols = [symbol for symbol in symbols if symbol in series]
        active_indices = [symbols.index(symbol) for symbol in active_symbols]
        matrix = np.array([[series[symbol][day] for symbol in active_symbols] for day in common_dates])
        benchmark = None
        if payload.objective == "min_tracking_error":
            benchmark_symbol = payload.benchmark_symbol or metadata["spec"]["target"].get("benchmark_symbol")
            if not benchmark_symbol:
                raise ValueError("最小跟踪误差必须指定基准")
            history = request.app.state.repo.get_index_daily(
                benchmark_symbol, start, end, columns=["date", "close"]
            ).sort("date").with_columns(pl.col("close").pct_change().alias("return"))
            benchmark_map = dict(zip(history["date"].to_list(), history["return"].to_list(), strict=True))
            if any(day not in benchmark_map or benchmark_map[day] is None for day in common_dates):
                raise ValueError("基准收益历史与候选区间不完整")
            benchmark = np.array([benchmark_map[day] for day in common_dates])
        result = PortfolioOptimizer().optimize(
            active_symbols, scores[active_indices], matrix, payload.objective,
            benchmark_returns=benchmark, max_positions=payload.max_positions,
            max_weight=payload.max_weight, industry_cap=payload.industry_cap,
            turnover_cap=payload.turnover_cap, asset_type=metadata["spec"]["asset_type"],
        )
        return {"date": str(latest), "model_version": payload.model_version, **result.__dict__}
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.post("/experiments")
def create_experiment(payload: ExperimentCreatePayload, request: Request):
    try:
        manager = _services(request)["manager"]
        if payload.kind == "ml_training":
            return manager.submit_ml(ModelSpec.model_validate(payload.spec))
        if payload.kind == "ml_backtest":
            return manager.submit_backtest(MLBacktestSpec.model_validate(payload.spec))
        if payload.kind == "ml_search":
            return manager.submit_search(MLSearchSpec.model_validate(payload.spec))
        raise ValueError("当前统一入口支持 ml_training、ml_search 和 ml_backtest")
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.get("/experiments")
def list_experiments(request: Request):
    services = _services(request)
    return {"experiments": [
        _experiment_response(item, services) for item in services["experiments"].list()
    ]}


@router.get("/experiments/compare")
def compare_experiments(request: Request, run_ids: str):
    ids = [item for item in run_ids.split(",") if item][:4]
    try:
        services = _services(request)
        return {"experiments": [
            _experiment_response(services["experiments"].get(item), services) for item in ids
        ]}
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.get("/experiments/{run_id}")
def get_experiment(run_id: str, request: Request):
    try:
        services = _services(request)
        return _experiment_response(services["experiments"].get(run_id), services)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/experiments/{run_id}/events")
async def experiment_events(run_id: str, request: Request):
    async def stream():
        last = None
        while True:
            if await request.is_disconnected():
                break
            try:
                manifest = _services(request)["experiments"].get(run_id)
            except Exception as exc:
                payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
                yield f"event: error\ndata: {payload}\n\n"
                break
            serialized = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, default=str)
            if serialized != last:
                yield f"event: progress\ndata: {serialized}\n\n"
                last = serialized
            if manifest.status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.75)
    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/experiments/{run_id}/cancel")
def cancel_experiment(run_id: str, request: Request):
    try:
        return _services(request)["manager"].cancel(run_id)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.post("/experiments/{run_id}/rerun")
def rerun_experiment(run_id: str, request: Request):
    try:
        return _services(request)["manager"].rerun(run_id)
    except Exception as exc:
        raise _bad_request(exc) from exc


@router.delete("/experiments/{run_id}")
def delete_experiment(run_id: str, request: Request):
    try:
        _services(request)["experiments"].delete(run_id)
        return {"deleted": True}
    except Exception as exc:
        raise _bad_request(exc) from exc
