from __future__ import annotations

import asyncio
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import HTTPException

from app.api import data as data_api
from app.services.data_catalog import (
    clear_registered_business_data,
    known_directory_roots,
    scan_catalog,
)
from app.services.depth_service import DepthService
from app.services.finance_news import FinanceNewsService


def _write_parquet(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def _find(dimensions: list[dict], dimension_id: str) -> dict:
    for dimension in dimensions:
        if dimension["id"] == dimension_id:
            return dimension
        try:
            return _find(dimension["children"], dimension_id)
        except LookupError:
            pass
    raise LookupError(dimension_id)


def _news_item(news_id: str, published_at: str) -> dict:
    return {
        "news_id": news_id,
        "source": "cls",
        "title": "",
        "content": "快讯正文",
        "published_at": published_at,
        "modified_at": published_at,
        "level": "B",
        "recommend": False,
        "subjects": [],
        "stocks": [],
    }


class _Repo:
    def __init__(self, data_dir) -> None:
        self.store = SimpleNamespace(data_dir=data_dir)
        self.clear_cache_calls = 0
        self.rebuild_views_calls = 0
        self.refresh_cache_calls = 0

    def execute_one(self, *_args, **_kwargs):
        raise RuntimeError("view unavailable")

    def execute_all(self, *_args, **_kwargs):
        raise RuntimeError("view unavailable")

    def clear_cache(self) -> None:
        self.clear_cache_calls += 1

    def rebuild_views(self) -> None:
        self.rebuild_views_calls += 1

    def refresh_cache(self) -> None:
        self.refresh_cache_calls += 1


def test_catalog_classifies_known_assets_and_reports_unknown(tmp_path) -> None:
    _write_parquet(
        tmp_path / "finance_news/cls/date=2026-07-17/part.parquet",
        [{"news_id": "1"}, {"news_id": "2"}],
    )
    _write_parquet(
        tmp_path / "finance_news/cls/date=2026-07-18/part.parquet",
        [{"news_id": "3"}],
    )
    _write_parquet(
        tmp_path / "depth5/date=2026-07-18/part.parquet",
        [{"symbol": "600000.SH"}],
    )
    _write_parquet(
        tmp_path / "kline_etf_minute/date=2026-07-18/part.parquet",
        [{"symbol": "510300.SH"}],
    )
    _write_parquet(
        tmp_path / "adj_factor_etf/all.parquet",
        [{"symbol": "510300.SH"}],
    )
    (tmp_path / "user_data/quant/models/v1.bin").parent.mkdir(parents=True)
    (tmp_path / "user_data/quant/models/v1.bin").write_bytes(b"model")
    (tmp_path / "user_data/secrets.json").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "user_data/secrets.json").write_text("{}", encoding="utf-8")
    (tmp_path / "future_dataset").mkdir()
    (tmp_path / "future_dataset/part.bin").write_bytes(b"unknown")
    (tmp_path / "empty_future_dataset").mkdir()

    snapshot = scan_catalog(tmp_path)

    news = _find(snapshot.dimensions, "finance_news")
    assert news["records"] == 3
    assert news["earliest_at"] == "2026-07-17"
    assert news["latest_at"] == "2026-07-18"
    assert _find(snapshot.dimensions, "depth5")["records"] == 1
    assert _find(snapshot.dimensions, "etf_minute")["files"] == 1
    assert _find(snapshot.dimensions, "etf_adj_factor")["files"] == 1
    assert _find(snapshot.dimensions, "quant_models")["category"] == "research"
    configuration = _find(snapshot.dimensions, "configuration")
    assert configuration["sensitive"] is True
    assert "paths" not in configuration
    assert snapshot.unclassified == {
        "groups": 2,
        "files": 1,
        "size_mb": 0.0,
    }

    category_size = sum(
        category["size_mb"] for category in snapshot.category_totals.values()
    )
    assert snapshot.total_size_mb == pytest.approx(
        category_size + snapshot.unclassified["size_mb"],
        abs=0.03,
    )


def test_catalog_roots_cover_all_datastore_directories() -> None:
    roots = set(known_directory_roots())
    expected = {
        "kline_daily",
        "kline_daily_enriched",
        "kline_index_daily",
        "kline_index_enriched",
        "kline_etf_daily",
        "kline_etf_enriched",
        "kline_etf_minute",
        "kline_minute",
        "adj_factor",
        "adj_factor_etf",
        "financials",
        "instruments",
        "instruments_index",
        "instruments_etf",
        "instruments_ext",
        "kline_ext",
        "pools",
        "backtest_results",
        "screener_results",
        "ai_cache",
        "user_data",
        "depth5",
        "finance_news",
    }
    assert expected <= roots


def test_registered_clear_preserves_extension_definition_and_research(tmp_path) -> None:
    _write_parquet(
        tmp_path / "kline_daily/date=2026-07-18/part.parquet",
        [{"symbol": "600000.SH"}],
    )
    _write_parquet(
        tmp_path / "ext_data/custom/date=2026-07-18/part.parquet",
        [{"symbol": "600000.SH"}],
    )
    config = tmp_path / "ext_data/custom/config.json"
    config.write_text('{"id":"custom"}', encoding="utf-8")
    report = tmp_path / "backtest_results/report.json"
    report.parent.mkdir()
    report.write_text("{}", encoding="utf-8")

    result = clear_registered_business_data(tmp_path)

    assert result["deleted_files"] == 2
    assert not any((tmp_path / "kline_daily").rglob("*.parquet"))
    assert not any((tmp_path / "ext_data").rglob("*.parquet"))
    assert config.exists()
    assert report.exists()


def test_status_exposes_news_depth_and_category_totals(tmp_path, monkeypatch) -> None:
    repo = _Repo(tmp_path)
    news = FinanceNewsService(tmp_path)
    news.store.upsert([
        _news_item("1", "2026-07-18T10:00:00+08:00"),
    ])
    state = news.store.load_state()
    state.update({
        "backfill_completed": True,
        "last_success_at": "2026-07-18T10:01:00+08:00",
    })
    news.store.save_state(state)
    _write_parquet(
        tmp_path / "depth5/date=2026-07-18/part.parquet",
        [{"symbol": "600000.SH"}],
    )
    depth = DepthService()
    depth.set_repo(repo)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                repo=repo,
                scheduler=None,
                finance_news_service=news,
                depth_service=depth,
                indicators_ready=True,
            )
        )
    )
    monkeypatch.setattr(data_api, "_last_finished", lambda _label: None)
    data_api.invalidate_data_cache()

    payload = data_api.status(request)

    assert _find(payload["dimensions"], "finance_news")["records"] == 1
    assert _find(payload["dimensions"], "finance_news")["sync"][
        "last_success_at"
    ] == "2026-07-18T10:01:00+08:00"
    assert _find(payload["dimensions"], "depth5")["records"] == 1
    assert payload["storage"]["category_totals"]["business"]["files"] >= 3
    assert payload["unclassified"]["files"] == 0


def test_clear_api_preserves_research_system_and_resets_business(
    tmp_path,
    monkeypatch,
) -> None:
    repo = _Repo(tmp_path)
    news = FinanceNewsService(tmp_path)
    news.store.upsert([
        _news_item("1", "2026-07-18T10:00:00+08:00"),
    ])
    depth = DepthService()
    depth.set_repo(repo)
    _write_parquet(
        tmp_path / "kline_daily/date=2026-07-18/part.parquet",
        [{"symbol": "600000.SH"}],
    )
    _write_parquet(
        tmp_path / "depth5/date=2026-07-18/part.parquet",
        [{"symbol": "600000.SH"}],
    )
    _write_parquet(
        tmp_path / "ext_data/custom/part.parquet",
        [{"symbol": "600000.SH"}],
    )
    (tmp_path / "ext_data/custom/config.json").write_text("{}", encoding="utf-8")
    for relative in (
        "backtest_results/report.json",
        "job_store/jobs.json",
        "user_data/alerts.jsonl",
        "user_data/preferences.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                repo=repo,
                finance_news_service=news,
                depth_service=depth,
            )
        )
    )
    monkeypatch.setattr(
        "app.services.screener.ScreenerService.clear_history_cache",
        lambda: None,
    )
    monkeypatch.setattr("app.api.overview.invalidate_overview_cache", lambda: None)

    result = asyncio.run(data_api.clear_data(request))

    assert result["deleted_files"] == 4
    assert result["rebuild_scheduled"] is True
    assert not any((tmp_path / "finance_news").rglob("*.parquet"))
    assert not any((tmp_path / "depth5").rglob("*.parquet"))
    assert not any((tmp_path / "kline_daily").rglob("*.parquet"))
    assert not any((tmp_path / "ext_data").rglob("*.parquet"))
    assert (tmp_path / "ext_data/custom/config.json").exists()
    assert (tmp_path / "backtest_results/report.json").exists()
    assert (tmp_path / "job_store/jobs.json").exists()
    assert (tmp_path / "user_data/alerts.jsonl").exists()
    assert (tmp_path / "user_data/preferences.json").exists()
    assert repo.rebuild_views_calls == 1


def test_clear_api_returns_409_while_news_syncs(tmp_path) -> None:
    repo = _Repo(tmp_path)
    news = FinanceNewsService(tmp_path)
    depth = DepthService()
    depth.set_repo(repo)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                repo=repo,
                finance_news_service=news,
                depth_service=depth,
            )
        )
    )
    async def run() -> None:
        await news._sync_lock.acquire()
        try:
            with pytest.raises(HTTPException) as exc:
                await data_api.clear_data(request)
            assert exc.value.status_code == 409
        finally:
            news._sync_lock.release()

    asyncio.run(run())


def test_clear_api_returns_409_while_depth_writes(tmp_path) -> None:
    repo = _Repo(tmp_path)
    news = FinanceNewsService(tmp_path)
    depth = DepthService()
    depth.set_repo(repo)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                repo=repo,
                finance_news_service=news,
                depth_service=depth,
            )
        )
    )
    async def run() -> None:
        depth._fetch_lock.acquire()
        try:
            with pytest.raises(HTTPException) as exc:
                await data_api.clear_data(request)
            assert exc.value.status_code == 409
        finally:
            depth._fetch_lock.release()

    asyncio.run(run())
