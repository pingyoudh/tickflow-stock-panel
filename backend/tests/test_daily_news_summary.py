from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import finance_news as finance_news_api
from app.services import daily_news_summary
from app.services.daily_news_summary import (
    DailyNewsSummaryStore,
    analyze_daily_news_stream,
    build_news_market_context,
    get_daily_summary_status,
)
from app.services.finance_news import BEIJING_TZ, FinanceNewsStore


def _item(news_id: str, published: datetime, content: str) -> dict:
    return {
        "news_id": news_id,
        "source": "cls",
        "url": f"https://example.test/{news_id}",
        "title": f"消息 {news_id}",
        "content": content,
        "published_at": published.isoformat(timespec="seconds"),
        "modified_at": published.isoformat(timespec="seconds"),
        "level": "A",
        "recommend": True,
        "subjects": [{"subject_id": 1, "subject_name": "机器人"}],
        "stocks": [{"stock_code": "600000.SH", "stock_name": "浦发银行"}],
    }


async def _collect_events(*args, **kwargs) -> list[dict]:
    return [json.loads(event) async for event in analyze_daily_news_stream(*args, **kwargs)]


def test_daily_summary_stream_caches_by_news_fingerprint(tmp_path, monkeypatch) -> None:
    target = date(2026, 7, 22)
    now = datetime(2026, 7, 22, 15, 0, tzinfo=BEIJING_TZ)
    store = FinanceNewsStore(tmp_path)
    store.upsert([_item("1", now - timedelta(minutes=5), "机器人产业发布新进展")])

    calls: list[list[dict]] = []

    async def fake_stream(messages, **kwargs):
        calls.append(messages)
        yield "# 2026-07-22 新闻总结\n\n## 一句话总览\n机器人消息受到关注 [N001]"

    monkeypatch.setattr(daily_news_summary, "stream_ai_text", fake_stream)
    monkeypatch.setattr(daily_news_summary, "current_ai_provider", lambda: "test")
    monkeypatch.setattr(daily_news_summary, "current_ai_model", lambda: "test-model")

    first = asyncio.run(_collect_events(tmp_path, store, target, now=now))
    second = asyncio.run(_collect_events(tmp_path, store, target, now=now))

    assert first[0]["type"] == "meta"
    assert first[0]["input_count"] == 1
    assert first[-1]["type"] == "done"
    assert first[-1]["cache_hit"] is False
    assert second[0]["cache_hit"] is True
    assert any(event["type"] == "delta" for event in second)
    assert len(calls) == 1
    assert DailyNewsSummaryStore(tmp_path).load(target)["model"] == "test-model"

    store.upsert([_item("2", now - timedelta(minutes=1), "半导体行业公布新的投资计划")])
    status = get_daily_summary_status(tmp_path, store, target, now=now)
    assert status["current_news_count"] == 2
    assert status["stale"] is True


def test_large_daily_news_uses_group_summaries_before_synthesis(tmp_path, monkeypatch) -> None:
    target = date(2026, 7, 22)
    now = datetime(2026, 7, 22, 12, 0, tzinfo=BEIJING_TZ)
    store = FinanceNewsStore(tmp_path)
    store.upsert([
        _item("1", now - timedelta(minutes=2), "第一组新闻"),
        _item("2", now - timedelta(minutes=1), "第二组新闻"),
    ])

    group_calls: list[str] = []
    final_calls: list[str] = []

    async def fake_generate(messages, **kwargs):
        group_calls.append(messages[-1]["content"])
        return f"分组摘要 {len(group_calls)} [N00{len(group_calls)}]"

    async def fake_stream(messages, **kwargs):
        final_calls.append(messages[-1]["content"])
        yield "# 综合总结"

    monkeypatch.setattr(daily_news_summary, "_split_chunks", lambda entries: entries)
    monkeypatch.setattr(daily_news_summary, "generate_ai_text", fake_generate)
    monkeypatch.setattr(daily_news_summary, "stream_ai_text", fake_stream)
    monkeypatch.setattr(daily_news_summary, "current_ai_provider", lambda: "test")
    monkeypatch.setattr(daily_news_summary, "current_ai_model", lambda: "test-model")

    events = asyncio.run(
        _collect_events(tmp_path, store, target, force=True, now=now)
    )

    assert len(group_calls) == 2
    assert "分组摘要 1" in final_calls[0]
    assert "分组摘要 2" in final_calls[0]
    assert [event["type"] for event in events].count("progress") == 3
    assert events[-1]["type"] == "done"


def test_market_context_only_exposes_current_tradeable_news_symbols(monkeypatch) -> None:
    target = date(2026, 7, 22)
    now = datetime(2026, 7, 22, 14, 45, tzinfo=BEIJING_TZ)
    frame = pl.DataFrame({
        "symbol": ["600000.SH", "000001.SZ", "600519.SH"],
        "date": [target, target, target],
        "name": ["浦发银行", "ST测试", "贵州茅台"],
        "close": [12.0, 4.5, 1500.0],
        "volume": [1_000_000.0, 500_000.0, 800_000.0],
        "amount": [500_000_000.0, 20_000_000.0, 900_000_000.0],
        "change_pct": [0.025, 0.049, 0.01],
        "turnover_rate": [2.2, 4.0, 0.7],
        "vol_ratio_5d": [1.6, 2.0, 1.1],
        "ma5": [11.8, 4.3, 1490.0],
        "ma20": [11.5, 4.1, 1480.0],
        "ma60": [11.0, 4.0, 1450.0],
        "signal_limit_up": [False, True, False],
        "signal_limit_down": [False, False, False],
        "signal_broken_limit_up": [False, False, False],
    })

    class FakeRepo:
        def get_instruments(self):
            return frame.select(["symbol", "name"])

    class FakeQuoteService:
        def get_enriched_today(self):
            return frame, target

    monkeypatch.setattr(
        daily_news_summary,
        "build_market_overview",
        lambda *args, **kwargs: {
            "as_of": target.isoformat(),
            "quote_status": {
                "enabled": True,
                "running": True,
                "is_trading_hours": True,
                "market_phase": "afternoon",
                "quote_age_ms": 10_000,
                "last_fetch_ms": now.timestamp() * 1000,
                "interval_s": 60,
            },
            "breadth": {"up": 3000, "down": 2000},
        },
    )
    news = [
        _item("1", now - timedelta(minutes=10), "银行行业出现政策催化"),
        {
            **_item("2", now - timedelta(minutes=5), "风险标的异动"),
            "stocks": [{"stock_code": "000001.SZ", "stock_name": "ST测试"}],
        },
    ]

    context = build_news_market_context(
        FakeRepo(),
        FakeQuoteService(),
        None,
        target,
        news,
        now=now,
    )

    assert context["selection_ready"] is True
    assert context["tail_window"] is True
    assert context["linked_symbol_count"] == 2
    assert context["eligible_count"] == 1
    assert [row["symbol"] for row in context["candidates"]] == ["600000.SH", "000001.SZ"]
    assert context["candidates"][0]["news_refs"] == ["N001"]
    assert context["candidates"][0]["change_pct"] == 2.5
    assert context["candidates"][1]["eligible"] is False
    assert "ST 风险标的" in context["candidates"][1]["exclusion_reasons"]
    assert "贵州茅台" not in json.dumps(context, ensure_ascii=False)


def test_daily_summary_api_exposes_status_and_ndjson_stream(tmp_path, monkeypatch) -> None:
    store = FinanceNewsStore(tmp_path)

    monkeypatch.setattr(
        finance_news_api,
        "get_daily_summary_status",
        lambda data_dir, news_store, as_of, **kwargs: {
            "as_of": as_of.isoformat(),
            "current_news_count": 3,
            "current_unique_count": 2,
            "stale": False,
            "summary": None,
        },
    )

    async def fake_analyze(data_dir, news_store, as_of, *, force=False, **kwargs):
        yield json.dumps({"type": "meta", "as_of": as_of.isoformat()})
        yield json.dumps({"type": "done", "cache_hit": False})

    monkeypatch.setattr(finance_news_api, "analyze_daily_news_stream", fake_analyze)

    app = FastAPI()
    app.state.finance_news_service = SimpleNamespace(store=store)
    app.include_router(finance_news_api.router)
    client = TestClient(app)

    status = client.get("/api/finance-news/daily-summary?as_of=2026-07-22")
    assert status.status_code == 200
    assert status.json()["current_unique_count"] == 2

    response = client.post(
        "/api/finance-news/daily-summary/analyze",
        json={"as_of": "2026-07-22", "force": True},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert [json.loads(line)["type"] for line in response.text.splitlines()] == ["meta", "done"]
