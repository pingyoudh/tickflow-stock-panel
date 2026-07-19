from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.finance_news import router
from app.services import ai_provider, market_recap
from app.services.finance_news import (
    BEIJING_TZ,
    ClsNewsClient,
    FinanceNewsService,
    FinanceNewsStore,
    FinanceNewsSyncInProgressError,
    generate_cls_sign,
    load_recap_news,
    normalize_cls_item,
)


def _item(
    news_id: str,
    published: datetime,
    *,
    modified: datetime | None = None,
    title: str = "",
    content: str = "快讯正文",
) -> dict:
    return {
        "news_id": news_id,
        "source": "cls",
        "title": title,
        "content": content,
        "published_at": published.isoformat(timespec="seconds"),
        "modified_at": (modified or published).isoformat(timespec="seconds"),
        "level": "B",
        "recommend": False,
        "subjects": [],
        "stocks": [],
    }


def _raw(news_id: str, published_ts: int, *, content: str = "快讯正文") -> dict:
    return {
        "id": news_id,
        "ctime": published_ts,
        "modified_time": published_ts,
        "sort_score": published_ts,
        "title": "",
        "content": content,
        "level": "C",
        "recommend": 0,
        "subjects": [],
        "stock_list": [],
    }


def test_cls_sign_uses_case_sensitive_sorted_keys() -> None:
    params = {
        "app": "CailianpressWeb",
        "lastTime": 1784339835,
        "last_time": 1784339835,
        "os": "web",
        "refresh_type": "1",
        "rn": 20,
        "sv": "8.4.6",
    }
    assert generate_cls_sign(params) == "fbf830c45704ff52e3d47046952cff41"


def test_normalize_cls_item_handles_subjects_and_a_share_codes() -> None:
    ts = int(datetime(2026, 7, 18, 10, 0, tzinfo=BEIJING_TZ).timestamp())
    item = normalize_cls_item({
        "id": 123,
        "ctime": ts,
        "modified_time": ts + 30,
        "title": None,
        "content": "测试正文",
        "level": "A",
        "recommend": 1,
        "subjects": [{"subject_id": 7, "subject_name": "机器人"}],
        "stock_list": [
            {"StockID": "sh600000", "name": "浦发银行"},
            {"StockID": "sz000001", "name": "平安银行"},
            {"StockID": "bj430047", "name": "诺思兰德"},
            {"StockID": "hk00700", "name": "腾讯"},
        ],
    })

    assert item["news_id"] == "123"
    assert item["url"] == (
        "https://api3.cls.cn/share/article/123"
        "?os=web&sv=8.4.6&app=CailianpressWeb"
    )
    assert item["title"] == ""
    assert item["published_at"] == "2026-07-18T10:00:00+08:00"
    assert item["modified_at"] == "2026-07-18T10:00:30+08:00"
    assert item["subjects"] == [{"subject_id": 7, "subject_name": "机器人"}]
    assert [stock["stock_code"] for stock in item["stocks"]] == [
        "600000.SH",
        "000001.SZ",
        "430047.BJ",
    ]


def test_cls_client_builds_valid_request_and_cursor() -> None:
    async def run() -> tuple[list[dict], int | None, httpx.Request | None]:
        seen_request: httpx.Request | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_request
            seen_request = request
            return httpx.Response(
                200,
                json={"errno": 0, "data": {"roll_data": [_raw("1", 100), _raw("2", 90)]}},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            items, cursor = await ClsNewsClient(retries=1).fetch_page(
                100,
                http_client=http_client,
            )
        return items, cursor, seen_request

    items, cursor, seen_request = asyncio.run(run())
    assert len(items) == 2
    assert cursor == 90
    assert seen_request is not None
    query = dict(seen_request.url.params)
    expected_sign = query.pop("sign")
    assert expected_sign == generate_cls_sign(query)


def test_cls_client_retries_transient_and_rejects_invalid_responses(monkeypatch) -> None:
    async def run() -> None:
        calls = 0

        def transient_handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(503, request=request)
            return httpx.Response(
                200,
                json={"errno": 0, "data": {"roll_data": [_raw("1", 100)]}},
                request=request,
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transient_handler)
        ) as http_client:
            items, _ = await ClsNewsClient(retries=2).fetch_page(
                100,
                http_client=http_client,
            )
        assert len(items) == 1
        assert calls == 2

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"errno": 500, "data": {}},
                    request=request,
                )
            )
        ) as http_client:
            with pytest.raises(ValueError, match="接口返回错误"):
                await ClsNewsClient(retries=1).fetch_page(
                    100,
                    http_client=http_client,
                )

    monkeypatch.setattr("app.services.finance_news.asyncio.sleep", AsyncMock())
    asyncio.run(run())


def test_store_upsert_update_and_stable_cursor_pagination(tmp_path) -> None:
    store = FinanceNewsStore(tmp_path)
    published = datetime(2026, 7, 18, 10, 0, tzinfo=BEIJING_TZ)
    items = [_item(str(news_id), published, content=f"正文 {news_id}") for news_id in (1, 2, 3)]

    assert store.upsert(items) == (3, 0)
    partition = store._partition_path("2026-07-18")
    first_mtime = partition.stat().st_mtime_ns
    assert store.upsert(items) == (0, 0)
    assert partition.stat().st_mtime_ns == first_mtime

    updated = _item(
        "2",
        published,
        modified=published + timedelta(minutes=1),
        content="更新后的正文",
    )
    assert store.upsert([updated]) == (0, 1)

    first = store.list_page(2)
    assert [item["news_id"] for item in first["items"]] == ["3", "2"]
    assert first["items"][0]["url"].startswith("https://api3.cls.cn/share/article/3?")
    assert first["has_more"] is True
    second = store.list_page(2, first["next_cursor"])
    assert [item["news_id"] for item in second["items"]] == ["1"]
    assert second["has_more"] is False
    assert second["next_cursor"] is None
    assert partition.exists()


def test_store_paginates_across_date_partitions(tmp_path) -> None:
    store = FinanceNewsStore(tmp_path)
    start = datetime(2026, 7, 18, 10, 0, tzinfo=BEIJING_TZ)
    store.upsert([
        _item(str(index), start - timedelta(days=index), title=f"消息 {index}")
        for index in range(5)
    ])

    news_ids = []
    cursor = None
    while True:
        page = store.list_page(2, cursor)
        news_ids.extend(item["news_id"] for item in page["items"])
        if not page["has_more"]:
            break
        cursor = page["next_cursor"]

    assert news_ids == ["0", "1", "2", "3", "4"]
    assert len(store._parquet_files()) == 5


def test_recap_news_uses_24h_window_without_future_items(tmp_path) -> None:
    store = FinanceNewsStore(tmp_path)
    target = datetime(2026, 7, 18, 15, 30, tzinfo=BEIJING_TZ)
    store.upsert([
        _item("included", target - timedelta(hours=1), content="机器人板块出现异动。后续内容"),
        _item("previous", target - timedelta(hours=23), title="前夜消息"),
        _item("future", target + timedelta(minutes=1), title="未来消息"),
        _item("old", target - timedelta(hours=25), title="过期消息"),
    ])

    result = load_recap_news(
        tmp_path,
        target.date(),
        now=datetime(2026, 7, 20, 9, 0, tzinfo=BEIJING_TZ),
    )

    assert [item["title"] for item in result] == ["机器人板块出现异动", "前夜消息"]
    assert all(item["source"] == "财联社" for item in result)


def test_current_recap_news_is_limited_and_truncated(tmp_path) -> None:
    now = datetime(2026, 7, 18, 10, 0, tzinfo=BEIJING_TZ)
    FinanceNewsStore(tmp_path).upsert([
        _item(
            str(index),
            now - timedelta(minutes=index),
            title=f"消息 {index}",
            content="正" * 400,
        )
        for index in range(10)
    ])

    result = load_recap_news(tmp_path, now.date(), now=now)

    assert len(result) == 8
    assert result[0]["title"] == "消息 0"
    assert len(result[0]["snippet"]) == 281
    assert result[0]["snippet"].endswith("…")


class _PagedClient:
    timeout_seconds = 1

    def __init__(self, pages: dict[int | None, tuple[list[dict], int | None]]) -> None:
        self.pages = pages
        self.calls: list[int | None] = []

    async def fetch_page(self, cursor=None, *, http_client=None):
        self.calls.append(cursor)
        return self.pages.get(cursor, ([], None))


def test_sync_resumes_backfill_and_remains_idempotent(tmp_path) -> None:
    async def run() -> None:
        now_ts = int(datetime.now(BEIJING_TZ).timestamp())
        client = _PagedClient({
            None: ([_raw("latest", now_ts)], now_ts - 10),
            now_ts - 10: ([_raw("middle", now_ts - 3600)], now_ts - 3700),
            now_ts - 3700: ([_raw("old", now_ts - 8 * 86400)], now_ts - 8 * 86400 - 10),
        })
        service = FinanceNewsService(
            tmp_path,
            client=client,
            backfill_page_limit=1,
            page_delay_seconds=0,
        )

        first = await service.sync()
        state = service.store.load_state()
        assert first["inserted"] == 2
        assert state["backfill_completed"] is False
        assert state["backfill_cursor"] == now_ts - 3700
        assert state["backfill_pages"] == 1
        assert state["backfill_oldest_published_at"] is not None

        second = await service.sync()
        state = service.store.load_state()
        assert second["inserted"] == 0
        assert state["backfill_completed"] is True
        assert [item["news_id"] for item in service.list_page(10)["items"]] == [
            "latest",
            "middle",
        ]

    asyncio.run(run())


def test_empty_store_resets_stale_completed_backfill_state(tmp_path) -> None:
    async def run() -> None:
        now_ts = int(datetime.now(BEIJING_TZ).timestamp())
        client = _PagedClient({
            None: ([_raw("latest", now_ts)], now_ts - 10),
            now_ts - 10: ([_raw("older", now_ts - 3600)], None),
        })
        service = FinanceNewsService(tmp_path, client=client, page_delay_seconds=0)
        stale_state = service.store.load_state()
        stale_state.update({
            "backfill_completed": True,
            "backfill_cutoff_ts": 1,
            "backfill_pages": 99,
        })
        service.store.save_state(stale_state)

        result = await service.sync()
        state = service.store.load_state()

        assert result["inserted"] == 2
        assert client.calls == [None, now_ts - 10]
        assert state["backfill_completed"] is True
        assert state["backfill_pages"] == 1

    asyncio.run(run())


def test_sync_failure_keeps_news_and_records_error(tmp_path) -> None:
    async def run() -> None:
        now = datetime.now(BEIJING_TZ)
        service = FinanceNewsService(tmp_path, page_delay_seconds=0)
        service.store.upsert([_item("saved", now, title="已保存快讯")])

        class FailingClient:
            timeout_seconds = 1

            async def fetch_page(self, cursor=None, *, http_client=None):
                raise httpx.ReadTimeout("request timed out")

        service.client = FailingClient()
        with pytest.raises(httpx.ReadTimeout):
            await service.sync()

        assert service.store.list_page(10)["items"][0]["news_id"] == "saved"
        assert "request timed out" in service.store.load_state()["last_error"]
        assert service.status()["syncing"] is False

    asyncio.run(run())


def test_sync_rejects_concurrent_run(tmp_path) -> None:
    async def run() -> None:
        release = asyncio.Event()

        class BlockingClient:
            timeout_seconds = 1

            async def fetch_page(self, cursor=None, *, http_client=None):
                await release.wait()
                return [_raw("1", int(datetime.now(BEIJING_TZ).timestamp()))], None

        service = FinanceNewsService(tmp_path, client=BlockingClient(), page_delay_seconds=0)
        running = asyncio.create_task(service.sync())
        await asyncio.sleep(0)

        with pytest.raises(FinanceNewsSyncInProgressError):
            await service.sync()

        release.set()
        await running

    asyncio.run(run())


def test_finance_news_api_paginates_and_reports_sync_conflict() -> None:
    class FakeService:
        def list_page(self, limit, cursor):
            if cursor == "bad":
                raise ValueError("无效的新闻分页游标")
            return {"items": [], "next_cursor": None, "has_more": False, "sync_status": {}}

        async def sync(self):
            raise FinanceNewsSyncInProgressError("财联社新闻正在同步中")

    app = FastAPI()
    app.state.finance_news_service = FakeService()
    app.include_router(router)
    client = TestClient(app)

    assert client.get("/api/finance-news?limit=10").status_code == 200
    assert client.get("/api/finance-news?cursor=bad").status_code == 400
    assert client.get("/api/finance-news?limit=0").status_code == 422
    assert client.post("/api/finance-news/refresh").status_code == 409


def test_finance_news_refresh_api_returns_sync_result() -> None:
    expected = {
        "fetched": 20,
        "inserted": 2,
        "updated": 1,
        "latest_published_at": "2026-07-18T10:00:00+08:00",
        "synced_at": "2026-07-18T10:01:00+08:00",
    }

    class FakeService:
        async def sync(self):
            return expected

    app = FastAPI()
    app.state.finance_news_service = FakeService()
    app.include_router(router)

    response = TestClient(app).post("/api/finance-news/refresh")

    assert response.status_code == 200
    assert response.json() == expected


def test_market_recap_automatically_injects_local_news(tmp_path, monkeypatch) -> None:
    async def run() -> None:
        now = datetime.now(BEIJING_TZ)
        FinanceNewsStore(tmp_path).upsert([
            _item("recap", now - timedelta(minutes=5), content="真实财联社快讯内容"),
        ])
        monkeypatch.setattr("app.config.settings.data_dir", tmp_path)
        monkeypatch.setattr(
            market_recap,
            "build_market_overview",
            lambda *args, **kwargs: {
                "as_of": now.date().isoformat(),
                "indices": [],
                "emotion": {},
                "limit": {},
                "amount": {},
            },
        )
        captured: list[str] = []

        async def fake_stream(messages, **kwargs):
            captured.append(messages[1]["content"])
            yield "复盘正文"

        monkeypatch.setattr(ai_provider, "stream_ai_text", fake_stream)
        events = [
            event
            async for event in market_recap.recap_market_stream(SimpleNamespace())
        ]

        assert any("复盘正文" in event for event in events)
        assert "真实财联社快讯内容" in captured[0]

    asyncio.run(run())
