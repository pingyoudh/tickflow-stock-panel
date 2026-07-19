"""财联社快讯同步、存储与复盘选取。"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from datetime import date, datetime, timedelta, timezone
from datetime import time as dt_time
from pathlib import Path
from typing import Any

import httpx
import polars as pl

logger = logging.getLogger(__name__)

BEIJING_TZ = timezone(timedelta(hours=8))
CLS_BASE_URL = "https://www.cls.cn/v1/roll/get_roll_list"
CLS_PARAMS = {
    "app": "CailianpressWeb",
    "os": "web",
    "refresh_type": "1",
    "rn": 20,
    "sv": "8.4.6",
}
CLS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.cls.cn/telegraph",
    "Accept": "application/json, text/plain, */*",
}
CLS_SHARE_URL = (
    "https://api3.cls.cn/share/article/{news_id}"
    "?os=web&sv=8.4.6&app=CailianpressWeb"
)

_STORAGE_SCHEMA = {
    "news_id": pl.Utf8,
    "source": pl.Utf8,
    "title": pl.Utf8,
    "content": pl.Utf8,
    "published_at": pl.Utf8,
    "published_ts": pl.Int64,
    "modified_at": pl.Utf8,
    "modified_ts": pl.Int64,
    "level": pl.Utf8,
    "recommend": pl.Boolean,
    "subjects_json": pl.Utf8,
    "stocks_json": pl.Utf8,
}
_A_SHARE_CODE = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")


class FinanceNewsSyncInProgressError(RuntimeError):
    """已有新闻同步任务正在执行。"""


def generate_cls_sign(params: Mapping[str, Any]) -> str:
    """按财联社网页端规则生成 SHA1 -> MD5 签名。"""
    plain = "&".join(f"{key}={params[key]}" for key in sorted(params))
    sha1_hex = hashlib.sha1(plain.encode("utf-8")).hexdigest()
    return hashlib.md5(sha1_hex.encode("utf-8")).hexdigest()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _iso_from_timestamp(value: int) -> str:
    return datetime.fromtimestamp(value, tz=BEIJING_TZ).isoformat(timespec="seconds")


def _normalize_stock_code(value: Any) -> str:
    raw = str(value or "").strip()
    lowered = raw.lower()
    prefix_map = {"sh": "SH", "sz": "SZ", "bj": "BJ"}
    for prefix, suffix in prefix_map.items():
        if lowered.startswith(prefix) and lowered[2:].isdigit():
            raw = f"{lowered[2:]}.{suffix}"
            break
    raw = raw.upper()
    return raw if _A_SHARE_CODE.fullmatch(raw) else ""


def _cls_news_url(news_id: Any) -> str:
    normalized = str(news_id or "").strip()
    if not normalized.isdigit():
        return "https://www.cls.cn/telegraph"
    return CLS_SHARE_URL.format(news_id=normalized)


def normalize_cls_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """把财联社 roll_data 项转换为稳定的公开新闻结构。"""
    modified_ts = _as_int(item.get("modified_time")) or _as_int(item.get("sort_score"))
    published_ts = _as_int(item.get("ctime")) or modified_ts
    if not published_ts:
        raise ValueError("财联社新闻缺少有效发布时间")
    if not modified_ts:
        modified_ts = published_ts

    subjects = []
    for subject in item.get("subjects") or []:
        if not isinstance(subject, Mapping):
            continue
        subject_id = _as_int(subject.get("subject_id"))
        subject_name = str(subject.get("subject_name") or "").strip()
        if subject_id or subject_name:
            subjects.append({"subject_id": subject_id, "subject_name": subject_name})

    stocks = []
    for stock in item.get("stock_list") or []:
        if not isinstance(stock, Mapping):
            continue
        raw_stock_code = stock.get("StockID") or stock.get("stock_id") or stock.get("stock_code")
        stock_code = _normalize_stock_code(raw_stock_code)
        stock_name = str(stock.get("name") or stock.get("stock_name") or "").strip()
        if raw_stock_code and not stock_code:
            continue
        if stock_code or stock_name:
            stocks.append({"stock_code": stock_code, "stock_name": stock_name})

    news_id = str(item.get("id") or "").strip()
    return {
        "news_id": news_id,
        "source": "cls",
        "url": _cls_news_url(news_id),
        "title": str(item.get("title") or "").strip(),
        "content": str(item.get("content") or "").strip(),
        "published_at": _iso_from_timestamp(published_ts),
        "modified_at": _iso_from_timestamp(modified_ts),
        "level": str(item.get("level") or "").strip(),
        "recommend": _as_int(item.get("recommend")) == 1,
        "subjects": subjects,
        "stocks": stocks,
    }


def _timestamp_from_iso(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp())


def _item_to_row(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "news_id": str(item.get("news_id") or ""),
        "source": str(item.get("source") or "cls"),
        "title": str(item.get("title") or ""),
        "content": str(item.get("content") or ""),
        "published_at": str(item.get("published_at") or ""),
        "published_ts": _timestamp_from_iso(str(item.get("published_at") or "")),
        "modified_at": str(item.get("modified_at") or ""),
        "modified_ts": _timestamp_from_iso(str(item.get("modified_at") or "")),
        "level": str(item.get("level") or ""),
        "recommend": bool(item.get("recommend")),
        "subjects_json": json.dumps(item.get("subjects") or [], ensure_ascii=False),
        "stocks_json": json.dumps(item.get("stocks") or [], ensure_ascii=False),
    }


def _decode_json_list(value: Any) -> list[dict[str, Any]]:
    try:
        decoded = json.loads(str(value or "[]"))
        return decoded if isinstance(decoded, list) else []
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def _row_to_item(row: Mapping[str, Any]) -> dict[str, Any]:
    news_id = str(row.get("news_id") or "")
    return {
        "news_id": news_id,
        "source": str(row.get("source") or "cls"),
        "url": _cls_news_url(news_id),
        "title": str(row.get("title") or ""),
        "content": str(row.get("content") or ""),
        "published_at": str(row.get("published_at") or ""),
        "modified_at": str(row.get("modified_at") or ""),
        "level": str(row.get("level") or ""),
        "recommend": bool(row.get("recommend")),
        "subjects": _decode_json_list(row.get("subjects_json")),
        "stocks": _decode_json_list(row.get("stocks_json")),
    }


def encode_news_cursor(published_ts: int, news_id: str) -> str:
    payload = json.dumps([published_ts, news_id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_news_cursor(cursor: str) -> tuple[int, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded).decode())
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], int)
            or not isinstance(value[1], str)
        ):
            raise ValueError
        return value[0], value[1]
    except Exception as exc:
        raise ValueError("无效的新闻分页游标") from exc


class ClsNewsClient:
    """财联社网页快讯接口客户端。"""

    def __init__(self, timeout_seconds: float = 15, retries: int = 3) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    async def fetch_page(
        self,
        last_time: int | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        cursor = int(last_time or time.time())
        params: dict[str, Any] = dict(CLS_PARAMS)
        params["lastTime"] = cursor
        params["last_time"] = cursor
        params["sign"] = generate_cls_sign(params)

        owns_client = http_client is None
        client = http_client or httpx.AsyncClient(
            timeout=self.timeout_seconds,
            headers=CLS_HEADERS,
        )
        try:
            for attempt in range(self.retries):
                try:
                    response = await client.get(CLS_BASE_URL, params=params, headers=CLS_HEADERS)
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict) or payload.get("errno") != 0:
                        raise ValueError(f"财联社接口返回错误: {payload.get('errno') if isinstance(payload, dict) else 'invalid'}")
                    roll_data = (payload.get("data") or {}).get("roll_data") or []
                    if not isinstance(roll_data, list):
                        raise ValueError("财联社接口 roll_data 格式异常")
                    if not roll_data:
                        return [], None
                    next_cursor = min(
                        _as_int(item.get("sort_score")) or _as_int(item.get("modified_time"))
                        for item in roll_data
                        if isinstance(item, Mapping)
                    )
                    return [dict(item) for item in roll_data if isinstance(item, Mapping)], next_cursor
                except (httpx.HTTPError, ValueError, json.JSONDecodeError):
                    if attempt + 1 >= self.retries:
                        raise
                    await asyncio.sleep(0.5 * (2**attempt))
        finally:
            if owns_client:
                await client.aclose()
        return [], None


class FinanceNewsStore:
    """财联社快讯的按日 Parquet 存储。"""

    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "finance_news" / "cls"
        self.state_path = self.root / "sync_state.json"
        self._lock = threading.RLock()

    def _partition_path(self, date_text: str) -> Path:
        return self.root / f"date={date_text}" / "part.parquet"

    def _parquet_files(self) -> list[Path]:
        return sorted(self.root.glob("date=*/part.parquet"))

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "backfill_completed": False,
            "backfill_cursor": None,
            "backfill_cutoff_ts": None,
            "backfill_started_at": None,
            "backfill_pages": 0,
            "backfill_oldest_published_at": None,
            "last_attempt_at": None,
            "last_success_at": None,
            "last_error": None,
            "latest_published_at": None,
        }

    def load_state(self) -> dict[str, Any]:
        with self._lock:
            state = self._empty_state()
            if not self.state_path.exists():
                return state
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    state.update(raw)
            except Exception as exc:
                logger.warning("财联社同步状态损坏,使用默认状态: %s", exc)
            return state

    def save_state(self, state: Mapping[str, Any]) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(dict(state), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self.state_path)

    @staticmethod
    def _atomic_write(frame: pl.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        frame.write_parquet(tmp)
        os.replace(tmp, path)

    def upsert(self, items: list[dict[str, Any]]) -> tuple[int, int]:
        if not items:
            return 0, 0

        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            row = _item_to_row(item)
            grouped.setdefault(row["published_at"][:10], []).append(row)

        inserted = 0
        updated = 0
        with self._lock:
            for date_text, new_rows in grouped.items():
                path = self._partition_path(date_text)
                existing_rows = (
                    pl.read_parquet(path).to_dicts()
                    if path.exists()
                    else []
                )
                by_id = {
                    (str(row["source"]), str(row["news_id"])): row
                    for row in existing_rows
                }
                partition_changed = False
                for row in new_rows:
                    key = (row["source"], row["news_id"])
                    previous = by_id.get(key)
                    if previous is None:
                        by_id[key] = row
                        inserted += 1
                        partition_changed = True
                    elif row["modified_ts"] > _as_int(previous.get("modified_ts")):
                        by_id[key] = row
                        updated += 1
                        partition_changed = True

                if not partition_changed:
                    continue

                frame = pl.DataFrame(
                    list(by_id.values()),
                    schema=_STORAGE_SCHEMA,
                ).sort(["published_ts", "news_id"], descending=[True, True])
                self._atomic_write(frame, path)
        return inserted, updated

    def _query(
        self,
        *,
        limit: int,
        cursor: tuple[int, str] | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        files = self._parquet_files()
        if not files:
            return []
        with self._lock:
            lazy = pl.scan_parquet([str(path) for path in files])
            if cursor:
                cursor_ts, cursor_id = cursor
                lazy = lazy.filter(
                    (pl.col("published_ts") < cursor_ts)
                    | (
                        (pl.col("published_ts") == cursor_ts)
                        & (pl.col("news_id") < cursor_id)
                    )
                )
            if start_ts is not None:
                lazy = lazy.filter(pl.col("published_ts") >= start_ts)
            if end_ts is not None:
                lazy = lazy.filter(pl.col("published_ts") <= end_ts)
            rows = (
                lazy.sort(
                    ["published_ts", "news_id"],
                    descending=[True, True],
                )
                .limit(limit)
                .collect()
                .to_dicts()
            )
        return rows

    def list_page(self, limit: int, cursor: str | None = None) -> dict[str, Any]:
        decoded_cursor = decode_news_cursor(cursor) if cursor else None
        rows = self._query(limit=limit + 1, cursor=decoded_cursor)
        has_more = len(rows) > limit
        visible = rows[:limit]
        next_cursor = None
        if has_more and visible:
            last = visible[-1]
            next_cursor = encode_news_cursor(_as_int(last["published_ts"]), str(last["news_id"]))
        return {
            "items": [_row_to_item(row) for row in visible],
            "next_cursor": next_cursor,
            "has_more": has_more,
        }

    def list_between(self, start_ts: int, end_ts: int, limit: int) -> list[dict[str, Any]]:
        rows = self._query(limit=limit, start_ts=start_ts, end_ts=end_ts)
        return [_row_to_item(row) for row in rows]

    def latest_modified_timestamp(self) -> int | None:
        files = self._parquet_files()
        if not files:
            return None
        with self._lock:
            value = (
                pl.scan_parquet([str(path) for path in files])
                .select(pl.col("modified_ts").max())
                .collect()
                .item()
            )
        return int(value) if value is not None else None

    def latest_published_at(self) -> str | None:
        files = self._parquet_files()
        if not files:
            return None
        with self._lock:
            row = (
                pl.scan_parquet([str(path) for path in files])
                .sort(["published_ts", "news_id"], descending=[True, True])
                .select("published_at")
                .limit(1)
                .collect()
            )
        return str(row.item()) if row.height else None

    def clear(self) -> dict[str, int]:
        """Clear news data and sync state while the service sync lock is held."""
        deleted_files = 0
        deleted_bytes = 0
        with self._lock:
            if not self.root.exists():
                return {"deleted_files": 0, "deleted_bytes": 0}
            files = [path for path in self.root.rglob("*") if path.is_file()]
            for path in files:
                try:
                    deleted_bytes += path.stat().st_size
                    path.unlink()
                    deleted_files += 1
                except OSError as exc:
                    logger.warning("清理财联社文件失败 %s: %s", path, exc)
            for directory in sorted(
                (path for path in self.root.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                with suppress(OSError):
                    directory.rmdir()
            with suppress(OSError):
                self.root.rmdir()
        return {
            "deleted_files": deleted_files,
            "deleted_bytes": deleted_bytes,
        }


class FinanceNewsService:
    """协调财联社增量抓取、历史回补和本地查询。"""

    def __init__(
        self,
        data_dir: Path,
        *,
        client: ClsNewsClient | None = None,
        backfill_days: int = 7,
        backfill_page_limit: int = 30,
        page_delay_seconds: float = 1.0,
    ) -> None:
        self.store = FinanceNewsStore(data_dir)
        self.client = client or ClsNewsClient()
        self.backfill_days = backfill_days
        self.backfill_page_limit = backfill_page_limit
        self.page_delay_seconds = page_delay_seconds
        self._sync_lock = asyncio.Lock()

    @asynccontextmanager
    async def exclusive(self):
        """Acquire the sync lock without waiting, for sync-adjacent maintenance."""
        if self._sync_lock.locked():
            raise FinanceNewsSyncInProgressError("财联社新闻正在同步中")
        await self._sync_lock.acquire()
        try:
            yield
        finally:
            self._sync_lock.release()

    def status(self) -> dict[str, Any]:
        state = self.store.load_state()
        if not state.get("latest_published_at"):
            state["latest_published_at"] = self.store.latest_published_at()
        return {
            "syncing": self._sync_lock.locked(),
            "backfill_completed": bool(state.get("backfill_completed")),
            "last_success_at": state.get("last_success_at"),
            "last_error": state.get("last_error"),
            "latest_published_at": state.get("latest_published_at"),
        }

    def list_page(self, limit: int, cursor: str | None = None) -> dict[str, Any]:
        result = self.store.list_page(limit, cursor)
        result["sync_status"] = self.status()
        return result

    async def _fetch_and_normalize(
        self,
        cursor: int | None,
        http_client: httpx.AsyncClient,
    ) -> tuple[list[dict[str, Any]], int | None, int]:
        raw_items, next_cursor = await self.client.fetch_page(cursor, http_client=http_client)
        normalized = []
        for raw in raw_items:
            try:
                item = normalize_cls_item(raw)
                if item["news_id"]:
                    normalized.append(item)
            except (TypeError, ValueError) as exc:
                logger.warning("跳过无效财联社新闻: %s", exc)
        return normalized, next_cursor, len(raw_items)

    async def sync(self) -> dict[str, Any]:
        if self._sync_lock.locked():
            raise FinanceNewsSyncInProgressError("财联社新闻正在同步中")
        await self._sync_lock.acquire()

        fetched = inserted = updated = 0
        state: dict[str, Any] | None = None
        try:
            state = self.store.load_state()
            now = datetime.now(BEIJING_TZ)
            now_iso = now.isoformat(timespec="seconds")
            known_latest = self.store.latest_modified_timestamp()
            if known_latest is None and (
                state.get("backfill_completed")
                or state.get("backfill_cursor") is not None
                or _as_int(state.get("backfill_pages"))
            ):
                state.update({
                    "backfill_completed": False,
                    "backfill_cursor": None,
                    "backfill_cutoff_ts": None,
                    "backfill_started_at": None,
                    "backfill_pages": 0,
                    "backfill_oldest_published_at": None,
                })

            state["last_attempt_at"] = now_iso
            if not state.get("backfill_cutoff_ts"):
                state["backfill_cutoff_ts"] = int(
                    (now - timedelta(days=self.backfill_days)).timestamp()
                )
                state["backfill_started_at"] = now_iso
            self.store.save_state(state)

            timeout = httpx.Timeout(self.client.timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout, headers=CLS_HEADERS) as http_client:
                newest, next_cursor, raw_count = await self._fetch_and_normalize(None, http_client)
                fetched += raw_count
                added, changed = self.store.upsert(newest)
                inserted += added
                updated += changed

                # 一分钟内超过一页的新消息时继续向后翻,直到触及同步前的最新记录。
                incremental_pages = 1
                current = newest
                while (
                    known_latest
                    and next_cursor
                    and current
                    and min(_timestamp_from_iso(item["modified_at"]) for item in current) > known_latest
                    and incremental_pages < 100
                ):
                    current, next_cursor, raw_count = await self._fetch_and_normalize(
                        next_cursor, http_client
                    )
                    fetched += raw_count
                    added, changed = self.store.upsert(current)
                    inserted += added
                    updated += changed
                    incremental_pages += 1

                if not state.get("backfill_completed"):
                    backfill_cursor = state.get("backfill_cursor")
                    if backfill_cursor is None:
                        backfill_cursor = next_cursor
                    cutoff = _as_int(state.get("backfill_cutoff_ts"))

                    for _ in range(self.backfill_page_limit):
                        if not backfill_cursor:
                            state["backfill_completed"] = True
                            state["backfill_cursor"] = None
                            break
                        page, page_cursor, raw_count = await self._fetch_and_normalize(
                            _as_int(backfill_cursor), http_client
                        )
                        fetched += raw_count
                        if not page:
                            state["backfill_completed"] = True
                            state["backfill_cursor"] = None
                            break

                        in_range = [
                            item
                            for item in page
                            if _timestamp_from_iso(item["published_at"]) >= cutoff
                        ]
                        added, changed = self.store.upsert(in_range)
                        inserted += added
                        updated += changed

                        state["backfill_pages"] = _as_int(state.get("backfill_pages")) + 1
                        state["backfill_oldest_published_at"] = min(
                            item["published_at"] for item in page
                        )
                        oldest = min(
                            _timestamp_from_iso(item["published_at"]) for item in page
                        )
                        if oldest <= cutoff or not page_cursor or page_cursor >= _as_int(backfill_cursor):
                            state["backfill_completed"] = True
                            state["backfill_cursor"] = None
                            self.store.save_state(state)
                            break

                        state["backfill_cursor"] = page_cursor
                        backfill_cursor = page_cursor
                        self.store.save_state(state)
                        if self.page_delay_seconds:
                            await asyncio.sleep(self.page_delay_seconds)

            completed_at = datetime.now(BEIJING_TZ).isoformat(timespec="seconds")
            latest_published_at = self.store.latest_published_at()
            state["last_success_at"] = completed_at
            state["last_error"] = None
            state["latest_published_at"] = latest_published_at
            self.store.save_state(state)
            return {
                "fetched": fetched,
                "inserted": inserted,
                "updated": updated,
                "latest_published_at": latest_published_at,
                "synced_at": completed_at,
            }
        except Exception as exc:
            if state is not None:
                state["last_error"] = str(exc)
                try:
                    self.store.save_state(state)
                except Exception:
                    logger.exception("财联社同步错误状态写入失败")
            raise
        finally:
            self._sync_lock.release()

    async def scheduled_sync(self) -> None:
        try:
            await self.sync()
        except FinanceNewsSyncInProgressError:
            logger.debug("财联社定时同步跳过: 已有任务运行")
        except Exception as exc:
            logger.warning("财联社定时同步失败,保留已有数据: %s", exc)


def _fallback_title(content: str, limit: int = 72) -> str:
    compact = " ".join(content.split())
    if not compact:
        return "财联社快讯"
    first = re.split(r"[\u3002\uFF01\uFF1F!?;\n]", compact, maxsplit=1)[0]
    title = first or compact
    return title if len(title) <= limit else title[:limit].rstrip() + "…"


def load_recap_news(
    data_dir: Path,
    as_of: date,
    *,
    now: datetime | None = None,
    limit: int = 8,
) -> list[dict[str, str]]:
    """读取复盘截止时间前 24 小时的新闻,历史日期固定截止 15:30。"""
    current = now.astimezone(BEIJING_TZ) if now else datetime.now(BEIJING_TZ)
    if as_of == current.date():
        end = current
    else:
        end = datetime.combine(as_of, dt_time(15, 30), tzinfo=BEIJING_TZ)
    start = end - timedelta(hours=24)

    items = FinanceNewsStore(data_dir).list_between(
        int(start.timestamp()),
        int(end.timestamp()),
        limit,
    )
    result = []
    for item in items:
        content = " ".join(str(item.get("content") or "").split())
        title = str(item.get("title") or "").strip() or _fallback_title(content)
        snippet = content if len(content) <= 280 else content[:280].rstrip() + "…"
        published = datetime.fromisoformat(item["published_at"]).astimezone(BEIJING_TZ)
        result.append({
            "title": title,
            "snippet": snippet,
            "source": "财联社",
            "published_date": published.strftime("%m-%d %H:%M"),
        })
    return result
