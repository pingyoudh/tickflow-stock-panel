"""Read-only THS Postgres gap audit and local ingest.

外部 Postgres 是严格只读来源。本模块只允许通过命名方法执行代码内白名单
SQL; 所有状态和同步结果只写本地 data/ 目录。
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from urllib.parse import (
    parse_qsl,
    quote,
    quote_from_bytes,
    unquote_to_bytes,
    urlencode,
    urlsplit,
    urlunsplit,
)

import duckdb
import polars as pl

from app import secrets_store
from app.config import settings
from app.services.ext_data import (
    ExtConfig,
    ExtConfigStore,
    ExtField,
    build_code_lookup,
    cast_df_to_schema,
    normalize_symbol,
)

logger = logging.getLogger(__name__)


GapStatus = Literal["covered", "missing", "snapshot_only", "not_usable", "deferred_heavy"]

ST_BLOCK_ID = "001005334012"
XST_BLOCK_ID = "001005334013"
FULL_BLOCK_MIN_COUNT = 100

ALLOWED_SQL_PREFIXES = (
    "SELECT",
    "WITH",
    "SHOW TRANSACTION_READ_ONLY",
    "BEGIN READ ONLY",
    "COMMIT",
    "ROLLBACK",
)

BANNED_SQL_RE = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|COPY|CALL|DO|"
    r"VACUUM|ANALYZE|LOCK|GRANT|REVOKE|SET\s+ROLE|TEMP|TEMPORARY|"
    r"MATERIALIZED|INDEX|PROCEDURE|FUNCTION"
    r")\b",
    re.IGNORECASE,
)


INDICATOR_FIELD_MAP = {
    "ths_total_float_shares_stock": "total_float_shares",
    "ths_market_value_stock": "market_value",
    "ths_sq_total_revenue_yoy_stock": "sq_total_revenue_yoy",
    "ths_revenue_yoy_sq_stock": "revenue_yoy_sq",
    "ths_sq_profit_dnrgal_yoy_stock": "sq_profit_dnrgal_yoy",
    "ths_fore_mbi_yoy_stock": "forecast_mbi_yoy",
    "ths_fore_op_yoy_stock": "forecast_op_yoy",
    "ths_interest_paid_date_stock": "interest_paid_date",
    "ths_roe_avg_by_ths_stock": "roe_avg_ths",
    "ths_corp_free_cf_stock": "corp_free_cf",
    "ths_np_stock": "net_profit",
    "ths_net_sales_margin_after_deduction_stock": "net_sales_margin_after_deduction",
    "ths_asset_liab_ratio_stock": "asset_liab_ratio",
    "ths_currency_fund_stock": "currency_fund",
    "ths_st_borrow_stock": "short_term_borrow",
    "ths_peg_lyr_stock": "peg_lyr",
    "ths_ps_ttm_stock": "ps_ttm",
    "ths_cf_interest_coverage_stock": "cf_interest_coverage",
    "ths_sales_gir_ttm_stock": "sales_gross_margin_ttm",
    "ths_gross_profit_lrr_sq_stock": "gross_profit_lrr_sq",
}

NUMERIC_INDICATOR_FIELDS = {
    field for indicator, field in INDICATOR_FIELD_MAP.items()
    if indicator != "ths_interest_paid_date_stock"
}
INDICATOR_NAMES = tuple(INDICATOR_FIELD_MAP)


@dataclass(frozen=True)
class SourceCoverage:
    rows: int | None = None
    min_date: str | None = None
    max_date: str | None = None
    symbols: int | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class LocalCoverage:
    rows: int | None = None
    min_date: str | None = None
    max_date: str | None = None
    symbols: int | None = None
    files: int = 0
    mode: str | None = None


class ThsPgSyncError(RuntimeError):
    """Base error for THS PG sync."""


class ReadonlyViolation(ThsPgSyncError):
    """Raised when readonly guarantees cannot be established."""


class ThsPgNotConfigured(ThsPgSyncError):
    """Raised when no DSN is configured."""


def assert_readonly_sql(sql: str) -> None:
    """Validate a SQL statement against the external-readonly contract."""
    compact = " ".join(sql.strip().split())
    upper = compact.upper()
    if not compact:
        raise ReadonlyViolation("empty SQL is not allowed")
    if ";" in compact.rstrip(";"):
        raise ReadonlyViolation("multiple SQL statements are not allowed")
    if BANNED_SQL_RE.search(compact):
        raise ReadonlyViolation("SQL contains a banned write/DDL keyword")
    if upper.startswith("WITH "):
        if " SELECT " not in f" {upper} ":
            raise ReadonlyViolation("WITH statements must be read-only SELECT queries")
        return
    if upper.startswith("SELECT "):
        return
    if upper in {"SHOW TRANSACTION_READ_ONLY", "BEGIN READ ONLY", "COMMIT", "ROLLBACK"}:
        return
    raise ReadonlyViolation(f"SQL is not in the readonly whitelist: {compact[:32]}")


def sanitize_error(exc: BaseException, dsn: str | None = None) -> str:
    text = str(exc)
    if dsn:
        text = text.replace(dsn, mask_dsn(dsn))
        text = text.replace(normalize_postgres_dsn(dsn), mask_dsn(dsn))
    return re.sub(r"://([^:/?#]+):([^@/?#]+)@", r"://\1:***@", text)


def _quote_userinfo_part(value: str) -> str:
    return quote_from_bytes(unquote_to_bytes(value), safe="")


def normalize_postgres_dsn(dsn: str) -> str:
    """Normalize URL userinfo so pasted raw passwords are valid libpq URLs."""
    try:
        parts = urlsplit(dsn)
    except Exception:
        return dsn
    if parts.scheme not in {"postgresql", "postgres"} or "@" not in parts.netloc:
        return dsn

    userinfo, host = parts.netloc.rsplit("@", 1)
    if ":" in userinfo:
        user, password = userinfo.split(":", 1)
        userinfo = f"{_quote_userinfo_part(user)}:{_quote_userinfo_part(password)}"
    else:
        userinfo = _quote_userinfo_part(userinfo)

    return urlunsplit((parts.scheme, f"{userinfo}@{host}", parts.path, parts.query, parts.fragment))


def mask_dsn(dsn: str) -> str:
    try:
        parts = urlsplit(normalize_postgres_dsn(dsn))
    except Exception:
        return secrets_store.mask(dsn)
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, host = netloc.rsplit("@", 1)
        user = userinfo.split(":", 1)[0]
        netloc = f"{user}:******@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def get_configured_dsn() -> str:
    return secrets_store.get_ths_pg_url() or getattr(settings, "ths_postgres_url", "") or ""


def readonly_dsn(dsn: str) -> str:
    """Return DSN with libpq readonly and timeout options appended."""
    dsn = normalize_postgres_dsn(dsn)
    parts = urlsplit(dsn)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    options = query.get("options", "")
    required_opts = [
        "-c default_transaction_read_only=on",
        "-c statement_timeout=600000",
        "-c lock_timeout=5000",
    ]
    for opt in required_opts:
        if opt not in options:
            options = f"{options} {opt}".strip()
    query["options"] = options
    query.setdefault("application_name", "tickflow_ths_pg_readonly")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, quote_via=quote), parts.fragment))


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


class ThsPgReadOnlyClient(AbstractContextManager):
    """External Postgres client with no public generic execute method."""

    def __init__(self, dsn: str | None = None) -> None:
        self._raw_dsn = dsn or get_configured_dsn()
        if not self._raw_dsn:
            raise ThsPgNotConfigured("THS Postgres 连接未配置")
        self._conn = None

    def __enter__(self) -> ThsPgReadOnlyClient:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except Exception as exc:
            raise ThsPgSyncError("缺少 psycopg 依赖, 请安装 psycopg[binary]") from exc
        try:
            self._conn = psycopg.connect(readonly_dsn(self._raw_dsn), row_factory=dict_row)
            self._readonly_preflight()
        except Exception as exc:
            if self._conn is not None:
                with suppress(Exception):
                    self._conn.close()
            if isinstance(exc, ThsPgSyncError):
                raise
            raise ThsPgSyncError(sanitize_error(exc, self._raw_dsn)) from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._conn is None:
            return
        try:
            sql = "ROLLBACK" if exc_type else "COMMIT"
            assert_readonly_sql(sql)
            with self._conn.cursor() as cur:
                cur.execute(sql)
        finally:
            self._conn.close()
            self._conn = None

    def _readonly_preflight(self) -> None:
        self._run_command("BEGIN READ ONLY")
        rows = self._query("SHOW TRANSACTION_READ_ONLY")
        val = str(rows[0].get("transaction_read_only", "") if rows else "").lower()
        if val != "on":
            raise ReadonlyViolation("外部 Postgres 未进入只读事务, 拒绝同步")

    def _run_command(self, sql: str) -> None:
        assert_readonly_sql(sql)
        if self._conn is None:
            raise ThsPgSyncError("Postgres connection is not open")
        with self._conn.cursor() as cur:
            cur.execute(sql)

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        assert_readonly_sql(sql)
        if self._conn is None:
            raise ThsPgSyncError("Postgres connection is not open")
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [dict(row) for row in rows]

    # ---- named readonly queries -------------------------------------------------

    def fetch_indicator_coverage(self) -> SourceCoverage:
        rows = self._query(
            """
            SELECT COUNT(*)::bigint AS rows,
                   MIN(date)::text AS min_date,
                   MAX(date)::text AS max_date,
                   COUNT(DISTINCT code)::bigint AS symbols
            FROM public.stock_indicators
            """
        )
        return _coverage_from_row(rows[0] if rows else {})

    def fetch_stock_basic_coverage(self) -> SourceCoverage:
        rows = self._query(
            """
            SELECT COUNT(*)::bigint AS rows,
                   MIN(data_date)::text AS min_date,
                   MAX(data_date)::text AS max_date,
                   COUNT(DISTINCT code)::bigint AS symbols
            FROM public.stock_basic_info
            """
        )
        return _coverage_from_row(rows[0] if rows else {})

    def fetch_block_coverage(self) -> SourceCoverage:
        rows = self._query(
            """
            SELECT COUNT(*)::bigint AS rows,
                   MIN(snapshot_date)::text AS min_date,
                   MAX(snapshot_date)::text AS max_date,
                   COUNT(DISTINCT stock_code)::bigint AS symbols,
                   COUNT(DISTINCT block_id)::bigint AS blocks
            FROM public.block_constituents_snapshots
            """
        )
        return _coverage_from_row(rows[0] if rows else {}, extra_keys=("blocks",))

    def fetch_latest_full_block_date(self) -> str | None:
        rows = self._query(
            """
            SELECT snapshot_date::text AS snapshot_date
            FROM public.block_constituents_snapshots
            GROUP BY snapshot_date
            HAVING COUNT(DISTINCT block_id) >= %s
            ORDER BY snapshot_date DESC
            LIMIT 1
            """,
            (FULL_BLOCK_MIN_COUNT,),
        )
        return rows[0]["snapshot_date"] if rows else None

    def fetch_st_coverage(self) -> SourceCoverage:
        rows = self._query(
            """
            SELECT COUNT(*)::bigint AS rows,
                   MIN(snapshot_date)::text AS min_date,
                   MAX(snapshot_date)::text AS max_date,
                   COUNT(DISTINCT stock_code)::bigint AS symbols
            FROM public.block_constituents_snapshots
            WHERE block_id IN (%s, %s)
            """,
            (ST_BLOCK_ID, XST_BLOCK_ID),
        )
        return _coverage_from_row(rows[0] if rows else {})

    def fetch_shareholder_coverage(self) -> SourceCoverage:
        rows = self._query(
            """
            SELECT
              (SELECT COUNT(*)::bigint FROM public.stock_shareholder_detail) AS detail_rows,
              (SELECT COUNT(*)::bigint FROM public.stock_shareholder_summary) AS summary_rows
            """
        )
        row = rows[0] if rows else {}
        total = int(row.get("detail_rows") or 0) + int(row.get("summary_rows") or 0)
        return SourceCoverage(
            rows=total,
            extra={
                "detail_rows": int(row.get("detail_rows") or 0),
                "summary_rows": int(row.get("summary_rows") or 0),
            },
        )

    def fetch_indicator_rows(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        where, params = _date_filter("date", start_date, end_date)
        conjunction = "AND" if where else "WHERE"
        placeholders = ", ".join(["%s"] * len(INDICATOR_NAMES))
        return self._query(
            f"""
            WITH latest AS (
                SELECT DISTINCT ON (indicator, code)
                       indicator, code, date, value, frequency_num
                FROM public.stock_indicators
                {where}
                {conjunction} indicator IN ({placeholders})
                ORDER BY indicator, code, date DESC,
                         frequency_num DESC NULLS LAST, value DESC NULLS LAST
            )
            SELECT indicator, code, date::text AS date, value, frequency_num
            FROM latest
            ORDER BY date, code, indicator
            """,
            (*params, *INDICATOR_NAMES),
        )

    def fetch_stock_basic_rows(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        where, params = _date_filter("data_date", start_date, end_date)
        return self._query(
            f"""
            SELECT DISTINCT ON (code)
                   code, data_date::text AS data_date, compvalue, pe, compname,
                   has_investigation, has_shareholding_change, roe_avg_ths,
                   dragon_tiger_net_buy_ratio
            FROM public.stock_basic_info
            {where}
            ORDER BY code, data_date DESC
            """,
            params,
        )

    def fetch_full_block_dates(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[str]:
        where, params = _date_filter("snapshot_date", start_date, end_date)
        rows = self._query(
            f"""
            SELECT snapshot_date::text AS snapshot_date
            FROM public.block_constituents_snapshots
            {where}
            GROUP BY snapshot_date
            HAVING COUNT(DISTINCT block_id) >= %s
            ORDER BY snapshot_date
            """,
            (*params, FULL_BLOCK_MIN_COUNT),
        )
        return [row["snapshot_date"] for row in rows]

    def fetch_block_memberships(self, snapshot_date: str) -> list[dict[str, Any]]:
        return self._query(
            """
            WITH latest_hierarchy AS (
                SELECT DISTINCT ON (block_id)
                       block_id, node_name, parent_block_id, block_type
                FROM public.block_hierarchy_snapshots
                WHERE snapshot_date <= %s
                ORDER BY block_id, snapshot_date DESC
            )
            SELECT c.snapshot_date::text AS snapshot_date,
                   c.block_id,
                   c.stock_code AS code,
                   c.stock_name,
                   COALESCE(c.block_type, h.block_type) AS block_type,
                   h.node_name AS block_name,
                   h.parent_block_id
            FROM public.block_constituents_snapshots c
            LEFT JOIN latest_hierarchy h ON h.block_id = c.block_id
            WHERE c.snapshot_date = %s
              AND c.block_id NOT IN (%s, %s)
            ORDER BY c.block_id, c.stock_code
            """,
            (snapshot_date, snapshot_date, ST_BLOCK_ID, XST_BLOCK_ID),
        )

    def fetch_st_dates(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[str]:
        where, params = _date_filter("snapshot_date", start_date, end_date)
        conjunction = "AND" if where else "WHERE"
        rows = self._query(
            f"""
            SELECT snapshot_date::text AS snapshot_date
            FROM public.block_constituents_snapshots
            {where}
            {conjunction} block_id IN (%s, %s)
            GROUP BY snapshot_date
            ORDER BY snapshot_date
            """,
            (*params, ST_BLOCK_ID, XST_BLOCK_ID),
        )
        return [row["snapshot_date"] for row in rows]

    def fetch_st_memberships(self, snapshot_date: str) -> list[dict[str, Any]]:
        return self._query(
            """
            SELECT snapshot_date::text AS snapshot_date,
                   block_id,
                   stock_code AS code,
                   stock_name
            FROM public.block_constituents_snapshots
            WHERE snapshot_date = %s
              AND block_id IN (%s, %s)
            ORDER BY stock_code, block_id
            """,
            (snapshot_date, ST_BLOCK_ID, XST_BLOCK_ID),
        )


def _coverage_from_row(row: dict[str, Any], extra_keys: Iterable[str] = ()) -> SourceCoverage:
    return SourceCoverage(
        rows=_maybe_int(row.get("rows")),
        min_date=_maybe_str(row.get("min_date")),
        max_date=_maybe_str(row.get("max_date")),
        symbols=_maybe_int(row.get("symbols")),
        extra={k: row.get(k) for k in extra_keys} if extra_keys else None,
    )


def _date_filter(column: str, start_date: str | None, end_date: str | None) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    params: list[Any] = []
    if start_date:
        clauses.append(f"{column} >= %s")
        params.append(start_date)
    if end_date:
        clauses.append(f"{column} <= %s")
        params.append(end_date)
    return (f"WHERE {' AND '.join(clauses)}" if clauses else "", tuple(params))


def _maybe_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _maybe_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _day_after(value: Any) -> str | None:
    if not value:
        return None
    try:
        return (date.fromisoformat(str(value)) + timedelta(days=1)).isoformat()
    except ValueError:
        return None


class ThsPgSyncState:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "user_data" / "ths_pg_sync_state.json"

    def load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"datasets": {}}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("ths_pg_sync_state.json malformed: %s", self._path)
            return {"datasets": {}}

    def save_dataset(self, dataset: str, payload: dict[str, Any]) -> None:
        state = self.load()
        datasets = state.setdefault("datasets", {})
        datasets[dataset] = {
            **datasets.get(dataset, {}),
            **payload,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(state, ensure_ascii=False, default=_json_default, indent=2),
            encoding="utf-8",
        )


class ThsPgSyncService:
    def __init__(
        self,
        data_dir: Path,
        client_factory: Callable[[], AbstractContextManager[Any]] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.state = ThsPgSyncState(data_dir)
        self._client_factory = client_factory or ThsPgReadOnlyClient

    def configured(self) -> bool:
        return bool(get_configured_dsn())

    def status(self) -> dict[str, Any]:
        dsn = get_configured_dsn()
        return {
            "configured": bool(dsn),
            "masked_dsn": mask_dsn(dsn) if dsn else "",
            "state": self.state.load(),
        }

    def audit_gaps(self) -> dict[str, Any]:
        local = self._local_coverages()
        source: dict[str, SourceCoverage] = {}
        readonly_ok = False
        error: str | None = None

        if self.configured():
            try:
                with self._client_factory() as client:
                    readonly_ok = True
                    source = {
                        "financial_metrics": client.fetch_indicator_coverage(),
                        "stock_basic_info": client.fetch_stock_basic_coverage(),
                        "block_membership": client.fetch_block_coverage(),
                        "st_status": client.fetch_st_coverage(),
                        "shareholders": client.fetch_shareholder_coverage(),
                    }
                    full_date = client.fetch_latest_full_block_date()
                    if full_date:
                        src = source["block_membership"]
                        source["block_membership"] = SourceCoverage(
                            rows=src.rows,
                            min_date=src.min_date,
                            max_date=src.max_date,
                            symbols=src.symbols,
                            extra={**(src.extra or {}), "latest_full_snapshot_date": full_date},
                        )
            except Exception as exc:
                error = sanitize_error(exc, get_configured_dsn())

        items = self._gap_items(local, source)
        return {
            "configured": self.configured(),
            "readonly_ok": readonly_ok,
            "error": error,
            "items": items,
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }

    def sync_recommended(
        self,
        on_progress: Callable[[str, int, str, int | None], None] | None = None,
    ) -> dict[str, Any]:
        def progress(stage: str, pct: int, msg: str, stage_pct: int | None = None) -> None:
            if on_progress:
                on_progress(stage, pct, msg, stage_pct)

        if not self.configured():
            raise ThsPgNotConfigured("THS Postgres 连接未配置")

        result: dict[str, Any] = {}
        progress("ths_pg_connect", 2, "连接外部 Postgres 并校验只读事务", 0)
        with self._client_factory() as client:
            progress("ths_pg_connect", 5, "只读事务已确认", 100)

            progress("ths_pg_financial_metrics", 10, "准备同步财务/估值指标", 0)
            result["financial_metrics"] = self.sync_financial_metrics(client, progress)

            progress("ths_pg_block_membership", 40, "读取完整概念/行业快照日期", 0)
            result["block_membership"] = self.sync_block_membership_history(client, progress)
            progress("ths_pg_block_membership", 75, "历史概念/行业成分完成", 100)

            progress("ths_pg_st_status", 80, "读取历史 ST/*ST 快照日期", 0)
            result["st_status"] = self.sync_st_status_history(client, progress)
            progress("ths_pg_st_status", 98, "历史 ST/*ST 状态完成", 100)

        result["synced_at"] = datetime.now().isoformat(timespec="seconds")
        progress("done", 100, "THS PG 数据缺口同步完成", 100)
        return result

    def sync_financial_metrics(
        self,
        client: Any,
        progress: Callable[[str, int, str, int | None], None] | None = None,
    ) -> dict[str, Any]:
        if progress:
            progress("ths_pg_financial_metrics", 11, "读取 stock_indicators", 5)
        indicator_rows = client.fetch_indicator_rows()
        if progress:
            progress(
                "ths_pg_financial_metrics",
                22,
                f"指标读取完成,共 {len(indicator_rows):,} 行",
                48,
            )

            progress("ths_pg_financial_metrics", 23, "读取 stock_basic_info", 52)
        basic_rows = client.fetch_stock_basic_rows()
        if progress:
            progress(
                "ths_pg_financial_metrics",
                29,
                f"主档读取完成,共 {len(basic_rows):,} 行",
                76,
            )
            progress("ths_pg_financial_metrics", 30, "转换并合并财务指标", 80)
        _, rows_latest = _build_financial_metric_frames(indicator_rows, basic_rows, self.data_dir)

        if progress:
            progress("ths_pg_financial_metrics", 33, "写入本地 Parquet", 92)
        ext_config = _financial_metrics_ext_config()
        _upsert_ext_config(self.data_dir, ext_config)
        snapshot = (
            rows_latest.drop("announce_date")
            if "announce_date" in rows_latest.columns
            else rows_latest
        )
        snapshot_rows_written = _write_ext_snapshot(snapshot, ext_config, self.data_dir)

        metrics_dir = self.data_dir / "financials" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        rows_latest.write_parquet(metrics_dir / "part.parquet")

        max_date = _max_date_from_df(rows_latest, "period_end")
        payload = {
            "last_success_at": datetime.now().isoformat(timespec="seconds"),
            "last_synced_date": max_date,
            "rows_written": len(rows_latest),
            "snapshot_rows_written": snapshot_rows_written,
            "timeseries_rows_written": None,
        }
        self.state.save_dataset("financial_metrics", payload)
        if progress:
            progress(
                "ths_pg_financial_metrics",
                35,
                f"财务/估值指标完成,共 {len(rows_latest):,} 行",
                100,
            )
        return payload

    def sync_block_membership_history(
        self,
        client: Any,
        progress: Callable[[str, int, str, int | None], None] | None = None,
    ) -> dict[str, Any]:
        config = _block_membership_ext_config()
        _upsert_ext_config(self.data_dir, config)

        previous = self.state.load().get("datasets", {}).get("block_membership", {})
        start_date = _day_after(previous.get("last_synced_date"))
        dates = client.fetch_full_block_dates(start_date=start_date)
        if progress:
            progress(
                "ths_pg_block_membership",
                42,
                f"发现 {len(dates):,} 个完整板块快照日",
                5,
            )
        total_rows = 0
        for idx, snap in enumerate(dates, start=1):
            rows = client.fetch_block_memberships(snap)
            df = _block_membership_frame(rows, self.data_dir)
            total_rows += _write_ext_timeseries_partition(df, config, self.data_dir, snap)
            if progress:
                stage_pct = int(idx * 100 / max(1, len(dates)))
                progress("ths_pg_block_membership", 40 + int(stage_pct * 0.35), f"{idx}/{len(dates)} {snap}", stage_pct)

        payload = {
            "last_success_at": datetime.now().isoformat(timespec="seconds"),
            "last_synced_date": dates[-1] if dates else previous.get("last_synced_date"),
            "rows_written": total_rows,
            "dates_written": len(dates),
        }
        self.state.save_dataset("block_membership", payload)
        return payload

    def sync_st_status_history(
        self,
        client: Any,
        progress: Callable[[str, int, str, int | None], None] | None = None,
    ) -> dict[str, Any]:
        config = _st_status_ext_config()
        _upsert_ext_config(self.data_dir, config)

        previous = self.state.load().get("datasets", {}).get("st_status", {})
        start_date = _day_after(previous.get("last_synced_date"))
        dates = client.fetch_st_dates(start_date=start_date)
        if progress:
            progress(
                "ths_pg_st_status",
                81,
                f"发现 {len(dates):,} 个 ST/*ST 快照日",
                5,
            )
        total_rows = 0
        for idx, snap in enumerate(dates, start=1):
            rows = client.fetch_st_memberships(snap)
            df = _st_status_frame(rows, self.data_dir)
            total_rows += _write_ext_timeseries_partition(df, config, self.data_dir, snap)
            if progress:
                stage_pct = int(idx * 100 / max(1, len(dates)))
                progress("ths_pg_st_status", 80 + int(stage_pct * 0.18), f"{idx}/{len(dates)} {snap}", stage_pct)

        payload = {
            "last_success_at": datetime.now().isoformat(timespec="seconds"),
            "last_synced_date": dates[-1] if dates else previous.get("last_synced_date"),
            "rows_written": total_rows,
            "dates_written": len(dates),
        }
        self.state.save_dataset("st_status", payload)
        return payload

    def _local_coverages(self) -> dict[str, LocalCoverage]:
        return {
            "daily": _parquet_coverage(self.data_dir, "kline_daily/**/*.parquet", "date", "symbol"),
            "index_daily": _parquet_coverage(self.data_dir, "kline_index_daily/**/*.parquet", "date", "symbol"),
            "etf_daily": _parquet_coverage(self.data_dir, "kline_etf_daily/**/*.parquet", "date", "symbol"),
            "minute": _parquet_coverage(self.data_dir, "kline_minute/**/*.parquet", "datetime", "symbol"),
            "adj_factor": _parquet_coverage(self.data_dir, "adj_factor/*.parquet", "trade_date", "symbol"),
            "finance_news": _parquet_coverage(self.data_dir, "finance_news/**/*.parquet", "published_at", None),
            "depth5": _parquet_coverage(self.data_dir, "depth5/**/*.parquet", "fetched_at", "symbol"),
            "financial_metrics": _parquet_coverage(self.data_dir, "financials/metrics/*.parquet", "period_end", "symbol"),
            "concept_snapshot": _ext_coverage(self.data_dir, "ext_gn_ths"),
            "industry_snapshot": _ext_coverage(self.data_dir, "ext_hy_ths"),
            "block_membership_history": _ext_coverage(self.data_dir, "ext_ths_block_membership_history"),
            "st_status_history": _ext_coverage(self.data_dir, "ext_ths_st_status_history"),
        }

    def _gap_items(
        self,
        local: dict[str, LocalCoverage],
        source: dict[str, SourceCoverage],
    ) -> list[dict[str, Any]]:
        financial = local["financial_metrics"]
        block_history = local["block_membership_history"]
        st_history = local["st_status_history"]
        shareholder = source.get("shareholders")

        return [
            _gap_item("daily", "日 K", "covered", local["daily"], None, False, "本地主行情覆盖充足"),
            _gap_item("index_daily", "指数日 K", "covered", local["index_daily"], None, False, "本地指数日线已存在"),
            _gap_item("etf_daily", "ETF 日 K", "covered", local["etf_daily"], None, False, "本地 ETF 日线已存在"),
            _gap_item("minute", "分钟 K", "covered", local["minute"], None, False, "本地分钟数据体量已很大"),
            _gap_item("adj_factor", "复权因子", "covered", local["adj_factor"], None, False, "本地复权因子已存在"),
            _gap_item("finance_news", "财联社快讯", "covered", local["finance_news"], None, False, "已有独立快讯同步"),
            _gap_item("depth5", "五档盘口", "covered", local["depth5"], None, False, "已有盘口本地采集"),
            _gap_item(
                "financial_metrics",
                "财务/估值指标",
                "missing" if not financial.rows else "covered",
                financial,
                source.get("financial_metrics"),
                not bool(financial.rows),
                "本地 financials/metrics 为空, 外库 stock_indicators 可补",
            ),
            _gap_item(
                "concept_industry_history",
                "历史概念/行业成分",
                "snapshot_only" if not block_history.rows else "covered",
                block_history,
                source.get("block_membership"),
                not bool(block_history.rows),
                "本地只有当前概念/行业快照, 外库有历史成分股快照",
            ),
            _gap_item(
                "st_status_history",
                "历史 ST/*ST 状态",
                "missing" if not st_history.rows else "covered",
                st_history,
                source.get("st_status"),
                not bool(st_history.rows),
                "本地主要靠名称判断, 外库有 ST/*ST 板块日快照",
            ),
            _gap_item(
                "shareholders",
                "股东明细/汇总",
                "not_usable",
                LocalCoverage(),
                shareholder,
                False,
                "外库股东明细/汇总当前无可用数据",
            ),
            _gap_item(
                "high_frequency",
                "高频全量",
                "deferred_heavy",
                local["minute"],
                None,
                False,
                "外库体量巨大且本地已有分钟数据, 首期不搬运",
            ),
        ]


def _gap_item(
    id: str,
    label: str,
    status: GapStatus,
    local: LocalCoverage,
    source: SourceCoverage | None,
    recommended: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "id": id,
        "label": label,
        "status": status,
        "recommended": recommended,
        "reason": reason,
        "local": local.__dict__,
        "source": source.__dict__ if source else None,
    }


def _parquet_coverage(
    data_dir: Path,
    pattern: str,
    date_col: str | None,
    symbol_col: str | None,
) -> LocalCoverage:
    files = sorted(data_dir.glob(pattern))
    if not files:
        return LocalCoverage(files=0)
    try:
        source = ",".join("'" + str(f).replace("\\", "/").replace("'", "''") + "'" for f in files)
        table_expr = f"read_parquet([{source}], union_by_name=true)"
        exprs = ["COUNT(*)::BIGINT AS row_count"]
        if date_col:
            exprs.append(f"MIN({date_col})::VARCHAR AS min_date")
            exprs.append(f"MAX({date_col})::VARCHAR AS max_date")
        if symbol_col:
            exprs.append(f"COUNT(DISTINCT {symbol_col})::BIGINT AS symbols")
        row = duckdb.connect().execute(f"SELECT {', '.join(exprs)} FROM {table_expr}").fetchone()
        return LocalCoverage(
            rows=int(row[0] or 0),
            min_date=str(row[1]) if date_col and row[1] is not None else None,
            max_date=str(row[2]) if date_col and row[2] is not None else None,
            symbols=int(row[3] or 0) if date_col and symbol_col and len(row) > 3 else None,
            files=len(files),
        )
    except Exception as exc:
        logger.warning("local parquet coverage failed for %s: %s", pattern, exc)
        return LocalCoverage(files=len(files))


def _ext_coverage(data_dir: Path, config_id: str) -> LocalCoverage:
    config = ExtConfigStore(data_dir).get(config_id)
    if not config:
        return LocalCoverage(files=0)
    if config.mode == "snapshot":
        cov = _parquet_coverage(data_dir, f"ext_data/{config_id}/*.parquet", None, "symbol")
        return LocalCoverage(rows=cov.rows, symbols=cov.symbols, files=cov.files, mode="snapshot")
    cov = _parquet_coverage(data_dir, f"ext_data/{config_id}/timeseries/**/*.parquet", "snapshot_date", "symbol")
    return LocalCoverage(
        rows=cov.rows,
        min_date=cov.min_date,
        max_date=cov.max_date,
        symbols=cov.symbols,
        files=cov.files,
        mode="timeseries",
    )


def _upsert_ext_config(data_dir: Path, config: ExtConfig) -> None:
    store = ExtConfigStore(data_dir)
    existing = store.get(config.id)
    if existing:
        existing.label = config.label
        existing.mode = config.mode
        existing.fields = config.fields
        existing.description = config.description
        existing.symbol_map = config.symbol_map
        existing.code_map = config.code_map
        store.upsert(existing)
    else:
        store.upsert(config)


def _financial_metrics_ext_config() -> ExtConfig:
    return ExtConfig(
        id="ext_ths_financial_metrics",
        label="THS 财务/估值指标",
        mode="snapshot",
        fields=[
            ExtField("symbol", "string", "标的代码"),
            ExtField("code", "string", "代码"),
            ExtField("period_end", "string", "最新指标日期"),
            *[ExtField(field, "float", field) for field in sorted(NUMERIC_INDICATOR_FIELDS)],
            ExtField("interest_paid_date", "string", "派息日"),
            ExtField("compname", "string", "公司名称"),
            ExtField("compvalue", "float", "公司价值"),
            ExtField("pe", "float", "市盈率"),
            ExtField("has_investigation", "bool", "近期调研"),
            ExtField("has_shareholding_change", "bool", "近期增减持"),
            ExtField("dragon_tiger_net_buy_ratio", "float", "龙虎榜净买入额占比"),
        ],
        description="从只读 THS Postgres 同步的最新财务/估值指标",
    )


def _block_membership_ext_config() -> ExtConfig:
    return ExtConfig(
        id="ext_ths_block_membership_history",
        label="THS 历史概念/行业成分",
        mode="timeseries",
        fields=[
            ExtField("symbol", "string", "标的代码"),
            ExtField("code", "string", "代码"),
            ExtField("股票代码", "string", "股票代码"),
            ExtField("股票简称", "string", "股票简称"),
            ExtField("snapshot_date", "string", "快照日期"),
            ExtField("block_id", "string", "板块ID"),
            ExtField("block_name", "string", "板块名称"),
            ExtField("parent_block_id", "string", "父板块ID"),
            ExtField("block_type", "string", "板块类型"),
        ],
        description="从只读 THS Postgres 同步的历史概念/行业成分股快照",
        symbol_map={"type": "computed", "from": "code", "method": "append_exchange"},
        code_map={"type": "computed", "from": "symbol", "method": "strip_exchange"},
    )


def _st_status_ext_config() -> ExtConfig:
    return ExtConfig(
        id="ext_ths_st_status_history",
        label="THS 历史 ST 状态",
        mode="timeseries",
        fields=[
            ExtField("symbol", "string", "标的代码"),
            ExtField("code", "string", "代码"),
            ExtField("股票代码", "string", "股票代码"),
            ExtField("股票简称", "string", "股票简称"),
            ExtField("snapshot_date", "string", "快照日期"),
            ExtField("is_st", "bool", "ST"),
            ExtField("is_xst", "bool", "*ST"),
        ],
        description="从只读 THS Postgres 同步的历史 ST/*ST 板块状态",
        symbol_map={"type": "computed", "from": "code", "method": "append_exchange"},
        code_map={"type": "computed", "from": "symbol", "method": "strip_exchange"},
    )


def _build_financial_metric_frames(
    indicator_rows: list[dict[str, Any]],
    basic_rows: list[dict[str, Any]],
    data_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not indicator_rows:
        empty = pl.DataFrame({"symbol": [], "code": [], "period_end": []})
        return empty, empty

    indicator_rows = [
        row for row in indicator_rows
        if row.get("indicator") in INDICATOR_FIELD_MAP
    ]
    if not indicator_rows:
        empty = pl.DataFrame({"symbol": [], "code": [], "period_end": []})
        return empty, empty

    lookup = build_code_lookup(data_dir)
    raw = _postgres_rows_to_frame(indicator_rows)
    fields = [INDICATOR_FIELD_MAP.get(item, item) for item in raw["indicator"].to_list()]
    raw = raw.with_columns(
        pl.col("code").cast(pl.Utf8),
        normalize_symbol(raw["code"].cast(pl.Utf8), lookup).alias("symbol"),
        pl.col("date").cast(pl.Date).alias("period_end"),
        pl.Series("field", fields),
        pl.col("value").cast(pl.Utf8).alias("value_text"),
        pl.col("value").cast(pl.Float64, strict=False).alias("numeric_value"),
    )

    numeric = raw.filter(pl.col("field").is_in(sorted(NUMERIC_INDICATOR_FIELDS))).pivot(
        values="numeric_value",
        index=["symbol", "code", "period_end"],
        on="field",
        aggregate_function="last",
    )
    text = raw.filter(pl.col("field") == "interest_paid_date").pivot(
        values="value_text",
        index=["symbol", "code", "period_end"],
        on="field",
        aggregate_function="last",
    )
    frames = [df for df in (numeric, text) if not df.is_empty()]
    ts = (
        frames[0]
        if len(frames) == 1
        else frames[0].join(frames[1], on=["symbol", "code", "period_end"], how="full", coalesce=True)
    )

    ts = ts.rename({"period_end": "date"}).with_columns(pl.col("date").alias("period_end"))
    latest_parts: list[pl.DataFrame] = []
    for field in sorted(set(raw["field"].to_list())):
        value_col = "value_text" if field == "interest_paid_date" else "numeric_value"
        part = (
            raw.filter(pl.col("field") == field)
            .sort(["symbol", "period_end"])
            .group_by("symbol", maintain_order=True)
            .tail(1)
            .select(["symbol", "code", pl.col("period_end").alias(f"{field}__date"), pl.col(value_col).alias(field)])
        )
        latest_parts.append(part)
    latest = latest_parts[0]
    for part in latest_parts[1:]:
        latest = latest.join(part, on=["symbol", "code"], how="full", coalesce=True)
    date_cols = [c for c in latest.columns if c.endswith("__date")]
    latest = latest.with_columns(
        pl.max_horizontal([pl.col(c) for c in date_cols]).alias("period_end")
    ).drop(date_cols)

    if basic_rows:
        basic = _latest_basic_frame(basic_rows, data_dir)
        latest = latest.join(basic, on=["symbol", "code"], how="left")
    latest = latest.with_columns(pl.lit(None).cast(pl.Date).alias("announce_date"))
    keep = ["symbol", "code", "period_end", "announce_date", *[c for c in latest.columns if c not in {"symbol", "code", "period_end", "announce_date"}]]
    return ts, latest.select(keep)


def _latest_basic_frame(rows: list[dict[str, Any]], data_dir: Path) -> pl.DataFrame:
    lookup = build_code_lookup(data_dir)
    df = _postgres_rows_to_frame(rows)
    return (
        df.with_columns(
            pl.col("code").cast(pl.Utf8),
            normalize_symbol(df["code"].cast(pl.Utf8), lookup).alias("symbol"),
            pl.col("data_date").cast(pl.Date),
            pl.col("compvalue").cast(pl.Float64, strict=False),
            pl.col("pe").cast(pl.Float64, strict=False),
            pl.col("roe_avg_ths").cast(pl.Float64, strict=False),
            pl.col("dragon_tiger_net_buy_ratio").cast(pl.Float64, strict=False),
            pl.col("has_investigation").cast(pl.Boolean, strict=False),
            pl.col("has_shareholding_change").cast(pl.Boolean, strict=False),
        )
        .sort(["symbol", "data_date"])
        .group_by("symbol", maintain_order=True)
        .tail(1)
        .select([
            "symbol", "code", "compname", "compvalue", "pe",
            "has_investigation", "has_shareholding_change", "dragon_tiger_net_buy_ratio",
        ])
    )


def _block_membership_frame(rows: list[dict[str, Any]], data_dir: Path) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    lookup = build_code_lookup(data_dir)
    df = _postgres_rows_to_frame(rows)
    symbols = normalize_symbol(df["code"].cast(pl.Utf8), lookup)
    df = df.with_columns(
        pl.col("code").cast(pl.Utf8),
        symbols.alias("symbol"),
        pl.col("code").cast(pl.Utf8).alias("股票代码"),
        pl.col("stock_name").cast(pl.Utf8).alias("股票简称"),
        pl.col("snapshot_date").cast(pl.Date),
    )
    return df.select([
        "symbol", "code", "股票代码", "股票简称", "snapshot_date",
        "block_id", "block_name", "parent_block_id", "block_type",
    ])


def _st_status_frame(rows: list[dict[str, Any]], data_dir: Path) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    lookup = build_code_lookup(data_dir)
    df = _postgres_rows_to_frame(rows)
    symbols = normalize_symbol(df["code"].cast(pl.Utf8), lookup)
    df = df.with_columns(
        pl.col("code").cast(pl.Utf8),
        symbols.alias("symbol"),
        pl.col("snapshot_date").cast(pl.Date),
        (pl.col("block_id") == ST_BLOCK_ID).alias("is_st_row"),
        (pl.col("block_id") == XST_BLOCK_ID).alias("is_xst_row"),
    )
    return (
        df.group_by(["symbol", "code", "snapshot_date"])
        .agg(
            pl.col("stock_name").drop_nulls().first().alias("股票简称"),
            pl.col("is_st_row").max().alias("is_st"),
            pl.col("is_xst_row").max().alias("is_xst"),
        )
        .with_columns(pl.col("code").alias("股票代码"))
        .select(["symbol", "code", "股票代码", "股票简称", "snapshot_date", "is_st", "is_xst"])
    )


def _postgres_rows_to_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a stable Polars frame from psycopg rows.

    PostgreSQL NUMERIC values arrive as Decimal objects whose scale can vary by
    row. Polars may infer a narrow decimal schema from early rows and then fail
    on later values, so normalize those values before schema inference.
    """
    normalized = [
        {
            key: float(value) if isinstance(value, Decimal) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    return pl.DataFrame(normalized, infer_schema_length=None)


def _write_ext_snapshot(df: pl.DataFrame, config: ExtConfig, data_dir: Path) -> int:
    cfg_dir = data_dir / "ext_data" / config.id
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if not df.is_empty() and "symbol" in df.columns:
        lookup = build_code_lookup(data_dir)
        df = df.with_columns(normalize_symbol(df["symbol"], lookup).alias("symbol"))
    df = cast_df_to_schema(df, config.fields)
    df.write_parquet(cfg_dir / "part.parquet")
    return len(df)


def _write_ext_timeseries_partition(
    df: pl.DataFrame,
    config: ExtConfig,
    data_dir: Path,
    snapshot_date: str,
) -> int:
    out_dir = data_dir / "ext_data" / config.id / "timeseries" / f"date={snapshot_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.is_empty():
        df = pl.DataFrame({f.name: [] for f in config.fields})
    elif "symbol" in df.columns:
        lookup = build_code_lookup(data_dir)
        df = df.with_columns(normalize_symbol(df["symbol"], lookup).alias("symbol"))
    df = cast_df_to_schema(df, config.fields)
    df.write_parquet(out_dir / "part.parquet")
    return len(df)


def _max_date_from_df(df: pl.DataFrame, column: str) -> str | None:
    if df.is_empty() or column not in df.columns:
        return None
    value = df.select(pl.col(column).max()).item()
    return value.isoformat() if hasattr(value, "isoformat") else str(value)
