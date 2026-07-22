from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import polars as pl
import pytest

from app.services import ths_pg_sync as ths_pg_sync_module
from app.services.ths_pg_sync import (
    ST_BLOCK_ID,
    XST_BLOCK_ID,
    ReadonlyViolation,
    SourceCoverage,
    ThsPgSyncService,
    assert_readonly_sql,
    mask_dsn,
    normalize_postgres_dsn,
    readonly_dsn,
)


def test_readonly_sql_guard_allows_only_safe_statements() -> None:
    for sql in (
        "SELECT 1",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "SHOW transaction_read_only",
        "BEGIN READ ONLY",
        "COMMIT",
        "ROLLBACK",
    ):
        assert_readonly_sql(sql)

    for sql in (
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1",
        "DELETE FROM t",
        "CREATE TEMP TABLE x(a int)",
        "COPY t TO STDOUT",
        "CALL refresh_cache()",
        "VACUUM public.stock_indicators",
        "SELECT 1; SELECT 2",
    ):
        with pytest.raises(ReadonlyViolation):
            assert_readonly_sql(sql)


def test_postgres_dsn_normalizes_raw_password_special_chars() -> None:
    raw = "postgresql://dengzhe:G4vcg^&3%a@nj-postgres-54r4t1bj.sql.tencentcdb.com:28092/fx-stock"

    normalized = normalize_postgres_dsn(raw)
    assert normalized.startswith("postgresql://dengzhe:G4vcg%5E%263%25a@")
    assert normalize_postgres_dsn("postgresql://u:p%40ss%25@example.invalid/db") == (
        "postgresql://u:p%40ss%25@example.invalid/db"
    )

    readonly = readonly_dsn(raw)
    assert "G4vcg%5E%263%25a" in readonly
    assert "+default_transaction_read_only" not in readonly
    assert "-c%20default_transaction_read_only%3Don" in readonly
    query = parse_qs(urlsplit(readonly).query)
    assert "default_transaction_read_only=on" in query["options"][0]
    assert "statement_timeout=600000" in query["options"][0]
    assert "lock_timeout=5000" in query["options"][0]

    masked = mask_dsn(raw)
    assert "G4vcg" not in masked
    assert "dengzhe:******@" in masked


class FakeClient:
    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def fetch_indicator_coverage(self) -> SourceCoverage:
        return SourceCoverage(rows=2, min_date="2026-07-17", max_date="2026-07-17", symbols=1)

    def fetch_stock_basic_coverage(self) -> SourceCoverage:
        return SourceCoverage(rows=1, min_date="2026-07-17", max_date="2026-07-17", symbols=1)

    def fetch_block_coverage(self) -> SourceCoverage:
        return SourceCoverage(rows=1, min_date="2026-07-15", max_date="2026-07-17", symbols=1)

    def fetch_latest_full_block_date(self) -> str:
        return "2026-07-15"

    def fetch_st_coverage(self) -> SourceCoverage:
        return SourceCoverage(rows=1, min_date="2026-07-17", max_date="2026-07-17", symbols=1)

    def fetch_shareholder_coverage(self) -> SourceCoverage:
        return SourceCoverage(rows=0, extra={"detail_rows": 0, "summary_rows": 0})

    def fetch_indicator_rows(self):
        return [
            {
                "indicator": "ths_market_value_stock",
                "code": "000001",
                "date": "2026-07-17",
                "value": "123.4",
                "frequency_num": 1,
            },
            {
                "indicator": "ths_ps_ttm_stock",
                "code": "000001",
                "date": "2026-07-17",
                "value": "2.5",
                "frequency_num": 1,
            },
        ]

    def fetch_stock_basic_rows(self):
        return [
            {
                "code": "000001",
                "data_date": "2026-07-17",
                "compvalue": "10",
                "pe": "12",
                "compname": "平安银行",
                "has_investigation": True,
                "has_shareholding_change": False,
                "roe_avg_ths": "9",
                "dragon_tiger_net_buy_ratio": "0.5",
            }
        ]

    def fetch_full_block_dates(self, start_date=None, end_date=None):
        dates = ["2026-07-15"]
        return [
            item for item in dates
            if (not start_date or item >= start_date) and (not end_date or item <= end_date)
        ]

    def fetch_block_memberships(self, snapshot_date: str):
        return [
            {
                "snapshot_date": snapshot_date,
                "block_id": "881001",
                "code": "000001",
                "stock_name": "平安银行",
                "block_type": "concept",
                "block_name": "金融科技",
                "parent_block_id": "root",
            }
        ]

    def fetch_st_dates(self, start_date=None, end_date=None):
        dates = ["2026-07-17"]
        return [
            item for item in dates
            if (not start_date or item >= start_date) and (not end_date or item <= end_date)
        ]

    def fetch_st_memberships(self, snapshot_date: str):
        return [
            {
                "snapshot_date": snapshot_date,
                "block_id": ST_BLOCK_ID,
                "code": "000001",
                "stock_name": "平安银行",
            },
            {
                "snapshot_date": snapshot_date,
                "block_id": XST_BLOCK_ID,
                "code": "000002",
                "stock_name": "万科A",
            },
        ]


def _service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ThsPgSyncService:
    monkeypatch.setattr(
        ths_pg_sync_module,
        "get_configured_dsn",
        lambda: "postgresql://readonly:secret@example.invalid/fx-stock",
    )
    (tmp_path / "instruments").mkdir()
    pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "code": ["000001", "000002"],
        }
    ).write_parquet(tmp_path / "instruments" / "instruments.parquet")
    return ThsPgSyncService(tmp_path, client_factory=FakeClient)


def test_gap_audit_classifies_missing_snapshot_only_and_unusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path
    (data_dir / "kline_daily" / "date=2026-07-17").mkdir(parents=True)
    pl.DataFrame(
        {"symbol": ["000001.SZ"], "date": ["2026-07-17"], "close": [10.0]}
    ).write_parquet(data_dir / "kline_daily" / "date=2026-07-17" / "part.parquet")

    result = _service(data_dir, monkeypatch).audit_gaps()
    items = {item["id"]: item for item in result["items"]}

    assert result["readonly_ok"] is True
    assert items["daily"]["status"] == "covered"
    assert items["financial_metrics"]["status"] == "missing"
    assert items["financial_metrics"]["recommended"] is True
    assert items["concept_industry_history"]["status"] == "snapshot_only"
    assert items["shareholders"]["status"] == "not_usable"
    assert items["high_frequency"]["status"] == "deferred_heavy"


def test_sync_recommended_writes_only_local_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, monkeypatch)
    progress_events: list[tuple[str, int, str, int | None]] = []
    result = service.sync_recommended(
        on_progress=lambda stage, pct, msg, stage_pct: progress_events.append(
            (stage, pct, msg, stage_pct)
        )
    )

    assert result["financial_metrics"]["rows_written"] == 1
    assert result["block_membership"]["rows_written"] == 1
    assert result["st_status"]["rows_written"] == 2
    assert progress_events[0][:2] == ("ths_pg_connect", 2)
    assert any("指标读取完成" in event[2] for event in progress_events)
    assert progress_events[-1][:2] == ("done", 100)

    metrics = pl.read_parquet(tmp_path / "financials" / "metrics" / "part.parquet")
    assert metrics.select("symbol").item() == "000001.SZ"
    assert metrics.select("market_value").item() == 123.4

    financial_snapshot = pl.read_parquet(
        tmp_path / "ext_data" / "ext_ths_financial_metrics" / "part.parquet"
    )
    assert len(financial_snapshot) == 1
    assert financial_snapshot.select("compname").item() == "平安银行"

    block = pl.read_parquet(
        tmp_path
        / "ext_data"
        / "ext_ths_block_membership_history"
        / "timeseries"
        / "date=2026-07-15"
        / "part.parquet"
    )
    assert block.select("block_name").item() == "金融科技"

    st = pl.read_parquet(
        tmp_path
        / "ext_data"
        / "ext_ths_st_status_history"
        / "timeseries"
        / "date=2026-07-17"
        / "part.parquet"
    )
    assert st.filter(pl.col("symbol") == "000001.SZ").select("is_st").item() is True
    assert st.filter(pl.col("symbol") == "000002.SZ").select("is_xst").item() is True

    incremental = service.sync_recommended()
    assert incremental["block_membership"]["dates_written"] == 0
    assert incremental["block_membership"]["last_synced_date"] == "2026-07-15"
    assert incremental["st_status"]["dates_written"] == 0
    assert incremental["st_status"]["last_synced_date"] == "2026-07-17"


def test_stock_basic_decimal_scales_are_normalized_before_polars_inference(
    tmp_path: Path,
) -> None:
    (tmp_path / "instruments").mkdir()
    pl.DataFrame(
        {"symbol": ["000001.SZ"], "code": ["000001"]}
    ).write_parquet(tmp_path / "instruments" / "instruments.parquet")

    rows = [
        {
            "code": "000001",
            "data_date": "2026-07-17",
            "compvalue": Decimal("10.00"),
            "pe": Decimal("12.00"),
            "compname": "平安银行",
            "has_investigation": True,
            "has_shareholding_change": False,
            "roe_avg_ths": Decimal("9.00"),
            "dragon_tiger_net_buy_ratio": Decimal("0.00"),
        }
        for _ in range(100)
    ]
    rows.append(
        {
            **rows[0],
            "data_date": "2026-07-18",
            "dragon_tiger_net_buy_ratio": Decimal("-22.8132"),
        }
    )

    result = ths_pg_sync_module._latest_basic_frame(rows, tmp_path)

    assert result.select("dragon_tiger_net_buy_ratio").item() == pytest.approx(-22.8132)
    assert result.schema["dragon_tiger_net_buy_ratio"] == pl.Float64


class FailingReadonlyClient:
    def __enter__(self):
        raise ReadonlyViolation("外部 Postgres 未进入只读事务, 拒绝同步")

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_readonly_preflight_failure_does_not_write_local_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ths_pg_sync_module,
        "get_configured_dsn",
        lambda: "postgresql://readonly:secret@example.invalid/fx-stock",
    )
    service = ThsPgSyncService(tmp_path, client_factory=FailingReadonlyClient)
    with pytest.raises(ReadonlyViolation):
        service.sync_recommended()
    assert not (tmp_path / "financials" / "metrics" / "part.parquet").exists()
