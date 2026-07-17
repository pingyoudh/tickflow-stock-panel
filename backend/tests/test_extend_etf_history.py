from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from app.services import extend_etf_history, index_sync, kline_sync
from app.tickflow import repository
from app.tickflow.capabilities import Cap, CapabilityLimits, CapabilitySet
from app.tickflow.repository import KlineRepository


class _Capabilities:
    @staticmethod
    def has(_cap) -> bool:
        return True


class _Repo:
    def __init__(self, data_dir) -> None:
        self.store = SimpleNamespace(data_dir=data_dir)
        self.refreshed = False

    @staticmethod
    def get_etf_instruments() -> pl.DataFrame:
        return pl.DataFrame({"symbol": ["510300.SH", "510500.SH"]})

    def refresh_index_views(self) -> None:
        self.refreshed = True


def test_extend_etf_history_uses_earliest_date_for_full_refresh(tmp_path, monkeypatch):
    earliest = date(2025, 7, 16)
    (tmp_path / "kline_etf_daily" / f"date={earliest.isoformat()}").mkdir(parents=True)
    repo = _Repo(tmp_path)
    captured = {}

    monkeypatch.setattr(extend_etf_history.index_sync, "sync_etf_instruments", lambda _repo: 2)

    def sync_adj(symbols, _repo, _caps, *, start_time, end_time, on_chunk_done):
        captured["adj"] = (symbols, start_time, end_time)
        on_chunk_done(1, 1)
        return 7, symbols

    def sync_daily(_repo, _caps, *, start_date, end_date, on_chunk_done):
        captured["daily"] = (start_date, end_date)
        on_chunk_done(1, 1)
        return 123

    monkeypatch.setattr(extend_etf_history.index_sync, "sync_etf_adj_factor", sync_adj)
    monkeypatch.setattr(
        extend_etf_history.index_sync, "sync_and_persist_etf_daily", sync_daily
    )

    result = extend_etf_history.run_extend_etf_history(
        repo, _Capabilities(), 4, "year"
    )

    expected_start = earliest.replace() - extend_etf_history.timedelta(days=4 * 365)
    assert captured["adj"][1].date() == expected_start
    assert captured["daily"][0].date() == expected_start
    assert captured["daily"][1].date() == date.today()
    assert result["earliest_before"] == earliest.isoformat()
    assert result["earliest_after"] == expected_start.isoformat()
    assert result["daily_rows"] == 123
    assert result["adj_factor_rows"] == 7
    assert repo.refreshed is True


def test_extend_etf_history_rejects_empty_instrument_table(tmp_path, monkeypatch):
    repo = _Repo(tmp_path)
    monkeypatch.setattr(repo, "get_etf_instruments", lambda: pl.DataFrame())
    monkeypatch.setattr(extend_etf_history.index_sync, "sync_etf_instruments", lambda _repo: 0)

    result = extend_etf_history.run_extend_etf_history(
        repo, _Capabilities(), 1, "year"
    )

    assert "error" in result
    assert "ETF 标的列表为空" in result["error"]


def test_etf_daily_batches_are_combined_before_persist(tmp_path, monkeypatch):
    class Repo:
        def __init__(self) -> None:
            self.store = SimpleNamespace(data_dir=tmp_path)
            self.daily_writes: list[pl.DataFrame] = []
            self.enriched_writes: list[pl.DataFrame] = []
            self.refreshed = False

        @staticmethod
        def get_etf_instruments() -> pl.DataFrame:
            return pl.DataFrame({"symbol": ["510300.SH", "510500.SH", "159915.SZ"]})

        def append_etf_daily(self, frame: pl.DataFrame) -> None:
            self.daily_writes.append(frame)

        def append_etf_enriched(self, frame: pl.DataFrame) -> None:
            self.enriched_writes.append(frame)

        def refresh_index_views(self) -> None:
            self.refreshed = True

    calls: list[list[str]] = []

    def fetch(symbols, **_kwargs):
        calls.append(symbols)
        return pl.DataFrame({
            "symbol": symbols,
            "date": [date(2026, 7, 16)] * len(symbols),
            "open": [1.0] * len(symbols),
            "high": [1.0] * len(symbols),
            "low": [1.0] * len(symbols),
            "close": [1.0] * len(symbols),
            "volume": [100.0] * len(symbols),
            "amount": [100.0] * len(symbols),
        })

    monkeypatch.setattr(index_sync.kline_sync, "sync_daily_batch", fetch)
    monkeypatch.setattr(index_sync, "sleep_between_batches", lambda *_args: None)
    monkeypatch.setattr(index_sync.preferences, "get_index_daily_batch_size", lambda: 2)
    monkeypatch.setattr(index_sync, "compute_enriched", lambda frame, **_kwargs: frame)
    progress: list[tuple[int, int]] = []
    repo = Repo()
    caps = CapabilitySet({Cap.KLINE_DAILY_BATCH: CapabilityLimits(batch=2)})

    rows = index_sync.sync_and_persist_etf_daily(
        repo,
        caps,
        on_chunk_done=lambda current, total: progress.append((current, total)),
    )

    assert len(calls) == 2
    assert rows == 3
    assert len(repo.daily_writes) == 1
    assert len(repo.enriched_writes) == 1
    assert repo.daily_writes[0].height == 3
    assert progress == [(1, 2), (2, 2)]
    assert repo.refreshed is True


@pytest.mark.parametrize(
    "writer",
    [KlineRepository._atomic_write_parquet, kline_sync._atomic_write_parquet],
)
def test_atomic_parquet_write_retries_transient_windows_lock(
    tmp_path, monkeypatch, writer
):
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(path: Path, target: Path):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "file is temporarily locked", str(target))
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(repository.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(kline_sync.time, "sleep", lambda _seconds: None)
    out = tmp_path / "part.parquet"

    writer(pl.DataFrame({"value": [1]}), out)

    assert attempts == 3
    assert pl.read_parquet(out).to_dict(as_series=False) == {"value": [1]}
