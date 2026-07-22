from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest

from app.api import financials
from app.tickflow.capabilities import CapabilityDenied, CapabilitySet


def _request(data_dir):
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=data_dir))
    state = SimpleNamespace(repo=repo, capabilities=CapabilitySet())
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_local_financial_metrics_are_readable_without_sync_capability(tmp_path, monkeypatch):
    metrics_dir = tmp_path / "financials" / "metrics"
    metrics_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "period_end": [date(2026, 7, 18)],
            "roe_avg_ths": [8.25],
        }
    ).write_parquet(metrics_dir / "part.parquet")
    request = _request(tmp_path)
    monkeypatch.setattr(financials, "_financial_allowed", lambda _capset: False)

    status = financials.financial_status(request)

    assert status["available"] is True
    assert status["can_sync"] is False
    assert status["tables"]["metrics"]["rows"] == 1
    assert status["tables"]["metrics"]["symbols"] == 1
    assert status["tables"]["metrics"]["updated_at"]
    assert status["tables"]["income"]["rows"] == 0

    result = financials.get_metrics(request, symbol="600000.SH")
    assert result["data"][0]["roe_avg_ths"] == 8.25

    with pytest.raises(CapabilityDenied):
        financials.sync_table(request, "metrics")


def test_financial_status_reports_no_local_data_without_capability(tmp_path, monkeypatch):
    request = _request(tmp_path)
    monkeypatch.setattr(financials, "_financial_allowed", lambda _capset: False)

    status = financials.financial_status(request)

    assert status["available"] is False
    assert status["can_sync"] is False
    assert all(info["rows"] == 0 for info in status["tables"].values())
