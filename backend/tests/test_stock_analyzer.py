from __future__ import annotations

import json
from datetime import date, datetime

import polars as pl

from app.services.stock_analyzer import _build_user_prompt, _load_financials


def test_stock_analysis_financial_context_is_json_safe(tmp_path):
    metrics_dir = tmp_path / "financials" / "metrics"
    income_dir = tmp_path / "financials" / "income"
    metrics_dir.mkdir(parents=True)
    income_dir.mkdir(parents=True)

    pl.DataFrame({
        "symbol": ["600889.SH"],
        "period_end": [date(2026, 3, 31)],
        "announce_date": [date(2026, 4, 28)],
        "updated_at": [datetime(2026, 4, 28, 15, 30)],
        "roe": [float("nan")],
    }).write_parquet(metrics_dir / "part.parquet")
    pl.DataFrame({
        "symbol": ["600889.SH"],
        "period_end": [date(2026, 3, 31)],
        "revenue": [123.45],
    }).write_parquet(income_dir / "part.parquet")

    fins = _load_financials(tmp_path, "600889.SH")

    assert fins["metrics"][0]["period_end"] == "2026-03-31"
    assert fins["metrics"][0]["announce_date"] == "2026-04-28"
    assert fins["metrics"][0]["updated_at"].startswith("2026-04-28T15:30")
    assert fins["metrics"][0]["roe"] is None
    json.dumps(fins, ensure_ascii=False)

    prompt = _build_user_prompt(
        [{"date": "2026-04-28", "close": 10.0}],
        fins,
        {"sr": []},
        10.0,
        "600889.SH",
        "",
    )
    assert "2026-03-31" in prompt
