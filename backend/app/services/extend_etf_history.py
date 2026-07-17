"""向前扩展 ETF 历史, 并重算对应的复权行情与技术指标。"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta

from app.services import index_sync
from app.tickflow.capabilities import Cap, CapabilitySet
from app.tickflow.repository import KlineRepository


def _noop(stage: str, pct: int, msg: str, **kwargs) -> None:
    pass


def _offset(value: int, unit: str) -> timedelta:
    if unit == "month":
        return timedelta(days=value * 30)
    if unit == "year":
        return timedelta(days=value * 365)
    raise ValueError("unit 只支持 month/year")


def earliest_etf_date(repo: KlineRepository) -> date | None:
    base = repo.store.data_dir / "kline_etf_daily"
    dates = sorted(
        path.name[5:]
        for path in base.glob("date=*")
        if path.is_dir() and path.name.startswith("date=")
    ) if base.exists() else []
    return date.fromisoformat(dates[0]) if dates else None


def run_extend_etf_history(
    repo: KlineRepository,
    capset: CapabilitySet,
    value: int,
    unit: str,
    on_progress: Callable | None = None,
) -> dict:
    """扩展 ETF 历史, 并重拉目标区间以保持复权与指标一致。"""
    emit = on_progress or _noop
    today = date.today()
    earliest = earliest_etf_date(repo)
    target_start = (earliest or today) - _offset(value, unit)

    emit("extend_etf_history", 3, "同步 ETF 维表…", stage_pct=3)
    instrument_count = index_sync.sync_etf_instruments(repo)
    instruments = repo.get_etf_instruments()
    if instruments.is_empty() or "symbol" not in instruments.columns:
        return {"error": "ETF 标的列表为空, 请检查数据源或权限"}
    symbols = sorted(set(instruments["symbol"].to_list()))
    emit(
        "extend_etf_history", 8, f"ETF 维表完成, 共 {len(symbols)} 只",
        stage_pct=8,
    )

    adj_rows = 0
    affected_symbols: list[str] = []
    if capset.has(Cap.ADJ_FACTOR):
        def adj_progress(current: int, total: int) -> None:
            stage_pct = int(100 * current / total) if total else 100
            emit(
                "extend_etf_history", 8 + int(17 * current / max(total, 1)),
                f"ETF 复权因子批次 {current}/{total}",
                stage_pct=stage_pct, skip_log=current < total,
            )

        adj_rows, affected_symbols = index_sync.sync_etf_adj_factor(
            symbols,
            repo,
            capset,
            start_time=datetime.combine(target_start, datetime.min.time()),
            end_time=datetime.combine(today, datetime.max.time()),
            on_chunk_done=adj_progress,
        )
        emit("extend_etf_history", 25, f"ETF 复权因子完成, 写入 {adj_rows} 行", stage_pct=100)
    else:
        emit("extend_etf_history", 25, "ETF 复权因子跳过(当前数据源不支持)", stage_pct=100)

    def daily_progress(current: int, total: int) -> None:
        stage_pct = int(100 * current / total) if total else 100
        emit(
            "extend_etf_history", 25 + int(70 * current / max(total, 1)),
            f"ETF 日K与指标批次 {current}/{total}",
            stage_pct=stage_pct, skip_log=current < total,
        )

    # 从目标起点完整重拉到今天, 让新补入的历史成为原有区间的指标预热数据。
    daily_rows = index_sync.sync_and_persist_etf_daily(
        repo,
        capset,
        start_date=datetime.combine(target_start, datetime.min.time()),
        end_date=datetime.combine(today, datetime.max.time()),
        on_chunk_done=daily_progress,
    )
    emit("extend_etf_history", 98, "刷新 ETF 数据视图…", stage_pct=100)
    repo.refresh_index_views()
    emit("extend_etf_history", 100, f"ETF 历史已扩展至 {target_start}", stage_pct=100)

    return {
        "asset_type": "etf",
        "earliest_before": earliest.isoformat() if earliest else None,
        "earliest_after": target_start.isoformat(),
        "latest_date": today.isoformat(),
        "instrument_count": instrument_count or len(symbols),
        "universe_size": len(symbols),
        "adj_factor_rows": adj_rows,
        "adj_factor_symbols": len(affected_symbols),
        "daily_rows": daily_rows,
    }
