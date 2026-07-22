"""财联社快讯查询与手动同步 API。"""
from __future__ import annotations

import logging
from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.daily_news_summary import (
    analyze_daily_news_stream,
    get_daily_summary_status,
)
from app.services.finance_news import BEIJING_TZ, FinanceNewsSyncInProgressError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/finance-news", tags=["finance-news"])


@router.get("")
def list_finance_news(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = Query(default=None),
) -> dict:
    service = request.app.state.finance_news_service
    try:
        return service.list_page(limit, cursor)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/refresh")
async def refresh_finance_news(request: Request) -> dict:
    service = request.app.state.finance_news_service
    try:
        return await service.sync()
    except FinanceNewsSyncInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("手动同步财联社快讯失败: %s", exc)
        raise HTTPException(status_code=502, detail=f"财联社快讯同步失败: {exc}") from exc


class DailySummaryRequest(BaseModel):
    as_of: date | None = None
    force: bool = False


@router.get("/daily-summary")
def get_daily_summary(
    request: Request,
    as_of: date | None = None,
) -> dict:
    service = request.app.state.finance_news_service
    target = as_of or datetime.now(BEIJING_TZ).date()
    try:
        return get_daily_summary_status(
            service.store.data_dir,
            service.store,
            target,
            repo=getattr(request.app.state, "repo", None),
            quote_service=getattr(request.app.state, "quote_service", None),
            depth_service=getattr(request.app.state, "depth_service", None),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/daily-summary/analyze")
async def analyze_daily_summary(request: Request, req: DailySummaryRequest):
    service = request.app.state.finance_news_service
    target = req.as_of or datetime.now(BEIJING_TZ).date()

    async def stream_gen():
        async for event in analyze_daily_news_stream(
            service.store.data_dir,
            service.store,
            target,
            force=req.force,
            repo=getattr(request.app.state, "repo", None),
            quote_service=getattr(request.app.state, "quote_service", None),
            depth_service=getattr(request.app.state, "depth_service", None),
        ):
            yield event + "\n"

    return StreamingResponse(
        stream_gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
