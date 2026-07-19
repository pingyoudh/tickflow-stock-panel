"""财联社快讯查询与手动同步 API。"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from app.services.finance_news import FinanceNewsSyncInProgressError

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
