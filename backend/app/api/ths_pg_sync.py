"""THS Postgres read-only gap audit and sync APIs."""
from __future__ import annotations

import asyncio
import concurrent.futures as _cf
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app import secrets_store
from app.api.data import invalidate_storage_cache
from app.services.pipeline_jobs import (
    LONG_JOB_STALL_TIMEOUT_S,
    LONG_JOB_TIMEOUT_S,
    job_store,
    release_run_slot,
    try_acquire_run_slot,
)
from app.services.ths_pg_sync import (
    ThsPgNotConfigured,
    ThsPgSyncError,
    ThsPgSyncService,
    mask_dsn,
    normalize_postgres_dsn,
    sanitize_error,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ths-pg", tags=["ths-pg"])

_executor = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ths-pg-sync")


class ThsPgConfigReq(BaseModel):
    url: str | None = None


def _service(request: Request) -> ThsPgSyncService:
    return ThsPgSyncService(request.app.state.repo.store.data_dir)


@router.get("/status")
def status(request: Request) -> dict:
    return _service(request).status()


@router.put("/config")
def save_config(body: ThsPgConfigReq) -> dict:
    url = (body.url or "").strip()
    if not url:
        secrets_store.clear("ths_pg_url")
        return {"configured": False, "masked_dsn": ""}
    if not url.startswith(("postgresql://", "postgres://")):
        raise HTTPException(400, "仅支持 postgresql:// DSN")
    normalized_url = normalize_postgres_dsn(url)
    secrets_store.save({"ths_pg_url": normalized_url})
    return {"configured": True, "masked_dsn": mask_dsn(normalized_url)}


@router.get("/gaps")
def gaps(request: Request) -> dict:
    return _service(request).audit_gaps()


@router.post("/sync")
async def sync(request: Request) -> dict:
    """Run recommended gap sync.

    外部 Postgres 访问由 ThsPgReadOnlyClient 强制只读; 本端只写本地数据目录。
    """
    service = _service(request)
    if not service.configured():
        raise HTTPException(400, "THS Postgres 连接未配置")

    job_store.reap_stale()
    job_id, is_new = job_store.create(
        timeout_s=LONG_JOB_TIMEOUT_S,
        stall_timeout_s=LONG_JOB_STALL_TIMEOUT_S,
    )
    if not is_new:
        return {"job_id": job_id, "reused": True}

    async def task() -> None:
        if not try_acquire_run_slot():
            job_store.fail(job_id, "已有数据任务在运行(或上一次任务卡死未结束),请稍后再试")
            return
        try:
            job_store.start(job_id)
            loop = asyncio.get_event_loop()

            def progress(stage: str, pct: int, msg: str, stage_pct: int | None = None) -> None:
                job_store.progress(job_id, stage, pct, msg, stage_pct=stage_pct)

            result = await loop.run_in_executor(
                _executor,
                lambda: service.sync_recommended(on_progress=progress),
            )
            job_store.succeed(job_id, result)
            invalidate_storage_cache()
            request.app.state.repo.refresh_cache()
        except ThsPgNotConfigured as exc:
            job_store.fail(job_id, str(exc))
        except ThsPgSyncError as exc:
            job_store.fail(job_id, str(exc))
        except Exception as exc:
            logger.exception("ths pg sync failed")
            job_store.fail(job_id, sanitize_error(exc))
        finally:
            release_run_slot()

    background_task = asyncio.create_task(task())
    background_task.add_done_callback(lambda done: done.exception())
    return {"job_id": job_id, "reused": False}
