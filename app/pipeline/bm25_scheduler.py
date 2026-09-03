"""BM25 重建调度器。

把 BM25 重建从文档变更的同步路径中解耦：
- 开发环境通过 FastAPI BackgroundTasks 异步执行。
- 生产环境写入 ingest_jobs 队列，按 collection_name 去重，实现防抖。
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import BackgroundTasks

from app.api.deps import (
    _get_settings,
    get_bm25_store_instance,
    get_document_store_instance,
    get_ingest_job_store_instance,
    get_vector_store_instance,
)
from app.settings.settings import AppConfig
from app.store.bm25_store import rebuild_bm25_after_change

logger = logging.getLogger(__name__)


async def schedule_bm25_rebuild(
    collection_name: str,
    tenant_id: str,
    settings: AppConfig,
    background_tasks: Optional[BackgroundTasks] = None,
) -> Optional[dict]:
    """调度一次指定 collection 的 BM25 重建。

    production：写入 `ingest_jobs`（job_type='bm25_rebuild'），同 collection 已存在
    queued/running job 时直接复用，实现防抖。
    dev：加入 `background_tasks` 异步执行。

    返回创建或复用的 job（production），或 None（dev）。
    """
    if not settings.retrieval.hybrid.enabled:
        return None

    if settings.app.env == "production":
        return await _schedule_bm25_rebuild_job(collection_name, tenant_id)

    if background_tasks is None:
        logger.warning("dev 模式下 schedule_bm25_rebuild 需要 background_tasks")
        return None
    background_tasks.add_task(_rebuild_bm25_in_background, collection_name)
    return None


async def _schedule_bm25_rebuild_job(
    collection_name: str,
    tenant_id: str,
) -> Optional[dict]:
    """production 模式：查询或创建 bm25_rebuild job。"""
    job_store = get_ingest_job_store_instance()

    atomic_get_or_create = getattr(job_store, "get_or_create_active_bm25_job", None)
    if callable(atomic_get_or_create):
        try:
            job = await atomic_get_or_create(
                tenant_id,
                collection_name=collection_name,
            )
            logger.info(
                "BM25 重建 job 已原子化排队/复用: collection=%s job_id=%s",
                collection_name,
                job["job_id"],
            )
            return job
        except Exception as exc:
            logger.warning("原子创建 BM25 重建 job 失败: %s", exc)
            return None

    def _is_active_rebuild(job: dict) -> bool:
        return (
            job.get("job_type") == "bm25_rebuild"
            and job.get("status") in ("queued", "running")
            and job.get("payload", {}).get("collection_name") == collection_name
        )

    try:
        active_jobs = await job_store.list_jobs(status="queued")
        active_jobs += await job_store.list_jobs(status="running")
        existing = next((j for j in active_jobs if _is_active_rebuild(j)), None)
        if existing:
            logger.info("BM25 重建 job 已存在，跳过创建: collection=%s", collection_name)
            return existing

        job = await job_store.create_job(
            tenant_id,
            "bm25_rebuild",
            payload={"collection_name": collection_name},
        )
        logger.info("BM25 重建 job 已创建: collection=%s job_id=%s", collection_name, job["job_id"])
        return job
    except Exception as exc:
        logger.warning("创建 BM25 重建 job 失败: %s", exc)
        return None


async def _rebuild_bm25_in_background(collection_name: str) -> None:
    """dev 模式 fire-and-forget 后台重建。"""
    try:
        await rebuild_bm25_for_collection(collection_name)
    except Exception as exc:
        logger.warning("BM25 后台重建失败，将在首次检索时懒加载: %s", exc)


async def rebuild_bm25_for_collection(collection_name: str, settings: Optional[AppConfig] = None) -> None:
    """直接重建指定 collection 的 BM25 索引。"""
    settings = settings or _get_settings()
    if not settings.retrieval.hybrid.enabled:
        return

    bm25_store = get_bm25_store_instance()
    vector_store = get_vector_store_instance()
    document_store = get_document_store_instance()

    # Cleanup-triggered rebuild jobs may run after the last vector was removed
    # and the physical collection was compacted.  A normal vector read uses
    # get-or-create semantics, so verify existence first to avoid recreating an
    # empty Chroma collection while trying to delete stale BM25 data.
    collection_exists = getattr(vector_store, "collection_exists", None)
    if callable(collection_exists):
        if not await collection_exists(collection_name=collection_name):
            bm25_store.invalidate_cache(collection_name)
            logger.info(
                "BM25 重建跳过：向量 collection 已不存在: collection=%s",
                collection_name,
            )
            return

    await rebuild_bm25_after_change(
        bm25_store,
        vector_store,
        document_store,
        settings,
        collection_name,
    )
