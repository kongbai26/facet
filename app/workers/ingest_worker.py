"""Persistent ingest job worker helpers."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import Optional

from app.api.deps import (
    _get_settings,
    get_bm25_store_instance,
    get_document_store_instance,
    get_embedding_provider_instance,
    get_ingest_job_store_instance,
    get_index_profile_store_instance,
    get_knowledge_base_store_instance,
    get_conversation_store_instance,
    get_namespace_store_instance,
    get_vector_store_instance,
)
from app.pipeline.bm25_scheduler import rebuild_bm25_for_collection
from app.pipeline.embedding_readiness import ensure_embedding_ready
from app.pipeline.ingest import ingest_document
from app.pipeline.tokenizer_policy import (
    TOKENIZER_CAPABILITY_STATUS_REASON,
    is_tokenizer_capability_error,
)
from app.pipeline.index_lifecycle import (
    finalize_pending_document_removals_after_cutover,
    reconcile_knowledge_base_index,
    schedule_knowledge_base_reconciliations_after_source_changes,
)
from app.pipeline.knowledge_base_lifecycle import purge_knowledge_base_runtime
from app.rag_scope import get_tenant_rag_collection_name, resolve_embedding_dimension
from app.utils.user_errors import sanitize_user_error_message

logger = logging.getLogger(__name__)

# The embedded queue worker runs in the same process as Facet. Keep a small
# in-memory heartbeat so system status can distinguish "dependencies are
# reachable" from "queued work is actually being consumed".
_worker_heartbeats: dict[str, float] = {}


def get_ingest_worker_status(*, stale_after_seconds: float) -> dict:
    """Return local embedded-worker liveness without exposing worker internals."""
    now = time.monotonic()
    latest = max(_worker_heartbeats.values(), default=None)
    age_seconds = None if latest is None else max(0.0, now - latest)
    return {
        "running": age_seconds is not None and age_seconds <= max(1.0, stale_after_seconds),
        "last_heartbeat_age_seconds": age_seconds,
    }


def _record_worker_heartbeat(worker_id: str) -> None:
    _worker_heartbeats[worker_id] = time.monotonic()


async def _heartbeat_ingest_worker_forever(worker_id: str, interval_seconds: float) -> None:
    """Keep liveness current while a long document job is being processed."""
    while True:
        _record_worker_heartbeat(worker_id)
        await asyncio.sleep(interval_seconds)


def _job_fencing_kwargs(job: dict) -> dict:
    """Return ownership fencing data for a claimed persistent job."""
    locked_by = job.get("locked_by")
    attempts = job.get("attempts")
    if locked_by is None or attempts is None:
        return {}
    return {"locked_by": locked_by, "attempts": int(attempts)}


async def _mark_job_succeeded(job_store, job: dict) -> bool:
    kwargs = _job_fencing_kwargs(job)
    try:
        result = await job_store.mark_succeeded(job["job_id"], **kwargs)
    except TypeError:
        # Compatibility with small test doubles and older integrations.
        result = await job_store.mark_succeeded(job["job_id"])
    return result is not False


async def _mark_job_failed(job_store, job: dict, error_message: str) -> bool:
    kwargs = _job_fencing_kwargs(job)
    try:
        result = await job_store.mark_failed(job["job_id"], error_message, **kwargs)
    except TypeError:
        result = await job_store.mark_failed(job["job_id"], error_message)
    return result is not False


def _source_doc_ids_for_job(job: dict) -> list[str]:
    """Return source document ids represented by an ingest/reindex job."""
    ids: list[str] = []
    if job.get("doc_id"):
        ids.append(str(job["doc_id"]))
    payload = job.get("payload") or {}
    if payload.get("doc_id"):
        ids.append(str(payload["doc_id"]))
    ids.extend(str(doc_id) for doc_id in payload.get("doc_ids") or [] if doc_id)
    return list(dict.fromkeys(ids))


def _source_failure_reason(error_message: str) -> str:
    if is_tokenizer_capability_error(error_message):
        return TOKENIZER_CAPABILITY_STATUS_REASON
    lowered = (error_message or "").lower()
    if any(token in lowered for token in ("embedding", "tokenizer", "模型服务", "模型配置")):
        return "embedding_unavailable"
    return "ingest_failed"


async def _renew_job_lease_forever(job_store, job: dict, settings) -> None:
    """Keep long-running builds from being reclaimed while still executing."""
    renew_lock = getattr(job_store, "renew_lock", None)
    if not callable(renew_lock):
        return
    fencing = _job_fencing_kwargs(job)
    if not fencing:
        return
    timeout_seconds = max(3, int(getattr(settings.queue, "lock_timeout_seconds", 300)))
    interval = max(1.0, min(30.0, timeout_seconds / 3))
    while True:
        await asyncio.sleep(interval)
        try:
            renewed = await renew_lock(job["job_id"], **fencing)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("job lease renewal failed: job_id=%s", job.get("job_id"), exc_info=True)
            continue
        if not renewed:
            logger.warning("job lease was lost while executing: job_id=%s", job.get("job_id"))
            return


async def execute_ingest_job(job: dict, settings=None) -> None:
    settings = settings or _get_settings()
    job_store = get_ingest_job_store_instance()
    document_store = get_document_store_instance()
    job_type = job.get("job_type") or "ingest"
    lease_task = None
    if _job_fencing_kwargs(job) and callable(getattr(job_store, "renew_lock", None)):
        lease_task = asyncio.create_task(_renew_job_lease_forever(job_store, job, settings))

    try:
        if job_type == "batch_ingest":
            await _execute_batch_ingest_job(job, settings, document_store)
        elif job_type == "batch_reindex":
            await _execute_batch_reindex_job(job, settings, document_store)
        elif job_type == "bm25_rebuild":
            await _execute_bm25_rebuild_job(job, settings, document_store)
        elif job_type == "index_candidate":
            while True:
                await _execute_index_candidate_job(job, settings, document_store)
                complete_iteration = getattr(
                    job_store,
                    "complete_index_candidate_iteration",
                    None,
                )
                if not callable(complete_iteration):
                    break
                try:
                    completion = await complete_iteration(
                        job["job_id"],
                        **_job_fencing_kwargs(job),
                    )
                except TypeError:
                    completion = await complete_iteration(job["job_id"])
                if completion.get("action") == "rerun":
                    job = completion["job"]
                    logger.info(
                        "candidate source changed while running; rebuilding latest snapshot: kb_id=%s",
                        job.get("kb_id"),
                    )
                    continue
                if completion.get("action") == "lost":
                    logger.warning(
                        "candidate completion fenced out because ownership changed: job_id=%s",
                        job.get("job_id"),
                    )
                return
        elif job_type == "knowledge_base_delete":
            await _execute_knowledge_base_delete_job(job, settings, document_store)
        elif job_type == "reindex":
            await _execute_single_reindex_job(job, settings, document_store)
        else:
            doc_id = job.get("doc_id")
            if not doc_id:
                raise RuntimeError("ingest job missing doc_id")
            doc = await document_store.get(doc_id)
            if not doc:
                raise RuntimeError("document deleted")
            if doc.get("status") != "processing":
                raise RuntimeError(f"document status is {doc.get('status')}")
            await _execute_ingest_job(job, settings, document_store)
        marked = await _mark_job_succeeded(job_store, job)
        if not marked:
            logger.warning(
                "job completion fenced out because ownership changed: job_id=%s",
                job.get("job_id"),
            )
    except Exception as exc:
        logger.exception("%s job failed: %s", job_type, job.get("job_id"))
        user_error = sanitize_user_error_message(str(exc), "任务执行失败，请稍后重试。")
        latest_candidate_payload = dict(job.get("payload") or {})
        atomic_candidate_failure = getattr(job_store, "mark_index_candidate_failed", None)
        if job_type == "index_candidate" and callable(atomic_candidate_failure):
            try:
                failure = await atomic_candidate_failure(
                    job["job_id"],
                    user_error,
                    **_job_fencing_kwargs(job),
                )
            except TypeError:
                failure = await atomic_candidate_failure(job["job_id"], user_error)
            marked = bool(failure.get("marked"))
            if failure.get("job") is not None:
                latest_candidate_payload = dict(failure["job"].get("payload") or {})
        else:
            if job_type == "index_candidate":
                get_job = getattr(job_store, "get", None)
                if callable(get_job):
                    try:
                        latest_job = await get_job(job["job_id"])
                        if latest_job is not None:
                            latest_candidate_payload = dict(latest_job.get("payload") or {})
                    except Exception:
                        logger.warning(
                            "failed to refresh coalesced candidate payload: job_id=%s",
                            job.get("job_id"),
                            exc_info=True,
                        )
            marked = await _mark_job_failed(
                job_store,
                job,
                user_error,
            )
        if not marked:
            logger.warning(
                "job failure fenced out because ownership changed: job_id=%s",
                job.get("job_id"),
            )
            return
        if job_type in {"ingest", "batch_ingest", "reindex", "batch_reindex"}:
            allowed_statuses = (
                ["processing"]
                if job_type in {"ingest", "batch_ingest"}
                else ["reindex_queued", "reindexing"]
            )
            failure_reason = _source_failure_reason(user_error)
            for doc_id in _source_doc_ids_for_job(job):
                try:
                    await document_store.update_status_if(
                        doc_id,
                        allowed_statuses,
                        "failed",
                        error_message=user_error,
                        chunks_count=0,
                        status_reason=failure_reason,
                    )
                except Exception:
                    logger.exception(
                        "failed to persist source document failure: doc_id=%s job_id=%s",
                        doc_id,
                        job.get("job_id"),
                    )
        if job_type == "knowledge_base_delete":
            payload = job.get("payload") or {}
            kb_id = job.get("kb_id") or payload.get("kb_id")
            tenant_id = job.get("tenant_id") or ""
            if kb_id and tenant_id:
                try:
                    await get_knowledge_base_store_instance().mark_delete_failed(
                        kb_id,
                        tenant_id,
                        user_error,
                    )
                except Exception:
                    logger.exception(
                        "failed to persist knowledge base deletion failure: kb_id=%s",
                        kb_id,
                    )
        candidate_snapshot_changed = job_type == "index_candidate" and "文档已变化" in str(exc)
        if job_type == "index_candidate" and not candidate_snapshot_changed:
            payload = latest_candidate_payload
            delete_doc_ids = list(payload.get("delete_doc_ids") or [])
            if payload.get("reason") == "document_delete" and payload.get("doc_id"):
                delete_doc_ids.append(payload["doc_id"])
            for doc_id in dict.fromkeys(str(item) for item in delete_doc_ids if item):
                try:
                    await document_store.update_status_if(
                        doc_id,
                        ["deleting"],
                        "delete_failed",
                        error_message=user_error,
                    )
                except Exception:
                    logger.exception(
                        "failed to persist document deletion failure: doc_id=%s",
                        doc_id,
                    )
        if candidate_snapshot_changed:
            payload = latest_candidate_payload
            kb_id = job.get("kb_id") or payload.get("kb_id")
            tenant_id = job.get("tenant_id") or ""
            if kb_id and tenant_id:
                retry_payload = {
                    **payload,
                    "kb_id": kb_id,
                    "tenant_slug": payload.get("tenant_slug") or "default",
                    "auto_activate": True,
                }
                retry_payload.pop("rerun_requested", None)
                if payload.get("reason") != "document_delete":
                    retry_payload["reason"] = "source_changed_during_candidate"
                await job_store.get_or_create_active_kb_job(
                    tenant_id,
                    "index_candidate",
                    kb_id=kb_id,
                    payload=retry_payload,
                )
    finally:
        if lease_task is not None:
            lease_task.cancel()
            with suppress(asyncio.CancelledError):
                await lease_task


async def _execute_ingest_job(job: dict, settings, document_store) -> None:
    payload = job.get("payload") or {}
    save_path_value = payload.get("save_path")
    if not save_path_value:
        raise RuntimeError("ingest job payload missing save_path")
    document = await document_store.get(job["doc_id"])
    if not document:
        raise RuntimeError("document deleted")
    await _ensure_document_kb_active(document)

    save_path = Path(save_path_value)
    embedding_provider = get_embedding_provider_instance()
    vector_store = get_vector_store_instance()
    bm25_store = get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None
    await ingest_document(
        save_path,
        job["doc_id"],
        settings,
        embedding_provider,
        vector_store,
        document_store,
        bm25_store,
    )
    document = await document_store.get(job["doc_id"])
    if document:
        await _refresh_active_knowledge_bases([document], settings, document_store)


async def _execute_single_reindex_job(job: dict, settings, document_store) -> None:
    doc_id = job.get("doc_id")
    if not doc_id:
        raise RuntimeError("reindex job missing doc_id")
    doc = await document_store.get(doc_id)
    if not doc:
        raise RuntimeError("document deleted")
    if doc.get("status") not in {"reindex_queued", "reindexing"}:
        raise RuntimeError(f"document status is {doc.get('status')}")
    await _ensure_document_kb_active(doc)
    await _reindex_single_doc(doc_id, job.get("payload") or {}, settings, document_store)
    document = await document_store.get(doc_id)
    if document:
        await _refresh_active_knowledge_bases([document], settings, document_store)


async def _refresh_active_knowledge_bases(documents: list[dict], settings, document_store) -> None:
    """Queue one serialized KB reconciliation after source writes complete."""
    await schedule_knowledge_base_reconciliations_after_source_changes(
        documents=documents,
        ingest_job_store=get_ingest_job_store_instance(),
        reason="source_changed",
    )


async def _ensure_document_kb_active(document: dict) -> None:
    kb_id = document.get("kb_id")
    if not kb_id:
        return
    knowledge_base = await get_knowledge_base_store_instance().get(kb_id)
    if knowledge_base is not None and knowledge_base.get("status") != "active":
        raise RuntimeError("knowledge base is not active")


async def _schedule_bm25_rebuild_after_delete(
    *,
    tenant_id: str,
    collection_name: str,
    settings,
) -> None:
    """Queue a durable rebuild, falling back to an immediate rebuild if queueing fails."""
    try:
        await get_ingest_job_store_instance().get_or_create_active_bm25_job(
            tenant_id,
            collection_name=collection_name,
        )
    except Exception:
        logger.exception(
            "failed to queue BM25 rebuild after deletion; rebuilding inline: collection=%s",
            collection_name,
        )
        try:
            await rebuild_bm25_for_collection(collection_name, settings)
        except Exception:
            # Deletion is already physically committed at this point. Keep it
            # successful and let readiness/next source change retry the derived index.
            logger.exception(
                "inline BM25 rebuild after deletion also failed: collection=%s",
                collection_name,
            )


async def _execute_knowledge_base_delete_job(job: dict, settings, document_store) -> None:
    kb_id = job.get("kb_id") or (job.get("payload") or {}).get("kb_id")
    tenant_id = job.get("tenant_id")
    if not kb_id or not tenant_id:
        raise RuntimeError("knowledge_base_delete job missing tenant_id or kb_id")

    knowledge_base_store = get_knowledge_base_store_instance()
    kb = await knowledge_base_store.get(kb_id)
    if not kb:
        return
    if kb.get("tenant_id") != tenant_id:
        raise RuntimeError("knowledge base tenant mismatch")
    if kb.get("status") in {"active", "delete_failed"}:
        await knowledge_base_store.mark_deleting(kb_id, tenant_id)
        kb = await knowledge_base_store.get(kb_id)

    job_store = get_ingest_job_store_instance()
    documents = await document_store.list_by_knowledge_base(kb_id)
    document_ids = {str(item["doc_id"]) for item in documents}
    legacy_collections = {
        get_tenant_rag_collection_name(
            settings.vectorstore.collection_prefix,
            item.get("embedding_model") or settings.embedding.openai.model_name,
            tenant_slug=item.get("tenant_slug"),
            embedding_dimension=item.get("embedding_dimension"),
        )
        for item in documents
    }

    def _affects_deleted_kb(candidate: dict) -> bool:
        if candidate.get("kb_id") == kb_id or candidate.get("doc_id") in document_ids:
            return True
        payload = candidate.get("payload") or {}
        if set(str(item) for item in (payload.get("doc_ids") or [])) & document_ids:
            return True
        return (
            candidate.get("job_type") == "bm25_rebuild"
            and payload.get("collection_name") in legacy_collections
        )

    # Do not delete physical data while a source job is still writing. Queued
    # work is cancelled; running work is allowed to finish and is fenced from
    # activating a generation after the KB entered deleting state.
    poll_interval = float(settings.queue.knowledge_base_delete_poll_interval_seconds)
    deadline = asyncio.get_running_loop().time() + float(
        settings.queue.knowledge_base_delete_wait_timeout_seconds
    )
    while True:
        await job_store.cancel_queued_jobs_for_kb(kb_id, except_job_id=job["job_id"])
        active_jobs = await job_store.list_jobs(
            status="running",
            tenant_id=tenant_id,
            limit=1000,
        )
        active_jobs = [
            item
            for item in active_jobs
            if item.get("job_id") != job["job_id"] and _affects_deleted_kb(item)
        ]
        if not active_jobs:
            break
        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError("knowledge base still has running source jobs")
        await asyncio.sleep(poll_interval)

    result = await purge_knowledge_base_runtime(
        kb=kb,
        settings=settings,
        document_store=document_store,
        vector_store=get_vector_store_instance(),
        index_profile_store=get_index_profile_store_instance(),
        conversation_store=get_conversation_store_instance(),
        namespace_store=get_namespace_store_instance(),
        knowledge_base_store=knowledge_base_store,
        bm25_store=get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None,
        job_store=job_store,
        delete_job_id=job["job_id"],
    )
    if settings.retrieval.hybrid.enabled:
        for collection_name in result.get("affected_legacy_collections") or []:
            await _schedule_bm25_rebuild_after_delete(
                tenant_id=tenant_id,
                collection_name=collection_name,
                settings=settings,
            )


async def _execute_batch_ingest_job(job: dict, settings, document_store) -> None:
    """处理批量上传 job：逐文档摄入（延迟重建），最后统一重建一次 BM25。"""
    payload = job.get("payload") or {}
    doc_ids = payload.get("doc_ids", [])
    save_paths = payload.get("save_paths", [])
    tenant_slug = payload.get("tenant_slug")

    if len(doc_ids) != len(save_paths):
        raise RuntimeError("batch_ingest job payload doc_ids and save_paths length mismatch")

    embedding_provider = get_embedding_provider_instance()
    vector_store = get_vector_store_instance()
    bm25_store = get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None
    failed_doc_ids: list[str] = []
    skipped_doc_ids: list[str] = []

    for doc_id, save_path_value in zip(doc_ids, save_paths):
        save_path = Path(save_path_value)
        doc = await document_store.get(doc_id)
        if not doc:
            logger.warning("batch_ingest job %s: document %s status is %s, skipping",
                           job.get("job_id"), doc_id, "missing")
            skipped_doc_ids.append(doc_id)
            continue
        if doc.get("status") != "processing":
            logger.warning(
                "batch_ingest job %s: document %s status is %s, skipping",
                job.get("job_id"),
                doc_id,
                doc.get("status"),
            )
            skipped_doc_ids.append(doc_id)
            continue
        try:
            await _ensure_document_kb_active(doc)
        except RuntimeError:
            skipped_doc_ids.append(doc_id)
            continue
        try:
            await ingest_document(
                save_path,
                doc_id,
                settings,
                embedding_provider,
                vector_store,
                document_store,
                bm25_store,
                defer_bm25_rebuild=True,
            )
        except Exception as exc:
            logger.exception("batch_ingest job %s: document %s ingest failed",
                             job.get("job_id"), doc_id)
            await document_store.update_status_if(
                doc_id,
                ["processing"],
                "failed",
                error_message=sanitize_user_error_message(
                    str(exc),
                    "文档摄入失败，请检查模型配置后重试。",
                ),
                chunks_count=0,
                status_reason="ingest_failed",
            )
            failed_doc_ids.append(doc_id)

    if bm25_store and settings.retrieval.hybrid.enabled:
        collection_name = payload.get("collection_name")
        if not collection_name:
            embedding_dimension = await resolve_embedding_dimension(embedding_provider)
            collection_name = get_tenant_rag_collection_name(
                settings.vectorstore.collection_prefix,
                payload.get("to_embedding_model") or settings.embedding.openai.model_name,
                tenant_slug=tenant_slug,
                embedding_dimension=payload.get("to_embedding_dimension") if payload.get("to_embedding_dimension") is not None else embedding_dimension,
            )
        await rebuild_bm25_for_collection(collection_name, settings)

    refreshed_documents: list[dict] = []
    for doc_id in doc_ids:
        document = await document_store.get(doc_id)
        if document and document.get("kb_id") and document.get("status") == "ready":
            refreshed_documents.append(document)
    await _refresh_active_knowledge_bases(refreshed_documents, settings, document_store)

    if skipped_doc_ids:
        logger.info("batch_ingest job %s skipped documents: %s", job.get("job_id"), skipped_doc_ids)
    if failed_doc_ids:
        raise RuntimeError(f"batch_ingest partial failures: {failed_doc_ids}")


async def _execute_batch_reindex_job(job: dict, settings, document_store) -> None:
    """处理批量重新索引 job：逐文档 reindex（延迟重建），最后统一重建一次 BM25。"""
    payload = job.get("payload") or {}
    doc_ids = payload.get("doc_ids", [])
    save_paths = payload.get("save_paths", [])
    target_collection_name = payload.get("collection_name")
    failed_doc_ids: list[str] = []
    old_collection_names: set[str] = set()
    skipped_doc_ids: list[str] = []

    if len(doc_ids) != len(save_paths):
        raise RuntimeError("batch_reindex job payload doc_ids and save_paths length mismatch")

    for doc_id, save_path_value in zip(doc_ids, save_paths):
        doc = await document_store.get(doc_id)
        if not doc:
            logger.warning(
                "batch_reindex job %s: document %s no longer exists, skipping",
                job.get("job_id"),
                doc_id,
            )
            skipped_doc_ids.append(doc_id)
            continue
        if doc.get("status") not in {"reindex_queued", "reindexing"}:
            logger.warning(
                "batch_reindex job %s: document %s status is %s, skipping",
                job.get("job_id"),
                doc_id,
                doc.get("status"),
            )
            skipped_doc_ids.append(doc_id)
            continue
        try:
            await _ensure_document_kb_active(doc)
        except RuntimeError:
            skipped_doc_ids.append(doc_id)
            continue
        try:
            item_payload = {
                "save_path": save_path_value,
                "from_embedding_model": doc.get("embedding_model") or settings.embedding.openai.model_name,
                "from_embedding_dimension": doc.get("embedding_dimension"),
                "tenant_slug": doc.get("tenant_slug"),
                "to_embedding_model": payload.get("to_embedding_model") or settings.embedding.openai.model_name,
                "to_embedding_dimension": payload.get("to_embedding_dimension"),
                "collection_name": target_collection_name,
            }
            old_collection = await _reindex_single_doc(
                doc_id,
                item_payload,
                settings,
                document_store,
                defer_bm25_rebuild=True,
            )
            if old_collection:
                old_collection_names.add(old_collection)
        except Exception as exc:
            logger.exception("batch_reindex job %s: document %s reindex failed",
                             job.get("job_id"), doc_id)
            await document_store.update_status_if(
                doc_id,
                ["reindex_queued", "reindexing"],
                "failed",
                error_message=sanitize_user_error_message(
                    str(exc),
                    "文档重新摄入失败，请检查模型配置后重试。",
                ),
                chunks_count=0,
                status_reason="reindex_failed",
            )
            failed_doc_ids.append(doc_id)

    if settings.retrieval.hybrid.enabled:
        bm25_store = get_bm25_store_instance()
        if bm25_store:
            for old_collection in old_collection_names:
                bm25_store.invalidate_cache(old_collection)
            collection_name = target_collection_name
            if not collection_name:
                embedding_provider = get_embedding_provider_instance()
                embedding_dimension = await resolve_embedding_dimension(embedding_provider)
                collection_name = get_tenant_rag_collection_name(
                    settings.vectorstore.collection_prefix,
                    settings.embedding.openai.model_name,
                    tenant_slug=None,
                    embedding_dimension=embedding_dimension,
                )
            await rebuild_bm25_for_collection(collection_name, settings)

    # Reindex replaces source vectors too.  Profile-routed KBs therefore need
    # the same generation refresh as normal uploads; omitting this left reads
    # pinned to a candidate built from the previous source representation.
    refreshed_documents = []
    for doc_id in doc_ids:
        document = await document_store.get(doc_id)
        if document and document.get("status") == "ready" and document.get("kb_id"):
            refreshed_documents.append(document)
    await _refresh_active_knowledge_bases(refreshed_documents, settings, document_store)

    if skipped_doc_ids:
        logger.info("batch_reindex job %s skipped documents: %s", job.get("job_id"), skipped_doc_ids)
    if failed_doc_ids:
        raise RuntimeError(f"batch_reindex partial failures: {failed_doc_ids}")


async def _execute_bm25_rebuild_job(job: dict, settings, document_store) -> None:
    """处理 BM25 重建 job。"""
    payload = job.get("payload") or {}
    collection_name = payload.get("collection_name")
    if not collection_name:
        raise RuntimeError("bm25_rebuild job payload missing collection_name")
    await rebuild_bm25_for_collection(collection_name, settings)


async def _execute_index_candidate_job(job: dict, settings, document_store) -> None:
    """Build a KB candidate and atomically activate automated rebuilds."""
    payload = job.get("payload") or {}
    kb_id = job.get("kb_id") or payload.get("kb_id")
    tenant_slug = payload.get("tenant_slug") or "default"
    if not kb_id:
        raise RuntimeError("index_candidate job missing kb_id")
    vector_store = get_vector_store_instance()
    profile_store = get_index_profile_store_instance()
    bm25_store = get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None
    await reconcile_knowledge_base_index(
        kb_id=kb_id,
        tenant_slug=tenant_slug,
        settings=settings,
        embedding_provider=get_embedding_provider_instance(),
        vector_store=vector_store,
        document_store=document_store,
        index_profile_store=profile_store,
        bm25_store=bm25_store,
        auto_activate=bool(payload.get("auto_activate")),
    )
    if payload.get("auto_activate"):
        pending_deletions = []
        if settings.retrieval.hybrid.enabled:
            pending_deletions = [
                item
                for item in await document_store.list_by_knowledge_base(kb_id)
                if item.get("status") in {"deleting", "delete_failed"}
            ]
        deleted_doc_ids = await finalize_pending_document_removals_after_cutover(
            kb_id=kb_id,
            settings=settings,
            vector_store=vector_store,
            document_store=document_store,
            index_profile_store=profile_store,
            bm25_store=bm25_store,
        )
        deleted_id_set = set(deleted_doc_ids)
        if settings.retrieval.hybrid.enabled and deleted_id_set:
            legacy_collections = {
                get_tenant_rag_collection_name(
                    settings.vectorstore.collection_prefix,
                    item.get("embedding_model") or settings.embedding.openai.model_name,
                    tenant_slug=item.get("tenant_slug"),
                    embedding_dimension=item.get("embedding_dimension"),
                )
                for item in pending_deletions
                if item.get("doc_id") in deleted_id_set
            }
            tenant_id = job.get("tenant_id") or ""
            for collection_name in legacy_collections:
                await _schedule_bm25_rebuild_after_delete(
                    tenant_id=tenant_id,
                    collection_name=collection_name,
                    settings=settings,
                )


async def _reindex_single_doc(
    doc_id: str,
    payload: dict,
    settings,
    document_store,
    *,
    defer_bm25_rebuild: bool = False,
) -> str | None:
    """对单篇文档执行重新索引。

    返回旧 collection 名称（若发生向量删除）。
    """
    save_path_value = payload.get("save_path")
    if not save_path_value:
        raise RuntimeError("reindex job payload missing save_path")

    save_path = Path(save_path_value)
    if not save_path.exists():
        await document_store.update_status_if(
            doc_id,
            ["reindex_queued", "reindexing"],
            "failed",
            error_message="上传原文件缺失，无法重新摄入。",
            chunks_count=0,
            status_reason="source_missing",
        )
        raise RuntimeError("reindex source file missing")

    current = await document_store.get(doc_id)
    if current and current.get("status") == "reindex_queued":
        await document_store.update_status_if(
            doc_id,
            ["reindex_queued"],
            "reindexing",
            error_message="",
            status_reason=current.get("status_reason") or "",
        )
    current = await document_store.get(doc_id)

    from_model = payload.get("from_embedding_model") or current.get("embedding_model") or settings.embedding.openai.model_name
    from_dimension = payload.get("from_embedding_dimension")
    to_model = payload.get("to_embedding_model") or settings.embedding.openai.model_name
    to_dimension = payload.get("to_embedding_dimension")
    tenant_slug = payload.get("tenant_slug") or current.get("tenant_slug")
    old_collection_name = get_tenant_rag_collection_name(
        settings.vectorstore.collection_prefix,
        from_model,
        tenant_slug=tenant_slug,
        embedding_dimension=from_dimension,
    )

    try:
        # Probe before deleting the old representation.  Manual reingest must
        # never turn a ready document into a failed document merely because
        # the replacement Embedding service is unavailable.
        embedding_provider = get_embedding_provider_instance()
        embedding_profile = await ensure_embedding_ready(
            settings,
            provider=embedding_provider,
        )
        vector_store = get_vector_store_instance()
        bm25_store = get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None
        await _delete_doc_vectors(
            vector_store,
            doc_id,
            old_collection_name,
            tenant_slug=tenant_slug,
            collection_prefix=settings.vectorstore.collection_prefix,
        )
        if not defer_bm25_rebuild and bm25_store and settings.retrieval.hybrid.enabled:
            bm25_store.invalidate_cache(old_collection_name)

        # 若队列创建时未获取到目标维度（如 embedding 服务短暂不可用），
        # 在 worker 执行时重新探测，避免使用文档旧维度。
        if to_dimension is None:
            to_dimension = embedding_profile.get("dimension")
        await document_store.update_embedding_model(doc_id, to_model, to_dimension)
        await ingest_document(
            save_path,
            doc_id,
            settings,
            embedding_provider,
            vector_store,
            document_store,
            bm25_store,
            defer_bm25_rebuild=defer_bm25_rebuild,
        )
        return old_collection_name
    except Exception as exc:
        await document_store.update_status_if(
            doc_id,
            ["reindex_queued", "reindexing"],
            "failed",
            error_message=sanitize_user_error_message(
                str(exc),
                "文档重新摄入失败，请检查模型配置后重试。",
            ),
            chunks_count=0,
            status_reason="reindex_failed",
        )
        raise


async def _delete_doc_vectors(
    vector_store,
    doc_id: str,
    collection_name: str,
    *,
    tenant_slug: str | None = None,
    collection_prefix: str = "rag",
) -> None:
    safe_delete = getattr(vector_store, "delete_by_doc_id_if_exists", None)
    if callable(safe_delete):
        await safe_delete(doc_id, collection_name=collection_name)
        return
    # Do not scan other collections.  A missing legacy collection must never
    # turn a reindex retry into a deletion from an active profile generation.
    await vector_store.delete_by_doc_id(doc_id, collection_name=collection_name)


async def process_next_ingest_job(
    worker_id: str = "local-worker",
    *,
    settings=None,
    max_attempts: Optional[int] = None,
    lock_timeout_seconds: Optional[int] = None,
) -> Optional[dict]:
    settings = settings or _get_settings()
    job_store = get_ingest_job_store_instance()
    job = await job_store.claim_next_job(
        worker_id,
        max_attempts=max_attempts or settings.queue.max_attempts,
        lock_timeout_seconds=lock_timeout_seconds or settings.queue.lock_timeout_seconds,
    )
    if not job:
        return None
    await execute_ingest_job(job, settings=settings)
    return await job_store.get(job["job_id"])


async def run_ingest_worker_forever(
    worker_id: str = "local-worker",
    *,
    settings=None,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    settings = settings or _get_settings()
    poll_interval = max(1, settings.queue.worker_poll_interval_seconds)
    heartbeat_task = asyncio.create_task(
        _heartbeat_ingest_worker_forever(worker_id, min(5.0, float(poll_interval))),
    )
    try:
        while True:
            if stop_event and stop_event.is_set():
                return

            try:
                job = await process_next_ingest_job(worker_id=worker_id, settings=settings)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("ingest worker loop crashed while polling")
                job = None

            if job is not None:
                continue

            if stop_event is None:
                await asyncio.sleep(poll_interval)
                continue

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                continue
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task
        _worker_heartbeats.pop(worker_id, None)
