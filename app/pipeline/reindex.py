"""Shared reindex queue helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.rag_scope import get_tenant_rag_collection_name


def embedding_configured(settings) -> bool:
    cfg = settings.embedding.openai
    return bool(cfg.api_base and cfg.model_name)


def auto_reindex_enabled(settings) -> bool:
    return (
        settings.app.enable_startup_recovery
        and settings.app.auto_reindex_on_embedding_change
        and settings.queue.backend == "db"
        and embedding_configured(settings)
    )


def document_upload_directory(upload_dir: str, doc: dict) -> Optional[Path]:
    """Resolve a document directory only when it stays under upload_dir."""
    upload_root = Path(upload_dir).resolve()
    doc_dir = (upload_root / str(doc.get("doc_id") or "")).resolve()
    try:
        doc_dir.relative_to(upload_root)
    except ValueError:
        return None
    return doc_dir


def find_document_file(upload_dir: str, doc: dict) -> Optional[Path]:
    doc_dir = document_upload_directory(upload_dir, doc)
    if doc_dir is None or not doc_dir.is_dir():
        return None

    filename = Path(str(doc.get("filename") or "")).name
    named = (doc_dir / filename).resolve()
    try:
        named.relative_to(doc_dir)
    except ValueError:
        named = None
    if named is not None and named.is_file():
        return named

    files = []
    for path in doc_dir.iterdir():
        resolved = path.resolve()
        try:
            resolved.relative_to(doc_dir)
        except ValueError:
            continue
        if resolved.is_file():
            files.append(resolved)
    files.sort()
    return files[0] if files else None


async def queue_batch_reindex(
    settings,
    document_store,
    ingest_job_store,
    docs: list[dict],
    *,
    tenant_id: str,
    trigger: str,
    embedding_provider=None,
    status_reason: str = "model_mismatch",
) -> dict:
    """把多份文档合并为单个 batch_reindex job。

    对每篇文档：
    - 检查源文件是否存在
    - 把状态迁移到 reindex_queued
    - 收集 doc_id 和 save_path

    最后创建一个 job_type='batch_reindex' 的 job。
    """
    from app.rag_scope import resolve_embedding_dimension

    to_model = settings.embedding.openai.model_name
    to_dimension = None
    if embedding_provider is not None:
        try:
            to_dimension = await resolve_embedding_dimension(embedding_provider)
        except Exception:
            # The worker probes again before writing vectors. A recovery job
            # must not turn an unavailable Embedding service into a startup
            # blocker merely to fill optional queue metadata.
            to_dimension = None

    doc_ids: list[str] = []
    save_paths: list[str] = []
    skipped_doc_ids: list[str] = []

    for doc in docs:
        save_path = find_document_file(settings.storage.upload_dir, doc)
        if not save_path:
            await document_store.update_status(
                doc["doc_id"],
                "failed",
                error_message="上传原文件缺失，无法重新摄入。",
                chunks_count=0,
                status_reason="source_missing",
            )
            skipped_doc_ids.append(doc["doc_id"])
            continue

        current_status = doc.get("status") or ""
        if current_status not in {"ready", "failed", "reindex_queued", "reindexing"}:
            skipped_doc_ids.append(doc["doc_id"])
            continue

        if current_status not in {"reindex_queued", "reindexing"}:
            result = await document_store.transition_status(
                doc["doc_id"],
                "reindex_queued",
                expected_statuses={current_status},
                error_message="",
                status_reason=status_reason,
            )
            if not result["ok"]:
                skipped_doc_ids.append(doc["doc_id"])
                continue

        doc_ids.append(doc["doc_id"])
        save_paths.append(str(save_path))

    if not doc_ids:
        return {
            "ok": False,
            "code": "no_documents_to_reindex",
            "message": "没有可重新索引的文档。",
            "document_ids": skipped_doc_ids,
        }

    tenant_slug = docs[0].get("tenant_slug") or "default"
    collection_name = get_tenant_rag_collection_name(
        settings.vectorstore.collection_prefix,
        to_model,
        tenant_slug=tenant_slug,
        embedding_dimension=to_dimension,
    )
    payload = {
        "doc_ids": doc_ids,
        "save_paths": save_paths,
        "trigger": trigger,
        "to_embedding_model": to_model,
        "to_embedding_dimension": to_dimension,
        "tenant_slug": tenant_slug,
        "collection_name": collection_name,
    }
    get_or_create_batch_job = getattr(
        ingest_job_store,
        "get_or_create_active_batch_reindex_job",
        None,
    )
    if callable(get_or_create_batch_job):
        job = await get_or_create_batch_job(
            tenant_id,
            doc_ids=doc_ids,
            payload=payload,
        )
    else:
        job = await ingest_job_store.create_job(
            tenant_id,
            "batch_reindex",
            payload=payload,
        )

    return {
        "ok": True,
        "job": job,
        "document_ids": doc_ids,
        "skipped_document_ids": skipped_doc_ids,
    }


async def queue_document_reindex(
    settings,
    document_store,
    ingest_job_store,
    doc: dict,
    *,
    tenant_id: str,
    trigger: str,
    embedding_provider=None,
) -> dict:
    from app.rag_scope import resolve_embedding_dimension

    save_path = find_document_file(settings.storage.upload_dir, doc)
    if not save_path:
        return {
            "ok": False,
            "code": "document_source_missing",
            "message": "上传原文件缺失，无法重新摄入。",
            "document": doc,
        }

    from_model = doc.get("embedding_model") or settings.embedding.openai.model_name
    from_dimension = doc.get("embedding_dimension")
    to_model = settings.embedding.openai.model_name
    to_dimension = None
    if embedding_provider is not None:
        try:
            to_dimension = await resolve_embedding_dimension(embedding_provider)
        except Exception:
            # Reindex jobs can be created while Embedding is temporarily
            # unavailable. The worker resolves the live dimension before it
            # writes vectors, so queueing must remain prompt and bounded.
            to_dimension = None
    current_status = doc.get("status") or ""
    current_reason = doc.get("status_reason") or ""
    next_reason = "model_mismatch" if (
        from_model != to_model or (to_dimension is not None and from_dimension != to_dimension)
    ) else ""

    active_job = await ingest_job_store.get_active_job_for_doc(doc["doc_id"], "reindex")
    next_status = "reindexing" if active_job and active_job.get("status") == "running" else "reindex_queued"

    if current_status not in {"ready", "failed", "reindex_queued", "reindexing"}:
        return {
            "ok": False,
            "code": "document_reingest_not_allowed",
            "message": f"文档状态 {current_status} 不允许重新摄入。",
            "document": doc,
        }

    if current_status != next_status:
        result = await document_store.transition_status(
            doc["doc_id"],
            next_status,
            expected_statuses={current_status},
            error_message="",
            status_reason=next_reason,
        )
        if not result["ok"]:
            current = result["document"] or doc
            return {
                "ok": False,
                "code": "document_status_conflict",
                "message": "状态变更失败，可能已被其他操作修改。",
                "document": current,
            }
        doc = result["document"]
    elif current_reason != next_reason:
        await document_store.update_status(
            doc["doc_id"],
            current_status,
            error_message=doc.get("error_message") or "",
            chunks_count=doc.get("chunks_count"),
            status_reason=next_reason,
        )
        doc = await document_store.get(doc["doc_id"]) or doc

    payload = {
        "save_path": str(save_path),
        "from_embedding_model": from_model,
        "from_embedding_dimension": from_dimension,
        "to_embedding_model": to_model,
        "to_embedding_dimension": to_dimension,
        "tenant_slug": doc.get("tenant_slug") or "",
        "collection_name": get_tenant_rag_collection_name(
            settings.vectorstore.collection_prefix,
            to_model,
            tenant_slug=doc.get("tenant_slug") or "",
            embedding_dimension=to_dimension,
        ),
        "trigger": trigger,
    }
    try:
        job = await ingest_job_store.get_or_create_active_doc_job(
            tenant_id,
            "reindex",
            doc_id=doc["doc_id"],
            payload=payload,
        )
    except Exception as exc:
        fallback_reason = next_reason or current_reason
        await document_store.update_status(
            doc["doc_id"],
            "failed",
            error_message=f"创建重新摄入任务失败: {exc}",
            chunks_count=0,
            status_reason=fallback_reason,
        )
        return {
            "ok": False,
            "code": "reingest_job_create_failed",
            "message": "创建重新摄入任务失败。",
            "document": await document_store.get(doc["doc_id"]) or doc,
        }

    refreshed = await document_store.get(doc["doc_id"]) or doc
    return {
        "ok": True,
        "job": job,
        "document": refreshed,
        "save_path": str(save_path),
    }
