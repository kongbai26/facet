"""文档列表/删除/重新摄入路由"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from app.api.authz import require_rag_read, require_rag_write
from app.api.deps import (
    _get_settings,
    get_bm25_store_instance,
    get_document_store_instance,
    get_embedding_provider_instance,
    get_ingest_job_store_instance,
    get_index_profile_store_instance,
    get_knowledge_base_store_instance,
    get_vector_store_instance,
    resolve_identity_tenant,
)
from app.api.errors import error_response
from app.pipeline.bm25_scheduler import schedule_bm25_rebuild
from app.pipeline.document_runtime_cleanup import cleanup_legacy_document_runtime
from app.pipeline.embedding_readiness import EmbeddingReadinessError, ensure_embedding_ready
from app.pipeline.reindex import queue_document_reindex
from app.utils.model_labels import display_model_name
from app.utils.user_errors import sanitize_diagnostic_detail

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


async def _resolve_job_tenant_id(identity: dict | None) -> str:
    tenant = await resolve_identity_tenant(identity)
    return tenant["tenant_id"]


async def _ensure_document_knowledge_base_active(document: dict, tenant_id: str) -> None:
    kb_id = document.get("kb_id")
    if not kb_id:
        return
    knowledge_base = await get_knowledge_base_store_instance().get(kb_id)
    if not knowledge_base or knowledge_base.get("tenant_id") != tenant_id:
        raise HTTPException(404, "Knowledge base not found")
    if knowledge_base.get("status") != "active":
        raise HTTPException(409, "Knowledge base is being deleted")


def _job_sort_key(job: dict) -> tuple[int, str, str]:
    status = job.get("status") or ""
    priority = 0 if status == "running" else 1
    return (priority, str(job.get("created_at") or ""), str(job.get("job_id") or ""))


def _public_document(document: dict) -> dict:
    """Hide internal model endpoints and path-like model identifiers from clients."""
    item = dict(document)
    if "embedding_model" in item:
        item["embedding_model"] = display_model_name(item.get("embedding_model"))
    item.pop("embedding_endpoint", None)
    if "error_message" in item:
        item["error_message"] = sanitize_diagnostic_detail(item.get("error_message"))
    return item


def _collect_active_jobs(jobs: list[dict]) -> tuple[dict[str, dict], list[dict], list[dict]]:
    doc_jobs: dict[str, dict] = {}
    bm25_jobs: list[dict] = []
    knowledge_base_jobs: list[dict] = []
    for job in sorted(jobs, key=_job_sort_key):
        job_type = job.get("job_type")
        if job_type == "bm25_rebuild":
            bm25_jobs.append(job)
            continue
        if job_type == "knowledge_base_delete":
            knowledge_base_jobs.append(job)
            continue

        # 普通索引重建没有 doc_id；如果不单独放进队列，它会在排队和
        # 运行阶段完全不可见，前端只能在历史记录里看到结果。
        if job_type == "index_candidate" and not _job_doc_ids(job):
            knowledge_base_jobs.append(job)
            continue

        for doc_id in _job_doc_ids(job):
            if doc_id not in doc_jobs:
                doc_jobs[doc_id] = job
    return doc_jobs, bm25_jobs, knowledge_base_jobs


def _job_doc_ids(job: dict) -> list[str]:
    doc_ids: list[str] = []
    if job.get("doc_id"):
        doc_ids.append(job["doc_id"])
    payload = job.get("payload") or {}
    if payload.get("doc_id"):
        doc_ids.append(payload["doc_id"])
    doc_ids.extend(doc_id for doc_id in payload.get("doc_ids") or [] if doc_id)
    doc_ids.extend(doc_id for doc_id in payload.get("delete_doc_ids") or [] if doc_id)
    return list(dict.fromkeys(doc_ids))


def _job_related_documents(job: dict, doc_lookup: dict[str, dict]) -> list[str]:
    payload = job.get("payload") or {}
    save_paths = [Path(str(path)) for path in payload.get("save_paths") or []]
    related_documents: list[str] = []

    for index, doc_id in enumerate(_job_doc_ids(job)):
        doc = doc_lookup.get(doc_id) or {}
        deletion_names = payload.get("delete_document_names") or {}
        title = doc.get("filename") or deletion_names.get(str(doc_id)) or ""
        if not title and index < len(save_paths):
            title = save_paths[index].name
        if not title and payload.get("save_path"):
            title = Path(str(payload["save_path"])).name
        related_documents.append(title or doc_id)

    if not related_documents and payload.get("save_path"):
        related_documents.append(Path(str(payload["save_path"])).name)

    return list(dict.fromkeys(related_documents))


def _job_summary(job: dict, related_documents: list[str]) -> str:
    job_type = job.get("job_type") or ""
    payload = job.get("payload") or {}

    if job_type == "knowledge_base_delete":
        name = payload.get("knowledge_base_name") or payload.get("kb_id") or job.get("kb_id") or "当前知识库"
        return f"知识库：{name}"

    if job_type == "index_candidate" and payload.get("reason") == "document_delete":
        return f"文档：{payload.get('document_name') or payload.get('doc_id') or '待删除文档'}"

    if job_type == "index_candidate":
        return "知识库索引自动重建。"

    if job_type == "bm25_rebuild":
        collection_name = payload.get("collection_name")
        if collection_name:
            return f"集合：{collection_name}"
        return "正在生成 BM25 索引。"

    if related_documents:
        if len(related_documents) == 1:
            return f"文档：{related_documents[0]}"
        head = "、".join(related_documents[:3])
        tail = " 等" if len(related_documents) > 3 else ""
        return f"包含 {len(related_documents)} 个文档：{head}{tail}"

    if payload.get("save_path"):
        return f"文件：{Path(str(payload['save_path'])).name}"
    if payload.get("collection_name"):
        return f"集合：{payload['collection_name']}"
    return sanitize_diagnostic_detail(job.get("error_message"), "任务处理中。")


def _job_detail(job: dict, related_documents: list[str]) -> str:
    job_type = job.get("job_type") or ""
    status = job.get("status") or ""

    if status == "running":
        if job_type == "index_candidate" and (job.get("payload") or {}).get("reason") == "document_delete":
            return "正在重新构建不含该文档的知识库索引；切换成功后会自动完成删除。"
        if job_type == "index_candidate":
            return "正在构建并校验新的知识库索引；完成后会自动安全切换。"
        if job_type == "knowledge_base_delete":
            return "知识库正在清理文档、索引、向量、缓存和关联元数据。"
        if job_type == "bm25_rebuild":
            return "BM25 索引正在后台建立。"
        if job_type in {"reindex", "batch_reindex"}:
            return "文档正在重新摄入并生成新索引。"
        return "文档正在后台处理。"

    if status == "queued":
        if job_type == "index_candidate" and (job.get("payload") or {}).get("reason") == "document_delete":
            return "文档删除正在等待重新构建知识库索引。"
        if job_type == "index_candidate":
            return "知识库索引重建已排队；当前检索继续使用上一个可用索引。"
        if job_type == "knowledge_base_delete":
            return "知识库删除任务正在排队等待清理。"
        if job_type == "bm25_rebuild":
            return "BM25 索引正在排队等待生成。"
        if job_type in {"reindex", "batch_reindex"}:
            return "文档正在排队等待重新摄入。"
        return "文档正在排队等待处理。"

    if status == "succeeded":
        if job_type == "index_candidate" and (job.get("payload") or {}).get("reason") == "document_delete":
            return "文档已从活动索引和运行时存储中删除。"
        if job_type == "index_candidate":
            return "知识库索引已校验并自动切换完成。"
        if job_type == "knowledge_base_delete":
            return "知识库及其运行时数据已清理完成。"
        if job_type == "bm25_rebuild":
            return "BM25 索引已完成，可用于检索。"
        if job_type in {"reindex", "batch_reindex"}:
            return "文档已重新摄入完成。"
        return "文档已完成处理。"

    if status == "cancelled":
        return sanitize_diagnostic_detail(job.get("error_message"), "任务已取消。")

    if status == "failed":
        return sanitize_diagnostic_detail(job.get("error_message"), "后台任务处理失败。")

    if related_documents:
        return f"涉及文档：{'、'.join(related_documents[:3])}"
    return sanitize_diagnostic_detail(job.get("error_message"), "任务详情不可用。")


def _job_label(job: dict, related_documents: list[str]) -> str:
    job_type = job.get("job_type") or ""
    status = job.get("status") or ""

    if status == "running":
        if job_type == "index_candidate" and (job.get("payload") or {}).get("reason") == "document_delete":
            return "重建索引后删除"
        if job_type == "index_candidate":
            return "索引重建中"
        if job_type == "knowledge_base_delete":
            return "知识库删除中"
        if job_type == "bm25_rebuild":
            return "BM25 索引建立中"
        if job_type in {"reindex", "batch_reindex"}:
            return "重建中"
        if job_type == "batch_ingest":
            return "批量上传中"
        return "上传中"
    if status == "queued":
        if job_type == "index_candidate" and (job.get("payload") or {}).get("reason") == "document_delete":
            return "等待重建索引"
        if job_type == "index_candidate":
            return "等待索引重建"
        if job_type == "knowledge_base_delete":
            return "知识库删除排队中"
        if job_type == "bm25_rebuild":
            return "BM25 索引排队中"
        if job_type in {"reindex", "batch_reindex"}:
            return "等待重建"
        if job_type == "batch_ingest":
            return "批量上传排队中"
        return "上传排队中"
    if status == "succeeded":
        if job_type == "index_candidate" and (job.get("payload") or {}).get("reason") == "document_delete":
            return "文档删除完成"
        if job_type == "index_candidate":
            return "索引重建完成"
        if job_type == "knowledge_base_delete":
            return "知识库删除完成"
        if job_type == "bm25_rebuild":
            return "BM25 索引完成"
        if job_type in {"reindex", "batch_reindex"}:
            return "重建完成"
        return "上传完成"
    if status == "cancelled":
        return "已取消"
    if status == "failed":
        if job_type == "knowledge_base_delete":
            return "知识库删除失败"
        if job_type == "index_candidate" and (job.get("payload") or {}).get("reason") == "document_delete":
            return "文档删除失败"
        if job_type == "index_candidate":
            return "索引重建失败"
        return "失败"
    return "任务"


def _queue_item_base(
    job: dict,
    *,
    kind: str,
    title: str,
    related_documents: list[str],
    detail: str,
    origin: str,
) -> dict:
    status = job.get("status") or "queued"
    if status not in {"queued", "running", "failed", "succeeded", "cancelled"}:
        status = "queued"
    return {
        "kind": kind,
        "status": status,
        "label": _job_label(job, related_documents),
        "detail": detail,
        "title": title,
        "job_type": job.get("job_type"),
        "reason": (job.get("payload") or {}).get("reason"),
        "job_id": job.get("job_id"),
        "kb_id": job.get("kb_id") or (job.get("payload") or {}).get("kb_id"),
        "doc_id": (
            job.get("doc_id")
            or (job.get("payload") or {}).get("doc_id")
            or next(iter((job.get("payload") or {}).get("delete_doc_ids") or []), None)
        ),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "related_documents": related_documents,
        "summary": _job_summary(job, related_documents),
        "origin": origin,
    }


def _document_queue_item(doc: dict, active_job: dict | None = None, doc_lookup: dict[str, dict] | None = None) -> dict:
    status = doc.get("status") or "processing"
    job_type = (active_job or {}).get("job_type")
    job_status = (active_job or {}).get("status")
    doc_lookup = doc_lookup or {}
    related_documents = _job_related_documents(active_job, doc_lookup) if active_job else [doc.get("filename") or doc["doc_id"]]

    if status == "processing":
        if job_type == "batch_ingest":
            label = "批量上传中" if job_status == "running" else "批量上传排队中"
            detail = "文档正在通过批量上传流程写入系统。"
            state = job_status or "running"
        elif job_type == "ingest":
            label = "上传中" if job_status == "running" else "上传排队中"
            detail = "文档正在进入后台摄入流程。"
            state = job_status or "running"
        else:
            label = "处理中"
            detail = "文档正在后台处理。"
            state = "running"
    elif status == "reindex_queued":
        label = "等待重建"
        detail = "检测到模型变化，正在等待后台重建索引。"
        state = "queued"
    elif status == "reindexing":
        label = "重建中"
        detail = "正在使用当前 embedding 重新建立索引。"
        state = "running"
    elif status == "deleting":
        label = "删除中"
        detail = "正在清理向量、文件和元数据。"
        state = "running"
    elif status == "delete_failed":
        label = "删除失败"
        detail = sanitize_diagnostic_detail(doc.get("error_message"), "删除过程中发生了错误。")
        state = "failed"
    else:
        label = status
        detail = sanitize_diagnostic_detail(doc.get("error_message"))
        state = "queued"

    return {
        "key": f"doc:{doc['doc_id']}",
        "kind": "document",
        "status": state,
        "label": label,
        "detail": detail,
        "title": doc.get("filename") or doc["doc_id"],
        "origin": "active",
        "doc_id": doc["doc_id"],
        "document_status": status,
        "job_type": job_type,
        "job_id": (active_job or {}).get("job_id"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "related_documents": related_documents,
        "summary": _job_summary(active_job, related_documents) if active_job else (doc.get("status_reason") or ""),
    }


def _bm25_queue_item(job: dict, ready_count: int) -> dict:
    detail = f"当前文档库正在后台生成 BM25 索引，已有 {ready_count} 个就绪文档。"
    item = _queue_item_base(
        job,
        kind="bm25",
        title="BM25 索引",
        related_documents=[],
        detail=detail,
        origin="active",
    )
    item["key"] = f"bm25:{job.get('payload', {}).get('collection_name') or job.get('job_id')}"
    item["collection_name"] = job.get("payload", {}).get("collection_name")
    return item


def _pending_bm25_queue_item(active_doc_count: int, ready_count: int) -> dict:
    detail = "文档处理完成后将自动建立 BM25 索引。"
    if ready_count > 0:
        detail = f"当前有 {ready_count} 个就绪文档，文档处理完成后将自动建立 BM25 索引。"
    return {
        "key": "bm25:pending",
        "kind": "bm25",
        "status": "queued",
        "label": "BM25 索引排队中",
        "detail": detail,
        "title": "BM25 索引",
        "job_type": "bm25_rebuild",
        "job_id": None,
        "created_at": None,
        "updated_at": None,
        "collection_name": None,
        "related_documents": [],
        "summary": f"当前有 {active_doc_count} 个文档任务在处理，BM25 将在文档完成后刷新。",
        "origin": "active",
    }


def _has_pending_bm25_refresh(active_docs: list[dict], active_doc_jobs: dict[str, dict]) -> bool:
    for doc in active_docs:
        if doc.get("status") in {"processing", "reindex_queued", "reindexing"}:
            return True

    for job in active_doc_jobs.values():
        if (job.get("job_type") or "") in {"ingest", "batch_ingest", "reindex", "batch_reindex"}:
            return True

    return False


def _history_queue_item(job: dict, doc_lookup: dict[str, dict]) -> dict:
    related_documents = _job_related_documents(job, doc_lookup)
    payload = job.get("payload") or {}
    if job.get("job_type") == "knowledge_base_delete":
        title = payload.get("knowledge_base_name") or payload.get("kb_id") or job.get("kb_id") or "知识库"
    elif job.get("job_type") == "index_candidate" and payload.get("reason") == "document_delete":
        title = payload.get("document_name") or payload.get("doc_id") or "文档删除"
    elif job.get("job_type") == "index_candidate":
        title = "知识库候选索引重建"
    elif job.get("job_type") == "bm25_rebuild":
        # collection_name 是内部物理名称，可能很长且对用户没有意义。
        title = "BM25 索引"
    else:
        title = related_documents[0] if related_documents else (payload.get("collection_name") or job.get("job_id") or "历史任务")
    item = _queue_item_base(
        job,
        kind=("bm25" if job.get("job_type") == "bm25_rebuild" else "knowledge_base" if job.get("job_type") == "knowledge_base_delete" else "document"),
        title=title,
        related_documents=related_documents,
        detail=_job_detail(job, related_documents),
        origin="history",
    )
    item["key"] = f"history:{job.get('job_id')}"
    item["collection_name"] = job.get("payload", {}).get("collection_name")
    return item


def _knowledge_base_queue_item(job: dict, knowledge_base_lookup: dict[str, dict]) -> dict:
    payload = job.get("payload") or {}
    kb_id = job.get("kb_id") or payload.get("kb_id")
    knowledge_base = knowledge_base_lookup.get(kb_id or "") or {}
    title = payload.get("knowledge_base_name") or knowledge_base.get("name") or kb_id or "知识库"
    item = _queue_item_base(
        job,
        kind="knowledge_base",
        title=str(title),
        related_documents=[],
        detail=_job_detail(job, []),
        origin="active",
    )
    item["key"] = f"knowledge-base:{job.get('job_id')}"
    return item


def _failed_knowledge_base_lifecycle_item(knowledge_base: dict) -> dict:
    return {
        "key": f"knowledge-base-lifecycle:{knowledge_base['kb_id']}",
        "kind": "knowledge_base",
        "status": "failed",
        "label": "知识库删除失败",
        "detail": sanitize_diagnostic_detail(
            knowledge_base.get("error_message"),
            "知识库删除未完成，请重试。",
        ),
        "title": knowledge_base.get("name") or knowledge_base["kb_id"],
        "job_type": "knowledge_base_delete",
        "job_id": None,
        "kb_id": knowledge_base["kb_id"],
        "created_at": knowledge_base.get("created_at"),
        "updated_at": knowledge_base.get("updated_at"),
        "related_documents": [],
        "summary": f"知识库：{knowledge_base.get('name') or knowledge_base['kb_id']}",
        "origin": "active",
    }


@router.get("")
async def list_documents(
    kb_id: str | None = Query(default=None),
    identity: dict = Depends(require_rag_read),
):
    """List documents in the selected knowledge base, or all documents when omitted."""
    document_store = get_document_store_instance()
    tenant = await resolve_identity_tenant(identity)
    if kb_id:
        knowledge_base = await get_knowledge_base_store_instance().get(kb_id)
        if (
            not knowledge_base
            or knowledge_base.get("tenant_id") != tenant["tenant_id"]
            or knowledge_base.get("status") != "active"
        ):
            raise HTTPException(404, "Knowledge base not found")
        docs = await document_store.list_by_knowledge_base(kb_id, tenant_id=tenant["tenant_id"])
    else:
        docs = await document_store.list_all(tenant_id=tenant["tenant_id"])
    return {"documents": [_public_document(doc) for doc in docs]}


@router.get("/revision")
async def document_revision(identity: dict = Depends(require_rag_read)):
    """Lightweight revision used to notice Agent-side document mutations."""
    tenant = await resolve_identity_tenant(identity)
    return await get_document_store_instance().tenant_revision(tenant["tenant_id"])


@router.get("/queue")
async def get_document_queue(identity: dict = Depends(require_rag_read)):
    """返回文档队列视图。"""
    settings = _get_settings()
    document_store = get_document_store_instance()
    job_store = get_ingest_job_store_instance()
    tenant = await resolve_identity_tenant(identity)
    tenant_id = tenant["tenant_id"]

    docs = await document_store.list_all(tenant_id=tenant_id)
    doc_lookup = {doc["doc_id"]: doc for doc in docs}
    ready_count = sum(1 for doc in docs if doc.get("status") == "ready")
    active_docs = [
        doc
        for doc in docs
        if doc.get("status") in {"processing", "reindex_queued", "reindexing", "deleting"}
    ]

    queued_jobs = await job_store.list_jobs(status="queued", tenant_id=tenant_id)
    running_jobs = await job_store.list_jobs(status="running", tenant_id=tenant_id)
    active_jobs = queued_jobs + running_jobs
    active_doc_jobs, active_bm25_jobs, active_knowledge_base_jobs = _collect_active_jobs(active_jobs)
    pending_bm25_refresh = _has_pending_bm25_refresh(active_docs, active_doc_jobs)

    items = [
        _document_queue_item(doc, active_doc_jobs.get(doc["doc_id"]), doc_lookup)
        for doc in sorted(active_docs, key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
    ]

    tenant_knowledge_bases = await get_knowledge_base_store_instance().list_by_tenant(tenant_id)
    knowledge_base_lookup = {
        item["kb_id"]: item
        for item in tenant_knowledge_bases
    }
    items.extend(
        _knowledge_base_queue_item(job, knowledge_base_lookup)
        for job in active_knowledge_base_jobs
    )
    active_delete_kb_ids = {
        str(job.get("kb_id") or (job.get("payload") or {}).get("kb_id"))
        for job in active_knowledge_base_jobs
    }
    items.extend(
        _failed_knowledge_base_lifecycle_item(knowledge_base)
        for knowledge_base in tenant_knowledge_bases
        if knowledge_base.get("status") == "delete_failed"
        and knowledge_base["kb_id"] not in active_delete_kb_ids
    )

    if settings.retrieval.hybrid.enabled:
        for job in active_bm25_jobs:
            items.append(_bm25_queue_item(job, ready_count))
        if not active_bm25_jobs and pending_bm25_refresh:
            items.append(_pending_bm25_queue_item(len(active_docs), ready_count))

    items.sort(key=lambda item: (0 if item["kind"] == "document" and item["status"] == "running" else 1, str(item.get("updated_at") or item.get("created_at") or ""), item["title"]))

    history_jobs: list[dict] = []
    for status in ("succeeded", "failed", "cancelled"):
        history_jobs.extend(await job_store.list_jobs(status=status, limit=10, tenant_id=tenant_id))
    history_jobs.sort(key=lambda job: (str(job.get("updated_at") or job.get("created_at") or ""), str(job.get("job_id") or "")), reverse=True)
    history_items = [_history_queue_item(job, doc_lookup) for job in history_jobs[:10]]

    return {
        "items": items,
        "history": history_items,
        "counts": {
            "documents": len(active_docs),
            "bm25": (len(active_bm25_jobs) + (1 if settings.retrieval.hybrid.enabled and not active_bm25_jobs and pending_bm25_refresh else 0)),
            "active": len(items),
            "history": len(history_items),
        },
    }


@router.post("/queue/history/clear")
async def clear_document_queue_history(identity: dict = Depends(require_rag_write)):
    """清空文档队列的历史记录。"""
    job_store = get_ingest_job_store_instance()
    tenant = await resolve_identity_tenant(identity)
    deleted = await job_store.clear_history(tenant["tenant_id"])
    return {"deleted": deleted}


@router.get("/{doc_id}")
async def get_document(doc_id: str, identity: dict = Depends(require_rag_read)):
    """获取文档详情。"""
    document_store = get_document_store_instance()
    tenant = await resolve_identity_tenant(identity)
    doc = await document_store.get(doc_id, tenant_id=tenant["tenant_id"])
    if not doc:
        raise HTTPException(404, "Document not found")
    await _ensure_document_knowledge_base_active(doc, tenant["tenant_id"])
    return _public_document(doc)


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    identity: dict = Depends(require_rag_write),
):
    """删除文档及其向量。BM25 重建异步调度，不阻塞接口返回。"""
    settings = _get_settings()
    document_store = get_document_store_instance()
    vector_store = get_vector_store_instance()
    tenant = await resolve_identity_tenant(identity)

    doc = await document_store.get(doc_id, tenant_id=tenant["tenant_id"])
    if not doc:
        raise HTTPException(404, "Document not found")
    await _ensure_document_knowledge_base_active(doc, tenant["tenant_id"])

    if doc["status"] in {"processing", "reindexing"}:
        return error_response(
            409,
            "document_processing",
            "文档正在处理中，暂不能删除。",
            doc_id=doc_id,
            status=doc["status"],
        )

    job_store = get_ingest_job_store_instance()
    prepare_deletion = getattr(job_store, "prepare_document_deletion", None)
    prepared_deletion = None
    if callable(prepare_deletion):
        try:
            prepared_deletion = await prepare_deletion(
                tenant["tenant_id"],
                doc_id=doc_id,
                candidate_payload={
                    "kb_id": doc.get("kb_id"),
                    "tenant_slug": doc.get("tenant_slug") or tenant.get("slug") or "default",
                },
            )
            doc = prepared_deletion["document"]
        except KeyError as exc:
            raise HTTPException(404, "Document not found") from exc
        except ValueError as exc:
            if str(exc) == "knowledge_base_not_active":
                raise HTTPException(409, "Knowledge base is being deleted") from exc
            return error_response(
                409,
                "document_processing" if str(exc) == "document_source_job_running" else "document_status_conflict",
                "文档仍有写任务在运行，暂不能删除。"
                if str(exc) == "document_source_job_running"
                else "文档当前状态暂不能删除。",
                doc_id=doc_id,
                status=doc.get("status"),
            )
    else:
        # Compatibility path for lightweight integrations. Production uses
        # the cross-table transaction above.
        result = await document_store.transition_status(
            doc_id,
            "deleting",
            expected_statuses={"ready", "failed", "delete_failed", "reindex_queued", "deleting"},
        )
        if not result["ok"]:
            current = result["document"]
            if not current:
                raise HTTPException(404, "Document not found")
            return error_response(
                409,
                "document_status_conflict",
                "文档当前状态暂不能删除。",
                doc_id=doc_id,
                status=current["status"],
            )
        try:
            await job_store.cancel_jobs_for_doc(doc_id)
        except Exception as e:
            logger.warning(f"取消文档 {doc_id} 关联 job 失败: {e}")

    tenant_slug = doc.get("tenant_slug")
    index_profile_store = get_index_profile_store_instance()
    active_index = (
        {"status": "active"}
        if prepared_deletion and prepared_deletion["requires_candidate"]
        else await index_profile_store.get_active_index(doc.get("kb_id") or "")
        if prepared_deletion is None and doc.get("kb_id")
        else None
    )
    bm25_store = get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None
    if active_index is not None:
        # A deletion changes the same KB snapshot as an upload or model
        # rebuild.  Queue it through the single KB-scoped candidate worker;
        # that worker activates the replacement then performs physical cleanup.
        job = prepared_deletion["job"] if prepared_deletion else await job_store.get_or_create_active_kb_job(
            tenant["tenant_id"],
            "index_candidate",
            kb_id=doc["kb_id"],
            payload={
                "kb_id": doc["kb_id"],
                "doc_id": doc_id,
                "document_name": doc.get("filename") or doc_id,
                "tenant_slug": tenant_slug or "default",
                "auto_activate": True,
                "reason": "document_delete",
            },
        )
        return JSONResponse(
            status_code=202,
            content={
                "status": "deleting",
                "job_id": job["job_id"],
                "message": "正在重新构建知识库索引；新索引切换成功后将自动完成删除。",
            },
        )

    # Profile generations are immutable. With no active profile, only the
    # shared legacy runtime representation can be removed directly.
    cleanup = await cleanup_legacy_document_runtime(
        document=doc,
        settings=settings,
        vector_store=vector_store,
        bm25_store=bm25_store,
        compact_empty_collection=True,
    )
    collection_name = cleanup.collection_name
    errors = list(cleanup.errors)

    if errors:
        error_message = "; ".join(errors)
        await document_store.transition_status(
            doc_id,
            "delete_failed",
            expected_statuses={"deleting"},
            error_message=error_message,
        )
        return error_response(
            500,
            "document_delete_failed",
            "删除文档时出现错误。",
            status="delete_failed",
            errors=[sanitize_diagnostic_detail(error, "删除文档时出现错误。") for error in errors],
        )

    await index_profile_store.retire_document_states(doc_id)

    try:
        await document_store.delete(doc_id)
    except Exception as e:
        error_message = f"元数据删除失败: {e}"
        logger.warning(f"元数据删除失败 {doc_id}: {e}")
        await document_store.transition_status(
            doc_id,
            "delete_failed",
            expected_statuses={"deleting"},
            error_message=error_message,
        )
        return error_response(
            500,
            "document_metadata_delete_failed",
            "删除文档元数据失败。",
            status="delete_failed",
            errors=[sanitize_diagnostic_detail(error_message, "删除文档元数据失败。")],
        )

    if settings.retrieval.hybrid.enabled and active_index is None:
        await schedule_bm25_rebuild(
            collection_name,
            tenant["tenant_id"],
            settings,
            background_tasks=background_tasks,
        )

    return {"status": "deleted"}


@router.post("/{doc_id}/reingest")
async def reingest_document(
    doc_id: str,
    identity: dict = Depends(require_rag_write),
):
    """重新摄入文档（重建向量索引）。"""
    settings = _get_settings()
    document_store = get_document_store_instance()
    tenant = await resolve_identity_tenant(identity)

    # 1. 检查文档存在
    doc = await document_store.get(doc_id, tenant_id=tenant["tenant_id"])
    if not doc:
        raise HTTPException(404, "Document not found")
    await _ensure_document_knowledge_base_active(doc, tenant["tenant_id"])

    # 2. 检查非 processing 状态
    if doc["status"] == "processing":
        return error_response(
            409,
            "document_processing",
            "文档正在处理中。",
            doc_id=doc_id,
            status="processing",
        )

    if doc["status"] not in ("failed", "ready", "reindex_queued", "reindexing"):
        return error_response(
            409,
            "document_reingest_not_allowed",
            f"文档状态 {doc['status']} 不允许重新摄入。",
            doc_id=doc_id,
            status=doc["status"],
        )

    if settings.app.env == "production":
        try:
            await ensure_embedding_ready(
                settings,
                provider=get_embedding_provider_instance(),
            )
        except EmbeddingReadinessError as exc:
            return error_response(
                503,
                "embedding_unavailable",
                str(exc),
                doc_id=doc_id,
                status=doc["status"],
            )

    tenant_id = await _resolve_job_tenant_id(identity)
    active_index = (
        await get_index_profile_store_instance().get_active_index(doc["kb_id"])
        if doc.get("kb_id") else None
    )
    if active_index is not None and doc["status"] == "ready":
        # Reingesting a source represented by an active immutable generation
        # must not delete-and-rewrite a live collection. Advance the explicit
        # source revision first, then use the same serialized KB candidate
        # path as a model or chunking-profile change.
        await document_store.bump_source_revision(doc_id)
        job = await get_ingest_job_store_instance().get_or_create_active_kb_job(
            tenant_id,
            "index_candidate",
            kb_id=doc["kb_id"],
            payload={
                "kb_id": doc["kb_id"],
                "tenant_slug": doc.get("tenant_slug") or tenant.get("slug") or "default",
                "auto_activate": True,
                "reason": "manual_reingest",
            },
        )
        return JSONResponse(
            status_code=202,
            content={
                "doc_id": doc_id,
                "filename": doc["filename"],
                "status": doc["status"],
                "job_id": job["job_id"],
                "message": "已加入知识库候选索引重建；切换完成前线上检索保持不变。",
            },
        )

    # A failed (or already queued) source cannot be repaired by a candidate
    # build: candidate generations intentionally include ready documents only.
    # Rebuild the source representation first; its worker subsequently queues
    # the KB-scoped candidate cutover without touching the live generation.
    result = await queue_document_reindex(
        settings,
        document_store,
        get_ingest_job_store_instance(),
        doc,
        tenant_id=tenant_id,
        trigger="manual",
    )
    if not result["ok"]:
        current = result.get("document") or doc
        code = result["code"]
        status_code = 400 if code == "document_source_missing" else 409 if code == "document_status_conflict" else 500
        if code == "document_reingest_not_allowed":
            status_code = 409
        return error_response(
            status_code,
            code,
            result["message"],
            doc_id=doc_id,
            status=current.get("status"),
        )

    current = result["document"]
    return JSONResponse(
        status_code=202,
        content={
            "doc_id": doc_id,
            "filename": current["filename"],
            "status": current["status"],
            "job_id": result["job"]["job_id"],
            "message": "已加入重新摄入队列",
        },
    )
