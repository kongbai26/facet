"""Knowledge-base selection and creation APIs."""

from __future__ import annotations

import re
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.authz import require_rag_read, require_rag_write
from app.api.deps import (
    _get_settings,
    get_bm25_store_instance,
    get_document_store_instance,
    get_index_profile_store_instance,
    get_ingest_job_store_instance,
    get_knowledge_base_store_instance,
    get_vector_store_instance,
    resolve_identity_tenant,
)
from app.pipeline.chat_flow import validate_full_context_document
from app.pipeline.index_lifecycle import (
    finalize_knowledge_base_index_activation,
    reclaim_inactive_knowledge_base_index,
)
from app.utils.model_labels import display_model_name
from app.utils.user_errors import sanitize_diagnostic_detail

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"])


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


def _slug_for_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", name.strip().lower()).strip("-")
    return (normalized[:40] or "knowledge-base") + "-" + uuid4().hex[:8]


async def _get_active_knowledge_base(kb_id: str, tenant_id: str) -> dict:
    """Resolve one active KB inside the caller's tenant for mutating flows."""
    knowledge_base = await get_knowledge_base_store_instance().get(kb_id)
    if (
        not knowledge_base
        or knowledge_base.get("tenant_id") != tenant_id
        or knowledge_base.get("status") != "active"
    ):
        raise HTTPException(404, "Knowledge base not found")
    return knowledge_base


@router.get("")
async def list_knowledge_bases(identity: dict = Depends(require_rag_read)):
    tenant = await resolve_identity_tenant(identity)
    # Inactive KBs are not valid retrieval targets.  Do not advertise them in
    # the Agent/UI discovery response as selectable resources.
    knowledge_bases = [
        item
        for item in await get_knowledge_base_store_instance().list_by_tenant(tenant["tenant_id"])
        if item.get("status") == "active"
    ]
    document_store = get_document_store_instance()
    response = []
    for knowledge_base in knowledge_bases:
        documents = await document_store.list_by_knowledge_base(
            knowledge_base["kb_id"], tenant_id=tenant["tenant_id"], status="ready",
        )
        response.append({
            **knowledge_base,
            "embedding_model": display_model_name(knowledge_base.get("embedding_model")),
            "llm_model": display_model_name(knowledge_base.get("llm_model")),
            "ready_documents_count": len(documents),
        })
    return {"knowledge_bases": response}


@router.get("/revision")
async def knowledge_base_revision(identity: dict = Depends(require_rag_read)):
    """Lightweight revision used to notice Agent-side KB mutations."""
    tenant = await resolve_identity_tenant(identity)
    return await get_knowledge_base_store_instance().tenant_revision(tenant["tenant_id"])


@router.get("/{kb_id}/documents")
async def list_knowledge_base_documents(kb_id: str, identity: dict = Depends(require_rag_read)):
    tenant = await resolve_identity_tenant(identity)
    await _get_active_knowledge_base(kb_id, tenant["tenant_id"])
    documents = await get_document_store_instance().list_by_knowledge_base(
        kb_id, tenant_id=tenant["tenant_id"], status="ready",
    )
    settings = _get_settings()
    active_index = await get_index_profile_store_instance().get_active_index(kb_id)
    active_parent_profile = active_index["index_id"] if active_index else "legacy"
    annotated_documents = []
    for document in documents:
        item = dict(document)
        item["embedding_model"] = display_model_name(item.get("embedding_model"))
        item.pop("embedding_endpoint", None)
        try:
            _text, _parents, token_count, token_budget = await validate_full_context_document(
                settings, item["doc_id"], profile_hash=active_parent_profile,
            )
            item.update({
                "full_context_available": True,
                "full_context_tokens": token_count,
                "full_context_token_budget": token_budget,
            })
        except ValueError as exc:
            item.update({
                "full_context_available": False,
                "full_context_reason": sanitize_diagnostic_detail(str(exc), "当前文档无法使用全文模式。"),
            })
        annotated_documents.append(item)
    return {"documents": annotated_documents}


@router.post("")
async def create_knowledge_base(
    req: KnowledgeBaseCreateRequest,
    identity: dict = Depends(require_rag_write),
):
    tenant = await resolve_identity_tenant(identity)
    settings = _get_settings()
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Knowledge base name is required")
    slug = _slug_for_name(name)
    try:
        knowledge_base = await get_knowledge_base_store_instance().create_with_namespace(
            tenant["tenant_id"],
            namespace_slug=f"kb-{slug}",
            kb_slug=slug,
            name=name,
            embedding_model=settings.embedding.openai.model_name,
            llm_model=settings.llm.model_name,
            max_namespaces=(settings.quota.default_max_namespaces if settings.quota.enabled else None),
        )
    except ValueError as exc:
        if str(exc) == "namespace_quota_exceeded":
            raise HTTPException(status_code=429, detail="Namespace quota exceeded") from exc
        if str(exc) == "knowledge_base_name_conflict":
            raise HTTPException(status_code=409, detail="知识库名称已存在") from exc
        raise
    return {
        **knowledge_base,
        "embedding_model": display_model_name(knowledge_base.get("embedding_model")),
        "llm_model": display_model_name(knowledge_base.get("llm_model")),
        "ready_documents_count": 0,
    }


@router.delete("/{kb_id}", status_code=202)
async def delete_knowledge_base(
    kb_id: str,
    identity: dict = Depends(require_rag_write),
):
    """Queue a durable deletion of one tenant knowledge base."""
    tenant = await resolve_identity_tenant(identity)
    knowledge_base_store = get_knowledge_base_store_instance()
    knowledge_base = await knowledge_base_store.get(kb_id)
    if not knowledge_base or knowledge_base.get("tenant_id") != tenant["tenant_id"]:
        raise HTTPException(404, "Knowledge base not found")
    if knowledge_base.get("slug") == "default":
        raise HTTPException(409, "默认知识库不能删除")
    if knowledge_base.get("status") not in {"active", "deleting", "delete_failed"}:
        raise HTTPException(409, "Knowledge base is not deletable in its current state")

    try:
        job = await get_ingest_job_store_instance().queue_knowledge_base_delete(
            tenant["tenant_id"],
            kb_id=kb_id,
            payload={
                "kb_id": kb_id,
                "knowledge_base_name": knowledge_base.get("name") or "知识库",
                "tenant_slug": tenant.get("slug") or "default",
                "reason": "api_delete",
            },
        )
    except KeyError as exc:
        raise HTTPException(404, "Knowledge base not found") from exc
    except ValueError as exc:
        if str(exc) == "default_knowledge_base_not_deletable":
            raise HTTPException(409, "默认知识库不能删除") from exc
        if str(exc) == "knowledge_base_not_deletable":
            raise HTTPException(409, "Knowledge base is not deletable in its current state") from exc
        raise
    return {
        "status": "deleting",
        "job_id": job["job_id"],
        "message": "知识库已停止对外提供服务，正在清理文档、索引、向量、缓存和关联元数据。",
    }


@router.get("/{kb_id}/indexes")
async def list_knowledge_base_indexes(kb_id: str, identity: dict = Depends(require_rag_read)):
    tenant = await resolve_identity_tenant(identity)
    await _get_active_knowledge_base(kb_id, tenant["tenant_id"])
    indexes = await get_index_profile_store_instance().list_knowledge_base_indexes(kb_id)
    return {"indexes": indexes}


@router.post("/{kb_id}/index-candidates", status_code=202)
async def build_index_candidate(
    kb_id: str,
    identity: dict = Depends(require_rag_write),
):
    """Queue a verified rebuild and atomically activate it when it is current."""
    tenant = await resolve_identity_tenant(identity)
    await _get_active_knowledge_base(kb_id, tenant["tenant_id"])
    job = await get_ingest_job_store_instance().get_or_create_active_kb_job(
        tenant["tenant_id"],
        "index_candidate",
        kb_id=kb_id,
        payload={
            "kb_id": kb_id,
            "tenant_slug": tenant.get("slug") or "default",
            "auto_activate": True,
            "reason": "manual_rebuild",
        },
    )
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "message": "索引重建已排队；完成校验后会自动切换，期间检索不中断。",
    }


@router.post("/{kb_id}/indexes/{index_id}/activate")
async def activate_knowledge_base_index_endpoint(
    kb_id: str,
    index_id: str,
    identity: dict = Depends(require_rag_write),
):
    """Atomically route a KB to a ready, source-current index generation."""
    tenant = await resolve_identity_tenant(identity)
    await _get_active_knowledge_base(kb_id, tenant["tenant_id"])
    try:
        settings = _get_settings()
        index = await finalize_knowledge_base_index_activation(
            kb_id=kb_id,
            index_id=index_id,
            settings=settings,
            vector_store=get_vector_store_instance(),
            document_store=get_document_store_instance(),
            index_profile_store=get_index_profile_store_instance(),
            bm25_store=get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None,
        )
    except KeyError:
        raise HTTPException(404, "Index not found") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return {
        "index": index,
        "message": "索引已切换；下一次检索会使用该集合及其对应的 Embedding 模型。",
    }


@router.delete("/{kb_id}/indexes/{index_id}")
async def reclaim_knowledge_base_index_endpoint(
    kb_id: str,
    index_id: str,
    identity: dict = Depends(require_rag_write),
):
    """Delete an old rollback generation and reclaim its vector/parent data."""
    tenant = await resolve_identity_tenant(identity)
    await _get_active_knowledge_base(kb_id, tenant["tenant_id"])
    settings = _get_settings()
    try:
        deleted = await reclaim_inactive_knowledge_base_index(
            kb_id=kb_id,
            index_id=index_id,
            vector_store=get_vector_store_instance(),
            index_profile_store=get_index_profile_store_instance(),
            bm25_store=get_bm25_store_instance() if settings.retrieval.hybrid.enabled else None,
        )
    except KeyError:
        raise HTTPException(404, "Index not found") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"deleted_index_id": deleted["index_id"], "status": "reclaimed"}
