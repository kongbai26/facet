"""Startup bootstrap helpers for default multi-tenant metadata."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def ensure_tenant_default_knowledge_base(
    settings,
    tenant: dict,
    namespace_store,
    knowledge_base_store,
) -> dict:
    """Ensure the default RAG namespace/KB belongs to the given tenant."""
    tenant_id = tenant["tenant_id"]
    namespace = await namespace_store.ensure_default(
        tenant_id,
        slug="default-kb",
        name="Default RAG Namespace",
        kind="rag",
    )
    knowledge_base = await knowledge_base_store.ensure_default(
        tenant_id,
        namespace["namespace_id"],
        slug="default",
        name="默认知识库",
        embedding_model=settings.embedding.openai.model_name,
        llm_model=settings.llm.model_name,
    )
    return {"namespace": namespace, "knowledge_base": knowledge_base}


async def ensure_default_workspace(
    settings,
    tenant_store,
    principal_store,
    namespace_store,
    knowledge_base_store,
) -> dict:
    """Ensure the default tenant/admin principal/RAG namespace/KB exist."""
    tenant = await tenant_store.ensure_default()
    principal = await principal_store.ensure_default_admin(tenant["tenant_id"])
    defaults = await ensure_tenant_default_knowledge_base(
        settings, tenant, namespace_store, knowledge_base_store,
    )
    return {
        "tenant": tenant,
        "principal": principal,
        **defaults,
    }


async def backfill_missing_tenant_metadata(
    document_store,
    conversation_store,
    tenant_id: str,
    tenant_slug: str,
    default_kb_id: str | None = None,
) -> dict:
    """Fill legacy rows that were created before tenant metadata existed."""
    normalized_slug = tenant_slug or "default"
    documents_updated = await document_store.assign_missing_tenant(tenant_id, normalized_slug)
    conversations_updated = await conversation_store.assign_missing_tenant(tenant_id)
    knowledge_base_updated = 0
    if default_kb_id:
        knowledge_base_updated = await document_store.assign_missing_knowledge_base(
            default_kb_id, tenant_id=tenant_id,
        )
    logger.info(
        "backfilled legacy tenant metadata: documents=%d conversations=%d knowledge_base=%d tenant_id=%s tenant_slug=%s",
        documents_updated,
        conversations_updated,
        knowledge_base_updated,
        tenant_id,
        normalized_slug,
    )
    return {
        "documents_updated": documents_updated,
        "conversations_updated": conversations_updated,
        "knowledge_base_updated": knowledge_base_updated,
    }
