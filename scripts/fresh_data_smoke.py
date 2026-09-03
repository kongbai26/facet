"""Build and query a clean, fresh knowledge-base generation with real services."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from app.config import get_config
from app.pipeline.chat_flow import prepare_retrieval_only
from app.pipeline.index_lifecycle import reconcile_knowledge_base_index
from app.pipeline.ingest import ingest_document
from app.pipeline.retrieval_target import RetrievalTarget
from app.providers.embedding.registry import get_embedding_provider
from app.store.bm25_store import BM25Store
from app.store.document_store import DocumentStore
from app.store.index_profile_store import IndexProfileStore
from app.store.vector_store import VectorStore


class NoopLLM:
    async def chat(self, *_args, **_kwargs):
        raise AssertionError("fresh-data retrieval smoke does not require an LLM call")


async def main() -> None:
    settings = get_config()
    docs = [
        ("fresh-guide", "上线运行手册.md"),
        ("fresh-cache", "缓存处置说明.md"),
    ]
    tenant_id = "fresh-smoke-tenant"
    tenant_slug = "fresh-smoke"
    kb_id = "fresh-smoke-kb"

    document_store = DocumentStore(settings.storage.metadata_db)
    vector_store = VectorStore(settings.vectorstore, settings.embedding.openai.model_name)
    profile_store = IndexProfileStore(settings.storage.metadata_db)
    bm25_store = BM25Store(
        cache_dir=settings.retrieval.hybrid.bm25_cache_dir,
        lexical_metadata_fields=settings.retrieval.exact_match.lexical_metadata_fields,
    )
    embedding_provider = get_embedding_provider(settings.embedding, settings.vectorstore)
    dimension = await embedding_provider.dimension()

    for doc_id, filename in docs:
        source = Path(settings.storage.upload_dir) / doc_id / filename
        content = source.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        existing = await document_store.get(doc_id)
        if existing is None:
            await document_store.create(
                doc_id,
                filename,
                len(content),
                content_hash=content_hash,
                embedding_model=settings.embedding.openai.model_name,
                embedding_dimension=dimension,
                tenant_id=tenant_id,
                tenant_slug=tenant_slug,
                kb_id=kb_id,
            )
            await ingest_document(
                source,
                doc_id,
                settings,
                embedding_provider,
                vector_store,
                document_store,
                bm25_store,
            )
        elif existing.get("content_hash") != content_hash or existing.get("status") != "ready":
            raise RuntimeError(
                f"fresh smoke document {doc_id} is not the expected ready source; clean runtime data first"
            )

    active = await reconcile_knowledge_base_index(
        kb_id=kb_id,
        tenant_slug=tenant_slug,
        settings=settings,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        document_store=document_store,
        index_profile_store=profile_store,
        bm25_store=bm25_store,
        auto_activate=True,
    )
    turn = await prepare_retrieval_only(
        "发布窗口内可以直接清空生产缓存吗？",
        [],
        settings,
        NoopLLM(),
        embedding_provider,
        vector_store,
        document_store,
        bm25_store,
        tenant_id=tenant_id,
        tenant_slug=tenant_slug,
        allowed_kb_ids=[kb_id],
        retrieval_targets=[RetrievalTarget(
            collection_name=active["collection_name"],
            embedding_provider=embedding_provider,
            kb_ids=(kb_id,),
            profile_hash=active["profile_hash"],
            index_id=active["index_id"],
        )],
    )
    source_doc_ids = [source["doc_id"] for source in turn.sources]
    if turn.decision != "RETRIEVE" or not source_doc_ids or source_doc_ids[0] != "fresh-guide":
        raise RuntimeError(f"fresh corpus retrieval failed: decision={turn.decision} sources={source_doc_ids}")
    print(json.dumps({
        "active_index_id": active["index_id"],
        "active_collection": active["collection_name"],
        "source_doc_ids": source_doc_ids,
    }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
