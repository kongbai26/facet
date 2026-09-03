"""文档摄入管道"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.chunkers.registry import get_chunker
from app.index_profile import INDEX_PIPELINE_VERSION
from app.parsers.base import ParserPrelude, StructuredBlock
from app.parsers.registry import get_parser
from app.chunkers.recursive import conservative_token_upper_bound, estimate_tokens
from app.pipeline.parent_child import build_parent_child_chunks, split_text_by_token_budget
from app.pipeline.tokenizer_policy import (
    TOKENIZER_CAPABILITY_STATUS_REASON,
    is_tokenizer_capability_error,
)
from app.providers.embedding.base import BaseEmbeddingProvider
from app.rag_scope import get_tenant_rag_collection_name, resolve_embedding_dimension, resolve_embedding_profile
from app.settings.settings import AppConfig
from app.store.bm25_store import BM25Store, rebuild_bm25_after_change
from app.store.document_store import DocumentStore
from app.store.parent_chunk_store import ParentChunkStore
from app.store.vector_store import VectorStore
from app.utils.user_errors import sanitize_user_error_message

logger = logging.getLogger(__name__)

EMPTY_CONTENT_STATUS_REASON = "empty_content"
EMPTY_CONTENT_ERROR_MESSAGE = "文档未解析出可检索文本，可能是扫描件、图片型 PDF，或正文为空。"


@dataclass(frozen=True)
class IngestDestination:
    """Describe where one ingest writes without mixing source and candidate modes.

    Source ingestion changes the document lifecycle and writes to the current
    collection.  Candidate construction must instead leave source metadata
    untouched and write under an immutable profile generation.  Keeping this
    decision as one value avoids callers combining independent boolean flags.
    """

    collection_name: str | None = None
    index_id: str = "legacy"
    allowed_source_statuses: frozenset[str] = frozenset({"processing", "reindexing"})
    finalize_document_status: bool = True
    update_document_embedding_metadata: bool = True

    @classmethod
    def candidate(cls, collection_name: str, index_id: str) -> "IngestDestination":
        return cls(
            collection_name=collection_name,
            index_id=index_id,
            allowed_source_statuses=frozenset({"ready"}),
            finalize_document_status=False,
            update_document_embedding_metadata=False,
        )


def _adaptive_child_tokens(settings: AppConfig, context_window: int) -> int:
    """Resolve the user target against the embedding model hard limit.

    The old implementation converted a token target into a character budget
    and scaled it by an arbitrary context ratio.  That made user settings
    ineffective and could create tiny chunks for a perfectly valid 512-token
    embedding model.  The target is now always in tokens; the splitter checks
    every final ``prefix + child`` payload against ``context_window``.
    """
    configured = getattr(settings.chunking, "child_target_tokens", None)
    if configured is None:
        configured = getattr(settings.chunking, "child_max_tokens", 512)
    hard_limit = max(1, int(context_window or settings.embedding.openai.max_tokens))
    return min(max(1, int(configured)), hard_limit)


def _normalize_leading_line(text: str) -> str:
    return " ".join((text or "").strip().split())


def _extract_first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        normalized = _normalize_leading_line(line)
        if normalized:
            return normalized
    return ""


def _build_prelude_segment(prelude: ParserPrelude, first_body_segment: str) -> str:
    lines: list[str] = []
    title = _normalize_leading_line(prelude.title)
    subtitle = _normalize_leading_line(prelude.subtitle)
    body_first_line = _extract_first_nonempty_line(first_body_segment)

    if title and title != body_first_line:
        lines.append(title)
    if subtitle and subtitle not in {title, body_first_line}:
        lines.append(subtitle)

    for raw_line in prelude.header_lines:
        line = _normalize_leading_line(raw_line)
        if line and line not in lines and line != body_first_line:
            lines.append(line)

    return "\n".join(lines).strip()


def _block_metadata(block: StructuredBlock) -> dict:
    metadata = dict(block.metadata or {})
    if block.kind:
        metadata.setdefault("block_kind", block.kind)
    if block.section_title:
        metadata.setdefault("section_title", block.section_title)
    if block.heading_path:
        metadata.setdefault("heading_path", " > ".join(block.heading_path))
    if block.table_headers:
        metadata.setdefault("table_headers", " | ".join(block.table_headers))
    if block.source_anchor:
        metadata.setdefault("source_anchor", block.source_anchor)
    if block.page is not None:
        metadata.setdefault("page", block.page)
    return metadata


async def ingest_document(
    file_path: Path,
    doc_id: str,
    settings: AppConfig,
    embedding_provider: BaseEmbeddingProvider,
    vector_store: VectorStore,
    document_store: DocumentStore,
    bm25_store: Optional[BM25Store] = None,
    *,
    defer_bm25_rebuild: bool = False,
    destination: IngestDestination | None = None,
    index_profile_store=None,
) -> int:
    """摄入单个文档，返回 chunk 数量。

    defer_bm25_rebuild=True 时只失效 BM25 缓存，不立即重建，由调用方统一重建。
    """
    destination = destination or IngestDestination()
    parser = get_parser(file_path.suffix)
    profile = await resolve_embedding_profile(embedding_provider)
    embedding_dimension = profile.get("dimension")
    runtime_context = int(profile.get("context_window") or settings.embedding.openai.max_tokens)
    # The chunker is only a first structural pass.  Every final payload is
    # counted below against the actual runtime hard limit.
    max_tokens = max(1, min(int(settings.embedding.openai.max_tokens), runtime_context))
    chunker = get_chunker(settings.chunking, max_tokens=max_tokens)
    batch_size = max(1, settings.ingest.batch_size)
    if embedding_dimension is None:
        embedding_dimension = await resolve_embedding_dimension(embedding_provider)

    provider_counter = getattr(embedding_provider, "count_tokens", None)
    provider_batch_counter = getattr(embedding_provider, "count_batch_tokens", None)
    tokenizer_verified = bool(profile.get("tokenizer_verified"))
    require_exact = bool(getattr(settings.chunking, "require_exact_tokenizer", False))
    fallback_token_reserve = int(getattr(settings.chunking, "fallback_token_reserve", 8))
    tokenizer_unavailable = not callable(provider_counter) or not tokenizer_verified
    token_count_strategy = "verified" if not tokenizer_unavailable else "conservative_utf8_bound"

    if tokenizer_unavailable and not require_exact:
        logger.info(
            "文档 %s: embedding 服务未提供已验证 tokenizer，使用保守 UTF-8 token 上界切片",
            doc_id,
        )

    token_cache: dict[str, int] = {}

    async def count_payload_tokens(text: str) -> int:
        cached = token_cache.get(text)
        if cached is not None:
            return cached
        try:
            if tokenizer_verified and callable(provider_counter):
                value = int(await provider_counter(text))
            else:
                raise NotImplementedError
        except Exception as exc:
            if require_exact:
                raise RuntimeError("当前 embedding 服务无法精确计算 token") from exc
            # Do not claim a heuristic is the model's token count.  The UTF-8
            # upper bound keeps every payload under its configured input
            # limit even for mixed scripts, emoji, or unbroken identifiers.
            value = max(
                estimate_tokens(text),
                conservative_token_upper_bound(text, reserve_tokens=fallback_token_reserve),
            )
        if value < 0:
            raise RuntimeError("embedding tokenizer 返回了非法 token 数")
        token_cache[text] = value
        return value

    async def count_batch_payload_tokens(items: list[dict]) -> int:
        texts = [str(item["text"]) for item in items]
        try:
            if tokenizer_verified and callable(provider_batch_counter):
                return int(await provider_batch_counter(texts))
        except Exception as exc:
            if require_exact:
                raise RuntimeError("当前 embedding 服务无法精确计算批处理 token") from exc
        return sum(int(item["embedding_tokens"]) for item in items)

    doc = await document_store.get(doc_id)
    current_status = doc.get("status") if doc else None
    allowed_statuses = destination.allowed_source_statuses
    if not doc or current_status not in allowed_statuses:
        raise RuntimeError(f"文档 {doc_id} 当前状态不允许摄入")

    # 统一使用当前 embedding provider 的维度作为目标 collection，
    # 避免 reindex 时仍使用文档旧维度导致 ChromaDB 维度不匹配。
    if destination.update_document_embedding_metadata:
        try:
            await document_store.update_embedding_model(
                doc_id,
                settings.embedding.openai.model_name,
                embedding_dimension,
                settings.embedding.openai.api_base,
                runtime_context,
            )
        except TypeError:
            # Compatibility for simple external/test stores implementing the
            # old three-argument document metadata contract.
            await document_store.update_embedding_model(
                doc_id,
                settings.embedding.openai.model_name,
                embedding_dimension,
            )

    collection_name = destination.collection_name or _get_collection_name(
        settings,
        doc.get("tenant_slug"),
        settings.embedding.openai.model_name,
        embedding_dimension,
    )
    logger.info(
        "文档 %s 摄入参数: batch_size=%d collection=%s",
        doc_id,
        batch_size,
        collection_name,
    )

    parent_store: ParentChunkStore | None = None
    try:
        if require_exact and tokenizer_unavailable:
            raise RuntimeError("当前 embedding 服务未提供已验证 tokenizer，生产索引已拒绝使用估算 token")
        if index_profile_store is not None:
            await index_profile_store.upsert_document_state(doc_id, destination.index_id, "building")
        if destination.collection_name:
            # Retrying a candidate rebuild replaces only that candidate's
            # representation.  The active collection and legacy parents stay
            # untouched until an explicit cutover.
            await _delete_doc_vectors(vector_store, doc_id, collection_name, embedding_dimension)
            await ParentChunkStore(settings.storage.metadata_db).delete_document(
                doc_id,
                profile_hash=destination.index_id,
            )
        total_chunks = 0
        next_chunk_index = 0
        pending_chunks = []
        pending_batch_tokens = 0
        prelude_getter = getattr(parser, "get_prelude", None)
        prelude = prelude_getter(file_path) if callable(prelude_getter) else ParserPrelude()
        parse_blocks_getter = getattr(parser, "parse_blocks", None)
        structured_blocks = parse_blocks_getter(file_path) if callable(parse_blocks_getter) else []
        if not structured_blocks:
            structured_blocks = [StructuredBlock(text=segment) for segment in parser.parse_stream(file_path) if (segment or "").strip()]
        first_body_segment = structured_blocks[0].text if structured_blocks else ""
        prelude_segment = _build_prelude_segment(prelude, first_body_segment)
        leading_blocks: list[StructuredBlock] = []
        if prelude_segment:
            leading_blocks.append(
                StructuredBlock(
                    text=prelude_segment,
                    kind="title",
                    section_title=prelude.title or prelude_segment,
                    heading_path=(prelude.title or prelude_segment,),
                )
            )

        def is_batch_capacity_error(exc: Exception) -> bool:
            message = str(exc).lower()
            return any(
                marker in message
                for marker in (
                    "context length",
                    "context window",
                    "maximum context",
                    "maximum input",
                    "max input",
                    "too many tokens",
                    "token limit",
                    "input too long",
                    "input is too long",
                    "exceeds the limit",
                    "exceeds context",
                )
            )

        async def embed_with_adaptive_batch_split(texts: list[str]) -> list[list[float]]:
            """Retry an unknown server batch limit by bisecting only that batch.

            OpenAI-compatible APIs do not standardise a total-token capacity.
            A deployment can configure it, but when it is unknown we learn the
            boundary from an explicit capacity error instead of failing the
            entire document.  Per-item limits remain guarded by the splitter.
            """
            try:
                return await embedding_provider.embed_texts(texts)
            except Exception as exc:
                if len(texts) <= 1 or not is_batch_capacity_error(exc):
                    raise
                midpoint = len(texts) // 2
                logger.warning(
                    "Embedding 批次超出服务容量，拆分重试: document=%s items=%d",
                    doc_id,
                    len(texts),
                )
                left = await embed_with_adaptive_batch_split(texts[:midpoint])
                right = await embed_with_adaptive_batch_split(texts[midpoint:])
                return [*left, *right]

        async def flush_batch() -> None:
            nonlocal total_chunks, next_chunk_index, pending_chunks, pending_batch_tokens
            if not pending_chunks:
                return

            texts = [item["text"] for item in pending_chunks]
            vectors = await embed_with_adaptive_batch_split(texts)
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"embedding 返回数量异常: expected={len(texts)} actual={len(vectors)}"
                )
            if embedding_dimension is not None:
                actual_dimensions = {len(vector) for vector in vectors}
                if actual_dimensions != {embedding_dimension}:
                    raise RuntimeError(
                        "embedding 维度在摄入过程中发生变化: "
                        f"expected={embedding_dimension} actual={sorted(actual_dimensions)}"
                    )
            metadatas = []
            ids = []
            filename = doc.get("filename") or file_path.name
            file_stem = Path(filename).stem
            extension = Path(filename).suffix.lower()
            for offset, item in enumerate(pending_chunks):
                chunk_index = next_chunk_index + offset
                chunk_id = f"{doc_id}_{chunk_index}"
                metadatas.append({
                    "doc_id": doc_id,
                    # A document belongs to exactly one knowledge base.  Keep
                    # that fact on every searchable chunk so retrieval can
                    # filter a selected KB set without first expanding it to
                    # a potentially large doc-id list.
                    "kb_id": doc.get("kb_id") or "",
                    "tenant_id": doc.get("tenant_id") or "",
                    "tenant_slug": doc.get("tenant_slug") or "",
                    "filename": filename,
                    "file_stem": file_stem,
                    "extension": extension,
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                    "embedding_model": settings.embedding.openai.model_name,
                    "embedding_dimension": embedding_dimension,
                    "embedding_tokens": int(item["embedding_tokens"]),
                    "embedding_token_count_strategy": token_count_strategy,
                    **item["metadata"],
                })
                ids.append(chunk_id)

            await _add_vectors(
                vector_store,
                texts=texts,
                vectors=vectors,
                metadatas=metadatas,
                ids=ids,
                collection_name=collection_name,
                embedding_dimension=embedding_dimension,
            )
            next_chunk_index += len(pending_chunks)
            total_chunks += len(pending_chunks)
            logger.info(f"文档 {doc_id}: 已处理 {total_chunks} 个 chunks")
            pending_chunks = []
            pending_batch_tokens = 0

        all_blocks = [*leading_blocks, *structured_blocks]
        if bool(getattr(settings.chunking, "parent_child_enabled", False)):
            parent_store = ParentChunkStore(settings.storage.metadata_db)
            parents, children = await build_parent_child_chunks(
                doc_id,
                all_blocks,
                title=prelude.title,
                parent_max_tokens=int(getattr(settings.chunking, "parent_max_tokens", 1024)),
                child_target_tokens=_adaptive_child_tokens(settings, runtime_context),
                child_hard_limit_tokens=runtime_context,
                child_continuity_tokens=(
                    getattr(settings.chunking, "child_continuity_tokens", None)
                    if getattr(settings.chunking, "child_continuity_tokens", None) is not None
                    else int(getattr(settings.chunking, "child_overlap_tokens", 40))
                ),
                token_counter=count_payload_tokens,
                index_profile_hash=destination.index_id,
            )
            await parent_store.replace_document(
                doc_id,
                [
                    {
                        "parent_id": parent.parent_id,
                        "parent_index": parent.parent_index,
                        "text": parent.text,
                        "metadata": {
                            **parent.metadata,
                            "kb_id": doc.get("kb_id") or "",
                        },
                    }
                    for parent in parents
                ],
                profile_hash=destination.index_id,
            )
            child_items = [
                {
                    "text": child.text,
                    "metadata": {
                        **child.metadata,
                        "chunk_local_index": local_index,
                        "chunk_block_kind": child.metadata.get("block_kind", "paragraph"),
                    },
                }
                for local_index, child in enumerate(children)
            ]
        else:
            child_items = []
            for block in all_blocks:
                metadata = _block_metadata(block)
                for local_index, chunk in enumerate(chunker.chunk(block.text)):
                    chunk_text = getattr(chunk, "text", chunk)
                    pieces = await split_text_by_token_budget(
                        chunk_text,
                        prefix="",
                        target_tokens=_adaptive_child_tokens(settings, runtime_context),
                        hard_limit_tokens=runtime_context,
                        continuity_tokens=0,
                        token_counter=count_payload_tokens,
                    )
                    for piece_index, piece in enumerate(pieces):
                        child_items.append({
                            "text": piece,
                            "metadata": {
                                **metadata,
                                "chunk_local_index": getattr(chunk, "index", local_index) + piece_index,
                                "chunk_block_kind": block.kind,
                            },
                        })

        for item in child_items:
            item["embedding_tokens"] = await count_payload_tokens(item["text"])
            if item["embedding_tokens"] > runtime_context:
                raise RuntimeError("最终 embedding 文本超过模型 token 上限")
            max_batch_tokens = getattr(settings.embedding.openai, "max_batch_tokens", None)
            if (
                max_batch_tokens is not None
                and pending_chunks
                and await count_batch_payload_tokens([*pending_chunks, item]) > int(max_batch_tokens)
            ):
                await flush_batch()
            pending_chunks.append(item)
            pending_batch_tokens += item["embedding_tokens"]
            if len(pending_chunks) >= batch_size:
                await flush_batch()

        await flush_batch()

        if total_chunks == 0:
            await document_store.update_status_if(
                doc_id,
                [current_status],
                "failed",
                error_message=EMPTY_CONTENT_ERROR_MESSAGE,
                chunks_count=0,
                status_reason=EMPTY_CONTENT_STATUS_REASON,
            )
            raise RuntimeError(EMPTY_CONTENT_ERROR_MESSAGE)

        if destination.finalize_document_status:
            updated = await document_store.update_status_if(
                doc_id,
                [current_status],
                "ready",
                chunks_count=total_chunks,
                status_reason="",
            )
            if not updated:
                raise RuntimeError(f"文档 {doc_id} 状态已变化，无法标记为 ready")
            update_index_version = getattr(document_store, "update_index_pipeline_version", None)
            if callable(update_index_version):
                await update_index_version(
                    doc_id,
                    INDEX_PIPELINE_VERSION,
                )

        if index_profile_store is not None:
            await index_profile_store.upsert_document_state(
                doc_id,
                destination.index_id,
                "ready",
                chunk_count=total_chunks,
            )

        if bm25_store and settings.retrieval.hybrid.enabled:
            bm25_store.invalidate_cache(collection_name)
            if not defer_bm25_rebuild:
                await rebuild_bm25_after_change(
                    bm25_store,
                    vector_store,
                    document_store,
                    settings,
                    collection_name,
                )
        logger.info(f"文档 {doc_id} 摄入完成，共 {total_chunks} 个 chunks")
        return total_chunks

    except Exception as e:
        logger.exception(f"文档 {doc_id} 摄入失败")
        cleanup_errors = await _cleanup_failed_ingest(
            doc_id,
            file_path,
            settings,
            vector_store,
            bm25_store,
            collection_name,
            embedding_dimension,
        )
        if parent_store is not None:
            try:
                await parent_store.delete_document(doc_id, profile_hash=destination.index_id)
            except Exception as parent_cleanup_error:
                cleanup_errors.append(f"父块清理失败: {parent_cleanup_error}")
        error_message = sanitize_user_error_message(
            str(e),
            "文档摄入失败，请检查模型配置后重试。",
        )
        if cleanup_errors:
            error_message = f"{error_message} 已完成残留清理，请稍后重试。"
        if index_profile_store is not None:
            await index_profile_store.upsert_document_state(
                doc_id,
                destination.index_id,
                "failed",
                error_message=error_message,
            )
        if destination.finalize_document_status:
            status_reason = (
                TOKENIZER_CAPABILITY_STATUS_REASON
                if is_tokenizer_capability_error(error_message)
                else ("reindex_failed" if current_status == "reindexing" else "")
            )
            await document_store.update_status_if(
                doc_id,
                [current_status],
                "failed",
                error_message=error_message,
                chunks_count=0,
                status_reason=status_reason,
            )
        raise


async def _cleanup_failed_ingest(
    doc_id: str,
    file_path: Path,
    settings: AppConfig,
    vector_store: VectorStore,
    bm25_store: Optional[BM25Store],
    collection_name: str,
    embedding_dimension: int | None = None,
) -> list[str]:
    errors = []
    try:
        await _delete_doc_vectors(vector_store, doc_id, collection_name, embedding_dimension)
    except Exception as e:
        errors.append(f"向量清理失败: {e}")
        logger.warning(f"文档 {doc_id} 清理向量失败: {e}")

    if bm25_store and settings.retrieval.hybrid.enabled:
        bm25_store.invalidate_cache(collection_name)

    return errors


def _get_collection_name(
    settings: AppConfig,
    tenant_slug: str | None = None,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
) -> str:
    return get_tenant_rag_collection_name(
        settings.vectorstore.collection_prefix,
        embedding_model or settings.embedding.openai.model_name,
        tenant_slug=tenant_slug,
        embedding_dimension=embedding_dimension,
    )


async def _add_vectors(
    vector_store,
    *,
    texts,
    vectors,
    metadatas,
    ids,
    collection_name: str,
    embedding_dimension: int | None = None,
) -> None:
    add_kwargs = {
        "texts": texts,
        "vectors": vectors,
        "metadatas": metadatas,
        "ids": ids,
        "collection_name": collection_name,
    }
    if embedding_dimension is not None:
        add_kwargs["dimension"] = embedding_dimension
    try:
        await vector_store.add(**add_kwargs)
    except TypeError:
        add_kwargs.pop("dimension", None)
        await vector_store.add(**add_kwargs)


async def _delete_doc_vectors(vector_store, doc_id: str, collection_name: str, embedding_dimension: int | None = None) -> None:
    try:
        await vector_store.delete_by_doc_id(doc_id, collection_name=collection_name, dimension=embedding_dimension)
    except TypeError:
        await vector_store.delete_by_doc_id(doc_id, collection_name=collection_name)
