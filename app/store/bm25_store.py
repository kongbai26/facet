"""BM25 索引管理"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rank_bm25 import BM25Okapi

from app.utils.retrieval_match import (
    build_lexical_search_text,
    compute_exact_match_bonus,
    normalize_filename,
)
from app.utils.text_utils import tokenize_mixed

logger = logging.getLogger(__name__)

BM25_CACHE_SCHEMA_VERSION = 6
DEFAULT_LEXICAL_METADATA_FIELDS = (
    "filename",
    "file_stem",
    "extension",
    "doc_id",
    "tenant_slug",
    "block_kind",
    "section_title",
    "heading_path",
    "table_headers",
    "source_anchor",
)


@dataclass(frozen=True)
class BM25Snapshot:
    """An immutable-in-practice index view used by one collection's search."""

    corpus: tuple[str, ...]
    doc_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    chunk_indexes: tuple[int, ...]
    metadatas: tuple[dict, ...]
    bm25: BM25Okapi | None
    empty_ready: bool
    lexical_metadata_fields: tuple[str, ...]


async def rebuild_bm25_after_change(
    bm25_store: Optional["BM25Store"],
    vector_store,
    document_store,
    settings,
    collection_name: str,
) -> None:
    """文档变更（摄入/删除/重新索引）后，若 hybrid 启用且仍有就绪文档则重建 BM25 索引。"""
    if not bm25_store or not settings.retrieval.hybrid.enabled:
        return
    try:
        if not await document_store.has_ready_documents():
            return
        await bm25_store.ensure_ready(
            vector_store,
            collection_name,
            document_store=document_store,
        )
        logger.info("BM25 变更后重建完成: collection=%s", collection_name)
    except Exception as exc:
        logger.warning("BM25 变更后重建失败，将在首次检索时懒加载: %s", exc)


class BM25Store:
    def __init__(
        self,
        cache_dir: str = "./data/bm25_cache",
        lexical_metadata_fields: Sequence[str] | None = None,
    ):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._corpus: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._doc_ids: List[str] = []
        self._chunk_ids: List[str] = []
        self._chunk_indexes: List[int] = []
        self._metadatas: List[dict] = []
        self._fingerprint = ""
        self._bm25: Optional[BM25Okapi] = None
        self._empty_ready = False
        self._active_collection_name: Optional[str] = None
        self._snapshots: dict[str, BM25Snapshot] = {}
        self._lock = asyncio.Lock()
        self._lexical_metadata_fields = tuple(lexical_metadata_fields or DEFAULT_LEXICAL_METADATA_FIELDS)

    async def build_from_chunks(
        self,
        chunks: List[Dict],
        save_cache: bool = True,
        collection_name: str = "default",
        fingerprint: str = "",
        lexical_metadata_fields: Sequence[str] | None = None,
    ) -> None:
        """从 chunk 列表构建 BM25 索引。"""
        self._corpus = [c["text"] for c in chunks]
        self._doc_ids = [c.get("doc_id", "") for c in chunks]
        self._chunk_ids = [c.get("chunk_id", "") for c in chunks]
        self._chunk_indexes = [int(c.get("chunk_index", idx)) for idx, c in enumerate(chunks)]
        self._metadatas = [self._prepare_metadata(c) for c in chunks]
        self._fingerprint = fingerprint or self._fingerprint_chunks(chunks)
        if lexical_metadata_fields is not None:
            self._lexical_metadata_fields = tuple(lexical_metadata_fields)

        if not self._corpus:
            self._set_empty_ready()
            self._active_collection_name = collection_name
            self._fingerprint = fingerprint or self._fingerprint_chunks([])
            if save_cache:
                await self.save_cache(collection_name)
            self._remember_snapshot(collection_name)
            return

        self._tokenized_corpus = await asyncio.to_thread(
            lambda: [tokenize_mixed(text) for text in self._corpus]
        )
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._empty_ready = False
        self._active_collection_name = collection_name
        logger.info(f"BM25 索引构建完成，共 {len(self._corpus)} 个 chunks")

        if save_cache:
            await self.save_cache(collection_name)
        self._remember_snapshot(collection_name)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        collection_name: str | None = None,
        exact_match_config=None,
        allowed_doc_ids: List[str] | None = None,
        allowed_kb_ids: List[str] | None = None,
    ) -> List[Dict]:
        """BM25 检索。"""
        snapshot = self._snapshot_for_search(collection_name)
        if snapshot is None or not snapshot.corpus:
            return []

        allowed_doc_id_set = {doc_id for doc_id in (allowed_doc_ids or []) if doc_id}
        allowed_kb_id_set = {kb_id for kb_id in (allowed_kb_ids or []) if kb_id}
        query_tokens = await asyncio.to_thread(tokenize_mixed, query)
        if snapshot.bm25 is None:
            return []
        scores = await asyncio.to_thread(snapshot.bm25.get_scores, query_tokens)

        indexed_scores = list(enumerate(scores))
        ranked_results = []
        for idx, score in indexed_scores:
            metadata = snapshot.metadatas[idx] if idx < len(snapshot.metadatas) else {}
            search_text = build_lexical_search_text(
                snapshot.corpus[idx],
                metadata,
                lexical_metadata_fields=snapshot.lexical_metadata_fields,
            )
            exact_bonus, exact_reasons = compute_exact_match_bonus(
                query,
                search_text,
                metadata,
                exact_match_config,
                lexical_metadata_fields=snapshot.lexical_metadata_fields,
            )
            rank_score = float(score) + exact_bonus
            ranked_results.append((idx, float(score), rank_score, exact_bonus, exact_reasons))

        ranked_results.sort(key=lambda item: (item[2], item[1]), reverse=True)

        results = []
        for idx, score, rank_score, exact_bonus, exact_reasons in ranked_results:
            if score <= 0 and exact_bonus <= 0:
                continue
            metadata = snapshot.metadatas[idx] if idx < len(snapshot.metadatas) else {
                "doc_id": snapshot.doc_ids[idx],
                "chunk_index": snapshot.chunk_indexes[idx],
                "chunk_id": snapshot.chunk_ids[idx],
            }
            if allowed_doc_id_set and metadata.get("doc_id") not in allowed_doc_id_set:
                continue
            if allowed_kb_id_set and metadata.get("kb_id") not in allowed_kb_id_set:
                continue
            results.append({
                "text": snapshot.corpus[idx],
                "metadata": metadata,
                "chunk_id": snapshot.chunk_ids[idx],
                "bm25_score": float(score),
                "bm25_rank_score": float(rank_score),
                "exact_match_bonus": float(exact_bonus),
                "exact_match_reasons": exact_reasons,
            })
            if len(results) >= top_k:
                break

        return results

    async def save_cache(self, collection_name: str) -> None:
        """缓存 BM25 索引到磁盘。"""
        cache_file = self._cache_file(collection_name)
        data = {
            "corpus": self._corpus,
            "tokenized_corpus": self._tokenized_corpus,
            "doc_ids": self._doc_ids,
            "chunk_ids": self._chunk_ids,
            "chunk_indexes": self._chunk_indexes,
            "metadatas": self._metadatas,
            "fingerprint": self._fingerprint,
            "count": len(self._corpus),
            "schema_version": BM25_CACHE_SCHEMA_VERSION,
        }
        await asyncio.to_thread(self._write_cache, cache_file, data)
        logger.info(f"BM25 缓存已保存: {cache_file} ({len(self._corpus)} chunks)")

    async def load_cache(self, collection_name: str) -> bool:
        """从磁盘加载 BM25 缓存。"""
        cache_file = self._cache_file(collection_name)
        if not cache_file.exists():
            return False

        try:
            data = await asyncio.to_thread(self._read_cache, cache_file)
        except Exception as e:
            logger.warning(f"BM25 缓存加载失败: {e}")
            return False

        if (
            data.get("schema_version") != BM25_CACHE_SCHEMA_VERSION
            or "chunk_indexes" not in data
            or "fingerprint" not in data
            or "metadatas" not in data
        ):
            logger.info(f"BM25 缓存版本过期或缺少元数据，视为过期: {cache_file}")
            return False

        self._corpus = data["corpus"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._doc_ids = data["doc_ids"]
        self._chunk_ids = data["chunk_ids"]
        self._chunk_indexes = data["chunk_indexes"]
        self._metadatas = data["metadatas"]
        self._fingerprint = data["fingerprint"]

        if not self._corpus:
            self._set_empty_ready()
            self._active_collection_name = collection_name
            self._remember_snapshot(collection_name)
            return True

        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._empty_ready = False
        self._active_collection_name = collection_name
        self._remember_snapshot(collection_name)
        logger.info(f"BM25 缓存已加载: {len(self._corpus)} 个 chunks")
        return True

    async def ensure_ready(
        self,
        vector_store,
        collection_name: str = "default",
        document_store=None,
    ) -> bool:
        """确保 BM25 可用，失败由调用方决定是否降级。"""
        if collection_name in self._snapshots:
            return True

        async with self._lock:
            if collection_name in self._snapshots:
                return True

            chunks = await self._chunks_from_collection(
                vector_store,
                collection_name,
            )
            if not chunks:
                await self.build_from_chunks(
                    [],
                    save_cache=True,
                    collection_name=collection_name,
                    fingerprint=self._fingerprint_chunks([]),
                )
                return True

            if document_store is not None:
                await self._backfill_missing_metadata(
                    vector_store,
                    collection_name,
                    chunks,
                    document_store,
                )
                chunks = await self._chunks_from_collection(
                    vector_store,
                    collection_name,
                )

            actual_count = len(chunks)
            actual_fingerprint = self._fingerprint_chunks(chunks)

            if actual_count == 0:
                await self.build_from_chunks(
                    [],
                    save_cache=True,
                    collection_name=collection_name,
                    fingerprint=actual_fingerprint,
                )
                return True

            cache_meta = self._load_cache_meta(collection_name)
            if (
                cache_meta
                and cache_meta.get("count") == actual_count
                and cache_meta.get("has_chunk_indexes")
                and cache_meta.get("has_metadatas")
                and cache_meta.get("fingerprint") == actual_fingerprint
            ):
                if await self.load_cache(collection_name):
                    return True

            await self.build_from_chunks(
                chunks,
                save_cache=True,
                collection_name=collection_name,
                fingerprint=actual_fingerprint,
                lexical_metadata_fields=self._lexical_metadata_fields,
            )
            logger.info(f"BM25 索引从向量库重建完成，共 {len(chunks)} 个 chunks")
            return True

    async def rebuild_from_vector_store(
        self,
        vector_store,
        collection_name: str = "default",
        document_store=None,
    ) -> bool:
        """从向量库重建 BM25 索引。"""
        chunks = await self._chunks_from_collection(vector_store, collection_name)
        if document_store is not None:
            await self._backfill_missing_metadata(vector_store, collection_name, chunks, document_store)
            chunks = await self._chunks_from_collection(vector_store, collection_name)
        await self.build_from_chunks(
            chunks,
            save_cache=True,
            collection_name=collection_name,
            lexical_metadata_fields=self._lexical_metadata_fields,
        )
        logger.info(f"BM25 索引从向量库重建完成，共 {len(chunks)} 个 chunks")
        return True

    async def _chunks_from_collection(self, vector_store, collection_name: str) -> List[Dict]:
        get_all_chunks = getattr(vector_store, "get_all_chunks", None)
        if callable(get_all_chunks):
            all_data = await get_all_chunks(collection_name=collection_name)
        else:
            collection = self._resolve_vector_collection(vector_store, collection_name)
            all_data = await asyncio.to_thread(
                collection.get,
                include=["documents", "metadatas"],
            )

        chunks = []
        documents = all_data.get("documents") or []
        metadatas = all_data.get("metadatas") or []
        ids = all_data.get("ids") or []
        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            metadata = meta or {}
            doc_id = metadata.get("doc_id", "")
            chunk_index = int(metadata.get("chunk_index", i))
            chunk_id = metadata.get("chunk_id")
            if not chunk_id and i < len(ids):
                chunk_id = ids[i]
            if not chunk_id:
                chunk_id = f"{doc_id}_{chunk_index}" if doc_id else str(chunk_index)
            metadata = self._prepare_metadata(
                {
                    **metadata,
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                }
            )
            chunks.append({
                "text": doc,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "metadata": metadata,
            })
        return chunks

    def _load_cache_meta(self, collection_name: str) -> Optional[Dict]:
        """加载缓存元数据（不加载完整数据）。"""
        cache_file = self._cache_file(collection_name)
        if not cache_file.exists():
            return None
        try:
            data = self._read_cache(cache_file)
            return {
                "count": data.get("count", 0),
                "has_chunk_indexes": "chunk_indexes" in data,
                "has_metadatas": "metadatas" in data,
                "fingerprint": data.get("fingerprint", ""),
            }
        except Exception:
            return None

    def invalidate_cache(self, collection_name: str = None) -> None:
        """标记缓存失效。"""
        if collection_name:
            for cache_file in self._cache_files(collection_name):
                if cache_file.exists():
                    cache_file.unlink(missing_ok=True)
                    logger.info(f"BM25 缓存已删除: {cache_file}")
            self._snapshots.pop(collection_name, None)
        if collection_name is None:
            self._snapshots.clear()
            self._clear_memory()
        elif self._active_collection_name == collection_name:
            self._clear_memory()

    def cleanup_orphaned_caches(self, valid_collection_names: set[str]) -> int:
        """Delete derived BM25 files that no longer have a RAG representation.

        BM25 files are disposable projections of Chroma data.  They do not
        need a time-based retention window: if no document or retained index
        generation points at a collection, the file can be regenerated only
        from data that no longer exists and should be removed at startup.
        """
        valid_names = {str(name) for name in valid_collection_names if name}
        deleted = 0
        for cache_file in [*self._cache_dir.glob("*.json"), *self._cache_dir.glob("*.pkl")]:
            collection_name = cache_file.stem
            # Legacy pickle files are always disposable and are never read.
            if cache_file.suffix == ".json" and collection_name in valid_names:
                continue
            try:
                cache_file.unlink()
                self._snapshots.pop(collection_name, None)
                if self._active_collection_name == collection_name:
                    self._clear_memory()
                deleted += 1
            except OSError as exc:
                logger.warning("清理孤立 BM25 缓存失败: %s", cache_file, exc_info=exc)
        return deleted

    def _clear_memory(self) -> None:
        self._bm25 = None
        self._corpus = []
        self._tokenized_corpus = []
        self._doc_ids = []
        self._chunk_ids = []
        self._chunk_indexes = []
        self._metadatas = []
        self._fingerprint = ""
        self._empty_ready = False
        self._active_collection_name = None

    def _set_empty_ready(self) -> None:
        self._bm25 = None
        self._corpus = []
        self._tokenized_corpus = []
        self._doc_ids = []
        self._chunk_ids = []
        self._chunk_indexes = []
        self._metadatas = []
        self._fingerprint = self._fingerprint_chunks([])
        self._empty_ready = True

    def _fingerprint_chunks(self, chunks: List[Dict]) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(self._lexical_metadata_fields, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
        for chunk in sorted(
            chunks,
            key=lambda item: (
                str(item.get("chunk_id", "")),
                str(item.get("doc_id", "")),
                int(item.get("chunk_index", 0)),
            ),
        ):
            metadata = chunk.get("metadata") or {
                key: value
                for key, value in chunk.items()
                if key not in {"text", "metadata"}
            }
            digest.update(str(chunk.get("chunk_id", "")).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(chunk.get("doc_id", "")).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(chunk.get("chunk_index", "")).encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(chunk.get("text", "").encode("utf-8")).hexdigest().encode("ascii"))
            digest.update(b"\0")
            digest.update(json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def _cache_file(self, collection_name: str) -> Path:
        return self._cache_dir / f"{collection_name}.json"

    def _cache_files(self, collection_name: str) -> tuple[Path, Path]:
        return (
            self._cache_file(collection_name),
            self._cache_dir / f"{collection_name}.pkl",
        )

    @staticmethod
    def _write_cache(cache_file: Path, data: Dict) -> None:
        """Write a disposable cache atomically without executable serialization."""
        temp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
        with open(temp_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, default=str)
        temp_file.replace(cache_file)

    @staticmethod
    def _read_cache(cache_file: Path) -> Dict:
        with open(cache_file, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("BM25 cache must contain a JSON object")
        return data

    @staticmethod
    def _resolve_vector_collection(vector_store, collection_name: str):
        return vector_store.get_collection(collection_name=collection_name)

    @property
    def is_ready(self) -> bool:
        return self._empty_ready or (self._bm25 is not None and len(self._corpus) > 0)

    def is_collection_ready(self, collection_name: str) -> bool:
        if collection_name in self._snapshots:
            return True
        return self._cache_file(collection_name).exists()

    def _remember_snapshot(self, collection_name: str) -> None:
        self._snapshots[collection_name] = BM25Snapshot(
            corpus=tuple(self._corpus),
            doc_ids=tuple(self._doc_ids),
            chunk_ids=tuple(self._chunk_ids),
            chunk_indexes=tuple(self._chunk_indexes),
            metadatas=tuple(self._metadatas),
            bm25=self._bm25,
            empty_ready=self._empty_ready,
            lexical_metadata_fields=tuple(self._lexical_metadata_fields),
        )

    def _snapshot_for_search(self, collection_name: str | None) -> BM25Snapshot | None:
        if collection_name:
            return self._snapshots.get(collection_name)
        if not self.is_ready:
            return None
        return BM25Snapshot(
            corpus=tuple(self._corpus),
            doc_ids=tuple(self._doc_ids),
            chunk_ids=tuple(self._chunk_ids),
            chunk_indexes=tuple(self._chunk_indexes),
            metadatas=tuple(self._metadatas),
            bm25=self._bm25,
            empty_ready=self._empty_ready,
            lexical_metadata_fields=tuple(self._lexical_metadata_fields),
        )

    def _prepare_metadata(self, chunk: Dict) -> dict:
        metadata = dict(chunk.get("metadata") or {})
        if chunk.get("doc_id") and not metadata.get("doc_id"):
            metadata["doc_id"] = chunk["doc_id"]
        if chunk.get("chunk_id") and not metadata.get("chunk_id"):
            metadata["chunk_id"] = chunk["chunk_id"]
        if chunk.get("chunk_index") is not None and metadata.get("chunk_index") is None:
            metadata["chunk_index"] = chunk["chunk_index"]

        filename = metadata.get("filename")
        if filename:
            stem, extension = normalize_filename(filename)
            metadata.setdefault("file_stem", stem)
            metadata.setdefault("extension", extension)
        return metadata

    async def _backfill_missing_metadata(
        self,
        vector_store,
        collection_name: str,
        chunks: List[Dict],
        document_store,
    ) -> None:
        missing = [
            chunk
            for chunk in chunks
            if self._needs_lexical_metadata_backfill(chunk)
        ]
        if not missing:
            return

        doc_ids = [chunk.get("doc_id", "") for chunk in missing if chunk.get("doc_id")]
        if not doc_ids:
            return

        loader = getattr(document_store, "list_by_doc_ids", None)
        if callable(loader):
            docs = await loader(doc_ids)
        else:
            docs = []
            for doc_id in dict.fromkeys(doc_ids):
                doc = await document_store.get(doc_id)
                if doc:
                    docs.append(doc)

        docs_by_id = {doc["doc_id"]: doc for doc in docs if doc.get("doc_id")}
        if not docs_by_id:
            return

        update_ids: List[str] = []
        update_metadatas: List[dict] = []
        for chunk in chunks:
            if not self._needs_lexical_metadata_backfill(chunk):
                continue
            doc = docs_by_id.get(chunk.get("doc_id", ""))
            if not doc:
                continue
            metadata = dict(chunk.get("metadata") or {})
            filename = doc.get("filename") or metadata.get("filename")
            if filename:
                stem, extension = normalize_filename(filename)
                metadata["filename"] = filename
                metadata["file_stem"] = stem
                metadata["extension"] = extension
            if doc.get("tenant_slug") and not metadata.get("tenant_slug"):
                metadata["tenant_slug"] = doc["tenant_slug"]
            if doc.get("tenant_id") and not metadata.get("tenant_id"):
                metadata["tenant_id"] = doc["tenant_id"]
            if doc.get("kb_id") and not metadata.get("kb_id"):
                metadata["kb_id"] = doc["kb_id"]
            metadata.setdefault("doc_id", chunk.get("doc_id", ""))
            metadata.setdefault("chunk_id", chunk.get("chunk_id", ""))
            metadata.setdefault("chunk_index", chunk.get("chunk_index", 0))
            update_ids.append(chunk.get("chunk_id", ""))
            update_metadatas.append(metadata)

        if not update_ids:
            return

        await vector_store.update_metadata(
            update_ids,
            update_metadatas,
            collection_name=collection_name,
        )
        logger.info("BM25 lexical metadata backfill finished: collection=%s chunks=%d", collection_name, len(update_ids))

    def _needs_lexical_metadata_backfill(self, chunk: Dict) -> bool:
        metadata = chunk.get("metadata") or {}
        required = ("filename", "file_stem", "extension", "kb_id")
        return any(not metadata.get(field) for field in required)
