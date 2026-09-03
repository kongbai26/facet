"""ChromaDB 封装"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidCollectionException, NotFoundError

from app.settings.settings import VectorStoreConfig
from app.utils.runtime_errors import IndexUnavailableError

logger = logging.getLogger(__name__)


def _clean_collection_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]", "_", value or "").lower().strip("_")
    return cleaned or "default"


def bound_collection_name(name: str, max_length: int = 63) -> str:
    """Keep Chroma collection names valid for arbitrary model IDs/paths."""
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", name or "").strip("_") or "rag_default"
    if len(normalized) <= max_length:
        return normalized
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    prefix_length = max_length - len(digest) - 1
    return f"{normalized[:prefix_length].rstrip('_-')}_{digest}"


def get_collection_name(prefix: str, model_name: str, dimension: int | None = None) -> str:
    """生成 collection 名称。

    collection 名称同时携带 embedding 维度，避免同名模型在不同维度下
    复用同一个 Chroma collection。
    """
    clean_model = _clean_collection_part(model_name)
    if dimension is None:
        return bound_collection_name(f"{prefix}_{clean_model}")
    return bound_collection_name(f"{prefix}_{clean_model}_d{int(dimension)}")


def load_cached_embedding_dimension(persist_dir: str, model_name: str) -> int | None:
    cache_file = Path(persist_dir) / ".dimension_cache.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("model") != model_name:
        return None
    try:
        return int(data.get("dimension"))
    except (TypeError, ValueError):
        return None


class VectorStore:
    def __init__(self, config: VectorStoreConfig, model_name: str):
        self._config = config
        self._default_model_name = model_name
        self._client = chromadb.PersistentClient(
            path=config.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collections: dict[str, Any] = {}

    def collection_name(self, model_name: Optional[str] = None, dimension: int | None = None) -> str:
        return get_collection_name(
            self._config.collection_prefix,
            model_name or self._default_model_name,
            dimension=dimension,
        )

    def _collection_cache_key(
        self,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> str:
        if collection_name:
            return collection_name
        return model_name or self._default_model_name

    def _resolve_chroma_collection_name(
        self,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> str:
        if collection_name:
            return collection_name
        return self.collection_name(model_name, dimension=dimension)

    def _get_or_create_collection(self, collection_name: str):
        return self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "rag_managed": "true"},
        )

    def get_collection(
        self,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ):
        # Cache by the resolved physical collection name.  The old key used
        # only model_name when the caller omitted collection_name, causing a
        # same-model/different-dimension request to reuse the wrong handle.
        chroma_name = self._resolve_chroma_collection_name(
            model_name,
            collection_name,
            dimension=dimension,
        )
        if chroma_name not in self._collections:
            self._collections[chroma_name] = self._get_or_create_collection(chroma_name)
            logger.info("ChromaDB collection: %s", chroma_name)
        return self._collections[chroma_name]

    def get_existing_collection(
        self,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ):
        """Read an existing collection without get-or-create side effects.

        Retrieval targets are durable metadata contracts.  If one points at a
        missing physical collection, returning an empty collection would turn
        an infrastructure fault into an ungrounded direct answer.  Resolve it
        from Chroma's catalog on every guarded read so a stale cached handle
        cannot hide an external deletion.
        """
        chroma_name = self._resolve_chroma_collection_name(
            model_name,
            collection_name,
            dimension=dimension,
        )
        try:
            collection = self._client.get_collection(chroma_name)
        except (NotFoundError, InvalidCollectionException) as exc:
            self._collections.pop(chroma_name, None)
            raise IndexUnavailableError(chroma_name) from exc
        self._collections[chroma_name] = collection
        return collection

    async def add(
        self,
        texts: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> None:
        def _add():
            self.get_collection(model_name, collection_name, dimension=dimension).add(
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )

        await asyncio.to_thread(_add)

    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        where: Optional[dict] = None,
        include_vectors: bool = False,
        include_documents: bool = True,
        include_metadatas: bool = True,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
        require_existing: bool = False,
    ) -> dict:
        def _query():
            collection = (
                self.get_existing_collection(model_name, collection_name, dimension=dimension)
                if require_existing
                else self.get_collection(model_name, collection_name, dimension=dimension)
            )
            # Chroma logs a warning when n_results exceeds a small collection.
            # Cap the request locally so evaluation and empty/small tenants do
            # not produce noisy non-actionable warnings.
            try:
                collection_size = int(collection.count())
            except (AttributeError, TypeError, ValueError):
                collection_size = 0
            if collection_size <= 0:
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            requested_top_k = min(max(1, int(top_k)), collection_size)
            include = ["distances"]
            if include_vectors:
                include.append("embeddings")
            if include_documents:
                include.append("documents")
            if include_metadatas:
                include.append("metadatas")
            kwargs: Dict[str, Any] = {
                "query_embeddings": [vector],
                "n_results": requested_top_k,
                "include": include,
            }
            if where:
                kwargs["where"] = where
            try:
                return collection.query(**kwargs)
            except (NotFoundError, InvalidCollectionException) as exc:
                chroma_name = self._resolve_chroma_collection_name(
                    model_name,
                    collection_name,
                    dimension=dimension,
                )
                self._collections.pop(chroma_name, None)
                raise IndexUnavailableError(chroma_name) from exc

        return await asyncio.to_thread(_query)

    async def get_all_chunks(
        self,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        where: Optional[dict] = None,
        dimension: int | None = None,
    ) -> dict:
        """Fetch all chunk records from a collection."""

        def _get():
            collection = self.get_collection(model_name, collection_name, dimension=dimension)
            kwargs: Dict[str, Any] = {
                "include": ["documents", "metadatas"],
            }
            if where:
                kwargs["where"] = where
            return collection.get(**kwargs)

        return await asyncio.to_thread(_get)

    async def delete_by_doc_id(
        self,
        doc_id: str,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> None:
        def _delete():
            self.get_collection(model_name, collection_name, dimension=dimension).delete(where={"doc_id": doc_id})

        await asyncio.to_thread(_delete)

    async def delete_by_doc_id_if_exists(
        self,
        doc_id: str,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> bool:
        """Delete a document only when the physical collection already exists.

        ``delete_by_doc_id`` intentionally uses ``get_collection`` for the
        normal ingest path, which creates a collection on first use.  Cleanup
        code must never have that side effect: retrying cleanup for an absent
        legacy collection otherwise recreates an empty collection on disk.
        """

        def _delete() -> bool:
            chroma_name = self._resolve_chroma_collection_name(
                model_name,
                collection_name,
                dimension=dimension,
            )
            try:
                collection = self._client.get_collection(chroma_name)
            except Exception:
                return False
            collection.delete(where={"doc_id": doc_id})
            self._collections.pop(chroma_name, None)
            return True

        return await asyncio.to_thread(_delete)

    async def count(
        self,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> int:
        return await asyncio.to_thread(self.get_collection(model_name, collection_name, dimension=dimension).count)

    async def count_by_doc_id(
        self,
        doc_id: str,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> int:
        """查询指定文档的向量数量"""

        def _count():
            result = self.get_collection(model_name, collection_name, dimension=dimension).get(where={"doc_id": doc_id})
            return len(result["ids"])

        return await asyncio.to_thread(_count)

    # ===================== Vector API 新增方法 =====================

    async def upsert(
        self,
        ids: List[str],
        vectors: Optional[List[List[float]]] = None,
        metadatas: Optional[List[dict]] = None,
        texts: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> None:
        """Upsert 向量：ID 已存在则更新，不存在则插入"""

        def _upsert():
            self.get_collection(model_name, collection_name, dimension=dimension).upsert(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=texts,
            )

        await asyncio.to_thread(_upsert)

    async def get_by_ids(
        self,
        ids: List[str],
        include_vectors: bool = False,
        include_documents: bool = True,
        include_metadatas: bool = True,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> dict:
        """按 ID 列表获取向量"""

        def _get():
            include = []
            if include_vectors:
                include.append("embeddings")
            if include_documents:
                include.append("documents")
            if include_metadatas:
                include.append("metadatas")
            return self.get_collection(model_name, collection_name, dimension=dimension).get(
                ids=ids,
                include=include or ["metadatas", "documents"],
            )

        return await asyncio.to_thread(_get)

    async def delete_by_ids(
        self,
        ids: List[str],
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> None:
        """按 ID 列表删除向量"""

        def _delete():
            self.get_collection(model_name, collection_name, dimension=dimension).delete(ids=ids)

        await asyncio.to_thread(_delete)

    async def delete(
        self,
        where: dict,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> None:
        """按 metadata filter 删除向量"""

        def _delete():
            self.get_collection(model_name, collection_name, dimension=dimension).delete(where=where)

        await asyncio.to_thread(_delete)

    async def update_metadata(
        self,
        ids: List[str],
        metadatas: List[dict],
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> None:
        """更新向量元数据（不改变向量和文本）"""

        def _update():
            self.get_collection(model_name, collection_name, dimension=dimension).update(ids=ids, metadatas=metadatas)

        await asyncio.to_thread(_update)

    async def peek(
        self,
        limit: int = 10,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> dict:
        """抽样查看集合中的向量"""
        return await asyncio.to_thread(
            lambda: self.get_collection(model_name, collection_name, dimension=dimension).peek(limit=limit)
        )

    async def get_collection_info(
        self,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> dict:
        """获取 collection 信息（name / count / metric / metadata），不存在则抛异常"""

        def _get():
            chroma_name = self._resolve_chroma_collection_name(model_name, collection_name, dimension=dimension)
            collection = self._client.get_collection(chroma_name)
            md = collection.metadata or {}
            return {
                "name": chroma_name,
                "count": collection.count(),
                "metric": md.get("hnsw:space", "cosine"),
                "metadata": md,
            }

        return await asyncio.to_thread(_get)

    async def collection_exists(
        self,
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: int | None = None,
    ) -> bool:
        """Check physical existence without creating or caching a collection."""

        def _exists() -> bool:
            chroma_name = self._resolve_chroma_collection_name(
                model_name,
                collection_name,
                dimension=dimension,
            )
            try:
                self._client.get_collection(chroma_name)
            except (NotFoundError, InvalidCollectionException):
                return False
            return True

        return await asyncio.to_thread(_exists)

    async def create_collection(
        self,
        user_name: str,
        metric: str = "cosine",
        collection_name: Optional[str] = None,
        *,
        logical_name: Optional[str] = None,
        tenant_slug: Optional[str] = None,
    ) -> None:
        """创建新 collection（支持指定 metric：cosine / l2 / ip）"""

        def _create():
            chroma_name = collection_name or self.collection_name(user_name)
            metadata = {"hnsw:space": metric}
            if logical_name:
                metadata["vector_api_logical_name"] = logical_name
            if tenant_slug:
                metadata["vector_api_tenant_slug"] = tenant_slug
            self._client.create_collection(name=chroma_name, metadata=metadata)

        await asyncio.to_thread(_create)

    async def list_collections(self) -> List[dict]:
        """列出当前 prefix 下所有 collection，返回 [{name, count, metric}]"""

        def _list():
            prefix = self._config.collection_prefix + "_"
            result = []
            for c in self._client.list_collections():
                name = str(c)
                if name.startswith(prefix):
                    collection = self._client.get_collection(name)
                    md = collection.metadata or {}
                    logical_name = md.get("vector_api_logical_name")
                    result.append({
                        "name": str(logical_name or name[len(prefix):]),
                        "count": collection.count(),
                        "metric": md.get("hnsw:space", "cosine"),
                        "metadata": md,
                    })
            return result

        return await asyncio.to_thread(_list)

    async def list_physical_collections(self) -> List[dict]:
        """列出 Chroma 中的物理 collection，保留真实名称和管理元数据。"""

        def _list() -> List[dict]:
            result: List[dict] = []
            for item in self._client.list_collections():
                name = str(item)
                collection = self._client.get_collection(name)
                metadata = dict(collection.metadata or {})
                result.append({
                    "name": name,
                    "count": int(collection.count()),
                    "metadata": metadata,
                })
            return result

        return await asyncio.to_thread(_list)

    def _vector_segment_ids(self, collection_name: str | None = None) -> set[str] | None:
        """Read Chroma's vector segment catalog without mutating it.

        Chroma 0.6.x removes catalog rows but can leave HNSW directories behind.
        The catalog is used only to identify directories that are still live;
        if the schema is unavailable, the caller must skip physical sweeping.
        """
        db_path = Path(self._config.persist_dir) / "chroma.sqlite3"
        if not db_path.exists():
            return None
        try:
            with sqlite3.connect(db_path) as connection:
                if collection_name is None:
                    rows = connection.execute(
                        "SELECT id FROM segments WHERE scope = 'VECTOR'"
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """SELECT segments.id
                           FROM segments
                           JOIN collections ON collections.id = segments.collection
                           WHERE segments.scope = 'VECTOR' AND collections.name = ?""",
                        (collection_name,),
                    ).fetchall()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("无法读取 Chroma segment catalog，跳过目录回收: %s", exc)
            return None
        return {str(row[0]) for row in rows if row and row[0]}

    @staticmethod
    def _remove_segment_directories(persist_dir: str, segment_ids: set[str]) -> int:
        """Remove only catalog-derived UUID directories and return byte count."""
        root = Path(persist_dir).resolve()
        removed_bytes = 0
        for segment_id in segment_ids:
            directory = (root / segment_id).resolve()
            if directory.parent != root or not directory.is_dir():
                continue
            try:
                size = sum(
                    path.stat().st_size
                    for path in directory.rglob("*")
                    if path.is_file()
                )
                shutil.rmtree(directory)
                removed_bytes += size
            except OSError as exc:
                logger.warning("Chroma segment 目录回收失败: %s: %s", directory, exc)
        return removed_bytes

    async def cleanup_orphaned_storage(
        self,
        *,
        protected_collection_names: set[str],
        orphan_grace_seconds: int = 86_400,
    ) -> dict:
        """Reclaim unreferenced managed collections and stale HNSW directories.

        User-owned Vector API collections are never removed here.  Only
        collections explicitly marked ``rag_managed`` and absent from the
        lifecycle's protected set are candidates, and non-empty collections
        are left untouched for safety.  HNSW directories are removed only when
        they are absent from Chroma's live VECTOR segment catalog and older
        than the configured grace period.
        """

        def _cleanup() -> dict:
            protected = {str(name) for name in protected_collection_names if name}
            deleted_collections = 0
            failed_collections = 0
            skipped_nonempty_collections = 0
            removed_bytes = 0

            for item in self._list_physical_collections_sync():
                name = item["name"]
                metadata = item.get("metadata") or {}
                is_rag_managed = str(metadata.get("rag_managed", "")).lower() == "true"
                is_vector_api_owned = any(
                    metadata.get(key)
                    for key in ("vector_api_logical_name", "vector_api_tenant_slug")
                )
                if (
                    name in protected
                    or not is_rag_managed
                    or is_vector_api_owned
                ):
                    continue
                if int(item.get("count") or 0) > 0:
                    skipped_nonempty_collections += 1
                    logger.warning(
                        "发现未登记但非空的 RAG collection，暂不自动删除: %s", name
                    )
                    continue
                segment_ids = self._vector_segment_ids(name) or set()
                try:
                    self._client.delete_collection(name)
                except NotFoundError:
                    pass
                except Exception as exc:
                    failed_collections += 1
                    logger.warning("孤立 RAG collection 回收失败，将重试: %s: %s", name, exc)
                    continue
                removed_bytes += self._remove_segment_directories(
                    self._config.persist_dir,
                    segment_ids,
                )
                self._collections.pop(name, None)
                deleted_collections += 1

            live_segment_ids = self._vector_segment_ids()
            root = Path(self._config.persist_dir).resolve()
            now = time.time()
            grace = max(0, int(orphan_grace_seconds))
            orphan_segment_ids: set[str] = set()
            if live_segment_ids is not None and root.exists():
                for directory in root.iterdir():
                    if not directory.is_dir() or directory.name in live_segment_ids:
                        continue
                    # Only UUID-like directory names can be Chroma HNSW
                    # segments; leave unrelated runtime files untouched.
                    if not re.fullmatch(
                        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                        directory.name,
                        re.IGNORECASE,
                    ):
                        continue
                    try:
                        if now - directory.stat().st_mtime < grace:
                            continue
                    except OSError:
                        continue
                    orphan_segment_ids.add(directory.name)
            removed_orphan_bytes = self._remove_segment_directories(
                self._config.persist_dir,
                orphan_segment_ids,
            )
            return {
                "deleted_collections": deleted_collections,
                "failed_collections": failed_collections,
                "skipped_nonempty_collections": skipped_nonempty_collections,
                "deleted_orphan_segments": len(orphan_segment_ids),
                "removed_bytes": removed_bytes + removed_orphan_bytes,
            }

        return await asyncio.to_thread(_cleanup)

    def _list_physical_collections_sync(self) -> List[dict]:
        result: List[dict] = []
        for item in self._client.list_collections():
            name = str(item)
            collection = self._client.get_collection(name)
            result.append({
                "name": name,
                "count": int(collection.count()),
                "metadata": dict(collection.metadata or {}),
            })
        return result

    async def delete_collection_by_name(
        self,
        user_name: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """删除 collection（按用户名称），同时清理内部缓存"""

        def _delete():
            chroma_name = collection_name or self.collection_name(user_name)
            segment_ids = self._vector_segment_ids(chroma_name) or set()
            try:
                self._client.delete_collection(chroma_name)
            except NotFoundError:
                # The catalog is already gone; still remove a segment
                # directory that was left by an earlier Chroma delete.
                pass
            self._remove_segment_directories(self._config.persist_dir, segment_ids)
            try:
                self._client.get_collection(chroma_name)
            except (NotFoundError, InvalidCollectionException):
                pass
            except Exception as exc:
                # The delete already committed.  A transient verification
                # failure must not turn a successful API operation into a
                # false error; the next maintenance pass will re-audit it.
                logger.warning("Chroma collection 删除后校验延迟: %s: %s", chroma_name, exc)
            else:
                raise RuntimeError(f"Chroma collection physical delete not verified: {chroma_name}")

        await asyncio.to_thread(_delete)
        # 清理缓存
        self._collections.pop(collection_name or user_name, None)
        self._collections.pop(user_name, None)
