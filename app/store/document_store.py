"""SQLite 元数据存储"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from app.store.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

UPLOAD_CONFLICT_STATUSES = (
    "ready",
    "processing",
    "reindex_queued",
    "reindexing",
    "deleting",
    "delete_failed",
)

UPLOAD_CONFLICT_STATUS_SQL = ", ".join(f"'{status}'" for status in UPLOAD_CONFLICT_STATUSES)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    tenant_id TEXT,
    tenant_slug TEXT,
    kb_id TEXT,
    filename TEXT NOT NULL,
    file_size INTEGER,
    chunks_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'processing',
    status_reason TEXT NOT NULL DEFAULT '',
    error_message TEXT,
    embedding_model TEXT,
    embedding_endpoint TEXT,
    embedding_context_window INTEGER,
    embedding_dimension INTEGER,
    index_pipeline_version TEXT NOT NULL DEFAULT 'legacy',
    content_hash TEXT,
    source_revision INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CONTENT_HASH_INDEX_SQL = """
DROP INDEX IF EXISTS idx_documents_content_hash_model_unique;
DROP INDEX IF EXISTS idx_documents_tenant_content_hash_model_unique;
DROP INDEX IF EXISTS idx_documents_tenant_kb_content_hash_model_unique;
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_tenant_kb_content_hash_model_unique
ON documents(tenant_id, kb_id, content_hash, embedding_model, embedding_dimension)
WHERE tenant_id IS NOT NULL
  AND tenant_id != ''
  AND kb_id IS NOT NULL
  AND kb_id != ''
  AND content_hash IS NOT NULL
  AND content_hash != ''
  AND embedding_model IS NOT NULL
  AND embedding_model != ''
  AND embedding_dimension IS NOT NULL
  AND status IN (%s);
""" % UPLOAD_CONFLICT_STATUS_SQL

LEGACY_CONTENT_HASH_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_documents_content_hash_model_lookup
ON documents(content_hash, embedding_model)
WHERE content_hash IS NOT NULL
  AND content_hash != ''
  AND embedding_model IS NOT NULL
  AND embedding_model != '';
"""

STATUS_PRIORITY_SQL = """
CASE status
    WHEN 'ready' THEN 1
    WHEN 'reindexing' THEN 2
    WHEN 'reindex_queued' THEN 3
    WHEN 'processing' THEN 4
    WHEN 'deleting' THEN 5
    WHEN 'failed' THEN 6
    WHEN 'delete_failed' THEN 7
    ELSE 8
END
"""

VALID_STATUSES = {
    "processing",
    "ready",
    "failed",
    "reindex_queued",
    "reindexing",
    "deleting",
    "delete_failed",
}

VALID_TRANSITIONS = {
    "processing": {"failed", "ready"},
    "ready": {"processing", "reindexing"},
    "failed": {"processing", "reindex_queued", "reindexing", "deleting"},
    "reindex_queued": {"ready", "failed"},
    "reindexing": {"reindex_queued"},
    "deleting": {"failed", "delete_failed", "reindex_queued"},
    "delete_failed": {"deleting"},
}

VALID_PREVIOUS_STATUSES = {
    destination: {
        source for source, destinations in VALID_TRANSITIONS.items()
        if destination in destinations
    }
    for destination in VALID_STATUSES
}


class DocumentStore(SQLiteStore):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(CREATE_TABLE_SQL)
            self._migrate_schema(conn)
            conn.executescript(CONTENT_HASH_INDEX_SQL)
            conn.executescript(LEGACY_CONTENT_HASH_INDEX_SQL)
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
                CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_documents_tenant_id ON documents(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_documents_tenant_kb_id ON documents(tenant_id, kb_id);
                """
            )

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(documents)").fetchall()
        }
        if "tenant_id" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN tenant_id TEXT")
        if "tenant_slug" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN tenant_slug TEXT")
        if "kb_id" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN kb_id TEXT")
        if "content_hash" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")
        if "source_revision" not in columns:
            conn.execute(
                "ALTER TABLE documents ADD COLUMN source_revision INTEGER NOT NULL DEFAULT 0"
            )
        if "embedding_dimension" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN embedding_dimension INTEGER")
        if "embedding_endpoint" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN embedding_endpoint TEXT")
        if "embedding_context_window" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN embedding_context_window INTEGER")
        if "status_reason" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN status_reason TEXT NOT NULL DEFAULT ''")
        if "index_pipeline_version" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN index_pipeline_version TEXT NOT NULL DEFAULT 'legacy'")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_tenant_id ON documents(tenant_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_tenant_kb_id ON documents(tenant_id, kb_id)"
        )

    async def create(
        self,
        doc_id: str,
        filename: str,
        file_size: int,
        content_hash: str = "",
        embedding_model: str = "",
        embedding_dimension: int | None = None,
        tenant_id: str | None = None,
        tenant_slug: str | None = None,
        kb_id: str | None = None,
    ) -> None:
        def _create():
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO documents
                       (doc_id, tenant_id, tenant_slug, kb_id, filename, file_size,
                        embedding_model, embedding_dimension, content_hash, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'processing')""",
                    (
                        doc_id,
                        tenant_id,
                        tenant_slug,
                        kb_id,
                        filename,
                        file_size,
                        embedding_model,
                        embedding_dimension,
                        content_hash,
                    ),
                )
        await self._execute_write(_create)

    async def claim_upload(
        self,
        doc_id: str,
        filename: str,
        file_size: int,
        content_hash: str,
        embedding_model: str,
        embedding_dimension: int | None = None,
        tenant_id: str | None = None,
        tenant_slug: str | None = None,
        kb_id: str | None = None,
    ) -> dict:
        """Atomically claim a content hash for upload/ingest."""
        def _claim():
            with self._conn() as conn:
                clauses = ["content_hash = ?"]
                query_params: list[object] = [content_hash]
                if tenant_id is None:
                    clauses.append("(tenant_id IS NULL OR tenant_id = '')")
                else:
                    clauses.append("tenant_id = ?")
                    query_params.append(tenant_id)
                if kb_id is None:
                    clauses.append("(kb_id IS NULL OR kb_id = '')")
                else:
                    # Startup backfills legacy rows to the default KB.  Keep
                    # this narrow compatibility branch for callers that run
                    # a store before bootstrap; after bootstrap all scoped
                    # uploads are isolated by their explicit kb_id.
                    clauses.append("(kb_id = ? OR kb_id IS NULL OR kb_id = '')")
                    query_params.append(kb_id)
                clauses.append(f"status IN ({UPLOAD_CONFLICT_STATUS_SQL})")
                row = conn.execute(
                    f"""SELECT * FROM documents
                        WHERE {' AND '.join(clauses)}
                        ORDER BY {STATUS_PRIORITY_SQL}, created_at DESC
                        LIMIT 1""",
                    tuple(query_params),
                ).fetchone()
                existing = dict(row) if row else None

                if existing:
                    status = existing["status"]
                    if status == "ready":
                        return {"action": "ready", "document": existing}
                    return {"action": "conflict", "document": existing}

                try:
                    conn.execute(
                        """INSERT INTO documents
                           (doc_id, tenant_id, tenant_slug, kb_id, filename, file_size,
                            embedding_model, embedding_dimension, content_hash, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'processing')""",
                        (
                            doc_id,
                            tenant_id,
                            tenant_slug,
                            kb_id,
                            filename,
                            file_size,
                            embedding_model,
                            embedding_dimension,
                            content_hash,
                        ),
                    )
                    return {
                        "action": "created",
                        "document": {
                            "doc_id": doc_id,
                            "tenant_id": tenant_id,
                            "tenant_slug": tenant_slug,
                            "kb_id": kb_id,
                            "filename": filename,
                            "file_size": file_size,
                            "embedding_model": embedding_model,
                            "embedding_dimension": embedding_dimension,
                            "content_hash": content_hash,
                            "status": "processing",
                        },
                    }
                except sqlite3.IntegrityError:
                    # 并发同时插入，回退到查询已有记录
                    row = conn.execute(
                        f"""SELECT * FROM documents
                            WHERE {' AND '.join(clauses)}
                            ORDER BY {STATUS_PRIORITY_SQL}, created_at DESC
                            LIMIT 1""",
                        tuple(query_params),
                    ).fetchone()
                    existing = dict(row) if row else None
                    if not existing:
                        raise
                    status = existing["status"]
                    if status == "ready":
                        return {"action": "ready", "document": existing}
                    return {"action": "conflict", "document": existing}

        return await self._execute_write(_claim)

    async def reset_for_retry(
        self,
        doc_id: str,
        filename: str,
        file_size: int,
        content_hash: str,
        embedding_model: str = "",
        embedding_dimension: int | None = None,
        tenant_id: str | None = None,
        tenant_slug: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        def _reset():
            with self._conn() as conn:
                conn.execute(
                    """UPDATE documents
                       SET tenant_id = ?, tenant_slug = ?, filename = ?, file_size = ?,
                           embedding_model = ?, embedding_dimension = ?, content_hash = ?,
                           status = 'processing', status_reason = '', error_message = '', chunks_count = 0, updated_at = ?
                       WHERE doc_id = ?""",
                    (tenant_id, tenant_slug, filename, file_size, embedding_model, embedding_dimension, content_hash, now, doc_id),
                )

        await self._execute_write(_reset)

    async def restore_snapshot(
        self,
        doc_id: str,
        snapshot: dict,
        *,
        status: str,
        error_message: str = "",
        chunks_count: Optional[int] = None,
        status_reason: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        def _restore():
            with self._conn() as conn:
                columns = [
                    "tenant_id = ?",
                    "tenant_slug = ?",
                    "filename = ?",
                    "file_size = ?",
                    "content_hash = ?",
                    "embedding_model = ?",
                    "embedding_endpoint = ?",
                    "embedding_context_window = ?",
                    "embedding_dimension = ?",
                    "status = ?",
                    "status_reason = ?",
                    "error_message = ?",
                    "updated_at = ?",
                ]
                params: list[object] = [
                    snapshot.get("tenant_id"),
                    snapshot.get("tenant_slug"),
                    snapshot.get("filename"),
                    snapshot.get("file_size"),
                    snapshot.get("content_hash"),
                    snapshot.get("embedding_model"),
                    snapshot.get("embedding_endpoint"),
                    snapshot.get("embedding_context_window"),
                    snapshot.get("embedding_dimension"),
                    status,
                    status_reason if status_reason is not None else snapshot.get("status_reason") or "",
                    error_message,
                    now,
                ]
                if chunks_count is not None:
                    columns.insert(9, "chunks_count = ?")
                    params.insert(9, chunks_count)
                cursor = conn.execute(
                    f"""UPDATE documents
                       SET {", ".join(columns)}
                       WHERE doc_id = ?""",
                    (*params, doc_id),
                )
                if cursor.rowcount == 0:
                    raise KeyError(f"Document not found: {doc_id}")

        await self._execute_write(_restore)

    async def get(self, doc_id: str, tenant_id: str | None = None) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                if tenant_id is None:
                    row = conn.execute(
                        "SELECT * FROM documents WHERE doc_id = ?",
                        (doc_id,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT * FROM documents WHERE doc_id = ? AND tenant_id = ?",
                        (doc_id, tenant_id),
                    ).fetchone()
                return dict(row) if row else None
        return await self._execute_read(_get)

    async def list_by_doc_ids(
        self,
        doc_ids: Iterable[str],
        tenant_id: str | None = None,
        status: str | None = None,
    ) -> List[dict]:
        """批量按 doc_id 查询文档。"""
        unique_ids = [doc_id for doc_id in dict.fromkeys(doc_ids) if doc_id]
        if not unique_ids:
            return []

        placeholders = ",".join("?" for _ in unique_ids)
        params: list[object] = list(unique_ids)
        conditions = [f"doc_id IN ({placeholders})"]
        if tenant_id is not None:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        def _list():
            with self._conn() as conn:
                row = conn.execute(
                    f"SELECT * FROM documents WHERE {' AND '.join(conditions)}",
                    tuple(params),
                ).fetchall()
                return [dict(item) for item in row]

        return await self._execute_read(_list)

    async def list_all(self, tenant_id: str | None = None) -> List[dict]:
        def _list():
            with self._conn() as conn:
                if tenant_id is None:
                    rows = conn.execute(
                        "SELECT * FROM documents ORDER BY created_at DESC"
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM documents WHERE tenant_id = ? ORDER BY created_at DESC",
                        (tenant_id,),
                    ).fetchall()
                return [dict(r) for r in rows]
        return await self._execute_read(_list)

    async def tenant_revision(self, tenant_id: str) -> dict:
        """Return a lightweight revision for Agent/UI document synchronization."""
        def _revision() -> dict:
            with self._conn() as conn:
                row = conn.execute(
                    """SELECT COUNT(*) AS item_count, COALESCE(MAX(updated_at), '') AS updated_at
                       FROM documents WHERE tenant_id = ?""",
                    (tenant_id,),
                ).fetchone()
                return {
                    "item_count": int(row["item_count"]),
                    "updated_at": str(row["updated_at"] or ""),
                }

        return await self._execute_read(_revision)

    async def list_by_knowledge_base(
        self,
        kb_id: str,
        *,
        tenant_id: str | None = None,
        status: str | None = None,
    ) -> List[dict]:
        conditions = ["kb_id = ?"]
        params: list[object] = [kb_id]
        if tenant_id is not None:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        def _list():
            with self._conn() as conn:
                rows = conn.execute(
                    f"SELECT * FROM documents WHERE {' AND '.join(conditions)} ORDER BY created_at DESC",
                    tuple(params),
                ).fetchall()
                return [dict(row) for row in rows]

        return await self._execute_read(_list)

    async def list_ready_doc_ids(self, kb_id: str, *, tenant_id: str | None = None) -> list[str]:
        documents = await self.list_by_knowledge_base(kb_id, tenant_id=tenant_id, status="ready")
        return [str(document["doc_id"]) for document in documents]

    async def assign_missing_knowledge_base(self, kb_id: str, *, tenant_id: str | None = None) -> int:
        """Backfill legacy documents into the tenant's default knowledge base."""
        def _assign():
            with self._conn() as conn:
                if tenant_id is None:
                    cursor = conn.execute("UPDATE documents SET kb_id = ? WHERE kb_id IS NULL OR kb_id = ''", (kb_id,))
                else:
                    cursor = conn.execute(
                        "UPDATE documents SET kb_id = ? WHERE tenant_id = ? AND (kb_id IS NULL OR kb_id = '')",
                        (kb_id, tenant_id),
                    )
                return cursor.rowcount

        return await self._execute_write(_assign)

    async def has_ready_documents(self, tenant_id: str | None = None) -> bool:
        def _check():
            with self._conn() as conn:
                if tenant_id is None:
                    row = conn.execute(
                        "SELECT 1 FROM documents WHERE status = 'ready' LIMIT 1"
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT 1 FROM documents WHERE tenant_id = ? AND status = 'ready' LIMIT 1",
                        (tenant_id,),
                    ).fetchone()
                return row is not None
        return await self._execute_read(_check)

    async def update_status(
        self,
        doc_id: str,
        status: str,
        error_message: str = "",
        chunks_count: Optional[int] = None,
        status_reason: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        def _update():
            with self._conn() as conn:
                reason_value = status_reason
                if chunks_count is not None:
                    if reason_value is None:
                        conn.execute(
                            """UPDATE documents
                               SET status = ?, error_message = ?, chunks_count = ?, updated_at = ?
                               WHERE doc_id = ?""",
                            (status, error_message, chunks_count, now, doc_id),
                        )
                    else:
                        conn.execute(
                            """UPDATE documents
                               SET status = ?, status_reason = ?, error_message = ?, chunks_count = ?, updated_at = ?
                               WHERE doc_id = ?""",
                            (status, reason_value, error_message, chunks_count, now, doc_id),
                        )
                else:
                    if reason_value is None:
                        conn.execute(
                            """UPDATE documents
                               SET status = ?, error_message = ?, updated_at = ?
                               WHERE doc_id = ?""",
                            (status, error_message, now, doc_id),
                        )
                    else:
                        conn.execute(
                            """UPDATE documents
                               SET status = ?, status_reason = ?, error_message = ?, updated_at = ?
                               WHERE doc_id = ?""",
                            (status, reason_value, error_message, now, doc_id),
                        )
        await self._execute_write(_update)

    async def update_index_pipeline_version(self, doc_id: str, version: str) -> None:
        now = datetime.now(timezone.utc).isoformat()

        def _update():
            with self._conn() as conn:
                cursor = conn.execute(
                    """UPDATE documents
                       SET index_pipeline_version = ?, updated_at = ?
                       WHERE doc_id = ?""",
                    (version, now, doc_id),
                )
                if cursor.rowcount == 0:
                    raise KeyError(f"Document not found: {doc_id}")

        await self._execute_write(_update)

    async def update_status_if(
        self,
        doc_id: str,
        allowed_statuses: Iterable[str],
        status: str,
        error_message: str = "",
        chunks_count: Optional[int] = None,
        status_reason: Optional[str] = None,
    ) -> bool:
        """Update status only when the current status matches one of allowed_statuses."""
        allowed = tuple(allowed_statuses)
        if not allowed:
            return False
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in allowed)

        def _update():
            with self._conn() as conn:
                reason_value = status_reason
                if chunks_count is not None:
                    if reason_value is None:
                        cursor = conn.execute(
                            f"""UPDATE documents
                                SET status = ?, error_message = ?, chunks_count = ?, updated_at = ?
                                WHERE doc_id = ? AND status IN ({placeholders})""",
                            (status, error_message, chunks_count, now, doc_id, *allowed),
                        )
                    else:
                        cursor = conn.execute(
                            f"""UPDATE documents
                                SET status = ?, status_reason = ?, error_message = ?, chunks_count = ?, updated_at = ?
                                WHERE doc_id = ? AND status IN ({placeholders})""",
                            (status, reason_value, error_message, chunks_count, now, doc_id, *allowed),
                        )
                else:
                    if reason_value is None:
                        cursor = conn.execute(
                            f"""UPDATE documents
                                SET status = ?, error_message = ?, updated_at = ?
                                WHERE doc_id = ? AND status IN ({placeholders})""",
                            (status, error_message, now, doc_id, *allowed),
                        )
                    else:
                        cursor = conn.execute(
                            f"""UPDATE documents
                                SET status = ?, status_reason = ?, error_message = ?, updated_at = ?
                                WHERE doc_id = ? AND status IN ({placeholders})""",
                            (status, reason_value, error_message, now, doc_id, *allowed),
                        )
                return cursor.rowcount > 0

        return await self._execute_write(_update)

    async def transition_status(
        self,
        doc_id: str,
        new_status: str,
        *,
        expected_statuses: Optional[Iterable[str]] = None,
        error_message: str = "",
        chunks_count: Optional[int] = None,
        status_reason: Optional[str] = None,
    ) -> dict:
        """集中式文档状态迁移。

        返回结构：
            {"ok": True, "previous": old_status, "current_status": new_status, "document": {...}}
            {"ok": False, "previous": None, "current_status": current_status, "document": {...}}
        """
        if new_status not in VALID_STATUSES:
            raise ValueError(f"非法文档状态: {new_status}")

        allowed = (
            set(expected_statuses)
            if expected_statuses
            else VALID_PREVIOUS_STATUSES.get(new_status, set())
        )
        if not allowed:
            raise ValueError(f"状态 {new_status} 没有定义允许的前置状态")

        doc = await self.get(doc_id)
        if not doc:
            return {"ok": False, "previous": None, "current_status": None, "document": None}

        previous = doc.get("status")
        if previous not in allowed:
            return {
                "ok": False,
                "previous": previous,
                "current_status": previous,
                "document": doc,
            }

        updated = await self.update_status_if(
            doc_id,
            allowed,
            new_status,
            error_message=error_message,
            chunks_count=chunks_count,
            status_reason=status_reason,
        )
        if not updated:
            current = await self.get(doc_id)
            return {
                "ok": False,
                "previous": previous,
                "current_status": current.get("status") if current else None,
                "document": current,
            }

        doc["status"] = new_status
        doc["error_message"] = error_message
        if status_reason is not None:
            doc["status_reason"] = status_reason
        if chunks_count is not None:
            doc["chunks_count"] = chunks_count
        return {
            "ok": True,
            "previous": previous,
            "current_status": new_status,
            "document": doc,
        }

    async def delete(self, doc_id: str) -> None:
        def _delete():
            with self._conn() as conn:
                cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                if cursor.rowcount == 0:
                    raise KeyError(f"Document not found: {doc_id}")
        await self._execute_write(_delete)

    async def get_by_content_hash(
        self,
        content_hash: str,
        embedding_model: Optional[str] = None,
        embedding_dimension: int | None = None,
        tenant_id: str | None = None,
    ) -> Optional[dict]:
        """按内容哈希查找优先级最高的现存文档。"""
        def _get():
            with self._conn() as conn:
                params = [content_hash]
                model_filter = ""
                if embedding_model is not None:
                    model_filter = "AND embedding_model = ?"
                    params.append(embedding_model)
                dimension_filter = ""
                if embedding_dimension is not None:
                    dimension_filter = "AND embedding_dimension = ?"
                    params.append(embedding_dimension)
                tenant_filter = ""
                if tenant_id is not None:
                    tenant_filter = "AND tenant_id = ?"
                    params.append(tenant_id)
                row = conn.execute(
                    f"""SELECT * FROM documents
                       WHERE content_hash = ? {model_filter} {dimension_filter} {tenant_filter}
                       ORDER BY {STATUS_PRIORITY_SQL},
                       created_at DESC
                       LIMIT 1""",
                    tuple(params),
                ).fetchone()
                return dict(row) if row else None
        return await self._execute_read(_get)

    async def list_by_content_hash_model(
        self,
        content_hash: str,
        embedding_model: str,
        embedding_dimension: int | None = None,
        tenant_id: str | None = None,
    ) -> List[dict]:
        def _list():
            with self._conn() as conn:
                dimension_filter = ""
                params: list[object] = [content_hash, embedding_model]
                if embedding_dimension is not None:
                    dimension_filter = " AND embedding_dimension = ?"
                    params.append(embedding_dimension)
                if tenant_id is None:
                    rows = conn.execute(
                        f"""SELECT * FROM documents
                            WHERE content_hash = ? AND embedding_model = ?
                            {dimension_filter}
                            ORDER BY {STATUS_PRIORITY_SQL}, created_at DESC""",
                        tuple(params),
                    ).fetchall()
                else:
                    params_with_tenant = params + [tenant_id]
                    rows = conn.execute(
                        f"""SELECT * FROM documents
                            WHERE content_hash = ? AND embedding_model = ? AND tenant_id = ?
                            {dimension_filter}
                            ORDER BY {STATUS_PRIORITY_SQL}, created_at DESC""",
                        tuple(params_with_tenant),
                    ).fetchall()
                return [dict(row) for row in rows]
        return await self._execute_read(_list)

    async def set_content_hash(
        self,
        doc_id: str,
        content_hash: str,
        embedding_model: Optional[str] = None,
        embedding_dimension: int | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        def _set():
            with self._conn() as conn:
                if embedding_model is None:
                    cursor = conn.execute(
                        """UPDATE documents
                           SET content_hash = ?, updated_at = ?
                           WHERE doc_id = ?""",
                        (content_hash, now, doc_id),
                    )
                else:
                    if embedding_dimension is None:
                        cursor = conn.execute(
                            """UPDATE documents
                               SET content_hash = ?, embedding_model = ?, updated_at = ?
                               WHERE doc_id = ?""",
                            (content_hash, embedding_model, now, doc_id),
                        )
                    else:
                        cursor = conn.execute(
                            """UPDATE documents
                               SET content_hash = ?, embedding_model = ?, embedding_dimension = ?, updated_at = ?
                               WHERE doc_id = ?""",
                            (content_hash, embedding_model, embedding_dimension, now, doc_id),
                        )
                if cursor.rowcount == 0:
                    raise KeyError(f"Document not found: {doc_id}")

        await self._execute_write(_set)

    async def bump_source_revision(self, doc_id: str) -> int:
        """Record a deliberate source rebuild even when file bytes are unchanged.

        ``updated_at`` also changes for ordinary status transitions, so it is
        not a safe source snapshot version. Candidate generations use this
        counter together with the content hash instead.
        """
        now = datetime.now(timezone.utc).isoformat()

        def _bump() -> int:
            with self._conn() as conn:
                cursor = conn.execute(
                    """UPDATE documents
                       SET source_revision = source_revision + 1, updated_at = ?
                       WHERE doc_id = ?""",
                    (now, doc_id),
                )
                if cursor.rowcount == 0:
                    raise KeyError(f"Document not found: {doc_id}")
                row = conn.execute(
                    "SELECT source_revision FROM documents WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()
                return int(row["source_revision"])

        return await self._execute_write(_bump)

    async def update_embedding_model(
        self,
        doc_id: str,
        embedding_model: str,
        embedding_dimension: int | None = None,
        embedding_endpoint: str | None = None,
        embedding_context_window: int | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()

        def _update():
            with self._conn() as conn:
                if embedding_dimension is None:
                    cursor = conn.execute(
                        """UPDATE documents
                           SET embedding_model = ?, embedding_endpoint = COALESCE(?, embedding_endpoint),
                               embedding_context_window = COALESCE(?, embedding_context_window), updated_at = ?
                           WHERE doc_id = ?""",
                        (embedding_model, embedding_endpoint, embedding_context_window, now, doc_id),
                    )
                else:
                    cursor = conn.execute(
                        """UPDATE documents
                           SET embedding_model = ?, embedding_dimension = ?,
                               embedding_endpoint = COALESCE(?, embedding_endpoint),
                               embedding_context_window = COALESCE(?, embedding_context_window), updated_at = ?
                           WHERE doc_id = ?""",
                        (embedding_model, embedding_dimension, embedding_endpoint,
                         embedding_context_window, now, doc_id),
                    )
                if cursor.rowcount == 0:
                    raise KeyError(f"Document not found: {doc_id}")

        await self._execute_write(_update)

    async def count_by_statuses(
        self,
        statuses: Iterable[str],
        *,
        status_reason: Optional[str] = None,
        tenant_id: str | None = None,
    ) -> int:
        values = tuple(statuses)
        if not values:
            return 0
        placeholders = ",".join("?" for _ in values)

        def _count():
            with self._conn() as conn:
                clauses = [f"status IN ({placeholders})"]
                params: list[object] = list(values)
                if status_reason is not None:
                    clauses.append("status_reason = ?")
                    params.append(status_reason)
                if tenant_id is not None:
                    clauses.append("tenant_id = ?")
                    params.append(tenant_id)
                row = conn.execute(
                    f"SELECT COUNT(1) AS count FROM documents WHERE {' AND '.join(clauses)}",
                    tuple(params),
                ).fetchone()
                return int(row["count"]) if row else 0

        return await self._execute_read(_count)

    async def ensure_unique_content_index(self) -> None:
        def _ensure():
            with self._conn() as conn:
                conn.executescript(CONTENT_HASH_INDEX_SQL)
                conn.executescript(LEGACY_CONTENT_HASH_INDEX_SQL)
        await self._execute_write(_ensure)

    async def assign_missing_tenant(self, tenant_id: str, tenant_slug: str) -> int:
        now = datetime.now(timezone.utc).isoformat()

        def _assign():
            with self._conn() as conn:
                cursor = conn.execute(
                    """UPDATE documents
                       SET tenant_id = ?, tenant_slug = ?, updated_at = ?
                       WHERE tenant_id IS NULL OR tenant_id = '' OR tenant_slug IS NULL OR tenant_slug = ''""",
                    (tenant_id, tenant_slug, now),
                )
                return cursor.rowcount

        return await self._execute_write(_assign)
