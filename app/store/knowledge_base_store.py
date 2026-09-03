"""Knowledge base metadata storage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.store.sqlite_store import SQLiteStore
from app.store.namespace_store import CREATE_NAMESPACES_TABLE_SQL

CREATE_KNOWLEDGE_BASES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_bases (
    kb_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    namespace_id TEXT NOT NULL,
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    llm_model TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    error_message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (tenant_id, slug),
    FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY(namespace_id) REFERENCES namespaces(namespace_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_knowledge_bases_tenant_id ON knowledge_bases(tenant_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeBaseStore(SQLiteStore):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(CREATE_NAMESPACES_TABLE_SQL)
            conn.executescript(CREATE_KNOWLEDGE_BASES_TABLE_SQL)
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(knowledge_bases)").fetchall()
            }
            if "error_message" not in columns:
                conn.execute(
                    "ALTER TABLE knowledge_bases ADD COLUMN error_message TEXT NOT NULL DEFAULT ''"
                )

    async def create_with_namespace(
        self,
        tenant_id: str,
        *,
        namespace_slug: str,
        kb_slug: str,
        name: str,
        embedding_model: str,
        llm_model: str = "",
        max_namespaces: int | None = None,
    ) -> dict:
        """Create a namespace and KB in one quota-checked SQLite transaction."""
        namespace_id = str(uuid4())
        kb_id = str(uuid4())
        now = _now()

        def _create() -> dict:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                existing_names = conn.execute(
                    "SELECT name FROM knowledge_bases WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchall()
                normalized_name = name.strip().casefold()
                if any(str(row["name"]).strip().casefold() == normalized_name for row in existing_names):
                    raise ValueError("knowledge_base_name_conflict")

                if max_namespaces is not None:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM namespaces WHERE tenant_id = ?",
                        (tenant_id,),
                    ).fetchone()[0]
                    if int(count) >= int(max_namespaces):
                        raise ValueError("namespace_quota_exceeded")

                conn.execute(
                    """INSERT INTO namespaces
                       (namespace_id, tenant_id, slug, name, kind, status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, 'rag', 'active', ?, ?)""",
                    (namespace_id, tenant_id, namespace_slug, name, now, now),
                )
                conn.execute(
                    """INSERT INTO knowledge_bases
                       (kb_id, tenant_id, namespace_id, slug, name, embedding_model, llm_model,
                        status, error_message, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'active', '', ?, ?)""",
                    (
                        kb_id,
                        tenant_id,
                        namespace_id,
                        kb_slug,
                        name,
                        embedding_model,
                        llm_model,
                        now,
                        now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM knowledge_bases WHERE kb_id = ?",
                    (kb_id,),
                ).fetchone()
                return dict(row)

        return await self._execute_write(_create)

    async def create(
        self,
        tenant_id: str,
        namespace_id: str,
        slug: str,
        name: str,
        embedding_model: str,
        *,
        kb_id: Optional[str] = None,
        llm_model: str = "",
        status: str = "active",
    ) -> dict:
        kb_id = kb_id or str(uuid4())
        now = _now()

        def _create():
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO knowledge_bases
                       (kb_id, tenant_id, namespace_id, slug, name, embedding_model, llm_model,
                        status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        kb_id,
                        tenant_id,
                        namespace_id,
                        slug,
                        name,
                        embedding_model,
                        llm_model,
                        status,
                        now,
                        now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM knowledge_bases WHERE kb_id = ?",
                    (kb_id,),
                ).fetchone()
                return dict(row)

        return await self._execute_write(_create)

    async def get(self, kb_id: str) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM knowledge_bases WHERE kb_id = ?",
                    (kb_id,),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_read(_get)

    async def get_by_slug(self, tenant_id: str, slug: str) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM knowledge_bases WHERE tenant_id = ? AND slug = ?",
                    (tenant_id, slug),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_read(_get)

    async def list_by_tenant(self, tenant_id: str) -> list[dict]:
        def _list():
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM knowledge_bases WHERE tenant_id = ? ORDER BY created_at ASC",
                    (tenant_id,),
                ).fetchall()
                return [dict(row) for row in rows]

        return await self._execute_read(_list)

    async def list_all(self) -> list[dict]:
        def _list():
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM knowledge_bases ORDER BY created_at ASC"
                ).fetchall()
                return [dict(row) for row in rows]

        return await self._execute_read(_list)

    async def tenant_revision(self, tenant_id: str) -> dict:
        """Return a lightweight cache revision for external Agent/UI synchronization."""
        def _revision() -> dict:
            with self._conn() as conn:
                row = conn.execute(
                    """SELECT COUNT(*) AS item_count, COALESCE(MAX(updated_at), '') AS updated_at
                       FROM knowledge_bases WHERE tenant_id = ?""",
                    (tenant_id,),
                ).fetchone()
                return {
                    "item_count": int(row["item_count"]),
                    "updated_at": str(row["updated_at"] or ""),
                }

        return await self._execute_read(_revision)

    async def ensure_default(
        self,
        tenant_id: str,
        namespace_id: str,
        *,
        slug: str = "default",
        name: str = "默认知识库",
        embedding_model: str,
        llm_model: str = "",
    ) -> dict:
        existing = await self.get_by_slug(tenant_id, slug)
        if existing:
            if (
                existing.get("embedding_model") == embedding_model
                and (existing.get("llm_model") or "") == llm_model
            ):
                if existing.get("name") == "Default Knowledge Base":
                    return await self.update_models(
                        existing["kb_id"],
                        embedding_model=embedding_model,
                        llm_model=llm_model,
                        name=name,
                    )
                return existing
            return await self.update_models(
                existing["kb_id"],
                embedding_model=embedding_model,
                llm_model=llm_model,
            )
        return await self.create(
            tenant_id=tenant_id,
            namespace_id=namespace_id,
            slug=slug,
            name=name,
            embedding_model=embedding_model,
            llm_model=llm_model,
        )

    async def update_models(
        self,
        kb_id: str,
        *,
        embedding_model: str,
        llm_model: str = "",
        name: str | None = None,
    ) -> dict:
        now = _now()

        def _update():
            with self._conn() as conn:
                if name is None:
                    conn.execute(
                        """UPDATE knowledge_bases
                           SET embedding_model = ?, llm_model = ?, updated_at = ?
                           WHERE kb_id = ?""",
                        (embedding_model, llm_model, now, kb_id),
                    )
                else:
                    conn.execute(
                        """UPDATE knowledge_bases
                           SET embedding_model = ?, llm_model = ?, name = ?, updated_at = ?
                           WHERE kb_id = ?""",
                        (embedding_model, llm_model, name, now, kb_id),
                    )
                row = conn.execute(
                    "SELECT * FROM knowledge_bases WHERE kb_id = ?",
                    (kb_id,),
                ).fetchone()
                return dict(row)

        return await self._execute_write(_update)

    async def mark_deleting(self, kb_id: str, tenant_id: str) -> dict:
        """Atomically hide a tenant KB before physical cleanup begins."""
        now = _now()

        def _mark() -> dict:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM knowledge_bases WHERE kb_id = ? AND tenant_id = ?",
                    (kb_id, tenant_id),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Knowledge base not found: {kb_id}")
                if row["status"] not in {"active", "deleting", "delete_failed"}:
                    raise ValueError(f"Knowledge base status is {row['status']}")
                conn.execute(
                    """UPDATE knowledge_bases
                       SET status = 'deleting', error_message = '', updated_at = ?
                       WHERE kb_id = ? AND tenant_id = ?""",
                    (now, kb_id, tenant_id),
                )
                updated = conn.execute(
                    "SELECT * FROM knowledge_bases WHERE kb_id = ?",
                    (kb_id,),
                ).fetchone()
                return dict(updated)

        return await self._execute_write(_mark)

    async def mark_delete_failed(self, kb_id: str, tenant_id: str, error_message: str) -> dict | None:
        """Persist a retryable deletion failure independently of job history."""
        now = _now()

        def _mark() -> dict | None:
            with self._conn() as conn:
                cursor = conn.execute(
                    """UPDATE knowledge_bases
                       SET status = 'delete_failed', error_message = ?, updated_at = ?
                       WHERE kb_id = ? AND tenant_id = ? AND status = 'deleting'""",
                    (error_message, now, kb_id, tenant_id),
                )
                if cursor.rowcount == 0:
                    return None
                row = conn.execute(
                    "SELECT * FROM knowledge_bases WHERE kb_id = ?",
                    (kb_id,),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_write(_mark)

    async def delete(self, kb_id: str, tenant_id: str) -> dict:
        """Remove a physically-cleaned, tenant-scoped knowledge base row."""
        def _delete() -> dict:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM knowledge_bases WHERE kb_id = ? AND tenant_id = ?",
                    (kb_id, tenant_id),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Knowledge base not found: {kb_id}")
                if row["status"] != "deleting":
                    raise ValueError("knowledge base must be marked deleting first")
                conn.execute(
                    "DELETE FROM knowledge_bases WHERE kb_id = ? AND tenant_id = ?",
                    (kb_id, tenant_id),
                )
                return dict(row)

        return await self._execute_write(_delete)
