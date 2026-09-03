"""Namespace metadata storage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.store.sqlite_store import SQLiteStore

CREATE_NAMESPACES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS namespaces (
    namespace_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (tenant_id, slug),
    FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_namespaces_tenant_id ON namespaces(tenant_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NamespaceStore(SQLiteStore):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(CREATE_NAMESPACES_TABLE_SQL)

    async def create(
        self,
        tenant_id: str,
        slug: str,
        name: str,
        *,
        namespace_id: Optional[str] = None,
        kind: str = "vector",
        status: str = "active",
    ) -> dict:
        namespace_id = namespace_id or str(uuid4())
        now = _now()

        def _create():
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO namespaces
                       (namespace_id, tenant_id, slug, name, kind, status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (namespace_id, tenant_id, slug, name, kind, status, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM namespaces WHERE namespace_id = ?",
                    (namespace_id,),
                ).fetchone()
                return dict(row)

        return await self._execute_write(_create)

    async def get(self, namespace_id: str) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM namespaces WHERE namespace_id = ?",
                    (namespace_id,),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_read(_get)

    async def get_by_slug(self, tenant_id: str, slug: str) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM namespaces WHERE tenant_id = ? AND slug = ?",
                    (tenant_id, slug),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_read(_get)

    async def list_by_tenant(self, tenant_id: str, kind: Optional[str] = None) -> list[dict]:
        def _list():
            query = "SELECT * FROM namespaces WHERE tenant_id = ?"
            params: list[object] = [tenant_id]
            if kind:
                query += " AND kind = ?"
                params.append(kind)
            query += " ORDER BY created_at ASC"

            with self._conn() as conn:
                rows = conn.execute(query, tuple(params)).fetchall()
                return [dict(row) for row in rows]

        return await self._execute_read(_list)

    async def ensure_default(
        self,
        tenant_id: str,
        *,
        slug: str,
        name: str,
        kind: str,
    ) -> dict:
        existing = await self.get_by_slug(tenant_id, slug)
        if existing:
            return existing
        return await self.create(tenant_id=tenant_id, slug=slug, name=name, kind=kind)

    async def delete_if_orphaned(self, namespace_id: str, tenant_id: str) -> bool:
        """Delete a namespace only when no knowledge base still references it."""
        def _delete() -> bool:
            with self._conn() as conn:
                cursor = conn.execute(
                    """DELETE FROM namespaces
                       WHERE namespace_id = ? AND tenant_id = ?
                         AND NOT EXISTS (
                             SELECT 1 FROM knowledge_bases
                             WHERE knowledge_bases.namespace_id = namespaces.namespace_id
                         )""",
                    (namespace_id, tenant_id),
                )
                return cursor.rowcount > 0

        return await self._execute_write(_delete)
