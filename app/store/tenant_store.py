"""Tenant metadata storage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.store.sqlite_store import SQLiteStore

CREATE_TENANTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TenantStore(SQLiteStore):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(CREATE_TENANTS_TABLE_SQL)

    async def create(
        self,
        slug: str,
        name: str,
        *,
        tenant_id: Optional[str] = None,
        status: str = "active",
    ) -> dict:
        tenant_id = tenant_id or str(uuid4())
        now = _now()

        def _create():
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO tenants
                       (tenant_id, slug, name, status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (tenant_id, slug, name, status, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM tenants WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchone()
                return dict(row)

        return await self._execute_write(_create)

    async def get(self, tenant_id: str) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM tenants WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_read(_get)

    async def get_by_slug(self, slug: str) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM tenants WHERE slug = ?",
                    (slug,),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_read(_get)

    async def list_all(self, limit: int = 100) -> list[dict]:
        def _list():
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM tenants ORDER BY created_at ASC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(row) for row in rows]

        return await self._execute_read(_list)

    async def ensure_default(self, slug: str = "default", name: str = "Default Tenant") -> dict:
        existing = await self.get_by_slug(slug)
        if existing:
            return existing
        return await self.create(slug=slug, name=name)
