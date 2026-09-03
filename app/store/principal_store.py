"""Principal metadata storage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.store.sqlite_store import SQLiteStore

CREATE_PRINCIPALS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS principals (
    principal_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    principal_type TEXT NOT NULL,
    name TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (tenant_id, principal_type, name),
    FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_principals_tenant_id ON principals(tenant_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PrincipalStore(SQLiteStore):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(CREATE_PRINCIPALS_TABLE_SQL)
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(principals)").fetchall()
            }
            if "is_admin" not in columns:
                conn.execute(
                    "ALTER TABLE principals ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0"
                )

    async def create(
        self,
        tenant_id: str,
        name: str,
        *,
        principal_id: Optional[str] = None,
        principal_type: str = "service",
        is_admin: bool = False,
        status: str = "active",
    ) -> dict:
        principal_id = principal_id or str(uuid4())
        now = _now()

        def _create():
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO principals
                       (principal_id, tenant_id, principal_type, name, is_admin, status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (principal_id, tenant_id, principal_type, name, int(is_admin), status, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM principals WHERE principal_id = ?",
                    (principal_id,),
                ).fetchone()
                return dict(row)

        return await self._execute_write(_create)

    async def get(self, principal_id: str) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM principals WHERE principal_id = ?",
                    (principal_id,),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_read(_get)

    async def get_by_name(self, tenant_id: str, principal_type: str, name: str) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    """SELECT * FROM principals
                       WHERE tenant_id = ? AND principal_type = ? AND name = ?""",
                    (tenant_id, principal_type, name),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_read(_get)

    async def list_by_tenant(self, tenant_id: str) -> list[dict]:
        def _list():
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM principals WHERE tenant_id = ? ORDER BY created_at ASC",
                    (tenant_id,),
                ).fetchall()
                return [dict(row) for row in rows]

        return await self._execute_read(_list)

    async def set_admin(self, principal_id: str, is_admin: bool) -> Optional[dict]:
        now = _now()

        def _update():
            with self._conn() as conn:
                conn.execute(
                    "UPDATE principals SET is_admin = ?, updated_at = ? WHERE principal_id = ?",
                    (int(is_admin), now, principal_id),
                )
                row = conn.execute(
                    "SELECT * FROM principals WHERE principal_id = ?",
                    (principal_id,),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_write(_update)

    async def migrate_legacy_default_admin(self, tenant_id: str, name: str = "local-admin") -> Optional[dict]:
        user_principal = await self.get_by_name(tenant_id, "user", name)
        legacy_principal = await self.get_by_name(tenant_id, "admin_session", name)

        if user_principal:
            if not user_principal.get("is_admin"):
                user_principal = await self.set_admin(user_principal["principal_id"], True)
            return user_principal

        if not legacy_principal:
            return None

        now = _now()

        def _migrate():
            with self._conn() as conn:
                conn.execute(
                    """
                    UPDATE principals
                    SET principal_type = 'user', is_admin = 1, updated_at = ?
                    WHERE principal_id = ?
                    """,
                    (now, legacy_principal["principal_id"]),
                )
                row = conn.execute(
                    "SELECT * FROM principals WHERE principal_id = ?",
                    (legacy_principal["principal_id"],),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_write(_migrate)

    async def ensure_default_admin(self, tenant_id: str, name: str = "local-admin") -> dict:
        existing = await self.migrate_legacy_default_admin(tenant_id, name)
        if existing:
            return existing
        return await self.create(
            tenant_id=tenant_id,
            name=name,
            principal_type="user",
            is_admin=True,
        )
