"""SQLite-backed local authentication credentials."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.store.sqlite_store import SQLiteStore

CREATE_AUTH_CREDENTIALS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS auth_credentials (
    credential_id TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL,
    credential_type TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_used_at TEXT,
    UNIQUE (principal_id, credential_type),
    FOREIGN KEY(principal_id) REFERENCES principals(principal_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_auth_credentials_principal_id
ON auth_credentials(principal_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuthCredentialStore(SQLiteStore):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(CREATE_AUTH_CREDENTIALS_TABLE_SQL)

    async def get_active_password_credential(self, principal_id: str) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM auth_credentials
                    WHERE principal_id = ? AND credential_type = 'password' AND is_active = 1
                    """,
                    (principal_id,),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_read(_get)

    async def upsert_password_credential(self, principal_id: str, password_hash: str) -> dict:
        now = _now()

        def _upsert():
            with self._conn() as conn:
                existing = conn.execute(
                    """
                    SELECT credential_id FROM auth_credentials
                    WHERE principal_id = ? AND credential_type = 'password'
                    """,
                    (principal_id,),
                ).fetchone()
                if existing:
                    credential_id = existing["credential_id"]
                    conn.execute(
                        """
                        UPDATE auth_credentials
                        SET password_hash = ?, is_active = 1, updated_at = ?
                        WHERE credential_id = ?
                        """,
                        (password_hash, now, credential_id),
                    )
                else:
                    credential_id = str(uuid4())
                    conn.execute(
                        """
                        INSERT INTO auth_credentials (
                            credential_id,
                            principal_id,
                            credential_type,
                            password_hash,
                            is_active,
                            created_at,
                            updated_at
                        ) VALUES (?, ?, 'password', ?, 1, ?, ?)
                        """,
                        (credential_id, principal_id, password_hash, now, now),
                    )
                row = conn.execute(
                    "SELECT * FROM auth_credentials WHERE credential_id = ?",
                    (credential_id,),
                ).fetchone()
                return dict(row)

        return await self._execute_write(_upsert)

    async def touch_last_used(self, credential_id: str) -> None:
        now = _now()

        def _touch():
            with self._conn() as conn:
                conn.execute(
                    """
                    UPDATE auth_credentials
                    SET last_used_at = ?, updated_at = ?
                    WHERE credential_id = ?
                    """,
                    (now, now, credential_id),
                )

        await self._execute_write(_touch)
