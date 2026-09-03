"""SQLite-backed browser session storage."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.store.sqlite_store import SQLiteStore

CREATE_SESSION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    subject_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
"""


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    return value.isoformat()


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


class SessionStore(SQLiteStore):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(CREATE_SESSION_TABLE_SQL)

    def _hash_token(self, token: str, secret: str) -> str:
        return hmac.new(
            secret.encode("utf-8"),
            token.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def create_session(
        self,
        secret: str,
        ttl_seconds: int,
        subject_type: str = "workspace",
        subject_id: str = "default",
    ) -> dict:
        token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(token, secret)
        session_id = secrets.token_urlsafe(18)
        now = _now_dt()
        expires_at = now + timedelta(seconds=ttl_seconds)

        def _create():
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO sessions
                       (session_id, token_hash, subject_type, subject_id,
                        expires_at, created_at, last_seen_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        token_hash,
                        subject_type,
                        subject_id,
                        _to_iso(expires_at),
                        _to_iso(now),
                        _to_iso(now),
                    ),
                )
                return {
                    "session_id": session_id,
                    "token": token,
                    "subject_type": subject_type,
                    "subject_id": subject_id,
                    "expires_at": _to_iso(expires_at),
                }

        return await self._execute_write(_create)

    async def get_session(
        self,
        token: str,
        secret: str,
        ttl_seconds: int | None = None,
        refresh_threshold_seconds: int | None = None,
    ) -> Optional[dict]:
        token_hash = self._hash_token(token, secret)
        now = _now_dt()

        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE token_hash = ?",
                    (token_hash,),
                ).fetchone()
                if not row:
                    return None
                data = dict(row)
                expires_at = _parse_iso(data["expires_at"])
                if expires_at <= now:
                    conn.execute(
                        "DELETE FROM sessions WHERE session_id = ?",
                        (data["session_id"],),
                    )
                    return None
                should_refresh = False
                if ttl_seconds is not None:
                    if refresh_threshold_seconds is None:
                        should_refresh = True
                    else:
                        remaining = expires_at - now
                        should_refresh = remaining <= timedelta(seconds=refresh_threshold_seconds)
                if should_refresh:
                    next_expires_at = _to_iso(now + timedelta(seconds=ttl_seconds))
                    next_seen_at = _to_iso(now)
                    conn.execute(
                        "UPDATE sessions SET expires_at = ?, last_seen_at = ? WHERE session_id = ?",
                        (next_expires_at, next_seen_at, data["session_id"]),
                    )
                    data["expires_at"] = next_expires_at
                    data["last_seen_at"] = next_seen_at
                data["refreshed"] = should_refresh
                return data

        return await self._execute_write(_get)

    async def delete_session(self, token: str, secret: str) -> None:
        token_hash = self._hash_token(token, secret)

        def _delete():
            with self._conn() as conn:
                conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))

        await self._execute_write(_delete)

    async def delete_subject_sessions(
        self,
        subject_id: str,
        *,
        subject_type: str | None = None,
    ) -> None:
        def _delete():
            with self._conn() as conn:
                if subject_type:
                    conn.execute(
                        "DELETE FROM sessions WHERE subject_id = ? AND subject_type = ?",
                        (subject_id, subject_type),
                    )
                else:
                    conn.execute("DELETE FROM sessions WHERE subject_id = ?", (subject_id,))

        await self._execute_write(_delete)

    async def delete_other_sessions(
        self,
        subject_id: str,
        *,
        secret: str,
        keep_token: str,
        subject_type: str | None = None,
    ) -> None:
        keep_hash = self._hash_token(keep_token, secret)

        def _delete():
            with self._conn() as conn:
                if subject_type:
                    conn.execute(
                        """
                        DELETE FROM sessions
                        WHERE subject_id = ? AND subject_type = ? AND token_hash != ?
                        """,
                        (subject_id, subject_type, keep_hash),
                    )
                else:
                    conn.execute(
                        "DELETE FROM sessions WHERE subject_id = ? AND token_hash != ?",
                        (subject_id, keep_hash),
                    )

        await self._execute_write(_delete)

    async def cleanup_expired(self) -> None:
        now = _to_iso(_now_dt())

        def _cleanup():
            with self._conn() as conn:
                conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))

        await self._execute_write(_cleanup)
