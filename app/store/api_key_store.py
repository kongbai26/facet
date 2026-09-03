"""API key storage and lifecycle helpers."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from app.store.sqlite_store import SQLiteStore

CREATE_API_KEYS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    scopes_json TEXT NOT NULL DEFAULT '[]',
    is_admin INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    requests_per_minute INTEGER,
    daily_quota INTEGER,
    expires_at TEXT,
    last_used_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY(principal_id) REFERENCES principals(principal_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant_id ON api_keys(tenant_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_principal_id ON api_keys(principal_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active);

CREATE TABLE IF NOT EXISTS api_key_usage (
    key_id TEXT PRIMARY KEY,
    minute_started_at TEXT NOT NULL,
    minute_count INTEGER NOT NULL DEFAULT 0,
    day_started TEXT NOT NULL,
    day_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(key_id) REFERENCES api_keys(key_id) ON DELETE CASCADE
);
"""


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    return value.isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _row_to_key(row) -> dict:
    data = dict(row)
    try:
        data["scopes"] = json.loads(data.pop("scopes_json") or "[]")
    except json.JSONDecodeError:
        data["scopes"] = []
    data["is_admin"] = bool(data.get("is_admin"))
    data["is_active"] = bool(data.get("is_active"))
    return data


class ApiKeyStore(SQLiteStore):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(CREATE_API_KEYS_TABLE_SQL)

    @staticmethod
    def hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    async def create_key(
        self,
        tenant_id: str,
        principal_id: str,
        name: str,
        *,
        scopes: Optional[list[str]] = None,
        is_admin: bool = False,
        expires_at: Optional[str] = None,
        requests_per_minute: Optional[int] = None,
        daily_quota: Optional[int] = None,
    ) -> dict:
        key_id = str(uuid4())
        raw_key = f"sk-{secrets.token_hex(32)}"
        key_hash = self.hash_key(raw_key)
        now = _to_iso(_now_dt())
        scopes_json = json.dumps(scopes or [], ensure_ascii=False)

        def _create():
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO api_keys
                       (key_id, tenant_id, principal_id, key_hash, name, scopes_json, is_admin,
                        is_active, requests_per_minute, daily_quota, expires_at, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)""",
                    (
                        key_id,
                        tenant_id,
                        principal_id,
                        key_hash,
                        name,
                        scopes_json,
                        1 if is_admin else 0,
                        requests_per_minute,
                        daily_quota,
                        expires_at,
                        now,
                        now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM api_keys WHERE key_id = ?",
                    (key_id,),
                ).fetchone()
                data = _row_to_key(row)
                data["raw_key"] = raw_key
                return data

        return await self._execute_write(_create)

    async def get(self, key_id: str) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM api_keys WHERE key_id = ?",
                    (key_id,),
                ).fetchone()
                return _row_to_key(row) if row else None

        return await self._execute_read(_get)

    async def get_key_by_hash(self, key_hash: str) -> Optional[dict]:
        now = _now_dt()

        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    """SELECT * FROM api_keys
                       WHERE key_hash = ? AND is_active = 1""",
                    (key_hash,),
                ).fetchone()
                if not row:
                    return None
                data = _row_to_key(row)
                expires_at = _parse_iso(data.get("expires_at"))
                if expires_at and expires_at <= now:
                    conn.execute(
                        """UPDATE api_keys
                           SET is_active = 0, updated_at = ?
                           WHERE key_id = ?""",
                        (_to_iso(now), data["key_id"]),
                    )
                    return None
                return data

        return await self._execute_write(_get)

    async def list_keys(self, tenant_id: Optional[str] = None) -> list[dict]:
        def _list():
            query = "SELECT * FROM api_keys"
            params: list[object] = []
            if tenant_id:
                query += " WHERE tenant_id = ?"
                params.append(tenant_id)
            query += " ORDER BY created_at DESC"

            with self._conn() as conn:
                rows = conn.execute(query, tuple(params)).fetchall()
                return [_row_to_key(row) for row in rows]

        return await self._execute_read(_list)

    async def list_empty_scope_keys(
        self,
        tenant_id: Optional[str] = None,
        *,
        active_only: bool = True,
        exclude_admin: bool = True,
    ) -> list[dict]:
        keys = await self.list_keys(tenant_id)
        result = []
        for key in keys:
            if active_only and not key.get("is_active"):
                continue
            if exclude_admin and key.get("is_admin"):
                continue
            if key.get("scopes"):
                continue
            result.append(key)
        return result

    async def update_scopes(self, key_id: str, scopes: list[str]) -> Optional[dict]:
        scopes_json = json.dumps(scopes or [], ensure_ascii=False)
        now = _to_iso(_now_dt())

        def _update():
            with self._conn() as conn:
                conn.execute(
                    """UPDATE api_keys
                       SET scopes_json = ?, updated_at = ?
                       WHERE key_id = ?""",
                    (scopes_json, now, key_id),
                )
                row = conn.execute(
                    "SELECT * FROM api_keys WHERE key_id = ?",
                    (key_id,),
                ).fetchone()
                return _row_to_key(row) if row else None

        return await self._execute_write(_update)

    async def backfill_empty_scope_keys(self, scopes: list[str]) -> int:
        keys = await self.list_empty_scope_keys()
        updated = 0
        for key in keys:
            await self.update_scopes(key["key_id"], scopes)
            updated += 1
        return updated

    async def revoke_key(self, key_id: str) -> bool:
        now = _to_iso(_now_dt())

        def _revoke():
            with self._conn() as conn:
                cursor = conn.execute(
                    """UPDATE api_keys
                       SET is_active = 0, updated_at = ?
                       WHERE key_id = ? AND is_active = 1""",
                    (now, key_id),
                )
                return cursor.rowcount > 0

        return await self._execute_write(_revoke)

    async def delete_revoked_keys(
        self,
        tenant_id: str,
        key_ids: Optional[list[str]] = None,
    ) -> int:
        """Permanently delete revoked keys owned by one tenant.

        Active keys are deliberately excluded so cleanup can never silently
        turn into a revocation action.
        """
        def _delete():
            with self._conn() as conn:
                if key_ids:
                    placeholders = ", ".join("?" for _ in key_ids)
                    cursor = conn.execute(
                        f"""DELETE FROM api_keys
                            WHERE tenant_id = ? AND is_active = 0
                              AND key_id IN ({placeholders})""",
                        (tenant_id, *key_ids),
                    )
                else:
                    cursor = conn.execute(
                        "DELETE FROM api_keys WHERE tenant_id = ? AND is_active = 0",
                        (tenant_id,),
                    )
                return cursor.rowcount

        return await self._execute_write(_delete)

    async def touch_last_used(self, key_id: str) -> None:
        now = _to_iso(_now_dt())

        def _touch():
            with self._conn() as conn:
                conn.execute(
                    """UPDATE api_keys
                       SET last_used_at = ?, updated_at = ?
                       WHERE key_id = ?""",
                    (now, now, key_id),
                )

        await self._execute_write(_touch)

    async def consume_usage(
        self,
        key_id: str,
        *,
        requests_per_minute: int | None = None,
        daily_quota: int | None = None,
    ) -> dict:
        """Atomically check and record one API-key request.

        Usage is deliberately stored in one row per key, so enabling limits
        does not create an unbounded request-log table.  Windows are UTC and
        reset lazily when the next request arrives.
        """
        if requests_per_minute is None and daily_quota is None:
            return {"allowed": True, "retry_after": 0, "reason": None}

        now = _now_dt()
        now_iso = _to_iso(now)
        day_value = now.date().isoformat()

        def _consume() -> dict:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM api_key_usage WHERE key_id = ?",
                    (key_id,),
                ).fetchone()
                if row is None:
                    minute_started_at = now
                    minute_count = 0
                    day_started = day_value
                    day_count = 0
                else:
                    minute_started_at = _parse_iso(row["minute_started_at"]) or now
                    minute_count = int(row["minute_count"] or 0)
                    day_started = row["day_started"] or day_value
                    day_count = int(row["day_count"] or 0)
                    if (now - minute_started_at).total_seconds() >= 60:
                        minute_started_at = now
                        minute_count = 0
                    if day_started != day_value:
                        day_started = day_value
                        day_count = 0

                if requests_per_minute is not None and minute_count >= int(requests_per_minute):
                    retry_after = max(
                        1,
                        int(60 - (now - minute_started_at).total_seconds()),
                    )
                    return {
                        "allowed": False,
                        "retry_after": retry_after,
                        "reason": "requests_per_minute",
                    }
                if daily_quota is not None and day_count >= int(daily_quota):
                    next_day = datetime.combine(
                        now.date() + timedelta(days=1),
                        datetime.min.time(),
                        tzinfo=timezone.utc,
                    )
                    retry_after = max(1, int((next_day - now).total_seconds()))
                    return {
                        "allowed": False,
                        "retry_after": retry_after,
                        "reason": "daily_quota",
                    }

                minute_count += 1
                day_count += 1
                conn.execute(
                    """INSERT INTO api_key_usage
                       (key_id, minute_started_at, minute_count, day_started, day_count, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(key_id) DO UPDATE SET
                         minute_started_at = excluded.minute_started_at,
                         minute_count = excluded.minute_count,
                         day_started = excluded.day_started,
                         day_count = excluded.day_count,
                         updated_at = excluded.updated_at""",
                    (
                        key_id,
                        _to_iso(minute_started_at),
                        minute_count,
                        day_started,
                        day_count,
                        now_iso,
                    ),
                )
                return {"allowed": True, "retry_after": 0, "reason": None}

        return await self._execute_write(_consume)

    async def cleanup_expired(self) -> int:
        now = _to_iso(_now_dt())

        def _cleanup():
            with self._conn() as conn:
                cursor = conn.execute(
                    """UPDATE api_keys
                       SET is_active = 0, updated_at = ?
                       WHERE is_active = 1
                         AND expires_at IS NOT NULL
                         AND expires_at <= ?""",
                    (now, now),
                )
                return cursor.rowcount

        return await self._execute_write(_cleanup)
