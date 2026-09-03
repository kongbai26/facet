"""Metadata for immutable RAG index profiles and knowledge-base cutovers."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from app.index_profile import canonical_json, profile_hash, source_fingerprint
from app.store.sqlite_store import SQLiteStore


INDEX_STATUSES = {"building", "ready", "active", "failed", "retired"}
DOCUMENT_INDEX_STATUSES = {"queued", "building", "ready", "failed", "retired"}

CREATE_INDEX_PROFILE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS index_profiles (
    profile_hash TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_base_indexes (
    kb_id TEXT NOT NULL,
    profile_hash TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT NOT NULL DEFAULT '',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    activated_at TEXT,
    PRIMARY KEY (kb_id, profile_hash),
    UNIQUE (collection_name)
);
CREATE INDEX IF NOT EXISTS idx_kb_indexes_active
ON knowledge_base_indexes(kb_id, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS document_index_states (
    doc_id TEXT NOT NULL,
    profile_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, profile_hash)
);
CREATE INDEX IF NOT EXISTS idx_document_index_states_profile
ON document_index_states(profile_hash, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_base_index_generations (
    index_id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    profile_hash TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT NOT NULL DEFAULT '',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    activated_at TEXT,
    UNIQUE (kb_id, profile_hash, source_fingerprint),
    UNIQUE (collection_name)
);
CREATE INDEX IF NOT EXISTS idx_kb_index_generations_active
ON knowledge_base_index_generations(kb_id, status, updated_at DESC);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IndexProfileStore(SQLiteStore):
    """Keeps candidate and active indexes independent from source documents.

    ``documents.status`` remains the source-file lifecycle.  This store owns
    the lifecycle of every representation of a source document in an immutable
    index profile, so an active index remains queryable during a rebuild.
    """

    def __init__(self, db_path: str):
        super().__init__(db_path)
        with self._conn() as conn:
            conn.executescript(CREATE_INDEX_PROFILE_TABLES_SQL)
            self._migrate_generations(conn)

    @staticmethod
    def _index_id(digest: str, source_fingerprint: str) -> str:
        return f"{digest}-{source_fingerprint[:16] or 'legacy'}"

    def _migrate_generations(self, conn) -> None:
        """Copy the first-generation table without losing an active pointer.

        The old primary key was ``(kb_id, profile_hash)``.  That cannot keep a
        currently-active collection while rebuilding the same profile for a
        later document revision, so all new reads use the generation table.
        """
        rows = conn.execute("SELECT * FROM knowledge_base_indexes").fetchall()
        for row in rows:
            item = dict(row)
            digest = item["profile_hash"]
            source_fingerprint = "legacy"
            conn.execute(
                """INSERT OR IGNORE INTO knowledge_base_index_generations
                   (index_id, kb_id, profile_hash, source_fingerprint, collection_name,
                    status, error_message, chunk_count, created_at, updated_at, activated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self._index_id(digest, source_fingerprint), item["kb_id"], digest,
                    source_fingerprint, item["collection_name"], item["status"],
                    item.get("error_message") or "", item.get("chunk_count") or 0,
                    item["created_at"], item["updated_at"], item.get("activated_at"),
                ),
            )

    @staticmethod
    def _resolve_index_id(conn, kb_id: str, identifier: str) -> str:
        """Accept the old profile-hash identifier when it is unambiguous."""
        direct = conn.execute(
            "SELECT index_id FROM knowledge_base_index_generations WHERE kb_id = ? AND index_id = ?",
            (kb_id, identifier),
        ).fetchone()
        if direct:
            return str(direct["index_id"])
        rows = conn.execute(
            """SELECT index_id FROM knowledge_base_index_generations
               WHERE kb_id = ? AND profile_hash = ?""",
            (kb_id, identifier),
        ).fetchall()
        if len(rows) == 1:
            return str(rows[0]["index_id"])
        return identifier

    async def ensure_profile(self, profile: Mapping[str, Any]) -> str:
        serialized = canonical_json(profile)
        digest = profile_hash(profile)
        created_at = _now()

        def _ensure() -> str:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT profile_json FROM index_profiles WHERE profile_hash = ?", (digest,)
                ).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO index_profiles (profile_hash, profile_json, created_at) VALUES (?, ?, ?)",
                        (digest, serialized, created_at),
                    )
                elif row["profile_json"] != serialized:
                    # A truncated hash collision must fail closed; never attach
                    # a candidate to a different semantic configuration.
                    raise RuntimeError("index profile hash collision")
            return digest

        return await self._execute_write(_ensure)

    async def get_profile(self, digest: str) -> Optional[dict[str, Any]]:
        def _get() -> Optional[dict[str, Any]]:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT profile_json FROM index_profiles WHERE profile_hash = ?", (digest,)
                ).fetchone()
            if row is None:
                return None
            import json
            return json.loads(row["profile_json"])

        return await self._execute_read(_get)

    async def ensure_knowledge_base_index(
        self,
        kb_id: str,
        digest: str,
        collection_name: str,
        *,
        source_fingerprint: str = "legacy",
        status: str = "building",
    ) -> dict:
        if status not in INDEX_STATUSES:
            raise ValueError(f"invalid index status: {status}")
        now = _now()
        index_id = self._index_id(digest, source_fingerprint)

        def _ensure() -> dict:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO knowledge_base_index_generations
                       (index_id, kb_id, profile_hash, source_fingerprint, collection_name,
                        status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(kb_id, profile_hash, source_fingerprint) DO NOTHING""",
                    (index_id, kb_id, digest, source_fingerprint, collection_name, status, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM knowledge_base_index_generations WHERE index_id = ?",
                    (index_id,),
                ).fetchone()
                return dict(row)

        return await self._execute_write(_ensure)

    async def get_active_index(self, kb_id: str) -> Optional[dict]:
        def _get() -> Optional[dict]:
            with self._conn() as conn:
                row = conn.execute(
                    """SELECT * FROM knowledge_base_index_generations
                       WHERE kb_id = ? AND status = 'active'
                       ORDER BY activated_at DESC, updated_at DESC LIMIT 1""",
                    (kb_id,),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_read(_get)

    async def get_knowledge_base_index(self, kb_id: str, index_id: str) -> Optional[dict]:
        def _get() -> Optional[dict]:
            with self._conn() as conn:
                row = conn.execute(
                    """SELECT * FROM knowledge_base_index_generations
                       WHERE kb_id = ? AND index_id = ?""",
                    (kb_id, index_id),
                ).fetchone()
                return dict(row) if row else None

        return await self._execute_read(_get)

    async def list_knowledge_base_indexes(self, kb_id: str) -> list[dict]:
        def _list() -> list[dict]:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM knowledge_base_index_generations WHERE kb_id = ? ORDER BY updated_at DESC",
                    (kb_id,),
                ).fetchall()
                return [dict(row) for row in rows]

        return await self._execute_read(_list)

    async def list_all_indexes(self) -> list[dict]:
        """Return all generation metadata for startup-derived-cache cleanup."""
        def _list() -> list[dict]:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM knowledge_base_index_generations ORDER BY updated_at DESC",
                ).fetchall()
                return [dict(row) for row in rows]

        return await self._execute_read(_list)

    async def list_expired_inactive_indexes(self, cutoff: str) -> list[dict]:
        """Return non-active generations whose last lifecycle update expired."""
        def _list() -> list[dict]:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT * FROM knowledge_base_index_generations
                       WHERE status IN ('ready', 'failed', 'retired') AND updated_at < ?
                       ORDER BY updated_at ASC""",
                    (cutoff,),
                ).fetchall()
                return [dict(row) for row in rows]

        return await self._execute_read(_list)

    async def mark_index(
        self,
        kb_id: str,
        index_id: str,
        status: str,
        *,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> dict:
        if status not in INDEX_STATUSES:
            raise ValueError(f"invalid index status: {status}")
        now = _now()

        def _mark() -> dict:
            with self._conn() as conn:
                resolved_index_id = self._resolve_index_id(conn, kb_id, index_id)
                row = conn.execute(
                    "SELECT * FROM knowledge_base_index_generations WHERE kb_id = ? AND index_id = ?",
                    (kb_id, resolved_index_id),
                ).fetchone()
                if row is None:
                    raise KeyError(f"index not found: {kb_id}/{index_id}")
                conn.execute(
                    """UPDATE knowledge_base_index_generations
                       SET status = ?, chunk_count = COALESCE(?, chunk_count),
                           error_message = COALESCE(?, error_message), updated_at = ?
                       WHERE kb_id = ? AND index_id = ?""",
                    (status, chunk_count, error_message, now, kb_id, resolved_index_id),
                )
                updated = conn.execute(
                    "SELECT * FROM knowledge_base_index_generations WHERE kb_id = ? AND index_id = ?",
                    (kb_id, resolved_index_id),
                ).fetchone()
                return dict(updated)

        return await self._execute_write(_mark)

    async def activate_index(self, kb_id: str, index_id: str) -> dict:
        """Atomically make one verified candidate the only active KB index."""
        now = _now()

        def _activate() -> dict:
            with self._conn() as conn:
                resolved_index_id = self._resolve_index_id(conn, kb_id, index_id)
                candidate = conn.execute(
                    "SELECT * FROM knowledge_base_index_generations WHERE kb_id = ? AND index_id = ?",
                    (kb_id, resolved_index_id),
                ).fetchone()
                if candidate is None:
                    raise KeyError(f"index not found: {kb_id}/{index_id}")
                if candidate["status"] not in {"ready", "active"}:
                    raise ValueError("only a ready index can become active")
                try:
                    kb = conn.execute(
                        "SELECT status FROM knowledge_bases WHERE kb_id = ?",
                        (kb_id,),
                    ).fetchone()
                except sqlite3.OperationalError:
                    kb = None
                if kb is not None and kb["status"] != "active":
                    raise ValueError("knowledge base is not active")
                conn.execute(
                    """UPDATE knowledge_base_index_generations
                       SET status = 'ready', updated_at = ?
                       WHERE kb_id = ? AND status = 'active' AND index_id != ?""",
                    (now, kb_id, resolved_index_id),
                )
                conn.execute(
                    """UPDATE knowledge_base_index_generations
                       SET status = 'active', activated_at = ?, updated_at = ?, error_message = ''
                       WHERE kb_id = ? AND index_id = ?""",
                    (now, now, kb_id, resolved_index_id),
                )
                row = conn.execute(
                    "SELECT * FROM knowledge_base_index_generations WHERE kb_id = ? AND index_id = ?",
                    (kb_id, resolved_index_id),
                ).fetchone()
                return dict(row)

        return await self._execute_write(_activate)

    async def activate_index_if_source_current(self, kb_id: str, index_id: str) -> dict:
        """Cut over a candidate only if its source snapshot is still current.

        The source read and active-pointer update share one SQLite write
        transaction.  Without this, a document can become ready after the
        caller's pre-check but before ``activate_index`` commits.  The
        immediate transaction also serializes this decision with source
        writes made by another process.
        """
        now = _now()

        def _activate() -> dict:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                resolved_index_id = self._resolve_index_id(conn, kb_id, index_id)
                candidate = conn.execute(
                    "SELECT * FROM knowledge_base_index_generations WHERE kb_id = ? AND index_id = ?",
                    (kb_id, resolved_index_id),
                ).fetchone()
                if candidate is None:
                    raise KeyError(f"index not found: {kb_id}/{index_id}")
                if candidate["status"] not in {"ready", "active"}:
                    raise ValueError("only a ready index can become active")

                try:
                    kb = conn.execute(
                        "SELECT status FROM knowledge_bases WHERE kb_id = ?",
                        (kb_id,),
                    ).fetchone()
                except sqlite3.OperationalError:
                    kb = None
                if kb is not None and kb["status"] != "active":
                    raise ValueError("knowledge base is not active")

                source_rows = conn.execute(
                    """SELECT doc_id, content_hash, file_size, source_revision
                       FROM documents
                       WHERE kb_id = ? AND status = 'ready'""",
                    (kb_id,),
                ).fetchall()
                current_fingerprint = source_fingerprint([dict(row) for row in source_rows])
                if candidate["source_fingerprint"] != current_fingerprint:
                    raise ValueError("候选索引构建期间文档已变化，请重新构建后再切换")

                conn.execute(
                    """UPDATE knowledge_base_index_generations
                       SET status = 'ready', updated_at = ?
                       WHERE kb_id = ? AND status = 'active' AND index_id != ?""",
                    (now, kb_id, resolved_index_id),
                )
                conn.execute(
                    """UPDATE knowledge_base_index_generations
                       SET status = 'active', activated_at = ?, updated_at = ?, error_message = ''
                       WHERE kb_id = ? AND index_id = ?""",
                    (now, now, kb_id, resolved_index_id),
                )
                row = conn.execute(
                    "SELECT * FROM knowledge_base_index_generations WHERE kb_id = ? AND index_id = ?",
                    (kb_id, resolved_index_id),
                ).fetchone()
                return dict(row)

        return await self._execute_write(_activate)

    async def delete_inactive_index(self, kb_id: str, index_id: str) -> dict:
        """Forget an inactive generation after its physical data is reclaimed."""
        def _delete() -> dict:
            with self._conn() as conn:
                resolved_index_id = self._resolve_index_id(conn, kb_id, index_id)
                row = conn.execute(
                    """SELECT * FROM knowledge_base_index_generations
                       WHERE kb_id = ? AND index_id = ?""",
                    (kb_id, resolved_index_id),
                ).fetchone()
                if row is None:
                    raise KeyError(f"index not found: {kb_id}/{index_id}")
                if row["status"] == "active":
                    raise ValueError("cannot reclaim the active index")
                item = dict(row)
                conn.execute(
                    "DELETE FROM document_index_states WHERE profile_hash = ?",
                    (resolved_index_id,),
                )
                conn.execute(
                    "DELETE FROM knowledge_base_index_generations WHERE index_id = ?",
                    (resolved_index_id,),
                )
                return item

        return await self._execute_write(_delete)

    async def delete_index_for_knowledge_base(self, kb_id: str, index_id: str) -> dict:
        """Delete one generation after its physical collection is gone."""
        def _delete() -> dict:
            with self._conn() as conn:
                resolved_index_id = self._resolve_index_id(conn, kb_id, index_id)
                row = conn.execute(
                    "SELECT * FROM knowledge_base_index_generations WHERE kb_id = ? AND index_id = ?",
                    (kb_id, resolved_index_id),
                ).fetchone()
                if row is None:
                    raise KeyError(f"index not found: {kb_id}/{index_id}")
                item = dict(row)
                conn.execute(
                    "DELETE FROM document_index_states WHERE profile_hash = ?",
                    (resolved_index_id,),
                )
                conn.execute(
                    "DELETE FROM knowledge_base_index_generations WHERE kb_id = ? AND index_id = ?",
                    (kb_id, resolved_index_id),
                )
                return item

        return await self._execute_write(_delete)

    async def upsert_document_state(
        self,
        doc_id: str,
        digest: str,
        status: str,
        *,
        chunk_count: int = 0,
        error_message: str = "",
    ) -> dict:
        if status not in DOCUMENT_INDEX_STATUSES:
            raise ValueError(f"invalid document index status: {status}")
        now = _now()

        def _upsert() -> dict:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO document_index_states
                       (doc_id, profile_hash, status, chunk_count, error_message, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(doc_id, profile_hash) DO UPDATE SET
                         status = excluded.status, chunk_count = excluded.chunk_count,
                         error_message = excluded.error_message, updated_at = excluded.updated_at""",
                    (doc_id, digest, status, int(chunk_count), error_message, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM document_index_states WHERE doc_id = ? AND profile_hash = ?",
                    (doc_id, digest),
                ).fetchone()
                return dict(row)

        return await self._execute_write(_upsert)

    async def list_document_states(self, doc_id: str) -> list[dict]:
        def _list() -> list[dict]:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM document_index_states WHERE doc_id = ? ORDER BY updated_at DESC",
                    (doc_id,),
                ).fetchall()
                return [dict(row) for row in rows]

        return await self._execute_read(_list)

    async def list_document_collections(self, doc_id: str) -> list[dict]:
        """Return every candidate collection that can still contain a document."""
        def _list() -> list[dict]:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT states.doc_id, states.profile_hash AS index_id,
                              states.status AS document_index_status,
                              indexes.kb_id, indexes.collection_name, indexes.status AS index_status
                       FROM document_index_states AS states
                       JOIN knowledge_base_index_generations AS indexes
                         ON indexes.index_id = states.profile_hash
                       WHERE states.doc_id = ?""",
                    (doc_id,),
                ).fetchall()
                return [dict(row) for row in rows]

        return await self._execute_read(_list)

    async def retire_document_states(self, doc_id: str) -> list[dict]:
        now = _now()

        def _retire() -> list[dict]:
            with self._conn() as conn:
                conn.execute(
                    """UPDATE document_index_states SET status = 'retired', updated_at = ?
                       WHERE doc_id = ? AND status != 'retired'""",
                    (now, doc_id),
                )
                rows = conn.execute(
                    "SELECT * FROM document_index_states WHERE doc_id = ?",
                    (doc_id,),
                ).fetchall()
                return [dict(row) for row in rows]

        return await self._execute_write(_retire)
