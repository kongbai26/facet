"""SQLite storage for parent chunks used as RAG evidence."""

from __future__ import annotations

import json
from typing import Iterable

from app.store.sqlite_store import SQLiteStore


CREATE_PARENT_CHUNKS_SQL = """
CREATE TABLE IF NOT EXISTS rag_parent_chunks (
    parent_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    profile_hash TEXT NOT NULL DEFAULT 'legacy',
    parent_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_rag_parent_chunks_doc_id
ON rag_parent_chunks(doc_id, parent_index);
"""


class ParentChunkStore(SQLiteStore):
    """Stores generation context separately from small retrieval chunks."""

    def __init__(self, db_path: str):
        super().__init__(db_path)
        with self._conn() as conn:
            conn.executescript(CREATE_PARENT_CHUNKS_SQL)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(rag_parent_chunks)").fetchall()}
            if "profile_hash" not in columns:
                conn.execute(
                    "ALTER TABLE rag_parent_chunks ADD COLUMN profile_hash TEXT NOT NULL DEFAULT 'legacy'"
                )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_rag_parent_chunks_doc_profile
                   ON rag_parent_chunks(doc_id, profile_hash, parent_index)"""
            )

    async def replace_document(
        self,
        doc_id: str,
        parents: Iterable[dict],
        *,
        profile_hash: str = "legacy",
    ) -> None:
        records = list(parents)

        def _replace() -> None:
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM rag_parent_chunks WHERE doc_id = ? AND profile_hash = ?",
                    (doc_id, profile_hash),
                )
                conn.executemany(
                    """INSERT INTO rag_parent_chunks
                       (parent_id, doc_id, profile_hash, parent_index, text, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            item["parent_id"],
                            doc_id,
                            profile_hash,
                            int(item["parent_index"]),
                            item["text"],
                            json.dumps(item.get("metadata") or {}, ensure_ascii=False),
                        )
                        for item in records
                    ],
                )

        await self._execute_write(_replace)

    async def delete_document(self, doc_id: str, profile_hash: str | None = None) -> None:
        def _delete() -> None:
            with self._conn() as conn:
                if profile_hash is None:
                    conn.execute("DELETE FROM rag_parent_chunks WHERE doc_id = ?", (doc_id,))
                else:
                    conn.execute(
                        "DELETE FROM rag_parent_chunks WHERE doc_id = ? AND profile_hash = ?",
                        (doc_id, profile_hash),
                    )

        await self._execute_write(_delete)

    async def delete_profile(self, profile_hash: str) -> None:
        """Remove all parent evidence for a disposable candidate generation."""
        def _delete() -> None:
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM rag_parent_chunks WHERE profile_hash = ?", (profile_hash,)
                )

        await self._execute_write(_delete)

    async def get_many(self, parent_ids: Iterable[str]) -> dict[str, dict]:
        ids = [parent_id for parent_id in dict.fromkeys(parent_ids) if parent_id]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)

        def _get() -> dict[str, dict]:
            with self._conn() as conn:
                rows = conn.execute(
                    f"""SELECT p.*,
                               (SELECT COUNT(*) FROM rag_parent_chunks AS siblings
                                WHERE siblings.doc_id = p.doc_id
                                  AND siblings.profile_hash = p.profile_hash) AS parent_count
                        FROM rag_parent_chunks AS p
                        WHERE p.parent_id IN ({placeholders})""",
                    tuple(ids),
                ).fetchall()
            result: dict[str, dict] = {}
            for row in rows:
                item = dict(row)
                try:
                    item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
                except json.JSONDecodeError:
                    item["metadata"] = {}
                result[item["parent_id"]] = item
            return result

        return await self._execute_read(_get)

    async def list_by_document(self, doc_id: str, profile_hash: str = "legacy") -> list[dict]:
        """Return the persisted parent evidence in original document order."""
        def _list() -> list[dict]:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT * FROM rag_parent_chunks
                       WHERE doc_id = ? AND profile_hash = ? ORDER BY parent_index ASC""",
                    (doc_id, profile_hash),
                ).fetchall()
            result: list[dict] = []
            for row in rows:
                item = dict(row)
                try:
                    item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
                except json.JSONDecodeError:
                    item["metadata"] = {}
                result.append(item)
            return result

        return await self._execute_read(_list)
