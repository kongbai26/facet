"""SQLite-backed conversation and message storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.store.sqlite_store import SQLiteStore

CREATE_CONVERSATION_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    tenant_id TEXT,
    title TEXT NOT NULL,
    knowledge_base_id TEXT,
    knowledge_scope TEXT NOT NULL DEFAULT 'all',
    knowledge_base_ids_json TEXT NOT NULL DEFAULT '[]',
    full_context_doc_id TEXT,
    grounding_mode TEXT NOT NULL DEFAULT 'auto',
    answer_quality_mode TEXT NOT NULL DEFAULT 'normal',
    llm_model TEXT NOT NULL DEFAULT '',
    thinking_effort TEXT NOT NULL DEFAULT '',
    stream_validation_mode TEXT NOT NULL DEFAULT 'realtime',
    stream_validation_mode_explicit INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_message_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'completed',
    grounding_mode TEXT NOT NULL DEFAULT 'knowledge',
    answer_quality_mode TEXT NOT NULL DEFAULT 'normal',
    evidence_status TEXT NOT NULL DEFAULT '',
    sources_json TEXT NOT NULL DEFAULT '[]',
    error_message TEXT NOT NULL DEFAULT '',
    seq INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    UNIQUE(conversation_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq
ON messages(conversation_id, seq);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _message_from_row(row) -> dict:
    data = dict(row)
    try:
        data["sources"] = json.loads(data.pop("sources_json") or "[]")
    except json.JSONDecodeError:
        data["sources"] = []
    return data


def _conversation_from_row(row) -> dict:
    data = dict(row)
    try:
        kb_ids = json.loads(data.get("knowledge_base_ids_json") or "[]")
    except json.JSONDecodeError:
        kb_ids = []
    data["knowledge_base_ids"] = [str(kb_id) for kb_id in kb_ids if kb_id]
    data.pop("knowledge_base_ids_json", None)
    return data


class ConversationStore(SQLiteStore):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(CREATE_CONVERSATION_TABLES_SQL)
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "tenant_id" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN tenant_id TEXT")
            if "knowledge_base_id" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN knowledge_base_id TEXT")
            if "knowledge_scope" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN knowledge_scope TEXT NOT NULL DEFAULT 'all'")
            if "knowledge_base_ids_json" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN knowledge_base_ids_json TEXT NOT NULL DEFAULT '[]'")
            if "full_context_doc_id" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN full_context_doc_id TEXT")
            if "grounding_mode" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN grounding_mode TEXT NOT NULL DEFAULT 'auto'")
            if "answer_quality_mode" not in columns:
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN answer_quality_mode TEXT NOT NULL DEFAULT 'normal'"
                )
            if "llm_model" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN llm_model TEXT NOT NULL DEFAULT ''")
            if "thinking_effort" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN thinking_effort TEXT NOT NULL DEFAULT ''")
            if "stream_validation_mode" not in columns:
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN stream_validation_mode TEXT NOT NULL DEFAULT 'realtime'"
                )
            if "stream_validation_mode_explicit" not in columns:
                # Older versions persisted ``validated`` as an invisible
                # default.  Upgrade that default to realtime once, while all
                # future user changes are recorded as explicit preferences.
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN stream_validation_mode_explicit "
                    "INTEGER NOT NULL DEFAULT 0"
                )
                conn.execute(
                    """UPDATE conversations
                       SET stream_validation_mode = 'realtime'
                       WHERE stream_validation_mode = 'validated'
                         AND stream_validation_mode_explicit = 0"""
                )
            legacy_rows = conn.execute(
                """SELECT conversation_id, knowledge_base_id, knowledge_scope, knowledge_base_ids_json
                   FROM conversations"""
            ).fetchall()
            for row in legacy_rows:
                try:
                    parsed_ids = [kb_id for kb_id in json.loads(row["knowledge_base_ids_json"] or "[]") if kb_id]
                except json.JSONDecodeError:
                    parsed_ids = []
                # New rows never retain a primary KB while declaring an all
                # scope.  That combination is the old single-KB schema after
                # columns were added with their defaults.
                legacy_kb_id = row["knowledge_base_id"]
                if (row["knowledge_scope"] == "selected" and parsed_ids) or (not legacy_kb_id and row["knowledge_scope"] == "all"):
                    continue
                scope = "selected" if legacy_kb_id else "all"
                ids_json = json.dumps([legacy_kb_id] if legacy_kb_id else [], ensure_ascii=False)
                conn.execute(
                    """UPDATE conversations
                       SET knowledge_scope = ?, knowledge_base_ids_json = ?
                       WHERE conversation_id = ?""",
                    (scope, ids_json, row["conversation_id"]),
                )
            message_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "grounding_mode" not in message_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN grounding_mode TEXT NOT NULL DEFAULT 'knowledge'")
            if "answer_quality_mode" not in message_columns:
                conn.execute(
                    "ALTER TABLE messages ADD COLUMN answer_quality_mode TEXT NOT NULL DEFAULT 'normal'"
                )
            if "evidence_status" not in message_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN evidence_status TEXT NOT NULL DEFAULT ''")
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
                ON conversations(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_conversations_tenant_id
                ON conversations(tenant_id);
                CREATE INDEX IF NOT EXISTS idx_conversations_tenant_kb
                ON conversations(tenant_id, knowledge_base_id);
                """
            )

    async def create_conversation(
        self,
        title: str = "新对话",
        tenant_id: str | None = None,
        knowledge_scope: str = "all",
        knowledge_base_ids: list[str] | None = None,
        full_context_doc_id: str | None = None,
        grounding_mode: str = "auto",
        llm_model: str = "",
        thinking_effort: str = "",
        stream_validation_mode: str = "realtime",
        answer_quality_mode: str = "normal",
    ) -> dict:
        conversation_id = str(uuid4())
        now = _now()
        normalized_kb_ids = list(dict.fromkeys(kb_id for kb_id in (knowledge_base_ids or []) if kb_id))
        if knowledge_scope == "all":
            normalized_kb_ids = []
        primary_kb_id = normalized_kb_ids[0] if len(normalized_kb_ids) == 1 else None

        def _create():
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO conversations
                       (conversation_id, tenant_id, title, knowledge_base_id, knowledge_scope, knowledge_base_ids_json,
                       full_context_doc_id, grounding_mode, answer_quality_mode, llm_model, thinking_effort,
                       stream_validation_mode, stream_validation_mode_explicit, created_at, updated_at,
                       last_message_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        conversation_id, tenant_id, title, primary_kb_id, knowledge_scope,
                        json.dumps(normalized_kb_ids, ensure_ascii=False), full_context_doc_id,
                        grounding_mode, answer_quality_mode, llm_model, thinking_effort,
                        stream_validation_mode, 1, now, now, now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM conversations WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                return _conversation_from_row(row)

        return await self._execute_write(_create)

    async def update_retrieval_scope(
        self,
        conversation_id: str,
        *,
        knowledge_scope: str,
        knowledge_base_ids: list[str] | None,
        full_context_doc_id: str | None,
        grounding_mode: str | None = None,
        answer_quality_mode: str | None = None,
        llm_model: str | None = None,
        thinking_effort: str | None = None,
        stream_validation_mode: str | None = None,
        tenant_id: str | None = None,
    ) -> dict:
        """Persist a conversation's retrieval boundary atomically."""
        now = _now()
        normalized_kb_ids = list(dict.fromkeys(kb_id for kb_id in (knowledge_base_ids or []) if kb_id))
        if knowledge_scope == "all":
            normalized_kb_ids = []
        primary_kb_id = normalized_kb_ids[0] if len(normalized_kb_ids) == 1 else None

        def _update():
            with self._conn() as conn:
                clauses = ["conversation_id = ?"]
                fields = [
                    "knowledge_base_id = ?",
                    "knowledge_scope = ?",
                    "knowledge_base_ids_json = ?",
                    "full_context_doc_id = ?",
                    "updated_at = ?",
                ]
                params: list[object] = [
                    primary_kb_id,
                    knowledge_scope,
                    json.dumps(normalized_kb_ids, ensure_ascii=False),
                    full_context_doc_id,
                    now,
                ]
                if grounding_mode is not None:
                    fields.append("grounding_mode = ?")
                    params.append(grounding_mode)
                if answer_quality_mode is not None:
                    fields.append("answer_quality_mode = ?")
                    params.append(answer_quality_mode)
                if llm_model is not None:
                    fields.append("llm_model = ?")
                    params.append(llm_model)
                if thinking_effort is not None:
                    fields.append("thinking_effort = ?")
                    params.append(thinking_effort)
                if stream_validation_mode is not None:
                    fields.append("stream_validation_mode = ?")
                    params.append(stream_validation_mode)
                    fields.append("stream_validation_mode_explicit = 1")
                params.append(conversation_id)
                if tenant_id is not None:
                    clauses.append("tenant_id = ?")
                    params.append(tenant_id)
                cursor = conn.execute(
                    f"""UPDATE conversations
                           SET {', '.join(fields)}
                           WHERE {' AND '.join(clauses)}""",
                    tuple(params),
                )
                if cursor.rowcount == 0:
                    raise KeyError(f"Conversation not found: {conversation_id}")
                row = conn.execute(
                    "SELECT * FROM conversations WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                return _conversation_from_row(row)

        return await self._execute_write(_update)

    async def list_conversations(self, limit: int = 100, tenant_id: str | None = None) -> list[dict]:
        def _list():
            with self._conn() as conn:
                if tenant_id is None:
                    rows = conn.execute(
                        """SELECT c.*,
                                  (SELECT COUNT(*) FROM messages m
                                   WHERE m.conversation_id = c.conversation_id) AS messages_count
                           FROM conversations c
                           ORDER BY COALESCE(c.last_message_at, c.updated_at) DESC
                           LIMIT ?""",
                        (limit,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT c.*,
                                  (SELECT COUNT(*) FROM messages m
                                   WHERE m.conversation_id = c.conversation_id) AS messages_count
                           FROM conversations c
                           WHERE c.tenant_id = ?
                           ORDER BY COALESCE(c.last_message_at, c.updated_at) DESC
                           LIMIT ?""",
                        (tenant_id, limit),
                    ).fetchall()
                return [_conversation_from_row(row) for row in rows]

        return await self._execute_read(_list)

    async def get_conversation(self, conversation_id: str, tenant_id: str | None = None) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                if tenant_id is None:
                    row = conn.execute(
                        "SELECT * FROM conversations WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT * FROM conversations WHERE conversation_id = ? AND tenant_id = ?",
                        (conversation_id, tenant_id),
                    ).fetchone()
                return _conversation_from_row(row) if row else None

        return await self._execute_read(_get)

    async def update_title(self, conversation_id: str, title: str, tenant_id: str | None = None) -> None:
        now = _now()

        def _update():
            with self._conn() as conn:
                if tenant_id is None:
                    cursor = conn.execute(
                        """UPDATE conversations
                           SET title = ?, updated_at = ?
                           WHERE conversation_id = ?""",
                        (title, now, conversation_id),
                    )
                else:
                    cursor = conn.execute(
                        """UPDATE conversations
                           SET title = ?, updated_at = ?
                           WHERE conversation_id = ? AND tenant_id = ?""",
                        (title, now, conversation_id, tenant_id),
                    )
                if cursor.rowcount == 0:
                    raise KeyError(f"Conversation not found: {conversation_id}")

        await self._execute_write(_update)

    async def delete_conversation(self, conversation_id: str, tenant_id: str | None = None) -> None:
        def _delete():
            with self._conn() as conn:
                if tenant_id is None:
                    cursor = conn.execute(
                        "DELETE FROM conversations WHERE conversation_id = ?",
                        (conversation_id,),
                    )
                else:
                    cursor = conn.execute(
                        "DELETE FROM conversations WHERE conversation_id = ? AND tenant_id = ?",
                        (conversation_id, tenant_id),
                    )
                if cursor.rowcount == 0:
                    raise KeyError(f"Conversation not found: {conversation_id}")

        await self._execute_write(_delete)

    async def delete_conversations(self, conversation_ids: list[str], tenant_id: str) -> int:
        """Delete a tenant-scoped batch atomically; messages cascade in SQLite."""
        ids = list(dict.fromkeys(str(item) for item in conversation_ids if item))
        if not ids:
            return 0

        def _delete() -> int:
            with self._conn() as conn:
                placeholders = ",".join("?" for _ in ids)
                cursor = conn.execute(
                    f"DELETE FROM conversations WHERE tenant_id = ? AND conversation_id IN ({placeholders})",
                    (tenant_id, *ids),
                )
                return int(cursor.rowcount)

        return await self._execute_write(_delete)

    async def detach_knowledge_base(self, kb_id: str, tenant_id: str) -> int:
        """Remove a deleted KB from conversation scopes without widening access."""
        now = _now()

        def _detach() -> int:
            changed = 0
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM conversations WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchall()
                for row in rows:
                    try:
                        selected_ids = [
                            str(item)
                            for item in json.loads(row["knowledge_base_ids_json"] or "[]")
                            if item
                        ]
                    except json.JSONDecodeError:
                        selected_ids = []
                    selected_ids = list(dict.fromkeys(selected_ids))
                    if kb_id not in selected_ids and row["knowledge_base_id"] != kb_id:
                        continue

                    remaining = [item for item in selected_ids if item != kb_id]
                    if row["knowledge_scope"] == "selected" and remaining:
                        conn.execute(
                            """UPDATE conversations
                               SET knowledge_base_id = ?, knowledge_base_ids_json = ?,
                                   full_context_doc_id = NULL, updated_at = ?
                               WHERE conversation_id = ? AND tenant_id = ?""",
                            (remaining[0] if len(remaining) == 1 else None,
                             json.dumps(remaining, ensure_ascii=False), now,
                             row["conversation_id"], tenant_id),
                        )
                    else:
                        conn.execute(
                            """UPDATE conversations
                               SET knowledge_base_id = NULL, knowledge_scope = 'all',
                                   knowledge_base_ids_json = '[]', full_context_doc_id = NULL,
                                   grounding_mode = 'assistant', updated_at = ?
                               WHERE conversation_id = ? AND tenant_id = ?""",
                            (now, row["conversation_id"], tenant_id),
                        )
                    changed += 1
            return changed

        return await self._execute_write(_detach)

    async def list_messages(self, conversation_id: str, tenant_id: str | None = None) -> list[dict]:
        def _list():
            with self._conn() as conn:
                if tenant_id is None:
                    rows = conn.execute(
                        """SELECT * FROM messages
                           WHERE conversation_id = ?
                           ORDER BY seq ASC""",
                        (conversation_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT m.*
                           FROM messages m
                           JOIN conversations c ON c.conversation_id = m.conversation_id
                           WHERE m.conversation_id = ? AND c.tenant_id = ?
                           ORDER BY m.seq ASC""",
                        (conversation_id, tenant_id),
                    ).fetchall()
                return [_message_from_row(row) for row in rows]

        return await self._execute_read(_list)

    async def get_message(self, message_id: str, tenant_id: str | None = None) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                if tenant_id is None:
                    row = conn.execute(
                        "SELECT * FROM messages WHERE message_id = ?",
                        (message_id,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """SELECT m.*
                           FROM messages m
                           JOIN conversations c ON c.conversation_id = m.conversation_id
                           WHERE m.message_id = ? AND c.tenant_id = ?""",
                        (message_id, tenant_id),
                    ).fetchone()
                return _message_from_row(row) if row else None

        return await self._execute_read(_get)

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        status: str = "completed",
        sources: Optional[list[dict]] = None,
        error_message: str = "",
        grounding_mode: str = "knowledge",
        evidence_status: str = "",
        tenant_id: str | None = None,
        answer_quality_mode: str = "normal",
    ) -> dict:
        message_id = str(uuid4())
        now = _now()
        sources_json = json.dumps(sources or [], ensure_ascii=False)

        def _append():
            with self._conn() as conn:
                if tenant_id is None:
                    conversation = conn.execute(
                        "SELECT * FROM conversations WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()
                else:
                    conversation = conn.execute(
                        "SELECT * FROM conversations WHERE conversation_id = ? AND tenant_id = ?",
                        (conversation_id, tenant_id),
                    ).fetchone()
                if not conversation:
                    raise KeyError(f"Conversation not found: {conversation_id}")

                row = conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                seq = int(row["next_seq"])
                conn.execute(
                    """INSERT INTO messages
                       (message_id, conversation_id, role, content, status, grounding_mode, answer_quality_mode,
                        evidence_status, sources_json, error_message, seq, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        message_id,
                        conversation_id,
                        role,
                        content,
                        status,
                        grounding_mode,
                        answer_quality_mode,
                        evidence_status,
                        sources_json,
                        error_message,
                        seq,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    """UPDATE conversations
                       SET updated_at = ?, last_message_at = ?
                       WHERE conversation_id = ?""",
                    (now, now, conversation_id),
                )
                row = conn.execute(
                    "SELECT * FROM messages WHERE message_id = ?",
                    (message_id,),
                ).fetchone()
                return _message_from_row(row)

        return await self._execute_write(_append)

    async def update_message(
        self,
        message_id: str,
        *,
        content: Optional[str] = None,
        status: Optional[str] = None,
        sources: Optional[list[dict]] = None,
        error_message: Optional[str] = None,
        grounding_mode: Optional[str] = None,
        answer_quality_mode: Optional[str] = None,
        evidence_status: Optional[str] = None,
    ) -> None:
        now = _now()

        def _update():
            fields = ["updated_at = ?"]
            values: list[object] = [now]
            if content is not None:
                fields.append("content = ?")
                values.append(content)
            if status is not None:
                fields.append("status = ?")
                values.append(status)
            if sources is not None:
                fields.append("sources_json = ?")
                values.append(json.dumps(sources, ensure_ascii=False))
            if error_message is not None:
                fields.append("error_message = ?")
                values.append(error_message)
            if grounding_mode is not None:
                fields.append("grounding_mode = ?")
                values.append(grounding_mode)
            if answer_quality_mode is not None:
                fields.append("answer_quality_mode = ?")
                values.append(answer_quality_mode)
            if evidence_status is not None:
                fields.append("evidence_status = ?")
                values.append(evidence_status)
            values.append(message_id)

            with self._conn() as conn:
                cursor = conn.execute(
                    f"UPDATE messages SET {', '.join(fields)} WHERE message_id = ?",
                    tuple(values),
                )
                if cursor.rowcount == 0:
                    raise KeyError(f"Message not found: {message_id}")
                row = conn.execute(
                    "SELECT conversation_id FROM messages WHERE message_id = ?",
                    (message_id,),
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                        (now, row["conversation_id"]),
                    )

        await self._execute_write(_update)

    async def delete_from_message(self, conversation_id: str, message_id: str) -> None:
        """Delete the selected message and all messages after it."""

        def _delete():
            with self._conn() as conn:
                row = conn.execute(
                    """SELECT seq FROM messages
                       WHERE conversation_id = ? AND message_id = ?""",
                    (conversation_id, message_id),
                ).fetchone()
                if not row:
                    raise KeyError(f"Message not found: {message_id}")
                conn.execute(
                    "DELETE FROM messages WHERE conversation_id = ? AND seq >= ?",
                    (conversation_id, row["seq"]),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                    (_now(), conversation_id),
                )

        await self._execute_write(_delete)

    async def mark_streaming_messages_stopped(self) -> None:
        now = _now()

        def _mark():
            with self._conn() as conn:
                conn.execute(
                    """UPDATE messages
                       SET status = 'stopped', updated_at = ?
                       WHERE status = 'streaming'""",
                    (now,),
                )

        await self._execute_write(_mark)

    async def assign_missing_tenant(self, tenant_id: str) -> int:
        now = _now()

        def _assign():
            with self._conn() as conn:
                cursor = conn.execute(
                    """UPDATE conversations
                       SET tenant_id = ?, updated_at = ?
                       WHERE tenant_id IS NULL OR tenant_id = ''""",
                    (tenant_id, now),
                )
                return cursor.rowcount

        return await self._execute_write(_assign)
