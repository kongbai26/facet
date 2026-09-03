"""Persistent ingest job storage."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from app.store.sqlite_store import SQLiteStore

CREATE_INGEST_JOBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    kb_id TEXT,
    doc_id TEXT,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT NOT NULL DEFAULT '',
    attempts INTEGER NOT NULL DEFAULT 0,
    locked_by TEXT,
    locked_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_status_created_at ON ingest_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_tenant_id ON ingest_jobs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_doc_id ON ingest_jobs(doc_id);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_active_batch_reindex
    ON ingest_jobs(tenant_id, job_type, status, created_at);
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


def _row_to_job(row) -> dict:
    data = dict(row)
    try:
        data["payload"] = json.loads(data.pop("payload_json") or "{}")
    except json.JSONDecodeError:
        data["payload"] = {}
    return data


def _merge_index_candidate_payload(
    existing_payload: dict,
    incoming_payload: Optional[dict],
    *,
    request_rerun: bool,
) -> dict:
    """Coalesce source-change intents without losing deletion or cutover requirements."""
    incoming = dict(incoming_payload or {})
    merged = dict(existing_payload or {})
    merged.update({key: value for key, value in incoming.items() if value is not None})
    merged["auto_activate"] = bool(
        existing_payload.get("auto_activate") or incoming.get("auto_activate")
    )

    reasons = [
        str(reason)
        for reason in [
            *(existing_payload.get("reasons") or []),
            existing_payload.get("reason"),
            *(incoming.get("reasons") or []),
            incoming.get("reason"),
        ]
        if reason
    ]
    if reasons:
        merged["reasons"] = list(dict.fromkeys(reasons))

    delete_doc_ids = [
        str(doc_id)
        for doc_id in [
            *(existing_payload.get("delete_doc_ids") or []),
            *(incoming.get("delete_doc_ids") or []),
        ]
        if doc_id
    ]
    if existing_payload.get("reason") == "document_delete" and existing_payload.get("doc_id"):
        delete_doc_ids.append(str(existing_payload["doc_id"]))
    if incoming.get("reason") == "document_delete" and incoming.get("doc_id"):
        delete_doc_ids.append(str(incoming["doc_id"]))
    if delete_doc_ids:
        merged["delete_doc_ids"] = list(dict.fromkeys(delete_doc_ids))
        merged["reason"] = "document_delete"

    document_names = dict(existing_payload.get("delete_document_names") or {})
    document_names.update(incoming.get("delete_document_names") or {})
    if incoming.get("reason") == "document_delete" and incoming.get("doc_id"):
        document_names[str(incoming["doc_id"])] = str(
            incoming.get("document_name") or incoming["doc_id"]
        )
    if document_names:
        merged["delete_document_names"] = document_names
    if request_rerun:
        merged["rerun_requested"] = True
    return merged


class IngestJobStore(SQLiteStore):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(CREATE_INGEST_JOBS_TABLE_SQL)
            self._migrate_schema(conn)

    def _migrate_schema(self, conn) -> None:
        duplicate_rows = conn.execute(
            """SELECT doc_id
               FROM ingest_jobs
               WHERE job_type = 'reindex'
                 AND status IN ('queued', 'running')
                 AND doc_id IS NOT NULL
               GROUP BY doc_id
               HAVING COUNT(1) > 1"""
        ).fetchall()
        now = _to_iso(_now_dt())
        for row in duplicate_rows:
            doc_id = row["doc_id"]
            jobs = conn.execute(
                """SELECT job_id, status
                   FROM ingest_jobs
                   WHERE doc_id = ? AND job_type = 'reindex' AND status IN ('queued', 'running')
                   ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at ASC""",
                (doc_id,),
            ).fetchall()
            for duplicate in jobs[1:]:
                conn.execute(
                    """UPDATE ingest_jobs
                       SET status = 'cancelled',
                           error_message = 'superseded duplicate reindex job',
                           locked_by = NULL,
                           locked_at = NULL,
                           updated_at = ?
                       WHERE job_id = ?""",
                    (now, duplicate["job_id"]),
                )
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_ingest_jobs_active_reindex_doc_unique
               ON ingest_jobs(doc_id)
               WHERE job_type = 'reindex'
                 AND doc_id IS NOT NULL
               AND status IN ('queued', 'running')"""
        )
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_ingest_jobs_active_index_candidate_kb_unique
               ON ingest_jobs(kb_id)
               WHERE job_type = 'index_candidate'
                 AND kb_id IS NOT NULL
               AND status IN ('queued', 'running')"""
        )
        duplicate_kb_delete_rows = conn.execute(
            """SELECT tenant_id, kb_id
               FROM ingest_jobs
               WHERE job_type = 'knowledge_base_delete'
                 AND status IN ('queued', 'running')
                 AND kb_id IS NOT NULL
               GROUP BY tenant_id, kb_id
               HAVING COUNT(1) > 1"""
        ).fetchall()
        for duplicate_group in duplicate_kb_delete_rows:
            jobs = conn.execute(
                """SELECT job_id, status
                   FROM ingest_jobs
                   WHERE tenant_id = ? AND kb_id = ?
                     AND job_type = 'knowledge_base_delete'
                     AND status IN ('queued', 'running')
                   ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at ASC""",
                (duplicate_group["tenant_id"], duplicate_group["kb_id"]),
            ).fetchall()
            for duplicate in jobs[1:]:
                conn.execute(
                    """UPDATE ingest_jobs
                       SET status = 'cancelled',
                           error_message = 'superseded duplicate knowledge base deletion job',
                           locked_by = NULL,
                           locked_at = NULL,
                           updated_at = ?
                       WHERE job_id = ?""",
                    (now, duplicate["job_id"]),
                )
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_ingest_jobs_active_kb_delete_unique
               ON ingest_jobs(tenant_id, kb_id)
               WHERE job_type = 'knowledge_base_delete'
                 AND kb_id IS NOT NULL
                 AND status IN ('queued', 'running')"""
        )

    async def create_job(
        self,
        tenant_id: str,
        job_type: str,
        *,
        kb_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        payload: Optional[dict] = None,
        status: str = "queued",
    ) -> dict:
        job_id = str(uuid4())
        now = _to_iso(_now_dt())
        payload_json = json.dumps(payload or {}, ensure_ascii=False)

        def _create():
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO ingest_jobs
                       (job_id, tenant_id, kb_id, doc_id, job_type, status, payload_json,
                        error_message, attempts, locked_by, locked_at, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, '', 0, NULL, NULL, ?, ?)""",
                    (job_id, tenant_id, kb_id, doc_id, job_type, status, payload_json, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM ingest_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                return _row_to_job(row)

        return await self._execute_write(_create)

    async def get_or_create_active_kb_job(
        self,
        tenant_id: str,
        job_type: str,
        *,
        kb_id: str,
        payload: Optional[dict] = None,
    ) -> dict:
        """Return one queued/running KB-scoped job, creating it atomically."""
        if not kb_id:
            raise ValueError("kb_id is required")
        job_id = str(uuid4())
        now = _to_iso(_now_dt())
        payload_json = json.dumps(payload or {}, ensure_ascii=False)

        def _claim() -> dict:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    """SELECT * FROM ingest_jobs
                       WHERE tenant_id = ? AND kb_id = ? AND job_type = ?
                         AND status IN ('queued', 'running')
                       ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at ASC
                       LIMIT 1""",
                    (tenant_id, kb_id, job_type),
                ).fetchone()
                if existing is not None:
                    if job_type == "index_candidate":
                        existing_job = _row_to_job(existing)
                        merged_payload = _merge_index_candidate_payload(
                            existing_job.get("payload") or {},
                            payload,
                            request_rerun=existing_job.get("status") == "running",
                        )
                        conn.execute(
                            """UPDATE ingest_jobs
                               SET payload_json = ?, updated_at = ?
                               WHERE job_id = ?""",
                            (
                                json.dumps(merged_payload, ensure_ascii=False),
                                now,
                                existing_job["job_id"],
                            ),
                        )
                        existing = conn.execute(
                            "SELECT * FROM ingest_jobs WHERE job_id = ?",
                            (existing_job["job_id"],),
                        ).fetchone()
                    return _row_to_job(existing)
                try:
                    conn.execute(
                        """INSERT INTO ingest_jobs
                           (job_id, tenant_id, kb_id, doc_id, job_type, status, payload_json,
                            error_message, attempts, locked_by, locked_at, created_at, updated_at)
                           VALUES (?, ?, ?, NULL, ?, 'queued', ?, '', 0, NULL, NULL, ?, ?)""",
                        (job_id, tenant_id, kb_id, job_type, payload_json, now, now),
                    )
                except Exception:
                    # The partial unique index may have won a concurrent
                    # insert between the query and INSERT. Return its winner.
                    existing = conn.execute(
                        """SELECT * FROM ingest_jobs
                           WHERE tenant_id = ? AND kb_id = ? AND job_type = ?
                             AND status IN ('queued', 'running')
                           ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at ASC
                           LIMIT 1""",
                        (tenant_id, kb_id, job_type),
                    ).fetchone()
                    if existing is None:
                        raise
                    return _row_to_job(existing)
                row = conn.execute("SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)).fetchone()
                return _row_to_job(row)

        return await self._execute_write(_claim)

    async def prepare_document_deletion(
        self,
        tenant_id: str,
        *,
        doc_id: str,
        candidate_payload: Optional[dict] = None,
    ) -> dict:
        """Fence source writes and atomically queue a cutover when an active index exists."""
        now = _to_iso(_now_dt())
        candidate_job_id = str(uuid4())

        def _prepare() -> dict:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                document = conn.execute(
                    "SELECT * FROM documents WHERE doc_id = ? AND tenant_id = ?",
                    (doc_id, tenant_id),
                ).fetchone()
                if document is None:
                    raise KeyError(f"document not found: {doc_id}")
                document_data = dict(document)
                if document_data.get("status") not in {
                    "ready", "failed", "delete_failed", "reindex_queued", "deleting",
                }:
                    raise ValueError("document_delete_status_conflict")

                kb_id = str(document_data.get("kb_id") or "")
                if kb_id:
                    knowledge_base = conn.execute(
                        "SELECT status FROM knowledge_bases WHERE kb_id = ? AND tenant_id = ?",
                        (kb_id, tenant_id),
                    ).fetchone()
                    if knowledge_base is None or knowledge_base["status"] != "active":
                        raise ValueError("knowledge_base_not_active")

                running_jobs = conn.execute(
                    """SELECT * FROM ingest_jobs
                       WHERE tenant_id = ? AND status = 'running'
                         AND job_type IN ('ingest', 'reindex', 'batch_ingest', 'batch_reindex')""",
                    (tenant_id,),
                ).fetchall()
                for running_job_row in running_jobs:
                    running_job = _row_to_job(running_job_row)
                    payload_doc_ids = {
                        str(item)
                        for item in (running_job.get("payload") or {}).get("doc_ids") or []
                        if item
                    }
                    if running_job.get("doc_id") == doc_id or doc_id in payload_doc_ids:
                        raise ValueError("document_source_job_running")

                conn.execute(
                    """UPDATE ingest_jobs
                       SET status = 'cancelled', error_message = 'document deleted',
                           locked_by = NULL, locked_at = NULL, updated_at = ?
                       WHERE doc_id = ? AND status = 'queued'""",
                    (now, doc_id),
                )
                conn.execute(
                    """UPDATE documents
                       SET status = 'deleting', status_reason = '', error_message = '', updated_at = ?
                       WHERE doc_id = ? AND tenant_id = ?""",
                    (now, doc_id, tenant_id),
                )

                has_active_index = False
                index_table = conn.execute(
                    """SELECT 1 FROM sqlite_master
                       WHERE type = 'table' AND name = 'knowledge_base_index_generations'"""
                ).fetchone()
                if kb_id and index_table is not None:
                    has_active_index = conn.execute(
                        """SELECT 1 FROM knowledge_base_index_generations
                           WHERE kb_id = ? AND status = 'active' LIMIT 1""",
                        (kb_id,),
                    ).fetchone() is not None

                job = None
                if has_active_index:
                    incoming_payload = {
                        **(candidate_payload or {}),
                        "kb_id": kb_id,
                        "doc_id": doc_id,
                        "document_name": document_data.get("filename") or doc_id,
                        "auto_activate": True,
                        "reason": "document_delete",
                    }
                    existing = conn.execute(
                        """SELECT * FROM ingest_jobs
                           WHERE tenant_id = ? AND kb_id = ? AND job_type = 'index_candidate'
                             AND status IN ('queued', 'running')
                           ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at ASC
                           LIMIT 1""",
                        (tenant_id, kb_id),
                    ).fetchone()
                    if existing is not None:
                        existing_job = _row_to_job(existing)
                        merged_payload = _merge_index_candidate_payload(
                            existing_job.get("payload") or {},
                            incoming_payload,
                            request_rerun=existing_job.get("status") == "running",
                        )
                        conn.execute(
                            "UPDATE ingest_jobs SET payload_json = ?, updated_at = ? WHERE job_id = ?",
                            (
                                json.dumps(merged_payload, ensure_ascii=False),
                                now,
                                existing_job["job_id"],
                            ),
                        )
                        existing = conn.execute(
                            "SELECT * FROM ingest_jobs WHERE job_id = ?",
                            (existing_job["job_id"],),
                        ).fetchone()
                        job = _row_to_job(existing)
                    else:
                        payload_json = json.dumps(
                            _merge_index_candidate_payload({}, incoming_payload, request_rerun=False),
                            ensure_ascii=False,
                        )
                        conn.execute(
                            """INSERT INTO ingest_jobs
                               (job_id, tenant_id, kb_id, doc_id, job_type, status, payload_json,
                                error_message, attempts, locked_by, locked_at, created_at, updated_at)
                               VALUES (?, ?, ?, NULL, 'index_candidate', 'queued', ?,
                                       '', 0, NULL, NULL, ?, ?)""",
                            (candidate_job_id, tenant_id, kb_id, payload_json, now, now),
                        )
                        row = conn.execute(
                            "SELECT * FROM ingest_jobs WHERE job_id = ?",
                            (candidate_job_id,),
                        ).fetchone()
                        job = _row_to_job(row)

                updated_document = conn.execute(
                    "SELECT * FROM documents WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()
                return {
                    "document": dict(updated_document),
                    "requires_candidate": has_active_index,
                    "job": job,
                }

        return await self._execute_write(_prepare)

    async def queue_knowledge_base_delete(
        self,
        tenant_id: str,
        *,
        kb_id: str,
        payload: Optional[dict] = None,
    ) -> dict:
        """Hide a KB and create its singleton delete job in one transaction."""
        if not kb_id:
            raise ValueError("kb_id is required")
        job_id = str(uuid4())
        now = _to_iso(_now_dt())
        payload_json = json.dumps(payload or {}, ensure_ascii=False)

        def _queue() -> dict:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                knowledge_base = conn.execute(
                    """SELECT kb_id, slug, status
                       FROM knowledge_bases
                       WHERE kb_id = ? AND tenant_id = ?""",
                    (kb_id, tenant_id),
                ).fetchone()
                if knowledge_base is None:
                    raise KeyError(f"knowledge base not found: {kb_id}")
                if knowledge_base["slug"] == "default":
                    raise ValueError("default_knowledge_base_not_deletable")
                if knowledge_base["status"] not in {"active", "deleting", "delete_failed"}:
                    raise ValueError("knowledge_base_not_deletable")

                conn.execute(
                    """UPDATE knowledge_bases
                       SET status = 'deleting', error_message = '', updated_at = ?
                       WHERE kb_id = ? AND tenant_id = ?""",
                    (now, kb_id, tenant_id),
                )
                existing = conn.execute(
                    """SELECT * FROM ingest_jobs
                       WHERE tenant_id = ? AND kb_id = ?
                         AND job_type = 'knowledge_base_delete'
                         AND status IN ('queued', 'running')
                       ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at ASC
                       LIMIT 1""",
                    (tenant_id, kb_id),
                ).fetchone()
                if existing is not None:
                    return _row_to_job(existing)

                conn.execute(
                    """INSERT INTO ingest_jobs
                       (job_id, tenant_id, kb_id, doc_id, job_type, status, payload_json,
                        error_message, attempts, locked_by, locked_at, created_at, updated_at)
                       VALUES (?, ?, ?, NULL, 'knowledge_base_delete', 'queued', ?,
                               '', 0, NULL, NULL, ?, ?)""",
                    (job_id, tenant_id, kb_id, payload_json, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM ingest_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                return _row_to_job(row)

        return await self._execute_write(_queue)

    async def get_active_job_for_doc(self, doc_id: str, job_type: str) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    """SELECT * FROM ingest_jobs
                       WHERE doc_id = ? AND job_type = ? AND status IN ('queued', 'running')
                       ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at ASC
                       LIMIT 1""",
                    (doc_id, job_type),
                ).fetchone()
                return _row_to_job(row) if row else None

        return await self._execute_read(_get)

    async def get_or_create_active_doc_job(
        self,
        tenant_id: str,
        job_type: str,
        *,
        doc_id: str,
        payload: Optional[dict] = None,
    ) -> dict:
        now = _to_iso(_now_dt())
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        job_id = str(uuid4())

        def _get_or_create():
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    """SELECT * FROM ingest_jobs
                       WHERE doc_id = ? AND job_type = ? AND status IN ('queued', 'running')
                       ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at ASC
                       LIMIT 1""",
                    (doc_id, job_type),
                ).fetchone()
                if existing:
                    return _row_to_job(existing)

                conn.execute(
                    """INSERT INTO ingest_jobs
                       (job_id, tenant_id, kb_id, doc_id, job_type, status, payload_json,
                        error_message, attempts, locked_by, locked_at, created_at, updated_at)
                       VALUES (?, ?, NULL, ?, ?, 'queued', ?, '', 0, NULL, NULL, ?, ?)""",
                    (job_id, tenant_id, doc_id, job_type, payload_json, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM ingest_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                return _row_to_job(row)

        return await self._execute_write(_get_or_create)

    async def get_or_create_active_bm25_job(
        self,
        tenant_id: str,
        *,
        collection_name: str,
    ) -> dict:
        """Atomically deduplicate one BM25 rebuild per tenant/collection."""
        if not collection_name:
            raise ValueError("collection_name is required")
        job_id = str(uuid4())
        now = _to_iso(_now_dt())
        payload_json = json.dumps({"collection_name": collection_name}, ensure_ascii=False)

        def _claim() -> dict:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(
                    """SELECT * FROM ingest_jobs
                       WHERE tenant_id = ? AND job_type = 'bm25_rebuild'
                         AND status IN ('queued', 'running')
                       ORDER BY created_at ASC""",
                    (tenant_id,),
                ).fetchall()
                for row in rows:
                    try:
                        payload = json.loads(row["payload_json"] or "{}")
                    except json.JSONDecodeError:
                        payload = {}
                    if payload.get("collection_name") == collection_name:
                        return _row_to_job(row)

                conn.execute(
                    """INSERT INTO ingest_jobs
                       (job_id, tenant_id, kb_id, doc_id, job_type, status, payload_json,
                        error_message, attempts, locked_by, locked_at, created_at, updated_at)
                       VALUES (?, ?, NULL, NULL, 'bm25_rebuild', 'queued', ?, '', 0, NULL, NULL, ?, ?)""",
                    (job_id, tenant_id, payload_json, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                return _row_to_job(row)

        return await self._execute_write(_claim)

    async def get_or_create_active_batch_reindex_job(
        self,
        tenant_id: str,
        *,
        doc_ids: list[str],
        payload: Optional[dict] = None,
    ) -> dict:
        """Atomically deduplicate startup/model-change batch reindex work.

        Recovery can run more than once before the previous worker finishes.
        The batch key is derived from the complete document set, so repeated
        recovery passes return the existing queued/running job instead of
        creating another job for the same snapshot.
        """
        normalized_doc_ids = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids if doc_id))
        if not normalized_doc_ids:
            raise ValueError("doc_ids is required")

        batch_key = hashlib.sha256(
            json.dumps(sorted(normalized_doc_ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        job_id = str(uuid4())
        now = _to_iso(_now_dt())
        job_payload = dict(payload or {})
        job_payload["doc_ids"] = normalized_doc_ids
        job_payload["batch_key"] = batch_key
        payload_json = json.dumps(job_payload, ensure_ascii=False)

        def _claim() -> dict:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(
                    """SELECT * FROM ingest_jobs
                       WHERE tenant_id = ? AND job_type = 'batch_reindex'
                         AND status IN ('queued', 'running')
                       ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END, created_at ASC""",
                    (tenant_id,),
                ).fetchall()
                requested = set(normalized_doc_ids)
                for row in rows:
                    try:
                        existing_payload = json.loads(row["payload_json"] or "{}")
                    except json.JSONDecodeError:
                        existing_payload = {}
                    existing_ids = {
                        str(doc_id)
                        for doc_id in (existing_payload.get("doc_ids") or [])
                        if doc_id
                    }
                    # Also recognize jobs created before batch_key was added.
                    if existing_payload.get("batch_key") == batch_key or requested <= existing_ids:
                        return _row_to_job(row)

                conn.execute(
                    """INSERT INTO ingest_jobs
                       (job_id, tenant_id, kb_id, doc_id, job_type, status, payload_json,
                        error_message, attempts, locked_by, locked_at, created_at, updated_at)
                       VALUES (?, ?, NULL, NULL, 'batch_reindex', 'queued', ?, '', 0, NULL, NULL, ?, ?)""",
                    (job_id, tenant_id, payload_json, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                return _row_to_job(row)

        return await self._execute_write(_claim)

    async def get(self, job_id: str) -> Optional[dict]:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM ingest_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                return _row_to_job(row) if row else None

        return await self._execute_read(_get)

    async def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        tenant_id: Optional[str] = None,
    ) -> list[dict]:
        def _list():
            query = "SELECT * FROM ingest_jobs"
            params: list[object] = []
            clauses = []
            if tenant_id is not None:
                clauses.append("tenant_id = ?")
                params.append(tenant_id)
            if status:
                clauses.append("status = ?")
                params.append(status)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY created_at ASC LIMIT ?"
            params.append(limit)

            with self._conn() as conn:
                rows = conn.execute(query, tuple(params)).fetchall()
                return [_row_to_job(row) for row in rows]

        return await self._execute_read(_list)

    async def claim_next_job(
        self,
        worker_id: str,
        *,
        max_attempts: int = 5,
        lock_timeout_seconds: int = 300,
    ) -> Optional[dict]:
        now = _now_dt()
        stale_before = _to_iso(now - timedelta(seconds=lock_timeout_seconds))
        now_iso = _to_iso(now)

        def _claim():
            with self._conn() as conn:
                # The in-process asyncio lock does not coordinate separate
                # Uvicorn/worker processes.  Selection and ownership update
                # must therefore happen in one SQLite write transaction.
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    """SELECT job_id FROM ingest_jobs
                       WHERE attempts < ?
                         AND (
                             status = 'queued'
                             OR (status = 'running' AND locked_at IS NOT NULL AND locked_at <= ?)
                         )
                       ORDER BY created_at ASC
                       LIMIT 1""",
                    (max_attempts, stale_before),
                ).fetchone()
                if not row:
                    return None

                job_id = row["job_id"]
                conn.execute(
                    """UPDATE ingest_jobs
                       SET status = 'running',
                           attempts = attempts + 1,
                           locked_by = ?,
                           locked_at = ?,
                           updated_at = ?
                       WHERE job_id = ?""",
                    (worker_id, now_iso, now_iso, job_id),
                )
                claimed = conn.execute(
                    "SELECT * FROM ingest_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                return _row_to_job(claimed)

        return await self._execute_write(_claim)

    async def mark_succeeded(
        self,
        job_id: str,
        *,
        locked_by: str | None = None,
        attempts: int | None = None,
    ) -> bool:
        now = _to_iso(_now_dt())

        def _update():
            with self._conn() as conn:
                conditions = ["job_id = ?"]
                params: list[object] = [job_id]
                if locked_by is not None and attempts is not None:
                    conditions.extend(["status = 'running'", "locked_by = ?", "attempts = ?"])
                    params.extend([locked_by, attempts])
                cursor = conn.execute(
                    f"""UPDATE ingest_jobs
                       SET status = 'succeeded', error_message = '', locked_by = NULL,
                           locked_at = NULL, updated_at = ?
                       WHERE {' AND '.join(conditions)}""",
                    (now, *params),
                )
                return cursor.rowcount > 0

        return await self._execute_write(_update)

    async def complete_index_candidate_iteration(
        self,
        job_id: str,
        *,
        locked_by: str | None = None,
        attempts: int | None = None,
    ) -> dict:
        """Atomically finish a candidate or consume a source-change rerun request."""
        now = _to_iso(_now_dt())

        def _complete() -> dict:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conditions = ["job_id = ?", "job_type = 'index_candidate'"]
                params: list[object] = [job_id]
                if locked_by is not None and attempts is not None:
                    conditions.extend(["status = 'running'", "locked_by = ?", "attempts = ?"])
                    params.extend([locked_by, attempts])
                row = conn.execute(
                    f"SELECT * FROM ingest_jobs WHERE {' AND '.join(conditions)}",
                    tuple(params),
                ).fetchone()
                if row is None:
                    return {"action": "lost", "job": None}

                current = _row_to_job(row)
                payload = dict(current.get("payload") or {})
                if payload.pop("rerun_requested", False):
                    conn.execute(
                        """UPDATE ingest_jobs
                           SET payload_json = ?, error_message = '', updated_at = ?
                           WHERE job_id = ?""",
                        (json.dumps(payload, ensure_ascii=False), now, job_id),
                    )
                    refreshed = conn.execute(
                        "SELECT * FROM ingest_jobs WHERE job_id = ?",
                        (job_id,),
                    ).fetchone()
                    return {"action": "rerun", "job": _row_to_job(refreshed)}

                conn.execute(
                    """UPDATE ingest_jobs
                       SET status = 'succeeded', error_message = '', locked_by = NULL,
                           locked_at = NULL, updated_at = ?
                       WHERE job_id = ?""",
                    (now, job_id),
                )
                refreshed = conn.execute(
                    "SELECT * FROM ingest_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                return {"action": "succeeded", "job": _row_to_job(refreshed)}

        return await self._execute_write(_complete)

    async def renew_lock(
        self,
        job_id: str,
        *,
        locked_by: str,
        attempts: int,
    ) -> bool:
        """Extend a running worker lease only for its current ownership."""
        now = _to_iso(_now_dt())

        def _renew() -> bool:
            with self._conn() as conn:
                cursor = conn.execute(
                    """UPDATE ingest_jobs
                       SET locked_at = ?, updated_at = ?
                       WHERE job_id = ? AND status = 'running'
                         AND locked_by = ? AND attempts = ?""",
                    (now, now, job_id, locked_by, attempts),
                )
                return cursor.rowcount > 0

        return await self._execute_write(_renew)

    async def mark_failed(
        self,
        job_id: str,
        error_message: str,
        *,
        locked_by: str | None = None,
        attempts: int | None = None,
    ) -> bool:
        now = _to_iso(_now_dt())

        def _update():
            with self._conn() as conn:
                conditions = ["job_id = ?"]
                params: list[object] = [job_id]
                if locked_by is not None and attempts is not None:
                    conditions.extend(["status = 'running'", "locked_by = ?", "attempts = ?"])
                    params.extend([locked_by, attempts])
                cursor = conn.execute(
                    f"""UPDATE ingest_jobs
                       SET status = 'failed', error_message = ?, locked_by = NULL,
                           locked_at = NULL, updated_at = ?
                       WHERE {' AND '.join(conditions)}""",
                    (error_message, now, *params),
                )
                return cursor.rowcount > 0

        return await self._execute_write(_update)

    async def mark_index_candidate_failed(
        self,
        job_id: str,
        error_message: str,
        *,
        locked_by: str | None = None,
        attempts: int | None = None,
    ) -> dict:
        """Fail a candidate and return its latest coalesced payload atomically."""
        now = _to_iso(_now_dt())

        def _fail() -> dict:
            with self._conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conditions = ["job_id = ?", "job_type = 'index_candidate'"]
                params: list[object] = [job_id]
                if locked_by is not None and attempts is not None:
                    conditions.extend(["status = 'running'", "locked_by = ?", "attempts = ?"])
                    params.extend([locked_by, attempts])
                row = conn.execute(
                    f"SELECT * FROM ingest_jobs WHERE {' AND '.join(conditions)}",
                    tuple(params),
                ).fetchone()
                if row is None:
                    return {"marked": False, "job": None}
                conn.execute(
                    """UPDATE ingest_jobs
                       SET status = 'failed', error_message = ?, locked_by = NULL,
                           locked_at = NULL, updated_at = ?
                       WHERE job_id = ?""",
                    (error_message, now, job_id),
                )
                failed = conn.execute(
                    "SELECT * FROM ingest_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                return {"marked": True, "job": _row_to_job(failed)}

        return await self._execute_write(_fail)

    async def cancel_jobs_for_doc(self, doc_id: str) -> int:
        """文档被删除时，取消其关联的 queued/running job（幂等）。"""
        now = _to_iso(_now_dt())

        def _cancel():
            with self._conn() as conn:
                cursor = conn.execute(
                    """UPDATE ingest_jobs
                       SET status = 'cancelled', error_message = 'document deleted',
                           locked_by = NULL, locked_at = NULL, updated_at = ?
                       WHERE doc_id = ? AND status IN ('queued', 'running')""",
                    (now, doc_id),
                )
                return cursor.rowcount

        return await self._execute_write(_cancel)

    async def cancel_queued_jobs_for_kb(self, kb_id: str, *, except_job_id: str | None = None) -> int:
        """Cancel queued KB jobs; running jobs remain fenced and observable."""
        now = _to_iso(_now_dt())

        def _cancel() -> int:
            with self._conn() as conn:
                clauses = ["kb_id = ?", "status = 'queued'"]
                params: list[object] = [kb_id]
                if except_job_id:
                    clauses.append("job_id != ?")
                    params.append(except_job_id)
                cursor = conn.execute(
                    f"""UPDATE ingest_jobs
                           SET status = 'cancelled', error_message = 'knowledge base deleted',
                               updated_at = ?
                           WHERE {' AND '.join(clauses)}""",
                    (now, *params),
                )
                return cursor.rowcount

        return await self._execute_write(_cancel)

    async def list_active_jobs_for_kb(self, kb_id: str, *, except_job_id: str | None = None) -> list[dict]:
        def _list() -> list[dict]:
            with self._conn() as conn:
                clauses = ["kb_id = ?", "status IN ('queued', 'running')"]
                params: list[object] = [kb_id]
                if except_job_id:
                    clauses.append("job_id != ?")
                    params.append(except_job_id)
                rows = conn.execute(
                    f"SELECT * FROM ingest_jobs WHERE {' AND '.join(clauses)} ORDER BY created_at ASC",
                    tuple(params),
                ).fetchall()
                return [_row_to_job(row) for row in rows]

        return await self._execute_read(_list)

    async def delete_terminal_jobs_for_kb(self, kb_id: str, *, except_job_id: str | None = None) -> int:
        def _delete() -> int:
            with self._conn() as conn:
                clauses = ["kb_id = ?", "status IN ('succeeded', 'failed', 'cancelled')"]
                params: list[object] = [kb_id]
                if except_job_id:
                    clauses.append("job_id != ?")
                    params.append(except_job_id)
                cursor = conn.execute(
                    f"DELETE FROM ingest_jobs WHERE {' AND '.join(clauses)}",
                    tuple(params),
                )
                return cursor.rowcount

        return await self._execute_write(_delete)

    async def clear_history(self, tenant_id: str) -> int:
        """清空已完成的历史任务，只保留 queued/running 任务。"""
        def _clear():
            with self._conn() as conn:
                cursor = conn.execute(
                    """DELETE FROM ingest_jobs
                       WHERE tenant_id = ?
                         AND status IN ('succeeded', 'failed', 'cancelled')""",
                    (tenant_id,),
                )
                return cursor.rowcount

        return await self._execute_write(_clear)

    async def prune_history(self, retention_days: int) -> int:
        """Delete aged terminal jobs without touching queued or running work."""
        cutoff = _to_iso(_now_dt() - timedelta(days=max(0, int(retention_days))))

        def _prune() -> int:
            with self._conn() as conn:
                cursor = conn.execute(
                    """DELETE FROM ingest_jobs
                       WHERE status IN ('succeeded', 'failed', 'cancelled') AND updated_at < ?""",
                    (cutoff,),
                )
                return cursor.rowcount

        return await self._execute_write(_prune)

    async def list_active_doc_ids(self, job_type: str) -> set[str]:
        def _list():
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT DISTINCT doc_id
                       FROM ingest_jobs
                       WHERE job_type = ? AND status IN ('queued', 'running') AND doc_id IS NOT NULL""",
                    (job_type,),
                ).fetchall()
                return {row["doc_id"] for row in rows if row["doc_id"]}

        return await self._execute_read(_list)
