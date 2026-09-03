"""Small, dependency-free diagnostics objects for the retrieval pipeline.

The retrieval path intentionally does not depend on this module yet.  It gives
callers a stable way to collect diagnostic data while keeping candidate bodies
out of traces, logs, and future admin responses.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import time
from typing import Any, Iterator, Mapping, Sequence


_SCORE_KEYS = (
    "score",
    "retrieval_score",
    "rank_score",
    "vector_score",
    "bm25_score",
    "rrf_score",
    "fusion_score",
    "rerank_score",
    "exact_match_bonus",
)
_DOCUMENT_NAME_KEYS = ("document_name", "file_name", "filename", "source_name", "title")
_SAFE_DETAIL_TYPES = (str, int, float, bool, type(None))


def _truncate(value: str, limit: int) -> str:
    """Return a bounded preview without adding information not in *value*."""
    if limit < 1:
        return ""
    return value if len(value) <= limit else f"{value[:limit]}…"


def _safe_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return round(float(value), 6)
    return None


@dataclass(frozen=True)
class CandidateSummary:
    """A candidate descriptor that deliberately excludes chunk/parent text."""

    rank: int
    chunk_id: str | None = None
    doc_id: str | None = None
    document_name: str | None = None
    chunk_index: int | None = None
    block_kind: str | None = None
    scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: Mapping[str, Any], *, rank: int) -> "CandidateSummary":
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}

        def text_value(*keys: str) -> str | None:
            for key in keys:
                value = result.get(key, metadata.get(key))
                if value not in (None, ""):
                    return str(value)
            return None

        raw_index = result.get("chunk_index", metadata.get("chunk_index"))
        chunk_index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else None
        scores = {
            key: number
            for key in _SCORE_KEYS
            if (number := _safe_number(result.get(key))) is not None
        }
        return cls(
            rank=rank,
            chunk_id=text_value("chunk_id"),
            doc_id=text_value("doc_id"),
            document_name=text_value(*_DOCUMENT_NAME_KEYS),
            chunk_index=chunk_index,
            block_kind=text_value("block_kind", "kind"),
            scores=scores,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"rank": self.rank}
        for key in ("chunk_id", "doc_id", "document_name", "chunk_index", "block_kind"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.scores:
            data["scores"] = dict(self.scores)
        return data


@dataclass
class StageTrace:
    """One named retrieval stage, such as ``vector`` or ``reranker``."""

    name: str
    candidate_count: int | None = None
    duration_ms: float | None = None
    reason: str | None = None
    details: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    candidates: list[CandidateSummary] = field(default_factory=list)

    def set_candidates(self, candidates: Sequence[Mapping[str, Any]], *, limit: int = 5) -> "StageTrace":
        """Store a bounded, body-free view of candidates and their source rank."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        self.candidate_count = len(candidates)
        self.candidates = [
            CandidateSummary.from_result(candidate, rank=index)
            for index, candidate in enumerate(candidates[:limit], start=1)
        ]
        return self

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name}
        if self.candidate_count is not None:
            data["candidate_count"] = self.candidate_count
        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms
        if self.reason:
            data["reason"] = self.reason
        if self.details:
            data["details"] = dict(self.details)
        if self.candidates:
            data["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return data


@dataclass
class AttemptTrace:
    """Trace for one actual retrieval attempt/query within a request."""

    query: str
    index: int = 0
    scope_type: str | None = None
    reason: str | None = None
    stages: list[StageTrace] = field(default_factory=list)
    _started_at: float = field(default_factory=time.perf_counter, repr=False)
    _finished_at: float | None = field(default=None, repr=False)

    @property
    def duration_ms(self) -> float:
        end = self._finished_at if self._finished_at is not None else time.perf_counter()
        return round((end - self._started_at) * 1000, 3)

    def finish(self, reason: str | None = None) -> "AttemptTrace":
        self._finished_at = time.perf_counter()
        if reason is not None:
            self.reason = reason
        return self

    def record_stage(
        self,
        name: str,
        *,
        candidate_count: int | None = None,
        duration_ms: float | None = None,
        reason: str | None = None,
        details: Mapping[str, Any] | None = None,
        candidates: Sequence[Mapping[str, Any]] | None = None,
        candidate_limit: int = 5,
    ) -> StageTrace:
        """Append a stage; all details are constrained to simple safe scalars."""
        if not name:
            raise ValueError("stage name must not be empty")
        if candidate_count is not None and candidate_count < 0:
            raise ValueError("candidate_count must be non-negative")
        if duration_ms is not None and duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")
        safe_details = {
            str(key): value
            for key, value in (details or {}).items()
            if isinstance(value, _SAFE_DETAIL_TYPES)
        }
        stage = StageTrace(
            name=name,
            candidate_count=candidate_count,
            duration_ms=round(duration_ms, 3) if duration_ms is not None else None,
            reason=reason,
            details=safe_details,
        )
        if candidates is not None:
            stage.set_candidates(candidates, limit=candidate_limit)
        self.stages.append(stage)
        return stage

    @contextmanager
    def measure_stage(self, name: str, **kwargs: Any) -> Iterator[StageTrace]:
        """Time a stage while leaving exception handling to the caller."""
        started_at = time.perf_counter()
        stage = self.record_stage(name, **kwargs)
        try:
            yield stage
        finally:
            stage.duration_ms = round((time.perf_counter() - started_at) * 1000, 3)

    def to_dict(self, *, query_limit: int = 160) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "query": _truncate(self.query, query_limit),
            "duration_ms": self.duration_ms,
            "stages": [stage.to_dict() for stage in self.stages],
        }
        if self.scope_type:
            data["scope_type"] = self.scope_type
        if self.reason:
            data["reason"] = self.reason
        return data


@dataclass
class RetrievalTrace:
    """Request-scoped retrieval diagnostics suitable for an admin-only endpoint."""

    request_id: str | None = None
    scope_type: str | None = None
    reason: str | None = None
    attempts: list[AttemptTrace] = field(default_factory=list)
    _started_at: float = field(default_factory=time.perf_counter, repr=False)
    _finished_at: float | None = field(default=None, repr=False)

    @property
    def duration_ms(self) -> float:
        end = self._finished_at if self._finished_at is not None else time.perf_counter()
        return round((end - self._started_at) * 1000, 3)

    def begin_attempt(self, query: str, *, scope_type: str | None = None) -> AttemptTrace:
        attempt = AttemptTrace(query=query, index=len(self.attempts), scope_type=scope_type or self.scope_type)
        self.attempts.append(attempt)
        return attempt

    def finish(self, reason: str | None = None) -> "RetrievalTrace":
        self._finished_at = time.perf_counter()
        if reason is not None:
            self.reason = reason
        for attempt in self.attempts:
            if attempt._finished_at is None:
                attempt.finish()
        return self

    def to_dict(self, *, query_limit: int = 160) -> dict[str, Any]:
        data: dict[str, Any] = {
            "duration_ms": self.duration_ms,
            "attempt_count": len(self.attempts),
            "attempts": [attempt.to_dict(query_limit=query_limit) for attempt in self.attempts],
        }
        if self.request_id:
            data["request_id"] = self.request_id
        if self.scope_type:
            data["scope_type"] = self.scope_type
        if self.reason:
            data["reason"] = self.reason
        return data
