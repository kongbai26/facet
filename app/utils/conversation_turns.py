"""Process-local coordination for persistent conversation turns.

The SQLite store serializes writes, but a chat turn also contains reads, history
construction and model generation.  These must remain ordered per conversation
to keep a later turn from observing an unfinished earlier turn.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

import anyio

from app.utils.runtime_errors import (
    ChatTurnDeadlineExceededError,
    ConversationTurnQueueTimeoutError,
    GenerationQueueTimeoutError,
)


@dataclass
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


_entries: dict[str, _LockEntry] = {}
_entries_guard = asyncio.Lock()
# A semaphore is bound to the event loop which created it.  TestClient,
# reloaders, and some ASGI deployments legitimately create more than one loop
# in one process; sharing a process-global semaphore across them can strand a
# later stream behind a cancelled earlier one.  Production retains one shared
# limiter per running loop and configured limit.
_generation_limiters: dict[tuple[int, int], asyncio.Semaphore] = {}


@asynccontextmanager
async def conversation_turn_lock(
    tenant_id: str | None,
    conversation_id: str | None,
    *,
    wait_timeout_seconds: float | None = None,
) -> AsyncIterator[None]:
    """Serialize persistent turns for one tenant-scoped conversation.

    A new conversation has no stable id before it is created, so it deliberately
    does not take a lock.  No other request can target it until its id is emitted.
    """
    if not conversation_id:
        yield
        return

    key = f"{tenant_id or ''}:{conversation_id}"
    async with _entries_guard:
        entry = _entries.get(key)
        if entry is None:
            entry = _LockEntry(lock=asyncio.Lock())
            _entries[key] = entry
        entry.users += 1

    acquired = False
    try:
        if wait_timeout_seconds is None:
            await entry.lock.acquire()
        else:
            timeout_seconds = max(0.01, float(wait_timeout_seconds))
            try:
                await asyncio.wait_for(entry.lock.acquire(), timeout=timeout_seconds)
            except asyncio.TimeoutError as exc:
                raise ConversationTurnQueueTimeoutError(timeout_seconds) from exc
        acquired = True
        yield
    finally:
        if acquired:
            entry.lock.release()
        async with _entries_guard:
            entry.users -= 1
            if entry.users == 0 and _entries.get(key) is entry:
                _entries.pop(key, None)


@asynccontextmanager
async def generation_slot(
    max_concurrent: int,
    *,
    wait_timeout_seconds: float | None = None,
) -> AsyncIterator[None]:
    """Bound concurrent retrieval/model work across conversations in this process.

    A semaphore without a queue budget can retain HTTP streams forever when a
    local model is overloaded.  Waiting is therefore bounded for foreground
    callers, while internal callers can deliberately opt out by passing
    ``None``.
    """
    loop = asyncio.get_running_loop()
    normalized_limit = max(1, int(max_concurrent))
    limiter_key = (id(loop), normalized_limit)
    # There is no await between lookup and insertion, so concurrent tasks on
    # this loop cannot race here. Avoiding a global asyncio.Lock also prevents
    # another cross-loop synchronization primitive.
    limiter = _generation_limiters.get(limiter_key)
    if limiter is None:
        limiter = asyncio.Semaphore(normalized_limit)
        _generation_limiters[limiter_key] = limiter
    acquired = False
    try:
        if wait_timeout_seconds is None:
            await limiter.acquire()
        else:
            timeout_seconds = max(0.01, float(wait_timeout_seconds))
            try:
                await asyncio.wait_for(limiter.acquire(), timeout=timeout_seconds)
            except asyncio.TimeoutError as exc:
                raise GenerationQueueTimeoutError(timeout_seconds) from exc
        acquired = True
        yield
    finally:
        if acquired:
            limiter.release()


@asynccontextmanager
async def foreground_generation_budget(
    max_concurrent: int,
    *,
    queue_wait_timeout_seconds: float,
    turn_timeout_seconds: float,
) -> AsyncIterator[None]:
    """Apply one end-to-end foreground budget around a generation slot.

    ``anyio.fail_after`` works on every Python version supported by FastAPI,
    unlike ``asyncio.timeout`` which was only added in Python 3.11.  The outer
    deadline intentionally includes queueing, retrieval, validation and model
    streaming so a request cannot outlive the UX contract through many small
    individually-bounded stages.
    """
    timeout_seconds = max(0.01, float(turn_timeout_seconds))
    try:
        with anyio.fail_after(timeout_seconds):
            async with generation_slot(
                max_concurrent,
                wait_timeout_seconds=queue_wait_timeout_seconds,
            ):
                yield
    except GenerationQueueTimeoutError:
        raise
    except TimeoutError as exc:
        raise ChatTurnDeadlineExceededError(timeout_seconds) from exc
