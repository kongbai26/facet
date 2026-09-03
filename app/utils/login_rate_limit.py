"""Small process-local guard for browser password guessing.

Facet is a single-process deployment by design.  This limiter intentionally
does not trust forwarded headers; a reverse proxy or VPN remains the boundary
for public deployments.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque


class LoginRateLimiter:
    def __init__(self) -> None:
        self._attempts: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def retry_after_seconds(self, key: str, *, limit: int, window_seconds: int) -> int:
        now = time.monotonic()
        async with self._lock:
            attempts = self._active_attempts(key, now, window_seconds)
            if len(attempts) < limit:
                return 0
            return max(1, math.ceil(window_seconds - (now - attempts[0])))

    async def record_failure(self, key: str, *, window_seconds: int) -> None:
        now = time.monotonic()
        async with self._lock:
            attempts = self._active_attempts(key, now, window_seconds)
            attempts.append(now)

    async def reset(self, key: str) -> None:
        async with self._lock:
            self._attempts.pop(key, None)

    def _active_attempts(self, key: str, now: float, window_seconds: int) -> deque[float]:
        attempts = self._attempts.setdefault(key, deque())
        cutoff = now - window_seconds
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()
        if not attempts:
            self._attempts.pop(key, None)
            attempts = self._attempts.setdefault(key, deque())
        return attempts


_login_rate_limiter = LoginRateLimiter()


def get_login_rate_limiter() -> LoginRateLimiter:
    return _login_rate_limiter
