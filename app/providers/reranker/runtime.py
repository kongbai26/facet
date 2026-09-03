"""Runtime availability, probing, and fail-open handling for rerankers."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any

from app.providers.reranker.base import BaseReranker
from app.settings.settings import RerankerConfig

logger = logging.getLogger(__name__)


class RerankerUnavailableError(RuntimeError):
    """Raised when a configured reranker is temporarily unavailable."""


def _error_text(exc: Exception) -> str:
    """Some HTTP client exceptions stringify to an empty string."""
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _normalized_model_name(value: str) -> str:
    normalized = (value or "").strip().lower().replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def _model_matches(expected: str, actual: str) -> bool:
    expected_name = _normalized_model_name(expected)
    actual_name = _normalized_model_name(actual)
    if not expected_name or not actual_name:
        return True
    return (
        expected_name == actual_name
        or expected_name in actual_name
        or actual_name in expected_name
    )


class RerankerRuntime(BaseReranker):
    """Keeps an optional HTTP reranker from becoming a RAG availability dependency."""

    def __init__(
        self,
        config: RerankerConfig,
        provider: BaseReranker | None,
        *,
        configuration_error: str | None = None,
    ):
        self._config = config
        self._provider = provider
        self._configuration_error = configuration_error
        self._active = False
        self._available = False
        self._last_error = configuration_error or ""
        self._last_probe_at: float | None = None
        self._failure_count = 0
        self._profile: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._probe_task: asyncio.Task[bool] | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    async def initialize(self) -> bool:
        """Run once during application startup without failing the application."""
        return await self.ensure_ready(force=True)

    def schedule_probe(self) -> bool:
        """Schedule an auto-recovery probe without delaying the caller."""
        if not self._is_enabled() or self._config.mode != "auto" or self._active:
            return False
        if self._probe_task is not None and not self._probe_task.done():
            return False
        now = time.monotonic()
        interval = max(1, int(self._config.reprobe_interval_seconds))
        if self._last_probe_at is not None and now - self._last_probe_at < interval:
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False

        task = loop.create_task(self.ensure_ready())
        self._probe_task = task

        def _release_probe(completed: asyncio.Task[bool]) -> None:
            if self._probe_task is completed:
                self._probe_task = None
            try:
                completed.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # ensure_ready normally absorbs failures
                logger.warning("reranker background probe failed: %s", _error_text(exc))

        task.add_done_callback(_release_probe)
        return True

    async def ensure_ready(self, *, force: bool = False) -> bool:
        """Probe an inactive service at a bounded interval and return active state."""
        if not self._is_enabled():
            return False
        if self._active:
            return True

        now = time.monotonic()
        # "on" is an explicit startup-only choice.  "auto" additionally
        # re-probes after a circuit opens so a recovered sidecar can return.
        if (
            self._config.mode == "on"
            and not force
            and self._last_probe_at is not None
        ):
            return False
        interval = max(1, int(self._config.reprobe_interval_seconds))
        if (
            not force
            and self._last_probe_at is not None
            and now - self._last_probe_at < interval
        ):
            return False

        async with self._lock:
            if self._active:
                return True
            now = time.monotonic()
            if (
                self._config.mode == "on"
                and not force
                and self._last_probe_at is not None
            ):
                return False
            if (
                not force
                and self._last_probe_at is not None
                and now - self._last_probe_at < interval
            ):
                return False
            self._last_probe_at = now
            if self._provider is None:
                self._available = False
                self._active = False
                return False

            probe = getattr(self._provider, "probe", None)
            try:
                profile = {}
                if callable(probe):
                    profile = probe()
                    if inspect.isawaitable(profile):
                        profile = await profile
                self._profile = dict(profile or {})
                actual_model = str(self._profile.get("model_name") or "")
                if (
                    self._config.strict_model_match
                    and self._config.expected_model
                    and not _model_matches(self._config.expected_model, actual_model)
                ):
                    raise RerankerUnavailableError(
                        "reranker model mismatch: "
                        f"expected={self._config.expected_model}, actual={actual_model or 'unknown'}"
                    )
                self._available = True
                self._active = True
                self._failure_count = 0
                self._last_error = ""
                logger.info(
                    "reranker active: model=%s endpoint=%s",
                    actual_model or self._config.expected_model or "unknown",
                    self._config.api_base,
                )
                return True
            except Exception as exc:
                self._available = False
                self._active = False
                self._last_error = _error_text(exc)
                logger.warning("reranker unavailable; using legacy retrieval: %s", self._last_error)
                return False

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not self._active:
            self.schedule_probe()
            raise RerankerUnavailableError(self._last_error or "reranker is inactive")
        if self._provider is None:
            raise RerankerUnavailableError(self._last_error or "reranker is not configured")
        try:
            return await self._provider.rerank(query, documents)
        except Exception as exc:
            await self.record_failure(exc)
            raise

    async def record_failure(self, exc: Exception) -> None:
        self._failure_count += 1
        self._last_error = _error_text(exc)
        limit = max(1, int(self._config.circuit_breaker_failures))
        if self._failure_count >= limit:
            self._active = False
            self._available = False
            self._last_probe_at = time.monotonic()
            logger.warning(
                "reranker circuit opened after %d failures; using legacy retrieval",
                self._failure_count,
            )
        else:
            logger.warning(
                "reranker request failed (%d/%d); current request will use legacy retrieval: %s",
                self._failure_count,
                limit,
                self._last_error,
            )

    def status(self) -> dict[str, Any]:
        return {
            "configured": self._is_enabled(),
            "active": self._active,
            "available": self._available,
            "mode": self._config.mode,
            "expected_model": self._config.expected_model,
            "model_name": self._profile.get("model_name") or "",
            "model_type": self._profile.get("model_type") or "",
            "context_length": self._profile.get("context_length"),
            "num_layers": self._profile.get("num_layers"),
            "last_error": self._last_error,
            "failure_count": self._failure_count,
            "probe_in_flight": bool(self._probe_task and not self._probe_task.done()),
        }

    def _is_enabled(self) -> bool:
        return bool(self._config.enabled and self._config.mode != "off")
