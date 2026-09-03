"""OpenAI 兼容 Embedding 实现"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from app.providers.embedding.base import BaseEmbeddingProvider
from app.settings.settings import OpenAIEmbeddingConfig, VectorStoreConfig

logger = logging.getLogger(__name__)

def _is_retryable_status_error(exc: APIStatusError) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code == 429 or (isinstance(status_code, int) and status_code >= 500)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, config: OpenAIEmbeddingConfig, vectorstore_config: VectorStoreConfig):
        self.config = config
        self._cache_dir = Path(vectorstore_config.persist_dir)
        self._dimension: Optional[int] = None
        self._runtime_profile: Optional[dict] = None
        self._client = AsyncOpenAI(
            api_key=config.api_key or None,
            base_url=config.api_base or None,
            timeout=config.request_timeout,
            # Retry policy is implemented by _retry_with_backoff below.
            # Disable the SDK's hidden retries so request limits remain real.
            max_retries=0,
        )
        # asyncio synchronization primitives become loop-bound after waiting.
        # A provider singleton can legitimately outlive a TestClient loop or
        # a development reload, so retain an equivalent limiter per loop.
        self._semaphores: dict[int, asyncio.Semaphore] = {}

    def _request_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        key = id(loop)
        semaphore = self._semaphores.get(key)
        if semaphore is None:
            semaphore = asyncio.Semaphore(max(1, int(self.config.concurrent_requests)))
            self._semaphores[key] = semaphore
        return semaphore

    async def _retry_with_backoff(self, request, *, total_timeout: float):
        """Retry inside one wall-clock budget instead of multiplying timeouts."""
        max_attempts = max(1, int(self.config.max_attempts))
        base_delay = max(0.0, float(self.config.retry_backoff_seconds))
        budget = max(0.1, float(total_timeout))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget
        for attempt in range(max_attempts):
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("embedding request exceeded its logical request budget")
            attempt_ratio = (attempt + 1) / max_attempts
            attempt_timeout = max(0.1, min(budget * attempt_ratio, remaining))
            try:
                return await asyncio.wait_for(
                    request(attempt_timeout),
                    timeout=attempt_timeout,
                )
            except (RateLimitError, APIStatusError) as exc:
                if isinstance(exc, APIStatusError) and not _is_retryable_status_error(exc):
                    raise
                if attempt == max_attempts - 1:
                    raise
                delay = min(
                    base_delay * (2 ** attempt),
                    max(0.0, deadline - loop.time()),
                )
                if delay <= 0:
                    raise TimeoutError(
                        "embedding request exceeded its logical request budget"
                    ) from exc
                logger.warning(
                    "Embedding request error %s, retrying in %.1fs (attempt %d/%d)",
                    getattr(exc, "status_code", "rate-limit"),
                    delay,
                    attempt + 1,
                    max_attempts,
                )
                await asyncio.sleep(delay)
            except (APIConnectionError, APITimeoutError, TimeoutError) as exc:
                if attempt == max_attempts - 1:
                    raise
                delay = min(
                    base_delay * (2 ** attempt),
                    max(0.0, deadline - loop.time()),
                )
                if delay <= 0:
                    raise TimeoutError(
                        "embedding request exceeded its logical request budget"
                    ) from exc
                logger.warning(
                    "Embedding connection/timeout error, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    max_attempts,
                )
                await asyncio.sleep(delay)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        async with self._request_semaphore():
            resp = await self._retry_with_backoff(
                lambda timeout: self._client.embeddings.create(
                    model=self.config.model_name,
                    input=texts,
                    timeout=timeout,
                ),
                total_timeout=self.config.request_timeout,
            )
            return [item.embedding for item in resp.data]

    async def embed_query(self, query: str) -> List[float]:
        async with self._request_semaphore():
            resp = await self._retry_with_backoff(
                lambda timeout: self._client.embeddings.create(
                    model=self.config.model_name,
                    input=[query],
                    timeout=timeout,
                ),
                total_timeout=self.config.query_timeout_seconds,
            )
            return resp.data[0].embedding

    async def dimension(self) -> int:
        if self._dimension is None:
            cache_file = self._cache_dir / ".dimension_cache.json"
            cached_dimension: Optional[int] = None
            if cache_file.exists():
                try:
                    cache = await asyncio.to_thread(
                        lambda: json.loads(cache_file.read_text(encoding="utf-8"))
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    logger.warning("embedding 维度缓存损坏，将重新探测: %s", cache_file)
                    cache = {}
                if isinstance(cache, dict) and cache.get("model") == self.config.model_name:
                    try:
                        cached_dimension = int(cache["dimension"])
                    except (TypeError, ValueError):
                        cached_dimension = None

            # The same model name can point at a newly loaded quantization or
            # server revision with a different dimension. Verify the cache on
            # first use so a stale collection name cannot receive wrong-sized
            # vectors after a model restart.
            try:
                result = await self.embed_query("dimension_probe")
                live_dimension = len(result)
            except Exception:
                if cached_dimension is None:
                    raise
                logger.warning(
                    "embedding 维度探测失败，暂时使用缓存维度=%d",
                    cached_dimension,
                )
                self._dimension = cached_dimension
                return self._dimension

            self._dimension = live_dimension
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                cache_file.write_text,
                json.dumps({"model": self.config.model_name, "dimension": self._dimension}),
            )
        return self._dimension

    def _tokenize_endpoint(self) -> str:
        """Resolve llama.cpp's tokenizer endpoint from an OpenAI base URL."""
        explicit = (getattr(self.config, "tokenizer_endpoint", "") or "").strip()
        if explicit:
            return explicit.rstrip("/")

        parsed = urlsplit((self.config.api_base or "").rstrip("/"))
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/tokenize", "", ""))

    async def _tokenize(self, content) -> list[int]:
        endpoint = self._tokenize_endpoint()
        if not endpoint:
            raise NotImplementedError("embedding tokenizer endpoint is not configured")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        timeout = httpx.Timeout(self.config.tokenizer_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await asyncio.wait_for(
                client.post(endpoint, headers=headers, json={"content": content}),
                timeout=float(self.config.tokenizer_timeout_seconds),
            )
            response.raise_for_status()
        payload = response.json()
        tokens = payload.get("tokens") if isinstance(payload, dict) else None
        if not isinstance(tokens, list):
            raise RuntimeError("embedding tokenizer response does not contain tokens")
        return tokens

    async def count_tokens(self, text: str) -> int:
        """Return exact llama.cpp tokenizer output for one embedding payload."""
        return len(await self._tokenize(text))

    async def count_batch_tokens(self, texts: List[str]) -> int:
        """Count a whole llama.cpp tokenizer request without heuristic sums."""
        return len(await self._tokenize(texts))

    async def runtime_profile(self) -> dict:
        """读取 OpenAI 兼容服务返回的实际模型能力。"""
        if self._runtime_profile is not None:
            return dict(self._runtime_profile)

        dimension = await self.dimension()
        profile = {
            "model_name": self.config.model_name,
            "dimension": dimension,
            "context_window": int(self.config.max_tokens),
            "tokenizer_id": "",
            "tokenizer_version": "",
            "tokenizer_verified": False,
        }
        try:
            models = await self._client.models.list()
            entries = list(getattr(models, "data", []) or [])
            selected = next(
                (item for item in entries if getattr(item, "id", None) == self.config.model_name),
                entries[0] if entries else None,
            )
            actual_model_name = getattr(selected, "id", None) if selected is not None else None
            if actual_model_name:
                profile["model_name"] = str(actual_model_name)
                if actual_model_name != self.config.model_name:
                    profile["configured_model_name"] = self.config.model_name
                    logger.warning(
                        "embedding 配置模型=%s 与服务实际模型=%s 不一致；画像以实际模型为准",
                        self.config.model_name,
                        actual_model_name,
                    )
            meta = getattr(selected, "meta", None) if selected is not None else None
            if isinstance(meta, dict):
                context_window = meta.get("n_ctx") or meta.get("context_window")
                trained_context = meta.get("n_ctx_train") or meta.get("max_position_embeddings")
                if context_window:
                    profile["context_window"] = int(context_window)
                if trained_context:
                    profile["trained_context_window"] = int(trained_context)
                if meta.get("n_embd") and int(meta["n_embd"]) != dimension:
                    logger.warning(
                        "embedding /models n_embd=%s 与实测维度=%s 不一致，以实测维度为准",
                        meta.get("n_embd"), dimension,
                    )
        except Exception as exc:
            logger.info("无法读取 embedding 模型能力元数据，使用配置上下限: %s", exc)

        try:
            await self.count_tokens("tokenizer profile probe")
            profile["tokenizer_id"] = "llama.cpp:/tokenize"
            profile["tokenizer_verified"] = True
        except Exception as exc:
            logger.info("embedding 服务未提供可验证 tokenizer，将仅保留兼容估算: %s", exc)

        self._runtime_profile = profile
        logger.info(
            "embedding runtime profile: model=%s dimension=%s context_window=%s",
            profile["model_name"], profile["dimension"], profile["context_window"],
        )
        return dict(profile)
