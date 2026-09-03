"""HTTP adapter for locally hosted cross-encoder rerankers."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import httpx

from app.providers.reranker.base import BaseReranker
from app.settings.settings import RerankerConfig


class HttpReranker(BaseReranker):
    """Uses POST /rerank with {query, documents} and indexed score results."""

    def __init__(self, config: RerankerConfig):
        self._config = config
        self._base_url = config.api_base.rstrip("/")
        self._url = f"{self._base_url}/rerank"

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        payload = await self._post_rerank(query, documents, timeout=self._config.request_timeout)
        return self._parse_scores(payload, len(documents))

    async def probe(self) -> dict[str, Any]:
        """Validate the actual scoring endpoint; metadata endpoints are optional."""
        try:
            return await asyncio.wait_for(
                self._probe_endpoints(),
                timeout=float(self._config.probe_timeout_seconds),
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"reranker probe exceeded {self._config.probe_timeout_seconds:g}s total deadline"
            ) from exc

    async def _probe_endpoints(self) -> dict[str, Any]:
        profile = await self._fetch_profile()
        payload = await self._post_rerank(
            "reranker capability probe",
            ["This document is used only to verify reranker availability."],
            timeout=self._config.startup_timeout,
        )
        self._parse_scores(payload, 1)
        if isinstance(payload, dict):
            model = payload.get("model") or payload.get("model_name")
            if isinstance(model, str) and model.strip():
                profile["model_name"] = model.strip()
        profile.setdefault("model_type", "reranker")
        return profile

    async def _post_rerank(self, query: str, documents: list[str], *, timeout: int) -> Any:
        request_timeout = max(1, int(timeout))
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            response = await client.post(
                self._url,
                json={"query": query, "documents": documents},
            )
            response.raise_for_status()
            return response.json()

    async def _fetch_profile(self) -> dict[str, Any]:
        endpoints = [f"{self._base_url}/models"]
        if not self._base_url.endswith("/v1"):
            endpoints.append(f"{self._base_url}/v1/models")
        timeout = max(1, int(self._config.startup_timeout))
        async with httpx.AsyncClient(timeout=timeout) as client:
            for endpoint in endpoints:
                try:
                    response = await client.get(endpoint)
                    response.raise_for_status()
                    profile = self._profile_from_payload(response.json())
                    if profile:
                        return profile
                except (httpx.HTTPError, ValueError):
                    continue
        return {}

    @staticmethod
    def _profile_from_payload(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        models = payload.get("data") or payload.get("models")
        if isinstance(models, list) and models:
            first = next((item for item in models if isinstance(item, dict)), {})
            if isinstance(first, dict):
                meta = first.get("meta") if isinstance(first.get("meta"), dict) else {}
                return {
                    "model_name": first.get("id") or first.get("name") or first.get("model"),
                    "model_type": first.get("type") or first.get("task") or "reranker",
                    "context_length": (
                        first.get("context_length")
                        or first.get("max_model_len")
                        or meta.get("n_ctx")
                        or meta.get("n_ctx_train")
                    ),
                    "num_layers": (
                        first.get("num_layers")
                        or first.get("layers")
                        or meta.get("n_layer")
                    ),
                }
        return {
            "model_name": payload.get("model") or payload.get("model_name") or payload.get("id"),
            "model_type": payload.get("type") or payload.get("task") or "reranker",
            "context_length": payload.get("context_length") or payload.get("max_model_len"),
            "num_layers": payload.get("num_layers") or payload.get("layers"),
        }

    def _parse_scores(self, payload: Any, document_count: int) -> list[float]:
        results = payload.get("results", payload) if isinstance(payload, dict) else payload
        scores = [float("-inf")] * document_count
        if not isinstance(results, list):
            raise ValueError("reranker response must contain a results list")
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            score = item.get("score", item.get("relevance_score"))
            if isinstance(index, int) and 0 <= index < len(scores) and score is not None:
                scores[index] = self._normalize_score(score)
        if all(score == float("-inf") for score in scores):
            raise ValueError("reranker response contains no usable scores")
        if any(score == float("-inf") for score in scores):
            raise ValueError("reranker response is missing one or more document scores")
        return scores

    def _normalize_score(self, value: Any) -> float:
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("reranker score must be finite")
        mode = self._config.score_mode.strip().lower()
        if mode == "logit":
            if score >= 0:
                return 1.0 / (1.0 + math.exp(-score))
            exp_score = math.exp(score)
            return exp_score / (1.0 + exp_score)
        if mode == "auto" and not 0.0 <= score <= 1.0:
            if score >= 0:
                return 1.0 / (1.0 + math.exp(-score))
            exp_score = math.exp(score)
            return exp_score / (1.0 + exp_score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("reranker probability score must be between 0 and 1")
        return score
