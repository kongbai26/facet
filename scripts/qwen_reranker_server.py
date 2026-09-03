"""HTTP adapter for Qwen/Qwen3-Reranker-0.6B.

Run after installing the optional dependency group:

    pip install -e ".[reranker]"
    RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B \\
      python -m uvicorn scripts.qwen_reranker_server:app --host 0.0.0.0 --port 7082

The RAG application only depends on the stable POST /rerank contract exposed
here; the Qwen-specific input template and score normalization stay isolated.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)


class QwenRerankerService:
    def __init__(self):
        self.model_name = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
        self.max_length = max(128, int(os.getenv("RERANKER_MAX_INPUT_TOKENS", "8192")))
        self.device = os.getenv("RERANKER_DEVICE", "") or None
        self._model: Any = None
        self._load_error = ""
        self._num_layers: int | None = None

    async def load(self) -> None:
        try:
            await asyncio.to_thread(self._load_sync)
        except Exception as exc:
            self._load_error = str(exc)
            raise

    def _load_sync(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "缺少 reranker 依赖。请执行 pip install -e '.[reranker]'"
            ) from exc

        kwargs: dict[str, Any] = {"max_length": self.max_length}
        if self.device:
            kwargs["device"] = self.device
        self._model = CrossEncoder(self.model_name, **kwargs)
        config = getattr(getattr(self._model, "model", None), "config", None)
        layers = getattr(config, "num_hidden_layers", None)
        self._num_layers = int(layers) if isinstance(layers, int) else None
        self._load_error = ""

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if self._model is None:
            raise RuntimeError(self._load_error or "reranker model is not loaded")
        return await asyncio.to_thread(self._score_sync, query, documents)

    def _score_sync(self, query: str, documents: list[str]) -> list[float]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("缺少 torch reranker 依赖") from exc
        pairs = [(query, document) for document in documents]
        scores = self._model.predict(pairs, activation_fn=torch.nn.Sigmoid())
        return [float(score) for score in scores]

    def profile(self) -> dict[str, Any]:
        return {
            "id": self.model_name,
            "type": "reranker",
            "context_length": self.max_length,
            "num_layers": self._num_layers,
            "loaded": self._model is not None,
            "error": self._load_error,
        }


service = QwenRerankerService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await service.load()
    yield


app = FastAPI(title="Qwen Reranker Adapter", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok" if service._model is not None else "error", **service.profile()}


@app.get("/models")
async def models() -> dict[str, list[dict[str, Any]]]:
    return {"data": [service.profile()]}


@app.post("/rerank")
async def rerank(request: RerankRequest) -> dict[str, Any]:
    try:
        scores = await service.rerank(request.query, request.documents)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="reranker unavailable") from exc
    return {
        "model": service.model_name,
        "results": [
            {"index": index, "score": score}
            for index, score in enumerate(scores)
        ],
    }
