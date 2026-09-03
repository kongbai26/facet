"""Structured API error helpers."""

from __future__ import annotations

from fastapi.responses import JSONResponse


def error_response(status_code: int, code: str, message: str, **extra) -> JSONResponse:
    payload = {"code": code, "message": message}
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=status_code, content=payload)
