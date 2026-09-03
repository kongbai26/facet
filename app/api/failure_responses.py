"""One public failure contract for foreground API work.

Chat, retrieval tools, and diagnostics can all acquire the same foreground
generation slot.  They must therefore report queue, deadline, and provider
failures consistently instead of making a side endpoint fall through to an
opaque framework 500 response.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse

from app.providers.llm.errors import failure_event_payload
from app.utils.user_errors import sanitize_user_error_message


def public_failure_payload(
    exc: BaseException,
    *,
    fallback: str,
    partial_output: bool = False,
) -> dict[str, str | bool]:
    """Return a redacted, typed payload suitable for JSON or SSE responses."""
    payload = dict(failure_event_payload(exc, fallback=fallback))
    if payload["code"] == "unknown":
        payload["error"] = sanitize_user_error_message(exc, str(payload["error"]))
    if partial_output:
        payload["partial_output"] = True
    return payload


def public_failure_response(
    exc: BaseException,
    *,
    fallback: str,
    non_retryable_status: int = 500,
    partial_output: bool = False,
) -> JSONResponse:
    """Build the stable JSON equivalent of the SSE failure contract."""
    payload = public_failure_payload(
        exc,
        fallback=fallback,
        partial_output=partial_output,
    )
    return JSONResponse(
        status_code=503 if payload["retryable"] else non_retryable_status,
        content={
            "error": payload["error"],
            "error_code": payload["code"],
            "message": fallback,
        },
    )
