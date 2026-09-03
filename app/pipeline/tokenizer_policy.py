"""Tokenizer capability policy shared by ingest, workers, and recovery."""

from __future__ import annotations


TOKENIZER_CAPABILITY_STATUS_REASON = "tokenizer_capability"


def is_tokenizer_capability_error(message: str) -> bool:
    """Whether an error means exact token counting was unavailable.

    This intentionally does not classify every generic embedding error as a
    tokenizer error.  The marker lets a later configuration change move only
    the affected source documents back into the durable reindex queue.
    """
    lowered = (message or "").lower()
    return "tokenizer" in lowered and any(
        marker in lowered
        for marker in (
            "verified",
            "exact",
            "精确",
            "已验证",
            "endpoint is not configured",
        )
    )


def is_recoverable_tokenizer_capability_failure(doc: dict, settings) -> bool:
    """Return true when a relaxed policy can safely retry a saved source."""
    if bool(getattr(settings.chunking, "require_exact_tokenizer", False)):
        return False
    if doc.get("status") != "failed":
        return False
    return (
        (doc.get("status_reason") or "") == TOKENIZER_CAPABILITY_STATUS_REASON
        or is_tokenizer_capability_error(str(doc.get("error_message") or ""))
    )
