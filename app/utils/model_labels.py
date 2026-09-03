"""Safe labels for model metadata returned to clients."""

from __future__ import annotations

import re


def display_model_name(value: str | None) -> str:
    """Return a model file/ID without exposing local or remote path details."""
    text = str(value or "").strip()
    if not text:
        return ""
    return re.split(r"[\\/]", text)[-1] or text
