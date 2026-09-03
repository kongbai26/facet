"""Filesystem helpers for document lifecycle operations."""

from __future__ import annotations

import shutil
from pathlib import Path


def remove_dir_strict(path: Path) -> None:
    """Remove a directory and fail if it still exists afterwards."""
    if path.exists():
        shutil.rmtree(path)
    if path.exists():
        raise RuntimeError(f"directory still exists after delete: {path}")
