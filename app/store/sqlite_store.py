"""Shared SQLite helpers for metadata-backed stores."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

_WRITE_LOCKS: dict[str, asyncio.Lock] = {}


def _normalized_db_path(db_path: str) -> str:
    return str(Path(db_path).resolve())


def get_write_lock(db_path: str) -> asyncio.Lock:
    """Return one write lock per SQLite database file."""
    normalized = _normalized_db_path(db_path)
    if normalized not in _WRITE_LOCKS:
        _WRITE_LOCKS[normalized] = asyncio.Lock()
    return _WRITE_LOCKS[normalized]


class SQLiteStore:
    """Base class for stores sharing the same SQLite database."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._write_lock = get_write_lock(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> str:
        """Shared metadata database path for coordinated stores."""
        return self._db_path

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        self._configure_connection(conn)
        return conn

    async def _execute_write(self, operation: Callable[[], T]) -> T:
        async with self._write_lock:
            return await asyncio.to_thread(operation)

    async def _execute_read(self, operation: Callable[[], T]) -> T:
        return await asyncio.to_thread(operation)
