"""SQLite-backed application preferences."""

from __future__ import annotations

from datetime import datetime, timezone

from app.prompt_profile import normalize_prompt_profile, PROMPT_PROFILE_AUTO
from app.providers.llm.thinking import normalize_thinking_mode
from app.store.sqlite_store import SQLiteStore

CREATE_APP_SETTINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS app_settings (
    setting_key TEXT PRIMARY KEY,
    setting_value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AppSettingsStore(SQLiteStore):
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(CREATE_APP_SETTINGS_TABLE_SQL)

    async def get_value(self, key: str) -> str | None:
        def _get():
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT setting_value FROM app_settings WHERE setting_key = ?",
                    (key,),
                ).fetchone()
                return None if row is None else str(row["setting_value"])

        return await self._execute_read(_get)

    async def set_value(self, key: str, value: str) -> None:
        now = _now_iso()

        def _set():
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO app_settings (setting_key, setting_value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(setting_key) DO UPDATE SET
                        setting_value = excluded.setting_value,
                        updated_at = excluded.updated_at
                    """,
                    (key, value, now),
                )

        await self._execute_write(_set)

    async def get_prompt_profile(self) -> str:
        value = await self.get_value("prompt_profile")
        return normalize_prompt_profile(value) if value is not None else PROMPT_PROFILE_AUTO

    async def set_prompt_profile(self, profile: str) -> str:
        normalized = normalize_prompt_profile(profile)
        await self.set_value("prompt_profile", normalized)
        return normalized

    async def get_llm_thinking_mode(self, default_mode: str = "auto") -> str:
        value = await self.get_value("llm_thinking_mode")
        return normalize_thinking_mode(value, default=default_mode)

    async def set_llm_thinking_mode(self, mode: str) -> str:
        normalized = normalize_thinking_mode(mode, default="auto")
        await self.set_value("llm_thinking_mode", normalized)
        return normalized
