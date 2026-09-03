"""Security helpers for password hashing and env file updates."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from pathlib import Path

PASSWORD_HASH_PREFIX = "pbkdf2_sha256"
PASSWORD_HASH_ITERATIONS = 390_000
PASSWORD_SALT_BYTES = 16
AUTH_PLACEHOLDERS = {
    "",
    "change-me",
    "change-me-to-a-secret",
    "change-me-to-a-password",
    "change-me-to-another-secret",
    "your-secret-key",
    "your-password",
}


def is_placeholder_secret(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in AUTH_PLACEHOLDERS


def is_password_hash(value: str | None) -> bool:
    if not value:
        return False
    return value.startswith(f"{PASSWORD_HASH_PREFIX}$")


def _b64_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(PASSWORD_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return (
        f"{PASSWORD_HASH_PREFIX}"
        f"${PASSWORD_HASH_ITERATIONS}"
        f"${_b64_encode(salt)}"
        f"${_b64_encode(digest)}"
    )


def verify_password(candidate: str, stored_value: str | None) -> bool:
    if not candidate or not stored_value:
        return False
    if is_password_hash(stored_value):
        try:
            _, iterations_raw, salt_raw, digest_raw = stored_value.split("$", 3)
            iterations = int(iterations_raw)
        except (ValueError, TypeError):
            return False
        salt = _b64_decode(salt_raw)
        expected = _b64_decode(digest_raw)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            candidate.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(actual, expected)
    return hmac.compare_digest(candidate, stored_value)


def generate_session_secret() -> str:
    return secrets.token_urlsafe(48)


def resolve_session_secret(session_secret: str = "") -> str:
    if session_secret and not is_placeholder_secret(session_secret):
        return session_secret.strip()
    return ""


def update_env_file(env_path: Path, updates: dict[str, str | None]) -> None:
    if not env_path.exists():
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("", encoding="utf-8")

    lines = env_path.read_text(encoding="utf-8").splitlines()
    rendered: list[str] = []
    seen: set[str] = set()

    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            rendered.append(line)
            continue

        key, _, _ = line.partition("=")
        key = key.strip()
        if key not in updates:
            rendered.append(line)
            continue

        seen.add(key)
        value = updates[key]
        if value is not None:
            rendered.append(f"{key}={value}")

    for key, value in updates.items():
        if key not in seen and value is not None:
            rendered.append(f"{key}={value}")

    env_path.write_text("\n".join(rendered).rstrip() + "\n", encoding="utf-8")
