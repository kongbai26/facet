"""Threshold checks for offline release reports."""

from __future__ import annotations

from typing import Mapping


def validate_minimum_metrics(
    metrics: Mapping[str, float],
    minimums: Mapping[str, float],
) -> list[str]:
    """Return readable failures; unknown metrics fail closed."""
    failures: list[str] = []
    for name, required in minimums.items():
        actual = metrics.get(name)
        if actual is None:
            failures.append(f"missing metric: {name}")
        elif float(actual) < float(required):
            failures.append(f"{name}={float(actual):.4f} < required {float(required):.4f}")
    return failures
