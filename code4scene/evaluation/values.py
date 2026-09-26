"""What counts as a value, before anything is scored on it.

A numeric string is not evidence — it is a number that lost its type on the
way — and the difference decides scores, so every atom asks these questions in
one place rather than each being slightly stricter than the last.
"""

from __future__ import annotations

import math
from typing import Any


def as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def nonnegative_finite(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None



def is_finite_json_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def validate_triplet(value: Any, name: str, errors: list[str]) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not is_finite_json_number(item) for item in value)
    ):
        errors.append(f"{name} must contain three finite JSON numbers")


__all__ = ["as_text", "is_finite_json_number", "nonnegative_finite", "validate_triplet"]
