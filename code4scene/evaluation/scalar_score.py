"""Migrate historical score evidence to the single conservative score format.

Only this compatibility reader knows historical endpoint field names. New
scorers calculate a scalar directly; unknown evidence receives no credit.
"""
from collections.abc import Mapping
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=4096)
def _endpoint(key: str, suffix: str) -> bool:
    return (key == suffix or key.endswith(f"score_{suffix}")
            or key in {f"validity_rate_{suffix}", f"mean_quality_on_valid_cases_{suffix}",
                       f"semantic_evidence_{suffix}"})


def without_intervals(value: Any) -> Any:
    """Return a copy with legacy scoring intervals replaced by scalar fields."""
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            if _endpoint(key, "upper_bound") or key in {"base_upper", "effective_upper"}:
                continue
            if _endpoint(key, "lower_bound"):
                scalar = "score" if key == "lower_bound" else key.removesuffix("_lower_bound")
                # The historical conservative endpoint is authoritative on migration.
                result[scalar] = without_intervals(child)
            elif key in {"base_lower", "effective_lower"}:
                result[key.replace("_lower", "_score")] = without_intervals(child)
            elif not (_endpoint(f"{key}_lower_bound", "lower_bound") and f"{key}_lower_bound" in value) and not (key == "score" and "lower_bound" in value):
                result[key] = without_intervals(child)
        return result
    if isinstance(value, tuple):
        return tuple(without_intervals(child) for child in value)
    if isinstance(value, list):
        return [without_intervals(child) for child in value]
    return value
