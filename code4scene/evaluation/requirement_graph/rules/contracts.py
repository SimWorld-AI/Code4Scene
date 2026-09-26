"""Shared contracts for deterministic RequirementGraph rule evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from code4scene.evaluation.assertions import Check, check, unevaluated
from code4scene.evaluation.semantic_graph import SemanticGraph
from code4scene.evaluation.values import as_text

# Rule type -> internal deterministic family.  Families are retained as compact
# evidence labels; they are not separately registered verifiers.
FAMILIES: Mapping[str, str] = {
    "object_presence": "CNT",
    "object_count": "CNT",
    "forbidden_object": "CNT",
    "attribute_value": "ATR",
    "object_architecture_relation": "OAR",
    "object_object_relation": "OOR",
    "around": "OOR",
}


@dataclass(frozen=True)
class RuleEvaluationContext:
    """Frozen rule items together with the scoped candidate semantic graph."""

    graph: SemanticGraph


def item_id(item: Mapping[str, Any], family: str) -> str:
    return as_text(item.get("id")) or f"unnamed-{family.lower()}-rule"


def inconclusive(
    item: Mapping[str, Any],
    family: str,
    reason: str,
    observed: Mapping[str, Any] | None = None,
) -> Check:
    """Return an explicit unknown when evidence cannot decide a rule."""

    return unevaluated(f"{family.lower()}.{item_id(item, family)}", reason, observed)


def decide(
    item: Mapping[str, Any],
    family: str,
    ok: bool,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]] = (),
    reason: str | None = None,
) -> Check:
    return check(
        f"{family.lower()}.{item_id(item, family)}",
        ok,
        {**expected, "prompt_span": item.get("prompt_span")},
        observed,
        evidence,
        reason or f"deterministic rule {item_id(item, family)} does not hold",
    )


__all__ = [
    "FAMILIES",
    "RuleEvaluationContext",
    "decide",
    "inconclusive",
    "item_id",
]
