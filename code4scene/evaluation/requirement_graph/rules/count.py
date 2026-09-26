"""Presence, count, and forbidden-object rules for RequirementGraph Stage 1.

The CNT family: presence, count, and the negative case. Counting is over
LOGICAL OBJECTS, not Actors — a table built from a top and four legs is one
table — because a prompt asking for one table is not satisfied or violated by
how many meshes the asset happens to contain.

`forbidden_object` is in the same family and is the reason the family earns a
verifier of its own: "and no cars" is a countable claim whose bound is zero,
and an agent that satisfies every positive rule while filling the plaza with
cars has not built the scene that was asked for.

An incomplete reading WITHHOLDS unless it cannot change the verdict. If a rule
needs at least one chair and four are already found, Actors the exporter left
uncategorised cannot make that false, so the rule is decided. If a rule needs
exactly four and four are found among Actors that might not be all of them,
it is not.

This module is internal to ``semantic_requirements``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from code4scene.evaluation.assertions import Check
from code4scene.evaluation.semantic_graph import object_evidence

from . import contracts
from .contracts import RuleEvaluationContext

FAMILY = "CNT"


def _bounds(item: Mapping[str, Any]) -> dict[str, Any]:
    if item.get("type") == "forbidden_object":
        return {"min_count": None, "max_count": 0, "exact_count": None}
    expected = item.get("expected") or {}
    bounds = {name: expected.get(name)
              for name in ("min_count", "max_count", "exact_count")}
    if item.get("type") == "object_presence" and all(
            value is None for value in bounds.values()):
        bounds["min_count"] = 1
    return bounds


def _checks(context: RuleEvaluationContext, item: Mapping[str, Any]) -> list[Check]:
    bounds = _bounds(item)
    if all(value is None for value in bounds.values()):
        # Every clause is `is None or ...`, so a rule with no bound is a
        # tautology that passes for any count. The case-spec vocabulary in
        # this repo spells the bound `count`, which is NOT one of these three
        # keys, so the mis-key is a live authoring mistake and not a
        # hypothetical.
        return [contracts.inconclusive(
            item, FAMILY,
            "the rule declares none of min_count, max_count or exact_count, "
            "so it asserts nothing and would pass for any observed count "
            "(note: the case-spec vocabulary spells this `count`)")]
    selection = context.graph.select(item.get("subject"))
    count = len(selection.objects)
    coverage = selection.coverage()

    # An unclassified remainder can only ADD objects, so it can only break a
    # lower bound's failure or an upper bound's pass.
    lower_already_met = (bounds["min_count"] is not None
                         and bounds["max_count"] is None
                         and bounds["exact_count"] is None
                         and count >= bounds["min_count"])
    upper_already_broken = (
        (bounds["max_count"] is not None and count > bounds["max_count"])
        or (bounds["exact_count"] is not None and count > bounds["exact_count"]))
    if not selection.complete and not (lower_already_met or upper_already_broken):
        return [contracts.inconclusive(
            item, FAMILY,
            f"the scene's category and grouping evidence is incomplete "
            f"({coverage['unclassified_object_count']} object(s) carry no "
            f"category), so a count of {count} could be an undercount",
            {"count": count, "selector_coverage": coverage})]

    ok = ((bounds["min_count"] is None or count >= bounds["min_count"])
          and (bounds["max_count"] is None or count <= bounds["max_count"])
          and (bounds["exact_count"] is None or count == bounds["exact_count"]))
    if not ok and item.get("visual_substitution_fallback") is True:
        return [contracts.inconclusive(
            item, FAMILY,
            "trusted structured identity did not cover the required count; "
            "visually equivalent replacements may use another asset identity",
            {"count": count, "count_unit": "logical_object",
             "selector_coverage": coverage})]
    return [contracts.decide(
        item, FAMILY, ok,
        {"selector": dict(selection.selector), **bounds},
        {"count": count, "count_unit": "logical_object",
         "selector_coverage": coverage},
        [object_evidence(entry) for entry in selection.objects],
        f"the scene holds {count} matching logical object(s), which does not "
        f"satisfy {contracts.item_id(item, FAMILY)}")]


def evaluate_item(
    context: RuleEvaluationContext, item: Mapping[str, Any]
) -> list[Check]:
    """Evaluate one frozen deterministic rule item."""

    return _checks(context, item)


__all__ = ["FAMILY", "evaluate_item"]
