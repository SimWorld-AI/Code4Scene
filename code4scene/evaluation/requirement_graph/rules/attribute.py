"""Attribute rules for RequirementGraph Stage 1.

The ATR family: "a RED sofa", "a bench at least two metres long", "warm
lighting". One rule, one attribute, one subject selection, and three ways of
quantifying it — at least one subject matches, at least N do, or all of them
must.

Two refusals matter more than the comparison:

* an attribute the snapshot does not expose is `not_evaluated`, never a
  failure. The exporter records what it was asked to record, and a scene is
  not wrong because nobody exported its colours.
* a compound object whose members DISAGREE is `not_evaluated` too. A two-tone
  sofa has two answers to "what colour is it", and resolving that to whichever
  Actor was exported first is a coin flip dressed as a measurement.

This module is internal to ``semantic_requirements``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from code4scene.evaluation.assertions import Check
from code4scene.evaluation.semantic_graph import attribute_matches, object_evidence

from . import contracts
from .contracts import RuleEvaluationContext

FAMILY = "ATR"


def _checks(context: RuleEvaluationContext, item: Mapping[str, Any]) -> list[Check]:
    expected = item.get("expected") or {}
    field = expected.get("field")
    if not field:
        return [contracts.inconclusive(
            item, FAMILY, "the rule names no attribute field to read")]
    selection = context.graph.select(item.get("subject"))
    if not selection.objects:
        return [contracts.inconclusive(
            item, FAMILY,
            "the rule's subject selector matched no logical object, so there "
            "is nothing whose attribute could be read",
            {"subject_count": 0, "selector_coverage": selection.coverage()})]

    readings = [(entry, entry.value_at(str(field))) for entry in selection.objects]
    unread = [entry for entry, value in readings if not value["available"]]
    if unread:
        return [contracts.inconclusive(
            item, FAMILY,
            f"the candidate snapshot does not expose {field!r} for "
            f"{len(unread)} of the {len(readings)} selected object(s)",
            {"subject_count": len(readings),
             "missing_attribute_object_ids": [entry.id for entry in unread][:20]})]
    ambiguous = [(entry, value) for entry, value in readings if value["ambiguous"]]
    if ambiguous:
        return [contracts.inconclusive(
            item, FAMILY,
            f"{len(ambiguous)} compound object(s) expose conflicting values "
            f"for {field!r}, and one of them is not the object's answer",
            {"ambiguous_objects": [
                {"evaluator_object_id": entry.id, "values": value["values"]}
                for entry, value in ambiguous][:20]})]

    matching = [entry for entry, value in readings
                if attribute_matches(value["value"], expected)]
    all_subjects = expected.get("all_subjects") is True
    required = (len(readings) if all_subjects
                else int(expected.get("minimum_matching_count") or 1))
    # Uncategorised Actors never enter a category selection, so an incomplete
    # reading can only be MISSING subjects — the same evidence the count and
    # relation families refuse. A missing subject cannot break "at least N
    # match" once N already do, and cannot rescue "all subjects match" once
    # one selected subject has failed it; every other verdict could be flipped
    # by an object that never entered the population.
    verdict_assured = ((not all_subjects and len(matching) >= required)
                       or (all_subjects and len(matching) < len(readings)))
    if not selection.complete and not verdict_assured:
        coverage = selection.coverage()
        return [contracts.inconclusive(
            item, FAMILY,
            f"the scene's category and grouping evidence is incomplete "
            f"({coverage['unclassified_object_count']} object(s) carry no "
            f"category), so the selected subjects may not be all of them and "
            f"a quantified attribute claim over the survivors is not decided",
            {"subject_count": len(readings), "matching_count": len(matching),
             "selector_coverage": coverage})]
    return [contracts.decide(
        item, FAMILY, len(matching) >= required,
        {"selector": dict(selection.selector), "attribute": dict(expected),
         "minimum_matching_count": required},
        {"subject_count": len(readings), "matching_count": len(matching)},
        [{**object_evidence(entry), "value": value["value"],
          "matches": entry in matching} for entry, value in readings],
        f"{len(matching)} of {len(readings)} selected object(s) satisfy "
        f"{field!r}, and {required} were required")]


def evaluate_item(
    context: RuleEvaluationContext, item: Mapping[str, Any]
) -> list[Check]:
    """Evaluate one frozen deterministic rule item."""

    return _checks(context, item)


__all__ = ["FAMILY", "evaluate_item"]
