"""Object-to-object relation rules for RequirementGraph Stage 1.

The OOR family. Most rules delegate to the shared relation vocabulary — near,
against, on top of, facing — evaluated between LOGICAL OBJECTS rather than
Actors, so "the chair is near the table" is about the chair, not about one of
its legs.

`around` is the family's own rule and the reason it is not just a wrapper over
`spatial_relations`. "The chairs surround the table" is not a
distance claim: four chairs in a row on one side satisfy every count and
distance rule and surround nothing. So it measures ANGULAR COVERAGE — the
circle around the object is cut into sectors and the rule asks how many the
subjects occupy — alongside a distance bound that scales with the object's own
footprint, because "around" means something different for a coffee table than
for a roundabout.

This module is internal to ``semantic_requirements``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from code4scene.evaluation import spatial_relations as relations
from code4scene.evaluation.assertions import Check
from code4scene.evaluation.scene_geometry import center_cm, extent_cm, rounded
from code4scene.evaluation.semantic_graph import object_evidence

from . import contracts
from .contracts import RuleEvaluationContext

FAMILY = "OOR"

#: How many sectors the circle is cut into for `around`. Eight is the coarsest
#: division that can tell "on three sides" from "all round".
SECTORS = 8
#: Default share of those sectors that must be occupied.
DEFAULT_MINIMUM_COVERAGE = 0.5
#: How far out an "around" subject may sit, as a multiple of the object's own
#: horizontal half-size. Object-relative because a chair 1.5 m from a coffee
#: table is around it and 1.5 m from a roundabout is on it.
DEFAULT_DISTANCE_FACTOR = 3.0


def _sides(context: RuleEvaluationContext, item: Mapping[str, Any]) -> tuple[Any, Any]:
    return (context.graph.select(item.get("subject")),
            context.graph.select(item.get("object")))


def _around(item: Mapping[str, Any], subjects, objects) -> list[Check]:
    expected = item.get("expected") or {}
    if len(objects.objects) != 1:
        return [contracts.inconclusive(
            item, FAMILY,
            f"`around` needs exactly one object to be around; the selector "
            f"matched {len(objects.objects)}",
            {"object_count": len(objects.objects)})]
    centre_object = objects.objects[0]
    centre = center_cm(centre_object.as_actor())
    half = extent_cm(centre_object.as_actor())
    reach = (max(half[0], half[1]) or 1.0) * float(
        expected.get("distance_factor") or DEFAULT_DISTANCE_FACTOR)
    minimum_subjects = int(expected.get("minimum_subject_count") or 3)
    minimum_coverage = float(expected.get("minimum_angular_coverage")
                             or DEFAULT_MINIMUM_COVERAGE)

    near = [entry for entry in subjects.objects
            if math.dist(center_cm(entry.as_actor())[:2], centre[:2]) <= reach]
    coverage = relations.angular_coverage(
        [entry.as_actor() for entry in near], centre, SECTORS)
    ok = (len(near) >= minimum_subjects
          and coverage["coverage"] >= minimum_coverage)
    return [contracts.decide(
        item, FAMILY, ok,
        {"relation": "around", "minimum_subject_count": minimum_subjects,
         "minimum_angular_coverage": minimum_coverage,
         "maximum_object_relative_distance_cm": rounded(reach, 3),
         "sector_count": SECTORS},
        {"subject_count": len(subjects.objects), "within_distance_count": len(near),
         "occupied_sectors": coverage["occupied_sectors"],
         "angular_coverage": coverage["coverage"]},
        [object_evidence(entry) for entry in near],
        f"{len(near)} object(s) sit within reach covering "
        f"{coverage['coverage']:.2f} of the circle, and the rule needs "
        f"{minimum_subjects} covering {minimum_coverage:.2f}")]


def _checks(context: RuleEvaluationContext, item: Mapping[str, Any]) -> list[Check]:
    subjects, objects = _sides(context, item)
    # Uncategorised Actors drop out of a category selector silently, so a
    # rule can "hold" over the survivors while the objects that would have
    # broken it were never in the population. The count family refuses this
    # evidence; so does this relation family — INCLUDING `around`, whose own
    # branch below used to return before this check ran, so a ring judged
    # over an incompletely selected scene could fail (or hold) on subjects
    # that were never all counted.
    for side, name_ in ((subjects, "subject"), (objects, "object")):
        if not side.complete:
            return [contracts.inconclusive(
                item, FAMILY,
                f"the {name_} selector's evidence is incomplete "
                f"({side.coverage()['unclassified_object_count']} object(s) "
                f"carry no category), so objects that would break this rule "
                f"may never have entered the population",
                side.coverage())]
    expected = item.get("expected") or {}
    minimum_subjects = int(expected.get("minimum_subject_count") or 1)
    if (
        item.get("visual_substitution_fallback") is True
        and len(subjects.objects) < minimum_subjects
    ):
        return [contracts.inconclusive(
            item, FAMILY,
            "trusted structured identity did not cover the full subject "
            "collection; inspect possible visual replacements",
            {"subject_count": len(subjects.objects),
             "minimum_subject_count": minimum_subjects,
             "object_count": len(objects.objects)})]
    if item.get("type") == "around":
        return _around(item, subjects, objects)

    rule = {**(item.get("expected") or {}), "id": contracts.item_id(item, FAMILY)}
    if not rule.get("relation"):
        return [contracts.inconclusive(item, FAMILY, "the rule names no relation")]
    if not subjects.objects or not objects.objects:
        return [contracts.inconclusive(
            item, FAMILY,
            "the rule's selectors did not resolve both sides of the relation",
            {"subject_count": len(subjects.objects),
             "object_count": len(objects.objects)})]
    try:
        result = relations.evaluate(
            rule, [entry.as_actor() for entry in subjects.objects],
            [entry.as_actor() for entry in objects.objects])
    except relations.RelationError as e:
        return [contracts.inconclusive(item, FAMILY, str(e))]
    return [contracts.decide(
        item, FAMILY, result["pass"], result["expected"], result["observed"],
        result["rows"],
        f"the declared {rule['relation']} relation does not hold")]


def evaluate_item(
    context: RuleEvaluationContext, item: Mapping[str, Any]
) -> list[Check]:
    """Evaluate one frozen deterministic rule item."""

    return _checks(context, item)


__all__ = ["DEFAULT_DISTANCE_FACTOR", "DEFAULT_MINIMUM_COVERAGE", "FAMILY",
           "SECTORS", "evaluate_item"]
