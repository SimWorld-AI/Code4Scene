"""Object-to-architecture relation rules for RequirementGraph Stage 1.

The OAR family shares the object-relation vocabulary,
restricted to the three claims a prompt makes about objects and the structure
around them: something is AGAINST a wall, something OCCUPIES a region, and
something BLOCKS one. They are separated from the object-to-object family
because they fail for a different reason — a scene where the furniture relates
correctly to itself but floats in the middle of the room is a specific,
common, and separately interesting failure.

**The architecture resolver is not ported.** The vendored evaluator inferred
which Actors were walls, floors and functional regions from an ontology and a
region resolver; here a rule NAMES its architecture with the same selector
shape everything else uses. That is a real reduction in what an author
can leave implicit, and it is deliberate: an inferred wall that is actually a
bookshelf turns a passing scene into a failing one with no way for a reader to
see why. A rule that needs inference is `not_evaluated`, and says so.

This module is internal to ``semantic_requirements``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from code4scene.evaluation import spatial_relations as relations
from code4scene.evaluation.assertions import Check

from . import contracts
from .contracts import RuleEvaluationContext

FAMILY = "OAR"

#: The relations this family admits. Anything else belongs to OOR.
ARCHITECTURE_RELATIONS = ("against", "occupies", "blocks")


def _checks(context: RuleEvaluationContext, item: Mapping[str, Any]) -> list[Check]:
    expected = item.get("expected") or {}
    relation = str(expected.get("relation") or "")
    if relation not in ARCHITECTURE_RELATIONS:
        return [contracts.inconclusive(
            item, FAMILY,
            f"object-to-architecture rules decide {', '.join(ARCHITECTURE_RELATIONS)};"
            f" {relation!r} is an object-to-object relation")]
    if not item.get("object"):
        return [contracts.inconclusive(
            item, FAMILY,
            "the rule does not name the architecture it is about. This build "
            "does not infer which Actors are walls or regions — an inferred "
            "wall that is really a bookshelf fails a correct scene invisibly — "
            "so the rule must select its architecture explicitly")]

    subjects = context.graph.select(item.get("subject"))
    architecture = context.graph.select(item.get("object"))
    # Uncategorised Actors drop out of a category selector silently, so a
    # rule can "hold" over the survivors while the objects that would have
    # broken it were never in the population. The count family refuses this
    # evidence; so does this relation family.
    for side, name_ in ((subjects, "subject"), (architecture, "architecture")):
        if not side.complete:
            return [contracts.inconclusive(
                item, FAMILY,
                f"the {name_} selector's evidence is incomplete "
                f"({side.coverage()['unclassified_object_count']} object(s) "
                f"carry no category), so objects that would break this rule "
                f"may never have entered the population",
                side.coverage())]
    if not subjects.objects or not architecture.objects:
        return [contracts.inconclusive(
            item, FAMILY,
            "the rule's selectors did not resolve both the objects and the "
            "architecture they are about",
            {"subject_count": len(subjects.objects),
             "architecture_count": len(architecture.objects)})]
    rule = {**expected, "id": contracts.item_id(item, FAMILY)}
    try:
        result = relations.evaluate(
            rule, [entry.as_actor() for entry in subjects.objects],
            [entry.as_actor() for entry in architecture.objects])
    except relations.RelationError as e:
        return [contracts.inconclusive(item, FAMILY, str(e))]
    return [contracts.decide(
        item, FAMILY, result["pass"], result["expected"], result["observed"],
        result["rows"],
        f"the declared {relation} relation with the architecture does not hold")]


def evaluate_item(
    context: RuleEvaluationContext, item: Mapping[str, Any]
) -> list[Check]:
    """Evaluate one frozen deterministic rule item."""

    return _checks(context, item)


__all__ = ["ARCHITECTURE_RELATIONS", "FAMILY", "evaluate_item"]
