"""``structure_additions`` — did the edit add anything nobody asked for.

The closed-world half of a structured case. `structure_concepts` asks whether
what was requested is present; this asks whether anything ELSE is, and the two
are separate because an agent that cannot build the thing tends to build many
things instead — a failure that looks like effort and scores like success
under an open-world contract.

An addition is unexpected when it matches no concept rule and no declared
component class. Matching TWO concept rules also lands here: an Actor that
could be either of two requested things has not established that it is one of
them, and the ambiguity belongs in the report rather than in whichever rule
was declared first.

Serves component `semantic.structured_requirements`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import assertions
from ..assertions import Case, Check
from ..context import Context
from ..scene_diff import actor_summary
from ..structure_rules import classify_additions, structure_spec



def _checks(case: Case, assertion: Mapping[str, Any]) -> list[Check]:
    spec = structure_spec([assertion])
    if spec is None:
        return [assertions.unevaluated(
            "structure.allowed_additions",
            "the structure assertion could not be read, so what counts as an "
            "allowed addition is undefined")]
    added = case.resolve_scope("all_additions")
    classified = classify_additions(added, spec)
    checks = [assertions.check(
        "structure.allowed_additions", not classified.unexpected,
        {"unexpected_added_actors": 0},
        {"unexpected_added_actors": len(classified.unexpected),
         "added_actors": len(added)},
        [actor_summary(actor) for actor in classified.unexpected],
        f"{len(classified.unexpected)} added Actor(s) match nothing the case "
        f"contract allows")]

    if spec.compound_rules:
        allowed_ids = {rule.logical_object_id for rule in spec.compound_rules}
        strays = [actor for actor in added
                  if actor.get("logical_object_id")
                  and actor.get("logical_object_id") not in allowed_ids]
        checks.append(assertions.check(
            "structure.allowed_compound_objects", not strays,
            {"unexpected_logical_objects": 0},
            {"unexpected_logical_objects": len(
                {str(actor.get("logical_object_id")) for actor in strays})},
            [actor_summary(actor) for actor in strays],
            "the candidate carries logical object ids the case contract does "
            "not allow"))
    return checks


def verify(context: Context) -> dict[str, Any]:
    return assertions.run("structure_additions", context, "structure",
                          "additions the case contract does not allow", _checks)


__all__ = ["verify"]
