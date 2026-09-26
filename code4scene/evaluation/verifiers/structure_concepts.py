"""``structure_concepts`` — is each thing the task asked for there, in its count.

One check per declared concept ("exactly one dining table", "exactly four
chairs"), and the score is the share of them satisfied. A partial credit is
the point: a scene with the table and three of four chairs is not the same
failure as an empty room, and a boolean would report them identically.

Compound objects are the same question one level down. When a case declares
them, a logical object must consist of EXACTLY the Actor roles it names —
a table missing a leg is not a table — and when it declares component rules
instead, each component class must appear its declared number of times.

Deliberately silent about additions nobody asked for: that is
`structure_additions`, because "everything asked for is present" and "nothing
else is" fail in different ways and a case author needs to see which.

Serves component `semantic.structured_requirements`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import assertions
from ..assertions import Case, Check
from ..context import Context
from ..scene_diff import actor_summary
from ..structure_rules import classify_additions, slug, structure_spec



def _checks(case: Case, assertion: Mapping[str, Any]) -> list[Check]:
    spec = structure_spec([assertion])
    if spec is None or not (spec.concepts or spec.compound_rules
                            or spec.companion_rules):
        return [assertions.unevaluated(
            "structure.required_concept",
            "the structure assertion declares no concepts, compound objects or "
            "component rules, so it states nothing about what must be present")]
    added = case.resolve_scope("all_additions")
    classified = classify_additions(added, spec)
    checks: list[Check] = []

    for rule in spec.concepts:
        actors = classified.concepts.get(rule.concept) or []
        checks.append(assertions.check(
            f"structure.required_concept.{slug(rule.concept)}",
            len(actors) == rule.count,
            {"concept": rule.concept, "count": rule.count,
             "allowed_asset_paths": list(rule.allowed_asset_paths),
             "allowed_categories": list(rule.allowed_categories),
             "allowed_classes": list(rule.allowed_classes)},
            {"concept": rule.concept, "count": len(actors)},
            [actor_summary(actor) for actor in actors],
            f"expected exactly {rule.count} {rule.concept} logical object(s), "
            f"found {len(actors)}"))

    # Compound rules and component rules are alternatives, not a sequence: a
    # case that names logical objects is describing the same population the
    # component rules would, by a stronger contract.
    if spec.compound_rules:
        for rule in spec.compound_rules:
            actors = [actor for actor in added
                      if actor.get("logical_object_id") == rule.logical_object_id]
            expected_roles = sorted(rule.required_roles)
            observed_roles = sorted(str(actor.get("actor_role")) for actor in actors)
            checks.append(assertions.check(
                f"structure.compound_object.{slug(rule.logical_object_id)}",
                observed_roles == expected_roles,
                {"logical_object_id": rule.logical_object_id,
                 "required_roles": expected_roles},
                {"logical_object_id": rule.logical_object_id,
                 "observed_roles": observed_roles},
                [actor_summary(actor) for actor in actors],
                f"logical object {rule.logical_object_id} does not consist of "
                f"exactly the required Actor roles"))
    else:
        for rule in spec.companion_rules:
            actors = classified.companions.get(rule.id) or []
            checks.append(assertions.check(
                f"structure.required_component.{slug(rule.id)}",
                len(actors) == rule.count,
                {"actor_class": rule.actor_class, "count": rule.count},
                {"actor_class": rule.actor_class, "count": len(actors)},
                [actor_summary(actor) for actor in actors],
                f"expected exactly {rule.count} {rule.id} component Actor(s), "
                f"found {len(actors)}"))
    return checks


def verify(context: Context) -> dict[str, Any]:
    return assertions.run("structure_concepts", context, "structure",
                          "share of declared concepts present in their declared "
                          "count", _checks)


__all__ = ["verify"]
