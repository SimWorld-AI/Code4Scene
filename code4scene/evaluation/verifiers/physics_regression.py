"""``physics_regression`` — did the edit make the scene physically worse.

Four questions, each asked twice — once of the scene the agent was handed,
once of the scene it left — and answered by the DIFFERENCE: collisions,
clearance violations, ground-contact failures, and Actors out of bounds. Only
what the candidate INTRODUCED counts. Violations that were already there are
reported as unchanged, and ones the edit removed are reported as fixed.

That framing is the whole point, and it is why this is open-ended despite the
component being called `gt_relative_regression`: imported levels arrive with
hundreds of pre-existing interpenetrations, and an absolute collision check
over one of them tells you about the import, not about the agent. The
comparison is against `input_scene` — what the agent was handed, not an answer
key — so it works on every editing task, including the ones that ship no
canonical scene.

The two measurement-backed halves REFUSE rather than assume when either side
is incompletely measured. A regression computed from a candidate that was
measured against an input that was not is a number about the measurement, and
it would read as "no new failures".

Serves component `physics.gt_relative_regression`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .. import assertions
from ..assertions import Case, Check
from ..context import Context
from ..physics_evidence import DEFAULT_PHYSICS_THRESHOLDS, physics_violations
from ..scene_geometry import (TOUCH_TOLERANCE_CM, aabb_overlaps_between,
                              clearance_violations)
from ..scene_semantics import is_non_solid
from ..selection import select_candidate_actors
from ..values import as_text



def _difference(before: Sequence[Mapping[str, Any]],
                after: Sequence[Mapping[str, Any]], key: Any) -> dict[str, Any]:
    """What is new, what is gone, and what was already there."""
    before_keys = {key(item): item for item in before}
    after_keys = {key(item): item for item in after}
    return {
        "introduced": [item for name, item in after_keys.items()
                       if name not in before_keys],
        "fixed": [item for name, item in before_keys.items()
                  if name not in after_keys],
        "unchanged": [item for name, item in after_keys.items()
                      if name in before_keys]}


def _regression(check_id: str, field: str, difference: Mapping[str, Any],
                what: str) -> Check:
    return assertions.check(
        check_id, not difference["introduced"], {field: 0},
        {field: len(difference["introduced"]),
         "fixed": len(difference["fixed"]),
         "unchanged": len(difference["unchanged"])},
        difference["introduced"],
        f"the edit introduced {len(difference['introduced'])} {what}")


def _pair_key(item: Mapping[str, Any]) -> str:
    return "\0".join(sorted([str(item.get("actor1_id") or item.get("actor1")),
                             str(item.get("actor2_id") or item.get("actor2"))]))


def _clearance_key(item: Mapping[str, Any]) -> str:
    return f"{item.get('actor_id') or item.get('actor')}\0{item.get('region')}"


def _actor_key(item: Mapping[str, Any]) -> str:
    return str(item.get("actor_id") or item.get("label"))


def _measured_regression(check_id: str, field: str, what: str,
                         before: Mapping[str, Any], after: Mapping[str, Any],
                         evaluable: str, missing: str,
                         measured_at_all: bool = True) -> Check:
    if not measured_at_all:
        # The in-editor probe emits `out_of_bounds` on every record and leaves
        # it null: it measures Actors, and the world plate is a task-level
        # concept it is never told. So this half is not a hole in the run's
        # evidence, it is a question this build's probe does not answer — and
        # `out_of_bounds` already answers it scene-wide from `measure_v1`.
        return assertions.inapplicable(
            check_id,
            "the in-editor physics probe does not report out-of-bounds; the "
            "world plate is scene-level and `out_of_bounds` measures it",
            {"probe_field": "out_of_bounds", "populated": False})
    if not before[evaluable] or not after[evaluable]:
        return assertions.unevaluated(
            check_id,
            "input and candidate physics measurements are incomplete, and a "
            "regression computed from one measured side is a number about the "
            "measurement",
            {"missing_input_actor_measurements": before[missing],
             "missing_candidate_actor_measurements": after[missing]})
    return _regression(check_id, field,
                       _difference(before[what], after[what], _actor_key),
                       what.replace("_", " "))


def _any_bounds(case: Case) -> bool:
    """Did the probe populate `out_of_bounds` for ANY Actor, on either side."""
    from ..physics_evidence import has_bounds_measurement, measurement_records

    for payload in (case.evidence.measurements, case.evidence.input_measurements):
        if any(has_bounds_measurement(item)
               for item in measurement_records(payload).values()):
            return True
    return False


def _checks(case: Case, assertion: Mapping[str, Any]) -> list[Check]:
    if case.diff is None:
        raise assertions.ScopeError(
            "a regression is measured against the scene the agent was handed, "
            "and the task supplied no input_scene")
    scope = as_text(assertion.get("scope")) or "candidate_all"
    against = as_text(assertion.get("against_scope")) or scope
    # Narrowed by `target_selector` like every other primitive. Scope alone
    # forced the population to be a whole scope, and the measured halves need
    # a MEASUREMENT for every Actor in it — so on a real level the regression
    # could only ever be evaluated by probing all 243 Actors, lights and sky
    # included, which the probe does not measure. The selector is how a case
    # says "these are the Actors this assertion is about", and this verifier
    # was the only one not listening to it.
    after_actors = case.population(assertion, default_scope=scope)
    after_against = case.resolve_scope(against)
    selector = assertion.get("target_selector")
    selector = selector if isinstance(selector, Mapping) else None

    def before_side(name: str) -> list[Mapping[str, Any]]:
        """The same selector, applied to the scene the agent was handed.

        Empty for an addition scope by construction: what was added did not
        exist before, so it can carry no prior violation.
        """
        if name not in ("candidate_all", "source_actors"):
            return []
        actors = case.input_actors
        return select_candidate_actors(actors, selector) if selector else actors

    before_actors = before_side(scope)
    before_against = (case.input_actors
                      if against in ("candidate_all", "source_actors") else [])

    tolerance = assertion.get("touch_tolerance_cm")
    tolerance = TOUCH_TOLERANCE_CM if tolerance is None else float(tolerance)
    # Solidity first, as `solid_penetration` does. Without it a flat region
    # marker — a dining zone, a nav volume, a trigger — is a box the furniture
    # placed INSIDE it necessarily intersects, so a scene that complied with
    # the task was reported as introducing five collisions for complying.
    solid = [actor for actor in after_actors if not is_non_solid(actor)]
    solid_against = [actor for actor in after_against if not is_non_solid(actor)]
    before_solid = [actor for actor in before_actors if not is_non_solid(actor)]
    before_solid_against = [actor for actor in before_against
                            if not is_non_solid(actor)]
    collisions = _difference(
        aabb_overlaps_between(before_solid, before_solid_against, tolerance),
        aabb_overlaps_between(solid, solid_against, tolerance), _pair_key)

    rectangles = assertion.get("protected_rectangles") or []
    routes = assertion.get("routes") or []
    clearance = _difference(
        clearance_violations(before_actors, rectangles, routes),
        clearance_violations(after_actors, rectangles, routes), _clearance_key)

    thresholds = {name: float(assertion[name]) for name in DEFAULT_PHYSICS_THRESHOLDS
                  if isinstance(assertion.get(name), (int, float))
                  and not isinstance(assertion.get(name), bool)}
    before_physics = physics_violations(before_actors,
                                        case.evidence.input_measurements, thresholds)
    after_physics = physics_violations(after_actors,
                                       case.evidence.measurements, thresholds)
    return [
        _regression("physics.regression.collision", "new_collision_pairs",
                    collisions, "AABB collision pair(s)"),
        _regression("physics.regression.clearance", "new_clearance_violations",
                    clearance, "protected-region or route violation(s)"),
        _measured_regression("physics.regression.ground_contact",
                             "new_ground_contact_failures", "ground_failures",
                             before_physics, after_physics,
                             "ground_evaluable", "missing_ground"),
        _measured_regression("physics.regression.out_of_bounds",
                             "new_out_of_bounds_actors", "out_of_bounds",
                             before_physics, after_physics,
                             "bounds_evaluable", "missing_bounds",
                             measured_at_all=_any_bounds(case)),
    ]


def _score_without_collision(checks: Sequence[Check]) -> float:
    """Score calibrated regression channels; keep collision as report-only."""
    decided = [
        item for item in checks
        if item.id != "physics.regression.collision"
        and item.status != "not_applicable"
    ]
    if not decided:
        return 1.0
    return sum(item.status == "pass" for item in decided) / len(decided)


def verify(context: Context) -> dict[str, Any]:
    report = assertions.run(
        "physics_regression",
        context,
        "physics_regression",
        "share of calibrated physical properties the edit did not worsen",
        _checks,
        _score_without_collision,
    )
    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        metrics["score_aggregation"] = (
            "mean_of_available_non_collision_regression_checks"
        )
        metrics["report_only_check_ids"] = ["physics.regression.collision"]
        for check in metrics.get("checks") or []:
            if check.get("id") == "physics.regression.collision":
                check["contributes_to_aggregate"] = False
                check["score_role"] = "report_only"
    return report


__all__ = ["verify"]
