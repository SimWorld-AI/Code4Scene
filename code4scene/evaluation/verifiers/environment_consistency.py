"""``environment_consistency`` — is anything standing at the bottom of a lake.

A scene can be perfectly grounded, perfectly non-intersecting, and still be
wrong in a way no other physics check sees: the bench is under the river, the
car is on the seabed. `floating` says a submerged Actor is fine — it IS resting
on something — and a screenshot of a water surface hides it completely.

Water is found by name and then by shape: an Actor is a water surface when it
is CALLED water AND spans at least two metres in both horizontal directions.
The second condition is what keeps a puddle decal and a prop called
`SM_WaterBottle` from flooding the level.

An Actor with no usable bounds WITHHOLDS the whole check rather than being
skipped. Skipping is how a scene that could not be measured comes back clean.

Serves component `physics.absolute`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import assertions
from ..assertions import Case, Check
from ..context import Context
from ..scene_semantics import (DEFAULT_WATERLINE_TOLERANCE_CM, actor_name,
                               bounding_box, names, overlaps_xy, water_evidence,
                               water_surface)

def _submersion(actors, scene_actors, options) -> dict[str, Any]:
    tolerance = float(options.get("waterline_tolerance_cm")
                      or DEFAULT_WATERLINE_TOLERANCE_CM)
    # An Actor NAMED like water whose bounds are unusable is dropped by
    # `water_surface`, which reads as "there is no water here" and clears
    # every submerged Actor. Counted instead, and the caller withholds.
    unusable = [actor_name(actor) for actor in scene_actors
                if water_evidence(actor) and bounding_box(actor) is None]
    surfaces = [surface for surface in
                (water_surface(actor, options) for actor in scene_actors)
                if surface is not None]
    fully: list[dict[str, Any]] = []
    partially: list[dict[str, Any]] = []
    missing: list[str] = []
    for actor in actors:
        bounds = bounding_box(actor)
        if bounds is None:
            missing.append(actor_name(actor))
            continue
        for surface in surfaces:
            if surface["actor"] is actor or not overlaps_xy(
                    bounds, surface["bounds"], tolerance):
                continue
            waterline = surface["surface_z_cm"] - tolerance
            under = bounds["max"][2] < waterline
            crossing = not under and bounds["min"][2] < waterline
            if not under and not crossing:
                continue
            (fully if under else partially).append({
                "actor": actor_name(actor), "actor_label": actor.get("label"),
                "surface_actor": actor_name(surface["actor"]),
                "surface_label": surface["actor"].get("label"),
                "actor_min_z_cm": bounds["min"][2],
                "actor_max_z_cm": bounds["max"][2],
                "water_surface_z_cm": surface["surface_z_cm"],
                "fully_submerged_depth_cm": (
                    surface["surface_z_cm"] - bounds["max"][2]) if under else 0,
                "waterline_height_within_actor_cm": (
                    surface["surface_z_cm"] - bounds["min"][2]) if crossing else None,
                "surface_semantic_evidence": surface["semantic_evidence"]})
    return {"surfaces": surfaces, "fully": fully, "partially": partially,
            "missing_bounds": missing, "unmeasurable_water": unusable}


def _checks(case: Case, assertion: Mapping[str, Any]) -> list[Check]:
    actors = assertions.actors_or_unresolved(case, assertion)
    observation = _submersion(actors, case.candidate, assertion)
    maximum_full = int(assertion.get("maximum_fully_submerged_actor_count") or 0)
    maximum_partial = int(
        assertion.get("maximum_partially_submerged_actor_count") or 0)
    expected = {"maximum_fully_submerged_actor_count": maximum_full,
                "maximum_partially_submerged_actor_count": maximum_partial}
    if observation["unmeasurable_water"]:
        return [assertions.unevaluated(
            "physics.environment_consistency",
            f"{len(observation['unmeasurable_water'])} Actor(s) named like "
            f"water carry no usable bounds "
            f"({', '.join(observation['unmeasurable_water'][:4])}), so the "
            f"waterline they define is unknown and everything under it would "
            f"be reported dry", {**expected})]
    if observation["missing_bounds"]:
        return [assertions.unevaluated(
            "physics.environment_consistency",
            f"{len(observation['missing_bounds'])} evaluated Actor(s) carry no "
            f"bounds, and an Actor whose size is unknown cannot be shown to be "
            f"out of the water",
            {**expected,
             "missing_bounds_actor_count": len(observation["missing_bounds"]),
             "detected_water_surface_count": len(observation["surfaces"])})]
    full_count = len(names(observation["fully"]))
    partial_count = len(names(observation["partially"]))
    return [assertions.check(
        "physics.environment_consistency",
        full_count <= maximum_full and partial_count <= maximum_partial,
        expected,
        {"evaluated_actor_count": len(actors),
         "detected_water_surface_count": len(observation["surfaces"]),
         "fully_submerged_actor_count": full_count,
         "partially_submerged_actor_count": partial_count},
        [*observation["fully"], *observation["partially"]],
        f"{full_count} Actor(s) are under a detected water surface and "
        f"{partial_count} cross one")]


def _score(checks) -> float:
    observed = checks[0].observed
    total = observed.get("evaluated_actor_count") or 0
    if not total:
        return 1.0
    submerged = (observed.get("fully_submerged_actor_count", 0)
                 + observed.get("partially_submerged_actor_count", 0))
    return 1.0 - submerged / total


def verify(context: Context) -> dict[str, Any]:
    return assertions.run("environment_consistency", context,
                          "environment_consistency",
                          "share of the declared Actors out of the water",
                          _checks, _score)


__all__ = ["verify"]
