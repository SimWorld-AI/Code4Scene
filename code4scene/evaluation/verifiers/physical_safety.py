"""Two-branch physical-safety policy.

The public ``physical_safety`` score intentionally contains only floating and
solid collision penetration. Text-to-scene policies retain scene-wide Physics
V3. Image-to-scene repair policies can freeze an ``edited_actors`` population;
global preservation then belongs to GT-repair rather than being re-measured
actor by actor here. Other registered physics atoms are standalone diagnostics
and must not be invoked or averaged by this parent.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .. import atomic_registry
from ..composite import composite_report
from ..context import Context, error
from ..evaluation_policy import load_evaluation_policy, task_mode


CLASS = "open_ended"

_LEAVES = (
    "floating",
    "solid_penetration",
)


def verify(context: Context) -> dict[str, Any]:
    try:
        policy = load_evaluation_policy(context.task)
    except Exception as exc:  # noqa: BLE001 - frozen policy refusal
        return error("physical_safety", context, f"{type(exc).__name__}: {exc}")
    mode = task_mode(context.task.kind)
    population_scope = policy.physics_profile.get(
        "selector", {}
    ).get("scope", "candidate_all")
    local_applicability_aggregation = population_scope == "edited_actors"
    leaves = []
    for leaf_id in _LEAVES:
        spec = policy.leaf_spec(context.task, leaf_id, context.spec)
        child = replace(context, spec=spec)
        try:
            report = atomic_registry.get(leaf_id).evaluate_report(
                child, policy.physics_profile
            )
        except Exception as exc:  # noqa: BLE001 - leaf remains visible
            report = error(
                leaf_id, context, f"{type(exc).__name__}: {exc}"
            )
        report["leaf_id"] = leaf_id
        leaves.append(report)
    return composite_report(
        "physical_safety",
        context,
        leaves,
        require_all_scored=True,
        exclude_not_applicable_from_required_scores=(
            local_applicability_aggregation
        ),
        evidence={
            **policy.evidence(),
            "task_mode": mode,
            "physics_profile_id": policy.physics_profile.get("profile_id"),
            "population_scope": population_scope,
            "score_requirement_policy": (
                "all_applicable_physics_leaves"
                if local_applicability_aggregation
                else "all_declared_physics_leaves"
            ),
            "global_coverage_owner": (
                "gt_repair.scene_diff"
                if population_scope == "edited_actors"
                else "physical_safety"
            ),
            "atomic_registry": list(atomic_registry.kinds("physics")),
            "active_physics_leaves": list(_LEAVES),
        },
    )


__all__ = ["verify"]
