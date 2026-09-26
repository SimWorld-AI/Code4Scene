"""Content-normalized overview planning for RequirementGraph evidence.

The shared implementation is the verifier-owned adaptation of an earlier
stand-alone scene-capture script. RequirementGraph uses it only for genuinely global
claims; actor-local claims continue to use their targeted AABB cameras.
"""

from __future__ import annotations

from typing import Any

from .. import scene_graph_capture
from .actor_inventory import ActorDescriptor, ActorInventorySnapshot
from .contracts import CameraPose, SceneBounds


def _scene_graph_actor(actor: ActorDescriptor) -> dict[str, Any]:
    """Project one eligible inventory Actor into the shared planner schema."""

    assert actor.bounds is not None
    return {
        "cls": actor.actor_class,
        "loc": list(actor.bounds.center_cm),
        "extent": list(actor.bounds.extent_cm),
    }


def plan_overview_poses(
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
    *,
    count: int,
) -> tuple[CameraPose, ...]:
    """Plan aerial overview poses around the Candidate's densest content.

    ``scene_graph_capture`` requires at least two views to establish a
    portfolio. A one-view caller still receives exactly one pose, selected
    from a deterministically planned two-view ring. No pose is returned when
    the authoritative inventory contains no frameable scene content; callers
    may then use their explicit legacy fallback rather than silently aiming at
    world origin.
    """

    if not isinstance(inventory, ActorInventorySnapshot):
        raise TypeError("inventory must be an ActorInventorySnapshot")
    if not isinstance(scene_bounds, SceneBounds):
        raise TypeError("scene_bounds must be a SceneBounds")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if count == 0:
        return ()

    actors = tuple(
        _scene_graph_actor(actor)
        for actor in inventory.actors
        if actor.bounds is not None and actor.eligible_for_stage1(scene_bounds)
    )
    if not actors:
        return ()
    plan = scene_graph_capture.plan_candidate_grounded_overview(
        actors,
        count=max(2, count),
    )
    scene_audit = plan.audit.get("scene")
    if not isinstance(scene_audit, dict) or int(
        scene_audit.get("content_count") or 0
    ) < 1:
        return ()
    return tuple(
        CameraPose(
            float(view.location[0]),
            float(view.location[1]),
            float(view.location[2]),
            float(view.rotation[1]),
            float(view.rotation[2]),
        )
        for view in plan.views[:count]
    )


__all__ = [
    "plan_overview_poses",
]
