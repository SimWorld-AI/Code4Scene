from __future__ import annotations

import pytest

from code4scene.evaluation.requirement_graph.actor_inventory import (
    ActorBounds,
    ActorInventorySnapshot,
    build_actor_descriptor,
)
from code4scene.evaluation.requirement_graph.contracts import SceneBounds
from code4scene.evaluation.requirement_graph.overview_planning import (
    plan_overview_poses,
)


def test_overviews_frame_dense_content_instead_of_world_origin():
    bounds = SceneBounds(
        (-20_000.0, -20_000.0, -500.0),
        (20_000.0, 20_000.0, 5_000.0),
    )
    inventory = ActorInventorySnapshot(
        (
            build_actor_descriptor(
                live_actor_id="building-a",
                actor_class="StaticMeshActor",
                bounds=ActorBounds((8_000.0, -4_000.0, 400.0), (700.0, 900.0, 400.0)),
                active=True,
                renderable=True,
                in_current_level=True,
            ),
            build_actor_descriptor(
                live_actor_id="building-b",
                actor_class="StaticMeshActor",
                bounds=ActorBounds((8_500.0, -3_500.0, 500.0), (600.0, 800.0, 500.0)),
                active=True,
                renderable=True,
                in_current_level=True,
            ),
            build_actor_descriptor(
                live_actor_id="sky-proxy",
                actor_class="SkySphereActor",
                bounds=ActorBounds((0.0, 0.0, 0.0), (19_000.0, 19_000.0, 19_000.0)),
                active=True,
                renderable=True,
                in_current_level=True,
            ),
        )
    )

    poses = plan_overview_poses(inventory, bounds, count=4)

    assert len(poses) == 4
    # Per-azimuth Actor clearance intentionally permits different ring radii,
    # so the arithmetic mean of the camera positions need not equal the look
    # target. The enclosing ring must still be centered on dense content.
    assert (
        min(value.x for value in poses) + max(value.x for value in poses)
    ) / 2 == pytest.approx(8_250.0)
    assert (
        min(value.y for value in poses) + max(value.y for value in poses)
    ) / 2 == pytest.approx(-3_750.0)
    assert all(value.z > 1_000.0 for value in poses)
    assert all(value.pitch < 0.0 for value in poses)


def test_overview_returns_no_origin_fallback_without_content():
    bounds = SceneBounds((-100.0, -100.0, -100.0), (100.0, 100.0, 100.0))

    assert (
        plan_overview_poses(ActorInventorySnapshot(), bounds, count=4)
        == ()
    )
