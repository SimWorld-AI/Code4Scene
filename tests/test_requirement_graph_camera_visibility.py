from __future__ import annotations

import numpy as np

from code4scene.evaluation.requirement_graph.actor_inventory import (
    ActorBounds,
    ActorInventorySnapshot,
    build_actor_descriptor,
)
from code4scene.evaluation.requirement_graph.camera_visibility import (
    camera_pose_key,
    resolve_visibility_camera_poses,
    visibility_pose_candidates,
)
from code4scene.evaluation.requirement_graph.contracts import CameraPose, SceneBounds
from code4scene.evaluation.requirement_graph.runtime import CapturedFrame
from code4scene.evaluation.requirement_graph.stage2_capture import (
    CaptureAttemptStatus,
    execute_stage2_capture_plan,
)
from code4scene.evaluation.requirement_graph.stage2_contracts import (
    CaptureProgram,
    CaptureRequest,
    CaptureShotRole,
    Stage2CapturePlan,
)
from code4scene.evaluation.requirement_graph.stage2_frames import FrameStore


SCENE_BOUNDS = SceneBounds(
    (-2000.0, -2000.0, -500.0),
    (2000.0, 2000.0, 2000.0),
)
TARGET_BOUNDS = ActorBounds((0.0, 0.0, 100.0), (50.0, 40.0, 60.0))


def _inventory() -> ActorInventorySnapshot:
    return ActorInventorySnapshot(
        (
            build_actor_descriptor(
                live_actor_id="actor-target",
                asset_path="/Game/Test/SM_Target.SM_Target",
                bounds=TARGET_BOUNDS,
                active=True,
                renderable=True,
                in_current_level=True,
            ),
        )
    )


def test_visibility_candidates_keep_requested_pose_and_cover_multiple_azimuths():
    preferred = CameraPose(500.0, 0.0, 180.0, -9.0, 180.0)

    values = visibility_pose_candidates(
        preferred,
        (TARGET_BOUNDS,),
        SCENE_BOUNDS,
    )

    assert values[0] == preferred
    assert len(values) >= 12
    azimuths = {
        round(
            np.degrees(
                np.arctan2(value.y - TARGET_BOUNDS.center_cm[1], value.x)
            )
            % 360.0
        )
        for value in values
    }
    assert len(azimuths) >= 6


def test_visibility_candidates_reject_camera_positions_too_close_to_small_actor():
    tiny = ActorBounds((0.0, 0.0, 100.0), (10.0, 8.0, 4.0))
    preferred = CameraPose(10.0, 0.0, 105.0, -9.0, 180.0)

    values = visibility_pose_candidates(preferred, (tiny,), SCENE_BOUNDS)

    assert values
    assert preferred not in values
    assert all(
        np.linalg.norm(np.asarray(value.location_cm) - np.asarray(tiny.center_cm))
        >= 49.999
        for value in values
    )


def test_camera_pose_key_canonicalizes_yaw_wraparound():
    negative_epsilon = CameraPose(1.0, 2.0, 3.0, 4.0, -1e-9)
    positive_epsilon = CameraPose(1.0, 2.0, 3.0, 4.0, 1e-9)

    assert camera_pose_key(negative_epsilon) == camera_pose_key(positive_epsilon)
    assert camera_pose_key(negative_epsilon)[-1] == 0.0


class _VisibilityBridge:
    def __init__(self, result):
        self.result = result
        self.scripts = []

    def exec_python_result(self, script, key, timeout=None):
        self.scripts.append((script, key, timeout))
        return self.result


def test_visibility_resolver_uses_ue_selected_pose_without_rendering():
    requested = CameraPose(500.0, 0.0, 180.0, -9.0, 180.0)
    candidates = visibility_pose_candidates(
        requested,
        (TARGET_BOUNDS,),
        SCENE_BOUNDS,
    )
    bridge = _VisibilityBridge(
        {
            "results": [
                {
                    "accepted": True,
                    "selected_index": 3,
                    "visible_actor_count": 1,
                    "target_actor_count": 1,
                    "clear_ray_fraction": 0.75,
                }
            ]
        }
    )

    result = resolve_visibility_camera_poses(
        bridge,
        _inventory(),
        SCENE_BOUNDS,
        (requested,),
        (("actor-target",),),
    )[0]

    assert result.pose == candidates[3]
    assert result.requested_pose == requested
    assert result.reason == "line_of_sight_resolved"
    assert result.clear_ray_fraction == 0.75
    assert result.required_actor_clear_fraction == 0.4
    assert len(bridge.scripts) == 1
    assert "sphere_trace_single" in bridge.scripts[0][0]
    assert "0.15 * min(_extent)" in bridge.scripts[0][0]
    assert "simcodearena.stable_actor_id=" in bridge.scripts[0][0]
    assert "'actor_id': 'actor-target'" in bridge.scripts[0][0]
    assert '"target_distance_cm"' in bridge.scripts[0][0]
    assert (
        bridge.scripts[0][0].index('-_value["target_distance_cm"]')
        < bridge.scripts[0][0].index('-_value["scene_center_distance_cm"]')
    )
    compile(bridge.scripts[0][0], "<camera-visibility-probe>", "exec")


def test_visibility_resolver_records_explicit_sparse_target_threshold():
    requested = CameraPose(500.0, 0.0, 180.0, -9.0, 180.0)
    bridge = _VisibilityBridge(
        {
            "results": [
                {
                    "accepted": True,
                    "selected_index": 0,
                    "visible_actor_count": 1,
                    "target_actor_count": 1,
                    "clear_ray_fraction": 0.2,
                    "minimum_actor_clear_fraction": 0.2,
                    "required_actor_clear_fraction": 0.2,
                }
            ]
        }
    )

    result = resolve_visibility_camera_poses(
        bridge,
        _inventory(),
        SCENE_BOUNDS,
        (requested,),
        (("actor-target",),),
        required_actor_clear_fraction=0.2,
    )[0]

    assert result.pose is not None
    assert result.required_actor_clear_fraction == 0.2
    assert "'required_actor_clear_fraction': 0.2" in bridge.scripts[0][0]


def test_visibility_resolver_records_opt_in_thin_surface_proxy_evidence():
    requested = CameraPose(500.0, 0.0, 180.0, -9.0, 180.0)
    bridge = _VisibilityBridge(
        {
            "results": [
                {
                    "accepted": True,
                    "selected_index": 0,
                    "visible_actor_count": 1,
                    "target_actor_count": 1,
                    "clear_ray_fraction": 0.2,
                    "minimum_actor_clear_fraction": 0.2,
                    "required_actor_clear_fraction": 0.2,
                    "direct_target_ray_count": 0,
                    "thin_surface_proxy_ray_count": 3,
                    "thin_surface_open_ray_count": 3,
                    "thin_surface_support_ray_count": 0,
                    "thin_surface_support_actors": [],
                    "nearest_blocker_distance_to_sample_cm": 1.25,
                    "nearest_blocker_actor": "Wall_7",
                    "thin_surface_proxy_enabled": True,
                }
            ]
        }
    )

    result = resolve_visibility_camera_poses(
        bridge,
        _inventory(),
        SCENE_BOUNDS,
        (requested,),
        (("actor-target",),),
        required_actor_clear_fraction=0.2,
        allow_thin_target_surface_proxy=True,
    )[0]

    assert result.pose is not None
    assert result.direct_target_ray_count == 0
    assert result.thin_surface_proxy_ray_count == 3
    assert result.thin_surface_open_ray_count == 3
    assert result.thin_surface_support_ray_count == 0
    assert result.thin_surface_support_actors == ()
    assert result.nearest_blocker_distance_to_sample_cm == 1.25
    assert result.nearest_blocker_actor == "Wall_7"
    assert result.thin_surface_proxy_enabled is True
    audit = result.to_dict()
    assert audit["thin_surface_proxy_ray_count"] == 3
    assert audit["thin_surface_open_ray_count"] == 3
    script = bridge.scripts[0][0]
    assert "'allow_thin_target_surface_proxy': True" in script
    assert "def _thin_surface_proxy_eligible" in script
    assert "def _hit_reaches_thin_target_surface" in script
    assert "def _blocking_actor_is_thin_target_support" in script
    assert "_camera_side * _support_side < 0.0" in script
    assert "_distance > _tolerance" in script
    assert "if _thin_surface_proxy:" in script
    assert "_thin_surface_open_rays += 1" in script
    compile(script, "<camera-visibility-probe>", "exec")


def test_visibility_resolver_rejects_invalid_explicit_target_threshold():
    requested = CameraPose(500.0, 0.0, 180.0, -9.0, 180.0)

    with np.testing.assert_raises_regex(
        ValueError,
        "required_actor_clear_fraction",
    ):
        resolve_visibility_camera_poses(
            _VisibilityBridge({"results": []}),
            _inventory(),
            SCENE_BOUNDS,
            (requested,),
            (("actor-target",),),
            required_actor_clear_fraction=0.0,
        )


def test_visibility_resolver_rejects_non_boolean_thin_surface_proxy():
    requested = CameraPose(500.0, 0.0, 180.0, -9.0, 180.0)

    with np.testing.assert_raises_regex(
        TypeError,
        "allow_thin_target_surface_proxy",
    ):
        resolve_visibility_camera_poses(
            _VisibilityBridge({"results": []}),
            _inventory(),
            SCENE_BOUNDS,
            (requested,),
            (("actor-target",),),
            allow_thin_target_surface_proxy=1,
        )


def test_visibility_resolver_uses_exact_shared_room_corner_candidates():
    requested = CameraPose(-300.0, -250.0, 150.0, -3.0, 45.0)
    inward_fallback = CameraPose(-250.0, -200.0, 150.0, -4.0, 45.0)
    bridge = _VisibilityBridge(
        {
            "results": [
                {
                    "accepted": True,
                    "selected_index": 1,
                    "visible_actor_count": 1,
                    "target_actor_count": 1,
                    "clear_ray_fraction": 0.6,
                    "camera_initial_overlap": False,
                    "camera_enclosure_ok": True,
                    "enclosure_up_hit": True,
                    "enclosure_up_trace_hit": False,
                    "enclosure_up_room_ceiling_fallback": True,
                    "enclosure_down_hit": True,
                    "enclosure_down_trace_hit": False,
                    "enclosure_down_room_floor_fallback": True,
                    "horizontal_enclosure_hits": 6,
                    "required_horizontal_enclosure_hits": 4,
                }
            ]
        }
    )

    result = resolve_visibility_camera_poses(
        bridge,
        _inventory(),
        SCENE_BOUNDS,
        (requested,),
        (("actor-target",),),
        candidate_poses_by_pose=((requested, inward_fallback),),
        overview_mode=True,
        require_enclosure=True,
        require_unique_camera_poses=True,
        enclosure_room_floor_z_cm=0.0,
        enclosure_room_ceiling_z_cm=300.0,
    )[0]

    assert result.pose == inward_fallback
    assert result.candidate_count == 2
    assert result.reason == "overview_visibility_resolved"
    assert result.camera_initial_overlap is False
    assert result.camera_enclosure_ok is True
    assert result.enclosure_up_hit is True
    assert result.enclosure_up_trace_hit is False
    assert result.enclosure_up_room_ceiling_fallback is True
    assert result.enclosure_down_hit is True
    assert result.enclosure_down_trace_hit is False
    assert result.enclosure_down_room_floor_fallback is True
    assert result.horizontal_enclosure_hits == 6
    assert result.required_horizontal_enclosure_hits == 4
    assert "'mode': 'overview'" in bridge.scripts[0][0]
    assert "'require_enclosure': True" in bridge.scripts[0][0]
    assert "'require_unique_camera_pose': True" in bridge.scripts[0][0]
    assert "'enclosure_room_floor_z_cm': 0.0" in bridge.scripts[0][0]
    assert "'enclosure_room_ceiling_z_cm': 300.0" in bridge.scripts[0][0]
    assert "def _camera_enclosure" in bridge.scripts[0][0]


def test_visibility_resolver_fails_closed_when_no_actor_pose_is_visible():
    requested = CameraPose(500.0, 0.0, 180.0, -9.0, 180.0)
    bridge = _VisibilityBridge(
        {
            "results": [
                {
                    "accepted": False,
                    "selected_index": 0,
                    "visible_actor_count": 0,
                    "target_actor_count": 1,
                    "clear_ray_fraction": 0.0,
                }
            ]
        }
    )

    result = resolve_visibility_camera_poses(
        bridge,
        _inventory(),
        SCENE_BOUNDS,
        (requested,),
        (("actor-target",),),
    )[0]

    assert result.pose is None
    assert result.reason == "no_unoccluded_camera_pose"


def test_visibility_resolver_grounds_actorless_overview_before_rgb_capture():
    requested = CameraPose(4500.0, 4500.0, 3500.0, -30.0, 225.0)
    bridge = _VisibilityBridge(
        {
            "results": [
                {
                    "accepted": True,
                    "selected_index": 0,
                    "visible_actor_count": 1,
                    "target_actor_count": 1,
                    "clear_ray_fraction": 0.5,
                }
            ]
        }
    )

    result = resolve_visibility_camera_poses(
        bridge,
        _inventory(),
        SCENE_BOUNDS,
        (requested,),
        ((),),
    )[0]

    assert result.pose is not None
    assert result.selected_index == 0
    assert result.reason == "overview_visibility_resolved"
    assert result.actor_ids == ("actor-target",)
    assert result.visible_actor_count == 1
    assert result.target_actor_count == 1
    assert len(bridge.scripts) == 1
    assert "'mode': 'overview'" in bridge.scripts[0][0]
    assert "_same_actor(_blocking_actor, _target_actor)" in bridge.scripts[0][0]


def test_stage2_actorless_overview_still_rejects_invalid_rgb():
    requested = CameraPose(4500.0, 4500.0, 3500.0, -30.0, 225.0)
    request = CaptureRequest(
        "capture-1",
        ("task-1",),
        CaptureShotRole.OVERVIEW,
        requested,
    )
    plan = Stage2CapturePlan(
        (CaptureProgram("program-1", ("task-1",), (request,)),)
    )

    class Provider:
        def resolve_camera_poses(self, poses, *, actor_ids_by_pose):
            resolutions = resolve_visibility_camera_poses(
                _VisibilityBridge(
                    {
                        "results": [
                            {
                                "accepted": True,
                                "selected_index": 0,
                                "visible_actor_count": 1,
                                "target_actor_count": 1,
                            }
                        ]
                    }
                ),
                _inventory(),
                SCENE_BOUNDS,
                poses,
                actor_ids_by_pose,
            )
            return tuple(value.pose for value in resolutions)

        def capture(self, poses, **kwargs):
            del kwargs
            return tuple(
                CapturedFrame(
                    f"provider-{index}",
                    pose,
                    np.zeros((24, 32, 3), dtype=np.uint8),
                )
                for index, pose in enumerate(poses)
            )

    result = execute_stage2_capture_plan(plan, Provider(), FrameStore())

    assert result.attempts[0].status is CaptureAttemptStatus.REJECTED
    assert result.attempts[0].reason == "near_constant"
    assert result.runtime_error is None


def test_stage2_renders_only_the_geometry_resolved_pose():
    requested = CameraPose(500.0, 0.0, 180.0, -9.0, 180.0)
    resolved = CameraPose(0.0, 500.0, 220.0, -13.0, 270.0)
    request = CaptureRequest(
        "capture-1",
        ("task-1",),
        CaptureShotRole.CLOSE,
        requested,
        actor_ids=("actor-target",),
    )
    plan = Stage2CapturePlan(
        (CaptureProgram("program-1", ("task-1",), (request,)),)
    )

    class Provider:
        def __init__(self):
            self.captured = []

        def resolve_camera_poses(self, poses, *, actor_ids_by_pose):
            assert tuple(poses) == (requested,)
            assert tuple(actor_ids_by_pose) == (("actor-target",),)
            return (resolved,)

        def capture(self, poses, **kwargs):
            self.captured.append((tuple(poses), kwargs))
            return (
                CapturedFrame(
                    "provider-frame",
                    resolved,
                    np.random.default_rng(1).integers(
                        0, 256, size=(24, 32, 3), dtype=np.uint8
                    ),
                ),
            )

    provider = Provider()
    result = execute_stage2_capture_plan(plan, provider, FrameStore())

    assert provider.captured[0][0] == (resolved,)
    assert result.attempts[0].status is CaptureAttemptStatus.ACCEPTED
    assert result.runtime_error is None


def test_stage2_does_not_render_an_unresolved_actor_pose():
    requested = CameraPose(500.0, 0.0, 180.0, -9.0, 180.0)
    request = CaptureRequest(
        "capture-1",
        ("task-1",),
        CaptureShotRole.CLOSE,
        requested,
        actor_ids=("actor-target",),
    )
    plan = Stage2CapturePlan(
        (CaptureProgram("program-1", ("task-1",), (request,)),)
    )

    class Provider:
        def resolve_camera_poses(self, poses, *, actor_ids_by_pose):
            return (None,)

        def capture(self, poses, **kwargs):
            raise AssertionError("an unresolved camera must not render RGB")

    result = execute_stage2_capture_plan(plan, Provider(), FrameStore())

    assert result.attempts[0].status is CaptureAttemptStatus.REJECTED
    assert result.attempts[0].reason == "camera_pose_unresolved"
    assert result.runtime_error is None
