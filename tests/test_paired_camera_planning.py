from __future__ import annotations

from types import SimpleNamespace

import pytest

from code4scene.evaluation import (
    paired_camera_planning,
    render,
    scene_graph_capture,
)
from code4scene.evaluation.render_evidence import (
    CAPTION_ENVIRONMENT_RENDER_PROTOCOL,
    GT_REPAIR_TARGET_VISUAL_ENVIRONMENT_RENDER_PROTOCOL,
    GT_REPAIR_TARGET_AUTHORED_VISUAL_PROTOCOL,
    GT_REPAIR_TARGET_VISUAL_PROTOCOL,
    GT_REPAIR_TARGET_VISUAL_QUALITY_GATE,
    GT_VISUAL_ENVIRONMENT_RENDER_PROTOCOL,
)
from code4scene.evaluation.verifiers import gt_repair
from code4scene.tasks import task as task_mod
from code4scene.evaluation.requirement_graph.contracts import CameraPose
from code4scene.evaluation.requirement_graph.evidence_adapter import (
    adapt_scene_snapshot,
)


def _actor(
    stable_id: str,
    *,
    center=(0.0, 0.0, 100.0),
    extent=(40.0, 40.0, 100.0),
    label="Chair",
    asset="/Game/Props/SM_Chair.SM_Chair",
):
    return {
        "stable_actor_id": stable_id,
        "name": f"SM_{stable_id}",
        "label": label,
        "class": "/Script/Engine.StaticMeshActor",
        "asset_path": asset,
        "component_asset_paths": [asset],
        "bounds": {"origin_cm": list(center), "extent_cm": list(extent)},
        "transform": {
            "location_cm": list(center),
            "rotation_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "export_diagnostics": {"hidden_in_editor": False},
    }


def _scene(*actors):
    return {
        "actors": list(actors),
        "export_metadata": {"status": "success"},
    }


def _runtime_observability(**accepted_by_view):
    return {
        name: {
            "policy": "current-task-input-gt-robust-rgb8-observability",
            "accepted": accepted,
            "candidate_used_for_acceptance": False,
            "gt_repeat_used_for_acceptance": False,
            "gates": {
                "weak_visibility": accepted,
                "strong_visibility": accepted,
                "locality": True,
                "camera_similarity": True,
            },
        }
        for name, accepted in accepted_by_view.items()
    }


def test_indoor_anchors_are_shared_unmodified_props_not_structure():
    chair = _actor("chair", center=(0.0, 0.0, 100.0))
    table = _actor(
        "table",
        center=(600.0, 0.0, 80.0),
        extent=(100.0, 60.0, 80.0),
        label="Table",
        asset="/Game/Props/SM_Table.SM_Table",
    )
    moved_gt = _actor("moved", center=(200.0, 400.0, 100.0))
    moved_input = _actor("moved", center=(300.0, 400.0, 100.0))
    floor = _actor(
        "floor",
        extent=(900.0, 900.0, 5.0),
        label="Floor",
        asset="/Game/Architecture/SM_Floor.SM_Floor",
    )
    thin_panel = _actor(
        "structural-panel",
        extent=(7.0, 351.0, 190.0),
        label="Glowing Box",
        asset="/Game/Architecture/SM_Glowing_Box.SM_Glowing_Box",
    )
    door = _actor(
        "door",
        extent=(42.0, 6.5, 111.0),
        label="Door",
        asset="/Game/Architecture/SM_Door.SM_Door",
    )
    input_scene = _scene(chair, table, moved_input, floor, thin_panel, door)
    gt_scene = _scene(chair, table, moved_gt, floor, thin_panel, door)
    inventory, _bounds = adapt_scene_snapshot(gt_scene)

    selected = paired_camera_planning.shared_indoor_anchor_ids(
        input_scene,
        gt_scene,
        inventory,
        count=4,
    )

    assert set(selected) == {"stable:chair", "stable:table"}


def test_indoor_overview_uses_four_shared_room_corner_portfolios(monkeypatch):
    actors = tuple(
        _actor(
            f"prop-{index}",
            center=(float(index * 300), float((index % 2) * 350), 100.0),
        )
        for index in range(5)
    )
    floor = _actor(
        "floor",
        center=(0.0, 0.0, 0.0),
        extent=(400.0, 350.0, 2.0),
        label="Floor",
        asset="/Game/Architecture/SM_Floor.SM_Floor",
    )
    ceiling = _actor(
        "ceiling",
        center=(0.0, 0.0, 300.0),
        extent=(400.0, 350.0, 5.0),
        label="Ceiling",
        asset="/Game/Architecture/SM_Ceiling.SM_Ceiling",
    )
    inserted = _actor(
        "inserted",
        center=(125.0, -75.0, 120.0),
        label="Inserted Lamp",
    )

    def resolve(_bridge, _inventory, _bounds, poses, actor_ids, **kwargs):
        groups = kwargs["candidate_poses_by_pose"]
        assert kwargs["overview_mode"] is True
        assert kwargs["require_enclosure"] is True
        assert kwargs["require_unique_camera_poses"] is True
        assert kwargs["enclosure_room_floor_z_cm"] == 2.0
        assert len(groups) == len(poses) == 4
        assert all(len(value) == 7 for value in groups)
        center_x, center_y = 0.0, 0.0
        assert groups[0][0].x < center_x and groups[0][0].y < center_y
        assert groups[1][0].x > center_x and groups[1][0].y < center_y
        assert groups[2][0].x > center_x and groups[2][0].y > center_y
        assert groups[3][0].x < center_x and groups[3][0].y > center_y
        assert all(value for value in actor_ids)
        return tuple(
            SimpleNamespace(
                pose=group[1],
                to_dict=lambda: {"reason": "overview_visibility_resolved"},
            )
            for group in groups
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    plan = paired_camera_planning.resolve_environment_overview_plan(
        object(),
        _scene(*actors, floor, ceiling),
        _scene(*actors, floor, ceiling, inserted),
        count=4,
        scene_environment="indoor",
        half_extent_m=None,
        timeout_s=300.0,
    )

    assert [value.name for value in plan.views] == [
        "view_1",
        "view_2",
        "view_3",
        "view_4",
    ]
    assert plan.audit["indoor_camera_position_policy"] == (
        "four-inset-room-corners"
    )
    assert plan.audit["indoor_room"]["boundary_kind"] == "shared_floor_bounds"
    assert plan.audit["corner_candidate_count_per_view"] == [7, 7, 7, 7]
    assert plan.audit["visibility_policy"] == (
        "indoor-room-corners-overlap-enclosure-los-unique"
    )
    assert plan.audit["single_scene_reframing_allowed"] is False


def test_indoor_overview_falls_back_to_shared_anchor_orbits_without_room(
    monkeypatch,
):
    chair = _actor("chair", center=(0.0, 0.0, 100.0))
    table = _actor(
        "table",
        center=(300.0, 150.0, 80.0),
        extent=(90.0, 60.0, 80.0),
        label="Table",
        asset="/Game/Props/SM_Table.SM_Table",
    )
    inserted = _actor("inserted", center=(125.0, -75.0, 120.0))

    def resolve(_bridge, _inventory, _bounds, poses, actor_ids, **kwargs):
        groups = kwargs["candidate_poses_by_pose"]
        assert groups is not None
        assert kwargs["overview_mode"] is True
        assert kwargs["require_enclosure"] is True
        assert kwargs["require_unique_camera_poses"] is True
        assert kwargs["enclosure_room_floor_z_cm"] is None
        assert kwargs["required_actor_clear_fraction"] == 0.20
        assert all(value == ("stable:chair", "stable:table") for value in actor_ids)
        assert len(groups) == len(poses) == 4
        assert all(1 < len(value) <= 64 for value in groups)
        assert all(value == groups[0] for value in groups[1:])
        assert kwargs["max_overview_candidates_per_pose"] == 64
        assert kwargs["allow_enclosed_overview_without_visible_actor"] is True
        return tuple(
            SimpleNamespace(
                pose=group[0],
                visible_actor_count=0 if index == 3 else 1,
                to_dict=lambda group=group: {
                    "reason": "overview_visibility_resolved",
                    "resolved_pose": group[0].to_dict(),
                },
            )
            for index, group in enumerate(groups)
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    plan = paired_camera_planning.resolve_environment_overview_plan(
        object(),
        _scene(chair, table),
        _scene(chair, table, inserted),
        count=4,
        scene_environment="indoor",
        half_extent_m=None,
        timeout_s=300.0,
    )

    assert plan.audit["indoor_camera_fallback_stage"] == (
        scene_graph_capture.INDOOR_OVERVIEW_CAMERA_FALLBACK_STAGE
    )
    assert plan.audit["indoor_camera_position_policy"] == (
        "shared-anchor-pool-orbit-enclosed-visibility"
    )
    assert plan.audit["fallback_candidate_count_per_view"]
    assert set(plan.audit["anchor_candidate_count_by_actor"]) == {
        "stable:chair",
        "stable:table",
    }
    assert plan.audit["planning_source"] == "input_gt_shared_actor_geometry"
    assert plan.audit["single_scene_reframing_allowed"] is False


def test_indoor_room_corner_visibility_failure_retries_shared_anchors(
    monkeypatch,
):
    chair = _actor("chair", center=(0.0, 0.0, 100.0))
    floor = _actor(
        "floor",
        center=(0.0, 0.0, 0.0),
        extent=(400.0, 350.0, 2.0),
        label="Floor",
        asset="/Game/Architecture/SM_Floor.SM_Floor",
    )
    inserted = _actor("inserted", center=(125.0, -75.0, 120.0))
    calls = []

    def resolve(_bridge, _inventory, _bounds, poses, _actor_ids, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return tuple(
                SimpleNamespace(
                    pose=None,
                    to_dict=lambda: {"reason": "overview_scene_not_visible"},
                )
                for _pose_value in poses
            )
        return tuple(
            SimpleNamespace(
                pose=pose,
                visible_actor_count=1,
                to_dict=lambda pose=pose: {
                    "reason": "overview_visibility_resolved",
                    "resolved_pose": pose.to_dict(),
                },
            )
            for pose in poses
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    plan = paired_camera_planning.resolve_environment_overview_plan(
        object(),
        _scene(chair, floor),
        _scene(chair, floor, inserted),
        count=4,
        scene_environment="indoor",
        half_extent_m=None,
        timeout_s=300.0,
    )

    assert len(calls) == 2
    assert calls[0]["candidate_poses_by_pose"] is not None
    assert calls[1]["candidate_poses_by_pose"] is not None
    assert plan.audit["indoor_camera_fallback_stage"] == (
        scene_graph_capture.INDOOR_OVERVIEW_CAMERA_FALLBACK_STAGE
    )


def test_indoor_local_camera_uses_ue_enclosure_when_room_is_modular(
    monkeypatch,
):
    chair = _actor("chair", center=(0.0, 0.0, 100.0))
    target = _actor("picture", extent=(60.0, 4.0, 45.0), label="Picture")

    def resolve(_bridge, _inventory, _bounds, poses, actor_ids, **kwargs):
        assert kwargs["candidate_poses_by_pose"] is None
        assert kwargs["require_enclosure"] is True
        assert kwargs["enclosure_room_floor_z_cm"] is None
        assert kwargs["enclosure_room_ceiling_z_cm"] is None
        assert all(value == ("stable:picture",) for value in actor_ids)
        return tuple(
            SimpleNamespace(
                pose=pose,
                to_dict=lambda pose=pose: {
                    "reason": "line_of_sight_resolved",
                    "resolved_pose": pose.to_dict(),
                    "camera_initial_overlap": False,
                    "camera_enclosure_ok": True,
                },
            )
            for pose in poses
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    plan = paired_camera_planning.resolve_repair_target_plan(
        object(),
        _scene(chair, target),
        (target,),
        input_scene=_scene(chair),
        gt_scene=_scene(chair, target),
        count=4,
        scene_environment="indoor",
        half_extent_m=None,
        timeout_s=300.0,
    )

    assert plan.audit["indoor_room"] is None
    assert plan.audit["indoor_room_inference_error"]
    assert plan.audit["indoor_room_requirement"] == (
        "ue_enclosure_trace_fallback"
    )
    assert plan.audit["single_scene_reframing_allowed"] is False


def test_indoor_local_vertical_thin_target_uses_surface_reach_last(
    monkeypatch,
):
    floor = _actor(
        "floor",
        center=(0.0, 0.0, 0.0),
        extent=(400.0, 350.0, 2.0),
        label="Floor",
        asset="/Game/Architecture/SM_Floor.SM_Floor",
    )
    ceiling = _actor(
        "ceiling",
        center=(0.0, 0.0, 300.0),
        extent=(400.0, 350.0, 5.0),
        label="Ceiling",
        asset="/Game/Architecture/SM_Ceiling.SM_Ceiling",
    )
    picture = _actor(
        "picture",
        center=(100.0, 340.0, 145.0),
        extent=(25.0, 1.15, 36.75),
        label="Picture",
        asset="/Game/Props/SM_Picture.SM_Picture",
    )
    wall = _actor(
        "wall",
        center=(100.0, 347.0, 145.0),
        extent=(100.0, 6.0, 120.0),
        label="Wall",
        asset="/Game/Architecture/SM_Wall.SM_Wall",
    )
    cameras = []
    for index, location in enumerate(
        (
            (-250.0, 100.0, 100.0),
            (-50.0, 150.0, 120.0),
            (150.0, 100.0, 110.0),
            (300.0, 175.0, 90.0),
        )
    ):
        camera = _actor(f"camera-{index}", center=location)
        camera["class"] = "/Script/CinematicCamera.CineCameraActor"
        camera["transform"]["location_cm"] = list(location)
        cameras.append(camera)
    calls = []

    def resolve(_bridge, _inventory, _bounds, poses, actor_ids, **kwargs):
        calls.append(kwargs)
        assert all(value == ("stable:picture",) for value in actor_ids)
        if len(calls) < 4:
            return tuple(
                SimpleNamespace(
                    pose=None,
                    to_dict=lambda: {"reason": "no_unoccluded_camera_pose"},
                )
                for _pose_value in poses
            )
        assert kwargs["allow_thin_target_surface_proxy"] is True
        assert kwargs["require_enclosure"] is True
        assert kwargs["require_unique_camera_poses"] is True
        assert kwargs["required_actor_clear_fraction"] == 0.20
        groups = kwargs["candidate_poses_by_pose"]
        assert groups is not None
        assert all(len(group) >= 4 for group in groups)
        assert all(
            pose.y < picture["bounds"]["origin_cm"][1]
            for group in groups
            for pose in group
        )
        return tuple(
            SimpleNamespace(
                pose=pose,
                to_dict=lambda pose=pose: {
                    "reason": "line_of_sight_resolved",
                    "resolved_pose": pose.to_dict(),
                    "direct_target_ray_count": 0,
                    "thin_surface_proxy_ray_count": 3,
                    "thin_surface_open_ray_count": 3,
                    "thin_surface_support_ray_count": 0,
                    "thin_surface_support_actors": [],
                    "thin_surface_proxy_enabled": True,
                },
            )
            for pose in poses
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    plan = paired_camera_planning.resolve_repair_target_plan(
        object(),
        _scene(floor, ceiling, wall, *cameras, picture),
        (picture,),
        input_scene=_scene(floor, ceiling, wall, *cameras),
        gt_scene=_scene(floor, ceiling, wall, *cameras, picture),
        count=4,
        scene_environment="indoor",
        half_extent_m=None,
        timeout_s=300.0,
    )

    assert len(calls) == 4
    assert all(
        call.get("allow_thin_target_surface_proxy") is not True
        for call in calls[:3]
    )
    assert plan.audit["indoor_camera_fallback_stage"] == (
        "target-thin-surface-reach"
    )
    assert plan.audit["visibility_policy"] == (
        "indoor-local-enclosed-thin-surface-reach-unique"
    )
    assert plan.audit["thin_target_support"]["actor_id"] == "stable:wall"
    assert plan.audit["thin_target_support"]["thin_axis"] == 1
    assert plan.audit["thin_target_support"]["exposed_sign"] == -1
    assert plan.audit["shared_authored_camera_position_count"] == 4
    assert plan.audit["shared_authored_camera_actor_ids"] == [
        "stable:camera-1",
        "stable:camera-2",
        "stable:camera-3",
        "stable:camera-0",
    ]
    assert (
        plan.audit["shared_authored_camera_neighborhood_candidate_count"]
        == 20
    )
    assert plan.audit["shared_authored_camera_neighborhood_offsets_cm"] == [
        [0.0, 0.0],
        [-20.0, 0.0],
        [20.0, 0.0],
        [0.0, -15.0],
        [0.0, 15.0],
    ]
    assert plan.audit["thin_surface_camera_side_policy"] == (
        "shared-authored-neighborhood-then-opposite-"
        "input-gt-support-wall"
    )
    assert plan.audit["single_scene_reframing_allowed"] is False


def test_indoor_local_non_thin_target_never_uses_surface_reach(monkeypatch):
    floor = _actor(
        "floor",
        center=(0.0, 0.0, 0.0),
        extent=(400.0, 350.0, 2.0),
        label="Floor",
        asset="/Game/Architecture/SM_Floor.SM_Floor",
    )
    crate = _actor(
        "crate",
        center=(100.0, 100.0, 45.0),
        extent=(45.0, 45.0, 45.0),
        label="Crate",
    )
    calls = []

    def resolve(_bridge, _inventory, _bounds, poses, _actor_ids, **kwargs):
        calls.append(kwargs)
        return tuple(
            SimpleNamespace(
                pose=None,
                to_dict=lambda: {"reason": "no_unoccluded_camera_pose"},
            )
            for _pose_value in poses
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    with pytest.raises(
        paired_camera_planning._CameraPortfolioUnavailable,
        match="no safe visible paired camera",
    ):
        paired_camera_planning.resolve_repair_target_plan(
            object(),
            _scene(floor, crate),
            (crate,),
            input_scene=_scene(floor),
            gt_scene=_scene(floor, crate),
            count=4,
            scene_environment="indoor",
            half_extent_m=None,
            timeout_s=300.0,
        )

    assert len(calls) == 3
    assert calls[0]["required_actor_clear_fraction"] is None
    assert calls[1]["required_actor_clear_fraction"] == 0.10
    assert calls[2]["required_actor_clear_fraction"] == 0.10
    assert all(
        call.get("allow_thin_target_surface_proxy") is not True
        for call in calls
    )


def test_local_plan_uses_resolved_camera_once_and_records_environment(monkeypatch):
    target = _actor("picture", extent=(60.0, 4.0, 45.0), label="Picture")
    floor = _actor(
        "floor",
        center=(0.0, 0.0, 0.0),
        extent=(400.0, 350.0, 2.0),
        label="Floor",
        asset="/Game/Architecture/SM_Floor.SM_Floor",
    )
    ceiling = _actor(
        "ceiling",
        center=(0.0, 0.0, 300.0),
        extent=(400.0, 350.0, 5.0),
        label="Ceiling",
        asset="/Game/Architecture/SM_Ceiling.SM_Ceiling",
    )

    def resolve(_bridge, _inventory, _bounds, poses, actor_ids, **kwargs):
        assert kwargs["fov_degrees"] == 55.0
        assert kwargs["candidate_poses_by_pose"] is None
        assert kwargs["require_enclosure"] is True
        assert kwargs["require_unique_camera_poses"] is True
        assert kwargs["enclosure_room_floor_z_cm"] == 2.0
        assert kwargs["enclosure_room_ceiling_z_cm"] == 295.0
        assert kwargs["required_actor_clear_fraction"] is None
        assert all(value == ("stable:picture",) for value in actor_ids)
        return tuple(
            SimpleNamespace(
                pose=CameraPose(
                    pose.x + 10.0,
                    pose.y,
                    pose.z,
                    pose.pitch,
                    pose.yaw,
                    pose.roll,
                ),
                to_dict=lambda pose=pose: {
                    "reason": "line_of_sight_resolved",
                    "requested_pose": pose.to_dict(),
                },
            )
            for pose in poses
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    plan = paired_camera_planning.resolve_repair_target_plan(
        object(),
        _scene(floor, ceiling, target),
        (target,),
        input_scene=_scene(floor, ceiling),
        gt_scene=_scene(floor, ceiling, target),
        count=4,
        scene_environment="indoor",
        half_extent_m=None,
        timeout_s=300.0,
    )

    assert plan.audit["planner_id"] == (
        scene_graph_capture.GT_REPAIR_TARGET_INDOOR_CAMERA_PLAN
    )
    assert plan.audit["scene_environment"] == "indoor"
    assert plan.audit["single_scene_reframing_allowed"] is False
    assert len(plan.audit["visibility_resolution"]) == 4
    assert plan.audit["camera_specs_sha256"]
    assert plan.audit["hanging_target_detected"] is False
    assert plan.audit["hanging_candidate_count_per_view"] == [0, 0, 0, 0]


def test_two_valid_case_cameras_are_kept_exact_without_generic_switch(monkeypatch):
    target = _actor("target", center=(0.0, 0.0, 100.0), extent=(40.0, 40.0, 40.0))
    floor = _actor(
        "floor",
        center=(0.0, 0.0, 0.0),
        extent=(800.0, 800.0, 2.0),
        label="Floor",
        asset="/Game/Architecture/SM_Floor.SM_Floor",
    )
    ceiling = _actor(
        "ceiling",
        center=(0.0, 0.0, 300.0),
        extent=(800.0, 800.0, 5.0),
        label="Ceiling",
        asset="/Game/Architecture/SM_Ceiling.SM_Ceiling",
    )
    task = SimpleNamespace(
        case_camera_manifest={
            "schema_version": "scenebench.indoor_review_case_cameras.v1",
            "views": [
                {
                    "name": "view-01",
                    "location_cm": [-500.0, 0.0, 100.0],
                    "target_cm": [0.0, 0.0, 100.0],
                    "fov_deg": 60.0,
                    "fill_intensity": 9000.0,
                },
                {
                    "name": "view-02",
                    "location_cm": [0.0, -500.0, 100.0],
                    "target_cm": [0.0, 0.0, 100.0],
                    "fov_deg": 58.0,
                    "exposure_bias": 5.0,
                },
            ],
        },
        case_camera_manifest_sha256="a" * 64,
    )
    portfolio = paired_camera_planning.case_camera_portfolio(task)
    portfolio = paired_camera_planning.expand_case_camera_runtime_neighborhood(
        portfolio,
        (target,),
        count=4,
    )
    calls = []

    def resolve(_bridge, _inventory, _bounds, poses, actor_ids, **kwargs):
        calls.append((poses, actor_ids, kwargs))
        return tuple(
            SimpleNamespace(
                pose=pose,
                visible_actor_count=1,
                to_dict=lambda pose=pose: {
                    "reason": "line_of_sight_resolved",
                    "resolved_pose": pose.to_dict(),
                    "camera_initial_overlap": False,
                    "camera_enclosure_ok": True,
                },
            )
            for pose in poses
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    plan = paired_camera_planning.resolve_repair_target_plan(
        object(),
        _scene(floor, ceiling, target),
        (target,),
        input_scene=_scene(floor, ceiling),
        gt_scene=_scene(floor, ceiling, target),
        count=4,
        scene_environment="indoor",
        half_extent_m=None,
        timeout_s=300.0,
        case_cameras=portfolio,
        case_camera_input_gt_observability=_runtime_observability(
            **{name: True for name in portfolio.authored_view_names}
        ),
    )

    assert len(calls) == 4
    assert [len(value[0]) for value in calls] == [1, 1, 1, 1]
    assert calls[0][2]["candidate_poses_by_pose"] == ((calls[0][0][0],),)
    assert calls[1][2]["candidate_poses_by_pose"] == ((calls[1][0][0],),)
    assert calls[0][2]["fov_degrees"] == 60.0
    assert calls[1][2]["fov_degrees"] == 58.0
    assert [tuple(value.location) for value in plan.views[:2]] == [
        (-500.0, 0.0, 100.0),
        (0.0, -500.0, 100.0),
    ]
    assert [value.fov_deg for value in plan.views[:2]] == [60.0, 58.0]
    assert plan.audit["planner_id"] == (
        scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN
    )
    assert plan.audit["case_camera_attempt"] == "accepted_exact_pose"
    assert plan.audit["case_camera_view_count"] == 4
    assert plan.audit["fill_view_count"] == 0
    assert plan.audit["view_count"] == 4
    assert plan.audit["requested_view_count"] == 4
    assert plan.audit["case_camera_direct_hit_satisfied"] is True
    assert plan.audit["case_camera_policy"] == "runtime-input-gt-authored-neighborhood-full-hit-else-generic-fallback"
    assert plan.audit["case_camera_candidate_used_for_acceptance"] is False
    assert portfolio.runtime_neighbor_sources == (
        None,
        None,
        "view-01",
        "view-02",
        "view-01",
        "view-02",
        "view-01",
        "view-02",
    )
    assert [tuple(value.location) for value in portfolio.views[2:]] == [
        (-500.0, 0.0, 100.0),
        (0.0, -500.0, 100.0),
        (-500.0, 0.0, 100.0),
        (0.0, -500.0, 100.0),
        (-500.0, 0.0, 100.0),
        (0.0, -500.0, 100.0),
    ]
    assert len(
        {
            paired_camera_planning.camera_pose_key(
                paired_camera_planning._pose(value)
            )
            for value in portfolio.views
        }
    ) == 8
    assert "fill_intensity" not in str(plan.audit)
    assert "exposure_bias" not in str(plan.audit)


def test_runtime_input_gt_makes_framing_and_los_advisory_on_direct_hit(
    monkeypatch,
):
    target = _actor("target", center=(0.0, 0.0, 100.0), extent=(40.0, 40.0, 40.0))
    passing_gates = {
        "camera_similarity": True,
        "gt_repeat_stable": True,
        "locality": True,
        "weak_visibility": True,
    }
    portfolio = paired_camera_planning.CaseCameraPortfolio(
        views=(
            render.Viewpoint(
                "case_view_1",
                [-500.0, 0.0, 100.0],
                [0.0, 0.0, 180.0],
                60.0,
            ),
            render.Viewpoint(
                "case_view_2",
                [0.0, -500.0, 100.0],
                [0.0, 0.0, 90.0],
                58.0,
            ),
        ),
        manifest_sha256="b" * 64,
        schema_version="scenebench.indoor_review_case_cameras.v1",
        authored_view_names=("view-01", "view-02"),
        paired_render_qa_sha256="c" * 64,
        paired_render_qa_by_view=tuple(
            {
                "view": name,
                "gates": passing_gates,
                "robust_delta": {
                    "fraction": fraction,
                    "policy": "gt-a-input-gt-b-robust-v1",
                },
            }
            for name, fraction in (("view-01", 0.02), ("view-02", 0.04))
        ),
    )
    portfolio = paired_camera_planning.expand_case_camera_runtime_neighborhood(
        portfolio,
        (target,),
        count=4,
    )
    calls = []

    def resolve(_bridge, _inventory, _bounds, poses, actor_ids, **kwargs):
        calls.append((poses, actor_ids, kwargs))
        if len(calls) <= 4:
            return (
                SimpleNamespace(
                    pose=None,
                    to_dict=lambda: {
                        "reason": "no_unoccluded_camera_pose",
                        "resolved_pose": None,
                        "visible_actor_count": 0,
                        "camera_initial_overlap": False,
                        "camera_enclosure_ok": True,
                    },
                ),
            )
        return tuple(
            SimpleNamespace(
                pose=pose,
                visible_actor_count=1,
                to_dict=lambda pose=pose: {
                    "reason": "line_of_sight_resolved",
                    "resolved_pose": pose.to_dict(),
                },
            )
            for pose in poses
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    plan = paired_camera_planning.resolve_repair_target_plan(
        object(),
        _scene(target),
        (target,),
        input_scene=_scene(),
        gt_scene=_scene(target),
        count=4,
        scene_environment="indoor",
        half_extent_m=None,
        timeout_s=300.0,
        case_cameras=portfolio,
        case_camera_input_gt_observability=_runtime_observability(
            **{name: True for name in portfolio.authored_view_names}
        ),
    )

    assert [len(value[0]) for value in calls] == [1, 1, 1, 1]
    assert [tuple(value.location) for value in plan.views[:2]] == [
        (-500.0, 0.0, 100.0),
        (0.0, -500.0, 100.0),
    ]
    assert plan.audit["case_camera_view_count"] == 4
    assert plan.audit["case_camera_rejected_view_count"] == 0
    assert plan.audit["fill_view_count"] == 0
    assert plan.audit["requested_view_count"] == 4
    assert plan.audit["case_camera_policy"] == "runtime-input-gt-authored-neighborhood-full-hit-else-generic-fallback"
    assert plan.audit["case_camera_acceptance_basis"] == (
        "current_task_input_gt_same_pose_rgb_observability"
    )
    assert plan.audit["case_camera_candidate_used_for_acceptance"] is False
    assert plan.audit["case_camera_release_qa_role"] == (
        "optional_provenance_only"
    )
    assert plan.audit["case_camera_geometry_gate_mode"] == (
        "overlap_enclosure_hard_framing_los_advisory"
    )
    assert plan.audit["case_camera_paired_render_qa_sha256"] == "c" * 64
    first_audit = plan.audit["visibility_resolution"][0]
    assert first_audit["selection"] == (
        "case_camera_runtime_input_gt_exact_pose"
    )
    assert first_audit["geometry_advisory"]["framing"]["healthy"] is False
    assert first_audit["geometry_advisory"]["live_visibility"]["reason"] == (
        "no_unoccluded_camera_pose"
    )


def test_rotation_neighborhood_can_fill_from_one_safe_authored_location(
    monkeypatch,
):
    target = _actor("target", center=(0.0, 0.0, 100.0), extent=(40.0, 40.0, 40.0))
    base = paired_camera_planning.CaseCameraPortfolio(
        views=(
            render.Viewpoint(
                "case_view_1",
                [-500.0, 0.0, 100.0],
                [0.0, 0.0, 0.0],
                60.0,
            ),
            render.Viewpoint(
                "case_view_2",
                [0.0, -500.0, 100.0],
                [0.0, 0.0, 90.0],
                58.0,
            ),
        ),
        manifest_sha256="b" * 64,
        schema_version="scenebench.indoor_review_case_cameras.v1",
        authored_view_names=("view-01", "view-02"),
    )
    portfolio = paired_camera_planning.expand_case_camera_runtime_neighborhood(
        base,
        (target,),
        count=4,
    )
    calls = []

    def resolve(_bridge, _inventory, _bounds, poses, actor_ids, **kwargs):
        del actor_ids, kwargs
        pose = poses[0]
        calls.append(pose)
        safe = pose.x == -500.0
        return (
            SimpleNamespace(
                pose=pose if safe else None,
                to_dict=lambda safe=safe, pose=pose: {
                    "reason": (
                        "line_of_sight_resolved"
                        if safe
                        else "camera_initial_overlap"
                    ),
                    "resolved_pose": pose.to_dict() if safe else None,
                    "camera_initial_overlap": not safe,
                    "camera_enclosure_ok": True,
                },
            ),
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    plan = paired_camera_planning.resolve_repair_target_plan(
        object(),
        _scene(target),
        (target,),
        input_scene=_scene(),
        gt_scene=_scene(target),
        count=4,
        scene_environment="indoor",
        half_extent_m=None,
        timeout_s=300.0,
        case_cameras=portfolio,
        case_camera_input_gt_observability=_runtime_observability(
            **{name: True for name in portfolio.authored_view_names}
        ),
    )

    assert len(calls) == 7
    assert {tuple(value.location) for value in plan.views} == {
        (-500.0, 0.0, 100.0)
    }
    assert plan.audit["case_camera_view_count"] == 4
    assert plan.audit["case_camera_rejected_view_count"] == 3
    assert plan.audit["fill_view_count"] == 0
    assert plan.audit["case_camera_direct_hit_satisfied"] is True


def test_failed_case_camera_is_replaced_while_an_exact_hit_is_kept(monkeypatch):
    target = _actor("target", center=(0.0, 0.0, 100.0), extent=(40.0, 40.0, 40.0))
    portfolio = paired_camera_planning.CaseCameraPortfolio(
        views=(
            render.Viewpoint(
                "case_view_1",
                [-500.0, 0.0, 100.0],
                [0.0, 0.0, 0.0],
                60.0,
            ),
            render.Viewpoint(
                "case_view_2",
                [0.0, -500.0, 100.0],
                [0.0, 0.0, 90.0],
                58.0,
            ),
        ),
        manifest_sha256="b" * 64,
        schema_version="scenebench.indoor_review_case_cameras.v1",
        authored_view_names=("view-01", "view-02"),
    )
    calls = []

    def resolve(_bridge, _inventory, _bounds, poses, actor_ids, **kwargs):
        calls.append((poses, actor_ids, kwargs))
        if len(calls) == 1:
            return (
                SimpleNamespace(
                    pose=None,
                    to_dict=lambda: {"reason": "no_unoccluded_camera_pose"},
                ),
            )
        return tuple(
            SimpleNamespace(
                pose=pose,
                to_dict=lambda pose=pose: {
                    "reason": "line_of_sight_resolved",
                    "resolved_pose": pose.to_dict(),
                    "camera_initial_overlap": False,
                    "camera_enclosure_ok": True,
                },
            )
            for pose in poses
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    plan = paired_camera_planning.resolve_repair_target_plan(
        object(),
        _scene(target),
        (target,),
        input_scene=_scene(),
        gt_scene=_scene(target),
        count=4,
        scene_environment="indoor",
        half_extent_m=None,
        timeout_s=300.0,
        case_cameras=portfolio,
        case_camera_input_gt_observability=_runtime_observability(
            **{"view-01": False, "view-02": True}
        ),
    )

    assert [len(value[0]) for value in calls] == [1, 1, 3]
    assert plan.audit["planner_id"] == (
        scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN
    )
    assert plan.audit["case_camera_attempt"] == "accepted_exact_pose"
    assert plan.audit["case_camera_view_count"] == 1
    assert plan.audit["case_camera_rejected_view_count"] == 1
    assert plan.audit["case_camera_rejections"][0]["authored_view"] == "view-01"
    assert plan.audit["fill_view_count"] == 3
    assert tuple(plan.views[0].location) == (0.0, -500.0, 100.0)
    assert [
        value["view"] for value in plan.audit["visibility_resolution"]
    ] == ["view_1", "view_2", "view_3", "view_4"]
    assert plan.audit["visibility_resolution"][0]["authored_view"] == "view-02"


def test_all_failed_case_cameras_cause_full_four_view_generic_fallback(monkeypatch):
    target = _actor("target", center=(0.0, 0.0, 100.0), extent=(40.0, 40.0, 40.0))
    portfolio = paired_camera_planning.CaseCameraPortfolio(
        views=(
            render.Viewpoint(
                "case_view_1",
                [-500.0, 0.0, 100.0],
                [0.0, 0.0, 0.0],
                60.0,
            ),
            render.Viewpoint(
                "case_view_2",
                [0.0, -500.0, 100.0],
                [0.0, 0.0, 90.0],
                58.0,
            ),
        ),
        manifest_sha256="b" * 64,
        schema_version="scenebench.indoor_review_case_cameras.v1",
        authored_view_names=("view-01", "view-02"),
    )
    calls = []

    def resolve(_bridge, _inventory, _bounds, poses, actor_ids, **kwargs):
        calls.append((poses, actor_ids, kwargs))
        if len(calls) <= 2:
            return (
                SimpleNamespace(
                    pose=None,
                    to_dict=lambda: {"reason": "no_unoccluded_camera_pose"},
                ),
            )
        return tuple(
            SimpleNamespace(
                pose=pose,
                to_dict=lambda pose=pose: {
                    "reason": "line_of_sight_resolved",
                    "resolved_pose": pose.to_dict(),
                },
            )
            for pose in poses
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    plan = paired_camera_planning.resolve_repair_target_plan(
        object(),
        _scene(target),
        (target,),
        input_scene=_scene(),
        gt_scene=_scene(target),
        count=4,
        scene_environment="indoor",
        half_extent_m=None,
        timeout_s=300.0,
        case_cameras=portfolio,
        case_camera_input_gt_observability=_runtime_observability(
            **{"view-01": False, "view-02": False}
        ),
    )

    assert [len(value[0]) for value in calls] == [1, 1, 4]
    assert plan.audit["planner_id"] == (
        scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN
    )
    assert plan.audit["case_camera_attempt"] == "full-generic-strategy-fallback"
    assert plan.audit["case_camera_view_count"] == 0
    assert plan.audit["fill_view_count"] == 4
    assert "no case camera passed" in plan.audit["case_camera_fallback_reason"]


def test_indoor_local_v4_adds_bounded_room_side_views_for_hanging_target(
    monkeypatch,
):
    floor = _actor(
        "floor",
        center=(0.0, 0.0, 0.0),
        extent=(400.0, 350.0, 2.0),
        label="Floor",
        asset="/Game/Architecture/SM_Floor.SM_Floor",
    )
    ceiling = _actor(
        "ceiling",
        center=(0.0, 0.0, 300.0),
        extent=(400.0, 350.0, 5.0),
        label="Ceiling",
        asset="/Game/Architecture/SM_Ceiling.SM_Ceiling",
    )
    lamp = _actor(
        "ceiling-lamp",
        center=(100.0, -75.0, 240.0),
        extent=(35.0, 35.0, 60.0),
        label="Unlabelled Target",
    )

    def resolve(_bridge, _inventory, _bounds, poses, actor_ids, **kwargs):
        groups = kwargs["candidate_poses_by_pose"]
        assert groups is not None
        assert kwargs["overview_mode"] is False
        assert kwargs["require_enclosure"] is True
        assert kwargs["require_unique_camera_poses"] is True
        assert kwargs["enclosure_room_floor_z_cm"] == 2.0
        assert kwargs["enclosure_room_ceiling_z_cm"] == 295.0
        assert kwargs["required_actor_clear_fraction"] == 0.20
        selected = []
        for group in groups:
            below = next(
                value
                for value in group
                if value.z < 240.0 and value.pitch > 0.0
            )
            assert -365.0 <= below.x <= 365.0
            assert -315.0 <= below.y <= 315.0
            selected.append(below)
        assert len({value.location_cm for value in selected}) == 4
        assert all(value == ("stable:ceiling-lamp",) for value in actor_ids)
        return tuple(
            SimpleNamespace(
                pose=pose,
                to_dict=lambda pose=pose: {
                    "reason": "line_of_sight_resolved",
                    "resolved_pose": pose.to_dict(),
                },
            )
            for pose in selected
        )

    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        resolve,
    )
    plan = paired_camera_planning.resolve_repair_target_plan(
        object(),
        _scene(floor, ceiling, lamp),
        (lamp,),
        input_scene=_scene(floor, ceiling),
        gt_scene=_scene(floor, ceiling, lamp),
        count=4,
        scene_environment="indoor",
        half_extent_m=None,
        timeout_s=300.0,
    )

    assert plan.audit["planner_id"] == (
        scene_graph_capture.GT_REPAIR_TARGET_INDOOR_CAMERA_PLAN
    )
    assert plan.audit["hanging_target_detected"] is True
    assert all(
        0 < value <= 72
        for value in plan.audit["hanging_candidate_count_per_view"]
    )
    assert all(value.location[2] < 240.0 for value in plan.views)
    assert all(value.rotation[1] > 0.0 for value in plan.views)
    assert len({tuple(value.location) for value in plan.views}) == 4


def test_outdoor_overview_preserves_dense_core_without_live_reframing(monkeypatch):
    actors = (
        _actor("one", center=(0.0, 0.0, 100.0)),
        _actor("two", center=(500.0, 200.0, 100.0)),
    )
    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        lambda *_args, **_kwargs: pytest.fail("Outdoor overview must not reframe"),
    )

    plan = paired_camera_planning.resolve_environment_overview_plan(
        object(),
        _scene(*actors),
        _scene(*actors),
        count=4,
        scene_environment="outdoor",
        half_extent_m=None,
        timeout_s=300.0,
    )

    assert plan.audit["scene_environment"] == "outdoor"
    assert plan.audit["planning_source"] == "gt"
    assert {
        value["reason"] for value in plan.audit["visibility_resolution"]
    } == {"outdoor-dense-core-preserved"}


def test_outdoor_target_preserves_current_framing_without_live_reframing(monkeypatch):
    target = _actor("bin", extent=(35.0, 35.0, 60.0), label="Trash Bin")
    monkeypatch.setattr(
        paired_camera_planning,
        "resolve_visibility_camera_poses",
        lambda *_args, **_kwargs: pytest.fail("Outdoor target must not reframe"),
    )

    plan = paired_camera_planning.resolve_repair_target_plan(
        object(),
        _scene(target),
        (target,),
        count=4,
        scene_environment="outdoor",
        half_extent_m=None,
        timeout_s=300.0,
    )

    assert plan.audit["scene_environment"] == "outdoor"
    assert plan.audit["planner_id"] == (
        scene_graph_capture.GT_REPAIR_TARGET_OUTDOOR_CAMERA_PLAN
    )
    assert plan.audit["pairing_policy"] == (
        "gt-repair-target-visibility-frozen-absolute-cameras"
    )
    assert plan.audit["planning_source"] == "gt_minus_input_target"
    assert plan.audit["visibility_policy"] == "outdoor-target-local-preserved"
    assert {
        value["reason"] for value in plan.audit["visibility_resolution"]
    } == {"outdoor-target-local-preserved"}


def test_v23_requires_explicit_environment():
    with pytest.raises(ValueError, match="scene_environment"):
        paired_camera_planning.require_scene_environment(SimpleNamespace())


@pytest.mark.parametrize("environment", ("indoor", "outdoor"))
def test_canonical_v23_gt_repair_routes_environment_specific_evidence(
    tmp_path,
    environment,
):
    from synthetic_tasks import image_to_scene

    task = task_mod.load(image_to_scene(tmp_path, environment))
    requests = gt_repair.evidence_requests(task, task.verifiers[0])
    by_protocol = {value.protocol: value for value in requests}

    local_protocol = (
        GT_REPAIR_TARGET_VISUAL_PROTOCOL
        if environment == "indoor"
        else GT_REPAIR_TARGET_VISUAL_ENVIRONMENT_RENDER_PROTOCOL
    )
    expected_protocols = {local_protocol}
    if environment == "outdoor":
        expected_protocols.update(
            {
                CAPTION_ENVIRONMENT_RENDER_PROTOCOL,
                GT_VISUAL_ENVIRONMENT_RENDER_PROTOCOL,
            }
        )
    assert set(by_protocol) == expected_protocols
    assert all(value.scene_environment == environment for value in requests)
    if environment == "outdoor":
        assert by_protocol[CAPTION_ENVIRONMENT_RENDER_PROTOCOL].camera_protocol == (
            render.GT_CAPTION_ENVIRONMENT_SCENE_GRAPH_PROTOCOL
        )
        assert by_protocol[
            GT_VISUAL_ENVIRONMENT_RENDER_PROTOCOL
        ].camera_plan_policy == (
            scene_graph_capture.GT_VISUAL_CAMERA_PLAN
        )
    assert by_protocol[local_protocol].camera_plan_policy == (
        scene_graph_capture.GT_REPAIR_TARGET_INDOOR_CAMERA_PLAN
        if environment == "indoor"
        else scene_graph_capture.GT_REPAIR_TARGET_OUTDOOR_CAMERA_PLAN
    )
    assert by_protocol[local_protocol].paired_visual_quality_gate == (
        GT_REPAIR_TARGET_VISUAL_QUALITY_GATE
        if environment == "indoor"
        else None
    )


def test_indoor_case_camera_task_uses_distinct_authored_local_evidence_protocol(tmp_path):
    from synthetic_tasks import image_to_scene

    root = tmp_path
    base = task_mod.load(image_to_scene(tmp_path, "indoor"))
    task = SimpleNamespace(
        path=base.path,
        half_extent_m=base.half_extent_m,
        kind=base.kind,
        case_type=base.case_type,
        data=base.data,
        verifiers=base.verifiers,
        scene_environment=base.scene_environment,
        case_camera_manifest_path=root / "camera-manifests" / "synthetic.json",
    )

    requests = gt_repair.evidence_requests(task, task.verifiers[0])
    by_protocol = {value.protocol: value for value in requests}

    assert set(by_protocol) == {
        GT_REPAIR_TARGET_AUTHORED_VISUAL_PROTOCOL
    }
    assert by_protocol[
        GT_REPAIR_TARGET_AUTHORED_VISUAL_PROTOCOL
    ].camera_plan_policy == scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN
    assert by_protocol[
        GT_REPAIR_TARGET_AUTHORED_VISUAL_PROTOCOL
    ].view_count == scene_graph_capture.GT_REPAIR_TARGET_CASE_CAMERA_VIEW_COUNT
    assert by_protocol[
        GT_REPAIR_TARGET_AUTHORED_VISUAL_PROTOCOL
    ].paired_visual_quality_gate == GT_REPAIR_TARGET_VISUAL_QUALITY_GATE
