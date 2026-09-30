"""Image-to-scene repair tasks route environment-specific visual evidence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from code4scene.evaluation import render, scene_graph_capture
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
