"""Regression tests for image-to-scene repair-target scene diff."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from code4scene.evaluation import contracts, scene_graph_capture
from code4scene.evaluation.context import Context
from code4scene.evaluation.gt_geometry_compare import compare_detailed
from code4scene.evaluation.repair_target_scope import (
    RepairTargetScope,
    measure_repair_target,
)
from code4scene.evaluation.requirement_graph.repair_target_authoring import (
    derive_repair_targets,
)
from code4scene.evaluation.verifiers import gt_repair, scene_diff


IDS = {"task_bundle_id": "bundle", "episode_id": "episode"}


def _actor(actor_id: str, x: float, *, asset: str | None = None) -> dict:
    mesh = asset or actor_id
    return {
        "stable_actor_id": actor_id,
        "label": actor_id,
        "actor_path": f"/Game/Test.Test:PersistentLevel.{actor_id}",
        "class": "/Script/Engine.StaticMeshActor",
        "asset_path": f"/Game/Test/{mesh}.{mesh}",
        "actor_tags": [],
        "transform": {
            "location_cm": [x, 0.0, 50.0],
            "rotation_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "bounds": {
            "origin_cm": [x, 0.0, 50.0],
            "extent_cm": [20.0, 20.0, 20.0],
        },
        "properties": {},
    }


def _scene(actors: list[dict]) -> dict:
    return {
        "actors": actors,
        "actor_count": len(actors),
        "export_metadata": {"status": "success"},
    }


def test_repair_target_projection_exposes_error_hidden_by_global_average():
    retained = [_actor(f"keep-{index}", index * 100.0) for index in range(12)]
    desired = [
        _actor("target-left", 2_000.0, asset="book-stack"),
        _actor("target-right", 2_050.0, asset="book-stack"),
    ]
    displaced = [
        _actor("generated-left", 2_400.0, asset="book-stack"),
        _actor("generated-right", 2_500.0, asset="book-stack"),
    ]
    unrelated = _actor("unrelated-lamp", 5_000.0, asset="lamp")
    input_scene = _scene(retained)
    gt_scene = _scene([*retained, *desired])
    candidate_scene = _scene([*retained, *displaced, unrelated])
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=gt_scene,
        targets=derive_repair_targets(input_scene, gt_scene),
    )

    comparison = compare_detailed(
        candidate_scene["actors"],
        gt_scene["actors"],
    )
    target = measure_repair_target(scope, candidate_scene, comparison)

    assert target["desired_actor_count"] == 2
    assert target["candidate_target_actor_count"] == 2
    assert target["matched_actor_count"] == 2
    assert target["actor_correspondence_score"] == pytest.approx(1.0)
    assert target["identity_score"] == pytest.approx(1.0)
    assert target["position_rmse_cm"] > comparison["metrics"][
        "aligned_position_rmse_cm"
    ]
    assert 0.0 < target["position_similarity"] < 0.25
    assert target["pairwise_layout_pair_count"] == 1
    assert target["pairwise_layout_similarity"] is not None
    assert target["mean_footprint_iou"] is not None
    assert "matched_logical_rows" not in target
    assert "fragmentation_similarity" not in target


def test_identical_repair_targets_are_exchangeable_after_global_alignment():
    retained = [
        _actor(f"keep-{index}", index * 250.0, asset=f"keep-{index}")
        for index in range(24)
    ]
    input_table = _actor("table-existing", 8_000.0, asset="table")
    desired_existing = _actor("table-existing", 10_000.0, asset="table")
    desired_added = _actor("table-added", 10_700.0, asset="table")
    candidate_existing = _actor("table-existing", 10_700.0, asset="table")
    candidate_added = _actor("table-generated", 10_000.0, asset="table")
    input_scene = _scene([*retained, input_table])
    gt_scene = _scene([*retained, desired_existing, desired_added])
    candidate_scene = _scene([
        *retained,
        candidate_existing,
        candidate_added,
    ])
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=gt_scene,
        targets=derive_repair_targets(input_scene, gt_scene),
    )
    comparison = compare_detailed(
        candidate_scene["actors"],
        gt_scene["actors"],
    )

    stable_rows = [
        row
        for row in comparison["audit"]["actor_matches"]
        if set(row["canonical"]["stable_actor_ids"])
        & {"table-existing", "table-added"}
    ]
    stable_rmse = sum(
        float(row["aligned_center_distance_cm"]) ** 2
        for row in stable_rows
    ) ** 0.5 / len(stable_rows) ** 0.5
    target = measure_repair_target(scope, candidate_scene, comparison)

    assert target["exchangeable_group_count"] == 1
    assert target["exchangeable_reassigned_actor_count"] == 2
    assert target["correspondence_policy"] == (
        "input_gt_frozen_targets_then_hard_compatibility_global_assignment"
    )
    assert target["matching_algorithm_version"] == (
        "repair-target-actor-correspondence.v2"
    )
    assert target["position_rmse_cm"] < stable_rmse * 0.2
    assert target["position_similarity"] > 0.5
    assert {
        row["identity_source"] for row in target["matched_actor_rows"]
    } == {"hard_compatible_global_min_cost_assignment"}
    matched_ids = {
        next(iter(row["candidate"]["stable_actor_ids"])):
        next(iter(row["canonical"]["stable_actor_ids"]))
        for row in target["matched_actor_rows"]
    }
    assert matched_ids == {
        "table-existing": "table-added",
        "table-generated": "table-existing",
    }


def test_materials_are_scored_after_structural_correspondence():
    retained = [
        _actor(f"keep-{index}", index * 250.0, asset=f"keep-{index}")
        for index in range(24)
    ]
    input_table = _actor("table-existing", 8_000.0, asset="table")
    desired_existing = _actor("table-existing", 10_000.0, asset="table")
    desired_added = _actor("table-added", 10_700.0, asset="table")
    candidate_existing = _actor("table-existing", 10_700.0, asset="table")
    candidate_added = _actor("table-generated", 10_000.0, asset="table")
    for actor in (input_table, desired_existing, candidate_existing):
        actor["material_paths"] = ["/Game/Test/Red.Red"]
    for actor in (desired_added, candidate_added):
        actor["material_paths"] = ["/Game/Test/Blue.Blue"]
    input_scene = _scene([*retained, input_table])
    gt_scene = _scene([*retained, desired_existing, desired_added])
    candidate_scene = _scene([
        *retained,
        candidate_existing,
        candidate_added,
    ])
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=gt_scene,
        targets=derive_repair_targets(input_scene, gt_scene),
    )
    comparison = compare_detailed(
        candidate_scene["actors"],
        gt_scene["actors"],
    )

    target = measure_repair_target(scope, candidate_scene, comparison)

    assert target["exchangeable_group_count"] == 1
    assert target["exchangeable_reassigned_actor_count"] == 2
    assert target["position_rmse_cm"] < 50.0
    assert target["material_set_match_rate"] == 0.0


def test_renamed_reguided_repathed_actor_matches_by_structure():
    desired = _actor("gt-chair", 1_000.0, asset="chair")
    candidate = _actor("candidate-chair", 1_000.0, asset="chair")
    desired.update({
        "name": "Chair_GT",
        "actor_guid": "GT-GUID",
        "actor_path": "/Game/GT.GT:PersistentLevel.Chair_GT",
    })
    candidate.update({
        "name": "Chair_New",
        "actor_guid": "CANDIDATE-GUID",
        "actor_path": "/Game/Candidate.Candidate:PersistentLevel.Chair_New",
    })
    input_scene = _scene([])
    gt_scene = _scene([desired])
    candidate_scene = _scene([candidate])
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=gt_scene,
        targets=derive_repair_targets(input_scene, gt_scene),
    )

    target = measure_repair_target(
        scope,
        candidate_scene,
        compare_detailed(candidate_scene["actors"], gt_scene["actors"]),
    )

    assert target["pair_coverage"] == 1.0
    assert target["matched_actor_count"] == 1
    assert target["name_guid_path_used_as_hint_only"] is True
    assert target["failure_classification"] == ["identity_changed"]
    assignment = target["final_assignment"][0]
    assert assignment["identity_hint_matches"] == []
    assert assignment["compatibility"]["compatible"] is True


def test_three_identical_actors_use_global_minimum_assignment():
    desired = [
        _actor(f"gt-{index}", x, asset="crate")
        for index, x in enumerate((1_000.0, 2_000.0, 3_000.0))
    ]
    candidates = [
        _actor(actor_id, x, asset="crate")
        for actor_id, x in (
            ("candidate-c", 3_000.0),
            ("candidate-a", 1_000.0),
            ("candidate-b", 2_000.0),
        )
    ]
    input_scene = _scene([])
    gt_scene = _scene(desired)
    candidate_scene = _scene(candidates)
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=gt_scene,
        targets=derive_repair_targets(input_scene, gt_scene),
    )

    target = measure_repair_target(
        scope,
        candidate_scene,
        compare_detailed(candidate_scene["actors"], gt_scene["actors"]),
    )

    assert target["matched_actor_count"] == 3
    assert target["position_rmse_cm"] == pytest.approx(0.0)
    positions = {
        row["candidate"]["center_cm"][0]: row["canonical"]["center_cm"][0]
        for row in target["matched_actor_rows"]
    }
    assert positions == {1_000.0: 1_000.0, 2_000.0: 2_000.0, 3_000.0: 3_000.0}


def test_correct_actor_at_wrong_position_is_measured_not_missing():
    retained = [
        _actor(f"keep-{index}", index * 200.0, asset=f"keep-{index}")
        for index in range(12)
    ]
    desired = _actor("gt-chair", 10_000.0, asset="chair")
    candidate = _actor("new-chair", 14_000.0, asset="chair")
    input_scene = _scene(retained)
    gt_scene = _scene([*retained, desired])
    candidate_scene = _scene([*retained, candidate])
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=gt_scene,
        targets=derive_repair_targets(input_scene, gt_scene),
    )

    target = measure_repair_target(
        scope,
        candidate_scene,
        compare_detailed(candidate_scene["actors"], gt_scene["actors"]),
    )

    assert target["matched_actor_count"] == 1
    assert target["pair_coverage"] == 1.0
    assert target["position_rmse_cm"] > 3_000.0
    assert 0.0 < target["position_similarity"] < 0.1


def test_wrong_mesh_is_measured_zero_instead_of_null():
    desired = _actor("gt-chair", 1_000.0, asset="chair")
    candidate = _actor("candidate-table", 1_000.0, asset="table")
    input_scene = _scene([])
    gt_scene = _scene([desired])
    candidate_scene = _scene([candidate])
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=gt_scene,
        targets=derive_repair_targets(input_scene, gt_scene),
    )
    target = measure_repair_target(
        scope,
        candidate_scene,
        compare_detailed(candidate_scene["actors"], gt_scene["actors"]),
    )

    assert target["matched_actor_count"] == 0
    assert target["actor_correspondence_score"] == 0.0
    assert target["identity_score"] == 0.0
    assert target["failure_classification"] == ["missing_or_wrong_asset"]
    report = gt_repair.report_from_repair_target_measurement(
        Context(record={}, task=None, ids=IDS, spec={}),
        target,
    )
    assert report["status"] == contracts.MEASURED
    assert report["score"] == 0.0
    assert report["evidence"]["actor_correspondence"][
        "matching_algorithm_version"
    ] == "repair-target-actor-correspondence.v2"


def test_missing_actor_penalizes_matched_channels_by_coverage():
    retained = [
        _actor(f"keep-{index}", index * 200.0, asset=f"keep-{index}")
        for index in range(12)
    ]
    desired = [
        _actor("gt-left", 1_000.0, asset="chair"),
        _actor("gt-right", 2_000.0, asset="chair"),
    ]
    candidate = _actor("candidate-left", 1_000.0, asset="chair")
    input_scene = _scene(retained)
    gt_scene = _scene([*retained, *desired])
    candidate_scene = _scene([*retained, candidate])
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=gt_scene,
        targets=derive_repair_targets(input_scene, gt_scene),
    )

    target = measure_repair_target(
        scope,
        candidate_scene,
        compare_detailed(candidate_scene["actors"], gt_scene["actors"]),
    )

    assert target["matched_actor_count"] == 1
    assert target["missing_actor_count"] == 1
    assert target["pair_coverage"] == 0.5
    assert target["position_similarity"] == pytest.approx(0.5)
    assert target["identity_score"] == pytest.approx(2.0 / 3.0)


def test_extra_compatible_actor_is_penalized():
    desired = _actor("gt-chair", 1_000.0, asset="chair")
    candidates = [
        _actor("candidate-chair", 1_000.0, asset="chair"),
        _actor("extra-chair", 1_500.0, asset="chair"),
    ]
    input_scene = _scene([])
    gt_scene = _scene([desired])
    candidate_scene = _scene(candidates)
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=gt_scene,
        targets=derive_repair_targets(input_scene, gt_scene),
    )

    target = measure_repair_target(
        scope,
        candidate_scene,
        compare_detailed(candidate_scene["actors"], gt_scene["actors"]),
    )

    assert target["candidate_target_actor_count"] == 2
    assert target["matched_actor_count"] == 1
    assert target["extra_actor_count"] == 1
    assert target["actor_correspondence_score"] == pytest.approx(2.0 / 3.0)


def test_unchanged_same_mesh_actor_cannot_satisfy_add_target():
    existing = _actor("existing-chair", 1_000.0, asset="chair")
    desired_addition = _actor("new-chair", 1_100.0, asset="chair")
    input_scene = _scene([existing])
    gt_scene = _scene([existing, desired_addition])
    candidate_scene = _scene([copy.deepcopy(existing)])
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=gt_scene,
        targets=derive_repair_targets(input_scene, gt_scene),
    )

    target = measure_repair_target(
        scope,
        candidate_scene,
        compare_detailed(candidate_scene["actors"], gt_scene["actors"]),
    )

    assert target["matched_actor_count"] == 0
    assert target["primary_classification"] == "missing_actor"
    outside = target["target_resolution"]["compatible_outside_resolution"]
    assert len(outside) == 1
    assert outside[0]["compatible_targets"][0][
        "plausible_target_resolution_miss"
    ] is False


def test_same_blueprint_with_different_component_asset_is_incompatible():
    desired = _actor("gt-blueprint", 1_000.0)
    candidate = _actor("candidate-blueprint", 1_000.0)
    blueprint_class = "/Game/Test/BP_Prop.BP_Prop_C"
    desired.update({
        "class": blueprint_class,
        "asset_path": blueprint_class,
        "component_asset_paths": ["/Game/Test/Mesh_A.Mesh_A"],
    })
    candidate.update({
        "class": blueprint_class,
        "asset_path": blueprint_class,
        "component_asset_paths": ["/Game/Test/Mesh_B.Mesh_B"],
    })
    input_scene = _scene([])
    gt_scene = _scene([desired])
    candidate_scene = _scene([candidate])
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=gt_scene,
        targets=derive_repair_targets(input_scene, gt_scene),
    )

    target = measure_repair_target(
        scope,
        candidate_scene,
        compare_detailed(candidate_scene["actors"], gt_scene["actors"]),
    )

    assert target["matched_actor_count"] == 0
    decision = target["compatibility_candidates"][0]["candidates"][0]
    assert decision["compatible"] is False
    assert "component_asset_signature_mismatch" in decision[
        "rejection_reasons"
    ]


def test_assignment_tie_is_deterministic_and_audited():
    desired = _actor("gt-chair", 1_000.0, asset="chair")
    candidates = [
        _actor("candidate-b", 1_000.0, asset="chair"),
        _actor("candidate-a", 1_000.0, asset="chair"),
    ]
    input_scene = _scene([])
    gt_scene = _scene([desired])

    assignments = []
    for values in (candidates, list(reversed(candidates))):
        candidate_scene = _scene(copy.deepcopy(values))
        scope = RepairTargetScope(
            input_scene=input_scene,
            canonical_scene=gt_scene,
            targets=derive_repair_targets(input_scene, gt_scene),
        )
        target = measure_repair_target(
            scope,
            candidate_scene,
            compare_detailed(candidate_scene["actors"], gt_scene["actors"]),
        )
        assignments.append(
            target["final_assignment"][0]["candidate"]["stable_actor_ids"][0]
        )
        assert target["ambiguity"]["ambiguous"] is True

    assert assignments == ["candidate-a", "candidate-a"]


def test_existing_exact_correspondence_keeps_expected_score():
    retained = [
        _actor(f"keep-{index}", index * 200.0, asset=f"keep-{index}")
        for index in range(12)
    ]
    input_actor = _actor("target-chair", 8_000.0, asset="chair")
    desired = _actor("target-chair", 10_000.0, asset="chair")
    candidate = copy.deepcopy(desired)
    input_scene = _scene([*retained, input_actor])
    gt_scene = _scene([*retained, desired])
    candidate_scene = _scene([*retained, candidate])
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=gt_scene,
        targets=derive_repair_targets(input_scene, gt_scene),
    )

    target = measure_repair_target(
        scope,
        candidate_scene,
        compare_detailed(candidate_scene["actors"], gt_scene["actors"]),
    )

    assert target["pair_coverage"] == 1.0
    assert target["actor_correspondence_score"] == 1.0
    assert target["identity_score"] == 1.0
    assert target["position_similarity"] == 1.0
    assert target["scale_similarity"] == 1.0


def test_scene_diff_stays_global_when_comparison_contains_repair_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    detailed = compare_detailed([_actor("chair", 0.0)], [_actor("chair", 0.0)])
    detailed["metrics"]["repair_target"] = {
        "target_derivation": {"target_count": 1},
        "normalization_scale_cm": 40.0,
        "desired_actor_count": 1,
        "candidate_target_actor_count": 1,
        "matched_actor_count": 1,
        "missing_actor_count": 0,
        "extra_actor_count": 0,
        "removal_target_count": 0,
        "remaining_removal_count": 0,
        "pair_coverage": 1.0,
        "actor_correspondence_score": 1.0,
        "identity_match_rate": 1.0,
        "identity_score": 1.0,
        "position_rmse_cm": 360.0,
        "position_mean_cm": 360.0,
        "position_similarity": 0.1,
        "rotation_mean_deg": 90.0,
        "rotation_similarity": 0.5,
        "scale_log_rmse": 0.0,
        "scale_similarity": 1.0,
        "mean_footprint_iou": 0.0,
        "footprint_similarity": 0.0,
        "bounds_size_log_rmse": 0.0,
        "bounds_size_similarity": 1.0,
        "pairwise_layout_error": None,
        "pairwise_layout_pair_count": 0,
        "pairwise_layout_similarity": None,
        "task_property_match_rate": None,
        "material_set_match_rate": None,
        "material_slot_match_rate": None,
        "attribute_similarity": None,
        "global_alignment": {"policy": "reuse_full_scene"},
    }

    def compared(_context: Context) -> dict:
        return {
            **contracts.base("gt_geometry", IDS),
            "status": contracts.MEASURED,
            "score": None,
            "metrics": {**detailed["metrics"], "metric_version": "test"},
            "evidence": {},
            "artifacts": {},
            "probes_used": ("gt_scene_correspondence",),
        }

    monkeypatch.setattr(scene_diff.gt_geometry, "verify", compared)
    task_path = tmp_path / "task.yaml"
    task_path.write_text("id: repair-target-test\n")
    task = SimpleNamespace(
        id="repair-target-test",
        path=task_path,
        kind="scene_generation",
        case_type="prompt_to_scene",
        data={},
        verifiers=[],
    )
    report = scene_diff._structured_report(
        Context(record={}, task=task, ids=IDS, spec={})
    )
    leaves = {
        value["leaf_id"]: value
        for value in report["metrics"]["leaf_results"]
    }

    assert report["evidence"]["primary_scope"] == "whole_scene"
    assert set(leaves) == {
        "actor_correspondence",
        "identity_diff",
        "transform_diff",
        "geometry_diff",
        "attribute_diff",
    }
    assert report["metrics"]["report_only_leaf_ids"] == []
    assert all(
        not leaf_id.startswith("repair_target_") for leaf_id in leaves
    )


def test_repair_target_camera_is_local_and_gt_frozen():
    target = _actor("restored-books", 25_000.0)
    plan = scene_graph_capture.plan_gt_repair_target_aerial([target], count=4)

    assert plan.audit["planner_id"] == (
        scene_graph_capture.GT_REPAIR_TARGET_AERIAL_SEED_PLAN
    )
    assert plan.audit["planning_source"] == "gt_minus_input_target"
    assert plan.anchor_cm[0] == pytest.approx(25_000.0)
    assert plan.audit["scene"]["reach_cm"] == pytest.approx(250.0)
    assert plan.audit["scene"]["reach_cm"] < scene_graph_capture.MIN_REACH_CM
    assert len(plan.audit["camera_specs_sha256"]) == 64
    assert len(plan.views) == 4


def test_repair_caption_remains_a_global_scene_diff_leaf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    task_path = tmp_path / "task.yaml"
    task_path.write_text("id: caption-data-test\n")
    task = SimpleNamespace(
        id="caption-data-test",
        path=task_path,
        kind="scene_repair",
        case_type="image_to_scene",
        data={},
        verifiers=[],
    )
    monkeypatch.setattr(
        scene_diff,
        "_caption_leaf",
        lambda _context: {
            **contracts.base("caption_similarity", IDS),
            "leaf_id": "caption_diff",
            "status": contracts.MEASURED,
            "score": 0.73,
            "metrics": {
                "cosine_similarity": 0.73,
                "cosine_distance": 0.27,
                "model_call_count": 3,
            },
            "evidence": {"model": "test-captioner"},
            "artifacts": {"captions": str(tmp_path / "captions.json")},
            "probes_used": ("independent_captioning",),
        },
    )

    report = scene_diff._visual_semantic_report(
        Context(
            record={},
            task=task,
            ids=IDS,
            spec={
                "visual_semantic_diff": {
                    "caption_diff": True,
                    "paired_visual": False,
                }
            },
        )
    )

    assert "caption_diff" in {
        value["leaf_id"] for value in report["metrics"]["leaf_results"]
    }
    caption = next(
        value
        for value in report["metrics"]["leaf_results"]
        if value["leaf_id"] == "caption_diff"
    )
    assert caption["score"] == pytest.approx(0.73)
    assert report["evidence"]["primary_scope"] == "whole_scene"
    assert "whole_scene_caption_data" not in report["metrics"]


def test_caption_failure_reduces_visual_coverage_without_erasing_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    class BrokenCaptionEvaluator:
        def evaluate_report(self, context, _policy):
            return {
                **contracts.base("caption_similarity", context.ids),
                "status": contracts.ERROR,
                "score": None,
                "failure_reason": "caption response format drift",
                "metrics": {},
                "evidence": {},
                "artifacts": {"failure_response": "/tmp/caption-failure.json"},
                "probes_used": ("vlm_caption",),
            }

    monkeypatch.setattr(
        scene_diff.atomic_registry,
        "get",
        lambda _name: BrokenCaptionEvaluator(),
    )
    monkeypatch.setattr(
        scene_diff,
        "_paired_visual_leaves",
        lambda _context: [
            {
                **contracts.base("paired_render_diff", IDS),
                "leaf_id": "paired_render_diff",
                "status": contracts.MEASURED,
                "score": 0.98,
                "contributes_to_aggregate": False,
                "metrics": {},
                "evidence": {},
                "artifacts": {},
                "probes_used": (),
            },
            {
                **contracts.base("visual_equivalence", IDS),
                "leaf_id": "visual_equivalence",
                "status": contracts.MEASURED,
                "score": 1.0,
                "contributes_to_aggregate": False,
                "metrics": {},
                "evidence": {},
                "artifacts": {},
                "probes_used": (),
            },
            {
                **contracts.base("calibrated_visual_score", IDS),
                "leaf_id": "calibrated_visual_score",
                "status": contracts.MEASURED,
                "score": 0.99,
                "metrics": {},
                "evidence": {},
                "artifacts": {},
                "probes_used": (),
            },
        ],
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text("id: caption-fallback-test\n")
    context = Context(
        record={},
        task=SimpleNamespace(path=task_path),
        ids=IDS,
        spec={
            "visual_semantic_diff": {
                "caption_diff": True,
                "paired_visual": True,
            }
        },
    )

    report = scene_diff._visual_semantic_report(context)
    leaves = {
        value["leaf_id"]: value
        for value in report["metrics"]["leaf_results"]
    }

    assert report["status"] == contracts.MEASURED
    assert report["score"] == pytest.approx(0.99)
    assert report["metrics"]["score_coverage"] == 0.5
    assert report["metrics"]["incomplete_leaf_ids"] == ["caption_diff"]
    assert leaves["caption_diff"]["status"] == "not_evaluated"
    assert leaves["caption_diff"]["metrics"]["upstream_status"] == contracts.ERROR
    assert leaves["caption_diff"]["artifacts"]["failure_response"] == (
        "/tmp/caption-failure.json"
    )
