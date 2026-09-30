"""The GT-repair interface keeps local and global scopes explicit."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from code4scene.evaluation import contracts, render, ue_evidence
from code4scene.evaluation.context import Context
from code4scene.evaluation.render_evidence import EvidenceRequest
from code4scene.evaluation.verifiers import gt_repair
from code4scene.tasks import task as task_mod


IDS = {"task_bundle_id": "bundle", "episode_id": "episode"}

def _composite(report: dict) -> float | None:
    """The legacy weighted composite, kept on the report as a diagnostic.

    The gt_repair report publishes Actor Repair F1 as its score; these tests
    pin the composite's internal arithmetic, which no benchmark score uses.
    """
    return report["metrics"]["diagnostics"]["legacy_gt_repair_composite"]


def _composite_status(report: dict) -> str:
    return report["metrics"]["diagnostics"]["legacy_gt_repair_composite_status"]



def _task(
    tmp_path: Path,
    *,
    prompt: str = "optional repair description",
    scene_environment: str | None = None,
):
    path = tmp_path / "repair.yaml"
    path.write_text("id: repair\n")
    return SimpleNamespace(
        id="repair",
        path=path,
        kind="scene_repair",
        case_type="image_to_scene",
        scene_environment=scene_environment,
        prompt=prompt,
        data={},
        verifiers=[],
        declared_reference_views=[tmp_path / "gt.png"],
    )


def _comparison() -> dict:
    return {
        **contracts.base("gt_geometry", IDS),
        "status": contracts.MEASURED,
        "score": None,
        "metrics": {
            "repair_target": {
                "target_derivation": {
                    "target_count": 2,
                    "operation_counts": {"add": 0, "remove": 0, "repair": 2},
                },
                "normalization_scale_cm": 100.0,
                "desired_actor_count": 2,
                "candidate_target_actor_count": 2,
                "matched_actor_count": 2,
                "missing_actor_count": 0,
                "extra_actor_count": 0,
                "removal_target_count": 0,
                "remaining_removal_count": 0,
                "actor_correspondence_score": 1.0,
                "identity_match_rate": 1.0,
                "identity_score": 1.0,
                "pair_coverage": 1.0,
                "position_rmse_cm": 100.0,
                "position_mean_cm": 100.0,
                "position_similarity": 0.5,
                "rotation_mean_deg": 0.0,
                "rotation_similarity": 1.0,
                "scale_log_rmse": 0.0,
                "scale_similarity": 1.0,
                "mean_footprint_iou": 0.5,
                "footprint_similarity": 0.5,
                "bounds_size_log_rmse": 0.0,
                "bounds_size_similarity": 1.0,
                "pairwise_layout_error": None,
                "pairwise_layout_pair_count": 0,
                "pairwise_layout_similarity": None,
                "task_property_match_rate": 0.5,
                "material_set_match_rate": 0.5,
                "material_slot_match_rate": 0.5,
                "attribute_similarity": 0.5,
                "global_alignment": {"policy": "shared"},
                "matched_actor_rows": [],
            },
            "metric_version": "test",
        },
        "evidence": {},
        "artifacts": {},
        "probes_used": ("gt_scene_correspondence",),
    }


def _pure_removal_comparison() -> dict:
    comparison = _comparison()
    comparison["metrics"]["repair_target"] = {
        "target_derivation": {
            "target_count": 1,
            "operation_counts": {"add": 0, "remove": 1, "repair": 0},
        },
        "normalization_scale_cm": 100.0,
        "desired_actor_count": 0,
        "candidate_target_actor_count": 0,
        "matched_actor_count": 0,
        "missing_actor_count": 0,
        "extra_actor_count": 0,
        "removal_target_count": 1,
        "remaining_removal_count": 0,
        "actor_correspondence_score": 1.0,
        "identity_match_rate": None,
        "identity_score": None,
        "pair_coverage": None,
        "global_alignment": {"policy": "shared"},
        "matched_actor_rows": [],
    }
    return comparison


def _missing_target_comparison() -> dict:
    comparison = _comparison()
    comparison["metrics"]["repair_target"] = {
        "target_derivation": {
            "target_count": 1,
            "operation_counts": {"add": 1, "remove": 0, "repair": 0},
        },
        "normalization_scale_cm": 100.0,
        "desired_actor_count": 1,
        "candidate_target_actor_count": 0,
        "matched_actor_count": 0,
        "missing_actor_count": 1,
        "extra_actor_count": 0,
        "removal_target_count": 0,
        "remaining_removal_count": 0,
        "actor_correspondence_score": 0.0,
        "identity_match_rate": 0.0,
        "identity_score": 0.0,
        "pair_coverage": 0.0,
        "global_alignment": {"policy": "shared"},
        "primary_classification": "missing_actor",
        "failure_classification": ["missing_or_wrong_asset"],
        "unmatched_gt_actors": [{"stable_actor_ids": ["missing"]}],
        "unmatched_candidate_actors": [],
        "matched_actor_rows": [],
    }
    return comparison


class _Policy:
    def __init__(self, source_snapshot=None):
        self.source_snapshot = source_snapshot or {
            "actors": [{"stable_actor_id": "input"}]
        }

    def leaf_spec(self, _task, _leaf_id, base_spec):
        return {**base_spec, "input_scene": self.source_snapshot}


def _measured_report(report_id: str, score: float) -> dict:
    return {
        **contracts.base(report_id, IDS),
        "status": contracts.MEASURED,
        "score": score,
        "metrics": {
            "leaf_results": [],
            "score_weight_policy_id": (
                gt_repair.repair_score.GLOBAL_POLICY_ID if report_id == "scene_diff"
                else gt_repair.LOCAL_REPAIR_WEIGHT_POLICY_ID
            ),
        },
        "evidence": {},
        "artifacts": {},
        "probes_used": (),
    }


def test_gt_repair_reuses_one_correspondence_for_local_and_global(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    comparison = _comparison()
    calls = {"gt": 0, "global": 0}

    def compare(_context):
        calls["gt"] += 1
        return comparison

    def global_report(_context, observed):
        calls["global"] += 1
        assert observed is comparison
        return _measured_report("scene_diff", 0.81)

    locality = _measured_report("gt_repair.locality_audit", 0.25)
    locality.update(
        {
            "leaf_id": "locality_audit",
            "contributes_to_aggregate": False,
            "score_role": "report_only",
        }
    )
    monkeypatch.setattr(gt_repair, "load_evaluation_policy", lambda _task: _Policy())
    monkeypatch.setattr(gt_repair.gt_geometry, "verify", compare)
    monkeypatch.setattr(gt_repair.scene_diff, "_report_from_comparison", global_report)
    monkeypatch.setattr(
        gt_repair, "_locality_audit", lambda _context, _comparison: locality
    )

    report = gt_repair.verify(
        Context(
            record={},
            task=_task(tmp_path),
            ids=IDS,
            spec={"name": "gt_repair", "ground_truth": "/Game/GT"},
            reference_images=[tmp_path / "gt.png"],
        )
    )

    assert calls == {"gt": 1, "global": 1}
    by_id = {value["leaf_id"]: value for value in report["metrics"]["leaf_results"]}
    local = by_id["repair_target_diff"]
    local_leaves = {value["leaf_id"]: value for value in local["metrics"]["leaf_results"]}
    assert set(local_leaves) == {
        "repair_success",
        "target_recovery",
        "target_identity",
        "target_transform_diff",
        "target_geometry_diff",
        "target_attribute_diff",
        "target_visual_diff",
    }
    assert local_leaves["target_transform_diff"]["score"] == pytest.approx(
        0.5 * 0.5 + 1.0 * 0.3 + 1.0 * 0.2
    )
    assert local_leaves["target_geometry_diff"]["score"] == pytest.approx(
        (0.5 * 0.5 + 1.0 * 0.3) / (0.5 + 0.3)
    )
    assert local_leaves["target_visual_diff"]["status"] == "not_applicable"
    assert local["score"] == pytest.approx(0.751953)
    assert local["metrics"]["score_weight_policy_id"] == (
        gt_repair.LOCAL_REPAIR_WEIGHT_POLICY_ID
    )
    assert "target_visual_diff" not in local["metrics"]["effective_score_weights"]
    assert sum(local["metrics"]["effective_score_weights"].values()) == (
        pytest.approx(1.0, abs=1e-5)
    )
    assert "locality_audit" in report["metrics"]["report_only_leaf_ids"]
    assert _composite(report) == pytest.approx(0.8 * local["score"] + 0.2 * 0.81, abs=1e-6)
    optional = report["evidence"]["optional_inputs"]
    assert optional["prompt"]["available"] is True
    assert optional["prompt"]["used_for_score"] is False
    assert optional["gt_or_reference_images"]["runtime_count"] == 1
    assert optional["gt_or_reference_images"]["used_for_score"] is False


def test_gt_repair_scores_pure_removal_without_target_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    comparison = _pure_removal_comparison()
    locality = _measured_report("gt_repair.locality_audit", 1.0)
    locality.update(
        {
            "leaf_id": "locality_audit",
            "contributes_to_aggregate": False,
            "score_role": "report_only",
        }
    )
    monkeypatch.setattr(gt_repair, "load_evaluation_policy", lambda _task: _Policy())
    monkeypatch.setattr(gt_repair.gt_geometry, "verify", lambda _context: comparison)
    monkeypatch.setattr(
        gt_repair.scene_diff,
        "_report_from_comparison",
        lambda _context, observed: _measured_report("scene_diff", 0.968),
    )
    monkeypatch.setattr(gt_repair, "_locality_audit", lambda *_args: locality)

    report = gt_repair.verify(
        Context(
            record={},
            task=_task(tmp_path),
            ids=IDS,
            spec={"name": "gt_repair", "ground_truth": "/Game/GT"},
            reference_images=[tmp_path / "gt.png"],
        )
    )

    by_id = {value["leaf_id"]: value for value in report["metrics"]["leaf_results"]}
    local = by_id["repair_target_diff"]
    local_leaves = {value["leaf_id"]: value for value in local["metrics"]["leaf_results"]}
    assert local_leaves["target_recovery"]["score"] == pytest.approx(1.0)
    assert local_leaves["target_identity"]["status"] == "not_applicable"
    assert local["metrics"]["score_aggregation"] == "direct_measured_leaf_score"
    assert local["metrics"]["direct_score_leaf_id"] == "target_recovery"
    assert local["metrics"]["prerequisite_leaf_ids"] == []
    assert local["metrics"]["prerequisite_values"] == {}
    assert local["metrics"]["prerequisite_gate"] == pytest.approx(1.0)
    assert local["metrics"]["prerequisite_selection"] == (
        "pure_removal_uses_recovery_directly"
    )
    assert local["score"] == pytest.approx(1.0)
    assert _composite(report) == pytest.approx(0.8 * 1.0 + 0.2 * 0.968, abs=1e-6)


def test_gt_repair_partial_pure_removal_scores_recovery_once(tmp_path: Path):
    target = _pure_removal_comparison()["metrics"]["repair_target"]
    target["target_derivation"] = {
        "target_count": 2,
        "operation_counts": {"add": 0, "remove": 2, "repair": 0},
    }
    target["removal_target_count"] = 2
    target["remaining_removal_count"] = 1
    target["actor_correspondence_score"] = 0.5

    report = gt_repair.report_from_repair_target_measurement(
        Context(
            record={},
            task=_task(tmp_path),
            ids=IDS,
            spec={"name": "gt_repair", "ground_truth": "/Game/GT"},
        ),
        target,
    )

    leaves = {value["leaf_id"]: value for value in report["metrics"]["leaf_results"]}
    assert leaves["target_recovery"]["score"] == pytest.approx(0.5)
    assert report["metrics"]["weighted_quality_mix"] == pytest.approx(0.5)
    assert report["metrics"]["prerequisite_gate"] == pytest.approx(1.0)
    assert report["score"] == pytest.approx(0.5)


def test_gt_repair_missing_target_short_circuits_paired_visual_to_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    comparison = _missing_target_comparison()
    locality = _measured_report("gt_repair.locality_audit", 1.0)
    locality.update(
        {
            "leaf_id": "locality_audit",
            "contributes_to_aggregate": False,
            "score_role": "report_only",
        }
    )
    visual_calls = 0

    def visual_verify(_context):
        nonlocal visual_calls
        visual_calls += 1
        raise AssertionError("missing target must not require paired renders")

    monkeypatch.setattr(gt_repair, "load_evaluation_policy", lambda _task: _Policy())
    monkeypatch.setattr(gt_repair.gt_geometry, "verify", lambda _context: comparison)
    monkeypatch.setattr(
        gt_repair.scene_diff,
        "_report_from_comparison",
        lambda _context, observed: _measured_report("scene_diff", 0.8),
    )
    monkeypatch.setattr(gt_repair, "_locality_audit", lambda *_args: locality)
    monkeypatch.setattr(gt_repair.visual_as_judge, "verify", visual_verify)

    report = gt_repair.verify(
        Context(
            record={},
            task=_task(tmp_path),
            ids=IDS,
            spec={
                "name": "gt_repair",
                "ground_truth": "/Game/GT",
                "visual_semantic_diff": {
                    "caption_diff": False,
                    "paired_visual": True,
                },
            },
        )
    )

    assert visual_calls == 0
    assert _composite_status(report) == contracts.MEASURED
    assert _composite(report) == pytest.approx(0.16)
    by_id = {value["leaf_id"]: value for value in report["metrics"]["leaf_results"]}
    local = by_id["repair_target_diff"]
    local_by_id = {value["leaf_id"]: value for value in local["metrics"]["leaf_results"]}
    assert local["status"] == contracts.MEASURED
    assert local["score"] == pytest.approx(0.0)
    assert local["metrics"]["prerequisite_gate"] == pytest.approx(0.0)
    assert local_by_id["target_recovery"]["score"] == pytest.approx(0.0)
    assert local_by_id["target_identity"]["score"] == pytest.approx(0.0)
    assert local_by_id["target_transform_diff"]["status"] == "not_applicable"
    assert local_by_id["target_geometry_diff"]["status"] == "not_applicable"
    assert local_by_id["target_attribute_diff"]["status"] == "not_applicable"
    visual = local_by_id["target_visual_diff"]
    assert visual["status"] == "not_applicable"
    assert visual["evidence"]["visual_evidence_required_for_score"] is False
    assert visual["evidence"]["target_visual_short_circuit_policy_id"] == (
        gt_repair.MISSING_TARGET_SHORT_CIRCUIT_POLICY_ID
    )
    assert local["evidence"]["missing_target_short_circuit"] is True


def test_gt_repair_missing_target_ignores_saved_visual_error(tmp_path: Path):
    target = _missing_target_comparison()["metrics"]["repair_target"]
    stale_visual_error = {
        **contracts.base("gt_repair.repair_target_diff.target_visual_diff", IDS),
        "leaf_id": "target_visual_diff",
        "status": contracts.ERROR,
        "score": None,
        "failure_reason": "gt_paired renders were not assembled",
        "metrics": {},
        "evidence": {},
        "artifacts": {},
        "probes_used": (),
    }

    report = gt_repair.report_from_repair_target_measurement(
        Context(
            record={},
            task=_task(tmp_path),
            ids=IDS,
            spec={"name": "gt_repair", "ground_truth": "/Game/GT"},
        ),
        target,
        target_visual=stale_visual_error,
    )

    visual = next(
        value
        for value in report["metrics"]["leaf_results"]
        if value["leaf_id"] == "target_visual_diff"
    )
    assert report["status"] == contracts.MEASURED
    assert report["score"] == pytest.approx(0.0)
    assert visual["status"] == "not_applicable"
    assert "gt_paired renders were not assembled" not in str(report)


def test_gt_repair_pure_removal_short_circuits_paired_visual(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    visual_calls = 0

    def visual_verify(_context):
        nonlocal visual_calls
        visual_calls += 1
        raise AssertionError("pure removal has no desired Actor to render")

    monkeypatch.setattr(gt_repair.visual_as_judge, "verify", visual_verify)
    target = _pure_removal_comparison()["metrics"]["repair_target"]
    report = gt_repair.report_from_repair_target_measurement(
        Context(
            record={},
            task=_task(tmp_path),
            ids=IDS,
            spec={
                "name": "gt_repair",
                "ground_truth": "/Game/GT",
                "visual_semantic_diff": {
                    "paired_visual": True,
                },
            },
        ),
        target,
    )

    visual = next(
        value
        for value in report["metrics"]["leaf_results"]
        if value["leaf_id"] == "target_visual_diff"
    )
    assert visual_calls == 0
    assert report["status"] == contracts.MEASURED
    assert report["score"] == pytest.approx(1.0)
    assert visual["status"] == "not_applicable"
    assert visual["evidence"]["target_visual_short_circuit_kind"] == (
        "pure_removal_without_desired_target"
    )
    assert visual["evidence"]["target_visual_short_circuit_policy_id"] == (
        gt_repair.PURE_REMOVAL_VISUAL_SHORT_CIRCUIT_POLICY_ID
    )


def test_gt_repair_full_metric_vector_includes_structured_visual_and_global(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    comparison = _comparison()
    visual = _measured_report("visual_as_judge", 0.6)
    visual.update(
        {
            "metrics": {
                "deterministic": {
                    "aggregate": {
                        "rgb_perceptual_similarity": 0.7,
                        "base_color_similarity": 0.5,
                        "foreground_silhouette_iou": 0.8,
                        "depth_edge_similarity": 0.4,
                    }
                },
                "vlm": {
                    "dimensions": {
                        "composition_layout": 0.75,
                        "silhouette_geometry": 0.5,
                        "material_fidelity": 0.25,
                        "landmark_content": 0.5,
                        "lighting_fidelity": 1.0,
                    }
                },
                "calibrated": {"overall_visual_similarity": 0.6},
            },
            "evidence": {
                "comparison_scope": "repair_target",
                "artifact_namespace": "repair_target",
            },
            "artifacts": {"paired_view_0_rgb": "/tmp/repair-target.png"},
        }
    )
    locality = _measured_report("gt_repair.locality_audit", 1.0)
    locality.update(
        {
            "leaf_id": "locality_audit",
            "contributes_to_aggregate": False,
            "score_role": "report_only",
        }
    )

    monkeypatch.setattr(gt_repair, "load_evaluation_policy", lambda _task: _Policy())
    monkeypatch.setattr(gt_repair.gt_geometry, "verify", lambda _context: comparison)
    monkeypatch.setattr(
        gt_repair.scene_diff,
        "_report_from_comparison",
        lambda _context, observed: _measured_report("scene_diff", 0.8),
    )
    monkeypatch.setattr(gt_repair, "_locality_audit", lambda *_args: locality)

    def visual_verify(context):
        assert context.spec["comparison_scope"] == "repair_target"
        return visual

    monkeypatch.setattr(gt_repair.visual_as_judge, "verify", visual_verify)
    report = gt_repair.verify(
        Context(
            record={},
            task=_task(tmp_path),
            ids=IDS,
            spec={
                "name": "gt_repair",
                "ground_truth": "/Game/GT",
                "visual_semantic_diff": {
                    "caption_diff": False,
                    "paired_visual": True,
                },
            },
        )
    )

    by_id = {value["leaf_id"]: value for value in report["metrics"]["leaf_results"]}
    local = by_id["repair_target_diff"]
    local_by_id = {value["leaf_id"]: value for value in local["metrics"]["leaf_results"]}
    assert {
        leaf_id: value["score"] for leaf_id, value in local_by_id.items()
        if leaf_id != "repair_success"
    } == pytest.approx(
        {
            "target_recovery": 1.0,
            "target_identity": 1.0,
            "target_transform_diff": 0.75,
            "target_geometry_diff": 0.6875,
            "target_attribute_diff": 0.5,
            "target_visual_diff": 0.6,
        }
    )
    assert local_by_id["target_visual_diff"]["metrics"] == visual["metrics"]
    assert local_by_id["repair_success"]["contributes_to_aggregate"] is False
    assert local_by_id["repair_success"]["score"] is None
    assert local_by_id["target_visual_diff"]["artifacts"] == visual["artifacts"]
    assert local["score"] == pytest.approx(0.721563)
    assert local["metrics"]["score_aggregation"] == (
        "prerequisite_gate_times_weighted_mean_of_available_normalized_leaf_scores"
    )
    assert local["metrics"]["prerequisite_gate"] == pytest.approx(1.0)
    assert local["metrics"]["prerequisite_values"] == pytest.approx(
        {"target_recovery": 1.0, "target_identity": 1.0}
    )
    assert local["metrics"]["prerequisite_selection"] == (
        "desired_targets_require_recovery_and_identity"
    )
    assert local["metrics"]["configured_score_weights"] == pytest.approx(
        gt_repair.LOCAL_REPAIR_WEIGHTS
    )
    assert local["metrics"]["effective_score_weights"] == pytest.approx(
        gt_repair.LOCAL_REPAIR_WEIGHTS
    )
    assert local_by_id["target_transform_diff"]["metrics"][
        "configured_channel_weights"
    ] == pytest.approx(gt_repair.TARGET_TRANSFORM_WEIGHTS)
    assert local_by_id["target_geometry_diff"]["metrics"][
        "effective_channel_weights"
    ] == pytest.approx(
        {
            "footprint_similarity": 0.625,
            "bounds_size_similarity": 0.375,
        }
    )
    assert report["metrics"]["aggregate_score_vector"] == pytest.approx(
        {
            "repair_target_diff": local["score"],
            "scene_diff": 0.8,
        }
    )
    assert _composite(report) == pytest.approx(0.8 * local["score"] + 0.2 * 0.8, abs=1e-6)
    assert report["metrics"]["configured_score_weights"] == pytest.approx(
        {"repair_target_diff": 0.8, "scene_diff": 0.2}
    )
    assert report["evidence"]["final_score_formula"] == (
        "0.8 * repair_target_diff + 0.2 * scene_diff"
    )


def test_gt_repair_local_prerequisites_gate_the_weighted_quality_mix():
    leaves = []
    for leaf_id, score in {
        "target_recovery": 0.0,
        "target_identity": 1.0,
        "target_transform_diff": 1.0,
        "target_geometry_diff": 1.0,
        "target_attribute_diff": 1.0,
        "target_visual_diff": 1.0,
    }.items():
        leaf = _measured_report(f"gt_repair.repair_target_diff.{leaf_id}", score)
        leaf["leaf_id"] = leaf_id
        leaves.append(leaf)
    report = {
        "status": contracts.MEASURED,
        "score": 1.0,
        "metrics": {"leaf_results": leaves},
        "evidence": {},
    }

    weighted = gt_repair._apply_local_repair_policy(report)

    assert weighted["metrics"]["weighted_quality_mix"] == pytest.approx(0.9)
    assert weighted["metrics"]["prerequisite_gate"] == pytest.approx(0.0)
    assert weighted["score"] == pytest.approx(0.0)


def test_gt_repair_requests_distinct_global_and_local_visual_protocols(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    scopes = []

    def request(_task, spec):
        scope = spec["comparison_scope"]
        scopes.append(scope)
        return (
            EvidenceRequest(
                protocol=f"test.{scope}",
                kind="paired_render",
                channels=("rgb",),
                view_count=1,
                width=64,
                height=64,
                timeout_s=1.0,
                camera_protocol=(
                    render.GT_PAIRED_ENVIRONMENT_SCENE_GRAPH_PROTOCOL
                ),
                scene_alignment=None,
            ),
        )

    monkeypatch.setattr(gt_repair.visual_as_judge, "evidence_requests", request)
    requests = gt_repair.evidence_requests(
        _task(tmp_path),
        {
            "name": "gt_repair",
            "ground_truth": "/Game/GT",
            "visual_semantic_diff": {
                "caption_diff": False,
                "paired_visual": True,
            },
        },
    )

    assert scopes == ["whole_scene", "repair_target"]
    assert {value.protocol for value in requests} == {
        "test.whole_scene",
        "test.repair_target",
    }


def test_low_quality_visual_leaf_is_removed_and_structural_weights_renormalize():
    leaves = []
    for leaf_id in gt_repair.LOCAL_REPAIR_WEIGHTS:
        visual = leaf_id == "target_visual_diff"
        leaves.append(
            {
                "leaf_id": leaf_id,
                "status": "not_evaluated" if visual else contracts.MEASURED,
                "score": None if visual else 0.8,
            }
        )
    report = {
        "status": contracts.MEASURED,
        "score": 0.0,
        "metrics": {"leaf_results": leaves},
        "evidence": {},
    }

    result = gt_repair._apply_weighted_leaf_policy(
        report,
        gt_repair.LOCAL_REPAIR_WEIGHTS,
        policy_id=gt_repair.LOCAL_REPAIR_WEIGHT_POLICY_ID,
        prerequisite_leaf_ids=gt_repair.LOCAL_REPAIR_PREREQUISITES,
    )

    assert result["score"] == pytest.approx(0.64)
    assert "target_visual_diff" not in result["metrics"][
        "effective_score_weights"
    ]
    assert sum(result["metrics"]["effective_score_weights"].values()) == (
        pytest.approx(1.0)
    )


def test_indoor_gt_repair_requests_only_local_visual_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    scopes = []

    def request(_task, spec):
        scope = spec["comparison_scope"]
        scopes.append(scope)
        return (
            EvidenceRequest(
                protocol=f"test.{scope}",
                kind="paired_render",
                channels=("rgb",),
                view_count=4,
                width=64,
                height=64,
                timeout_s=1.0,
                camera_protocol=(
                    render.GT_PAIRED_ENVIRONMENT_SCENE_GRAPH_PROTOCOL
                ),
                scene_alignment=None,
            ),
        )

    monkeypatch.setattr(gt_repair.visual_as_judge, "evidence_requests", request)
    monkeypatch.setattr(
        gt_repair.caption_similarity,
        "evidence_requests",
        lambda *_args, **_kwargs: pytest.fail(
            "Indoor repair must not request caption overview evidence"
        ),
    )
    requests = gt_repair.evidence_requests(
        _task(tmp_path, scene_environment="indoor"),
        {
            "name": "gt_repair",
            "ground_truth": "/Game/GT",
            "visual_semantic_diff": {
                "caption_diff": True,
                "paired_visual": True,
            },
        },
    )

    assert scopes == ["repair_target"]
    assert [value.protocol for value in requests] == ["test.repair_target"]


def test_indoor_gt_repair_keeps_global_structure_without_overview_visual(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    comparison = _comparison()
    local = _measured_report("gt_repair.repair_target_diff", 0.75)
    global_report = _measured_report("scene_diff", 0.8)
    locality = _measured_report("gt_repair.locality_audit", 1.0)
    locality.update(
        {
            "leaf_id": "locality_audit",
            "contributes_to_aggregate": False,
            "score_role": "report_only",
        }
    )

    monkeypatch.setattr(gt_repair, "load_evaluation_policy", lambda _task: _Policy())
    monkeypatch.setattr(gt_repair.gt_geometry, "verify", lambda _context: comparison)

    def local_report(context, observed):
        assert observed is comparison
        assert context.spec["visual_semantic_diff"] == {
            "caption_diff": True,
            "paired_visual": True,
        }
        return local

    def structured_global_report(context, observed):
        assert observed is comparison
        assert context.spec["visual_semantic_diff"] == {
            "caption_diff": False,
            "paired_visual": False,
        }
        assert context.spec["visual_evidence_routing"]["policy_id"] == (
            gt_repair.INDOOR_LOCAL_VISUAL_ROUTING_POLICY_ID
        )
        return global_report

    monkeypatch.setattr(gt_repair, "_repair_target_report", local_report)
    monkeypatch.setattr(
        gt_repair.scene_diff,
        "_report_from_comparison",
        structured_global_report,
    )
    monkeypatch.setattr(gt_repair, "_locality_audit", lambda *_args: locality)

    report = gt_repair.verify(
        Context(
            record={},
            task=_task(tmp_path, scene_environment="indoor"),
            ids=IDS,
            spec={
                "name": "gt_repair",
                "ground_truth": "/Game/GT",
                "visual_semantic_diff": {
                    "caption_diff": True,
                    "paired_visual": True,
                },
            },
        )
    )

    assert _composite_status(report) == contracts.MEASURED
    assert _composite(report) == pytest.approx(0.8 * 0.75 + 0.2 * 0.8, abs=1e-6)
    assert report["evidence"]["visual_evidence_routing"] == {
        "policy_id": gt_repair.INDOOR_LOCAL_VISUAL_ROUTING_POLICY_ID,
        "repair_target_visual": "score_when_available",
        "whole_scene_visual": "omitted",
        "whole_scene_structured_diff": "required",
    }
    by_id = {value["leaf_id"]: value for value in report["metrics"]["leaf_results"]}
    assert by_id["scene_diff"]["evidence"][
        "visual_evidence_routing_policy_id"
    ] == gt_repair.INDOOR_LOCAL_VISUAL_ROUTING_POLICY_ID


@pytest.mark.parametrize("environment", ["indoor", "outdoor"])
@pytest.mark.parametrize("enabled", [False, True])
def test_visual_selection_controls_execution_and_optional_score_contribution(
    monkeypatch, tmp_path, environment, enabled,
):
    from code4scene.evaluation import primary_score, verifiers

    task = _task(tmp_path, scene_environment=environment)
    spec = {
        "name": "gt_repair", "ground_truth": "/Game/GT",
        "visual_semantic_diff": {"caption_diff": enabled, "paired_visual": enabled},
    }
    task.verifiers = [spec]
    requests = verifiers.render_evidence_requests(task)
    assert len(requests) == ((1 if environment == "indoor" else 3) if enabled else 0)

    model_calls = []

    def visual(context):
        model_calls.append(context.spec["comparison_scope"])
        report = _measured_report("visual_as_judge", 0.6)
        report["metrics"]["calibrated_visual_score"] = {"score": 0.6}
        return report

    def caption(context):
        model_calls.append("caption")
        return _measured_report("caption_similarity", 0.6)

    monkeypatch.setattr(gt_repair, "load_evaluation_policy", lambda _: _Policy())
    monkeypatch.setattr(gt_repair.gt_geometry, "verify", lambda _: _comparison())
    monkeypatch.setattr(gt_repair.visual_as_judge, "verify", visual)
    monkeypatch.setattr(
        gt_repair.scene_diff.atomic_registry, "get",
        lambda _: SimpleNamespace(evaluate_report=lambda context, spec: caption(context)),
    )
    monkeypatch.setattr(
        gt_repair.scene_diff, "_structured_report_from_comparison",
        lambda *_: _measured_report("structured_scene_diff", 0.8),
    )
    monkeypatch.setattr(gt_repair, "_locality_audit", lambda *_: {
        **_measured_report("locality_audit", 1.0),
        "leaf_id": "locality_audit", "contributes_to_aggregate": False,
    })
    report = gt_repair.verify(Context(record={}, task=task, ids=IDS, spec=spec))
    assert model_calls == (
        (["repair_target"] if environment == "indoor" else ["repair_target", "caption", "whole_scene"])
        if enabled else []
    )
    assert _composite_status(report) == "measured"
    heads = {child["leaf_id"]: child for child in report["metrics"]["leaf_results"]}
    expected_global = 0.7 if enabled and environment == "outdoor" else 0.8
    expected_local = 0.721563 if enabled else 0.751953
    assert heads["scene_diff"]["score"] == expected_global
    assert heads["repair_target_diff"]["score"] == expected_local
    if not enabled:
        local_visual = heads["repair_target_diff"]["metrics"]["leaf_results"][-1]
        assert local_visual["score"] is None
        assert "disabled" in local_visual["failure_reason"]
        assert local_visual["contributes_to_aggregate"] is False
    primary = primary_score.from_reports([
        {"report_id": "candidate_integrity", "status": "valid"}, report,
        _measured_report("physical_safety", 0.5),
    ])
    # The synthetic context exports no scene snapshots, so Actor Repair F1 is
    # unmeasurable and scores zero at its fixed weight; the composite these
    # leaves feed is reported only as a diagnostic.
    assert primary["reason_codes"] == ["repair_f1_missing_score_zero"]
    assert primary["known_coverage"] == 0.2
    assert primary["score"] == round(0.2 * 0.5, 6)
    assert primary["diagnostics"]["legacy_gt_repair_composite"] == round(
        0.8 * expected_local + 0.2 * expected_global, 6)


def test_gt_repair_schema_accepts_prompt_modality_with_frozen_3d_inputs(
    tmp_path: Path,
):
    document = {
        "id": "repair",
        "kind": "scene_repair",
        "case_type": "prompt_to_scene",
        "inputs": {"prompt": "repair", "init_map": "/Game/Input"},
        "source": {"pack": "Dungeon"},
        "assets": {"packs": ["Input"]},
        "verifiers": [{"name": "gt_repair", "ground_truth": "/Game/GT"}],
    }
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(document))

    loaded = task_mod.load(path)

    assert loaded.case_type == "prompt_to_scene"
    assert loaded.verifiers == [{"name": "gt_repair", "ground_truth": "/Game/GT"}]


def _actor(actor_id: str, x_cm: float, *, asset: str) -> dict:
    return {
        "stable_actor_id": actor_id,
        "label": actor_id,
        "actor_path": f"/Game/Test.Test:PersistentLevel.{actor_id}",
        "class": "/Script/Engine.StaticMeshActor",
        "asset_path": f"/Game/Test/{asset}.{asset}",
        "actor_tags": [],
        "transform": {
            "location_cm": [x_cm, 0.0, 50.0],
            "rotation_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "bounds": {
            "origin_cm": [x_cm, 0.0, 50.0],
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


def test_gt_repair_reads_code_owned_input_candidate_and_gt_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    retained = _actor("retained", 0.0, asset="wall")
    target = _actor("restored-chair", 200.0, asset="chair")
    input_scene = _scene([retained])
    candidate_scene = _scene([retained, target])
    task_document = {
        "id": "document-backed-repair",
        "kind": "scene_repair",
        "case_type": "image_to_scene",
        "inputs": {
            "prompt": "optional and not scored",
            "init_map": "/Game/Input",
        },
        "assets": {"packs": ["Input", "GT"]},
        "source": {"pack": "Dungeon"},
        "verifiers": [
            {
                "name": "gt_repair",
                "ground_truth": "/Game/GT",
                "candidate_scene": candidate_scene,
            }
        ],
    }
    task_path = tmp_path / "document-backed-repair.yaml"
    task_path.write_text(yaml.safe_dump(task_document, sort_keys=False))
    task_path.with_suffix(".label.json").write_text(
        json.dumps(
            {
                "gt_id": "document-backed-repair-gt",
                "canonical_map": "/Game/GT",
                "canonical_actors": [retained, target],
            }
        )
    )
    task = task_mod.load(task_path)
    monkeypatch.setattr(
        gt_repair,
        "load_evaluation_policy",
        lambda _task: _Policy(input_scene),
    )
    report = gt_repair.verify(
        Context(
            record={},
            task=task,
            ids=IDS,
            spec=task.verifiers[0],
            scoring=SimpleNamespace(measure_path=None),
            artifacts_dir=tmp_path / "artifacts",
            out_dir=tmp_path / "output",
        )
    )

    assert report["status"] == contracts.MEASURED
    assert report["score"] == pytest.approx(1.0)
    by_id = {value["leaf_id"]: value for value in report["metrics"]["leaf_results"]}
    local = by_id["repair_target_diff"]
    local_leaves = {value["leaf_id"]: value for value in local["metrics"]["leaf_results"]}
    assert local_leaves["target_recovery"]["score"] == pytest.approx(1.0)
    assert local_leaves["target_identity"]["score"] == pytest.approx(1.0)
    assert by_id["scene_diff"]["status"] == contracts.MEASURED
    assert report["evidence"]["required_inputs"] == {
        "input": "evaluation_policy.source_snapshot",
        "candidate": "independent_scoring_editor_export",
        "gt": "answer_key.canonical_actors_and_canonical_map",
    }


def test_gt_repair_runtime_triplet_needs_no_frozen_scene_graphs(
    tmp_path: Path,
    monkeypatch,
):
    retained = _actor("retained", 0.0, asset="wall")
    target = _actor("restored-chair", 200.0, asset="chair")
    input_scene = _scene([retained])
    candidate_scene = _scene([retained, target])
    canonical_scene = _scene([retained, target])
    task_document = {
        "id": "runtime-repair",
        "kind": "scene_repair",
        "case_type": "image_to_scene",
        "inputs": {"prompt": "repair", "init_map": "/Game/Input"},
        "assets": {"packs": ["Input", "GT"]},
        "source": {"pack": "Dungeon"},
        "verifiers": [
            {
                "name": "gt_repair",
                "ground_truth": "/Game/GT",
                "candidate_scene": candidate_scene,
            }
        ],
    }
    task_path = tmp_path / "runtime-repair.yaml"
    task_path.write_text(yaml.safe_dump(task_document, sort_keys=False))
    task = task_mod.load(task_path)

    def capture_runtime(_context, package, *, role, candidate_scene=None):
        assert package == "/Game/Input"
        assert role == "input"
        assert candidate_scene is not None
        return input_scene, tmp_path / "episode" / "input.scene.json"

    def capture_gt(_context, *, candidate_scene=None):
        assert candidate_scene is not None
        return canonical_scene, tmp_path / "episode" / "ground_truth.scene.json"

    monkeypatch.setattr(
        ue_evidence,
        "capture_runtime_map_scene",
        capture_runtime,
    )
    monkeypatch.setattr(
        ue_evidence,
        "capture_task_ground_truth",
        capture_gt,
    )

    report = gt_repair.verify(
        Context(
            record={},
            task=task,
            ids=IDS,
            spec=task.verifiers[0],
            scoring=SimpleNamespace(measure_path=None),
            artifacts_dir=tmp_path / "artifacts",
            out_dir=tmp_path / "output",
        )
    )

    assert report["status"] == contracts.MEASURED
    assert report["score"] == pytest.approx(1.0)
    assert report["evidence"]["required_inputs"] == {
        "input": "independent_runtime_export_of_task.inputs.init_map",
        "candidate": "independent_scoring_editor_export",
        "gt": "independent_runtime_export_of_task.ground_truth_map",
    }


def test_runtime_map_capture_restores_candidate_after_success(
    tmp_path: Path,
    monkeypatch,
):
    loaded = []
    current = {"map": "/Game/Candidate"}

    def load(_context, package, _timeout):
        loaded.append(package)
        current["map"] = package

    monkeypatch.setattr(ue_evidence, "_load_runtime_package", load)
    monkeypatch.setattr(
        ue_evidence.core_inventory,
        "read_scene_snapshot",
        lambda _bridge, *, timeout: {
            "map_path": current["map"],
            "actors": [],
            "export_metadata": {"status": "success"},
        },
    )
    context = SimpleNamespace(
        spec={},
        scoring=SimpleNamespace(bridge=object(), measure_path=None),
        artifacts_dir=tmp_path,
        out_dir=tmp_path,
        ids={"task_bundle_id": "bundle", "episode_id": "episode"},
        cache={},
        record={},
    )

    scene, output = ue_evidence.capture_runtime_map_scene(
        context,
        "/Game/Input",
        role="input",
        candidate_scene={"map_path": "/Game/Candidate"},
    )

    assert scene["map_path"] == "/Game/Input"
    assert output.exists()
    assert loaded == ["/Game/Input", "/Game/Candidate"]
    assert current["map"] == "/Game/Candidate"


def test_runtime_map_capture_restores_candidate_after_export_failure(
    tmp_path: Path,
    monkeypatch,
):
    loaded = []

    def load(_context, package, _timeout):
        loaded.append(package)

    def fail_export(_bridge, *, timeout):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(ue_evidence, "_load_runtime_package", load)
    monkeypatch.setattr(
        ue_evidence.core_inventory,
        "read_scene_snapshot",
        fail_export,
    )
    context = SimpleNamespace(
        spec={},
        scoring=SimpleNamespace(bridge=object(), measure_path=None),
        artifacts_dir=tmp_path,
        out_dir=tmp_path,
        ids={"task_bundle_id": "bundle", "episode_id": "episode"},
        cache={},
        record={},
    )

    with pytest.raises(RuntimeError, match="snapshot failed"):
        ue_evidence.capture_runtime_map_scene(
            context,
            "/Game/GT",
            role="ground_truth",
            candidate_scene={"map_path": "/Game/Candidate"},
        )

    assert loaded == ["/Game/GT", "/Game/Candidate"]


def _raise(exc):
    def fail(*_args, **_kwargs):
        raise exc
    return fail


@pytest.mark.parametrize("where", ["scene_diff", "aggregation"])
def test_a_failed_diagnostic_keeps_the_actor_repair_f1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, where):
    monkeypatch.setattr(gt_repair, "load_evaluation_policy", lambda _task: _Policy())
    monkeypatch.setattr(gt_repair.gt_geometry, "verify", lambda _context: _comparison())
    monkeypatch.setattr(gt_repair, "_locality_audit", lambda _context, _comparison: _measured_report(
        "gt_repair.locality_audit", 0.5))
    monkeypatch.setattr(gt_repair, "_actor_repair_f1", lambda _context: {
        "policy": "actor-repair-f1", "status": contracts.MEASURED, "f1": 0.75})
    if where == "scene_diff":
        monkeypatch.setattr(gt_repair.scene_diff, "_report_from_comparison", _raise(KeyError("missing slot")))
    else:
        monkeypatch.setattr(gt_repair.scene_diff, "_report_from_comparison",
                            lambda _context, _comparison: _measured_report("scene_diff", 0.8))
        monkeypatch.setattr(gt_repair.repair_score, "apply", _raise(ValueError("bad weights")))

    report = gt_repair.verify(Context(record={}, task=_task(tmp_path), ids=IDS,
                                      spec={"name": "gt_repair", "ground_truth": "/Game/GT"}))

    assert report["status"] == contracts.MEASURED and report["score"] == 0.75
    assert report["metrics"]["actor_repair_f1"]["f1"] == 0.75
    (diagnostics,) = report["metrics"]["leaf_results"]
    assert diagnostics["leaf_id"] == "diagnostics" and diagnostics["status"] == contracts.ERROR
    assert ("KeyError" if where == "scene_diff" else "ValueError") in diagnostics["failure_reason"]
