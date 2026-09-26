from __future__ import annotations

import math
from pathlib import Path

from code4scene.evaluation.saved_gt_repair_recompute import rebuild_result
from code4scene.evaluation import repair_score


IDS = {
    "task_bundle_id": "case",
    "episode_id": "case--episode",
}


def _report(
    report_id: str,
    leaf_id: str,
    score: float | None,
    *,
    status: str = "measured",
    contributes: bool = True,
) -> dict:
    value = {
        **IDS,
        "report_id": report_id,
        "leaf_id": leaf_id,
        "status": status,
        "score": score,
        "metrics": {},
        "evidence": {},
        "artifacts": {},
        "probes_used": (),
    }
    if not contributes:
        value["contributes_to_aggregate"] = False
    return value


def test_rebuild_preserves_frozen_leaves_and_uses_production_scores(
    tmp_path: Path,
) -> None:
    visual = _report(
        "gt_repair.repair_target_diff.target_visual_diff",
        "target_visual_diff",
        0.9,
    )
    visual["evidence"] = {"frozen_visual": True}
    old_local = _report(
        "gt_repair.repair_target_diff",
        "repair_target_diff",
        0.1,
    )
    old_local["metrics"] = {"leaf_results": [visual]}
    scene = _report("gt_repair.scene_diff", "scene_diff", 0.8)
    scene["metrics"]["score_weight_policy_id"] = repair_score.GLOBAL_POLICY_ID
    locality = _report(
        "gt_repair.locality_audit",
        "locality_audit",
        1.0,
        contributes=False,
    )
    old_root = {
        **IDS,
        "report_id": "gt_repair",
        "status": "measured",
        "score": 0.2,
        "metrics": {"leaf_results": [old_local, scene, locality]},
        "evidence": {"original": True},
        "artifacts": {},
        "probes_used": (),
    }
    source = {
        "task_id": "case",
        "batch_status": "complete",
        "error_report_count": 0,
        "reports": [old_root],
    }
    target = {
        "target_derivation": {"derivation": "test"},
        "normalization_scale_cm": 100.0,
        "pair_coverage": 1.0,
        "global_alignment": {"anchor_count": 1},
        "correspondence_policy": "test_exchangeable",
        "exchangeable_assignment_cost": "test_cost",
        "exchangeable_group_count": 1,
        "exchangeable_reassigned_actor_count": 2,
        "actor_correspondence_score": 1.0,
        "identity_score": 1.0,
        "position_similarity": 0.8,
        "rotation_similarity": 0.6,
        "scale_similarity": 1.0,
        "footprint_similarity": 0.4,
        "bounds_size_similarity": 0.9,
        "pairwise_layout_similarity": 0.7,
        "attribute_similarity": 0.95,
    }

    rebuilt = rebuild_result(
        source,
        target,
        output_case_dir=tmp_path,
        provenance={"scope": "test"},
    )

    root = rebuilt["reports"][0]
    leaves = {
        value["leaf_id"]: value
        for value in root["metrics"]["leaf_results"]
    }
    local = leaves["repair_target_diff"]
    local_leaves = {
        value["leaf_id"]: value
        for value in local["metrics"]["leaf_results"]
    }
    assert local_leaves["target_visual_diff"]["score"] == visual["score"]
    assert local_leaves["target_visual_diff"]["evidence"] == visual["evidence"]
    assert local_leaves["target_visual_diff"]["contributes_to_aggregate"] is True
    assert leaves["scene_diff"] == scene
    assert leaves["locality_audit"] == locality
    assert local_leaves["target_transform_diff"]["score"] == 0.78
    assert local_leaves["target_geometry_diff"]["score"] == 0.61
    assert local["score"] == 0.84375
    # The rebuilt composite is a diagnostic; the report publishes Actor Repair F1.
    assert root["metrics"]["diagnostics"]["legacy_gt_repair_composite"] == round(
        0.8 * 0.84375 + 0.2 * 0.8, 6)
    assert root["evidence"]["saved_evidence_recomputation"] == {
        "scope": "test"
    }
    assert math.isclose(root["metrics"]["repair_target_score"], 0.84375)
    assert source["reports"][0]["score"] == 0.2
