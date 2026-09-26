from __future__ import annotations

from pathlib import Path

from code4scene.evaluation import repair_score
from code4scene.evaluation.saved_caption_recompute import rebuild_result


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


def test_rebuild_result_replaces_only_caption_and_ancestors(tmp_path: Path) -> None:
    caption_error = _report(
        "scene_diff.visual_semantic_diff.caption_diff",
        "caption_diff",
        None,
        status="error",
    )
    caption_error["failure_reason"] = "old malformed tool call"
    paired = _report(
        "scene_diff.visual_semantic_diff.paired_render_diff",
        "paired_render_diff",
        0.9,
        contributes=False,
    )
    equivalence = _report(
        "scene_diff.visual_semantic_diff.visual_equivalence",
        "visual_equivalence",
        0.8,
        contributes=False,
    )
    calibrated = _report(
        "scene_diff.visual_semantic_diff.calibrated_visual_score",
        "calibrated_visual_score",
        0.6,
    )
    old_visual = _report(
        "scene_diff.visual_semantic_diff",
        "visual_semantic_diff",
        None,
        status="error",
    )
    old_visual["metrics"] = {
        "leaf_results": [caption_error, paired, equivalence, calibrated]
    }
    structured = _report(
        "scene_diff.structured_scene_diff",
        "structured_scene_diff",
        0.9,
    )
    old_scene = _report("scene_diff", "scene_diff", None, status="error")
    old_scene["metrics"] = {"leaf_results": [structured, old_visual]}
    local = _report(
        "gt_repair.repair_target_diff",
        "repair_target_diff",
        0.5,
    )
    local["metrics"]["score_weight_policy_id"] = repair_score.LOCAL_POLICY_ID
    locality = _report(
        "gt_repair.locality_audit",
        "locality_audit",
        1.0,
        contributes=False,
    )
    old_root = {
        **IDS,
        "report_id": "gt_repair",
        "status": "error",
        "score": None,
        "metrics": {"leaf_results": [local, old_scene, locality]},
        "evidence": {"original": True},
        "artifacts": {},
        "probes_used": (),
    }
    # The paper score reads the recorded Actor Repair F1; the caption leaf only
    # feeds the diagnostic composite.
    old_root.setdefault("metrics", {})["actor_repair_f1"] = {"status": "measured", "f1": 0.5}
    source = {
        "task_id": "case",
        "batch_status": "complete_with_errors",
        "error_report_count": 1,
        "reports": [
            {
                **IDS,
                "report_id": "candidate_integrity",
                "status": "valid",
                "score": None,
            },
            old_root,
        ],
    }
    new_caption = _report(
        "gt_caption_similarity",
        "caption_diff",
        0.8,
    )

    rebuilt = rebuild_result(
        source,
        new_caption,
        output_case_dir=tmp_path,
        provenance={"scope": "test"},
    )

    root = next(
        value for value in rebuilt["reports"] if value["report_id"] == "gt_repair"
    )
    new_scene = next(
        value
        for value in root["metrics"]["leaf_results"]
        if value["leaf_id"] == "scene_diff"
    )
    new_visual = next(
        value
        for value in new_scene["metrics"]["leaf_results"]
        if value["leaf_id"] == "visual_semantic_diff"
    )
    assert new_visual["score"] == 0.7
    assert new_visual["metrics"]["score_coverage"] == 1.0
    assert new_visual["contributes_to_aggregate"] is True
    assert new_scene["score"] == 0.8
    assert root["metrics"]["diagnostics"]["legacy_gt_repair_composite"] == round(
        0.8 * 0.5 + 0.2 * 0.8, 6)
    assert root["score"] == 0.5  # published Actor Repair F1
    assert rebuilt["overall_score"] == round(0.8 * root["score"], 6)
    assert "score_breakdown" in rebuilt
    assert root["status"] == "measured"
    assert rebuilt["batch_status"] == "complete"
    assert rebuilt["error_report_count"] == 0
    assert source["reports"][1]["status"] == "error"
    assert source["reports"][1]["score"] is None
