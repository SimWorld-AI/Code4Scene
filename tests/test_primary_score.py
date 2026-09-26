from __future__ import annotations

import json
import pytest

from code4scene.evaluation import case_outcome, primary_score, repair_score


def _repair(score, *, status="measured"):
    """A published gt_repair report: its score is the Actor Repair F1."""
    measured = ({"status": "measured", "f1": score, "true_positive": 1}
                if status == "measured" else {"status": status, "failure_reason": "synthetic"})
    return {"report_id": "gt_repair", "status": status,
            "score": score if status == "measured" else None,
            "metrics": {"score_weight_policy_id": repair_score.POLICY_ID,
                        "published_score": repair_score.PUBLISHED_SCORE,
                        repair_score.PUBLISHED_SCORE: measured}}


def _integrity(classification: str, reason: str | None = None) -> dict[str, object]:
    outcome = case_outcome.make(
        classification,
        reason_codes=((reason,) if reason else ()),
        source="test",
    )
    return {
        "report_id": "candidate_integrity",
        "status": ("valid" if classification == case_outcome.VALID else
                   "error" if classification == case_outcome.INFRASTRUCTURE_ERROR else "invalid"),
        "evidence": {"case_outcome": outcome},
    }


def test_image_to_scene_combines_repair_and_physics() -> None:
    result = primary_score.from_reports(
        [
            _integrity(case_outcome.VALID),
            _repair(0.625),
            {"report_id": "physical_safety", "status": "measured", "score": 0.9},
        ]
    )

    assert result["track"] == "image_to_scene"
    assert result["source_id"] == primary_score.IMAGE_POLICY_ID
    assert result["score"] == 0.68
    assert "lower_bound" not in result and "upper_bound" not in result
    assert result["components"]["physical_safety"]["weight"] == 0.2


def test_image_to_scene_model_invalid_overrides_stale_repair_quality() -> None:
    result = primary_score.from_reports(
        [
            _integrity(case_outcome.MODEL_INVALID, "empty_scene"),
            {"report_id": "gt_repair", "status": "measured", "score": 0.99},
        ]
    )

    assert result["status"] == "model_invalid_zero"
    assert result["score"] == 0.0
    assert result["reason_codes"] == ["empty_scene"]


def test_image_to_scene_infrastructure_error_remains_unresolved() -> None:
    result = primary_score.from_reports(
        [
            _integrity(case_outcome.INFRASTRUCTURE_ERROR, "dependency_failure"),
            {"report_id": "gt_repair", "status": "error", "score": None},
        ]
    )

    assert result["status"] == "unresolved"
    assert result["score"] is None
    assert "lower_bound" not in result and "upper_bound" not in result
    assert result["reason_codes"] == ["dependency_failure"]


def test_image_to_scene_offline_recompute_matches_online_fields() -> None:
    source = {
        "batch_status": "complete",
        "reports": [
            _integrity(case_outcome.VALID),
            _repair(0.42),
            {"report_id": "physical_safety", "status": "measured", "score": 0.42},
        ],
    }
    before = json.dumps(source, sort_keys=True)

    rebuilt = primary_score.apply_to_result(source)

    assert json.dumps(source, sort_keys=True) == before
    assert rebuilt["overall_score"] == 0.42
    assert rebuilt["primary_score"]["score"] == 0.42
    assert rebuilt["case_outcome"]["classification"] == "valid"


@pytest.mark.parametrize("physics", [None, {"status": "not_evaluated", "score": None},
                                    {"status": "not_applicable", "score": None},
                                    {"status": "measured", "score": float("nan")}])
def test_missing_physics_counts_as_zero_without_renormalizing(physics) -> None:
    reports = [_integrity(case_outcome.VALID),
               _repair(0.5)]
    if physics:
        reports.append({"report_id": "physical_safety", **physics})
    result = primary_score.from_reports(reports)
    assert result["score"] == 0.4
    assert result["status"] == "resolved"
    assert "lower_bound" not in result and "upper_bound" not in result
    assert result["known_coverage"] == 0.8


def test_unmeasurable_repair_f1_is_zero_with_reason() -> None:
    reports = [_integrity(case_outcome.VALID),
               {"report_id": "physical_safety", "status": "measured", "score": 0.9},
               _repair(None, status="error")]
    result = primary_score.from_reports(reports)
    assert result["score"] == 0.18
    assert result["reason_codes"] == ["repair_f1_missing_score_zero"]
    assert result["components"]["repair_f1"]["missing_evidence"]


def test_the_legacy_composite_is_a_diagnostic_and_never_scored() -> None:
    report = {"report_id": "gt_repair", "status": "measured", "score": 0.5,
              "metrics": {repair_score.PUBLISHED_SCORE: {"status": "measured", "f1": 0.25},
                          "leaf_results": [
             {"leaf_id": "repair_target_diff", "status": "measured", "score": 0.5,
              "metrics": {"direct_score_leaf_id": "target_recovery", "leaf_results": [
                  {"leaf_id": "target_recovery", "status": "measured", "score": 0.5},
                  {"leaf_id": "target_visual_diff", "status": "not_applicable", "score": None},
              ]}},
             {"leaf_id": "scene_diff", "status": "measured", "score": 0.5,
              "metrics": {"score_weight_policy_id": repair_score.GLOBAL_POLICY_ID}},
             {"leaf_id": "locality_audit", "status": "error", "score": None,
              "contributes_to_aggregate": False},
         ]}}
    rebuilt = primary_score.apply_to_result({"batch_status": "complete", "reports": [
        _integrity(case_outcome.VALID),
        {"report_id": "physical_safety", "status": "measured", "score": 1.0}, report]})
    published = next(r for r in rebuilt["reports"] if r["report_id"] == "gt_repair")
    assert published["score"] == 0.25
    assert published["metrics"]["diagnostics"]["legacy_gt_repair_composite"] == 0.5
    assert rebuilt["overall_score"] == round(0.8 * 0.25 + 0.2 * 1.0, 6)
    assert rebuilt["primary_score"]["diagnostics"]["legacy_gt_repair_composite"] == 0.5
    assert "gt_repair" not in rebuilt["primary_score"]["components"]


def test_a_result_without_actor_repair_f1_is_unresolved_not_relabelled() -> None:
    result = primary_score.from_reports([
        _integrity(case_outcome.VALID),
        {"report_id": "gt_repair", "status": "measured", "score": 0.9},
        {"report_id": "physical_safety", "status": "measured", "score": 1.0},
    ])
    assert result["score"] is None
    assert result["reason_codes"] == ["actor_repair_f1_not_recorded"]


def test_breakdown_exposes_raw_and_normalized_physics_leaf() -> None:
    result = primary_score.breakdown([{
        "report_id": "physical_safety", "status": "measured", "score": 0.8,
        "metrics": {"leaf_results": [{
            "report_id": "physical_safety.floating", "leaf_id": "floating",
            "status": "measured", "score": 0.2,
            "metadata": {"score_direction": "lower_is_better"},
        }]},
    }], {})
    leaf = result["reports"][0]["children"][0]
    assert leaf["raw_score"] == 0.2
    assert leaf["quality_score"] == 0.8
