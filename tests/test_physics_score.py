from copy import deepcopy

import pytest

from code4scene.evaluation import physics_score, score_policy, primary_score, case_outcome, repair_score
from test_score_policy import POLICY, _reports


def report(penetration_status="not_evaluated", penetration_score=None):
    return {"report_id": "physical_safety", "status": "measured", "score": None,
            "metrics": {"score_aggregation": None, "leaf_results": [
                {"leaf_id": "floating", "status": "measured", "score": .1778,
                 "metadata": {"score_direction": "lower_is_better"}},
                {"leaf_id": "solid_penetration", "status": penetration_status, "score": penetration_score}]}}


def test_partial_physics_retains_measured_leaf_without_mutating_evidence():
    raw = report()
    before = deepcopy(raw)
    result = physics_score.project(raw)
    assert result["score"] == .4111
    assert result["known_coverage"] == .5
    assert result["effective_weights"] == {"floating": .5, "solid_penetration": .5}
    assert raw == before


def test_absent_required_leaf_keeps_its_weight():
    raw = report()
    raw["metrics"]["leaf_results"].pop()
    assert physics_score.project(raw)["score"] == .4111


def test_true_na_is_zero_without_renormalization():
    raw = report("not_applicable")
    before = deepcopy(raw)
    result = physics_score.project(raw)
    assert result["score"] == .4111
    assert result["effective_weights"] == {"floating": .5, "solid_penetration": .5}
    assert result["raw_not_applicable_leaf_ids"] == ["solid_penetration"]
    assert result["children"][1]["source_status"] == "not_evaluated"
    assert raw == before


def test_both_missing_are_zero():
    assert physics_score.project(None)["score"] == 0


def test_complete_scalar_legacy_report_is_preserved():
    assert physics_score.project({"status": "measured", "score": .9234})["score"] == .9234


def test_measured_zero_is_not_missing_and_penetration_not_inverted():
    result = physics_score.project(report("measured", 0.0))
    assert result["score"] == .4111
    assert result["known_coverage"] == 1.0
    assert physics_score.project(report("measured", .9))["score"] == .8611


def test_t2s_uses_leafwise_physics():
    reports = _reports()
    reports[1] = report()
    result = score_policy.aggregate_reports(reports, score_policy.load(POLICY))
    assert result["components"]["physics"]["score"] == .4111
    assert result["score"] == round(.2 * .54 + .6 * .7 + .2 * .4111, 4)
    assert result["known_coverage"] == pytest.approx(.9)


def test_i2s_uses_same_leafwise_physics_on_legacy_report():
    reports = [{"report_id": "candidate_integrity", "status": "valid",
                "evidence": {"case_outcome": case_outcome.make(case_outcome.VALID, source="test")}},
               {"report_id": "gt_repair", "status": "measured", "score": 1.,
                "metrics": {"score_weight_policy_id": repair_score.POLICY_ID,
                            repair_score.PUBLISHED_SCORE: {"status": "measured", "f1": 1.0}}},
               report()]
    result = primary_score.from_reports(reports)
    assert result["components"]["physical_safety"]["score"] == .4111
    assert result["score"] == .88222
