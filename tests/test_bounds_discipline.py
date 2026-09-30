"""Plate discipline: corrections per actor, and when there is nothing to score.

These lived in `test_contracts.py` while the scoring did.
"""

from __future__ import annotations

from types import SimpleNamespace


from code4scene.evaluation import contracts
from code4scene.evaluation.verifiers import bounds_discipline

IDS = {"task_bundle_id": "bundle-1", "episode_id": "ep-1"}


def record(**overrides):
    base = {
        "task": "t-village", "agent": "example-agent", "model": "m-1", "mode": "contracts",
        "exit_reason": "completed", "tool_calls": 90, "wall_minutes": 12.0,
        "infra_error": None, "rounds": [{"round": 1, "parser_stalled": False}],
        "usage": {"input_tokens": 1200, "output_tokens": 340},
        "metrics": {"actors": 400, "structural_collision_rate": 0.05,
                    "floating_rate": 0.01, "oob_rate": 0.0},
        "edge_discipline": {"total_clamped": 0, "total_deleted": 0, "passes": []},
        "umap": "/runs/ep-1/scene.umap",
        "provenance": {"metric_ids": ["measure_v1", "bounds_v1"]},
    }
    return {**base, **overrides}


# ── the schema accepts what we produce ────────────────────────────────────


def bounds_verify(rec):
    return bounds_discipline.verify(SimpleNamespace(record=rec, ids=IDS))



def test_bounds_report_is_a_real_verifier_report(infra):
    corrected = record(edge_discipline={"total_clamped": 8, "total_deleted": 2,
                                        "passes": []})
    report = infra.VerifierReport.from_json_dict(
        bounds_verify(corrected))
    assert report.status is infra.ReportStatus.MEASURED
    assert report.failure_reason is None
    assert report.score == round(1 - 10 / 400, 4)


def test_corrections_without_a_measured_scene_withhold_the_score():
    unmeasured = record(metrics={},
                        edge_discipline={"total_clamped": 3, "total_deleted": 0})
    report = bounds_verify(unmeasured)
    assert report["metrics"]["corrections_per_actor"] is None
    assert report["score"] is None and report["status"] == contracts.ERROR


def test_no_boundary_requirement_is_not_applicable():
    report = bounds_verify(record())
    assert report["status"] == "not_applicable"
    assert report["score"] is None
    assert report["evidence"]["evaluation_bounds"] is None
