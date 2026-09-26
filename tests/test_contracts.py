"""Lowering into the repository's shared result schema.

The point of these tests is that "conforms to VerifierReport" is checked by
constructing a real VerifierReport, not asserted in a docstring — the first
version of this module invented field names the schema does not have.
"""

from __future__ import annotations


import pytest

from code4scene.evaluation import contracts
from code4scene.evaluation.contracts import MetricResult

IDS = {"task_bundle_id": "bundle-1", "episode_id": "ep-1"}


def record(**overrides):
    base = {
        "task": "t-village", "agent": "codex", "model": "m-1", "mode": "contracts",
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

def test_metric_sidecar_is_valid_verifier_report_metrics(infra):
    metric = MetricResult(
        id="physics.ground_gap",
        instance_id="physics.ground_gap:physics-grounded",
        metric_version="ground-gap-v1",
        dimension="actor-to-ground surface separation",
        applicability_policy="configured candidate population",
        required_evidence=("ground_gap_cm", "measurement_method"),
        applicable=True,
        status="pass",
        coverage=1.0,
        raw={"unit": "cm", "max": 0.0},
        score=1.0,
        normalization_policy="ground-gap-threshold-fraction-v1",
        normalization_parameters={"maximum_ground_gap_cm": 5.0},
        calibration_status="not_human_validated_diagnostic",
        contributes_to_aggregate=False,
    ).to_json_dict()
    report = infra.VerifierReport.from_json_dict({
        "report_id": "ground_gap",
        **IDS,
        "status": "pass",
        "score": 1.0,
        "metrics": {"metric_results": [metric]},
        "evidence": {},
        "artifacts": {},
        "probes_used": (),
    })

    assert report.metrics["metric_results"][0]["id"] == "physics.ground_gap"
    assert report.metrics["metric_results"][0]["contributes_to_aggregate"] is False


# ── the parts that are judgement, not plumbing ────────────────────────────

def test_ids_are_required_because_the_schema_requires_them():
    """A report the schema cannot identify is a report nothing can file."""
    with pytest.raises(ValueError, match="episode_id"):
        contracts.base("floating", {"task_bundle_id": "b"})


def test_only_measured_quality_reports_produce_an_aggregate_score():
    assert contracts.score_for_aggregate({
        "report_id": "floating", "status": "pass", "score": 0.75,
    }) == 0.75
    assert contracts.score_for_aggregate({
        "report_id": "floating", "status": "error", "score": None,
    }) is None
    assert contracts.score_for_aggregate({
        "report_id": "floating", "status": "measured", "score": 0.75,
    }) == 0.75
    assert contracts.score_for_aggregate({
        "report_id": "candidate_integrity", "status": "valid", "score": None,
    }) is None

    with pytest.raises(ValueError, match="must carry score=None"):
        contracts.score_for_aggregate({
            "report_id": "floating", "status": "error", "score": 0.0,
        })


def test_measured_metric_may_contribute_without_a_pass_fail_gate():
    result = MetricResult(
        id="physics.floating",
        instance_id="physics.floating:episode",
        metric_version="floating-v1",
        dimension="unsupported scene actors",
        applicability_policy="scene-wide closing measurement",
        required_evidence=("measure_scene",),
        applicable=True,
        status="measured",
        coverage=1.0,
        raw={"unit": "ratio", "floating_rate": 0.25},
        score=0.75,
        normalization_policy="one_minus_floating_rate_report_only",
        normalization_parameters={},
        calibration_status="continuous_report_only",
        contributes_to_aggregate=False,
    )

    assert result.to_json_dict()["status"] == "measured"

    contributing = MetricResult(
        **{
            **result.to_json_dict(),
            "required_evidence": result.required_evidence,
            "contributes_to_aggregate": True,
        }
    )
    assert contributing.score == 0.75
