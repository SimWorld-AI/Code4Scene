from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from code4scene.evaluation import case_outcome, score_policy
from code4scene.resources import config_file


POLICY = config_file("score-policies", "text-to-scene-human-aligned.yaml")


def _reports() -> list[dict[str, object]]:
    families = {
        "identity_environment": {
            "applicable": True,
            "lower_bound": 0.8,
            "upper_bound": 0.8,
            "known_coverage": 1.0,
            "active_requirement_count": 2,
            "clause_count": 2,
        },
        "content_quantity": {
            "applicable": True,
            "lower_bound": 0.6,
            "upper_bound": 0.6,
            "known_coverage": 1.0,
            "active_requirement_count": 3,
            "clause_count": 3,
        },
        "attributes_materials": {
            "applicable": True,
            "lower_bound": 0.4,
            "upper_bound": 0.4,
            "known_coverage": 1.0,
            "active_requirement_count": 1,
            "clause_count": 1,
        },
        "spatial_composition": {
            "applicable": True,
            "lower_bound": 0.2,
            "upper_bound": 0.2,
            "known_coverage": 1.0,
            "active_requirement_count": 1,
            "clause_count": 1,
        },
    }
    return [
        {
            "report_id": "semantic_requirements",
            "status": "measured",
            "score": 0.5,
            "metrics": {
                "score_policy": "old-policy",
                "semantic_subscores": families,
            },
        },
        {"report_id": "physical_safety", "status": "measured", "score": 0.9},
        {
            "report_id": "overview_prompt_alignment",
            "status": "measured",
            "score": 0.7,
            "contributes_to_aggregate": False,
            "score_role": "report_only",
            "report_only_reason": "not active without an explicit policy",
        },
    ]


def test_checked_in_policy_has_human_aligned_weights() -> None:
    policy = score_policy.load(POLICY)

    assert policy.semantic_family_weights == {
        "identity_environment": 0.25,
        "content_quantity": 0.40,
        "attributes_materials": 0.15,
        "spatial_composition": 0.20,
    }
    assert policy.total_weights == {
        "semantic": 0.20,
        "physics": 0.20,
        "holistic_overview": 0.60,
    }


def test_automatic_policy_matches_explicit_policy_and_keeps_family_scores() -> None:
    from types import SimpleNamespace

    task = SimpleNamespace(verifiers=[
        {"name": "semantic_requirements"}, {"name": "overview_prompt_alignment"}
    ])
    automatic = score_policy.default_for_task(task)
    explicit = score_policy.load(POLICY)
    assert automatic is not None
    assert automatic.total_weights == explicit.total_weights
    assert automatic.semantic_family_weights == explicit.semantic_family_weights
    outcome = case_outcome.make(case_outcome.VALID, source="test")
    result = score_policy.apply_to_result({"reports": [
        {"report_id": "candidate_integrity", "status": "valid",
         "evidence": {"case_outcome": outcome}}, *_reports(),
    ]}, automatic)
    assert result["overall_score"] == 0.708
    assert result["score_breakdown"]["components"]["semantic"]["families"]["content_quantity"]["score"] == 0.6
    assert score_policy.default_for_task(SimpleNamespace(verifiers=[{"name": "gt_repair"}])) is None


def test_policy_computes_family_semantic_and_final_score() -> None:
    policy = score_policy.load(POLICY)
    reports = _reports()

    result = score_policy.aggregate_reports(reports, policy, activate_reports=True)

    assert result["components"]["semantic"]["score"] == 0.54
    assert result["score"] == 0.708
    assert "lower_bound" not in result and "upper_bound" not in result
    assert result["known_coverage"] == 1.0
    assert reports[0]["score"] == 0.54
    assert reports[0]["metrics"]["score_before_score_policy"] == 0.5
    assert reports[0]["metrics"]["semantic_subscores"]["content_quantity"][
        "family_weight"
    ] == 0.4
    assert reports[2]["contributes_to_aggregate"] is True
    assert reports[2]["score_role"] == "weighted_overall_component"
    assert reports[2]["aggregate_weight"] == 0.6
    assert "report_only_reason" not in reports[2]


def test_unknown_component_keeps_its_weight_and_receives_zero() -> None:
    policy = score_policy.load(POLICY)
    reports = _reports()
    reports[2] = {
        "report_id": "overview_prompt_alignment",
        "status": "error",
        "score": None,
    }

    result = score_policy.aggregate_reports(reports, policy)

    assert result["score"] == 0.288
    assert "lower_bound" not in result and "upper_bound" not in result
    assert result["known_coverage"] == 0.4
    assert result["components"]["holistic_overview"]["status"] == "unknown"


def test_model_invalid_candidate_hard_gates_the_case_to_zero() -> None:
    policy = score_policy.load(POLICY)
    reports = _reports()
    reports.append(
        {
            "report_id": "candidate_integrity",
            "status": "invalid",
            "score": None,
            "evidence": {
                "case_outcome": case_outcome.make(
                    case_outcome.MODEL_INVALID,
                    reason_codes=("empty_scene",),
                    source="test",
                )
            },
        }
    )

    result = score_policy.aggregate_reports(reports, policy)

    assert result["score"] == 0.0
    assert "lower_bound" not in result and "upper_bound" not in result
    assert result["known_coverage"] == 1.0
    assert result["hard_gate"]["applied"] is True
    assert result["quality_before_gate"]["score"] == 0.708


def test_infrastructure_error_does_not_trigger_the_model_penalty() -> None:
    policy = score_policy.load(POLICY)
    reports = _reports()
    reports.append(
        {
            "report_id": "candidate_integrity",
            "status": "error",
            "score": None,
            "evidence": {
                "case_outcome": case_outcome.make(
                    case_outcome.INFRASTRUCTURE_ERROR,
                    reason_codes=("content_release_drift",),
                    source="test",
                )
            },
        }
    )

    result = score_policy.aggregate_reports(reports, policy)

    assert result["hard_gate"]["applied"] is False
    assert result["case_outcome"]["score_disposition"] == "withheld"
    assert result["score"] == 0.708


def test_legacy_composite_invalid_with_clear_model_evidence_is_zero() -> None:
    policy = score_policy.load(POLICY)
    reports = _reports()
    reports.append(
        {
            "report_id": "candidate_integrity",
            "status": "invalid",
            "score": None,
            "evidence": {},
            "metrics": {
                "leaf_results": [
                    {
                        "leaf_id": "candidate_snapshot_integrity",
                        "status": "invalid",
                        "metrics": {
                            "actor_count": 0,
                            "minimum_actor_count": 1,
                            "execution_failure_count": 0,
                            "schema_error_count": 0,
                            "provenance_error_count": 0,
                        },
                        "evidence": {},
                    },
                    {"leaf_id": "content_and_dependency_parity", "status": "valid"},
                    {"leaf_id": "asset_library_manifest_parity", "status": "valid"},
                ]
            },
        }
    )

    result = score_policy.aggregate_reports(reports, policy)

    assert result["score"] == 0.0
    assert result["case_outcome"]["classification"] == "model_invalid"


def test_non_applicable_family_is_renormalized_not_scored_as_failure() -> None:
    policy = score_policy.load(POLICY)
    reports = _reports()
    attributes = reports[0]["metrics"]["semantic_subscores"][
        "attributes_materials"
    ]
    attributes.update(
        {
            "applicable": False,
            "lower_bound": None,
            "upper_bound": None,
            "known_coverage": None,
        }
    )

    result = score_policy.aggregate_reports(reports, policy)
    semantic = result["components"]["semantic"]

    assert semantic["score"] == 0.5647
    assert semantic["families"]["attributes_materials"]["effective_weight"] == 0.0


def test_offline_application_keeps_source_document_unchanged() -> None:
    policy = score_policy.load(POLICY)
    reports = _reports()
    reports.append(
        {
            "report_id": "candidate_integrity",
            "status": "valid",
            "score": None,
            "evidence": {
                "case_outcome": case_outcome.make(
                    case_outcome.VALID,
                    source="test",
                )
            },
        }
    )
    source = {"batch_status": "complete", "reports": reports}
    encoded_before = json.dumps(source, sort_keys=True)

    rebuilt = score_policy.apply_to_result(source, policy)

    assert json.dumps(source, sort_keys=True) == encoded_before
    assert rebuilt["overall_score"] == 0.708
    assert rebuilt["primary_score"]["track"] == "text_to_scene"
    assert rebuilt["primary_score"]["score"] == 0.708
    assert rebuilt["score_policy_result"]["policy"]["sha256"] == policy.sha256


def test_policy_rejects_weights_that_do_not_sum_to_one(tmp_path: Path) -> None:
    document = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    document["total_weights"]["semantic"] = 0.3
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(score_policy.ScorePolicyError, match="sum to 1.0"):
        score_policy.load(invalid)
