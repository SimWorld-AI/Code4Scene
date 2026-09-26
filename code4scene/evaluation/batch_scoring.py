"""Case-level score resolution and fixed-denominator batch aggregation.

Moved out of the internal artifact-batch launcher so that scoring saved
results needs no launcher, editor, or content packs. The functions only read
already-recorded verifier outputs.

``score_resolution`` decides whether one saved case result carries a resolved
score (or an authoritative invalid verdict that scores a fixed zero).
``aggregate_summary_rows`` averages resolved rows over the fixed selected-case
denominator. The paper's headline aggregation over settings is implemented in
:mod:`code4scene.protocol.aggregate`; this module is the per-batch view used
by the verifier pipeline itself.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from . import case_outcome, missing_score, primary_score

PHYSICS_SCORE_CONTRACT = {
    "contract_id": "physical-safety-two-branch-role-aware-v3",
    "active_physics_leaves": ["floating", "solid_penetration"],
    "physics_profile_id": "absolute-physical-safety-candidate-all-v3",
    "solid_penetration_metric_version": "solid-penetration-v3",
    "adaptive_penetration_tolerance": True,
    "relative_penetration_tolerance_fraction": 0.05,
    "minimum_penetration_tolerance_cm": 5.0,
    "maximum_penetration_tolerance_cm": 50.0,
    "adaptive_tolerance_policy": "actor_shortest_full_aabb_span_clamped_v1",
    "depth_weighting": "binary_plus_log1p_size_normalized_excess_v2",
    "primary_scoring_policy": "eligible_actor_collision_free_rate_v3",
    "physics_role_policy": "deterministic_scene_relative_physics_roles_v1",
    "environment_proxy_contacts": "actor_level_containment_failure",
}


def score_resolution(result: Mapping[str, Any]) -> dict[str, Any]:
    """Classify score availability without guessing from a failed status."""

    outcome = case_outcome.from_result(result)
    classification = outcome["classification"]
    outcome_reasons = list(outcome.get("reason_codes") or ())
    if classification in {case_outcome.MODEL_INVALID, case_outcome.INVALID}:
        # Once an authoritative integrity gate says invalid, errors in quality
        # verifiers cannot turn the fixed zero back into a missing case.
        if result.get("batch_status") not in {"complete", "complete_with_errors"}:
            if outcome.get("failure_owner") == "evaluation_infrastructure":
                return {
                    "schema_version": "scenebenchmark.score_resolution.v1",
                    "classification": "evaluation_infrastructure",
                    "resolved": False,
                    "reason_codes": outcome_reasons
                    or ["candidate_infrastructure_error"],
                }
            return {
                "schema_version": "scenebenchmark.score_resolution.v1",
                "classification": "unknown",
                "resolved": False,
                "reason_codes": [
                    f"unattributed_batch_status:{result.get('batch_status') or 'missing'}"
                ],
            }
        if result.get("authoritative") is not True:
            return {
                "schema_version": "scenebenchmark.score_resolution.v1",
                "classification": "unknown",
                "resolved": False,
                "reason_codes": ["result_not_authoritative"],
            }
        return {
            "schema_version": "scenebenchmark.score_resolution.v1",
            "classification": classification,
            "resolved": True,
            "reason_codes": outcome_reasons or ["model_invalid_unspecified"],
        }
    if classification == case_outcome.INFRASTRUCTURE_ERROR:
        return {
            "schema_version": "scenebenchmark.score_resolution.v1",
            "classification": "evaluation_infrastructure",
            "resolved": False,
            "reason_codes": outcome_reasons or ["candidate_infrastructure_error"],
        }
    if classification == case_outcome.UNKNOWN:
        return {
            "schema_version": "scenebenchmark.score_resolution.v1",
            "classification": "unknown",
            "resolved": False,
            "reason_codes": outcome_reasons or ["candidate_outcome_unknown"],
        }

    reports = result.get("reports")
    reports = reports if isinstance(reports, list) else []
    report_errors = [
        report
        for report in reports
        if isinstance(report, Mapping) and report.get("status") == "error"
    ]
    primary_policy = (result.get("primary_score") or {}).get("policy") or {}
    scored_errors_as_zero = (
        result.get("batch_status") == "complete_with_errors"
        and primary_policy.get("missing_evidence_policy") == missing_score.POLICY_ID
    )
    if result.get("batch_status") != "complete" and not scored_errors_as_zero:
        if report_errors:
            reason_codes = []
            for report in report_errors:
                evidence = report.get("evidence")
                evidence = evidence if isinstance(evidence, Mapping) else {}
                reason_codes.append(
                    str(
                        evidence.get("reason_code")
                        or f"verifier_error:{report.get('report_id') or 'unknown'}"
                    )
                )
            resolution_class = "evaluation_infrastructure"
        else:
            reason_codes = [
                f"unattributed_batch_status:{result.get('batch_status') or 'missing'}"
            ]
            resolution_class = "unknown"
        return {
            "schema_version": "scenebenchmark.score_resolution.v1",
            "classification": resolution_class,
            "resolved": False,
            "reason_codes": sorted(set(reason_codes)),
        }
    if result.get("authoritative") is not True:
        return {
            "schema_version": "scenebenchmark.score_resolution.v1",
            "classification": "unknown",
            "resolved": False,
            "reason_codes": ["result_not_authoritative"],
        }

    primary = result.get("primary_score")
    primary = primary if isinstance(primary, Mapping) else {}
    if (
        primary.get("schema_version")
        != primary_score.SCHEMA_VERSION
        or primary.get("status") != "resolved"
    ):
        return {
            "schema_version": "scenebenchmark.score_resolution.v1",
            "classification": "unknown",
            "resolved": False,
            "reason_codes": list(primary.get("reason_codes") or ())
            or ["strict_primary_score_contract_missing"],
        }
    if primary.get("track") not in {"text_to_scene", "image_to_scene"}:
        return {
            "schema_version": "scenebenchmark.score_resolution.v1",
            "classification": "unknown",
            "resolved": False,
            "reason_codes": ["primary_score_track_invalid"],
        }
    if (
        primary.get("track") == "image_to_scene"
        and primary.get("source_id") != primary_score.IMAGE_POLICY_ID
    ):
        return {
            "schema_version": "scenebenchmark.score_resolution.v1",
            "classification": "unknown", "resolved": False,
            "reason_codes": ["i2s_score_policy_mismatch"],
        }

    score = result.get("overall_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        return {
            "schema_version": "scenebenchmark.score_resolution.v1",
            "classification": "unknown",
            "resolved": False,
            "reason_codes": list(primary.get("reason_codes") or ())
            or ["primary_score_missing_or_non_finite"],
        }
    primary_value = primary.get("score")
    if (
        isinstance(primary_value, bool)
        or not isinstance(primary_value, (int, float))
        or not math.isfinite(float(primary_value))
        or not math.isclose(float(primary_value), float(score), abs_tol=1e-9)
    ):
        return {
            "schema_version": "scenebenchmark.score_resolution.v1",
            "classification": "unknown",
            "resolved": False,
            "reason_codes": ["primary_score_alias_mismatch"],
        }
    if not 0.0 <= float(score) <= 1.0:
        return {
            "schema_version": "scenebenchmark.score_resolution.v1",
            "classification": "unknown",
            "resolved": False,
            "reason_codes": ["primary_score_outside_unit_interval"],
        }
    aliases = (
        ("known_coverage", "overall_known_coverage"),
    )
    parsed: dict[str, float] = {"score": float(score)}
    for primary_key, result_key in aliases:
        primary_alias = primary.get(primary_key)
        result_alias = result.get(result_key)
        if (
            isinstance(primary_alias, bool)
            or not isinstance(primary_alias, (int, float))
            or not math.isfinite(float(primary_alias))
            or not 0.0 <= float(primary_alias) <= 1.0
            or isinstance(result_alias, bool)
            or not isinstance(result_alias, (int, float))
            or not math.isfinite(float(result_alias))
            or not math.isclose(
                float(primary_alias),
                float(result_alias),
                abs_tol=1e-9,
            )
        ):
            return {
                "schema_version": "scenebenchmark.score_resolution.v1",
                "classification": "unknown",
                "resolved": False,
                "reason_codes": [f"primary_score_{primary_key}_alias_invalid"],
            }
        parsed[primary_key] = float(primary_alias)
    return {
        "schema_version": "scenebenchmark.score_resolution.v1",
        "classification": "valid",
        "resolved": True,
        "reason_codes": [],
    }


def aggregate_eligibility(
    result: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Return whether one case has a resolved score for the batch headline."""

    resolution = score_resolution(result)
    if resolution["resolved"]:
        return True, None
    reasons = ", ".join(resolution.get("reason_codes") or ())
    return False, f"{resolution['classification']}: {reasons}"


def effective_case_score(result: Mapping[str, Any]) -> float | None:
    """Return quality for a valid case or the fixed penalty for model invalidity."""

    resolution = score_resolution(result)
    if resolution["resolved"] and resolution["classification"] in {"model_invalid", "invalid"}:
        return 0.0
    if not resolution["resolved"]:
        return None
    score = result.get("overall_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        return None
    return float(score)


def aggregate_summary_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate over the fixed selected-case denominator.

    Resolved invalid cases contribute exactly zero. Infrastructure errors and
    missing whole-case results keep the headline unresolved. No intervals are
    calculated or published.
    """

    selected = [row for row in rows if isinstance(row, Mapping)]
    included = [row for row in selected if row.get("aggregate_included") is True]
    excluded = [row for row in selected if row.get("aggregate_included") is not True]

    def classification(row: Mapping[str, Any]) -> str:
        resolution = row.get("score_resolution")
        if isinstance(resolution, Mapping):
            value = str(resolution.get("classification") or "")
            if value in {
                "valid",
                "model_invalid",
                "invalid",
                "evaluation_infrastructure",
                "unknown",
            }:
                return value
        return "unknown"

    model_invalid = [
        row for row in included if classification(row) == "model_invalid"
    ]
    invalid = [row for row in included if classification(row) == "invalid"]
    valid_scored = [
        row for row in included if classification(row) not in {"model_invalid", "invalid"}
    ]
    valid_scores = [
        float(row.get("effective_score", row.get("overall_score")))
        for row in valid_scored
    ]

    valid_known_coverages = []
    for row in valid_scored:
        primary = row.get("primary_score")
        primary = primary if isinstance(primary, Mapping) else {}
        coverage = primary.get("known_coverage")
        if (
            not isinstance(coverage, bool)
            and isinstance(coverage, (int, float))
            and math.isfinite(float(coverage))
            and 0.0 <= float(coverage) <= 1.0
        ):
            valid_known_coverages.append(float(coverage))
    infrastructure_errors = [
        row
        for row in excluded
        if classification(row) == "evaluation_infrastructure"
    ]
    unknown = [row for row in excluded if row not in infrastructure_errors]

    failure_rows = [*model_invalid, *invalid, *excluded]
    by_reason: dict[str, list[str]] = {}
    for row in failure_rows:
        resolution = row.get("score_resolution")
        resolution = resolution if isinstance(resolution, Mapping) else {}
        reason_codes = list(resolution.get("reason_codes") or ())
        if not reason_codes:
            reason_codes = [f"{classification(row)}_unspecified"]
        for reason_code in set(str(value) for value in reason_codes):
            by_reason.setdefault(reason_code, []).append(str(row.get("case")))

    attempt_counts = [
        int(row.get("generation_attempt_count"))
        for row in selected
        if isinstance(row.get("generation_attempt_count"), int)
        and not isinstance(row.get("generation_attempt_count"), bool)
        and int(row.get("generation_attempt_count")) >= 0
    ]
    retry_attempted = [row for row in selected if row.get("retry_attempted") is True]
    retry_recovered = [
        row
        for row in retry_attempted
        if row.get("retry_recovered") is True
    ]

    selected_count = len(selected)
    unresolved_count = len(excluded)
    exact = (round(math.fsum(valid_scores) / selected_count, 4)
             if selected_count and not unresolved_count else None)
    valid_count = len(valid_scored)
    validity_rate = (valid_count / selected_count
                     if selected_count and not unresolved_count else None)
    conditional_quality = (
        math.fsum(valid_scores) / len(valid_scores) if valid_scores else None
    )
    conditional_known_coverage = (
        math.fsum(valid_known_coverages) / len(valid_known_coverages)
        if len(valid_known_coverages) == len(valid_scored) and valid_scored
        else None
    )

    return {
        "policy_id": "all-selected-primary-score-invalid-zero-scalar-v3",
        "score_field": "effective_score",
        "selected_count": selected_count,
        "included_count": len(included),
        "excluded_count": len(excluded),
        "resolved_count": len(included),
        "unresolved_count": unresolved_count,
        "valid_scored_count": valid_count,
        "model_invalid_count": len(model_invalid),
        "invalid_count": len(model_invalid) + len(invalid),
        "mean_overall_score": exact,
        "unconditional_mean_score": exact,
        "score_coverage": (
            round(len(included) / selected_count, 4) if selected_count else None
        ),
        "validity_rate": (
            round(validity_rate, 4) if validity_rate is not None else None
        ),
        "mean_quality_on_valid_cases": (
            round(conditional_quality, 4)
            if conditional_quality is not None
            else None
        ),
        "mean_known_coverage_on_valid_cases": (
            round(conditional_known_coverage, 4)
            if conditional_known_coverage is not None
            else None
        ),
        "failure_counts": {
            "model_invalid": len(model_invalid),
            "candidate_invalid": len(invalid),
            "evaluation_infrastructure": len(infrastructure_errors),
            "missing_or_unknown": len(unknown),
        },
        "failure_breakdown": {
            reason: {
                "count": len(cases),
                "cases": sorted(cases),
            }
            for reason, cases in sorted(by_reason.items())
        },
        "generation_attempt_total": sum(attempt_counts),
        "generation_attempt_observed_count": len(attempt_counts),
        "retry_attempted_count": len(retry_attempted),
        "retry_recovered_count": len(retry_recovered),
        "retry_not_recovered_count": len(retry_attempted) - len(retry_recovered),
        "retry_recovery_rate": (
            round(len(retry_recovered) / len(retry_attempted), 4)
            if retry_attempted
            else None
        ),
        "included_cases": [str(row.get("case")) for row in included],
        "excluded_cases": [
            {
                "case": row.get("case"),
                "status": row.get("status"),
                "reason": row.get("aggregate_exclusion_reason"),
            }
            for row in excluded
        ],
    }


