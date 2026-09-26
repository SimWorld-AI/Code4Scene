"""One modality-independent primary score for an evaluated scene case.

This is the single "overall" number of a case result (``overall_score`` and
``primary_score.score`` in result.json). It is the paper's case score:

* text-to-scene: ``0.20 Detailed + 0.60 Overview + 0.20 Physics`` (computed by
  :mod:`code4scene.evaluation.score_policy` with the packaged policy);
* image-to-scene: ``0.80 Actor Repair F1 + 0.20 Physics`` (policy
  ``i2s-actor-f1-0.8-physics-0.2-case-macro.v1``).

The formulas themselves live in :mod:`code4scene.protocol`; this module only
reads the verifier reports and records provenance. The legacy weighted
GT-repair composite is carried under ``diagnostics`` and is never scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from typing import Any

from ..protocol import constants as paper
from ..protocol import i2s as paper_i2s
from ..protocol import physics as paper_physics
from . import case_outcome, contracts, missing_score, physics_score, repair_score
from .scalar_score import without_intervals


SCHEMA_VERSION = "code4scene.case_primary_score.v3"
IMAGE_POLICY_ID = paper.I2S_CASE_POLICY
IMAGE_WEIGHTS = {"repair_f1": paper.I2S_REPAIR_WEIGHT,
                 "physical_safety": paper.I2S_PHYSICS_WEIGHT}


def benchmark_track(task: Any) -> str | None:
    """Identify the benchmark stratum from the frozen task, never the candidate."""
    data = task if isinstance(task, Mapping) else {
        "case_type": getattr(task, "case_type", None),
        "scene_environment": getattr(task, "scene_environment", None),
        "verifiers": getattr(task, "verifiers", ()),
    }
    names = {v.get("name") for v in data.get("verifiers", ()) if isinstance(v, Mapping)}
    if data.get("case_type") == "image_to_scene" or "gt_repair" in names:
        environment = data.get("scene_environment")
        return f"image_to_scene_{environment}" if environment in {"indoor", "outdoor"} else None
    if data.get("case_type") in {None, "prompt_to_scene"}:
        return "text_to_scene"
    return None


def physics_component(report: Mapping[str, Any] | None, *, weight: float) -> dict[str, Any]:
    """The Physical Safety component under the paper protocol."""

    measured = paper_physics.score_from_report(report)
    unavailable = [name for name, leaf in measured["leaves"].items() if leaf["safety"] is None]
    coverage = math.fsum(
        paper.PHYSICS_LEAF_WEIGHTS[name] for name, leaf in measured["leaves"].items()
        if leaf["safety"] is not None)
    return {
        "report_id": "physical_safety", "weight": weight,
        "status": "scored_with_missing_zero" if unavailable else "measured",
        "reported_score": (report or {}).get("score"),
        "score": measured["score"], "known_coverage": coverage,
        "missing_evidence": [
            {"path": f"physical_safety.{name}",
             "status": measured["leaves"][name]["status"],
             "reason": "unavailable physics leaf receives zero at its fixed weight"}
            for name in unavailable],
        "leaves": measured["leaves"], "policy_id": paper.PHYSICS_POLICY,
        "scoring_tree": physics_score.project(report),
    }


def _image_score(reports: Sequence[Mapping[str, Any]], *, scene_environment: str | None = None) -> dict[str, Any]:
    repair = _report(reports, "gt_repair")
    metrics = (repair or {}).get("metrics") or {}
    measured = metrics.get(repair_score.PUBLISHED_SCORE)
    if not isinstance(measured, Mapping):
        return unresolved(
            track="image_to_scene", source_kind="score_policy", source_id=IMAGE_POLICY_ID,
            reason_codes=("actor_repair_f1_not_recorded",),
        )
    reasons = []
    if measured.get("status") == contracts.MEASURED:
        f1, f1_missing = float(measured["f1"]), []
    else:
        f1 = 0.0
        f1_missing = [{"path": "gt_repair.actor_repair_f1", "status": measured.get("status"),
                       "reason": str(measured.get("failure_reason") or "not measured")}]
        reasons.append("repair_f1_missing_score_zero")
    physics = physics_component(_report(reports, "physical_safety"),
                                weight=IMAGE_WEIGHTS["physical_safety"])
    if physics["missing_evidence"]:
        reasons.append("physical_safety_missing_score_zero")
    case = paper_i2s.case_score(valid=True, repair_f1=f1, physics_score=physics["score"])
    components = {
        "repair_f1": {
            "report_id": "gt_repair", "weight": IMAGE_WEIGHTS["repair_f1"],
            "status": "scored_with_missing_zero" if f1_missing else "measured",
            "score": f1, "known_coverage": 0.0 if f1_missing else 1.0,
            "missing_evidence": f1_missing, "policy_id": paper.ACTOR_F1_POLICY,
            "counts": {k: measured.get(k) for k in (
                "true_positive", "false_positive", "false_negative",
                "precision", "recall", "desired_count", "candidate_count")},
        },
        "physical_safety": physics,
    }
    coverage = round(math.fsum(
        row["weight"] * row["known_coverage"] for row in components.values()), 6)
    diagnostics = metrics.get("diagnostics") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "resolved",
        "track": "image_to_scene", "source_kind": "score_policy",
        "source_id": IMAGE_POLICY_ID,
        "score": case["score"],
        "known_coverage": coverage, "reason_codes": reasons,
        "policy": {
            "policy_id": IMAGE_POLICY_ID, "weights": dict(IMAGE_WEIGHTS),
            "formula": "round(0.8 * actor_repair_f1 + 0.2 * physical_safety, 6)",
            "repair_metric": paper.ACTOR_F1_POLICY,
            "physics_policy": paper.PHYSICS_POLICY,
            "missing_component_policy": "zero_without_renormalization",
            "missing_evidence_policy": missing_score.POLICY_ID,
        },
        "components": components,
        "diagnostics": {
            repair_score.LEGACY_DIAGNOSTIC_KEY: diagnostics.get(
                repair_score.LEGACY_DIAGNOSTIC_KEY),
            repair_score.LEGACY_DIAGNOSTIC_KEY + "_policy_id": diagnostics.get(
                repair_score.LEGACY_DIAGNOSTIC_KEY + "_policy_id"),
            "role": "diagnostic_not_scored",
            "scene_environment": scene_environment,
        },
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and 0.0 <= result <= 1.0 else None


def _report(
    reports: Sequence[Mapping[str, Any]], report_id: str
) -> Mapping[str, Any] | None:
    return next(
        (
            report
            for report in reports
            if isinstance(report, Mapping) and report.get("report_id") == report_id
        ),
        None,
    )


def unresolved(
    *,
    track: str | None,
    source_kind: str | None,
    source_id: str | None = None,
    reason_codes: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "unresolved",
        "track": track,
        "source_kind": source_kind,
        "source_id": source_id,
        "score": None,
        "known_coverage": 0.0,
        "reason_codes": sorted({str(value) for value in reason_codes}),
    }


def from_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    score_policy_result: Mapping[str, Any] | None = None,
    track: str | None = None,
    scene_environment: str | None = None,
) -> dict[str, Any]:
    """Resolve the public case score from its modality's canonical source."""

    outcome = case_outcome.from_reports(reports)
    repair = _report(reports, "gt_repair")
    track = "text_to_scene" if score_policy_result is not None else (
        "image_to_scene" if repair is not None else track
    )
    source_kind = "score_policy" if score_policy_result is not None else (
        "score_policy" if track == "image_to_scene" else None
    )
    source_id = (
        (score_policy_result.get("policy") or {}).get("policy_id")
        if score_policy_result is not None
        else IMAGE_POLICY_ID
        if track == "image_to_scene"
        else None
    )

    if outcome["classification"] in {case_outcome.MODEL_INVALID, case_outcome.INVALID}:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "model_invalid_zero" if outcome["classification"] == case_outcome.MODEL_INVALID else "invalid_zero",
            "track": track,
            "source_kind": "candidate_integrity_gate",
            "source_id": source_id,
            "score": 0.0,
            "known_coverage": 1.0,
            "reason_codes": list(outcome.get("reason_codes") or ()),
        }
    if outcome["classification"] != case_outcome.VALID:
        return unresolved(
            track=track,
            source_kind=source_kind,
            source_id=source_id,
            reason_codes=(
                outcome.get("reason_codes")
                or ("candidate_outcome_unresolved",)
            ),
        )

    if score_policy_result is not None:
        score_policy_result = without_intervals(score_policy_result)
        score = _number(score_policy_result.get("score"))
        coverage = _number(score_policy_result.get("known_coverage"))
        if (
            score is None
            or coverage is None
        ):
            return unresolved(
                track=track,
                source_kind=source_kind,
                source_id=source_id,
                reason_codes=("text_score_policy_result_invalid",),
            )
        components = deepcopy(score_policy_result.get("components", {}))
        reasons = []
        for name, component in components.items():
            if component.get("known_coverage", 0.0) < 1.0:
                component["status"] = "scored_with_missing_zero"
                reasons.append(f"{name}_missing_score_zero")
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "resolved",
            "track": track,
            "source_kind": source_kind,
            "source_id": source_id,
            "score": score,
            "known_coverage": coverage,
            "reason_codes": reasons,
            "policy": {
                **deepcopy(score_policy_result.get("policy") or {}),
                "missing_evidence_policy": missing_score.POLICY_ID,
            },
            "components": components,
        }

    if track == "image_to_scene":
        return _image_score(reports, scene_environment=scene_environment)

    return unresolved(
        track=None,
        source_kind=None,
        reason_codes=("primary_score_source_missing",),
    )


def result_fields(primary: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable result.json aliases consumed by batch aggregation."""

    primary = without_intervals(primary)
    return {
        "primary_score": without_intervals(primary),
        "overall_score": primary.get("score"),
        "overall_known_coverage": primary.get("known_coverage"),
    }


def apply_to_result(result: Mapping[str, Any], *, scene_environment: str | None = None) -> dict[str, Any]:
    """Rebuild an Image-to-Scene primary score from saved verifier reports."""

    rebuilt = without_intervals(result)
    declared = rebuilt.get("scene_environment")
    track_environment = {
        "image_to_scene_indoor": "indoor", "image_to_scene_outdoor": "outdoor",
    }.get(rebuilt.get("benchmark_track"))
    environments = {value for value in (scene_environment, declared, track_environment) if value}
    if not environments.issubset({"indoor", "outdoor"}) or len(environments) > 1:
        raise ValueError("image_to_scene_environment_conflict")
    scene_environment = next(iter(environments), None)
    if scene_environment is not None:
        rebuilt["scene_environment"] = scene_environment
    reports = rebuilt.get("reports")
    if not isinstance(reports, list):
        raise ValueError("artifact result must contain a reports list")
    rebuilt["case_outcome"] = case_outcome.from_reports(reports)
    for index, report in enumerate(reports):
        if (
            report.get("report_id") == "gt_repair"
            and rebuilt["case_outcome"]["classification"] not in {case_outcome.MODEL_INVALID, case_outcome.INVALID}
        ):
            try:
                recomputed = repair_score.apply(report, scene_environment=scene_environment)
            except ValueError as exc:
                # The diagnostic composite cannot be rebuilt from this legacy
                # report; the published Actor Repair F1 does not depend on it.
                recomputed = deepcopy(dict(report))
                recomputed.setdefault("metrics", {}).setdefault("diagnostics", {})[
                    "legacy_gt_repair_composite_recompute_error"] = str(exc)
            reports[index] = repair_score.publish_actor_f1(recomputed)
    if rebuilt.get("batch_status") in {"complete", "complete_with_errors"}:
        error_count = sum(report.get("status") == contracts.ERROR for report in reports)
        rebuilt["error_report_count"] = error_count
        rebuilt["batch_status"] = "complete_with_errors" if error_count else "complete"
    previous = rebuilt.get("primary_score") or {}
    primary = from_reports(
        reports,
        track="image_to_scene" if previous.get("track") == "image_to_scene" else None,
        scene_environment=scene_environment,
    )
    rebuilt.update(result_fields(primary))
    rebuilt["score_breakdown"] = breakdown(reports, primary)
    return rebuilt


def breakdown(reports: Sequence[Mapping[str, Any]], primary: Mapping[str, Any]) -> dict[str, Any]:
    """Compact score tree; raw reports retain all evidence and per-actor details."""
    def node(report: Mapping[str, Any]) -> dict[str, Any]:
        metrics = report.get("metrics") or {}
        try:
            quality = contracts.score_for_aggregate(report)
        except ValueError:
            quality = None
        result = {
            "report_id": report.get("report_id"), "leaf_id": report.get("leaf_id"),
            "status": report.get("status"), "raw_score": report.get("score"),
            "quality_score": quality,
            "score_direction": (report.get("metadata") or {}).get("score_direction", "higher_is_better"),
            "failure_reason": report.get("failure_reason"),
            "contributes_to_aggregate": report.get("contributes_to_aggregate", True),
            "score_role": report.get("score_role"),
        }
        for key in (
            "score_aggregation", "configured_score_weights", "effective_score_weights",
            "weighted_score_contributions", "prerequisite_gate", "semantic_subscores",
            "dimensions", "normalized_channels", "report_only_leaf_ids",
            "semantic_subscores_policy",
            "optional_score_leaf_ids", "visual_coverage", "score_weight_groups",
            "repair_success",
        ):
            if key in metrics:
                result[key] = without_intervals(metrics[key])
        result["children"] = [node(child) for child in metrics.get("leaf_results") or ()]
        return result
    return {
        "schema_version": "scenebenchmark.score_breakdown.v1",
        "components": deepcopy(primary.get("components", {})),
        "visual_coverage": deepcopy(primary.get("visual_coverage", {})),
        "missing_evidence_policy": missing_score.POLICY_ID,
        "reports": [node(report) for report in reports],
    }


__all__ = [
    "IMAGE_POLICY_ID",
    "SCHEMA_VERSION",
    "apply_to_result",
    "physics_component",
    "benchmark_track",
    "breakdown",
    "from_reports",
    "result_fields",
    "unresolved",
]
