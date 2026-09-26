"""The one canonical Candidate-to-GT public interface.

One gt_geometry comparison owns logical-object correspondence. Every dimension
below is a projection of that same comparison and correspondence audit; no
structured leaf runs an independent matcher. Caption and paired-visual
evidence remain separate protocols even though their reports share this public
tree.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .. import atomic_registry, contracts
from ..composite import composite_report, run_leaf
from ..context import Context
from ..render_evidence import (
    CAPTION_ENVIRONMENT_RENDER_PROTOCOL,
    EvidenceRequest,
    GT_VISUAL_ENVIRONMENT_RENDER_PROTOCOL,
)
from . import caption_similarity, gt_geometry, visual_as_judge


CLASS = "gt"


def _leaf(
    context: Context,
    leaf_id: str,
    metrics: Mapping[str, Any],
    *,
    status: str = contracts.MEASURED,
    score: float | None = None,
    reason: str | None = None,
    evidence: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    result = {
        **contracts.base(
            f"scene_diff.structured_scene_diff.{leaf_id}", context.ids
        ),
        "leaf_id": leaf_id,
        "status": status,
        "score": score,
        "metrics": dict(metrics),
        "evidence": dict(evidence or {}),
        "artifacts": dict(artifacts or {}),
        "probes_used": ("gt_scene_correspondence",),
    }
    if status not in {contracts.MEASURED, contracts.PASS}:
        result["failure_reason"] = reason or f"{leaf_id} was not evaluated"
    return result


def _pick(metrics: Mapping[str, Any], *names: str) -> dict[str, Any]:
    return {name: metrics.get(name) for name in names}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _unit(value: Any) -> float | None:
    number = _number(value)
    return None if number is None else max(0.0, min(1.0, number))


def _mean_available(values: list[float | None]) -> float | None:
    measured = [value for value in values if value is not None]
    return round(sum(measured) / len(measured), 6) if measured else None


def _set_f1(matched: Any, candidate_count: Any, gt_count: Any) -> float | None:
    match = _number(matched)
    candidate = _number(candidate_count)
    gt = _number(gt_count)
    if match is None or candidate is None or gt is None or candidate + gt <= 0:
        return None
    return max(0.0, min(1.0, 2.0 * match / (candidate + gt)))


def _transform_score(metrics: Mapping[str, Any]) -> tuple[float | None, dict[str, Any]]:
    scale = _number(metrics.get("normalization_scale_cm"))
    rmse = _number(metrics.get("aligned_position_rmse_cm"))
    rotation = _number(metrics.get("aligned_rotation_mean_deg"))
    scale_rmse = _number(metrics.get("scale_log_rmse"))
    channels = {
        "aligned_position_similarity": (
            1.0 / (1.0 + rmse / scale)
            if rmse is not None and scale is not None and scale > 0
            else None
        ),
        "cyclic_rotation_similarity": (
            max(0.0, 1.0 - rotation / 180.0)
            if rotation is not None
            else None
        ),
        "scale_ratio_similarity": (
            math.exp(-scale_rmse) if scale_rmse is not None else None
        ),
    }
    return _mean_available(list(channels.values())), channels


def _geometry_score(metrics: Mapping[str, Any]) -> tuple[float | None, dict[str, Any]]:
    fragmentation = _unit(metrics.get("fragmentation_error_rate"))
    bounds_log_rmse = _number(metrics.get("bounds_size_log_rmse"))
    layout_error = _number(metrics.get("pairwise_layout_error"))
    channels = {
        "footprint_iou": _unit(metrics.get("mean_footprint_iou")),
        "fragmentation_similarity": (
            1.0 - fragmentation if fragmentation is not None else None
        ),
        "bounds_size_similarity": (
            math.exp(-bounds_log_rmse) if bounds_log_rmse is not None else None
        ),
        "pairwise_layout_similarity": (
            1.0 / (1.0 + layout_error) if layout_error is not None else None
        ),
    }
    return _mean_available(list(channels.values())), channels


def _attribute_leaf(
    context: Context,
    metrics: Mapping[str, Any],
    evidence: Mapping[str, Any],
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    property_status = str(metrics.get("task_property_status") or "not_evaluated")
    material_set_status = str(
        metrics.get("material_set_status") or "not_evaluated"
    )
    material_slot_status = str(
        metrics.get("material_slot_status") or "not_evaluated"
    )
    submetrics = {
        "schema_contract": _pick(
            metrics,
            "attribute_schema_required_fields",
            "candidate_attribute_schema_complete_pair_count",
            "candidate_attribute_schema_missing_pair_count",
            "candidate_attribute_schema_coverage",
            "candidate_attribute_schema_missing_by_field",
            "canonical_attribute_schema_complete_pair_count",
            "canonical_attribute_schema_missing_pair_count",
            "canonical_attribute_schema_coverage",
            "canonical_attribute_schema_missing_by_field",
            "paired_attribute_schema_complete_pair_count",
            "paired_attribute_schema_coverage",
        ),
        "task_relevant_properties": {
            "status": property_status,
            **_pick(
                metrics,
                "task_property_pair_count",
                "task_property_mismatch_count",
                "task_property_match_rate",
            ),
        },
        # Kept for old snapshots. This is explicitly an unordered set metric,
        # not a substitute for component/slot identity.
        "legacy_material_set": {
            "status": material_set_status,
            **_pick(
                metrics,
                "material_set_pair_count",
                "material_set_mismatch_count",
                "material_set_match_rate",
            ),
        },
        "component_material_slots": {
            "status": material_slot_status,
            **_pick(
                metrics,
                "material_slot_pair_count",
                "material_slot_mismatch_count",
                "material_slot_match_rate",
                "material_slot_coverage",
                "material_slot_missing_pair_count",
                "material_slot_dynamic_refused_pair_count",
                "material_slot_invalid_pair_count",
            ),
        },
        "single_actor_correspondence_count": metrics.get(
            "attribute_single_actor_pair_count"
        ),
    }
    measured = {
        name
        for name, status in (
            ("task_relevant_properties", property_status),
            ("legacy_material_set", material_set_status),
            ("component_material_slots", material_slot_status),
        )
        if status == "measured"
    }
    if measured:
        rates = {
            "task_relevant_properties": _unit(metrics.get("task_property_match_rate")),
            "legacy_material_set": _unit(metrics.get("material_set_match_rate")),
            "component_material_slots": _unit(metrics.get("material_slot_match_rate")),
        }
        return _leaf(
            context,
            "attribute_diff",
            submetrics,
            score=_mean_available([rates[name] for name in sorted(measured)]),
            evidence={
                **evidence,
                "measured_submetrics": sorted(measured),
                "normalized_channels": rates,
                "score_formula": "macro_mean_of_measured_attribute_match_rates",
                "slot_identity_policy": (
                    "exact_normalized_ue_object_path_with_component_and_slot"
                ),
                "slot_refusal_policy": (
                    "missing_invalid_or_dynamic_evidence_is_not_evaluated"
                ),
            },
            artifacts=artifacts,
        )
    return _leaf(
        context,
        "attribute_diff",
        submetrics,
        status="not_evaluated",
        reason=(
            "no matched single-Actor correspondence carried reliable property, "
            "material-set, or component material-slot evidence; "
            "missing schema pairs: candidate={} canonical={}".format(
                metrics.get("candidate_attribute_schema_missing_pair_count"),
                metrics.get("canonical_attribute_schema_missing_pair_count"),
            )
        ),
        evidence=evidence,
        artifacts=artifacts,
    )


def _structured_report(context: Context) -> dict[str, Any]:
    comparison = run_leaf(context, gt_geometry.verify, spec=context.spec)
    return _structured_report_from_comparison(context, comparison)


def _structured_report_from_comparison(
    context: Context,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one already-computed Candidate/GT correspondence globally."""

    comparison = dict(comparison)
    if comparison.get("status") not in {contracts.MEASURED, contracts.PASS}:
        comparison["leaf_id"] = "actor_correspondence"
        return composite_report(
            "scene_diff.structured_scene_diff", context, [comparison]
        )

    metrics = comparison.get("metrics") or {}
    evidence = comparison.get("evidence") or {}
    artifacts = comparison.get("artifacts") or {}
    common_evidence = {
        "gt_id": evidence.get("gt_id"),
        "candidate_exported_from": evidence.get("candidate_exported_from"),
        "correspondence_policy": evidence.get("logical_object_policy"),
        "metric_version": metrics.get("metric_version"),
    }
    actor_f1 = _set_f1(
        metrics.get("matched_actor_count"),
        metrics.get("candidate_content_actor_count"),
        metrics.get("canonical_content_actor_count"),
    )
    logical_f1 = _set_f1(
        metrics.get("matched_logical_object_count"),
        metrics.get("candidate_logical_object_count"),
        metrics.get("canonical_logical_object_count"),
    )
    identity_rate = _unit(metrics.get("identity_match_rate"))
    identity_score = (
        logical_f1 * identity_rate
        if logical_f1 is not None and identity_rate is not None
        else logical_f1
    )
    transform_score, transform_channels = _transform_score(metrics)
    geometry_score, geometry_channels = _geometry_score(metrics)
    leaves = [
        _leaf(
            context,
            "actor_correspondence",
            _pick(
                metrics,
                "candidate_exported_actor_count",
                "canonical_exported_actor_count",
                "candidate_content_actor_count",
                "canonical_content_actor_count",
                "matched_actor_count",
                "missing_actor_count",
                "extra_actor_count",
                "candidate_logical_object_count",
                "canonical_logical_object_count",
                "matched_logical_object_count",
                "actor_assignment_cost_matrix_cell_count",
                "actor_assignment_cost_matrix_reduction_ratio",
                "assignment_cost_matrix_cell_count",
                "assignment_cost_matrix_reduction_ratio",
            ),
            score=actor_f1,
            evidence={
                **common_evidence,
                "score_formula": "2 * matched_actor_count / (candidate_content_actor_count + canonical_content_actor_count)",
                "metric_family": "set_correspondence_f1",
            },
            artifacts=artifacts,
        ),
        _leaf(
            context,
            "identity_diff",
            _pick(
                metrics,
                "matched_logical_object_count",
                "missing_logical_object_count",
                "extra_logical_object_count",
                "identity_comparison_pair_count",
                "identity_mismatch_count",
                "identity_match_rate",
                "matched_actor_count",
                "missing_actor_count",
                "extra_actor_count",
            ),
            score=identity_score,
            evidence={
                **common_evidence,
                "logical_object_correspondence_f1": logical_f1,
                "score_formula": "logical_object_correspondence_f1 * identity_match_rate",
            },
        ),
        _leaf(
            context,
            "transform_diff",
            _pick(
                metrics,
                "global_center_offset_vector_cm",
                "transform_correspondence_coverage",
                "absolute_location_error_cm",
                "absolute_location_axis_error_cm",
                "aligned_location_error_cm",
                "aligned_location_axis_error_cm",
                "cyclic_rotation_error_deg",
                "scale_log_error",
                "global_center_offset_cm",
                "global_translation_vector_cm",
                "global_translation_error_cm",
                "global_translation_normalized",
                "global_yaw_error_deg",
                "alignment_anchor_count",
                "alignment_anchor_rmse_cm",
                "aligned_position_rmse_cm",
                "aligned_position_mean_cm",
                "aligned_rotation_mean_deg",
                "rotation_pair_count",
                "scale_log_rmse",
            ),
            score=transform_score,
            evidence={
                **common_evidence,
                "location_units": "centimetres",
                "rotation_units": "cyclic_degrees",
                "scale_units": "log_ratio_rmse",
                "normalized_channels": transform_channels,
                "score_formula": "macro_mean(position_similarity, rotation_similarity, scale_similarity)",
            },
        ),
        _leaf(
            context,
            "geometry_diff",
            _pick(
                metrics,
                "raw_actor_count_delta",
                "logical_object_correspondence_coverage",
                "over_fragmented_actor_count",
                "under_segmented_actor_count",
                "fragmentation_error_rate",
                "bounds_measured",
                "bounds_size_log_rmse",
                "mean_footprint_iou",
                "pairwise_layout_error",
                "pairwise_layout_pair_count",
                "normalization_scale_cm",
            ),
            score=geometry_score,
            evidence={
                **common_evidence,
                "normalized_channels": geometry_channels,
                "score_formula": "macro_mean_of_available_geometry_similarity_channels",
            },
        ),
        _attribute_leaf(context, metrics, common_evidence, artifacts),
    ]
    return composite_report(
        "scene_diff.structured_scene_diff",
        context,
        leaves,
        evidence={
            **common_evidence,
            "primary_scope": "whole_scene",
            "correspondence_reused_by": [
                "identity_diff",
                "transform_diff",
                "geometry_diff",
                "attribute_diff",
            ],
        },
    )


def _visual_config(spec: Mapping[str, Any]) -> dict[str, Any]:
    configured = spec.get("visual_semantic_diff")
    return dict(configured) if isinstance(configured, Mapping) else {}


def _whole_scene_visual_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": "visual_as_judge",
        "mode": "gt_paired",
        "ground_truth": spec.get("ground_truth"),
        "comparison_scope": "whole_scene",
    }


def evidence_requests(
    task: Any,
    spec: dict[str, Any],
) -> tuple[EvidenceRequest, ...]:
    """Plan each selected leaf's evidence without sharing RenderSets."""

    configured = _visual_config(spec)
    requests: list[EvidenceRequest] = []
    if configured.get("caption_diff") is True:
        requests.extend(caption_similarity.evidence_requests(task, spec))
    if configured.get("paired_visual") is True:
        requests.extend(
            visual_as_judge.evidence_requests(task, _whole_scene_visual_spec(spec))
        )
    return tuple(requests)


def _not_applicable_leaf(
    context: Context,
    leaf_id: str,
    reason: str,
) -> dict[str, Any]:
    return {
        **contracts.base(
            f"scene_diff.visual_semantic_diff.{leaf_id}", context.ids
        ),
        "leaf_id": leaf_id,
        "status": "not_applicable",
        "score": None,
        "failure_reason": reason,
        "metrics": {},
        "evidence": {"selection_source": "frozen_scene_diff_spec"},
        "artifacts": {},
        "probes_used": (),
    }


def _caption_leaf(context: Context) -> dict[str, Any]:
    evaluator = atomic_registry.get("independent_caption_embedding_distance")
    leaf = run_leaf(
        context,
        lambda child: evaluator.evaluate_report(child, {}),
        spec=context.spec,
    )
    leaf["leaf_id"] = "caption_diff"
    evidence = dict(leaf.get("evidence") or {})
    evidence.update(
        {
            "independent_captioning": True,
            "comparison_space": "text_embedding_cosine_distance",
        }
    )
    if leaf.get("status") == contracts.ERROR:
        metrics = dict(leaf.get("metrics") or {})
        metrics.update(
            {
                "upstream_status": contracts.ERROR,
                "partial_aggregation_policy": "omit_with_coverage_penalty",
            }
        )
        leaf["metrics"] = metrics
        leaf["status"] = "not_evaluated"
        evidence["partial_aggregation_policy"] = (
            "caption failure remains visible but does not erase available "
            "paired-visual evidence"
        )
    leaf["evidence"] = evidence
    return leaf


def _visual_leaf(
    context: Context,
    leaf_id: str,
    *,
    status: str,
    score: float | None = None,
    metrics: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str] | None = None,
    probes_used: tuple[str, ...] = (),
    failure_reason: str | None = None,
) -> dict[str, Any]:
    result = {
        **contracts.base(
            f"scene_diff.visual_semantic_diff.{leaf_id}", context.ids
        ),
        "leaf_id": leaf_id,
        "status": status,
        "score": score,
        "metrics": dict(metrics or {}),
        "evidence": dict(evidence or {}),
        "artifacts": dict(artifacts or {}),
        "probes_used": probes_used,
    }
    if status not in {contracts.MEASURED, contracts.PASS}:
        result["failure_reason"] = failure_reason or (
            f"{leaf_id} was not measured under the frozen paired-visual policy"
        )
    return result


def _paired_visual_leaves(context: Context) -> list[dict[str, Any]]:
    report = run_leaf(
        context,
        visual_as_judge.verify,
        spec=_whole_scene_visual_spec(context.spec),
    )
    status = str(report.get("status") or contracts.ERROR)
    reason = str(report.get("failure_reason") or status)
    metrics = dict(report.get("metrics") or {})
    paired = dict(metrics.get("paired_render_diff") or {})
    equivalence = dict(metrics.get("visual_equivalence") or {})
    calibrated_value = metrics.get("calibrated_visual_score")
    calibrated = (
        dict(calibrated_value)
        if isinstance(calibrated_value, Mapping)
        else {}
    )
    report_evidence = dict(report.get("evidence") or {})
    measured = status in {contracts.MEASURED, contracts.PASS, contracts.FAIL}
    paired_measured = paired.get("status") == "measured"
    paired_status = (
        contracts.MEASURED
        if paired_measured
        else contracts.ERROR
        if status == contracts.ERROR
        else "not_evaluated"
    )
    visual_status = (
        contracts.MEASURED
        if measured
        else contracts.ERROR
        if status == contracts.ERROR
        else "not_evaluated"
    )
    score = calibrated.get("score")
    valid_score = (
        not isinstance(score, bool)
        and isinstance(score, (int, float))
        and math.isfinite(float(score))
        and 0.0 <= float(score) <= 1.0
    )
    calibrated_status = (
        contracts.MEASURED
        if measured and valid_score
        else contracts.ERROR
        if status == contracts.ERROR or measured
        else "not_evaluated"
    )
    calibrated_reason = (
        reason
        if status == contracts.ERROR or valid_score
        else "paired visual produced no finite calibrated score in [0, 1]"
    )
    provenance_fields = (
        "judge_policy",
        "calibration_status",
        "render_protocol",
        "rubric",
        "aggregation",
        "camera_alignment",
        "lighting_alignment",
    )
    provenance = {
        key: report_evidence[key]
        for key in provenance_fields
        if key in report_evidence
    }
    deterministic_aggregate = dict(paired.get("metrics") or {}).get("aggregate")
    deterministic_aggregate = (
        dict(deterministic_aggregate)
        if isinstance(deterministic_aggregate, Mapping)
        else {}
    )
    paired_score = _mean_available(
        [_unit(value) for value in deterministic_aggregate.values()]
    )
    visual_dimensions = equivalence.get("dimensions")
    visual_dimensions = (
        dict(visual_dimensions)
        if isinstance(visual_dimensions, Mapping)
        else {}
    )
    visual_score = _mean_available(
        [_unit(value) for value in visual_dimensions.values()]
    )
    paired_leaf = _visual_leaf(
        context,
        "paired_render_diff",
        status=(
            paired_status
            if paired_score is not None or paired_status != contracts.MEASURED
            else "not_evaluated"
        ),
        score=paired_score,
        metrics={
            **paired,
            "score_formula": "macro_mean_of_available_deterministic_similarity_channels",
        },
        evidence={
            **provenance,
            "measurement_kind": "deterministic_aligned_multichannel",
        },
        probes_used=("paired_render_metrics",),
        failure_reason=reason,
    )
    paired_leaf.update(
        {
            "contributes_to_aggregate": False,
            "score_role": "report_only",
            "report_only_reason": (
                "this deterministic channel is an input to the frozen "
                "calibrated_visual_score and must not vote a second time"
            ),
        }
    )
    equivalence_leaf = _visual_leaf(
        context,
        "visual_equivalence",
        status=(
            visual_status
            if visual_score is not None or visual_status != contracts.MEASURED
            else "not_evaluated"
        ),
        score=visual_score,
        metrics={
            **equivalence,
            "score_formula": "macro_mean_of_raw_vlm_dimension_scores",
        },
        evidence=report_evidence,
        artifacts=report.get("artifacts") or {},
        probes_used=("paired_view_vlm_judgement",),
        failure_reason=reason,
    )
    equivalence_leaf.update(
        {
            "contributes_to_aggregate": False,
            "score_role": "report_only",
            "report_only_reason": (
                "these raw VLM dimensions are inputs to the frozen "
                "calibrated_visual_score and must not vote a second time"
            ),
        }
    )
    calibrated_leaf = _visual_leaf(
        context,
        "calibrated_visual_score",
        status=calibrated_status,
        score=float(score) if valid_score else None,
        metrics={
            **calibrated,
            "contributes_to_gt_aggregate": True,
        },
        evidence=provenance,
        probes_used=("frozen_visual_score_aggregation",),
        failure_reason=calibrated_reason,
    )
    return [
        paired_leaf,
        equivalence_leaf,
        calibrated_leaf,
    ]


def _visual_semantic_report(context: Context) -> dict[str, Any]:
    configured = _visual_config(context.spec)
    explicit = bool(configured)
    caption_enabled = configured.get("caption_diff") is True
    paired_enabled = configured.get("paired_visual") is True
    leaves: list[dict[str, Any]] = []
    if caption_enabled:
        leaves.append(_caption_leaf(context))
    else:
        leaves.append(
            _not_applicable_leaf(
                context,
                "caption_diff",
                "caption_diff is disabled by the frozen scene_diff spec",
            )
        )
    if paired_enabled:
        leaves.extend(_paired_visual_leaves(context))
    else:
        for leaf_id in (
            "paired_render_diff",
            "visual_equivalence",
            "calibrated_visual_score",
        ):
            leaves.append(
                _not_applicable_leaf(
                    context,
                    leaf_id,
                    "paired_visual is disabled by the frozen scene_diff spec",
                )
            )
    return composite_report(
        "scene_diff.visual_semantic_diff",
        context,
        leaves,
        evidence={
            "configuration_profile": (
                "explicit-visual-selection"
                if explicit
                else "structured-only-default"
            ),
            "caption_diff_enabled": caption_enabled,
            "paired_visual_enabled": paired_enabled,
            "primary_scope": "whole_scene",
            "evidence_isolation": {
                "caption_diff": CAPTION_ENVIRONMENT_RENDER_PROTOCOL,
                "paired_visual": GT_VISUAL_ENVIRONMENT_RENDER_PROTOCOL,
            },
        },
    )


def verify(context: Context) -> dict[str, Any]:
    comparison = run_leaf(context, gt_geometry.verify, spec=context.spec)
    return _report_from_comparison(context, comparison)


def _report_from_comparison(
    context: Context,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the global scene-diff tree from one shared correspondence."""

    structured = _structured_report_from_comparison(context, comparison)
    structured["leaf_id"] = "structured_scene_diff"
    visual = _visual_semantic_report(context)
    visual["leaf_id"] = "visual_semantic_diff"
    return composite_report(
        "scene_diff",
        context,
        [structured, visual],
        evidence={
            "public_interface": "scene_diff",
            "report_schema_version": "scene-diff-public.v2",
            "legacy_top_level_replacements": {
                "caption_similarity": (
                    "scene_diff.visual_semantic_diff.caption_diff"
                ),
                "visual_as_judge": "scene_diff.visual_semantic_diff",
            },
        },
    )


__all__ = ["evidence_requests", "verify"]
