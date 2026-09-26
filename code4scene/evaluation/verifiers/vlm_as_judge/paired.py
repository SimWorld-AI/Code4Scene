"""Strict GT/candidate visual comparison for ``vlm_as_judge``.

This module deliberately has no fallback to candidate-only judging.  Its
numbers are meaningful only after ``strict_pair`` has proved that every GT
and candidate channel came from the same scene-relative camera protocol.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

from ... import contracts, image_metrics, scene_graph_capture
from ...context import Context
from ...paired_views import PairingError, pair, strict_pair
from ...vlm_concurrency import (
    parallel_map as parallel_vlm_map,
    runtime_snapshot as vlm_runtime_snapshot,
)
from .evidence import collect_paired
from .judge import Backend, Judge, JudgeError
from .policy import JudgePolicy


def _policy_evidence(policy: JudgePolicy, backend: Backend) -> dict[str, Any]:
    evidence = policy.evidence()
    effective_base_url = getattr(backend, "base_url", None) or policy.base_url
    evidence["effective_base_url"] = str(effective_base_url).rstrip("/")
    return evidence


def _json_finite(value: Any) -> Any:
    """Make metric output JSON-native without turning infinity into a score."""
    if isinstance(value, dict):
        return {str(key): _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"paired metric contains non-JSON value {type(value).__name__}")


def _similarity(metrics: dict[str, Any]) -> float:
    """Use measured global SSIM as a bounded reproducible similarity input."""
    return round(max(0.0, min(1.0, float(metrics["ssim"]))), 6)


def _paths_by_view(pairing: Any, channel: str) -> dict[str, tuple[Path, Path]]:
    return {
        view: (Path(reference), Path(candidate))
        for view, reference, candidate in pairing.pairs[channel]
    }


def _optional_mask_metrics(renders: Any, pairing: Any, view: str) -> dict[str, Any]:
    """Measure semantic/instance IoU when a capture provider supplied masks."""
    result: dict[str, Any] = {}
    for channel in ("semantic_mask", "instance_mask"):
        try:
            optional = pair(renders, channel)
        except PairingError:
            continue
        paths = {
            item_view: (Path(reference), Path(candidate))
            for item_view, reference, candidate in optional.pairs
        }
        if view not in paths:
            continue
        reference, candidate = paths[view]
        measured = image_metrics.semantic_mask_iou(
            image_metrics.load_mask(reference),
            image_metrics.load_mask(candidate),
        )
        result[channel] = measured
    if "semantic_mask" in result:
        result["semantic_mask_iou"] = result["semantic_mask"]["mean_iou"]
    elif "instance_mask" in result:
        result["semantic_mask_iou"] = result["instance_mask"]["mean_iou"]
    return result


def measure_view(
    renders: Any,
    pairing: Any,
    view: str,
    policy: JudgePolicy,
) -> dict[str, Any]:
    """Deterministic RGB, BaseColor and depth measurements for one camera."""
    rgb_reference, rgb_candidate = _paths_by_view(pairing, "rgb")[view]
    base_reference, base_candidate = _paths_by_view(
        pairing, "base_color")[view]
    depth_reference, depth_candidate = _paths_by_view(
        pairing, "scene_depth")[view]

    rgb = image_metrics.compare(
        image_metrics.load(rgb_reference), image_metrics.load(rgb_candidate))
    base_color = image_metrics.compare(
        image_metrics.load(base_reference), image_metrics.load(base_candidate))
    reference_depth = image_metrics.load_depth(depth_reference)
    candidate_depth = image_metrics.load_depth(depth_candidate)
    try:
        depth = image_metrics.compare_depth(
            reference_depth,
            candidate_depth,
            maximum_valid_depth_cm=policy.depth_far_cm,
            data_range_cm=policy.depth_far_cm - policy.depth_near_cm,
        )
    except image_metrics.ImageError as error:
        depth = {"status": "not_evaluated", "failure_reason": str(error)}
    geometry = image_metrics.compare_depth_geometry(
        reference_depth,
        candidate_depth,
        maximum_valid_depth_cm=policy.depth_far_cm,
        edge_threshold_cm=policy.aggregation["depth_edge_threshold_cm"],
        chamfer_normalization_px=(
            policy.aggregation["depth_edge_chamfer_normalization_px"]),
    )
    return _json_finite({
        "rgb": rgb,
        "base_color": base_color,
        "scene_depth": depth,
        "depth_geometry": geometry,
        "rgb_perceptual_similarity": _similarity(rgb),
        "base_color_similarity": _similarity(base_color),
        "foreground_silhouette_iou": geometry["foreground_silhouette_iou"],
        "depth_edge_similarity": geometry["depth_edge_similarity"],
        **_optional_mask_metrics(renders, pairing, view),
    })


def _blend_dimension(
    dimension: str,
    vlm_score: float,
    deterministic: dict[str, Any],
    policy: JudgePolicy,
) -> tuple[float, dict[str, Any]]:
    inputs = {"vlm": vlm_score, **deterministic}
    weights = policy.aggregation["deterministic_blend"][dimension]
    missing = [name for name in weights if inputs.get(name) is None]
    if missing:
        raise ValueError(
            f"{dimension} aggregation lacks required measured inputs {missing}")
    contributions = {
        name: round(float(inputs[name]) * float(weight), 6)
        for name, weight in weights.items()
    }
    return round(sum(contributions.values()), 6), {
        "inputs": {name: inputs[name] for name in weights},
        "weights": dict(weights),
        "contributions": contributions,
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty set of paired viewpoints")
    return round(sum(values) / len(values), 6)


def _artifact_namespace(context: Context) -> str:
    """Keep independent comparison scopes from overwriting audit evidence."""
    value = context.spec.get("comparison_scope", "whole_scene")
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(
            "comparison_scope must be a non-empty artifact-safe identifier"
        )
    return value


def _paired_artifacts(bundle: Any) -> dict[str, Any]:
    """Flatten every already-captured strict-pair image for audit/reporting."""

    return {
        f"paired_{view}_{channel}": paths["paired_image"]
        for view, channels in bundle.paired_images.items()
        for channel, paths in channels.items()
    }


def _camera_plan_policy(context: Context, policy: JudgePolicy) -> str | None:
    if context.spec.get("comparison_scope") == "repair_target":
        has_case_cameras = (
            getattr(context.task, "case_camera_manifest_path", None) is not None
        )
        return (
            (
                scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN
                if (
                    getattr(context.task, "scene_environment", None) == "indoor"
                    and has_case_cameras
                )
                else scene_graph_capture.GT_REPAIR_TARGET_INDOOR_CAMERA_PLAN
                if getattr(context.task, "scene_environment", None) == "indoor"
                else scene_graph_capture.GT_REPAIR_TARGET_OUTDOOR_CAMERA_PLAN
            )
            if policy.camera_plan_policy
            == scene_graph_capture.GT_VISUAL_CAMERA_PLAN
            else None
        )
    return policy.camera_plan_policy


def _expected_view_count(policy: JudgePolicy, camera_plan_policy: str | None) -> int:
    if (
        camera_plan_policy
        == scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN
    ):
        return scene_graph_capture.GT_REPAIR_TARGET_CASE_CAMERA_VIEW_COUNT
    return policy.view_count


def _scoring_views(context: Context, captured_views: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve an internal, capture-certified subset without weakening pairing."""

    configured = context.spec.get("_paired_visual_eligible_views")
    if configured is None:
        return captured_views
    if (
        not isinstance(configured, list)
        or not configured
        or any(not isinstance(value, str) for value in configured)
        or len(configured) != len(set(configured))
    ):
        raise ValueError("paired visual eligible views must be unique view ids")
    unknown = sorted(set(configured) - set(captured_views))
    if unknown:
        raise ValueError(
            f"paired visual quality gate selected uncaptured views {unknown}"
        )
    return tuple(view for view in captured_views if view in set(configured))


def _vlm_dimension_reason(
    dimension: str,
    score: float,
    per_view: dict[str, dict[str, Any]],
) -> str:
    rationales = [
        value["vlm_rationales"].get(dimension, "").strip()
        for value in per_view.values()
        if value["vlm_rationales"].get(dimension, "").strip()
    ]
    reason = " ".join(rationales)
    if not reason:
        reason = "No VLM rationale was returned for this dimension."
    return (
        f"Aggregated raw VLM score {score:.3f} across {len(per_view)} aligned "
        f"views. {reason}"
    )


def evaluate(
    context: Context,
    policy: JudgePolicy,
    backend: Backend,
) -> dict[str, Any]:
    """Run strict paired measurement and one stateless VLM call per view."""
    started = time.monotonic()
    camera_plan_policy = _camera_plan_policy(context, policy)
    pairing = strict_pair(
        context.renders,
        policy.channels,
        expected_view_count=_expected_view_count(policy, camera_plan_policy),
        camera_protocol=policy.render_protocol,
        camera_plan_policy=camera_plan_policy,
        lighting_policy=policy.lighting_policy,
        frame_quality_policy=policy.frame_quality_policy,
    )
    scoring_views = _scoring_views(context, pairing.views)
    artifact_namespace = _artifact_namespace(context)
    if context.out_dir is not None:
        evidence_dir = (
            Path(context.out_dir)
            / "judge_evidence"
            / "gt_paired"
            / artifact_namespace
        )
    else:
        first_path = next(
            Path(path)
            for views in context.renders.images.values()
            for channels in views.values()
            for path in channels.values()
        )
        evidence_dir = (
            first_path.parent
            / "judge_evidence"
            / "gt_paired"
            / artifact_namespace
        )
    bundle = collect_paired(
        context.renders,
        pairing,
        policy.rubric,
        evidence_dir,
        near_cm=policy.depth_near_cm,
        far_cm=policy.depth_far_cm,
        base_color_encoding=policy.base_color_encoding,
    )

    judge = Judge(
        rubric=policy.rubric,
        backend=backend,
        model=policy.model,
        min_images=1,
    )
    def score_view(view: str) -> tuple[str, dict[str, Any], JudgeError | None]:
        deterministic = measure_view(context.renders, pairing, view, policy)
        try:
            verdict = judge.score(
                prompt=context.task.prompt,
                images=[],
                # Keep the semantic judge independent from the deterministic
                # branch. Both become inputs to the frozen calibration below,
                # but neither consumes the other's result.
                metrics={},
                evidence=bundle.images_by_view[view],
                evidence_layout=policy.evidence_layout,
                comparison_mode="gt_paired",
            )
        except JudgeError as error:
            return view, {"deterministic_metrics": deterministic}, error
        vlm_scores = {
            key: round(value / policy.rubric.scale_max, 6)
            for key, value in verdict.scores.items()
        }
        calibrated_dimension_scores: dict[str, float] = {}
        blend_evidence: dict[str, Any] = {}
        for dimension in policy.aggregation["dimension_weights"]:
            calibrated_dimension_scores[dimension], blend_evidence[dimension] = (
                _blend_dimension(
                    dimension, vlm_scores[dimension], deterministic, policy))
        return view, {
            "deterministic_metrics": deterministic,
            "vlm_scores": vlm_scores,
            "vlm_rationales": verdict.rationales,
            "vlm_raw_structured_response": verdict.raw_response,
            "structured_output_recovery": verdict.structured_output_recovery,
            "calibrated_dimension_scores": calibrated_dimension_scores,
            "calibration": blend_evidence,
            "paired_images": bundle.paired_images[view],
        }, None

    scored_views = parallel_vlm_map(
        scoring_views,
        score_view,
        thread_name_prefix="code4scene-paired-vlm",
    )
    model_call_count = len(scored_views)
    per_view = {
        view: value for view, value, error in scored_views if error is None
    }
    failures = tuple(
        (view, value, error)
        for view, value, error in scored_views
        if error is not None
    )
    if failures:
        failed_view, _, failure = failures[0]
        # All stateless calls were launched concurrently. Preserve every
        # deterministic measurement even when one VLM verdict is malformed.
        partial_deterministic = {
            view: value["deterministic_metrics"]
            for view, value, _ in scored_views
        }
        policy_evidence = _policy_evidence(policy, backend)
        policy_evidence["render_protocol"] = {
            **policy_evidence["render_protocol"],
            "camera_plan_policy": camera_plan_policy,
        }
        return _json_finite({
            **contracts.base("vlm_as_judge", context.ids),
            "status": contracts.ERROR,
            "score": None,
            "failure_reason": f"JudgeError: {failure}",
            "metrics": {
                "deterministic": {"per_view": partial_deterministic},
                "model_call_count": model_call_count,
                "structured_output_recovery_count": sum(
                    bool(value["structured_output_recovery"])
                    for value in per_view.values()
                ),
                "total_time_s": round(time.monotonic() - started, 6),
            },
            "evidence": {
                **policy_evidence,
                **pairing.evidence(),
                "artifact_namespace": artifact_namespace,
                "views_captured": len(pairing.views),
                "views_judged_before_error": len(per_view),
                "judge_failure": {
                    "failed_view": failed_view,
                    "failure_kind": "evaluator_structured_verdict_error",
                },
                "paired_images": bundle.paired_images,
                "base_color_normalization": bundle.base_color_normalization,
                "depth_visualization": bundle.depth_visualization,
                "vlm_runtime": vlm_runtime_snapshot(),
            },
            "artifacts": _paired_artifacts(bundle),
            "probes_used": ("vlm_as_judge",),
        })

    vlm_dimensions = {
        dimension: _mean([
            result["vlm_scores"][dimension]
            for result in per_view.values()
        ])
        for dimension in policy.aggregation["dimension_weights"]
    }
    calibrated_dimensions = {
        dimension: _mean([
            result["calibrated_dimension_scores"][dimension]
            for result in per_view.values()
        ])
        for dimension in policy.aggregation["dimension_weights"]
    }
    overall = round(sum(
        calibrated_dimensions[dimension] * weight
        for dimension, weight in
        policy.aggregation["dimension_weights"].items()
    ), 6)
    deterministic_aggregate = {
        name: _mean([
            float(result["deterministic_metrics"][name])
            for result in per_view.values()
            if result["deterministic_metrics"].get(name) is not None
        ])
        for name in (
            "rgb_perceptual_similarity", "base_color_similarity",
            "foreground_silhouette_iou", "depth_edge_similarity",
        )
    }
    observations = [
        {
            "dimension": dimension,
            "score": score,
            "reason": _vlm_dimension_reason(dimension, score, per_view),
        }
        for dimension, score in vlm_dimensions.items()
    ]
    policy_evidence = _policy_evidence(policy, backend)
    policy_evidence["render_protocol"] = {
        **policy_evidence["render_protocol"],
        "camera_plan_policy": camera_plan_policy,
    }
    result = {
        **contracts.base("vlm_as_judge", context.ids),
        "status": contracts.MEASURED,
        "score": overall,
        "metrics": {
            "deterministic": {
                "aggregate": deterministic_aggregate,
                "per_view": {
                    view: value["deterministic_metrics"]
                    for view, value in per_view.items()
                },
            },
            "vlm": {
                "dimensions": vlm_dimensions,
                "per_view": {
                    view: value["vlm_scores"]
                    for view, value in per_view.items()
                },
            },
            "calibrated": {
                "dimensions": calibrated_dimensions,
                "overall_visual_similarity": overall,
            },
            "model_call_count": model_call_count,
            "structured_output_recovery_count": sum(
                bool(value["structured_output_recovery"])
                for value in per_view.values()
            ),
            "total_time_s": round(time.monotonic() - started, 6),
        },
        "evidence": {
            **policy_evidence,
            **pairing.evidence(),
            "artifact_namespace": artifact_namespace,
            "views_captured": len(pairing.views),
            "captured_views": list(pairing.views),
            "views_compared": len(scoring_views),
            "scored_views": list(scoring_views),
            "quality_excluded_views": sorted(
                set(pairing.views) - set(scoring_views)
            ),
            "observations": observations,
            "per_view": per_view,
            "paired_images": bundle.paired_images,
            "base_color_normalization": bundle.base_color_normalization,
            "depth_visualization": bundle.depth_visualization,
            "vlm_raw_structured_responses": {
                view: value["vlm_raw_structured_response"]
                for view, value in per_view.items()
            },
            "vlm_runtime": vlm_runtime_snapshot(),
        },
        "artifacts": _paired_artifacts(bundle),
        "probes_used": ("vlm_as_judge",),
    }
    return _json_finite(result)


__all__ = ["evaluate", "measure_view"]
