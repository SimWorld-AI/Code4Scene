"""Internal GT-paired visual leaf used by ``scene_diff``.

Candidate-only prompt judgement belongs to RequirementGraph visual leaves.
This entry accepts only the frozen gt_paired policy.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .. import contracts
from ..context import Context, error
from ..render_evidence import (
    EvidenceRequest,
    GT_REPAIR_TARGET_VISUAL_ENVIRONMENT_RENDER_PROTOCOL,
    GT_REPAIR_TARGET_VISUAL_QUALITY_GATE,
    GT_REPAIR_TARGET_AUTHORED_VISUAL_PROTOCOL,
    GT_REPAIR_TARGET_VISUAL_PROTOCOL,
    GT_VISUAL_ENVIRONMENT_RENDER_PROTOCOL,
)
from ..repair_target_scope import is_repair_target_task
from .. import scene_graph_capture
from . import vlm_as_judge


_COMPARISON_SCOPES = frozenset({"auto", "whole_scene", "repair_target"})


def _quality_gate_policy(task: Any, comparison_scope: str) -> str | None:
    """Apply the calibrated Input/GT gate without task-level policy switches."""

    environment = str(
        getattr(task, "scene_environment", "") or ""
    ).strip().casefold()
    return (
        GT_REPAIR_TARGET_VISUAL_QUALITY_GATE
        if comparison_scope == "repair_target" and environment == "indoor"
        else None
    )


def _quality_gate_certificate(
    renders: Any,
    policy: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return one complete capture-time certificate or a withholding reason."""

    if policy is None:
        return None, None
    matches = [
        value
        for value in (getattr(renders, "capture_overrides", None) or ())
        if isinstance(value, dict) and value.get("policy") == policy
    ]
    if len(matches) != 1:
        return None, (
            "paired visual quality gate needs exactly one Input/GT "
            f"certificate; found {len(matches)}"
        )
    certificate = dict(matches[0])
    eligible = certificate.get("eligible_views")
    minimum = certificate.get("minimum_eligible_view_count")
    if (
        certificate.get("candidate_used_for_acceptance") is not False
        or not isinstance(eligible, list)
        or len(eligible) != len(set(eligible))
        or isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum < 2
    ):
        return certificate, "paired visual quality certificate is malformed"
    available = set((getattr(renders, "images", None) or {}).get("gt", {}))
    available &= set(
        (getattr(renders, "images", None) or {}).get("candidate", {})
    )
    if not set(eligible).issubset(available):
        return certificate, (
            "paired visual quality certificate selects uncaptured views"
        )
    if certificate.get("accepted") is not True or len(eligible) < minimum:
        return certificate, str(
            certificate.get("failure_reason")
            or "fewer than two high-quality Input/GT target views are available"
        )
    return certificate, None


def _raw_render_artifacts(renders: Any) -> dict[str, str]:
    """Keep every attempted Candidate/GT screenshot on a withheld visual leaf."""

    return {
        f"captured_{scene}_{view}_{channel}": str(path)
        for scene, views in sorted(
            (getattr(renders, "images", None) or {}).items()
        )
        for view, channels in sorted(views.items())
        for channel, path in sorted(channels.items())
    }


def _quality_withheld_report(
    context: Context,
    renders: Any,
    certificate: dict[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    artifacts = _raw_render_artifacts(renders)
    if certificate and certificate.get("artifact"):
        artifacts["paired_visual_quality_gate"] = str(certificate["artifact"])
    return {
        **contracts.base("visual_as_judge", context.ids),
        "status": "not_evaluated",
        "score": None,
        "failure_reason": reason,
        "metrics": {"model_call_count": 0},
        "evidence": {
            "paired_visual_quality_gate": certificate or {},
            "visual_score_role": "withheld_low_quality_evidence",
            "candidate_used_for_quality_gate": False,
            "screenshots_retained_for_audit": True,
        },
        "artifacts": artifacts,
        "probes_used": ("input_gt_paired_visual_quality_gate",),
    }


def _comparison_scope(task: Any, spec: dict[str, Any]) -> str:
    """Resolve the paired-visual scope selected by the owning verifier."""

    configured = str(spec.get("comparison_scope") or "auto").strip().casefold()
    if configured not in _COMPARISON_SCOPES:
        raise ValueError(
            "visual_as_judge.comparison_scope must be auto, whole_scene, or "
            "repair_target"
        )
    if configured == "auto":
        return "repair_target" if is_repair_target_task(task) else "whole_scene"
    if configured == "repair_target" and not is_repair_target_task(task):
        raise ValueError(
            "repair_target visual comparison needs a GT repair task with a "
            "frozen Input snapshot"
        )
    return configured




def evidence_requests(task: Any, spec: dict[str, Any]) -> tuple[EvidenceRequest, ...]:
    policy = vlm_as_judge.load_policy()
    if policy.mode != "gt_paired":
        raise ValueError("visual_as_judge evidence policy must be gt_paired")
    comparison_scope = _comparison_scope(task, spec)
    quality_gate = _quality_gate_policy(task, comparison_scope)
    scene_environment = getattr(task, "scene_environment", None)
    has_case_cameras = (
        getattr(task, "case_camera_manifest_path", None) is not None
    )
    if scene_environment not in {"indoor", "outdoor"}:
        raise ValueError(
            "visual evidence requires explicit scene_environment: indoor|outdoor"
        )
    if comparison_scope == "repair_target":
        protocol = (
            GT_REPAIR_TARGET_AUTHORED_VISUAL_PROTOCOL
            if scene_environment == "indoor" and has_case_cameras
            else GT_REPAIR_TARGET_VISUAL_PROTOCOL
            if scene_environment == "indoor"
            else GT_REPAIR_TARGET_VISUAL_ENVIRONMENT_RENDER_PROTOCOL
        )
        camera_plan_policy = (
            scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN
            if scene_environment == "indoor" and has_case_cameras
            else scene_graph_capture.GT_REPAIR_TARGET_INDOOR_CAMERA_PLAN
            if scene_environment == "indoor"
            else scene_graph_capture.GT_REPAIR_TARGET_OUTDOOR_CAMERA_PLAN
        )
    else:
        protocol = GT_VISUAL_ENVIRONMENT_RENDER_PROTOCOL
        camera_plan_policy = policy.camera_plan_policy
    view_count = (
        scene_graph_capture.GT_REPAIR_TARGET_CASE_CAMERA_VIEW_COUNT
        if camera_plan_policy
        == scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN
        else policy.view_count
    )
    return (
        EvidenceRequest(
            protocol=protocol,
            kind="paired_render",
            channels=policy.channels,
            view_count=view_count,
            width=policy.width,
            height=policy.height,
            timeout_s=policy.timeout_s,
            camera_protocol=policy.render_protocol,
            scene_alignment=policy.scene_alignment,
            lighting_policy=policy.lighting_policy,
            exposure_policy=policy.exposure_policy,
            camera_plan_policy=camera_plan_policy,
            frame_quality_policy=policy.frame_quality_policy,
            scene_environment=scene_environment,
            paired_visual_quality_gate=quality_gate,
        ),
    )


def verify(context: Context) -> dict[str, Any]:
    configured_mode = str(context.spec.get("mode") or "gt_paired")
    if configured_mode != "gt_paired":
        return error(
            "visual_as_judge",
            context,
            "canonical visual_as_judge accepts only frozen gt_paired policy; "
            "candidate-only prompt visual claims belong to semantic_requirements",
        )
    spec = {**context.spec, "mode": "gt_paired"}
    comparison_scope = _comparison_scope(context.task, spec)
    render_protocol = evidence_requests(context.task, spec)[0].protocol
    renders = context.renders_for(render_protocol)
    quality_policy = _quality_gate_policy(context.task, comparison_scope)
    certificate, gate_failure = _quality_gate_certificate(
        renders, quality_policy
    )
    if gate_failure is not None:
        result = _quality_withheld_report(
            context, renders, certificate, gate_failure
        )
    else:
        scoring_spec = dict(spec)
        if certificate is not None:
            scoring_spec["_paired_visual_eligible_views"] = list(
                certificate["eligible_views"]
            )
        result = vlm_as_judge.verify(
            replace(context, spec=scoring_spec, renders=renders)
        )
        if certificate is not None:
            result.setdefault("evidence", {})[
                "paired_visual_quality_gate"
            ] = certificate
    result["report_id"] = "visual_as_judge"
    metrics = dict(result.get("metrics") or {})
    deterministic = metrics.get("deterministic") or {}
    vlm = metrics.get("vlm") or {}
    calibrated = metrics.get("calibrated") or {}
    visual_measured = result.get("status") in {
        contracts.MEASURED,
        contracts.PASS,
        contracts.FAIL,
    }
    gate_evidence = (
        certificate.get("per_view") if isinstance(certificate, dict) else None
    )
    result["metrics"] = {
        "paired_render_diff": {
            "status": (
                "measured" if deterministic else "not_evaluated"
            ),
            "channels": ["rgb", "base_color", "scene_depth"],
            "metrics": deterministic,
            "units": {
                "rgb_mae": "normalized_pixel_value",
                "base_color_mae": "normalized_pixel_value",
                "scene_depth_mae": "cm",
                "foreground_silhouette_iou": "ratio",
            },
            "contributes_to_visual_score": visual_measured,
        },
        "visual_equivalence": {
            "status": (
                "measured"
                if result.get("status") in {"measured", "pass", "fail"}
                else "not_evaluated"
            ),
            "dimensions": vlm.get("dimensions") or {},
            "per_view": vlm.get("per_view") or {},
            "observations": (result.get("evidence") or {}).get("observations", []),
            "judgement_kind": "raw_vlm_semantic_equivalence",
            "contributes_to_visual_score": visual_measured,
        },
        "edit_region": {
            "status": "measured" if gate_evidence is not None else "not_applicable",
            "reason": (
                "aligned Input/GT views certify target observability but do not "
                "score Candidate"
                if gate_evidence is not None
                else "canonical GT-paired capture has no aligned frozen Input views"
            ),
            "per_view": gate_evidence or {},
            "contributes_to_visual_score": False,
        },
        "calibrated_visual_score": {
            "score": result.get("score"),
            "dimensions": calibrated.get("dimensions") or {},
            "aggregation": "frozen_deterministic_vlm_blend",
            "contributes_to_visual_score": visual_measured,
        },
        "model_call_count": metrics.get("model_call_count"),
        "total_time_s": metrics.get("total_time_s"),
    }
    result.setdefault("evidence", {}).update(
        {
            "canonical_interface": "visual_as_judge",
            "comparison_scope": (
                "gt_minus_input_repair_target"
                if comparison_scope == "repair_target"
                else "whole_scene"
            ),
            "submetric_policy": (
                "paired_render_diff and visual_equivalence stay independent; "
                "only calibrated_visual_score blends them under the frozen policy"
            ),
            "visual_coverage": {
                "quality_gate_policy": quality_policy,
                "captured_view_count": (
                    certificate.get("captured_view_count")
                    if isinstance(certificate, dict)
                    else None
                ),
                "eligible_view_count": (
                    certificate.get("eligible_view_count")
                    if isinstance(certificate, dict)
                    else None
                ),
                "eligible_views": (
                    certificate.get("eligible_views", [])
                    if isinstance(certificate, dict)
                    else []
                ),
                "visual_score_available": visual_measured,
            },
        }
    )
    return result


__all__ = ["evidence_requests", "verify"]
