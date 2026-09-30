"""Global prompt alignment from four candidate overview renders.

One VLM call directly compares the original prompt with all four images. A
second prompt-blind call independently checks structural geometry integrity.
The visible-scene summary is explanatory output only and is never used as an
intermediate scoring input.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import contracts, render, scene_graph_capture, vlm_model_config
from ..context import Context, error
from ..offline_overview_prompt_alignment import (
    METRIC_ID,
    SCHEMA_VERSION,
    STRUCTURAL_INTEGRITY_PROTOCOL,
    VIEW_COUNT,
    evaluate_overview_frames,
)
from ..render_evidence import EvidenceRequest

CLASS = "open_ended"

RENDER_PROTOCOL = "overview_prompt_alignment.candidate_clearance_gallery"
REPORT_ONLY_REASON = (
    "global overview alignment counts toward the score only under the "
    "text-to-scene score policy, whose component weights were fitted to human "
    "preferences"
)


def evidence_requests(
    _task: Any,
    _spec: dict[str, Any],
) -> tuple[EvidenceRequest, ...]:
    """Request one reusable candidate-only four-view scene-graph gallery."""

    return (
        EvidenceRequest(
            protocol=RENDER_PROTOCOL,
            kind="candidate_render",
            channels=("rgb",),
            view_count=VIEW_COUNT,
            width=render.DEFAULT_WIDTH,
            height=render.DEFAULT_HEIGHT,
            timeout_s=300.0,
            camera_protocol=render.REFERENCE_OVERVIEW_SCENE_GRAPH_PROTOCOL,
            scene_alignment="scene-graph-grounded-core",
            lighting_policy=render.LIGHTING_NORMALIZATION_POLICY,
            exposure_policy=None,
            camera_plan_policy=scene_graph_capture.REFERENCE_CAMERA_PLAN,
            frame_quality_policy=scene_graph_capture.REFERENCE_FRAME_QUALITY,
        ),
    )


def _natural_view_key(value: str) -> tuple[int, str]:
    suffix = value.rsplit("_", 1)[-1]
    return (
        int(suffix) if value.startswith("view_") and suffix.isdigit() else 1_000_000,
        value,
    )


def _frames(renders: Any) -> tuple[dict[str, Any], ...]:
    images = getattr(renders, "images", {})
    candidate = images.get("candidate", {}) if isinstance(images, Mapping) else {}
    if not isinstance(candidate, Mapping):
        candidate = {}
    camera_specs = getattr(renders, "camera_specs", {})
    candidate_specs = (
        camera_specs.get("candidate", {})
        if isinstance(camera_specs, Mapping)
        else {}
    )
    frames = []
    for view, channels in sorted(candidate.items(), key=lambda item: _natural_view_key(str(item[0]))):
        if not isinstance(channels, Mapping) or not channels.get("rgb"):
            continue
        name = str(view)
        frames.append(
            {
                "frame_id": f"formal_{name}",
                "shot_role": "overview",
                "phase": "formal_render_evidence",
                "path": str(channels["rgb"]),
                "camera_spec": (
                    dict(candidate_specs.get(name) or {})
                    if isinstance(candidate_specs, Mapping)
                    else {}
                ),
            }
        )
    if len(frames) != VIEW_COUNT:
        raise ValueError(
            f"{RENDER_PROTOCOL} needs exactly {VIEW_COUNT} candidate RGB views; "
            f"found {len(frames)}"
        )
    return tuple(frames)


def _write_artifact(context: Context, payload: Mapping[str, Any]) -> str | None:
    root = context.out_dir or context.artifacts_dir
    if root is None:
        return None
    path = Path(root) / "overview_prompt_alignment" / "comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(path)


def verify(context: Context) -> dict[str, Any]:
    renders = context.renders_for(RENDER_PROTOCOL)
    if renders is None:
        return error(
            "overview_prompt_alignment",
            context,
            f"required candidate render evidence is unavailable: {RENDER_PROTOCOL}",
        )
    try:
        frames = _frames(renders)
        measured = evaluate_overview_frames(
            context.task.prompt,
            frames,
            model_label=str(
                context.spec.get("model_label") or vlm_model_config.MODEL
            ),
        )
        payload = {
            **measured,
            "authoritative": False,
            "offline": False,
            "ue_recapture_performed": True,
            "source": {
                "render_protocol": RENDER_PROTOCOL,
                "frames": measured["frames"],
                "camera_plan_audit": dict(
                    getattr(renders, "camera_plan_audit", {}) or {}
                ),
                "capture_overrides": list(
                    getattr(renders, "capture_overrides", []) or []
                ),
            },
        }
        artifact = _write_artifact(context, payload)
    except Exception as exc:  # noqa: BLE001 - verifier failure stays isolated
        return error(
            "overview_prompt_alignment",
            context,
            f"{type(exc).__name__}: {exc}",
        )

    artifacts = {"overview_prompt_alignment": artifact} if artifact else {}
    return {
        **contracts.base("overview_prompt_alignment", context.ids),
        "status": contracts.MEASURED,
        "score": measured["score"],
        "contributes_to_aggregate": False,
        "score_role": "report_only",
        "report_only_reason": REPORT_ONLY_REASON,
        "metrics": {
            "overview_score": measured["score"],
            "overview_alignment_score": measured["overview_alignment_score"],
            "structural_integrity_score": measured[
                "structural_integrity_score"
            ],
            "structural_integrity_status": measured[
                "structural_integrity_status"
            ],
            "structural_adjustment_multiplier": measured[
                "structural_adjustment_multiplier"
            ],
            "overview_score_before_severe_cap": measured[
                "overview_score_before_severe_cap"
            ],
            "severe_structural_cap_eligible": measured[
                "severe_structural_cap_eligible"
            ],
            "severe_structural_cap_applied": measured[
                "severe_structural_cap_applied"
            ],
            "severe_structural_cap": measured["severe_structural_cap"],
            "structural_adjustment_policy": measured[
                "structural_adjustment_policy"
            ],
            "dimensions": measured["dimensions"],
            "dimension_weights": measured["dimension_weights"],
            "score_policy": measured["score_policy"],
            "calibration_status": measured["calibration_status"],
            "model_call_count": measured["calls"]["count"],
        },
        "evidence": {
            "schema_version": SCHEMA_VERSION,
            "metric_version": METRIC_ID,
            "render_protocol": RENDER_PROTOCOL,
            "structural_integrity_protocol": STRUCTURAL_INTEGRITY_PROTOCOL,
            "prompt_sha256": measured["prompt_sha256"],
            "visible_scene_summary": measured["visible_scene_summary"],
            "matched_prompt_elements": measured["matched_prompt_elements"],
            "missing_or_unsupported_prompt_elements": measured[
                "missing_or_unsupported_prompt_elements"
            ],
            "allowed_nonstandard_geometry": measured[
                "allowed_nonstandard_geometry"
            ],
            "structural_geometry_integrity": measured[
                "structural_geometry_integrity"
            ],
            "summary": measured["summary"],
            "frames": measured["frames"],
            "model": measured["model"],
            "calls": measured["calls"],
            "direct_multimodal_prompt_comparison": True,
            "caption_used_as_scoring_intermediate": False,
        },
        "artifacts": artifacts,
        "probes_used": ("overview_prompt_alignment", RENDER_PROTOCOL),
    }


__all__ = [
    "CLASS",
    "RENDER_PROTOCOL",
    "evidence_requests",
    "verify",
]
