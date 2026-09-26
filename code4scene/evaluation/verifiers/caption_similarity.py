"""GT/Candidate independent-caption continuous similarity."""

from __future__ import annotations

from typing import Any

from .. import atomic_registry, render, scene_graph_capture
from ..composite import composite_report, run_leaf
from ..context import Context
from ..render_evidence import (
    CAPTION_ENVIRONMENT_RENDER_PROTOCOL,
    EvidenceRequest,
)
from . import gt_caption_similarity


def render_protocol(task: Any) -> str:
    environment = getattr(task, "scene_environment", None)
    if environment not in {"indoor", "outdoor"}:
        raise ValueError(
            "caption evidence requires explicit scene_environment: indoor|outdoor"
        )
    return CAPTION_ENVIRONMENT_RENDER_PROTOCOL


def evidence_requests(task: Any, _spec: dict[str, Any]) -> tuple[EvidenceRequest, ...]:
    environment = getattr(task, "scene_environment", None)
    if environment not in {"indoor", "outdoor"}:
        raise ValueError(
            "caption evidence requires explicit scene_environment: indoor|outdoor"
        )
    return (
        EvidenceRequest(
            protocol=render_protocol(task),
            kind="paired_render",
            channels=("rgb",),
            view_count=gt_caption_similarity.VIEW_COUNT,
            width=render.DEFAULT_WIDTH,
            height=render.DEFAULT_HEIGHT,
            timeout_s=300.0,
            camera_protocol=render.GT_CAPTION_ENVIRONMENT_SCENE_GRAPH_PROTOCOL,
            scene_alignment="input-gt-environment-routed-frozen",
            lighting_policy=render.LIGHTING_NORMALIZATION_POLICY,
            exposure_policy=render.EXPOSURE_NORMALIZATION_POLICY,
            camera_plan_policy=scene_graph_capture.CAPTION_CAMERA_PLAN,
            frame_quality_policy=scene_graph_capture.CAPTION_FRAME_QUALITY,
            scene_environment=str(environment),
        ),
    )


def verify(context: Context) -> dict[str, Any]:
    evaluator = atomic_registry.get("independent_caption_embedding_distance")
    leaf = run_leaf(
        context,
        lambda child: evaluator.evaluate_report(child, {}),
        spec=context.spec,
    )
    leaf["leaf_id"] = "independent_caption_embedding_distance"
    return composite_report("caption_similarity", context, [leaf])


__all__ = ["evidence_requests", "render_protocol", "verify"]
