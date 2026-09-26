"""Protocol-keyed render evidence requests.

Verifiers own what must be photographed. Episode infrastructure only executes
these frozen requests and stores one independent ``RenderSet`` per protocol.
Different protocols are never merged merely because they use the same scene.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from . import render, scene_graph_capture


CAPTION_ENVIRONMENT_RENDER_PROTOCOL = (
    "caption_similarity.environment_visibility_gallery_rgb"
)
GT_VISUAL_ENVIRONMENT_RENDER_PROTOCOL = (
    "visual_as_judge.gt_environment_visibility_per_view_photometry_multichannel"
)
GT_REPAIR_TARGET_VISUAL_ENVIRONMENT_RENDER_PROTOCOL = (
    "visual_as_judge.gt_repair_target_visibility_per_view_photometry_"
    "multichannel"
)
GT_REPAIR_TARGET_VISUAL_PROTOCOL = (
    "visual_as_judge.gt_repair_target_visibility_per_view_photometry_"
    "input_gt_quality_gated_multichannel"
)
GT_REPAIR_TARGET_AUTHORED_VISUAL_PROTOCOL = (
    "visual_as_judge.gt_repair_target_authored_camera_priority_per_view_"
    "photometry_input_gt_quality_gated_multichannel"
)
GT_REPAIR_TARGET_VISUAL_QUALITY_GATE = (
    "input-gt-target-observability-min-two"
)
SUPPORTED_PAIRED_VISUAL_QUALITY_GATE_POLICIES = frozenset({
    GT_REPAIR_TARGET_VISUAL_QUALITY_GATE,
})

EvidenceKind = Literal["candidate_render", "paired_render"]


@dataclass(frozen=True, slots=True)
class EvidenceRequest:
    """One complete render protocol and its immutable capture parameters."""

    protocol: str
    kind: EvidenceKind
    channels: tuple[str, ...]
    view_count: int
    width: int
    height: int
    timeout_s: float
    camera_protocol: str
    scene_alignment: str | None
    lighting_policy: str | None = None
    exposure_policy: str | None = None
    camera_plan_policy: str | None = None
    frame_quality_policy: str | None = None
    scene_environment: str | None = None
    paired_visual_quality_gate: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, str) or not self.protocol.strip():
            raise ValueError("render evidence protocol must be a non-empty string")
        if self.kind not in {"candidate_render", "paired_render"}:
            raise ValueError(f"unsupported render evidence kind {self.kind!r}")
        if (
            not isinstance(self.channels, tuple)
            or not self.channels
            or len(set(self.channels)) != len(self.channels)
            or any(channel not in render.IMPLEMENTED_CHANNELS for channel in self.channels)
        ):
            raise ValueError("render evidence channels must be unique implemented channels")
        for name, value in (
            ("view_count", self.view_count),
            ("width", self.width),
            ("height", self.height),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(float(self.timeout_s))
            or float(self.timeout_s) <= 0.0
        ):
            raise ValueError("timeout_s must be a positive finite number")
        if self.camera_protocol not in render.SUPPORTED_RING_PROTOCOLS:
            raise ValueError(
                f"unsupported camera protocol {self.camera_protocol!r}"
            )
        if (
            self.camera_plan_policy is not None
            and self.camera_plan_policy
            not in scene_graph_capture.SUPPORTED_CAMERA_PLAN_POLICIES
        ):
            raise ValueError(
                f"unsupported camera plan policy {self.camera_plan_policy!r}"
            )
        if (
            self.frame_quality_policy is not None
            and self.frame_quality_policy
            not in scene_graph_capture.SUPPORTED_FRAME_QUALITY_POLICIES
        ):
            raise ValueError(
                f"unsupported frame quality policy {self.frame_quality_policy!r}"
            )
        if self.scene_environment not in {None, "indoor", "outdoor"}:
            raise ValueError(
                f"unsupported scene environment {self.scene_environment!r}"
            )
        if (
            self.paired_visual_quality_gate is not None
            and self.paired_visual_quality_gate
            not in SUPPORTED_PAIRED_VISUAL_QUALITY_GATE_POLICIES
        ):
            raise ValueError(
                "unsupported paired visual quality gate "
                f"{self.paired_visual_quality_gate!r}"
            )
        if self.paired_visual_quality_gate is not None and (
            self.kind != "paired_render"
            or "rgb" not in self.channels
            or self.camera_plan_policy not in {
                scene_graph_capture.GT_REPAIR_TARGET_INDOOR_CAMERA_PLAN,
                scene_graph_capture.GT_REPAIR_TARGET_AUTHORED_CAMERA_PLAN,
            }
            or self.frame_quality_policy
            != scene_graph_capture.GT_VISUAL_FRAME_QUALITY
            or self.view_count < 2
        ):
            raise ValueError(
                "paired visual quality gate requires a multi-view indoor "
                "repair-target RGB protocol with per-view photometry"
            )

    def to_json_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["channels"] = list(self.channels)
        return value


def merge_requests(values: list[EvidenceRequest]) -> tuple[EvidenceRequest, ...]:
    """Deduplicate identical protocols and reject protocol/config collisions."""

    by_protocol: dict[str, EvidenceRequest] = {}
    for value in values:
        previous = by_protocol.get(value.protocol)
        if previous is not None and previous != value:
            raise ValueError(
                f"render protocol {value.protocol!r} was requested with "
                "conflicting capture parameters"
            )
        by_protocol[value.protocol] = value
    return tuple(by_protocol[key] for key in sorted(by_protocol))


__all__ = [
    "CAPTION_ENVIRONMENT_RENDER_PROTOCOL",
    "EvidenceKind",
    "EvidenceRequest",
    "GT_VISUAL_ENVIRONMENT_RENDER_PROTOCOL",
    "GT_REPAIR_TARGET_VISUAL_ENVIRONMENT_RENDER_PROTOCOL",
    "GT_REPAIR_TARGET_VISUAL_PROTOCOL",
    "GT_REPAIR_TARGET_AUTHORED_VISUAL_PROTOCOL",
    "GT_REPAIR_TARGET_VISUAL_QUALITY_GATE",
    "SUPPORTED_PAIRED_VISUAL_QUALITY_GATE_POLICIES",
    "merge_requests",
]
