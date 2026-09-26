"""Canonical frozen policy for the formal visual judge.

The release has one executable policy. Tasks choose whether paired visual
scoring is enabled; they do not choose an implementation generation. The
policy remains file-backed and hashable so score provenance can pin exact
content without carrying an algorithm version selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ... import render, scene_graph_capture
from . import model_config
from .rubric import Rubric, load_rubric


class JudgePolicyError(ValueError):
    """The formal judge policy is incomplete, changed, or unsupported."""


@dataclass(frozen=True)
class JudgePolicy:
    id: str
    path: Path
    rubric: Rubric
    rubric_path: Path
    mode: str
    aggregation: dict[str, Any]
    backend: str
    base_url: str
    model: str
    max_tokens: int
    judge_timeout_s: float
    temperature: float
    seed: int
    enable_thinking: bool
    structured_output: bool
    structured_output_method: str
    evidence_layout: str
    render_protocol: str
    lighting_policy: str
    base_color_encoding: str
    exposure_policy: str
    scene_alignment: str
    camera_plan_policy: str
    frame_quality_policy: str
    view_count: int
    minimum_complete_viewpoints: int
    width: int
    height: int
    fov_deg: float
    radius_half_extent_multiplier: float
    height_radius_multiplier: float
    channels: tuple[str, ...]
    timeout_s: float
    depth_policy: str
    depth_near_cm: float
    depth_far_cm: float
    calibration_status: str

    @property
    def name(self) -> str:
        return self.id

    def evidence(self) -> dict[str, Any]:
        """JSON-native provenance carried by every formal visual report."""

        return {
            "judge_policy": self.name,
            "judge_policy_status": "frozen",
            "scoring_policy": "formal",
            "mode": self.mode,
            "aggregation": self.aggregation,
            "rubric": self.rubric.name,
            "backend": self.backend,
            "base_url": self.base_url,
            "model": self.model,
            "judge_parameters": {
                "structured_output": self.structured_output,
                "structured_output_method": self.structured_output_method,
                "max_tokens": self.max_tokens,
                "timeout_s": self.judge_timeout_s,
                "temperature": self.temperature,
                "seed": self.seed,
                "enable_thinking": self.enable_thinking,
                "evidence_layout": self.evidence_layout,
            },
            "render_protocol": {
                "id": self.render_protocol,
                "lighting_normalization": self.lighting_policy,
                "base_color_encoding": self.base_color_encoding,
                "exposure_normalization": self.exposure_policy,
                "scene_alignment": self.scene_alignment,
                "camera_plan_policy": self.camera_plan_policy,
                "frame_quality_policy": self.frame_quality_policy,
                "view_count": self.view_count,
                "minimum_complete_viewpoints": self.minimum_complete_viewpoints,
                "width": self.width,
                "height": self.height,
                "fov_deg": self.fov_deg,
                "radius_half_extent_multiplier": self.radius_half_extent_multiplier,
                "height_radius_multiplier": self.height_radius_multiplier,
                "channels": list(self.channels),
                "timeout_s": self.timeout_s,
                "depth_visualization": {
                    "policy": self.depth_policy,
                    "near_cm": self.depth_near_cm,
                    "far_cm": self.depth_far_cm,
                },
            },
            "calibration_status": self.calibration_status,
        }


def canonical_policy_path() -> Path:
    """Return the only packaged formal visual policy."""

    from ....resources import config_file

    return config_file("judge-policies", "visual-gt-paired-formal.yaml")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JudgePolicyError(f"{name} must be a mapping")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise JudgePolicyError(f"{name} has " + "; ".join(details))


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise JudgePolicyError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0 or parsed != value:
        raise JudgePolicyError(f"{name} must be a positive integer")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise JudgePolicyError(f"{name} must be a positive number")
    parsed = float(value)
    if parsed <= 0:
        raise JudgePolicyError(f"{name} must be a positive number")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise JudgePolicyError(f"{name} must be a non-negative integer")
    parsed = int(value)
    if parsed < 0 or parsed != value:
        raise JudgePolicyError(f"{name} must be a non-negative integer")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise JudgePolicyError(f"{name} must be a non-negative number")
    parsed = float(value)
    if parsed < 0:
        raise JudgePolicyError(f"{name} must be a non-negative number")
    return parsed


def _aggregation(
    value: Any,
    rubric: Rubric,
) -> dict[str, Any]:
    aggregation = _mapping(value, "judge policy aggregation")
    _exact_keys(
        aggregation,
        {
            "protocol",
            "view_aggregation",
            "dimension_weights",
            "deterministic_blend",
            "depth_edge_threshold_cm",
            "depth_edge_chamfer_normalization_px",
        },
        "judge policy aggregation",
    )
    if aggregation["protocol"] != "gt-paired-weighted":
        raise JudgePolicyError("unsupported gt_paired aggregation protocol")
    if aggregation["view_aggregation"] != "arithmetic-mean":
        raise JudgePolicyError("unsupported gt_paired view aggregation")
    criterion_ids = {criterion.id for criterion in rubric.criteria}
    weights = _mapping(
        aggregation["dimension_weights"], "aggregation dimension_weights"
    )
    blend = _mapping(
        aggregation["deterministic_blend"], "aggregation deterministic_blend"
    )
    _exact_keys(weights, criterion_ids, "aggregation dimension_weights")
    _exact_keys(blend, criterion_ids, "aggregation deterministic_blend")
    parsed_weights = {
        key: _nonnegative_float(value, f"dimension_weights.{key}")
        for key, value in weights.items()
    }
    if abs(sum(parsed_weights.values()) - 1.0) > 1e-9:
        raise JudgePolicyError("gt_paired dimension weights must sum to 1")
    allowed_inputs = {
        "vlm",
        "rgb_perceptual_similarity",
        "base_color_similarity",
        "foreground_silhouette_iou",
        "depth_edge_similarity",
        "semantic_mask_iou",
    }
    parsed_blend: dict[str, dict[str, float]] = {}
    for dimension, raw_parts in blend.items():
        parts = _mapping(raw_parts, f"deterministic_blend.{dimension}")
        unknown = sorted(set(parts) - allowed_inputs)
        if unknown:
            raise JudgePolicyError(
                f"deterministic_blend.{dimension} has unknown inputs {unknown}"
            )
        parsed = {
            key: _nonnegative_float(weight, f"{dimension}.{key}")
            for key, weight in parts.items()
        }
        if abs(sum(parsed.values()) - 1.0) > 1e-9:
            raise JudgePolicyError(
                f"deterministic_blend.{dimension} weights must sum to 1"
            )
        parsed_blend[dimension] = parsed
    return {
        **aggregation,
        "dimension_weights": parsed_weights,
        "deterministic_blend": parsed_blend,
        "depth_edge_threshold_cm": _positive_float(
            aggregation["depth_edge_threshold_cm"],
            "aggregation.depth_edge_threshold_cm",
        ),
        "depth_edge_chamfer_normalization_px": _positive_float(
            aggregation["depth_edge_chamfer_normalization_px"],
            "aggregation.depth_edge_chamfer_normalization_px",
        ),
    }


def load_policy(path: str | Path | None = None) -> JudgePolicy:
    """Load current policy content from the package or a frozen artifact copy.

    Public task configuration cannot choose this path. Accepting a copy here
    is required for self-contained evaluation artifacts; the exact schema and
    implementation identifiers below still reject historical policy content.
    """

    policy_path = Path(path or canonical_policy_path()).resolve()
    try:
        data = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise JudgePolicyError(
            f"cannot read judge policy {policy_path}: {error}"
        ) from error
    data = _mapping(data, "judge policy")
    _exact_keys(
        data,
        {
            "id",
            "status",
            "scoring_policy",
            "mode",
            "rubric",
            "judge",
            "render",
            "aggregation",
            "calibration_status",
        },
        "judge policy",
    )
    if (
        data["status"] != "frozen"
        or data["scoring_policy"] != "formal"
        or data["mode"] != "gt_paired"
    ):
        raise JudgePolicyError(
            "canonical policy requires status=frozen, scoring_policy=formal, "
            "and mode=gt_paired"
        )

    rubric_spec = _mapping(data["rubric"], "judge policy rubric")
    _exact_keys(rubric_spec, {"path"}, "judge policy rubric")
    rubric_path = Path(str(rubric_spec["path"]))
    if not rubric_path.is_absolute():
        rubric_path = (policy_path.parent / rubric_path).resolve()
    rubric = load_rubric(rubric_path)
    aggregation = _aggregation(data["aggregation"], rubric)

    judge = _mapping(data["judge"], "judge policy judge")
    _exact_keys(
        judge,
        {
            "backend",
            "model",
            "structured_output",
            "structured_output_method",
            "max_tokens",
            "timeout_s",
            "temperature",
            "seed",
            "enable_thinking",
            "evidence_layout",
        },
        "judge policy judge",
    )
    if judge["structured_output"] is not True:
        raise JudgePolicyError("formal visual policy requires structured_output=true")
    if judge["structured_output_method"] != "json_schema_response_format":
        raise JudgePolicyError(
            "formal visual policy requires JSON-schema response format"
        )
    if judge["enable_thinking"] is not False:
        raise JudgePolicyError("formal visual policy requires enable_thinking=false")
    if judge["evidence_layout"] != "channel_major_blocks":
        raise JudgePolicyError(
            "formal visual policy requires channel-major evidence blocks"
        )
    max_tokens = _positive_int(judge["max_tokens"], "judge.max_tokens")
    judge_timeout_s = _positive_float(judge["timeout_s"], "judge.timeout_s")
    temperature = _nonnegative_float(judge["temperature"], "judge.temperature")
    seed = _nonnegative_int(judge["seed"], "judge.seed")
    declared_judge = {
        "backend": str(judge["backend"]),
        "model": str(judge["model"]),
        "structured_output": True,
        "max_tokens": max_tokens,
        "timeout_s": judge_timeout_s,
        "temperature": temperature,
        "seed": seed,
        "enable_thinking": False,
    }
    implementation_judge = {
        "backend": model_config.BACKEND,
        "model": model_config.MODEL,
        "structured_output": model_config.STRUCTURED_OUTPUT,
        "max_tokens": model_config.MAX_TOKENS,
        "timeout_s": model_config.TIMEOUT_S,
        "temperature": model_config.TEMPERATURE,
        "seed": model_config.SEED,
        "enable_thinking": model_config.ENABLE_THINKING,
    }
    if declared_judge != implementation_judge:
        raise JudgePolicyError(
            f"frozen judge deployment {declared_judge} does not match the "
            f"runtime deployment {implementation_judge}"
        )

    render_spec = _mapping(data["render"], "judge policy render")
    _exact_keys(
        render_spec,
        {
            "protocol",
            "lighting_normalization",
            "exposure_normalization",
            "scene_alignment",
            "camera_plan_policy",
            "frame_quality_policy",
            "base_color_encoding",
            "view_count",
            "minimum_complete_viewpoints",
            "width",
            "height",
            "fov_deg",
            "radius_half_extent_multiplier",
            "height_radius_multiplier",
            "channels",
            "timeout_s",
            "depth_visualization",
        },
        "judge policy render",
    )
    expected_identifiers = {
        "protocol": render.GT_PAIRED_ENVIRONMENT_SCENE_GRAPH_PROTOCOL,
        "lighting_normalization": render.PAIRED_PHOTOMETRY_LIGHTING_POLICY,
        "exposure_normalization": render.PAIRED_PHOTOMETRY_EXPOSURE_POLICY,
        "scene_alignment": "gt-scene-graph-environment-routed-frozen",
        "camera_plan_policy": scene_graph_capture.GT_VISUAL_CAMERA_PLAN,
        "frame_quality_policy": scene_graph_capture.GT_VISUAL_FRAME_QUALITY,
        "base_color_encoding": "rgb8-opaque-png",
    }
    declared_identifiers = {
        key: str(render_spec[key]) for key in expected_identifiers
    }
    if declared_identifiers != expected_identifiers:
        raise JudgePolicyError(
            "canonical render identifiers do not match the current implementation"
        )
    channels = tuple(str(value) for value in render_spec["channels"])
    if channels != rubric.channels:
        raise JudgePolicyError(
            f"policy channels {channels} do not equal rubric channels {rubric.channels}"
        )
    view_count = _positive_int(render_spec["view_count"], "render.view_count")
    minimum_views = _positive_int(
        render_spec["minimum_complete_viewpoints"],
        "render.minimum_complete_viewpoints",
    )
    if minimum_views > view_count:
        raise JudgePolicyError("minimum_complete_viewpoints cannot exceed view_count")
    width = _positive_int(render_spec["width"], "render.width")
    height = _positive_int(render_spec["height"], "render.height")
    fov_deg = _positive_float(render_spec["fov_deg"], "render.fov_deg")
    radius_multiplier = _positive_float(
        render_spec["radius_half_extent_multiplier"],
        "render.radius_half_extent_multiplier",
    )
    height_multiplier = _positive_float(
        render_spec["height_radius_multiplier"],
        "render.height_radius_multiplier",
    )
    declared_geometry = {
        "view_count": view_count,
        "width": width,
        "height": height,
        "fov_deg": fov_deg,
        "radius_half_extent_multiplier": radius_multiplier,
        "height_radius_multiplier": height_multiplier,
    }
    implementation_geometry = {
        "view_count": render.DEFAULT_VIEWS,
        "width": render.DEFAULT_WIDTH,
        "height": render.DEFAULT_HEIGHT,
        "fov_deg": render.DEFAULT_FOV_DEG,
        "radius_half_extent_multiplier": render.RADIUS_HALF_EXTENT_MULTIPLIER,
        "height_radius_multiplier": render.HEIGHT_RADIUS_MULTIPLIER,
    }
    if declared_geometry != implementation_geometry:
        raise JudgePolicyError(
            f"frozen render geometry {declared_geometry} does not match "
            f"implementation {implementation_geometry}"
        )

    depth = _mapping(
        render_spec["depth_visualization"],
        "judge policy depth_visualization",
    )
    _exact_keys(
        depth,
        {"policy", "near_cm", "far_cm"},
        "judge policy depth_visualization",
    )
    if depth["policy"] != "linear-fixed-camera-space-cm":
        raise JudgePolicyError("unsupported depth visualization policy")
    near_cm = float(depth["near_cm"])
    far_cm = float(depth["far_cm"])
    if near_cm < 0 or far_cm <= near_cm:
        raise JudgePolicyError("depth visualization needs 0 <= near_cm < far_cm")

    identifier = str(data["id"]).strip()
    model = str(judge["model"]).strip()
    calibration_status = str(data["calibration_status"]).strip()
    if not identifier or not model or not calibration_status:
        raise JudgePolicyError(
            "policy id, judge model, and calibration_status are required"
        )
    return JudgePolicy(
        id=identifier,
        path=policy_path,
        rubric=rubric,
        rubric_path=rubric_path,
        mode="gt_paired",
        aggregation=aggregation,
        backend=str(judge["backend"]),
        # Transport only: the serving location is user-supplied at run time
        # and recorded in provenance; it is not part of the frozen policy.
        base_url=model_config.base_url(),
        model=model,
        max_tokens=max_tokens,
        judge_timeout_s=judge_timeout_s,
        temperature=temperature,
        seed=seed,
        enable_thinking=False,
        structured_output=True,
        structured_output_method=str(judge["structured_output_method"]),
        evidence_layout=str(judge["evidence_layout"]),
        render_protocol=str(render_spec["protocol"]),
        lighting_policy=str(render_spec["lighting_normalization"]),
        base_color_encoding=str(render_spec["base_color_encoding"]),
        exposure_policy=str(render_spec["exposure_normalization"]),
        scene_alignment=str(render_spec["scene_alignment"]),
        camera_plan_policy=str(render_spec["camera_plan_policy"]),
        frame_quality_policy=str(render_spec["frame_quality_policy"]),
        view_count=view_count,
        minimum_complete_viewpoints=minimum_views,
        width=width,
        height=height,
        fov_deg=fov_deg,
        radius_half_extent_multiplier=radius_multiplier,
        height_radius_multiplier=height_multiplier,
        channels=channels,
        timeout_s=_positive_float(render_spec["timeout_s"], "render.timeout_s"),
        depth_policy=str(depth["policy"]),
        depth_near_cm=near_cm,
        depth_far_cm=far_cm,
        calibration_status=calibration_status,
    )


__all__ = [
    "JudgePolicy",
    "JudgePolicyError",
    "canonical_policy_path",
    "load_policy",
]
