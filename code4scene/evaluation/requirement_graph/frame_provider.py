"""Code4Scene-owned RGB provider for merged Stage 2 and Stage 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np
from PIL import Image

from code4scene.evaluation import render
from code4scene.evaluation.context import Context

from .actor_inventory import ActorInventorySnapshot
from .camera_visibility import resolve_visibility_camera_poses
from .contracts import CameraPose, SceneBounds
from .evidence_adapter import SceneInventoryEvidence
from .runtime import CapturedFrame


class SceneBenchmarkFrameProvider:
    """Borrow one live editor; never enter, reset, or close its episode."""

    def __init__(
        self,
        bridge: Any,
        out_dir: Path,
        scene: SceneInventoryEvidence,
        *,
        width: int = 1280,
        height: int = 720,
        timeout_s: float = 300.0,
        lighting_policy: str | None = None,
    ) -> None:
        if bridge is None:
            raise ValueError("a live editor bridge is required for visual capture")
        self.bridge = bridge
        self.out_dir = Path(out_dir)
        self.scene = scene
        self.width = int(width)
        self.height = int(height)
        self.timeout_s = float(timeout_s)
        self.lighting_policy = lighting_policy
        self._counter = 0
        self._capture_overrides: list[dict[str, Any]] = []
        self._camera_visibility_audits: list[dict[str, Any]] = []

    def __enter__(self) -> SceneBenchmarkFrameProvider:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def capture(
        self,
        poses: Sequence[CameraPose],
        *,
        phase: str = "capture",
        frame_id_prefix: str = "frame",
    ) -> list[CapturedFrame]:
        values = tuple(poses)
        if any(not isinstance(value, CameraPose) for value in values):
            raise TypeError("poses must contain CameraPose values")
        if not values:
            return []
        views: list[render.Viewpoint] = []
        frame_ids: list[str] = []
        for pose in values:
            self._counter += 1
            frame_id = f"{frame_id_prefix}_{self._counter:04d}"
            frame_ids.append(frame_id)
            views.append(
                render.Viewpoint(
                    name=frame_id,
                    location=list(pose.location_cm),
                    # Viewpoint follows Unreal's positional Rotator order.
                    rotation=[pose.roll, pose.pitch, pose.yaw],
                )
            )
        target = self.out_dir / phase
        paths = render.capture(
            self.bridge,
            target,
            views=views,
            channels=("rgb",),
            width=self.width,
            height=self.height,
            timeout=self.timeout_s,
            retain_last_camera=True,
            lighting_policy=self.lighting_policy,
            capture_overrides=self._capture_overrides,
        )
        by_name = {Path(value).stem: Path(value) for value in paths}
        captured: list[CapturedFrame] = []
        for frame_id, pose in zip(frame_ids, values, strict=True):
            path = by_name.get(frame_id)
            if path is None:
                continue
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            captured.append(
                CapturedFrame(
                    frame_id=frame_id,
                    pose=pose,
                    rgb=rgb,
                    metadata={
                        "path": str(path),
                        "provider": "scenebench_render",
                        "capture_override": (
                            self._capture_overrides[-1]
                            if self._capture_overrides else None
                        ),
                    },
                    phase=phase,
                )
            )
        return captured

    def resolve_camera_poses(
        self,
        poses: Sequence[CameraPose],
        *,
        actor_ids_by_pose: Sequence[Sequence[str]],
    ) -> tuple[CameraPose | None, ...]:
        """Choose line-of-sight Actor cameras before any RGB is rendered."""

        results = resolve_visibility_camera_poses(
            self.bridge,
            self.scene.inventory,
            self.scene.scene_bounds,
            poses,
            actor_ids_by_pose,
            timeout_s=self.timeout_s,
        )
        self._camera_visibility_audits.extend(
            value.to_dict() for value in results
        )
        return tuple(value.pose for value in results)

    @property
    def camera_visibility_audits(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(value) for value in self._camera_visibility_audits)

    def get_scene_bounds(
        self, explicit_bounds: SceneBounds | None = None
    ) -> SceneBounds:
        return explicit_bounds or self.scene.scene_bounds

    def get_actor_inventory(
        self,
        *,
        closed_categories: Sequence[str] = (),
        canonical_identity_closed_categories: Sequence[str] = (),
        scene_bounds: SceneBounds | None = None,
        assume_loaded_world_complete: bool = False,
    ) -> ActorInventorySnapshot:
        del closed_categories, canonical_identity_closed_categories
        del scene_bounds, assume_loaded_world_complete
        return self.scene.inventory

    def close(self) -> None:
        """The provider borrows the bridge; its owner closes the episode."""
        render.remove_shot_cameras(self.bridge, timeout=self.timeout_s)


def frame_provider_from_context(
    context: Context,
    scene: SceneInventoryEvidence,
    out_dir: Path,
) -> SceneBenchmarkFrameProvider:
    if context.scoring is not None:
        bridge = context.scoring.bridge
    else:
        if context.spec.get("require_independent", False):
            raise ValueError(
                "require_independent visual verification needs a scoring editor"
            )
        bridge = context.bridge
    return SceneBenchmarkFrameProvider(
        bridge,
        out_dir,
        scene,
        width=int(context.spec.get("visual_width") or 1280),
        height=int(context.spec.get("visual_height") or 720),
        timeout_s=float(context.spec.get("visual_timeout_s") or 300.0),
        lighting_policy=render.LIGHTING_NORMALIZATION_POLICY,
    )


def reusable_overview_frames_from_context(
    context: Context,
) -> tuple[CapturedFrame, ...]:
    """Load already-frozen candidate RGB renders as Stage 2 observations.

    Prefer the exact camera specs stored beside scene-graph-planned renders.
    Legacy ring renders fall back to :func:`render.views_for`. Missing,
    custom-named, or unreadable images are skipped; Stage 2 will then acquire
    only the overview frames that are still missing.
    """

    renders = getattr(context, "visual_renders", None)
    images = getattr(renders, "images", {})
    candidate = images.get("candidate", {}) if isinstance(images, dict) else {}
    if not isinstance(candidate, dict) or not candidate:
        return ()
    names = sorted(
        (
            name
            for name, channels in candidate.items()
            if isinstance(channels, dict) and channels.get("rgb")
        ),
        key=lambda value: (
            int(value.rsplit("_", 1)[-1])
            if value.startswith("view_") and value.rsplit("_", 1)[-1].isdigit()
            else 1_000_000,
            value,
        ),
    )
    if not names or any(not name.startswith("view_") for name in names):
        return ()
    camera_specs = getattr(renders, "camera_specs", {})
    candidate_specs = (
        camera_specs.get("candidate", {})
        if isinstance(camera_specs, dict)
        else {}
    )
    anchors = getattr(renders, "scene_anchors_cm", {})
    anchor = (
        anchors.get("candidate", (0.0, 0.0, 0.0))
        if isinstance(anchors, dict)
        else (0.0, 0.0, 0.0)
    )
    if not isinstance(anchor, (list, tuple)) or len(anchor) != 3:
        anchor = (0.0, 0.0, 0.0)
    fallback = {value.name: value for value in render.views_for(
        getattr(context.task, "half_extent_m", None), count=len(names)
    )}
    frames: list[CapturedFrame] = []
    for name in names:
        spec = (
            candidate_specs.get(name, {})
            if isinstance(candidate_specs, dict)
            else {}
        )
        relative = spec.get("relative_location_cm") if isinstance(spec, dict) else None
        rotation = spec.get("rotation_deg") if isinstance(spec, dict) else None
        stored_pose_used = False
        if (
            isinstance(relative, (list, tuple))
            and len(relative) == 3
            and isinstance(rotation, (list, tuple))
            and len(rotation) == 3
        ):
            try:
                location = tuple(
                    float(anchor[index]) + float(relative[index])
                    for index in range(3)
                )
                orientation = tuple(float(value) for value in rotation)
            except (TypeError, ValueError):
                pass
            else:
                stored_pose_used = True
        if not stored_pose_used:
            viewpoint = fallback.get(name)
            if viewpoint is None:
                continue
            location = tuple(float(value) for value in viewpoint.location)
            orientation = tuple(float(value) for value in viewpoint.rotation)
        path = Path(str(candidate[name]["rgb"]))
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        try:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        except (OSError, ValueError):
            continue
        frames.append(
            CapturedFrame(
                frame_id=f"formal_{name}",
                pose=CameraPose(
                    x=location[0],
                    y=location[1],
                    z=location[2],
                    pitch=orientation[1],
                    yaw=orientation[2],
                    roll=orientation[0],
                ),
                rgb=rgb,
                metadata={
                    "path": str(path),
                    "provider": "formal_visual_render_reuse",
                    "view": name,
                    "camera_spec_source": (
                        "render_set" if stored_pose_used else "legacy_ring"
                    ),
                },
                phase="formal_overview_reuse",
            )
        )
    return tuple(frames)


__all__ = [
    "SceneBenchmarkFrameProvider",
    "frame_provider_from_context",
    "reusable_overview_frames_from_context",
]
