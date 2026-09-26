"""Prepare addressed candidate-only render evidence for a vision judge.

RGB and Base Color are emitted as PNGs, but UE Base Color captures may retain
useful RGB under an all-zero alpha channel. A vision provider commonly
composites that image onto white and therefore sees a blank frame. The frozen
opaque-RGB policy drops alpha without compositing and keeps the raw capture as
the source artifact. Scene Depth remains floating-point EXR and is converted
to a deterministic PNG solely for visual judging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ... import image_metrics
from .judge import EvidenceImage, JudgeError
from .rubric import Rubric


@dataclass(frozen=True)
class EvidenceBundle:
    images: tuple[EvidenceImage, ...]
    base_color_normalization: dict[str, Any]
    depth_visualization: dict[str, Any]
    dropped_views: dict[str, list[str]]

@dataclass(frozen=True)
class PairedEvidenceBundle:
    images_by_view: dict[str, tuple[EvidenceImage, ...]]
    paired_images: dict[str, dict[str, dict[str, str]]]
    base_color_normalization: dict[str, Any]
    depth_visualization: dict[str, Any]



def normalize_base_color(source: Path, target: Path, *, policy: str) -> dict[str, Any]:
    """Preserve hidden RGB while removing UE's fully transparent alpha."""

    if policy != "rgb8-opaque-png":
        raise JudgeError(f"unsupported Base Color encoding policy {policy!r}")
    try:
        from PIL import Image
    except ImportError as error:               # pragma: no cover - environment
        raise JudgeError(
            "Base Color normalization needs the optional imaging extra: "
            "pip install code4scene"
        ) from error
    try:
        with Image.open(source) as image:
            source_mode = image.mode
            alpha_extrema = (
                list(image.getchannel("A").getextrema())
                if "A" in image.getbands()
                else None
            )
            rgb = image.convert("RGB")
            target.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(target)
    except (OSError, ValueError) as error:
        raise JudgeError(
            f"cannot normalize Base Color evidence {source}: {error}"
        ) from error
    return {
        "source_png": str(source),
        "normalized_png": str(target),
        "policy": policy,
        "source_mode": source_mode,
        "source_alpha_extrema": alpha_extrema,
        "output_mode": "RGB",
    }


def visualize_depth(source: Path, target: Path, *, near_cm: float,
                    far_cm: float) -> dict[str, Any]:
    """Write a fixed-linear-range depth PNG without changing pixel alignment.

    White is ``near_cm``, black is ``far_cm`` or farther, and magenta marks a
    NaN/negative depth. Unlike per-image min/max normalisation, the same world
    distance has the same tone in every candidate and every camera.
    """
    if near_cm < 0 or far_cm <= near_cm:
        raise JudgeError(
            f"invalid depth visualization range {near_cm:g}..{far_cm:g} cm")
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:               # pragma: no cover - environment
        raise JudgeError(
            "three-channel visual judging needs the optional imaging extra "
            "to convert Scene Depth EXR to PNG: pip install "
            "'scenebench[images]'") from error

    try:
        depth = image_metrics.load_depth(source)[:, :, 0]
    except image_metrics.ImageError as error:
        raise JudgeError(str(error)) from error
    valid = np.isfinite(depth) & (depth >= 0)
    safe_depth = np.where(valid, depth, far_cm)
    clipped = np.clip(safe_depth, near_cm, far_cm)
    intensity = np.rint(
        (1.0 - (clipped - near_cm) / (far_cm - near_cm)) * 255.0
    ).astype(np.uint8)
    rgb = np.repeat(intensity[:, :, None], 3, axis=2)
    rgb[~valid] = np.array([255, 0, 255], dtype=np.uint8)

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(rgb, mode="RGB").save(target)
    except (OSError, ValueError) as error:
        raise JudgeError(f"cannot write depth visualization {target}: {error}") from error

    valid_depth = depth[valid]
    return {
        "source_exr": str(source),
        "visualization_png": str(target),
        "normalization": "linear-fixed-camera-space-cm",
        "near_cm": float(near_cm),
        "far_cm": float(far_cm),
        "valid_fraction": round(float(valid.mean()), 6),
        "far_clipped_fraction": round(float((valid & (depth >= far_cm)).mean()), 6),
        "observed_min_cm": (round(float(valid_depth.min()), 3)
                            if valid_depth.size else None),
        "observed_max_cm": (round(float(valid_depth.max()), 3)
                            if valid_depth.size else None),
    }


def collect(renders: Any, rubric: Rubric, out_dir: Path, *,
            near_cm: float, far_cm: float,
            base_color_encoding: str | None = None,
            minimum_complete_viewpoints: int = 2,
            scene: str = "candidate") -> EvidenceBundle:
    """Collect complete same-view channel groups from a candidate RenderSet."""
    scene_images = getattr(renders, "images", {}).get(scene, {})
    if not scene_images:
        raise JudgeError(
            f"no addressed renders were captured for scene {scene!r}")

    expected = set(rubric.channels)
    complete: list[str] = []
    dropped: dict[str, list[str]] = {}
    for view, channels in sorted(scene_images.items()):
        missing = [channel for channel in rubric.channels
                   if channel not in channels
                   or not Path(channels[channel]).is_file()
                   or Path(channels[channel]).stat().st_size <= 0]
        unexpected = sorted(set(channels) - expected)
        if missing:
            dropped[view] = [f"missing:{channel}" for channel in missing]
        elif unexpected:
            # Extra artifacts are harmless and are not sent to the model, but
            # recording them here makes the evidence selection auditable.
            dropped[view] = [f"unused:{channel}" for channel in unexpected]
            complete.append(view)
        else:
            complete.append(view)

    if minimum_complete_viewpoints < 2:
        raise JudgeError("minimum_complete_viewpoints must be at least 2")
    if len(complete) < minimum_complete_viewpoints:
        details = "; ".join(
            f"{view} ({', '.join(reasons)})" for view, reasons in dropped.items())
        raise JudgeError(
            f"only {len(complete)} complete multi-channel viewpoint(s) were "
            f"captured; at least {minimum_complete_viewpoints} are required"
            + (f": {details}" if details else ""))

    evidence: list[EvidenceImage] = []
    base_color_records: dict[str, Any] = {}
    depth_records: dict[str, Any] = {}
    for view in complete:
        channels = scene_images[view]
        for channel in rubric.channels:
            source = Path(channels[channel])
            description = ""
            path = source
            if channel == "rgb":
                description = "lit RGB appearance"
            elif channel == "base_color":
                if base_color_encoding is not None:
                    path = out_dir / "base_color" / f"{view}.png"
                    base_color_records[view] = normalize_base_color(
                        source, path, policy=base_color_encoding,
                    )
                description = (
                    "unlit Base Color in opaque RGB; no lighting or shadow evidence"
                    if base_color_encoding is not None
                    else "unlit Base Color; no lighting or shadow evidence"
                )
            elif channel == "scene_depth":
                path = out_dir / "scene_depth" / f"{view}.png"
                depth_records[view] = visualize_depth(
                    source, path, near_cm=near_cm, far_cm=far_cm)
                description = (
                    f"Scene Depth in camera-space cm, fixed linear range "
                    f"{near_cm:g}..{far_cm:g}; white near, black far, "
                    "magenta invalid")
            evidence.append(EvidenceImage(
                path=path, view=view, channel=channel,
                description=description))

    return EvidenceBundle(
        images=tuple(evidence),
        base_color_normalization={
            "policy": base_color_encoding,
            "views": base_color_records,
        },
        depth_visualization={
            "policy": "linear-fixed-camera-space-cm",
            "near_cm": float(near_cm), "far_cm": float(far_cm),
            "views": depth_records,
        },
        dropped_views=dropped,
    )


def _pair_board(reference: Path, candidate: Path, target: Path, *,
                channel: str) -> None:
    """Write a labelled, pixel-preserving GT/candidate board for one channel."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:                 # pragma: no cover - environment
        raise JudgeError(
            "paired visual evidence needs Pillow") from error
    try:
        with Image.open(reference) as gt_source:
            gt = gt_source.convert("RGB")
        with Image.open(candidate) as candidate_source:
            rendered = candidate_source.convert("RGB")
        if gt.size != rendered.size:
            raise JudgeError(
                f"paired {channel} images differ in size: {gt.size} vs "
                f"{rendered.size}")
        header = 28
        board = Image.new("RGB", (gt.width * 2, gt.height + header), "black")
        board.paste(gt, (0, header))
        board.paste(rendered, (gt.width, header))
        draw = ImageDraw.Draw(board)
        draw.text((8, 7), f"GT — {channel}", fill="white")
        draw.text((gt.width + 8, 7), f"CANDIDATE — {channel}", fill="white")
        target.parent.mkdir(parents=True, exist_ok=True)
        board.save(target)
    except JudgeError:
        raise
    except (OSError, ValueError) as error:
        raise JudgeError(
            f"cannot create paired evidence board {target}: {error}") from error


def collect_paired(
    renders: Any,
    pairing: Any,
    rubric: Rubric,
    out_dir: Path,
    *,
    near_cm: float,
    far_cm: float,
    base_color_encoding: str | None,
) -> PairedEvidenceBundle:
    """Normalize and pair every strict GT/candidate view and channel."""
    reference = collect(
        renders,
        rubric,
        out_dir / "gt",
        near_cm=near_cm,
        far_cm=far_cm,
        base_color_encoding=base_color_encoding,
        minimum_complete_viewpoints=len(pairing.views),
        scene=pairing.reference_scene,
    )
    candidate = collect(
        renders,
        rubric,
        out_dir / "candidate",
        near_cm=near_cm,
        far_cm=far_cm,
        base_color_encoding=base_color_encoding,
        minimum_complete_viewpoints=len(pairing.views),
        scene="candidate",
    )
    addressed: dict[tuple[str, str, str], EvidenceImage] = {}
    for role, bundle in (("gt", reference), ("candidate", candidate)):
        for item in bundle.images:
            addressed[(role, item.view, item.channel)] = EvidenceImage(
                path=item.path,
                view=item.view,
                channel=item.channel,
                description=item.description,
                scene=role,
            )

    images_by_view: dict[str, tuple[EvidenceImage, ...]] = {}
    paired_images: dict[str, dict[str, dict[str, str]]] = {}
    for view in pairing.views:
        view_evidence = []
        channel_records = {}
        for channel in rubric.channels:
            gt = addressed[("gt", view, channel)]
            rendered = addressed[("candidate", view, channel)]
            view_evidence.extend((gt, rendered))
            board = out_dir / "paired" / view / f"{channel}.png"
            _pair_board(gt.path, rendered.path, board, channel=channel)
            channel_records[channel] = {
                "gt": str(gt.path),
                "candidate": str(rendered.path),
                "paired_image": str(board),
            }
        images_by_view[view] = tuple(view_evidence)
        paired_images[view] = channel_records

    return PairedEvidenceBundle(
        images_by_view=images_by_view,
        paired_images=paired_images,
        base_color_normalization={
            "policy": base_color_encoding,
            "gt": reference.base_color_normalization,
            "candidate": candidate.base_color_normalization,
        },
        depth_visualization={
            "policy": "linear-fixed-camera-space-cm",
            "near_cm": float(near_cm),
            "far_cm": float(far_cm),
            "gt": reference.depth_visualization,
            "candidate": candidate.depth_visualization,
        },
    )


__all__ = [
    "EvidenceBundle", "PairedEvidenceBundle", "collect", "collect_paired",
    "normalize_base_color", "visualize_depth",
]
