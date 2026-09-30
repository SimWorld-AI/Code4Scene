"""Arithmetic on two renders of the same camera.

MAE, MSE, PSNR, SSIM and the changed-pixel fraction, over a pair of images
taken from the identical viewpoint in two scenes. That last condition is the
whole basis: these numbers mean nothing between two different framings, so a
caller that cannot prove the cameras match has no business calling this.

NumPy and Pillow come with the package; OpenEXR, for Scene Depth images, is the
optional `depth` extra (`pip install -e ".[depth]"` from the repository). A harness that only runs the deterministic geometry pass should not have to
carry NumPy and Pillow, so its absence is a reported refusal rather than an
import error at start-up — `available()` says whether the arithmetic can run,
and every verifier that uses it checks first.

SSIM is the global variant — one window over the whole image — rather than the
windowed mean. It is cheaper, it is what the vendored evaluator computed, and
the difference matters: global SSIM is insensitive to where the change is,
which is exactly why `edit_region` exists to ask separately.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any

#: Two images differ at a pixel when any channel differs by more than this,
#: on a 0-255 scale. Small enough to catch a re-lit surface, large enough to
#: ignore encoder noise.
CHANGED_PIXEL_TOLERANCE = 2.0
# Runtime authored-camera admission deliberately uses the release audit's
# coarser RGB8 threshold. Unlike formal Candidate/GT scoring this gate asks
# only whether the current task's Input mutation is plainly observable from a
# proposed fixed camera; small rendering noise must not select a camera.
INPUT_GT_OBSERVABILITY_TOLERANCE = 12.0
INPUT_GT_MIN_CHANGED_FRACTION = 0.0002
INPUT_GT_STRONG_CHANGED_FRACTION = 0.001
INPUT_GT_MAX_CHANGED_FRACTION = 0.2
INPUT_GT_MIN_COMPONENT_FRACTION = 0.0002
INPUT_GT_MIN_COMPONENT_SPAN_PX = 8
INPUT_GT_MAX_COMPONENT_BBOX_FRACTION = 0.5
INPUT_GT_MIN_SSIM = 0.85
# Formal visual scoring needs a stronger signal than camera admission.  The
# latter may keep a small but real edit as a useful seed; the former must not
# let a barely visible target influence a Candidate score.  These thresholds
# deliberately reject the previously observed ~27x49-pixel / 0.14% case while
# retaining target-centred views with a plainly visible changed region.
INPUT_GT_SCORING_MIN_CHANGED_FRACTION = 0.003
INPUT_GT_SCORING_MIN_COMPONENT_FRACTION = 0.001
INPUT_GT_SCORING_MIN_COMPONENT_BBOX_FRACTION = 0.003
INPUT_GT_SCORING_MIN_LONGEST_BBOX_FRACTION = 0.08
INPUT_GT_SCORING_MAX_CENTER_OFFSET = 0.85
# A target-centred close-up can legitimately change more than the conservative
# whole-frame camera-admission ceiling above.  Admit that case only when the
# delta is dominated by one bounded coherent region; diffuse/global changes
# remain ineligible.
INPUT_GT_SCORING_LARGE_TARGET_MAX_CHANGED_FRACTION = 0.45
INPUT_GT_SCORING_LARGE_TARGET_MAX_COMPONENT_FRACTION = 0.45
INPUT_GT_SCORING_LARGE_TARGET_MAX_BBOX_FRACTION = 0.70
INPUT_GT_SCORING_LARGE_TARGET_MIN_COMPONENT_SHARE = 0.80
#: Ten centimetres is larger than floating-point/export noise and smaller than
#: a placement error visible in the benchmark scenes.
DEPTH_CHANGED_TOLERANCE_CM = 10.0
DEPTH_PSNR_DATA_RANGE_CM = 10000.0
DEPTH_EDGE_THRESHOLD_CM = 50.0
DEPTH_EDGE_CHAMFER_NORMALIZATION_PX = 32.0


class ImageError(Exception):
    """The images could not be read or compared."""


def available(channel: str = "rgb") -> bool:
    """Is the optional stack for this render channel installed."""
    try:
        import numpy            # noqa: F401
        from PIL import Image   # noqa: F401
        if channel == "scene_depth":
            import Imath        # noqa: F401
            import OpenEXR      # noqa: F401
    except ImportError:
        return False
    return True


def load(path: str | Path) -> Any:
    """One image as a float array, shape (height, width, channels), 0-255."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:                       # pragma: no cover - env
        raise ImageError(
            "comparing renders needs NumPy and Pillow: "
            "pip install -e . from the repository") from e
    try:
        with Image.open(path) as handle:
            return np.asarray(handle.convert("RGB"), dtype="float64")
    except (OSError, ValueError) as e:
        raise ImageError(f"cannot read render {path}: {e}") from e


def load_depth(path: str | Path) -> Any:
    """A SceneDepth EXR as ``(height, width, 1)`` float centimetres."""
    try:
        import Imath
        import numpy as np
        import OpenEXR
    except ImportError as error:                   # pragma: no cover - env
        raise ImageError(
            "reading scene_depth needs OpenEXR: "
            "pip install -e \".[depth]\" from the repository") from error
    try:
        handle = OpenEXR.InputFile(str(path))
        header = handle.header()
        window = header["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        channel = next((name for name in ("R", "Z", "Y")
                        if name in header.get("channels", {})), None)
        if channel is None:
            raise ImageError(f"depth EXR {path} carries no R, Z, or Y channel")
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        values = np.frombuffer(handle.channel(channel, pixel_type), dtype=np.float32)
        handle.close()
        return values.reshape((height, width, 1)).astype("float64")
    except ImageError:
        raise
    except Exception as error:                     # noqa: BLE001 — optional lib
        raise ImageError(f"cannot read scene-depth EXR {path}: {error}") from error


def compare(left: Any, right: Any, *, data_range: float = 255.0,
            changed_tolerance: float = CHANGED_PIXEL_TOLERANCE) -> dict[str, Any]:
    """Every whole-image number, for one pair of same-camera renders."""
    import numpy as np

    if left.shape != right.shape:
        raise ImageError(
            f"renders differ in size ({left.shape} vs {right.shape}); these "
            f"metrics compare the same camera in two scenes and a resize would "
            f"produce a number about the resampling")
    difference = left - right
    absolute = np.abs(difference)
    mae = float(absolute.mean())
    mse = float((difference ** 2).mean())
    changed = float((absolute.max(axis=2) > changed_tolerance).mean())
    return {"mae": round(mae, 6), "mse": round(mse, 6),
            "psnr_db": round(_psnr(mse, data_range), 4),
            "ssim": round(global_ssim(left, right, data_range=data_range), 6),
            "changed_pixel_fraction": round(changed, 6),
            "pixel_count": int(left.shape[0] * left.shape[1])}


def assess_input_gt_observability(left: Any, right: Any) -> dict[str, Any]:
    """Assess whether one frozen camera exposes a localized Input/GT edit.

    This is an admission gate for authored camera poses, not a Candidate score.
    It intentionally uses only the current task's Input and GT renders. The
    largest 8-connected robust-difference component rejects both tiny noise
    and diffuse whole-frame photometric drift while allowing framing and
    actor-identity traces to remain advisory.
    """

    import numpy as np

    if left.shape != right.shape:
        raise ImageError(
            f"renders differ in size ({left.shape} vs {right.shape}); authored "
            "camera observability requires one frozen Input/GT camera"
        )
    if left.ndim != 3 or left.shape[2] < 1:
        raise ImageError(
            "authored camera observability requires HxWxC image arrays"
        )
    height, width = int(left.shape[0]), int(left.shape[1])
    pixel_count = height * width
    robust_mask = (
        np.abs(left - right).max(axis=2) > INPUT_GT_OBSERVABILITY_TOLERANCE
    )
    changed_count = int(robust_mask.sum())
    changed_fraction = changed_count / pixel_count if pixel_count else 0.0

    visited = np.zeros((height, width), dtype=bool)
    largest_count = 0
    largest_bbox = (0, 0, 0, 0)
    ys, xs = np.nonzero(robust_mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist(), strict=True):
        if visited[start_y, start_x]:
            continue
        queue = deque(((start_y, start_x),))
        visited[start_y, start_x] = True
        component_count = 0
        min_x = max_x = start_x
        min_y = max_y = start_y
        while queue:
            y, x = queue.popleft()
            component_count += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    if (
                        not visited[neighbor_y, neighbor_x]
                        and robust_mask[neighbor_y, neighbor_x]
                    ):
                        visited[neighbor_y, neighbor_x] = True
                        queue.append((neighbor_y, neighbor_x))
        if component_count > largest_count:
            largest_count = component_count
            largest_bbox = (
                min_x,
                min_y,
                max_x - min_x + 1,
                max_y - min_y + 1,
            )

    bbox_x, bbox_y, bbox_width, bbox_height = largest_bbox
    component_fraction = largest_count / pixel_count if pixel_count else 0.0
    bbox_fraction = (
        bbox_width * bbox_height / pixel_count if pixel_count else 0.0
    )
    similarity = float(global_ssim(left, right))
    gates = {
        "weak_visibility": (
            changed_fraction >= INPUT_GT_MIN_CHANGED_FRACTION
            and component_fraction >= INPUT_GT_MIN_COMPONENT_FRACTION
            and bbox_width >= INPUT_GT_MIN_COMPONENT_SPAN_PX
            and bbox_height >= INPUT_GT_MIN_COMPONENT_SPAN_PX
        ),
        "strong_visibility": (
            changed_fraction >= INPUT_GT_STRONG_CHANGED_FRACTION
        ),
        "locality": (
            changed_fraction <= INPUT_GT_MAX_CHANGED_FRACTION
            and bbox_fraction <= INPUT_GT_MAX_COMPONENT_BBOX_FRACTION
        ),
        "camera_similarity": similarity >= INPUT_GT_MIN_SSIM,
    }
    accepted = all(
        gates[name]
        for name in ("weak_visibility", "locality", "camera_similarity")
    )
    return {
        "policy": "current-task-input-gt-robust-rgb8-observability",
        "accepted": accepted,
        "robust_delta": {
            "tolerance_rgb8": INPUT_GT_OBSERVABILITY_TOLERANCE,
            "pixel_count": changed_count,
            "fraction": round(changed_fraction, 6),
        },
        "dominant_component": {
            "pixel_count": largest_count,
            "fraction": round(component_fraction, 6),
            "bbox_xywh": [bbox_x, bbox_y, bbox_width, bbox_height],
            "bbox_area_fraction": round(bbox_fraction, 6),
        },
        "ssim": round(similarity, 6),
        "gates": gates,
        "thresholds": {
            "min_changed_fraction": INPUT_GT_MIN_CHANGED_FRACTION,
            "strong_changed_fraction": INPUT_GT_STRONG_CHANGED_FRACTION,
            "max_changed_fraction": INPUT_GT_MAX_CHANGED_FRACTION,
            "min_component_fraction": INPUT_GT_MIN_COMPONENT_FRACTION,
            "min_component_span_px": INPUT_GT_MIN_COMPONENT_SPAN_PX,
            "max_component_bbox_fraction": (
                INPUT_GT_MAX_COMPONENT_BBOX_FRACTION
            ),
            "min_ssim": INPUT_GT_MIN_SSIM,
        },
    }


def assess_input_gt_scoring_view(left: Any, right: Any) -> dict[str, Any]:
    """Certify one Candidate-independent view for formal visual scoring.

    Camera admission and score admission answer different questions.  A tiny
    Input/GT delta is enough to prove that an authored pose is relevant, but it
    is not enough to let a noisy overview vote on Candidate quality.  This
    stricter wrapper requires a substantial, coherent, reasonably centred
    Input/GT edit region.  It never reads Candidate pixels.
    """

    base = assess_input_gt_observability(left, right)
    height, width = int(left.shape[0]), int(left.shape[1])
    component = dict(base["dominant_component"])
    bbox_x, bbox_y, bbox_width, bbox_height = component["bbox_xywh"]
    center_x = (float(bbox_x) + float(bbox_width) / 2.0) / max(width, 1)
    center_y = (float(bbox_y) + float(bbox_height) / 2.0) / max(height, 1)
    center_offset = math.hypot(
        2.0 * (center_x - 0.5),
        2.0 * (center_y - 0.5),
    )
    longest_bbox_fraction = max(
        float(bbox_width) / max(width, 1),
        float(bbox_height) / max(height, 1),
    )
    robust_fraction = float(base["robust_delta"]["fraction"])
    component_fraction = float(component["fraction"])
    bbox_area_fraction = float(component["bbox_area_fraction"])
    component_share = (
        component_fraction / robust_fraction if robust_fraction > 0.0 else 0.0
    )
    standard_camera_admission = base.get("accepted") is True
    large_coherent_target_admission = (
        robust_fraction
        <= INPUT_GT_SCORING_LARGE_TARGET_MAX_CHANGED_FRACTION
        and component_fraction
        <= INPUT_GT_SCORING_LARGE_TARGET_MAX_COMPONENT_FRACTION
        and bbox_area_fraction
        <= INPUT_GT_SCORING_LARGE_TARGET_MAX_BBOX_FRACTION
        and component_share
        >= INPUT_GT_SCORING_LARGE_TARGET_MIN_COMPONENT_SHARE
    )
    camera_admission = (
        standard_camera_admission or large_coherent_target_admission
    )
    score_gates = {
        "camera_admission": camera_admission,
        "plainly_visible_delta": (
            robust_fraction >= INPUT_GT_SCORING_MIN_CHANGED_FRACTION
        ),
        "coherent_component": (
            component_fraction >= INPUT_GT_SCORING_MIN_COMPONENT_FRACTION
            and bbox_area_fraction
            >= INPUT_GT_SCORING_MIN_COMPONENT_BBOX_FRACTION
        ),
        "useful_target_scale": (
            longest_bbox_fraction
            >= INPUT_GT_SCORING_MIN_LONGEST_BBOX_FRACTION
        ),
        "target_region_focused": (
            center_offset <= INPUT_GT_SCORING_MAX_CENTER_OFFSET
        ),
    }
    return {
        **base,
        "policy": "current-task-input-gt-visual-score-view-quality",
        "accepted": all(score_gates.values()),
        "candidate_used_for_acceptance": False,
        "camera_admission_mode": (
            "standard"
            if standard_camera_admission
            else (
                "large_coherent_target_override"
                if large_coherent_target_admission
                else "rejected"
            )
        ),
        "score_gates": score_gates,
        "dominant_component": {
            **component,
            "longest_bbox_fraction": round(longest_bbox_fraction, 6),
            "center_offset": round(center_offset, 6),
            "share_of_robust_delta": round(component_share, 6),
        },
        "image_size_px": [width, height],
        "scoring_thresholds": {
            "min_changed_fraction": (
                INPUT_GT_SCORING_MIN_CHANGED_FRACTION
            ),
            "min_component_fraction": (
                INPUT_GT_SCORING_MIN_COMPONENT_FRACTION
            ),
            "min_component_bbox_fraction": (
                INPUT_GT_SCORING_MIN_COMPONENT_BBOX_FRACTION
            ),
            "min_longest_bbox_fraction": (
                INPUT_GT_SCORING_MIN_LONGEST_BBOX_FRACTION
            ),
            "max_center_offset": INPUT_GT_SCORING_MAX_CENTER_OFFSET,
            "large_target_max_changed_fraction": (
                INPUT_GT_SCORING_LARGE_TARGET_MAX_CHANGED_FRACTION
            ),
            "large_target_max_component_fraction": (
                INPUT_GT_SCORING_LARGE_TARGET_MAX_COMPONENT_FRACTION
            ),
            "large_target_max_bbox_fraction": (
                INPUT_GT_SCORING_LARGE_TARGET_MAX_BBOX_FRACTION
            ),
            "large_target_min_component_share": (
                INPUT_GT_SCORING_LARGE_TARGET_MIN_COMPONENT_SHARE
            ),
        },
    }


def _psnr(mse: float, data_range: float = 255.0) -> float:
    """Infinite for identical images, which is honest and unusable as a score."""
    if mse <= 0:
        return math.inf
    return 20 * math.log10(data_range) - 10 * math.log10(mse)


def global_ssim(left: Any, right: Any, *, data_range: float = 255.0) -> float:
    """Structural similarity over the whole image, on luminance."""
    import numpy as np

    def luminance(image: Any) -> Any:
        if image.shape[2] == 1:
            return image[:, :, 0]
        return image @ np.array([0.299, 0.587, 0.114])

    a, b = luminance(left), luminance(right)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var()), float(b.var())
    covariance = float(((a - mean_a) * (b - mean_b)).mean())
    numerator = (2 * mean_a * mean_b + c1) * (2 * covariance + c2)
    denominator = (mean_a ** 2 + mean_b ** 2 + c1) * (var_a + var_b + c2)
    return numerator / denominator if denominator else 0.0


def compare_depth(left: Any, right: Any,
                  *, maximum_valid_depth_cm: float = 6_000_000.0,
                  data_range_cm: float = DEPTH_PSNR_DATA_RANGE_CM,
                  changed_tolerance_cm: float = DEPTH_CHANGED_TOLERANCE_CM,
                  ) -> dict[str, Any]:
    """Metrics over pixels where both depth renders carry finite world depth."""
    import numpy as np

    if left.shape != right.shape:
        raise ImageError(
            f"depth renders differ in size ({left.shape} vs {right.shape})")
    valid = (np.isfinite(left[:, :, 0]) & np.isfinite(right[:, :, 0])
             & (left[:, :, 0] >= 0) & (right[:, :, 0] >= 0)
             & (left[:, :, 0] <= maximum_valid_depth_cm)
             & (right[:, :, 0] <= maximum_valid_depth_cm))
    if not valid.any():
        raise ImageError("the depth pair shares no finite in-range pixels")
    a, b = left[:, :, 0][valid], right[:, :, 0][valid]
    difference = a - b
    absolute = np.abs(difference)
    mae = float(absolute.mean())
    mse = float((difference ** 2).mean())
    # SSIM is defined over the common valid samples.  Preserve them as a
    # single-column image rather than filling invalid pixels with a value that
    # would become evidence.
    a_image = a.reshape((-1, 1, 1))
    b_image = b.reshape((-1, 1, 1))
    return {"mae_cm": round(mae, 6), "mse_cm2": round(mse, 6),
            "psnr_db": round(_psnr(mse, data_range_cm), 4),
            "ssim": round(global_ssim(
                a_image, b_image, data_range=data_range_cm), 6),
            "changed_pixel_fraction": round(float(
                (absolute > changed_tolerance_cm).mean()), 6),
            "valid_pixel_count": int(valid.sum()),
            "valid_pixel_fraction": round(float(valid.mean()), 6),
            "depth_unit": "cm"}


def binary_iou(left: Any, right: Any) -> float:
    """Intersection-over-union for two boolean pixel masks.

    Two empty masks agree perfectly. This is a measured pixel region IoU;
    callers must never substitute a VLM estimate for it.
    """
    import numpy as np

    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    if a.shape != b.shape:
        raise ImageError(f"masks differ in size ({a.shape} vs {b.shape})")
    union = np.logical_or(a, b)
    if not union.any():
        return 1.0
    return float(np.logical_and(a, b).sum() / union.sum())


def _depth_edges(depth: Any, *, maximum_valid_depth_cm: float,
                 threshold_cm: float) -> tuple[Any, Any]:
    """Finite foreground and discontinuity masks from one depth raster."""
    import numpy as np

    values = depth[:, :, 0]
    foreground = (
        np.isfinite(values) & (values >= 0) &
        (values <= maximum_valid_depth_cm)
    )
    edges = np.zeros(values.shape, dtype=bool)
    horizontal_valid = foreground[:, 1:] & foreground[:, :-1]
    vertical_valid = foreground[1:, :] & foreground[:-1, :]
    horizontal = (
        (foreground[:, 1:] != foreground[:, :-1]) |
        (horizontal_valid &
         (np.abs(values[:, 1:] - values[:, :-1]) > threshold_cm))
    )
    vertical = (
        (foreground[1:, :] != foreground[:-1, :]) |
        (vertical_valid &
         (np.abs(values[1:, :] - values[:-1, :]) > threshold_cm))
    )
    edges[:, 1:] |= horizontal
    edges[:, :-1] |= horizontal
    edges[1:, :] |= vertical
    edges[:-1, :] |= vertical
    return foreground, edges


def _l1_distance_to_true(mask: Any) -> Any:
    """City-block distance transform without an optional SciPy dependency."""
    import numpy as np

    height, width = mask.shape
    maximum = height + width + 1
    distance = np.where(mask, 0.0, float(maximum))
    for row in range(height):
        for column in range(width):
            best = distance[row, column]
            if row:
                best = min(best, distance[row - 1, column] + 1.0)
            if column:
                best = min(best, distance[row, column - 1] + 1.0)
            distance[row, column] = best
    for row in range(height - 1, -1, -1):
        for column in range(width - 1, -1, -1):
            best = distance[row, column]
            if row + 1 < height:
                best = min(best, distance[row + 1, column] + 1.0)
            if column + 1 < width:
                best = min(best, distance[row, column + 1] + 1.0)
            distance[row, column] = best
    return distance


def compare_depth_geometry(
    left: Any,
    right: Any,
    *,
    maximum_valid_depth_cm: float,
    edge_threshold_cm: float = DEPTH_EDGE_THRESHOLD_CM,
    chamfer_normalization_px: float = DEPTH_EDGE_CHAMFER_NORMALIZATION_PX,
) -> dict[str, Any]:
    """Silhouette IoU and symmetric depth-edge Chamfer for aligned cameras.

    The implementation uses deterministic L1 pixel distance and names that
    choice in its output. The raw distance remains in the report.
    """
    import numpy as np

    if left.shape != right.shape:
        raise ImageError(
            f"depth renders differ in size ({left.shape} vs {right.shape})")
    if maximum_valid_depth_cm <= 0:
        raise ImageError("maximum_valid_depth_cm must be positive")
    if edge_threshold_cm <= 0 or chamfer_normalization_px <= 0:
        raise ImageError("depth edge thresholds must be positive")

    left_foreground, left_edges = _depth_edges(
        left,
        maximum_valid_depth_cm=maximum_valid_depth_cm,
        threshold_cm=edge_threshold_cm,
    )
    right_foreground, right_edges = _depth_edges(
        right,
        maximum_valid_depth_cm=maximum_valid_depth_cm,
        threshold_cm=edge_threshold_cm,
    )
    if not left_edges.any() and not right_edges.any():
        chamfer = 0.0
    elif not left_edges.any() or not right_edges.any():
        chamfer = float(chamfer_normalization_px)
    else:
        left_distance = _l1_distance_to_true(left_edges)
        right_distance = _l1_distance_to_true(right_edges)
        chamfer = float(
            (right_distance[left_edges].mean() +
             left_distance[right_edges].mean()) / 2.0
        )
    similarity = 1.0 - min(chamfer / chamfer_normalization_px, 1.0)
    return {
        "foreground_silhouette_iou": round(
            binary_iou(left_foreground, right_foreground), 6),
        "depth_edge_chamfer_px": round(chamfer, 6),
        "depth_edge_similarity": round(similarity, 6),
        "depth_edge_distance": "symmetric-l1-pixel-chamfer",
        "depth_edge_threshold_cm": float(edge_threshold_cm),
        "depth_edge_chamfer_normalization_px": float(
            chamfer_normalization_px),
        "reference_edge_pixel_count": int(np.count_nonzero(left_edges)),
        "candidate_edge_pixel_count": int(np.count_nonzero(right_edges)),
    }


def load_mask(path: str | Path) -> Any:
    """One semantic/instance mask as an integer raster without resampling."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:                 # pragma: no cover - environment
        raise ImageError(
            "reading segmentation masks needs numpy and Pillow") from error
    try:
        with Image.open(path) as image:
            return np.asarray(image)
    except (OSError, ValueError) as error:
        raise ImageError(f"cannot read segmentation mask {path}: {error}") from error


def semantic_mask_iou(left: Any, right: Any) -> dict[str, Any]:
    """Mean class IoU over labels present in either aligned mask."""
    import numpy as np

    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        raise ImageError(f"masks differ in size ({a.shape} vs {b.shape})")
    if a.ndim == 3:
        a = a[:, :, 0]
    if b.ndim == 3:
        b = b[:, :, 0]
    labels = sorted(set(np.unique(a).tolist()) | set(np.unique(b).tolist()))
    per_label = {
        str(label): round(binary_iou(a == label, b == label), 6)
        for label in labels
    }
    return {
        "mean_iou": round(sum(per_label.values()) / len(per_label), 6)
        if per_label else 1.0,
        "per_label_iou": per_label,
        "label_count": len(labels),
    }


def masked(left: Any, right: Any, mask: Any) -> dict[str, Any]:
    """The same numbers over a subset of pixels, and over its complement."""
    import numpy as np

    if not mask.any():
        return {"pixel_count": 0, "mae": None, "changed_pixel_fraction": None}
    difference = np.abs(left - right)
    per_pixel = difference.max(axis=2)
    inside = per_pixel[mask]
    return {"pixel_count": int(mask.sum()),
            "mae": round(float(difference[mask].mean()), 6),
            "changed_pixel_fraction": round(
                float((inside > CHANGED_PIXEL_TOLERANCE).mean()), 6)}


def changed_mask(left: Any, right: Any) -> Any:
    """Where two renders differ at all — the region an edit is visible in."""
    import numpy as np

    return np.abs(left - right).max(axis=2) > CHANGED_PIXEL_TOLERANCE


__all__ = ["CHANGED_PIXEL_TOLERANCE", "DEPTH_CHANGED_TOLERANCE_CM",
           "DEPTH_EDGE_CHAMFER_NORMALIZATION_PX", "DEPTH_EDGE_THRESHOLD_CM",
           "DEPTH_PSNR_DATA_RANGE_CM", "ImageError", "available", "binary_iou",
           "changed_mask", "compare", "compare_depth", "compare_depth_geometry",
           "global_ssim", "load", "load_depth", "load_mask", "masked",
           "semantic_mask_iou"]
