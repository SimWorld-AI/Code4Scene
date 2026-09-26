from __future__ import annotations

import numpy as np
import pytest

from code4scene.evaluation.requirement_graph.stage2_frames import (
    FrameQualityError,
    FrameRejectionReason,
    validate_frame_quality,
)


def test_smooth_normally_exposed_wall_is_rejected_as_low_detail():
    ramp = np.linspace(82, 118, 320, dtype=np.uint8)
    gray = np.broadcast_to(ramp[None, :, None], (180, 320, 3)).copy()

    with pytest.raises(FrameQualityError) as caught:
        validate_frame_quality(gray)

    assert caught.value.reason is FrameRejectionReason.LOW_SPATIAL_DETAIL
    assert caught.value.metrics["p999_spatial_gradient"] < 6.0
    assert caught.value.metrics["maximum_spatial_gradient"] < 12.0


def test_small_hard_silhouette_remains_valid_visual_evidence():
    rgb = np.full((180, 320, 3), 96, dtype=np.uint8)
    rgb[70:110, 145:175] = (220, 40, 40)

    _, quality = validate_frame_quality(rgb)

    assert quality.maximum_spatial_gradient >= 12.0
    assert quality.to_dict()["p999_spatial_gradient"] >= 0.0
