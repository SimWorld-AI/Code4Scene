from __future__ import annotations

import numpy as np

from code4scene.evaluation import image_metrics


def test_input_gt_observability_accepts_localized_strong_edit():
    input_image = np.full((200, 200, 3), 128.0)
    gt_image = input_image.copy()
    gt_image[70:82, 90:102, :] = 160.0

    result = image_metrics.assess_input_gt_observability(
        input_image,
        gt_image,
    )

    assert result["accepted"] is True
    assert result["gates"] == {
        "weak_visibility": True,
        "strong_visibility": True,
        "locality": True,
        "camera_similarity": True,
    }
    assert result["robust_delta"]["fraction"] == 0.0036
    assert result["dominant_component"]["bbox_xywh"] == [90, 70, 12, 12]


def test_input_gt_observability_rejects_tiny_delta():
    input_image = np.zeros((200, 200, 3), dtype=float)
    gt_image = input_image.copy()
    gt_image[10:14, 20:24, :] = 255.0

    result = image_metrics.assess_input_gt_observability(
        input_image,
        gt_image,
    )

    assert result["accepted"] is False
    assert result["gates"]["weak_visibility"] is False
    assert result["dominant_component"]["bbox_xywh"] == [20, 10, 4, 4]


def test_input_gt_observability_rejects_diffuse_whole_frame_change():
    input_image = np.full((100, 100, 3), 80.0)
    gt_image = input_image.copy()
    gt_image[:60, :, :] = 110.0

    result = image_metrics.assess_input_gt_observability(
        input_image,
        gt_image,
    )

    assert result["accepted"] is False
    assert result["gates"]["locality"] is False
    assert result["gates"]["camera_similarity"] is False


def test_input_gt_scoring_view_rejects_barely_visible_admission_delta():
    input_image = np.full((720, 1280, 3), 96.0)
    gt_image = input_image.copy()
    gt_image[330:379, 620:647, :] = 140.0

    result = image_metrics.assess_input_gt_scoring_view(input_image, gt_image)

    assert result["gates"]["weak_visibility"] is True
    assert result["accepted"] is False
    assert result["score_gates"]["plainly_visible_delta"] is False
    assert result["score_gates"]["coherent_component"] is False
    assert result["candidate_used_for_acceptance"] is False


def test_input_gt_scoring_view_accepts_large_focused_target_delta():
    input_image = np.full((720, 1280, 3), 96.0)
    gt_image = input_image.copy()
    gt_image[270:447, 488:792, :] = 109.0

    result = image_metrics.assess_input_gt_scoring_view(input_image, gt_image)

    assert result["accepted"] is True
    assert all(result["score_gates"].values())
    assert result["dominant_component"]["bbox_xywh"] == [488, 270, 304, 177]
    assert result["dominant_component"]["center_offset"] < 0.01


def test_input_gt_scoring_view_accepts_bounded_close_up_target_delta():
    input_image = np.full((720, 1280, 3), 96.0)
    gt_image = input_image.copy()
    gt_image[160:560, 240:1040, :] = 140.0

    result = image_metrics.assess_input_gt_scoring_view(input_image, gt_image)

    assert result["gates"]["locality"] is False
    assert result["accepted"] is True
    assert result["camera_admission_mode"] == (
        "large_coherent_target_override"
    )
    assert result["dominant_component"]["share_of_robust_delta"] == 1.0


def test_input_gt_scoring_view_rejects_diffuse_majority_frame_delta():
    input_image = np.full((720, 1280, 3), 96.0)
    gt_image = input_image.copy()
    gt_image[:400, :, :] = 140.0

    result = image_metrics.assess_input_gt_scoring_view(input_image, gt_image)

    assert result["accepted"] is False
    assert result["score_gates"]["camera_admission"] is False
    assert result["camera_admission_mode"] == "rejected"
