"""Paired renders reach the pixel metrics, and a missing pair says what is missing.

`RenderSet` addresses renders by scene and camera, and the paired pixel metrics
need one. When it is absent, empty, or holds only one side of a pair, the
refusal names that case rather than claiming that no renders were captured.
"""

from __future__ import annotations

import pytest

from code4scene.evaluation import paired_views
from code4scene.evaluation.render import RenderSet


def set_with(**scenes) -> RenderSet:
    """A render set holding the named scenes, each with the given views."""
    renders = RenderSet()
    for scene, views in scenes.items():
        for view in views:
            renders.add(scene, view, "rgb", f"/runs/ep/{scene}/{view}.png")
    return renders


# ── each absence names itself ─────────────────────────────────────────────


def test_no_render_set_at_all_says_so_rather_than_blaming_the_camera():
    with pytest.raises(paired_views.PairingError) as raised:
        paired_views.pair(None)
    message = str(raised.value)
    assert "no render set" in message
    assert "captured" not in message, (
        "this is the wiring case: claiming nothing was photographed sends the "
        "reader to the render path, which is where an hour went")


def test_an_empty_set_is_a_different_sentence():
    with pytest.raises(paired_views.PairingError) as raised:
        paired_views.pair(RenderSet())
    assert "empty" in str(raised.value)


def test_a_candidate_with_no_reference_names_the_scoring_environment():
    """The common real case, and the one worth being precise about.

    Only the scoring environment mounts the canonical content, so a missing
    reference is a deployment fact and not a bad scene.
    """
    with pytest.raises(paired_views.PairingError) as raised:
        paired_views.pair(set_with(candidate=["view_0", "view_1"]))
    assert "scoring environment" in str(raised.value)


def test_a_reference_with_no_candidate_is_not_mistaken_for_one():
    with pytest.raises(paired_views.PairingError) as raised:
        paired_views.pair(set_with(gt=["view_0"]))
    assert "candidate" in str(raised.value)


# ── and a complete set pairs by camera ────────────────────────────────────


def test_the_same_camera_in_both_scenes_is_what_pairs():
    pairing = paired_views.pair(set_with(candidate=["view_0", "view_1"],
                                         gt=["view_0", "view_1"]))
    assert pairing.reference_scene == "gt"
    assert [view for view, _, _ in pairing.pairs] == ["view_0", "view_1"]
    assert not pairing.dropped


def test_a_camera_only_one_side_has_is_dropped_and_recorded():
    """Dropped, not silently averaged over: a metric computed across a
    different number of cameras than the run took is a different metric."""
    pairing = paired_views.pair(set_with(candidate=["view_0", "view_1"],
                                         gt=["view_0"]))
    assert [view for view, _, _ in pairing.pairs] == ["view_0"]
    assert [d["view"] for d in pairing.dropped] == ["view_1"]




