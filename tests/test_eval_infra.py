"""The seams an image-comparison eval plugs into.

No verifier here, and no metric. What is tested is that the scorer can HAND
one what it needs — a clean editor, the
same cameras across scenes, renders addressable pairwise, and a place for
artifacts to meet.
"""

from __future__ import annotations

import pytest

from code4scene.evaluation import render, verifiers


# ── a clean editor the agent never drove ──────────────────────────────────

def test_a_verifier_can_be_given_a_second_editor():
    """The canonical scene cannot live in the agent's instance: arbitrary code
    in the agent's editor gets past any path filter. So the boundary is a
    different instance, and a verifier has to be able to reach it."""
    seen = {}

    def spy(context):
        seen["scoring"] = context.scoring
        seen["artifacts"] = context.artifacts_dir
        return {"report_id": "candidate_integrity", "status": "pass", "score": None,
                "task_bundle_id": "b", "episode_id": "e", "metrics": {},
                "evidence": {}, "artifacts": {}, "probes_used": ()}

    class Task:
        # The public registry is deliberately closed: exercise the first
        # canonical automatic policy rather than injecting an undeclared ID.
        verifiers = []
        id = "t"
        path = None
        kind = "scene_generation"
        data = {"kind": kind, "inputs": {}}

    original = verifiers.REGISTRY["candidate_integrity"]
    verifiers.REGISTRY["candidate_integrity"] = spy
    try:
        verifiers.run(Task(), {}, {"task_bundle_id": "b", "episode_id": "e"},
                      scoring="CLEAN-EDITOR", artifacts_dir="/artifacts")
    finally:
        verifiers.REGISTRY["candidate_integrity"] = original

    assert seen["scoring"] == "CLEAN-EDITOR"
    assert seen["artifacts"] == "/artifacts"


def test_no_scoring_environment_is_a_reportable_state_not_a_crash():
    """A dev run has one editor. A verifier that needs two must be able to say
    so rather than fail on an attribute."""
    context = verifiers.Context(record={}, task=None, ids={})
    assert context.scoring is None and context.renders is None


# ── the same camera, several scenes ───────────────────────────────────────

def test_cameras_can_be_supplied_rather_than_derived_per_scene():
    """Cameras computed separately per scene would differ by framing as well
    as by content, and the comparison would measure both."""
    import inspect

    signature = inspect.signature(render.capture)
    assert "views" in signature.parameters
    assert "channels" in signature.parameters


def test_only_the_implemented_channel_vocabulary_is_accepted():
    """RGB, Base Color and raw Scene Depth have real producers; aliases and
    unknown channels are refused rather than silently returning colour."""
    assert render.check_channels(render.CHANNELS) == render.CHANNELS

    with pytest.raises(render.RenderError, match="unknown render channel"):
        render.check_channels(["rgb", "depth"])
    with pytest.raises(render.RenderError, match="unknown render channel"):
        render.check_channels(["ultraviolet"])


def test_the_named_channels_cover_what_the_eval_asks_for():
    assert set(render.CHANNELS) == {"rgb", "scene_depth", "base_color"}


# ── renders that can be compared pairwise ─────────────────────────────────

def test_renders_are_addressable_by_scene_camera_and_channel():
    """A flat list cannot say which image pairs with which, and every
    comparison in this eval is a pair taken from one camera."""
    shots = render.RenderSet()
    for scene in ("canonical", "corrupted", "candidate"):
        shots.add(scene, "actor_17", "rgb", f"/artifacts/{scene}/actor_17.png")
    shots.add("candidate", "actor_17", "depth", "/artifacts/candidate/d.png")

    assert shots.scenes() == ["candidate", "canonical", "corrupted"]

    pair = shots.paired("actor_17", "rgb")
    assert set(pair) == {"canonical", "corrupted", "candidate"}
    assert pair["canonical"].endswith("canonical/actor_17.png")


def test_a_camera_missing_from_one_scene_is_simply_absent():
    """Three renders are wanted and two may be what exists; the comparison has
    to be able to tell, not to receive a path that was never written."""
    shots = render.RenderSet()
    shots.add("canonical", "actor_1", "rgb", "/a.png")
    shots.add("candidate", "actor_1", "rgb", "/b.png")

    pair = shots.paired("actor_1", "rgb")
    assert set(pair) == {"canonical", "candidate"}
    assert shots.paired("actor_1", "depth") == {}
