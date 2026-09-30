"""A scene whose lighting does not work photographs as pure black.

A black frame is not evidence of anything: a judge shown one is being asked to
score an empty image. Whether a frame comes out black does not follow from the
lights a level contains (a sun pointed above the horizon lights nothing), so
the fallback keys on the symptom rather than the cause: shoot, look at what
came back, and re-shoot the black ones under a temporary rig that is removed
afterwards. `viewmode unlit` does not help, because the screenshot tool does
not photograph the viewport.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from code4scene.evaluation import render

Image = pytest.importorskip("PIL.Image", reason="the fallback reads the frames")


def _write(path: Path, level: int) -> None:
    Image.new("RGB", (64, 36), (level, level, level)).save(path)


class _Editor:
    """An editor whose scene is black until a light rig is raised in it."""

    def __init__(self, lit_level: int, unlit_level: int = 120) -> None:
        self.lit_level, self.unlit_level = lit_level, unlit_level
        self.mode = "lit"
        self.console: list[str] = []
        self.shots: list[tuple[str, tuple]] = []

    def command(self, name, args, timeout=300.0):
        if name == "take_screenshot":
            self.shots.append((Path(args["filepath"]).name,
                               tuple(args["camera_location"])))
            _write(Path(args["filepath"]),
                   self.unlit_level if self.mode == "unlit" else self.lit_level)
        return {"status": "success"}

    def exec_python(self, script, timeout=60.0):
        return {}

    def exec_python_result(self, script, key, timeout=120.0):
        if "_SB_RELIGHT" in script:
            self.mode = "unlit"                 # the rig is up: the scene lights
            self.console.append("relight")
            return {"spawned": 2}
        if "_SB_UNRELIGHT" in script:
            self.mode = "lit"
            self.console.append("unrelight")
            return {"removed": 2}
        return {"removed": 0}


def test_a_black_frame_is_reshot_under_a_rig(tmp_path):
    editor = _Editor(lit_level=0)

    produced = render.capture(editor, tmp_path, half_extent_m=65)

    assert len(produced) == render.DEFAULT_VIEWS
    for path in produced:
        assert render.mean_luma(path) >= render.BLACK_LUMA, (
            f"{Path(path).name} is still black after the fallback")
    assert editor.console[-1] == "unrelight", (
        "the photographer's lamps were left standing in the scene")


def test_the_fallback_uses_the_very_same_cameras(tmp_path):
    """Re-framing would make the fallback incomparable with the lit path."""
    editor = _Editor(lit_level=0)

    render.capture(editor, tmp_path, half_extent_m=65)

    first = editor.shots[:render.DEFAULT_VIEWS]
    second = editor.shots[render.DEFAULT_VIEWS:]
    assert len(second) == len(first), "every black viewpoint should be re-shot"
    assert [c for _, c in second] == [c for _, c in first]


def test_the_black_original_is_kept(tmp_path):
    """That a scene renders black IS a finding; the fallback must not erase it."""
    editor = _Editor(lit_level=0)

    render.capture(editor, tmp_path, half_extent_m=65)

    originals = sorted(tmp_path.glob("*.lit.png"))
    assert len(originals) == render.DEFAULT_VIEWS
    for path in originals:
        assert render.mean_luma(path) == 0.0


def test_a_scene_that_photographs_is_left_alone(tmp_path):
    """A dark night scene is still a night scene — only black triggers this."""
    editor = _Editor(lit_level=90)

    render.capture(editor, tmp_path, half_extent_m=65)

    assert editor.console == [], "a rig was raised for a scene that already lit"
    assert len(editor.shots) == render.DEFAULT_VIEWS, "nothing should be re-shot"
    assert not list(tmp_path.glob("*.lit.png"))


def test_the_record_says_which_frames_are_which(tmp_path):
    """A viewer comparing two models must know which lighting each was shot in."""
    editor = _Editor(lit_level=0)

    render.capture(editor, tmp_path, half_extent_m=65)
    modes = json.loads((tmp_path / "capture_modes.json").read_text())

    assert len(modes) == render.DEFAULT_VIEWS
    assert set(modes.values()) == {"relit"}


def test_a_relight_that_did_not_take_is_recorded_as_such(tmp_path):
    """Measured, not assumed: `viewmode unlit` was the first attempt and it left
    all four frames at 0.0 on the real editor. A frame recorded as relit while
    still black would be this pass lying about its own evidence."""
    editor = _Editor(lit_level=0, unlit_level=0)     # switching changes nothing

    produced = render.capture(editor, tmp_path, half_extent_m=65)
    modes = json.loads((tmp_path / "capture_modes.json").read_text())

    assert set(modes.values()) == {"relight_ineffective"}
    assert len(produced) == render.DEFAULT_VIEWS
    assert all(render.mean_luma(p) == 0.0 for p in produced)
    assert not list(tmp_path.glob("*.lit.png")), (
        "the fallback achieved nothing, so it should leave no second copy")


def test_the_fallback_is_skipped_when_no_rig_could_be_spawned(tmp_path):
    """An editor that refuses the rig must leave the black frames, not lose them."""
    editor = _Editor(lit_level=0)
    editor.exec_python_result = lambda script, key, timeout=120.0: (
        {"spawned": 0} if "_SB_RELIGHT" in script else {"removed": 0})

    produced = render.capture(editor, tmp_path, half_extent_m=65)

    assert len(produced) == render.DEFAULT_VIEWS
    assert all(render.mean_luma(p) == 0.0 for p in produced)
    assert not list(tmp_path.glob("*.lit.png")), (
        "nothing was re-shot, so nothing should have been renamed")


def test_partial_relight_installation_is_rolled_back(monkeypatch):
    class Editor:
        def __init__(self):
            self.scripts = []

        def exec_python(self, script, timeout=60.0):
            del timeout
            self.scripts.append(script)

        def exec_python_result(self, script, key, timeout=120.0):
            del key, timeout
            self.scripts.append(script)
            return {"spawned": 1}

    editor = Editor()
    removals = []

    def remove(bridge, timeout=300.0):
        removals.append((bridge, timeout))
        return 1

    monkeypatch.setattr(render, "remove_relight_rig", remove)

    assert render.add_relight_rig(editor, timeout=45.0) == 0
    assert removals == [(editor, 45.0)]
    assert any("if _n != len(_specs):" in script for script in editor.scripts)
