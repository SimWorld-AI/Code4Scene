"""Directories two users must agree on.

The scorer creates directories on the shared volumes and the editor, which may
run as a different user outside the scorer's group, writes the files. A
directory created at the umask's 775 would refuse that write with a bare
PermissionError, so the code that creates the directory is the side that has
to open it.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from code4scene.core.sharedfs import mkdir_shared


@pytest.fixture
def strict_umask():
    """The exact umask that produced the 775 directories on a shared volume."""
    previous = os.umask(0o022)
    yield
    os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_created_directories_are_open_to_the_other_uid(tmp_path, strict_umask):
    target = tmp_path / "artifacts" / "paired_renders" / "run-1" / "gt"

    mkdir_shared(target)

    for directory in (target, target.parent, target.parent.parent):
        assert _mode(directory) == 0o777, (
            f"{directory} would refuse the editor's uid; mkdir alone is "
            f"filtered by the umask, which is how the 775 traps were made")


def test_an_existing_parent_is_not_widened(tmp_path, strict_umask):
    """The mount root keeps its own mode; only what THIS call created is
    opened up."""
    mount = tmp_path / "measure"
    mount.mkdir(mode=0o750)
    mount.chmod(0o750)

    mkdir_shared(mount / "scene_evidence" / "task-ep")

    assert _mode(mount) == 0o750, "an existing directory keeps its mode"
    assert _mode(mount / "scene_evidence") == 0o777
    assert _mode(mount / "scene_evidence" / "task-ep") == 0o777


def test_idempotent_over_an_existing_target(tmp_path, strict_umask):
    target = tmp_path / "renders"
    mkdir_shared(target)
    target.chmod(0o700)

    mkdir_shared(target)                    # a second call must not fail...

    assert _mode(target) == 0o700, "...and must not re-widen what it did not create"


def test_the_editor_written_render_directory_is_writable_cross_uid(
        tmp_path, strict_umask):
    """The capture path end to end: the scorer asks for a screenshot, and the
    fake editor (any other user) writes the file into a directory the scorer
    created a moment earlier."""
    from code4scene.evaluation import render as render_eval

    written = {}

    class Bridge:
        def command(self, name, params, timeout=0):
            # The real editor would now open() this path as another user; the
            # directory's mode is the whole question.
            target = Path(params["filepath"])
            written["dir_mode"] = _mode(target.parent)
            target.write_bytes(b"px")

        def exec_python(self, script, timeout=0):        # shot-camera cleanup
            return {}

        def exec_python_result(self, script, key, timeout=0):
            return {"removed": 0}

    out_dir = tmp_path / "artifacts" / "renders_candidate" / "run-1"
    produced = render_eval.capture(
        Bridge(), out_dir, 10.0,
        views=[render_eval.Viewpoint(name="view_0", location=[0, 0, 100],
                                     rotation=[0, -30, 0])])

    assert produced and written["dir_mode"] == 0o777, (
        "the editor is a different uid; a 775 directory here is the "
        "shared-volume trap baked into code")


def test_the_editor_written_evidence_directory_is_writable_cross_uid(
        tmp_path, strict_umask):
    """`_run_editor_export` names an output path and the EDITOR writes it —
    same hand-off, same requirement, for scene evidence and the dependency
    manifest."""
    from code4scene.evaluation import ue_evidence

    output = tmp_path / "measure" / "scene_evidence" / "t-ep" / "out.json"

    class Bridge:
        def exec_python_result(self, script, key, timeout=0):
            assert _mode(output.parent) == 0o777, (
                "the editor's uid cannot write into a 775 directory the "
                "scorer just created")
            output.write_text('{"status": "success"}')
            return {"finished": True}

    payload = ue_evidence._run_editor_export(
        Bridge(), ue_evidence.DEPENDENCIES_SCRIPT, "PRELUDE = 1", output,
        "_SB_TEST", timeout=1.0)

    assert payload == {"status": "success"}
