"""The editor finishes a screenshot after the call that asked for it returns.

`take_screenshot` queues the shot and the editor writes the file on a later
frame. The RGB path used to check for that file the instant the bridge replied,
so a heavy scene — where the write takes longest — was the one most likely to
be recorded as having produced nothing.

One missed overview shot leaves every requirement it covers without an image.
The message names the directory the image is sitting in.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from code4scene.evaluation import render

Image = pytest.importorskip("PIL.Image", reason="the frames have to be real")


class _SlowEditor:
    """Answers at once and writes the file `delay` seconds later, on its own.

    A fake that only wrote when it was next CALLED would pass against any
    implementation, including the broken one — the waiter polls the
    filesystem and never speaks to the editor again. The editor's clock has
    to be independent of ours for this to test anything.
    """

    def __init__(self, delay: float = 0.4) -> None:
        self.delay = delay
        self.timers: list[threading.Timer] = []

    @staticmethod
    def _write(path: Path) -> None:
        Image.new("RGB", (64, 36), (140, 140, 140)).save(path)

    def command(self, name, args, timeout=300.0):
        if name == "take_screenshot":
            timer = threading.Timer(
                self.delay, self._write, args=(Path(args["filepath"]),))
            timer.daemon = True
            timer.start()
            self.timers.append(timer)
        return {"status": "success"}

    def exec_python(self, script, timeout=60.0):
        return {}

    def exec_python_result(self, script, key, timeout=120.0):
        return {"removed": 0}


def test_a_screenshot_that_lands_late_is_still_a_screenshot(tmp_path):
    editor = _SlowEditor(delay=0.4)

    produced = render.capture(editor, tmp_path, half_extent_m=65)

    assert len(produced) == render.DEFAULT_VIEWS, (
        "a frame the editor was still writing was counted as never written")
    for path in produced:
        assert Path(path).stat().st_size > 0


def test_a_frame_that_never_arrives_still_fails(tmp_path):
    """The wait is bounded: waiting is not the same as accepting anything."""

    class _MuteEditor(_SlowEditor):
        def command(self, name, args, timeout=300.0):
            return {"status": "success"}      # asked for, never written

    with pytest.raises(render.RenderError, match="no viewpoint produced"):
        render.capture(_MuteEditor(), tmp_path, half_extent_m=65, timeout=0.3)
