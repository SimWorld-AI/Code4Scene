"""Scene measurement wrapper: send the canonical payload, read the report.

Python port of an earlier JavaScript ``measureScene`` tool (file
protocol: stale-unlink → execute payload → poll for the JSON file → parse).
Failures raise :class:`MeasureError` — unlike the JS original, callers never
receive an empty report to silently score.

``out_path`` is required and must be private to the environment instance (an
older fixed shared-temp-file convention collided across tenants; the caller
derives a per-run path from configuration).
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any

from code4scene.core import sharedfs
from code4scene.core.bridge import Bridge, BridgeError
from . import measure_payload

METRIC_ID = measure_payload.METRIC_ID

# File-appearance poll after the bridge call returns: the editor can flush the
# payload a beat late right after a spawn burst. Matching the JS exactly, the
# first read happens immediately and is followed by at most 24 sleep-then-read
# retries — 25 reads, the last at t=6.0s.
_POLL_RETRIES = 24
_POLL_INTERVAL_S = 0.25


class MeasureError(Exception):
    """Measurement failed: bridge error, editor-side error, or no output."""


def payload_script(gh_cm: float, out_path: str) -> str:
    """The editor-python script: this module's source + the entry call."""
    source = inspect.getsource(measure_payload)
    return source + f"\nmain({float(gh_cm)!r}, {str(out_path)!r})\n"


def derive_rates(report: dict[str, Any]) -> dict[str, Any]:
    """Attach convenience rates. Counts are the stored truth; rates are
    presentation (rounded to 4 places — JS used ``toFixed(4)``, which may
    differ at half-ulp boundaries; comparisons should use the counts)."""
    n = report.get("checked", 0) or 0
    report["structural_collision_rate"] = round(len(report.get("structural_collision_actors", [])) / n, 4) if n else 0
    report["floating_rate"] = round(len(report.get("floating", [])) / n, 4) if n else 0
    report["oob_rate"] = round(len(report.get("out_of_bounds", [])) / n, 4) if n else 0
    return report


def measure(bridge: Bridge, out_path: str | Path,
            gh_cm: float = measure_payload.DEFAULT_GH_CM,
            timeout: float = 600.0,
            payload_out: str | Path | None = None,
            expect_map: str | None = None) -> dict[str, Any]:
    """Run the measurement pass; return ``{'actors', 'report', 'by_label'}``.

    Pass ``gh_cm = size_m * 50`` so the out-of-bounds gate matches the
    episode's world plate — never rely on the historical ±13000 default.

    ``expect_map`` is the level this measurement is FOR. When given, the loaded
    editor world must be that exact package or the pass raises — measurement no
    longer trusts that "whatever is live" is the run's scene. This is what
    stopped three generation runs from all scoring the same stale ``empty``
    city while their agents built elsewhere.

    ``payload_out`` is where the EDITOR writes the payload; ``out_path`` is
    where the run keeps it. They differ when an environment has its own
    scratch location, and the harness must be able to read the former — the
    payload goes through a file rather than the log because a dense scene's
    per-actor data overflows the editor-log capture.
    """
    if expect_map is not None:
        _assert_world(bridge, expect_map)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    editor_out = Path(payload_out) if payload_out else out
    # The EDITOR writes the payload; on a shared volume it is a different uid,
    # so a directory this process creates at the umask's 775 is one the editor
    # cannot write into (shared-volume permissions, done by the code now).
    sharedfs.mkdir_shared(editor_out.parent)
    # Delete any stale payload FIRST so an editor-side failure can never leave
    # us reading a previous run's data.
    editor_out.unlink(missing_ok=True)
    if editor_out != out:
        out.unlink(missing_ok=True)

    # 600 s, not 120. The old default was exactly the far side's own wait, so
    # a measurement that outran it hit the SOCKET deadline first and the
    # "still running" answer — the thing that distinguishes a slow pass from a
    # failed one — could never arrive. Measuring 553 actors runs an O(n^2) pair
    # loop on the blocked game thread; on a dense scene that is minutes.
    response = None
    try:
        response = bridge.exec_python(payload_script(gh_cm, str(editor_out)),
                                      timeout=timeout)
        bridge._await_job(response, time.monotonic() + timeout)
    except BridgeError as e:
        # Not raised yet. The editor writes the payload to a file this side can
        # read, and it may land a beat after the bridge gives up — the poll
        # below exists for exactly that. Raising here skipped it, so a
        # completed measurement sitting on disk was reported as no measurement
        # at all, and the record's metrics then described a different scene
        # from the saved .umap.
        if _read_payload(editor_out) is None:
            raise MeasureError(f"bridge failure during measurement: {e}") from e

    parsed = _read_payload(editor_out)
    for _ in range(_POLL_RETRIES):
        if parsed is not None:
            break
        time.sleep(_POLL_INTERVAL_S)
        parsed = _read_payload(editor_out)
    if parsed is None:
        logs = bridge.python_logs(response)
        marker = next((line for line in logs.splitlines()
                       if measure_payload.MARK_ERR in line), None)
        raise MeasureError(
            marker or f"no measure output at {editor_out} (editor logs: "
                      f"{logs[-500:] or 'empty'}). The harness must be able to "
                      f"read the path the editor writes to.")

    # Unpacking is guarded: the payload arrives through a file an adversarial
    # agent shares a filesystem with, so a wrong shape must be a clean
    # measurement failure, not an AttributeError from deep in the caller.
    try:
        actors = parsed["actors"]
        report = derive_rates(parsed["report"])
        by_label = {str(a["label"]): a for a in actors}
    except (KeyError, TypeError, AttributeError) as e:
        raise MeasureError(f"malformed measure payload at {editor_out}: {e}") from e
    if editor_out != out:
        # Keep the payload with the run, not only in the environment's scratch.
        out.write_text(json.dumps(parsed))
    return {"actors": actors, "report": report, "by_label": by_label}


def _assert_world(bridge: Bridge, expect_map: str) -> None:
    """Raise unless the editor's current world is exactly ``expect_map``.

    A measurement scores whatever level is loaded. If the run's own level is
    not the one loaded, the number describes a different scene — the silent
    failure that let generation runs score a stale shared map. We refuse to
    measure rather than mislabel.
    """
    probe = "\n".join([
        "import unreal",
        "_w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)"
        ".get_editor_world()",
        "_actual = ''",
        "if _w:",
        "    try:",
        "        _actual = _w.get_outer().get_path_name()",
        "    except Exception:",
        "        _actual = _w.get_path_name().split(chr(58))[0]"
        ".rsplit(chr(46), 1)[0]",
        "globals()['_SB_WORLD'] = {'actual': _actual}",
    ])
    try:
        bridge.exec_python("globals().pop('_SB_WORLD', None)", timeout=60.0)
        result = bridge.exec_python_result(probe, "_SB_WORLD", timeout=120.0)
    except BridgeError as e:
        raise MeasureError(
            f"could not confirm the measured world is {expect_map}: {e}") from e
    actual = result.get("actual")
    if actual != expect_map:
        raise MeasureError(
            f"refusing to measure: expected the run's level {expect_map!r} but "
            f"the editor is on {actual!r}. Scoring the wrong world is how a run "
            f"gets a number for a scene it never built.")


def _read_payload(path: Path) -> dict[str, Any] | None:
    """Read the measurement file, or None while it is absent/incomplete."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
