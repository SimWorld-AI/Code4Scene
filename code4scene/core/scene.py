"""Harness-side scene saving.

The scored artifact must be the scored state. The agent's own save happens
before the harness's final plate enforcement, so a scene saved by the agent
cannot reproduce the recorded metrics — and official scoring reloads the saved
level in a separate, agent-inaccessible instance. The harness therefore saves
the level itself, after the final bounds pass and immediately before the final
measurement.
"""

from __future__ import annotations

import time
from pathlib import Path

from code4scene.core.bridge import Bridge, BridgeError

MARK = "SCENE_SAVED"
#: Printed by `load_script`. Saving and loading announce themselves
#: differently, and a reader greps for one of the two.
LOAD_MARK = "SCENE_LOADED"
DEFAULT_ROOT = "/Game/SavedScenes"

#: A genuinely-empty base level. The retired ``/Game/Maps/empty`` had a
#: procedural city (846 actors) baked into it, which every generation run then
#: silently measured; NOTHING may assume that name again. A generation task
#: names this base as its ``init_map`` to mean "start blank" — the harness then
#: gives the run its OWN fresh level (see :func:`new_canvas`) rather than
#: sharing this one.
BLANK_STAGE = "/Game/SceneBench/BlankStage"

#: `/Game` roots an instance owns rather than mounts: provisioning makes the
#: first three and the editor makes the last two. They are never content, so a
#: task cannot declare them and a level a generator wrote into one is not a
#: level from a pack. Kept here because both the task loader and provisioning
#: need the same answer, and they live in different layers.
INSTANCE_OWNED_ROOTS = ("SavedScenes", "_Runs", "Generated",
                        "Collections", "Developers")


class SceneError(Exception):
    """The level could not be saved."""


#: Editor python runs in ONE persistent interpreter -- the globals popped
#: elsewhere in this file are the proof -- so a name bound to a UWorld by one
#: script is still bound when the next one runs. UE refuses to tear down a
#: world something still references and calls it "World Memory Leaks", which
#: is a fatal, not a warning: the editor dies mid-script, and because it died
#: mid-script the supervisor's liveness probe reads the capture file it never
#: renamed as work still in flight, and so never restarts it.
#:
#: Dropped immediately BEFORE a level swap rather than after each bind. Where
#: a reference is released does not matter; what matters is that nothing
#: holds one at the moment the swap runs, and doing it here covers every
#: script that ever bound one -- including the render and measure payloads,
#: and any written later.
#:
#: This cannot cover names the AGENT bound. Those are its own, clearing them
#: would change what an agent may rely on between its calls, and an agent
#: that holds a world across a load can still fatal the editor as before.
RELEASE_WORLDS = "for _n in ('_w', '_world'): globals().pop(_n, None)"


def _ue_name(text: str) -> str:
    """A UE-package-safe token: only ``[A-Za-z0-9_]``, collapsed."""
    import re
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_") or "scene"


def run_scene_name(task_id: str, run_slug: str, ts: str) -> str:
    """Run-scoped, namespaced, timestamped level name — never shared.

    Two runs of one task must not collide on a well-known name, and a verifier
    must be able to name the exact level a run produced. ``ts`` is a
    ``YYYYMMDD-HHMMSS`` stamp; ``run_slug`` is the run's out-dir identity.
    """
    return (f"scenebench_{_ue_name(task_id)}"
            f"__{_ue_name(ts)}__{_ue_name(run_slug)}")


def payload_script(package_path: str) -> str:
    return "\n".join([
        "import unreal",
        "_w = unreal.EditorLevelLibrary.get_editor_world()",
        f"_p = {package_path!r}",
        "unreal.EditorLoadingAndSavingUtils.save_map(_w, _p)",
        f"print('{MARK} ' + _p)",
        # The print is for a human reading the editor log. The harness reads
        # this instead: a marker printed after the save cannot be relied on
        # (see Bridge.exec_python_result).
        "globals()['_SB_SAVED'] = {'saved': True, 'package': _p}",
    ])


def save(bridge: Bridge, name: str, root: str = DEFAULT_ROOT,
         timeout: float = 300.0, saved_scenes_dir: Path | None = None) -> str:
    """Save the current level as ``<root>/<name>``; return the package path.

    ``saved_scenes_dir`` is where the editor writes levels. Pass it and the
    LEVEL ON DISK decides: the marker is read back out of an editor global,
    and that read can fail — on a long save, or when the editor is busy —
    after ``save_map`` has already written the file. Believing the marker over
    the artifact turned a finished 115-actor scene into ``infra_error`` with
    ``umap: null``, and every verifier refused to score a level that was
    sitting on disk the whole time. A missing file is still a failure; only a
    missing *marker* is now survivable.
    """
    package_path = f"{root.rstrip('/')}/{name}"
    started = time.time()

    def landed() -> bool:
        if saved_scenes_dir is None:
            return False
        umap = Path(saved_scenes_dir) / f"{name}.umap"
        try:
            # Written by THIS attempt: an older file of the same name is a
            # previous run's artifact, and accepting it would report a save
            # that never happened. Allow a little slack for clock skew.
            return umap.exists() and umap.stat().st_mtime >= started - 5
        except OSError:
            return False

    # Clear any previous result first: without this a save that raised would
    # be confirmed by the PREVIOUS episode's value still sitting in the global.
    try:
        bridge.exec_python("globals().pop('_SB_SAVED', None)", timeout=timeout)
        result = bridge.exec_python_result(payload_script(package_path),
                                           "_SB_SAVED", timeout=timeout)
    except BridgeError as e:
        if landed():
            return package_path
        raise SceneError(f"bridge failure while saving the scene: {e}") from e
    if not result.get("saved") and not landed():
        raise SceneError(f"the editor did not save {package_path}")
    return package_path


def new_canvas_script(package_path: str) -> str:
    """Editor-python that creates a fresh, empty level and makes it current."""
    return "\n".join([
        "import unreal",
        f"_p = {package_path!r}",
        RELEASE_WORLDS,
        "_les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)",
        # A template-free new level: 0 actors, and it becomes the editor world.
        "_ok = _les.new_level(_p)",
        "_w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)"
        ".get_editor_world()",
        "_actual = ''",
        "if _w:",
        "    try:",
        "        _actual = _w.get_outer().get_path_name()",
        "    except Exception:",
        "        _actual = _w.get_path_name().split(chr(58))[0]"
        ".rsplit(chr(46), 1)[0]",
        "_n = len(unreal.get_editor_subsystem(unreal.EditorActorSubsystem)"
        ".get_all_level_actors())",
        "globals()['_SB_CANVAS'] = {'created': bool(_ok) and _actual == _p,"
        " 'level': _p, 'actual': _actual, 'reported': bool(_ok), 'actors': _n}",
    ])


def new_canvas(bridge: Bridge, package_path: str,
               timeout: float = 300.0) -> dict:
    """Create a fresh, run-private, empty level at ``package_path`` and open it.

    This replaces opening a shared ``/Game/Maps/empty`` canvas. That map was a
    global mutable resource every generation run built into and every
    measurement implicitly read — and a procedural city got baked into it, so
    every run scored the city. Each run now builds in its own level.
    """
    try:
        bridge.exec_python("globals().pop('_SB_CANVAS', None)", timeout=timeout)
        result = bridge.exec_python_result(new_canvas_script(package_path),
                                           "_SB_CANVAS", timeout=timeout)
    except BridgeError as e:
        raise SceneError(f"bridge failure creating the run canvas: {e}") from e
    if not result.get("created"):
        raise SceneError(
            f"the editor did not open a fresh canvas at {package_path}; "
            f"it is on {result.get('actual')!r}")
    # EMPTY is the other half of the contract, and the half that was taken on
    # trust. `/Game/Maps/empty` was named for being empty right up until a
    # procedural city was baked into it, and every generation run afterwards
    # measured the city — the canvas reported "created" the whole time. A
    # generation score is a count of what the AGENT put there, so a canvas
    # that starts with anything in it is not a canvas, whatever it is called.
    actors = result.get("actors")
    if not isinstance(actors, int) or actors > 0:
        raise SceneError(
            f"the fresh canvas at {package_path} reports {actors!r} actors, "
            f"not 0; a generation run measures what the agent built, and "
            f"anything already standing in the level would be counted as its "
            f"work")
    return result


def load_script(package_path: str) -> str:
    """Editor-python that opens a saved level."""
    return "\n".join([
        "import unreal",
        f"_p = {package_path!r}",
        RELEASE_WORLDS,
        "_les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)",
        "_ok = _les.load_level(_p)",
        # What the editor is ACTUALLY on afterwards. load_level returns True
        # even when it did not load the level: a level whose dependencies are
        # not present leaves the editor on a fresh /Temp/Untitled world, and
        # says it succeeded. Measured on a live editor — the return value said
        # True and the world was /Temp/Untitled_0.Untitled.
        "_w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)"
        ".get_editor_world()",
        # The world's PACKAGE, which is what a task names. get_path_name()
        # returns the object — /Game/Hangar/Maps/Hangar.Hangar — so comparing
        # it to the requested path fails on every successful load, and this
        # check rejected the levels it was written to admit. The outer of a
        # world is its package; the string form is the fallback for a world
        # with no outer, and it also drops the PIE suffix after a colon.
        "_actual = ''",
        "if _w:",
        "    try:",
        "        _actual = _w.get_outer().get_path_name()",
        "    except Exception:",
        "        _actual = _w.get_path_name().split(chr(58))[0]"
        ".rsplit(chr(46), 1)[0]",
        f"print('{LOAD_MARK} ' + _p + ' ' + str(bool(_ok)))",
        # Stored as well as printed: loading a level takes long enough that
        # the marker falls outside the returned log window.
        # `loaded` is the effect, not the announcement: the requested level
        # must be the one the editor ended up on.
        "globals()['_SB_LOADED'] = {'loaded': bool(_ok) and _actual == _p,"
        " 'level': _p, 'actual': _actual, 'reported': bool(_ok)}",
    ])
