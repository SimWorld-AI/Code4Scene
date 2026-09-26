"""Photograph the finished scene, so something can look at it.

The judge package is complete — a rubric, a schema, two backends, a ceiling the
deterministic metrics impose — and had no caller, because it scores IMAGES and
nothing in the harness ever made any. A verifier declaring ``kind: judge``
could only ever report that it had not run.

So this is the missing half. It is not a new capability: the editor has taken
screenshots from a given camera all along, and the harness simply never asked.

**Several viewpoints, not one.** A single camera can be aimed at the one corner
that was built and miss that the rest is empty, and a judge shown that image is
being asked the wrong question. The viewpoints here ring the plate at a fixed
height and all look at its centre, so what varies between runs is the scene
rather than the framing — which is the only way two runs' verdicts are
comparable.

**Written by the EDITOR.** The path handed in must be one the editor can write
and the harness can read. They are the same inside a container and are not on a
lab host, which is the same two-sided path the measurement pass has.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence

from code4scene.core.bridge import Bridge, BridgeError
from code4scene.core.sharedfs import mkdir_shared

#: Enough viewpoints that one lucky corner cannot carry a scene, few enough
#: that the pass costs seconds. The judge refuses fewer than two.
DEFAULT_VIEWS = 4

#: Current candidate-only visual framing. Any change is a scoring-protocol
#: change and must be reviewed together with the canonical policy.
RING_PROTOCOL = "candidate-ring-known-setup-sun-normalized"
GT_CAPTION_ENVIRONMENT_SCENE_GRAPH_PROTOCOL = (
    "gt-caption-environment-routed-visibility-gallery"
)
REFERENCE_OVERVIEW_SCENE_GRAPH_PROTOCOL = (
    "reference-candidate-clearance-overview-gallery"
)
GT_PAIRED_ENVIRONMENT_SCENE_GRAPH_PROTOCOL = (
    "gt-paired-scene-graph-environment-routed-visibility"
)
GT_PAIRED_SCENE_GRAPH_PROTOCOLS = frozenset({
    GT_PAIRED_ENVIRONMENT_SCENE_GRAPH_PROTOCOL,
})
GT_PAIRED_PROTOCOLS = GT_PAIRED_SCENE_GRAPH_PROTOCOLS
SUPPORTED_RING_PROTOCOLS = frozenset({
    RING_PROTOCOL,
    GT_CAPTION_ENVIRONMENT_SCENE_GRAPH_PROTOCOL,
    REFERENCE_OVERVIEW_SCENE_GRAPH_PROTOCOL,
    *GT_PAIRED_SCENE_GRAPH_PROTOCOLS,
})

# Evaluation-only correction for maps produced by the historical
# setup_environment Rotator positional-argument bug. This is not generic
# auto-lighting.
LIGHTING_NORMALIZATION_POLICY = "known-setup-environment-sun-rotation"
DETERMINISTIC_RELIGHT_POLICY = "deterministic-neutral-relight"
PAIRED_PHOTOMETRY_LIGHTING_POLICY = (
    "gt-paired-per-view-independent-photometry"
)
EXPOSURE_NORMALIZATION_POLICY = "eye-adaptation-disabled"
PAIRED_PHOTOMETRY_EXPOSURE_POLICY = (
    "camera-manual-fixed-per-view-bias-search"
)
POST_PROCESS_BIAS_CAMERA_EXPOSURE_MODE = "camera-component-post-process-bias"
PHOTOMETRY_CAMERA_EXPOSURE_MODE = (
    "camera-component-manual-fixed-bias"
)
SUPPORTED_CAMERA_EXPOSURE_MODES = frozenset({
    POST_PROCESS_BIAS_CAMERA_EXPOSURE_MODE,
    PHOTOMETRY_CAMERA_EXPOSURE_MODE,
})
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FOV_DEG = 55.0
RADIUS_HALF_EXTENT_MULTIPLIER = 1.7
HEIGHT_RADIUS_MULTIPLIER = 0.55

# A formal paired visual score cannot depend on one map happening to contain a
# working sun while the other produces valid-but-black PNGs.  This fill rig is
# added to both worlds with exactly these values and removed without saving.
# Existing scene lights remain untouched: the rig normalizes minimum
# visibility, not the authored lighting difference the visual score measures.
DETERMINISTIC_FILL_RIG = (
    {
        "class": "DirectionalLight",
        "label": "_SbRelight_DirectionalLight",
        "location_cm": [0.0, 0.0, 20_000.0],
        "rotation_deg": [0.0, -45.0, 30.0],
        # 10 lux is effectively moonlight once automatic exposure is disabled.
        # A moderate daylight key keeps opaque indoor geometry legible while
        # remaining identical across Input, Candidate, and GT.
        "intensity": 10_000.0,
        "indirect_lighting_intensity": 1.0,
    },
    {
        "class": "SkyLight",
        "label": "_SbRelight_SkyLight",
        "location_cm": [0.0, 0.0, 20_000.0],
        "rotation_deg": [0.0, 0.0, 0.0],
        "intensity": 4.0,
    },
)
DETERMINISTIC_FILL_RIG_SHA256 = hashlib.sha256(
    json.dumps(
        DETERMINISTIC_FILL_RIG,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

def _neutral_fill_level(
    identifier: str,
    directional_intensity: float,
    sky_intensity: float,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "lighting_policy": (
            "gt-paired-neutral-fill-" + identifier.removeprefix("neutral_fill_")
        ).replace("_", "-"),
        "rig_spec": (
            {
                "class": "DirectionalLight",
                "label": "_SbRelight_DirectionalLight",
                "location_cm": [0.0, 0.0, 20_000.0],
                "rotation_deg": [0.0, -45.0, 30.0],
                "intensity": directional_intensity,
                "indirect_lighting_intensity": 1.0,
            },
            {
                "class": "SkyLight",
                "label": "_SbRelight_SkyLight",
                "location_cm": [0.0, 0.0, 20_000.0],
                "rotation_deg": [0.0, 0.0, 0.0],
                "intensity": sky_intensity,
            },
        ),
    }


# This is the one released bounded photometry search ladder.  Keep both order
# and numeric values stable: the executor accepts the first readable setting.
PHOTOMETRY_FILL_LEVELS = (
    _neutral_fill_level("neutral_fill_micro", 10.0, 0.005),
    _neutral_fill_level("neutral_fill_very_low", 25.0, 0.0125),
    _neutral_fill_level("neutral_fill_low_30", 30.0, 0.015),
    _neutral_fill_level("neutral_fill_low_50", 50.0, 0.025),
    _neutral_fill_level("neutral_fill_minimal", 100.0, 0.05),
    _neutral_fill_level("neutral_fill_low", 250.0, 0.125),
    _neutral_fill_level("neutral_fill_weak", 500.0, 0.25),
    _neutral_fill_level("neutral_fill_medium", 1_500.0, 0.6),
    _neutral_fill_level("neutral_fill_strong", 3_500.0, 1.4),
)
PHOTOMETRY_FILL_LEVELS_BY_LIGHTING_POLICY = {
    PAIRED_PHOTOMETRY_LIGHTING_POLICY: PHOTOMETRY_FILL_LEVELS,
}
PHOTOMETRY_FILL_BY_POLICY = {
    str(level["lighting_policy"]): level
    for levels in PHOTOMETRY_FILL_LEVELS_BY_LIGHTING_POLICY.values()
    for level in levels
}
PHOTOMETRY_FILL_RIG_SHA256 = {
    policy: hashlib.sha256(
        json.dumps(
            level["rig_spec"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    for policy, level in PHOTOMETRY_FILL_BY_POLICY.items()
}

def _exposure_level(bias_ev: float) -> dict[str, Any]:
    direction = "plus" if bias_ev > 0 else "minus"
    magnitude = int(abs(bias_ev))
    return {
        "id": f"exposure_{direction}_{magnitude}",
        "exposure_policy": f"camera-manual-fixed-bias-{direction}-{magnitude}",
        "bias_ev": bias_ev,
    }


PHOTOMETRY_EXPOSURE_LEVELS = tuple(
    _exposure_level(value)
    for value in (-1.0, 1.0, -2.0, 2.0, -3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
)
PHOTOMETRY_EXPOSURE_BY_POLICY = {
    str(level["exposure_policy"]): level
    for level in PHOTOMETRY_EXPOSURE_LEVELS
}

#: The label the editor's screenshot tool gives the camera it spawns. It
#: destroys stale ones at the START of each call and never removes the last,
#: so photographing a scene left a camera standing in it — and this pass runs
#: BEFORE the verifiers, so the restoration scorer counted that camera as
#: something the agent had added. Coupled to the tool's own constant on
#: purpose: whoever changes it should change this too.
SHOT_CAMERA_LABEL = "_SwShotCam"

#: Where cameras sit when a task declares no plate, in centimetres. Chosen to
#: frame the historical ±13000 default from outside it.
DEFAULT_RADIUS_CM = 22000.0

#: One vocabulary across capture, pairing and metrics. ``scene_depth`` is
#: explicit because a generic image called ``depth`` is commonly a normalised
#: visualisation; this channel is floating-point camera-space depth in UE cm.
CHANNELS = ("rgb", "base_color", "scene_depth")
IMPLEMENTED_CHANNELS = CHANNELS


class RenderError(Exception):
    """The scene could not be photographed."""


@contextmanager
def normalized_lighting(
    bridge: Bridge,
    policy: str | None,
    timeout: float = 120.0,
):
    """Temporarily repair only the known harness-authored sun rotation.

    The actor label, class, and erroneous transform must all match.  Model-
    authored lights are left untouched.  The original transform is restored
    in ``finally`` and the level is never saved.
    """

    if policy is None:
        yield None
        return
    if policy in PHOTOMETRY_FILL_LEVELS_BY_LIGHTING_POLICY:
        stale_removed = remove_relight_rig(bridge, timeout=timeout)
        yield {
            "policy": policy,
            "acquisition_setting": "native",
            "stale_removed_before_capture": stale_removed,
            "saved_level": False,
        }
        return
    if (
        policy == DETERMINISTIC_RELIGHT_POLICY
        or policy in PHOTOMETRY_FILL_BY_POLICY
    ):
        if policy == DETERMINISTIC_RELIGHT_POLICY:
            rig_spec = DETERMINISTIC_FILL_RIG
            rig_sha256 = DETERMINISTIC_FILL_RIG_SHA256
            acquisition_setting = "deterministic_relight"
        else:
            level = PHOTOMETRY_FILL_BY_POLICY[policy]
            rig_spec = level["rig_spec"]
            rig_sha256 = PHOTOMETRY_FILL_RIG_SHA256[policy]
            acquisition_setting = str(level["id"])
        stale_removed = remove_relight_rig(bridge, timeout=timeout)
        result = add_deterministic_fill_rig(
            bridge,
            timeout=timeout,
            rig_spec=rig_spec,
            policy=policy,
        )
        result["acquisition_setting"] = acquisition_setting
        result["stale_removed_before_capture"] = stale_removed
        if (
            result.get("policy") != policy
            or result.get("rig_sha256") != rig_sha256
            or result.get("spawned") != len(rig_spec)
            or result.get("configured") != len(rig_spec)
            or result.get("errors")
        ):
            remove_relight_rig(bridge, timeout=timeout)
            raise RenderError(
                "paired fill rig was not installed exactly: "
                f"{result}"
            )
        try:
            yield result
        finally:
            removed = remove_relight_rig(bridge, timeout=timeout)
            result["removed_after_capture"] = removed
            if removed != len(rig_spec):
                raise RenderError(
                    "paired fill cleanup was incomplete: "
                    f"expected {len(rig_spec)}, got {removed}"
                )
        return
    if policy != LIGHTING_NORMALIZATION_POLICY:
        raise RenderError(f"unknown lighting normalization policy {policy!r}")

    apply_script = "\n".join(
        [
            "import unreal",
            "_subs = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)",
            "_records = []",
            "for _actor in list(_subs.get_all_level_actors()):",
            "    try:",
            "        _label = _actor.get_actor_label()",
            "        _class = _actor.get_class().get_name()",
            "        if _label != 'SimWorldStudio_Sun' or _class != 'DirectionalLight':",
            "            continue",
            "        _rotation = _actor.get_actor_rotation()",
            "        _before = {'roll': float(_rotation.roll), 'pitch': float(_rotation.pitch), 'yaw': float(_rotation.yaw)}",
            "        _matches = (abs(_before['roll'] + 45.0) <= 0.25 and abs(_before['pitch'] - 30.0) <= 0.25 and abs(_before['yaw']) <= 0.25)",
            "        _record = {'actor_name': _actor.get_name(), 'actor_label': _label, 'actor_class': _class, 'before': _before, 'matched_known_bug': _matches, 'applied': False}",
            "        if _matches:",
            "            _actor.set_actor_rotation(unreal.Rotator(roll=0.0, pitch=-45.0, yaw=30.0), False)",
            "            _record['after'] = {'roll': 0.0, 'pitch': -45.0, 'yaw': 30.0}",
            "            _record['applied'] = True",
            "        _records.append(_record)",
            "    except Exception as _error:",
            "        _records.append({'applied': False, 'inspection_error': str(_error)})",
            f"globals()['_SB_LIGHTING_NORMALIZATION'] = {{'policy': {policy!r}, 'records': _records, 'saved_level': False}}",
        ]
    )
    try:
        bridge.exec_python(
            "globals().pop('_SB_LIGHTING_NORMALIZATION', None)", timeout=60.0
        )
        result = bridge.exec_python_result(
            apply_script, "_SB_LIGHTING_NORMALIZATION", timeout=timeout
        )
    except BridgeError as error:
        raise RenderError(f"lighting normalization failed: {error}") from error
    if not isinstance(result, dict) or result.get("policy") != policy:
        raise RenderError("editor did not return lighting normalization provenance")

    try:
        yield result
    finally:
        originals = [
            {"actor_name": value.get("actor_name"), "before": value.get("before")}
            for value in result.get("records", [])
            if value.get("applied") is True
        ]
        restore_script = "\n".join(
            [
                "import unreal",
                "_subs = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)",
                f"_originals = {originals!r}",
                "_by_name = {value['actor_name']: value['before'] for value in _originals}",
                "_restored = []",
                "for _actor in list(_subs.get_all_level_actors()):",
                "    if _actor.get_name() not in _by_name:",
                "        continue",
                "    _before = _by_name[_actor.get_name()]",
                "    _actor.set_actor_rotation(unreal.Rotator(roll=_before['roll'], pitch=_before['pitch'], yaw=_before['yaw']), False)",
                "    _restored.append(_actor.get_name())",
                "globals()['_SB_LIGHTING_RESTORE'] = {'restored': _restored}",
            ]
        )
        try:
            bridge.exec_python(
                "globals().pop('_SB_LIGHTING_RESTORE', None)", timeout=60.0
            )
            restored = bridge.exec_python_result(
                restore_script, "_SB_LIGHTING_RESTORE", timeout=timeout
            )
        except BridgeError as error:
            raise RenderError(f"lighting restoration failed: {error}") from error
        expected = sorted(value["actor_name"] for value in originals)
        actual = (
            sorted(restored.get("restored", []))
            if isinstance(restored, dict)
            else []
        )
        if actual != expected:
            raise RenderError(
                f"lighting restoration was incomplete: expected {expected}, got {actual}"
            )


@contextmanager
def normalized_exposure(
    bridge: Bridge,
    policy: str | None,
    camera_exposure_mode: str = POST_PROCESS_BIAS_CAMERA_EXPOSURE_MODE,
    timeout: float = 120.0,
):
    """Freeze exposure and declare the per-camera paired EV contract.

    Paired visual capture uses a CameraComponent in manual metering mode.
    That mode needs eye-adaptation quality enabled for the post-process
    exposure bias to affect pixels, while manual metering itself keeps the
    exposure temporally fixed.
    """
    if policy is None:
        yield None
        return
    bias_level = PHOTOMETRY_EXPOSURE_BY_POLICY.get(policy)
    paired_search = policy == PAIRED_PHOTOMETRY_EXPOSURE_POLICY
    if (
        policy != EXPOSURE_NORMALIZATION_POLICY
        and not paired_search
        and bias_level is None
    ):
        raise RenderError(f"unknown exposure normalization policy {policy!r}")
    bias_ev = (
        float(bias_level["bias_ev"])
        if bias_level is not None
        else 0.0
    )
    camera_override = paired_search or bias_level is not None
    if camera_override and camera_exposure_mode not in SUPPORTED_CAMERA_EXPOSURE_MODES:
        raise RenderError(
            f"unknown camera exposure mode {camera_exposure_mode!r}"
        )
    eye_adaptation_quality = (
        2
        if (
            camera_override
            and camera_exposure_mode
            == PHOTOMETRY_CAMERA_EXPOSURE_MODE
        )
        else 0
    )
    script = "\n".join([
        "import unreal",
        "_before = int(unreal.SystemLibrary.get_console_variable_int_value('r.EyeAdaptationQuality'))",
        f"unreal.SystemLibrary.execute_console_command(None, 'r.EyeAdaptationQuality {eye_adaptation_quality}')",
        f"_policy = {policy!r}",
        f"_bias_ev = {bias_ev!r}",
        f"_paired_search = {bool(camera_override)!r}",
        f"_camera_override_mode = {camera_exposure_mode!r}",
        "globals()['_SB_EXPOSURE_NORMALIZATION'] = {",
        "    'policy': _policy, 'before': _before, 'applied': True,",
        "    'paired_bias_search': _paired_search, 'bias_ev': _bias_ev,",
        "    'camera_override_mode': (_camera_override_mode",
        "        if _paired_search else None),",
        "    'camera_override_applied': False,",
        "    'camera_override_applied_view_count': 0,",
        "    'saved_level': False,",
        "}",
    ])
    try:
        bridge.exec_python(
            "globals().pop('_SB_EXPOSURE_NORMALIZATION', None)", timeout=60.0
        )
        record = bridge.exec_python_result(
            script, "_SB_EXPOSURE_NORMALIZATION", timeout=timeout
        )
    except BridgeError as error:
        raise RenderError(f"exposure normalization failed: {error}") from error
    if not isinstance(record, dict) or record.get("policy") != policy:
        raise RenderError("editor did not return exposure normalization provenance")
    try:
        yield record
    finally:
        before = int(record.get("before", 1))
        try:
            bridge.exec_python(
                "import unreal\n"
                f"unreal.SystemLibrary.execute_console_command(None, "
                f"'r.EyeAdaptationQuality {before}')\n",
                timeout=timeout,
            )
        except BridgeError as error:
            raise RenderError(f"exposure restoration failed: {error}") from error


@contextmanager
def normalized_capture(
    bridge: Bridge,
    *,
    lighting_policy: str | None,
    exposure_policy: str | None,
    camera_exposure_mode: str = POST_PROCESS_BIAS_CAMERA_EXPOSURE_MODE,
    timeout: float,
):
    """Apply and audit the frozen render normalizations as one scope."""
    with normalized_exposure(
        bridge,
        exposure_policy,
        camera_exposure_mode=camera_exposure_mode,
        timeout=timeout,
    ) as exposure:
        with normalized_lighting(
            bridge, lighting_policy, timeout=timeout
        ) as lighting:
            yield tuple(value for value in (exposure, lighting) if value is not None)


@dataclass(frozen=True)
class Viewpoint:
    """One camera, in the editor's own coordinates."""

    name: str
    location: list[float]
    rotation: list[float]          # [roll, pitch, yaw] — the editor adapter feeds
                                   # unreal.Rotator(r0, r1, r2) positionally, and that
                                   # constructor's order is (roll, pitch, yaw).
    fov_deg: float = DEFAULT_FOV_DEG


def ring(count: int = DEFAULT_VIEWS, radius_cm: float = DEFAULT_RADIUS_CM,
         height_cm: float | None = None,
         center_cm: tuple[float, float, float] = (0.0, 0.0, 0.0),
         ) -> list[Viewpoint]:
    """Cameras evenly around the scene, all aimed at its centre.

    Fixed framing on purpose: a judge comparing two runs must be comparing the
    scenes, not one run's luckier camera.

    ``center_cm`` is where the ring stands and where every camera aims. The
    default is the world origin, which is right for a generation run on a
    fresh canvas — the plate is origin-centred and so is what gets built. It
    is WRONG for photographing an authored level that sits off to one side:
    Hangar's content is centred 79 m from the origin, and an origin ring
    sized to its 380 m plate produced input views that were mostly sky with
    the scene as a distant silhouette — the third framing defect of this
    shape (the Stage 2 scout cameras and the plate measurement came first),
    each one a camera aimed at a coordinate system instead of at the scene.
    """
    if count < 2:
        raise RenderError(
            f"{count} viewpoint(s) is not a look at a scene — one camera can be "
            f"aimed at the only corner that was built. The judge refuses fewer "
            f"than two for the same reason.")
    cx, cy, cz = (float(v) for v in center_cm)
    height = radius_cm * HEIGHT_RADIUS_MULTIPLIER if height_cm is None else height_cm
    views = []
    for i in range(count):
        angle = 2 * math.pi * i / count
        x, y = radius_cm * math.cos(angle), radius_cm * math.sin(angle)
        # Aim back at the ring's centre: yaw along the inward vector, pitch
        # down by however far above it the camera sits.
        yaw = math.degrees(math.atan2(-y, -x))
        pitch = -math.degrees(math.atan2(height, math.hypot(x, y)))
        views.append(Viewpoint(name=f"view_{i}",
                               location=[round(cx + x, 1), round(cy + y, 1),
                                         round(cz + height, 1)],
                               rotation=[0.0, round(pitch, 1), round(yaw, 1)]))
    return views


def content_framing(actors: list[dict[str, Any]],
                    ) -> tuple[tuple[float, float, float], float]:
    """Where a scene's content is, and how far it reaches — in cm.

    THE placement authority for photographing an authored scene. The same
    origin-centred assumption was independently baked into three camera
    systems — this ring, the Stage 2 scouts, and the case builder — and was
    right for all of them until cases began deriving from assembled
    environments, which sit wherever their authors built them. Each copy then
    failed the same way on its own schedule: a camera aimed at a coordinate
    system instead of at the scene. One implementation, imported by everyone,
    is the fix for the fourth occurrence.

    Measured over the actors the bounds pass acts on: the scenery prefixes
    are excluded because the sky sphere sits 16 km out and would put any ring
    framed to include it in orbit.
    """
    from code4scene.core.inventory import SCENERY_PREFIXES

    xs, ys, zs = [], [], []
    for actor in actors:
        if str(actor.get("label") or "").startswith(SCENERY_PREFIXES):
            continue
        loc = actor.get("loc")
        if not loc or len(loc) < 3:
            continue
        xs.append(float(loc[0]))
        ys.append(float(loc[1]))
        zs.append(float(loc[2]))
    if not xs:
        return (0.0, 0.0, 0.0), 10000.0
    center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, min(zs))
    reach = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 or 5000.0
    return center, reach


def views_for(half_extent_m: float | None, count: int = DEFAULT_VIEWS
              ) -> list[Viewpoint]:
    """Viewpoints framing a task's plate, or the default when it has none."""
    if half_extent_m:
        # Far enough out that the whole plate is in shot with room around it.
        return ring(
            count,
            radius_cm=half_extent_m * 100 * RADIUS_HALF_EXTENT_MULTIPLIER,
        )
    return ring(count)


def translated_views(
    views: Sequence[Viewpoint],
    anchor_cm: Sequence[float],
) -> list[Viewpoint]:
    """Apply a scene anchor to one shared scene-relative camera specification."""
    if len(anchor_cm) != 3:
        raise RenderError("scene anchor must contain exactly three coordinates")
    anchor = [float(value) for value in anchor_cm]
    return [
        Viewpoint(
            name=view.name,
            location=[
                round(float(value) + anchor[index], 6)
                for index, value in enumerate(view.location)
            ],
            rotation=list(view.rotation),
            fov_deg=float(view.fov_deg),
        )
        for view in views
    ]


def check_channels(channels: Sequence[str]) -> tuple[str, ...]:
    """Reject a channel this layer cannot actually produce."""
    unknown = [c for c in channels if c not in CHANNELS]
    if unknown:
        raise RenderError(
            f"unknown render channel(s): {', '.join(unknown)}. Known: "
            f"{', '.join(CHANNELS)}")
    return tuple(channels)


def _target(out_dir: Path, view: Viewpoint, channel: str) -> Path:
    if channel == "rgb":
        return out_dir / f"{view.name}.png"
    suffix = ".exr" if channel == "scene_depth" else ".png"
    return out_dir / channel / f"{view.name}{suffix}"


def _wait_for_file(path: Path, timeout: float) -> None:
    """Wait for a render export to become visible and non-empty."""
    deadline = time.monotonic() + timeout
    previous = -1
    stable = 0
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if size > 0 and size == previous:
            stable += 1
            if stable >= 1:
                return
        else:
            stable = 0
            previous = size
        time.sleep(0.1)
    raise RenderError(f"UE did not write a stable {path.name} render at {path}")


def _capture_deferred(bridge: Bridge, target: Path, view: Viewpoint,
                      channel: str, width: int, height: int,
                      timeout: float) -> None:
    """Capture Base Color or floating-point Scene Depth with SceneCapture2D."""
    if channel not in ("base_color", "scene_depth"):
        raise RenderError(f"deferred capture does not implement {channel!r}")
    # The EDITOR writes this file, and on a shared volume it is a different
    # uid: a directory this process creates at the umask's 775 is one the
    # editor cannot write into (shared-volume permissions, now done by the code).
    mkdir_shared(target.parent)
    target.unlink(missing_ok=True)
    extensionless = target.with_suffix("")
    extensionless.unlink(missing_ok=True)
    capture_source = ("SCS_BASE_COLOR" if channel == "base_color"
                      else "SCS_SCENE_DEPTH")
    target_format = ("RTF_RGBA8" if channel == "base_color"
                     else "RTF_RGBA32F")
    encoding = ("float_exr_scene_depth_cm" if channel == "scene_depth"
                else "rgba8_png_base_color")
    script = "\n".join([
        "import unreal, os",
        f"_target = {str(target)!r}",
        "os.makedirs(os.path.dirname(_target), exist_ok=True)",
        "_actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)",
        "_world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()",
        f"_location = unreal.Vector(*{list(view.location)!r})",
        f"_rotation = unreal.Rotator(*{list(view.rotation)!r})",
        "_camera = _actors.spawn_actor_from_class(unreal.SceneCapture2D, _location, _rotation)",
        "if not _camera:",
        "    raise RuntimeError('could not spawn SceneCapture2D')",
        f"_camera.set_actor_label({SHOT_CAMERA_LABEL!r} + '_{channel}')",
        "_component = _camera.get_editor_property('capture_component2d')",
        f"_component.set_editor_property('fov_angle', {float(view.fov_deg)!r})",
        f"_texture = unreal.RenderingLibrary.create_render_target2d(_world, {width}, {height}, unreal.TextureRenderTargetFormat.{target_format}, unreal.LinearColor(0, 0, 0, 1), False, False)",
        "if not _texture:",
        "    raise RuntimeError('could not create render target')",
        f"_component.set_editor_property('capture_source', unreal.SceneCaptureSource.{capture_source})",
        "_component.set_editor_property('capture_every_frame', False)",
        "_component.set_editor_property('capture_on_movement', False)",
        "_component.set_editor_property('texture_target', _texture)",
        "_component.capture_scene()",
        "unreal.RenderingLibrary.export_render_target(_world, _texture, os.path.dirname(_target), os.path.splitext(os.path.basename(_target))[0])",
        "globals()['_SB_RENDER_CHANNEL'] = {'finished': True, 'target': _target, "
        f"'channel': {channel!r}, 'encoding': {encoding!r}}}",
    ])
    try:
        bridge.exec_python("globals().pop('_SB_RENDER_CHANNEL', None)", timeout=60.0)
        result = bridge.exec_python_result(
            script, "_SB_RENDER_CHANNEL", timeout=timeout)
    except BridgeError as error:
        raise RenderError(f"{channel} capture failed: {error}") from error
    if not isinstance(result, dict) or result.get("finished") is not True:
        raise RenderError(f"UE did not finish the {channel} capture")
    if not target.is_file() and extensionless.is_file():
        extensionless.replace(target)
    _wait_for_file(target, min(timeout, 5.0))


def _capture_deferred_batch(
    bridge: Bridge,
    values: Sequence[tuple[Viewpoint, Path, str]],
    width: int,
    height: int,
    timeout: float,
) -> set[Path] | None:
    """Capture every deferred channel through one UE Python submission.

    Base Color and Scene Depth used to pay two bridge round trips, one
    ``SceneCapture2D`` spawn and one render-target allocation per image.  On
    the production editor the file itself appeared in roughly three seconds,
    but the next request did not start for about 27 seconds.  Reuse one camera
    and one target per channel across the fixed pose sweep while preserving
    the exact per-view files and encodings.

    ``None`` is an explicit compatibility fallback.  A partial successful
    batch returns the files that appeared; the caller re-shoots only missing
    addresses with the established single-image path.
    """
    if len(values) < 2:
        return None
    specs: list[dict[str, Any]] = []
    for view, target, channel in values:
        if channel not in ("base_color", "scene_depth"):
            raise RenderError(f"deferred capture does not implement {channel!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        target.with_suffix("").unlink(missing_ok=True)
        specs.append({
            "target": str(target),
            "channel": channel,
            "location": list(view.location),
            "rotation": list(view.rotation),
            "fov_deg": float(view.fov_deg),
        })
    script = "\n".join([
        "import unreal, os",
        f"_specs = {specs!r}",
        "globals()['_SB_RENDER_CHANNEL_BATCH'] = {",
        "    'finished': False, 'records': [],",
        "    'camera_policy': 'single-scene-capture-component-batch',",
        "}",
        "_actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)",
        "_world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()",
        "_first = _specs[0]",
        "_camera = _actors.spawn_actor_from_class(",
        "    unreal.SceneCapture2D, unreal.Vector(*_first['location']),",
        "    unreal.Rotator(*_first['rotation']))",
        "if not _camera:",
        "    raise RuntimeError('could not spawn batched SceneCapture2D')",
        f"_camera.set_actor_label({SHOT_CAMERA_LABEL!r} + '_deferred_batch')",
        "_component = _camera.get_editor_property('capture_component2d')",
        "_component.set_editor_property('capture_every_frame', False)",
        "_component.set_editor_property('capture_on_movement', False)",
        "_textures = {",
        f"    'base_color': unreal.RenderingLibrary.create_render_target2d(_world, {width}, {height}, unreal.TextureRenderTargetFormat.RTF_RGBA8, unreal.LinearColor(0, 0, 0, 1), False, False),",
        f"    'scene_depth': unreal.RenderingLibrary.create_render_target2d(_world, {width}, {height}, unreal.TextureRenderTargetFormat.RTF_RGBA32F, unreal.LinearColor(0, 0, 0, 1), False, False),",
        "}",
        "if not all(_textures.values()):",
        "    raise RuntimeError('could not create batched render targets')",
        "_records = []",
        "for _spec in _specs:",
        "    _target = _spec['target']",
        "    os.makedirs(os.path.dirname(_target), exist_ok=True)",
        "    _camera.set_actor_location(unreal.Vector(*_spec['location']), False, False)",
        "    _camera.set_actor_rotation(unreal.Rotator(*_spec['rotation']), False)",
        "    _component.set_editor_property('fov_angle', _spec['fov_deg'])",
        "    if _spec['channel'] == 'base_color':",
        "        _component.set_editor_property('capture_source', unreal.SceneCaptureSource.SCS_BASE_COLOR)",
        "    else:",
        "        _component.set_editor_property('capture_source', unreal.SceneCaptureSource.SCS_SCENE_DEPTH)",
        "    _component.set_editor_property('texture_target', _textures[_spec['channel']])",
        "    _component.capture_scene()",
        "    unreal.RenderingLibrary.export_render_target(",
        "        _world, _textures[_spec['channel']], os.path.dirname(_target),",
        "        os.path.splitext(os.path.basename(_target))[0])",
        "    _records.append({'target': _target, 'channel': _spec['channel']})",
        "globals()['_SB_RENDER_CHANNEL_BATCH'] = {",
        "    'finished': True, 'records': _records,",
        "    'camera_policy': 'single-scene-capture-component-batch',",
        "}",
    ])
    try:
        bridge.exec_python(
            "globals().pop('_SB_RENDER_CHANNEL_BATCH', None)", timeout=60.0)
        result = bridge.exec_python_result(
            script,
            "_SB_RENDER_CHANNEL_BATCH",
            timeout=max(timeout, timeout * len(values)),
        )
    except Exception:  # noqa: BLE001 - compatibility/failure fallback
        return None
    if (
        not isinstance(result, dict)
        or result.get("finished") is not True
        or result.get("camera_policy") !=
        "single-scene-capture-component-batch"
    ):
        return None
    produced: set[Path] = set()
    for _, target, _ in values:
        extensionless = target.with_suffix("")
        if not target.is_file() and extensionless.is_file():
            extensionless.replace(target)
        try:
            _wait_for_file(target, min(timeout, 5.0))
        except RenderError:
            continue
        produced.add(target)
    return produced


def _capture_one(
    bridge: Bridge,
    target: Path,
    view: Viewpoint,
    channel: str,
    width: int,
    height: int,
    timeout: float,
    exposure_bias_ev: float | None = None,
    camera_exposure_mode: str = POST_PROCESS_BIAS_CAMERA_EXPOSURE_MODE,
) -> None:
    if channel != "rgb":
        _capture_deferred(bridge, target, view, channel, width, height, timeout)
        return
    # Editor-written, cross-uid on a shared volume — see _capture_deferred.
    mkdir_shared(target.parent)
    target.unlink(missing_ok=True)
    payload = {
        "filepath": str(target),
        "width": width,
        "height": height,
        "camera_location": view.location,
        "camera_rotation": view.rotation,
        "field_of_view": view.fov_deg,
    }
    if exposure_bias_ev is not None:
        payload["exposure_bias_ev"] = float(exposure_bias_ev)
        payload["exposure_mode"] = camera_exposure_mode
    reply = bridge.command("take_screenshot", payload, timeout=timeout)
    # An editor that reported a failure is not one to wait on: the wait below
    # exists for a shot that IS coming, and this one said it is not.
    if isinstance(reply, dict) and reply.get("status") == "error":
        raise RenderError(
            f"the editor refused the screenshot at {target}: "
            f"{reply.get('error', 'no reason given')}")
    if exposure_bias_ev is not None and (
        not isinstance(reply, dict)
        or reply.get("camera_exposure_policy")
        != camera_exposure_mode
        or float(reply.get("camera_exposure_bias_ev", float("nan")))
        != float(exposure_bias_ev)
    ):
        raise RenderError(
            "the editor did not confirm the requested camera exposure bias "
            f"{exposure_bias_ev}"
        )
    # The editor finishes the shot on a later frame and returns before the file
    # is on disk, so checking here decides against a write that has not
    # happened yet. Measured on a 736-actor scene: the frame landed two seconds
    # after the call returned, every viewpoint was recorded as having produced
    # nothing, and semantic_requirements failed for all seven requirements with
    # "no viewpoint produced an image" — beside the image.
    #
    # Longer than the deferred channels wait, because those get an explicit
    # `finished: True` from the editor before they start watching and this has
    # no handshake at all — but still bounded, because a shot that is never
    # coming has to fail in reasonable time. The measured lag was two seconds.
    _wait_for_file(target, min(timeout, 15.0))


def _capture_rgb_batch(
    bridge: Bridge,
    values: Sequence[tuple[Viewpoint, Path]],
    width: int,
    height: int,
    timeout: float,
    exposure_bias_ev: float | None = None,
    camera_exposure_mode: str = POST_PROCESS_BIAS_CAMERA_EXPOSURE_MODE,
) -> set[Path] | None:
    """Capture one or more RGB poses through one internal bridge request.

    New compatibility bridges execute the whole pose sweep with one persistent
    CameraActor and one UE Python submission.  ``None`` means the deployed
    bridge predates that internal command, so callers transparently retain the
    one-frame path instead of making the verifier version-dependent.
    """

    if not values:
        return None
    shots: list[dict[str, Any]] = []
    for view, target in values:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        shot = {
            "filepath": str(target),
            "width": width,
            "height": height,
            "camera_location": list(view.location),
            "camera_rotation": list(view.rotation),
            "field_of_view": float(view.fov_deg),
        }
        if exposure_bias_ev is not None:
            shot["exposure_bias_ev"] = float(exposure_bias_ev)
            shot["exposure_mode"] = camera_exposure_mode
        shots.append(shot)
    try:
        response = bridge.command(
            "take_screenshot_batch",
            {"shots": shots},
            timeout=max(timeout, timeout * len(values)),
        )
    except Exception:  # noqa: BLE001 - old/fake bridge capability fallback
        return None
    if (
        not isinstance(response, dict)
        or response.get("camera_policy") != "single-camera-slate-tick-batch"
        or response.get("status") not in {"success", "partial"}
    ):
        return None
    if exposure_bias_ev is not None and (
        response.get("camera_exposure_policy")
        != camera_exposure_mode
        or float(response.get("camera_exposure_bias_ev", float("nan")))
        != float(exposure_bias_ev)
    ):
        return None
    return {
        target
        for _, target in values
        if target.is_file() and target.stat().st_size > 0
    }


#: Below this mean luma (0-255) a frame shows a viewer nothing at all. It is not
#: "dark": the black frames this was written for measure exactly 0.0 across every
#: pixel of all four viewpoints, while the darkest frame anyone could still read
#: content in measured 2.4.
BLACK_LUMA = 3.0

#: The view mode a black frame would ideally be re-shot in — base colour with
#: the lighting term dropped, which answers "what is in this scene" when "how
#: does this scene look" has no answer.
#:
#: It does not work here, and that is measured rather than assumed: on the lab
#: editor `viewmode unlit` leaves all four frames at 0.0. `viewmode` is a
#: VIEWPORT console command and the screenshot tool does not photograph the
#: viewport. The real unlit channel is `_capture_deferred`'s SCS_BASE_COLOR
#: pass above — implemented since the multichannel work — and the judge's
#: frozen policy requires it precisely so scoring never depends on lighting.
#: This fallback remains for the RGB channel alone.
UNLIT_MODE = "unlit"

#: So the fallback lights the scene for the photograph instead, with a rig of
#: its own that is removed afterwards. Labels are prefixed so the cleanup can
#: find them even if this process dies between the two.
#:
#: What made this necessary: `setup_environment` aims the sun it creates with
#: `unreal.Rotator(-45, 30, 0)` intending pitch -45 and yaw 30 — but that
#: constructor takes (roll, pitch, yaw) positionally, so the sun ends up rolled
#: 45 degrees and pitched THIRTY UP, shining into the sky. The level has a sun,
#: it is visible, it affects the world, and it lights nothing. Twenty of the
#: first sixty frames are black for that reason and no other, so charging the
#: agent's evidence for it would be scoring the tool's bug as the model's work.
RELIT_PREFIX = "_SbRelight_"


def mean_luma(path: Path | str) -> float | None:
    """Average brightness of an image, or None if it cannot be read."""
    try:
        from PIL import Image
    except ImportError:      # pragma: no cover - depends on the environment
        return None
    try:
        with Image.open(path) as image:
            small = image.convert("L").resize((64, 36))
    except Exception:
        return None
    pixels = list(small.tobytes())
    return sum(pixels) / len(pixels) if pixels else None


def image_quality(path: Path | str) -> dict[str, float] | None:
    """Return robust low-resolution RGB acquisition statistics for one frame.

    The tile metrics intentionally describe image structure rather than scene
    semantics.  They let an evidence policy detect a camera pressed against a
    large wall (or otherwise dominated by one flat foreground region) without
    asking a model to choose or approve the camera.
    """

    try:
        from PIL import Image
    except ImportError:      # pragma: no cover - depends on the environment
        return None
    try:
        with Image.open(path) as image:
            raw = image.convert("RGB").resize((64, 36)).tobytes()
            pixels = list(zip(raw[0::3], raw[1::3], raw[2::3], strict=True))
    except Exception:
        return None
    if not pixels:
        return None
    lumas_unsorted = [
        0.2126 * red + 0.7152 * green + 0.0722 * blue
        for red, green, blue in pixels
    ]
    lumas = sorted(lumas_unsorted)

    def percentile(fraction: float) -> float:
        index = round((len(lumas) - 1) * fraction)
        return float(lumas[index])

    gradients = []
    width = 64
    height = 36
    for row in range(height):
        offset = row * width
        gradients.extend(
            abs(lumas_unsorted[offset + column + 1] - lumas_unsorted[offset + column])
            for column in range(width - 1)
        )
    gradients.extend(
        abs(lumas_unsorted[index + width] - lumas_unsorted[index])
        for index in range(width * (height - 1))
    )

    tile_columns = 8
    tile_rows = 6
    tile_width = width // tile_columns
    tile_height = height // tile_rows
    flat_tiles: list[list[bool]] = []
    for tile_row in range(tile_rows):
        row_values: list[bool] = []
        row_start = tile_row * tile_height
        for tile_column in range(tile_columns):
            column_start = tile_column * tile_width
            tile_lumas = [
                lumas_unsorted[row * width + column]
                for row in range(row_start, row_start + tile_height)
                for column in range(column_start, column_start + tile_width)
            ]
            tile_gradients = [
                abs(
                    lumas_unsorted[row * width + column + 1]
                    - lumas_unsorted[row * width + column]
                )
                for row in range(row_start, row_start + tile_height)
                for column in range(
                    column_start,
                    column_start + tile_width - 1,
                )
            ]
            tile_gradients.extend(
                abs(
                    lumas_unsorted[(row + 1) * width + column]
                    - lumas_unsorted[row * width + column]
                )
                for row in range(row_start, row_start + tile_height - 1)
                for column in range(column_start, column_start + tile_width)
            )
            tile_edge_fraction = (
                sum(value >= 10.0 for value in tile_gradients)
                / float(len(tile_gradients))
                if tile_gradients
                else 0.0
            )
            row_values.append(
                max(tile_lumas) - min(tile_lumas) <= 18.0
                and tile_edge_fraction <= 0.12
            )
        flat_tiles.append(row_values)

    largest_flat_component = 0
    visited: set[tuple[int, int]] = set()
    for tile_row in range(tile_rows):
        for tile_column in range(tile_columns):
            origin = (tile_row, tile_column)
            if not flat_tiles[tile_row][tile_column] or origin in visited:
                continue
            pending = [origin]
            visited.add(origin)
            component_size = 0
            while pending:
                row, column = pending.pop()
                component_size += 1
                for adjacent in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    adjacent_row, adjacent_column = adjacent
                    if not (
                        0 <= adjacent_row < tile_rows
                        and 0 <= adjacent_column < tile_columns
                    ):
                        continue
                    if adjacent in visited:
                        continue
                    if not flat_tiles[adjacent_row][adjacent_column]:
                        continue
                    visited.add(adjacent)
                    pending.append(adjacent)
            largest_flat_component = max(
                largest_flat_component,
                component_size,
            )

    central_tiles = [
        flat_tiles[row][column]
        for row in range(1, tile_rows - 1)
        for column in range(2, tile_columns - 2)
    ]
    flat_tile_count = sum(value for row in flat_tiles for value in row)
    tile_count = float(tile_rows * tile_columns)
    count = float(len(pixels))
    return {
        "mean_luma": float(sum(lumas) / count),
        "p05_luma": percentile(0.05),
        "p50_luma": percentile(0.50),
        "p95_luma": percentile(0.95),
        "dark_fraction": sum(value <= 8.0 for value in lumas) / count,
        "white_clip_fraction": sum(
            red >= 250 and green >= 250 and blue >= 250
            for red, green, blue in pixels
        ) / count,
        "channel_clip_fraction": sum(
            red >= 250 or green >= 250 or blue >= 250
            for red, green, blue in pixels
        ) / count,
        "detail_edge_fraction": (
            sum(value >= 10.0 for value in gradients) / float(len(gradients))
            if gradients else 0.0
        ),
        "p99_spatial_gradient": (
            float(sorted(gradients)[round((len(gradients) - 1) * 0.99)])
            if gradients else 0.0
        ),
        "flat_tile_fraction": flat_tile_count / tile_count,
        "largest_flat_component_fraction": (
            largest_flat_component / tile_count
        ),
        "central_flat_tile_fraction": (
            sum(central_tiles) / float(len(central_tiles))
            if central_tiles
            else 0.0
        ),
    }


def set_view_mode(bridge: Bridge, mode: str, timeout: float = 60.0) -> bool:
    """Switch the editor viewport's view mode; True if the command went through.

    There is no Python API for this, so it goes through the console — which
    means a failure is silent on the editor's side and has to be caught here.
    """
    script = "\n".join([
        "import unreal",
        "_w = unreal.EditorLevelLibrary.get_editor_world()",
        f"unreal.SystemLibrary.execute_console_command(_w, 'viewmode {mode}')",
        "globals()['_SB_VIEWMODE'] = {'ok': True}",
    ])
    try:
        bridge.exec_python("globals().pop('_SB_VIEWMODE', None)", timeout=60.0)
        return bool(bridge.exec_python_result(script, "_SB_VIEWMODE",
                                              timeout=timeout)["ok"])
    except BridgeError:
        return False


def add_deterministic_fill_rig(
    bridge: Bridge,
    timeout: float = 300.0,
    *,
    rig_spec: Sequence[dict[str, Any]] = DETERMINISTIC_FILL_RIG,
    policy: str = DETERMINISTIC_RELIGHT_POLICY,
) -> dict[str, Any]:
    """Install one frozen paired fill rig and return measured provenance."""

    specs = [dict(value) for value in rig_spec]
    rig_sha256 = hashlib.sha256(
        json.dumps(specs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    script = "\n".join([
        "import unreal",
        "_subs = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)",
        f"_specs = {specs!r}",
        "_records = []",
        "_errors = []",
        "for _spec in _specs:",
        "    _actor = None",
        "    try:",
        "        _cls = getattr(unreal, _spec['class'])",
        "        _actor = _subs.spawn_actor_from_class(",
        "            _cls, unreal.Vector(*_spec['location_cm']),",
        "            unreal.Rotator(*_spec['rotation_deg']))",
        "        if not _actor:",
        "            raise RuntimeError('spawn returned no actor')",
        "        _actor.set_actor_label(_spec['label'])",
        "        if _spec['class'] == 'DirectionalLight':",
        "            _component = _actor.get_component_by_class(",
        "                unreal.DirectionalLightComponent)",
        "            if not _component:",
        "                raise RuntimeError('missing DirectionalLightComponent')",
        "            try:",
        "                _component.set_mobility(unreal.ComponentMobility.MOVABLE)",
        "            except Exception:",
        "                pass",
        "            _component.set_editor_property(",
        "                'intensity', float(_spec['intensity']))",
        "            _component.set_editor_property(",
        "                'indirect_lighting_intensity',",
        "                float(_spec['indirect_lighting_intensity']))",
        "            _actual = {",
        "                'intensity': float(_component.get_editor_property('intensity')),",
        "                'indirect_lighting_intensity': float(_component.get_editor_property('indirect_lighting_intensity')),",
        "            }",
        "        else:",
        "            _component = _actor.get_component_by_class(",
        "                unreal.SkyLightComponent)",
        "            if not _component:",
        "                raise RuntimeError('missing SkyLightComponent')",
        "            try:",
        "                _component.set_mobility(unreal.ComponentMobility.MOVABLE)",
        "            except Exception:",
        "                pass",
        "            _component.set_editor_property(",
        "                'intensity', float(_spec['intensity']))",
        "            try:",
        "                _component.recapture_sky()",
        "            except Exception:",
        "                pass",
        "            _actual = {",
        "                'intensity': float(_component.get_editor_property('intensity')),",
        "            }",
        "        _records.append({",
        "            'class': _spec['class'],",
        "            'actor_name': _actor.get_name(),",
        "            'actor_label': _actor.get_actor_label(),",
        "            'configured': True,",
        "            'actual': _actual,",
        "        })",
        "    except Exception as _error:",
        "        _errors.append({'class': _spec.get('class'), 'error': str(_error)})",
        "globals()['_SB_DETERMINISTIC_FILL'] = {",
        f"    'policy': {policy!r},",
        f"    'rig_sha256': {rig_sha256!r},",
        "    'rig_spec': _specs,",
        "    'spawned': len([value for value in _records if value.get('actor_name')]),",
        "    'configured': len([value for value in _records if value.get('configured')]),",
        "    'records': _records,",
        "    'errors': _errors,",
        "    'saved_level': False,",
        "}",
    ])
    try:
        bridge.exec_python(
            "globals().pop('_SB_DETERMINISTIC_FILL', None)", timeout=60.0
        )
        result = bridge.exec_python_result(
            script,
            "_SB_DETERMINISTIC_FILL",
            timeout=timeout,
        )
    except BridgeError as error:
        raise RenderError(
            f"deterministic paired fill installation failed: {error}"
        ) from error
    if not isinstance(result, dict):
        raise RenderError("editor returned no deterministic fill provenance")
    return result


def add_relight_rig(bridge: Bridge, timeout: float = 300.0) -> int:
    """Add a configured neutral key and sky light; return the spawn count.

    Aimed DOWN — `unreal.Rotator(roll, pitch, yaw)` positionally, pitch
    negative — which is the bug this exists to photograph around.  Merely
    spawning the classes leaves engine-default intensities and an uncaptured
    SkyLight; enclosed rooms can remain black even though both Actors exist.
    Reuse the frozen deterministic rig values, but retain the historical int
    return contract used by caption/reference acquisition.
    """
    script = "\n".join([
        "import unreal",
        "_s = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)",
        f"_p = {RELIT_PREFIX!r}",
        f"_specs = {[dict(value) for value in DETERMINISTIC_FILL_RIG]!r}",
        "_n = 0",
        "_created = []",
        "for _spec in _specs:",
        "    try:",
        "        _cls = getattr(unreal, _spec['class'])",
        "        _a = _s.spawn_actor_from_class(",
        "            _cls, unreal.Vector(*_spec['location_cm']),",
        "            unreal.Rotator(*_spec['rotation_deg']))",
        "        if _a:",
        "            _created.append(_a)",
        "            _a.set_actor_label(_p + _spec['class'])",
        "            if _spec['class'] == 'DirectionalLight':",
        "                _c = _a.get_component_by_class(",
        "                    unreal.DirectionalLightComponent)",
        "                if not _c:",
        "                    raise RuntimeError('missing DirectionalLightComponent')",
        "                try:",
        "                    _c.set_mobility(unreal.ComponentMobility.MOVABLE)",
        "                except Exception:",
        "                    pass",
        "                _c.set_editor_property(",
        "                    'intensity', float(_spec['intensity']))",
        "                _c.set_editor_property(",
        "                    'indirect_lighting_intensity',",
        "                    float(_spec['indirect_lighting_intensity']))",
        "            else:",
        "                _c = _a.get_component_by_class(",
        "                    unreal.SkyLightComponent)",
        "                if not _c:",
        "                    raise RuntimeError('missing SkyLightComponent')",
        "                try:",
        "                    _c.set_mobility(unreal.ComponentMobility.MOVABLE)",
        "                except Exception:",
        "                    pass",
        "                _c.set_editor_property(",
        "                    'intensity', float(_spec['intensity']))",
        "                try:",
        "                    _c.recapture_sky()",
        "                except Exception:",
        "                    pass",
        "            _n += 1",
        "    except Exception:",
        "        pass",
        "if _n != len(_specs):",
        "    for _a in reversed(_created):",
        "        try:",
        "            _s.destroy_actor(_a)",
        "        except Exception:",
        "            pass",
        "    _n = 0",
        "globals()['_SB_RELIGHT'] = {'spawned': _n}",
    ])
    try:
        bridge.exec_python("globals().pop('_SB_RELIGHT', None)", timeout=60.0)
        spawned = int(bridge.exec_python_result(
            script, "_SB_RELIGHT", timeout=timeout)["spawned"])
    except (BridgeError, KeyError, TypeError, ValueError):
        remove_relight_rig(bridge, timeout=timeout)
        return 0
    if spawned != len(DETERMINISTIC_FILL_RIG):
        remove_relight_rig(bridge, timeout=timeout)
        return 0
    return spawned


def remove_relight_rig(bridge: Bridge, timeout: float = 300.0) -> int:
    """Take the photographer's lamps back out; returns how many were removed."""
    script = "\n".join([
        "import unreal",
        "_s = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)",
        f"_p = {RELIT_PREFIX!r}",
        "_g = 0",
        "for _a in list(_s.get_all_level_actors()):",
        "    try:",
        "        if _a.get_actor_label().startswith(_p):",
        "            _s.destroy_actor(_a)",
        "            _g += 1",
        "    except Exception:",
        "        pass",
        "globals()['_SB_UNRELIGHT'] = {'removed': _g}",
    ])
    try:
        bridge.exec_python("globals().pop('_SB_UNRELIGHT', None)", timeout=60.0)
        return int(bridge.exec_python_result(script, "_SB_UNRELIGHT",
                                             timeout=timeout)["removed"])
    except BridgeError:
        return 0


def hide(bridge: Bridge, labels: Sequence[str], hidden: bool = True,
         timeout: float = 120.0) -> int:
    """Temporarily hide Actors by label; returns how many were affected.

    Editor-only visibility, so it changes what a camera sees and nothing
    else — no transform, no property, nothing a measurement would read. The
    caller is responsible for putting them back, and `capture` does.
    """
    script = "\n".join([
        "import unreal",
        "_subs = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)",
        f"_want = set({list(labels)!r})",
        f"_hidden = {bool(hidden)!r}",
        "_n = 0",
        "for _a in list(_subs.get_all_level_actors()):",
        "    try:",
        "        if _a.get_actor_label() in _want:",
        "            _a.set_is_temporarily_hidden_in_editor(_hidden)",
        "            _n += 1",
        "    except Exception:",
        "        pass",
        "globals()['_SB_HIDE'] = {'affected': _n}",
    ])
    bridge.exec_python("globals().pop('_SB_HIDE', None)", timeout=60.0)
    return int(bridge.exec_python_result(script, "_SB_HIDE",
                                         timeout=timeout)["affected"])


def capture(bridge: Bridge, out_dir: Path, half_extent_m: float | None = None,
            count: int = DEFAULT_VIEWS, width: int = DEFAULT_WIDTH,
            height: int = DEFAULT_HEIGHT,
            timeout: float = 300.0,
            views: Sequence[Viewpoint] | None = None,
            channels: Sequence[str] = ("rgb",),
            hidden_labels: Sequence[str] = (),
            allow_empty: bool = False,
            retain_last_camera: bool = False,
            lighting_policy: str | None = None,
            exposure_policy: str | None = None,
            camera_exposure_mode: str = POST_PROCESS_BIAS_CAMERA_EXPOSURE_MODE,
            allow_relight_fallback: bool = True,
            capture_overrides: list[dict[str, Any]] | None = None,
            stop_after_rgb: Callable[[Viewpoint, Path], bool] | None = None,
            ) -> list[str]:
    """Photograph the current scene; return the paths that actually appeared.

    A viewpoint that fails is skipped rather than fatal — a judge works from
    three images as well as four — but a pass that produced NOTHING raises,
    because handing the judge an empty list would have it decline to score and
    that would read as the scene's fault.

    ``views`` overrides the default ring. An eval that compares the same camera
    across several scenes supplies its own, and must: cameras computed
    separately per scene would differ by framing as well as by content.

    ``stop_after_rgb`` makes an RGB-only probe sequential.  It is evaluated
    immediately after each readable frame and can stop the remaining views
    while the same temporary lighting context is still active.  Formal final
    evidence never uses it; it exists so one black pilot frame can trigger a
    different acquisition setting without paying for the rest of the probe.
    """
    check_channels(channels)
    if stop_after_rgb is not None and tuple(channels) != ("rgb",):
        raise RenderError(
            "stop_after_rgb is only valid for an RGB-only probe capture"
        )
    out_dir = Path(out_dir)
    # Editor-written, cross-uid on a shared volume — see _capture_deferred.
    mkdir_shared(out_dir)
    exposure_level = PHOTOMETRY_EXPOSURE_BY_POLICY.get(exposure_policy)
    exposure_bias_ev = (
        float(exposure_level["bias_ev"])
        if exposure_level is not None
        else (
            0.0
            if exposure_policy == PAIRED_PHOTOMETRY_EXPOSURE_POLICY
            else None
        )
    )

    def shoot(view: Viewpoint, target: Path, channel: str = "rgb") -> bool:
        try:
            _capture_one(
                bridge,
                target,
                view,
                channel,
                width,
                height,
                timeout,
                exposure_bias_ev,
                camera_exposure_mode,
            )
        except (BridgeError, RenderError, OSError):
            return False
        return target.is_file() and target.stat().st_size > 0

    produced: list[str] = []
    shot: list[tuple[Viewpoint, Path]] = []
    actual_views = views if views is not None else views_for(half_extent_m, count)
    try:
        if hidden_labels:
            hide(bridge, hidden_labels, True, timeout=timeout)
        with normalized_capture(
            bridge, lighting_policy=lighting_policy,
            exposure_policy=exposure_policy,
            camera_exposure_mode=camera_exposure_mode,
            timeout=timeout,
        ) as overrides:
            if capture_overrides is not None:
                capture_overrides.extend(overrides)
            # Send the RGB subset through one bridge request so UE reuses a
            # single camera across the sweep.  This applies both to RGB-only
            # RequirementGraph capture and to the RGB portion of formal
            # multi-channel renders; Base Color and Scene Depth retain their
            # separate channel-addressed capture contract.
            rgb_batch = (
                tuple((view, _target(out_dir, view, "rgb")) for view in actual_views)
                if "rgb" in channels and stop_after_rgb is None
                else ()
            )
            batched_paths = (
                _capture_rgb_batch(
                    bridge,
                    rgb_batch,
                    width,
                    height,
                    timeout,
                    exposure_bias_ev,
                    camera_exposure_mode,
                )
                if rgb_batch
                else None
            )
            if batched_paths is not None:
                for view, target in rgb_batch:
                    if target in batched_paths:
                        produced.append(str(target))
                        shot.append((view, target))
            deferred_batch = tuple(
                (view, _target(out_dir, view, channel), channel)
                for view in actual_views
                for channel in channels
                if channel != "rgb"
            )
            batched_deferred_paths = (
                _capture_deferred_batch(
                    bridge, deferred_batch, width, height, timeout
                )
                if deferred_batch
                else None
            )
            if batched_deferred_paths is not None:
                produced.extend(str(path) for path in batched_deferred_paths)
            stop_requested = False
            for view in actual_views:
                for channel in channels:
                    target = _target(out_dir, view, channel)
                    if (
                        channel == "rgb"
                        and batched_paths is not None
                        and target in batched_paths
                    ):
                        continue
                    if (
                        channel != "rgb"
                        and batched_deferred_paths is not None
                        and target in batched_deferred_paths
                    ):
                        continue
                    if shoot(view, target, channel):
                        produced.append(str(target))
                        if channel == "rgb":
                            shot.append((view, target))
                            if (
                                stop_after_rgb is not None
                                and stop_after_rgb(view, target)
                            ):
                                stop_requested = True
                                break
                if stop_requested:
                    break
            if exposure_bias_ev is not None:
                exposure_audits = [
                    value
                    for value in overrides
                    if value.get("policy") == exposure_policy
                ]
                if len(exposure_audits) != 1:
                    raise RenderError(
                        "camera exposure capture produced invalid provenance"
                    )
                exposure_audits[0].update({
                    "camera_override_applied": bool(shot),
                    "camera_override_applied_view_count": len(shot),
                })

        # The black-frame fallback remains as a second, separately audited
        # guard for maps not covered by the exact known-sun normalization.
        dark = [
            (view, target)
            for view, target in shot
            if (luma := mean_luma(target)) is not None and luma < BLACK_LUMA
        ]
        modes = {target.name: "lit" for _, target in shot}
        if allow_relight_fallback and dark and add_relight_rig(bridge, timeout=timeout):
            relit_views: list[str] = []
            ineffective_views: list[str] = []
            try:
                for view, target in dark:
                    # Keep the original: black output remains an auditable
                    # finding even when a fallback image is usable.
                    kept = target.with_suffix(".lit.png")
                    target.replace(kept)
                    if shoot(view, target) and (mean_luma(target) or 0.0) >= BLACK_LUMA:
                        modes[target.name] = "relit"
                        relit_views.append(view.name)
                    else:
                        modes[target.name] = "relight_ineffective"
                        ineffective_views.append(view.name)
                        kept.replace(target)
            finally:
                remove_relight_rig(bridge, timeout=timeout)
            if capture_overrides is not None:
                capture_overrides.append(
                    {
                        "policy": "black-rgb-relight-fallback",
                        "relit_views": relit_views,
                        "ineffective_views": ineffective_views,
                        "saved_level": False,
                    }
                )
        if shot:
            (out_dir / "capture_modes.json").write_text(json.dumps(modes, indent=2))
    finally:
        # A failed capture may still leave hidden actors or a transient camera.
        if hidden_labels:
            hide(bridge, hidden_labels, False, timeout=timeout)
        # A session-owning Stage 2/3 provider retains the last camera until
        # close to avoid destroying an RHI object still in use.
        if not retain_last_camera:
            remove_shot_cameras(bridge)

    if not produced and not allow_empty:
        raise RenderError(
            f"no viewpoint produced an image in {out_dir}. The editor writes "
            f"these, so this path has to be one it can write and this process "
            f"can read — the same two-sided path the measurement pass needs.")
    return produced


def capture_set(bridge: Bridge, out_dir: Path, scene: str,
                half_extent_m: float | None = None,
                count: int = DEFAULT_VIEWS, width: int = DEFAULT_WIDTH,
                height: int = DEFAULT_HEIGHT,
                timeout: float = 300.0,
                views: Sequence[Viewpoint] | None = None,
                channels: Sequence[str] = CHANNELS,
                hidden_labels: Sequence[str] = (),
                renders: RenderSet | None = None,
                lighting_policy: str | None = None,
                exposure_policy: str | None = None,
                camera_exposure_mode: str = POST_PROCESS_BIAS_CAMERA_EXPOSURE_MODE,
                camera_protocol: str = RING_PROTOCOL,
                scene_anchor_cm: Sequence[float] = (0.0, 0.0, 0.0),
                allow_relight_fallback: bool = True,
                ) -> RenderSet:
    """Capture one scene into the addressed render contract.

    A caller can pass the same ``RenderSet`` while it loads Candidate and GT
    in turn.  Pairing then proves equality of view id and channel instead of
    relying on list position or file-name coincidence.
    """
    actual_views = list(views if views is not None else views_for(half_extent_m, count))
    result = renders or RenderSet()
    anchor = [float(value) for value in scene_anchor_cm]
    if len(anchor) != 3:
        raise RenderError("scene_anchor_cm must contain exactly three coordinates")
    result.scene_anchors_cm[scene] = anchor
    result.capture_settings[scene] = {
        "camera_protocol": camera_protocol,
        "width": int(width),
        "height": int(height),
        "lighting_policy": lighting_policy,
        "exposure_policy": exposure_policy,
        "camera_exposure_mode": camera_exposure_mode,
    }
    camera_specs = {
        view.name: {
            "location_cm": [
                round(float(value), 6) for value in view.location
            ],
            "relative_location_cm": [
                round(float(value) - anchor[index], 6)
                for index, value in enumerate(view.location)
            ],
            "rotation_deg": [float(value) for value in view.rotation],
            "fov_deg": float(view.fov_deg),
            "width": int(width),
            "height": int(height),
            "camera_protocol": camera_protocol,
            "lighting_policy": lighting_policy,
            "exposure_policy": exposure_policy,
            "camera_exposure_mode": camera_exposure_mode,
        }
        for view in actual_views
    }
    result.camera_specs.setdefault(scene, {}).update(camera_specs)
    produced = capture(
        bridge, out_dir, half_extent_m, count=count, width=width, height=height,
        timeout=timeout, views=actual_views, channels=channels,
        hidden_labels=hidden_labels, allow_empty=True,
        lighting_policy=lighting_policy,
        exposure_policy=exposure_policy,
        camera_exposure_mode=camera_exposure_mode,
        allow_relight_fallback=(
            allow_relight_fallback and camera_protocol not in GT_PAIRED_PROTOCOLS
        ),
        capture_overrides=result.capture_overrides)
    lookup = {str(_target(Path(out_dir), view, channel)): (view.name, channel)
              for view in actual_views for channel in channels}
    produced_set = set(produced)
    for path in produced:
        address = lookup.get(path)
        if address is not None:
            result.add(scene, address[0], address[1], path)
    for path, (view, channel) in lookup.items():
        if path not in produced_set:
            result.capture_failures.append({
                "scene": scene, "view": view, "channel": channel,
                "path": path,
                "failure_reason": "capture did not produce a non-empty artifact",
            })
    return result


def remove_shot_cameras(bridge: Bridge, timeout: float = 120.0) -> int:
    """Take the photographer out of the photograph.

    Measuring a scene must not be able to change it. Returns how many cameras
    were removed, so a pass that quietly stopped working is visible.
    """
    script = "\n".join([
        "import unreal",
        "_subs = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)",
        f"_label = {SHOT_CAMERA_LABEL!r}",
        "_gone = 0",
        "for _a in list(_subs.get_all_level_actors()):",
        "    try:",
        "        if _a.get_actor_label().startswith(_label):",
        "            _subs.destroy_actor(_a)",
        "            _gone += 1",
        "    except Exception:",
        "        pass",
        # Deferred captures run in UE's persistent Python namespace.  Release
        # strong references before collecting transient RHI resources.
        "for _name in ('_camera', '_component', '_texture'):",
        "    globals().pop(_name, None)",
        "try:",
        "    unreal.SystemLibrary.collect_garbage()",
        "except Exception:",
        "    pass",
        "globals()['_SB_SHOTCAM'] = {'removed': _gone}",
    ])
    try:
        bridge.exec_python("globals().pop('_SB_SHOTCAM', None)", timeout=60.0)
        return int(bridge.exec_python_result(script, "_SB_SHOTCAM",
                                             timeout=timeout)["removed"])
    except BridgeError:
        return 0


def render_report(paths: list[str]) -> dict[str, Any]:
    """What the run record keeps about the renders."""
    return {"views": len(paths), "images": sorted(paths)}


@dataclass
class RenderSet:
    """Renders addressed by the scene they came from and the camera used.

    A flat list is enough for a judge looking at one scene. It is not enough
    for an eval that compares the SAME camera across the canonical, corrupted
    and candidate scenes: those comparisons are only meaningful pairwise, by
    camera, and a list cannot say which image pairs with which.

    ``images[scene][view][channel] -> path``.
    """

    images: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    capture_failures: list[dict[str, str]] = field(default_factory=list)
    capture_overrides: list[dict[str, Any]] = field(default_factory=list)
    camera_specs: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    capture_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    scene_anchors_cm: dict[str, list[float]] = field(default_factory=dict)
    camera_plan_audit: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, scene: str, view: str, channel: str, path: str) -> None:
        self.images.setdefault(scene, {}).setdefault(view, {})[channel] = path

    def scenes(self) -> list[str]:
        return sorted(self.images)

    def paired(self, view: str, channel: str = "rgb") -> dict[str, str]:
        """The same camera and channel across every scene that has it."""
        return {scene: views[view][channel]
                for scene, views in self.images.items()
                if view in views and channel in views[view]}

    def as_dict(self) -> dict[str, Any]:
        return {"scenes": self.scenes(),
                "views": sorted({v for s in self.images.values() for v in s}),
                "images": self.images,
                "camera_specs": self.camera_specs,
                "capture_settings": self.capture_settings,
                "scene_anchors_cm": self.scene_anchors_cm,
                "camera_plan_audit": self.camera_plan_audit,
                "channel_validity": channel_validity(self),
                "capture_overrides": list(self.capture_overrides),
                "capture_failures": list(self.capture_failures)}


def channel_validity(renders: RenderSet) -> dict[str, Any]:
    """Report whether every addressed artifact exists and is non-empty."""
    records = []
    addressed = set()
    for scene, views in sorted(renders.images.items()):
        for view, channels in sorted(views.items()):
            for channel, value in sorted(channels.items()):
                addressed.add((scene, view, channel))
                path = Path(value)
                exists = path.is_file()
                size = path.stat().st_size if exists else 0
                records.append({
                    "scene": scene, "view": view, "channel": channel,
                    "path": str(path),
                    "status": "valid" if size > 0 else "not_evaluated",
                    "file_size_bytes": size,
                    "failure_reason": (
                        None if size > 0 else "render artifact is absent or empty"),
                })
    for failure in renders.capture_failures:
        key = (failure.get("scene", ""), failure.get("view", ""),
               failure.get("channel", ""))
        if key in addressed:
            continue
        records.append({
            "scene": key[0], "view": key[1], "channel": key[2],
            "path": failure.get("path", ""), "status": "not_evaluated",
            "file_size_bytes": 0,
            "failure_reason": (failure.get("failure_reason")
                               or "render capture failed"),
        })
    return {
        "policy": "artifact-presence-nonempty",
        "records": records,
        "valid_count": sum(item["status"] == "valid" for item in records),
        "not_evaluated_count": sum(
            item["status"] == "not_evaluated" for item in records),
    }


__all__ = ["CHANNELS", "DEFAULT_FOV_DEG", "DEFAULT_HEIGHT", "DEFAULT_RADIUS_CM",
           "DEFAULT_VIEWS", "DEFAULT_WIDTH", "EXPOSURE_NORMALIZATION_POLICY",
           "PAIRED_PHOTOMETRY_EXPOSURE_POLICY",
           "POST_PROCESS_BIAS_CAMERA_EXPOSURE_MODE",
           "PHOTOMETRY_CAMERA_EXPOSURE_MODE",
           "GT_PAIRED_PROTOCOLS", "HEIGHT_RADIUS_MULTIPLIER",
           "IMPLEMENTED_CHANNELS",
           "LIGHTING_NORMALIZATION_POLICY",
           "DETERMINISTIC_FILL_RIG",
           "DETERMINISTIC_FILL_RIG_SHA256",
           "PHOTOMETRY_FILL_BY_POLICY",
           "PHOTOMETRY_FILL_LEVELS_BY_LIGHTING_POLICY",
           "PHOTOMETRY_FILL_LEVELS",
           "PHOTOMETRY_EXPOSURE_LEVELS",
           "PHOTOMETRY_EXPOSURE_BY_POLICY",
           "PHOTOMETRY_FILL_RIG_SHA256",
           "PAIRED_PHOTOMETRY_LIGHTING_POLICY",
           "DETERMINISTIC_RELIGHT_POLICY",
           "RADIUS_HALF_EXTENT_MULTIPLIER",
           "content_framing", "RING_PROTOCOL", "SUPPORTED_RING_PROTOCOLS",
           "GT_CAPTION_ENVIRONMENT_SCENE_GRAPH_PROTOCOL",
           "REFERENCE_OVERVIEW_SCENE_GRAPH_PROTOCOL",
           "GT_PAIRED_ENVIRONMENT_SCENE_GRAPH_PROTOCOL",
           "GT_PAIRED_SCENE_GRAPH_PROTOCOLS",
           "RenderError", "RenderSet", "SHOT_CAMERA_LABEL", "Viewpoint",
           "capture", "capture_set", "channel_validity", "check_channels",
           "image_quality", "normalized_capture", "normalized_exposure",
           "normalized_lighting",
           "remove_shot_cameras", "render_report", "ring", "translated_views",
           "views_for"]
