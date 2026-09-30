"""Photographing the finished scene.

The judge scores images of the finished scene. What matters here is that the
framing is fixed — two runs' verdicts are comparable only if
what varies between them is the scene and not the camera.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from code4scene.evaluation import render


def test_the_cameras_ring_the_scene_and_look_at_its_centre():
    views = render.ring(count=4, radius_cm=10000, height_cm=5000)

    assert len(views) == 4
    for view in views:
        x, y, z = view.location
        assert math.isclose(math.hypot(x, y), 10000, rel_tol=1e-6)
        assert z == 5000
        # rotation is [roll, pitch, yaw]: the editor adapter passes it
        # positionally into unreal.Rotator, whose constructor order is
        # (roll, pitch, yaw). The previous ordering put pitch in the roll
        # slot, which rolled the horizon and aimed every render at nothing —
        # found when the judge's gallery came back as sky.
        inward = math.degrees(math.atan2(-y, -x))
        assert view.rotation[0] == 0.0, "no roll — a tilted horizon is a bug"
        assert view.rotation[1] < 0, "pitch looks down, not up"
        assert math.isclose(view.rotation[2], round(inward, 1), abs_tol=0.2), \
            "yaw points back at the origin"


def test_one_viewpoint_is_refused():
    """A single camera can be aimed at the only corner that was built, and a
    judge shown that image is being asked the wrong question."""
    with pytest.raises(render.RenderError, match="not a look at a scene"):
        render.ring(count=1)


def test_the_plate_decides_how_far_out_the_cameras_sit():
    """Fixed framing relative to the task, so the scene is what varies."""
    small = render.views_for(half_extent_m=50)
    large = render.views_for(half_extent_m=200)

    assert math.hypot(*small[0].location[:2]) < math.hypot(*large[0].location[:2])
    assert [v.name for v in small] == [v.name for v in large]


def test_a_task_with_no_plate_still_gets_framed():
    assert len(render.views_for(half_extent_m=None)) == render.DEFAULT_VIEWS


def test_known_setup_sun_fix_is_exact_temporary_and_recorded(tmp_path):
    calls = []

    class Bridge:
        def command(self, name, args, timeout=300.0):
            from pathlib import Path

            Path(args["filepath"]).write_bytes(b"\x89PNG")
            return {"status": "success"}

        def exec_python(self, script, timeout=60.0):
            return {}

        def exec_python_result(self, script, key, timeout=120.0):
            calls.append((key, script))
            if key == "_SB_LIGHTING_NORMALIZATION":
                assert "SimWorldStudio_Sun" in script
                assert "abs(_before['roll'] + 45.0) <= 0.25" in script
                return {
                    "policy": render.LIGHTING_NORMALIZATION_POLICY,
                    "records": [{
                        "actor_name": "DirectionalLight_0",
                        "actor_label": "SimWorldStudio_Sun",
                        "actor_class": "DirectionalLight",
                        "before": {"roll": -45.0, "pitch": 30.0, "yaw": 0.0},
                        "after": {"roll": 0.0, "pitch": -45.0, "yaw": 30.0},
                        "matched_known_bug": True,
                        "applied": True,
                    }],
                    "saved_level": False,
                }
            if key == "_SB_LIGHTING_RESTORE":
                assert "roll=_before['roll']" in script
                return {"restored": ["DirectionalLight_0"]}
            return {"removed": 1}

    overrides = []
    view = render.Viewpoint(
        "view_0", [1000.0, 0.0, 500.0], [0.0, -20.0, 180.0]
    )
    produced = render.capture(
        Bridge(), tmp_path, views=[view],
        lighting_policy=render.LIGHTING_NORMALIZATION_POLICY,
        capture_overrides=overrides,
    )

    assert len(produced) == 1
    assert overrides[0]["records"][0]["applied"] is True
    assert overrides[0]["saved_level"] is False
    assert [key for key, _ in calls].count("_SB_LIGHTING_RESTORE") == 1


def test_unknown_lighting_normalization_is_refused():
    with pytest.raises(render.RenderError, match="unknown lighting"):
        with render.normalized_lighting(object(), "auto_light_any_black_frame"):
            pass


def test_photometry_native_capture_clears_a_stale_fill(monkeypatch):
    removed = []

    def remove(_bridge, timeout=300.0):
        removed.append(timeout)
        return 2

    monkeypatch.setattr(render, "remove_relight_rig", remove)

    with render.normalized_lighting(
        object(), render.PAIRED_PHOTOMETRY_LIGHTING_POLICY
    ) as audit:
        assert audit["acquisition_setting"] == "native"
        assert audit["stale_removed_before_capture"] == 2

    assert removed == [120.0]


def test_photometry_fill_ladder_preserves_current_order_and_values():
    levels = render.PHOTOMETRY_FILL_LEVELS
    assert [value["id"] for value in levels] == [
        "neutral_fill_micro",
        "neutral_fill_very_low",
        "neutral_fill_low_30",
        "neutral_fill_low_50",
        "neutral_fill_minimal",
        "neutral_fill_low",
        "neutral_fill_weak",
        "neutral_fill_medium",
        "neutral_fill_strong",
    ]
    assert [value["rig_spec"][0]["intensity"] for value in levels] == [
        10.0, 25.0, 30.0, 50.0, 100.0, 250.0, 500.0, 1_500.0, 3_500.0,
    ]


def test_paired_exposure_declares_camera_override_and_restores_adaptation():
    scripts = []

    class Bridge:
        def exec_python(self, script, timeout=60.0):
            scripts.append(script)
            return {}

        def exec_python_result(self, script, key, timeout=120.0):
            scripts.append(script)
            assert key == "_SB_EXPOSURE_NORMALIZATION"
            assert "unreal.PostProcessVolume" not in script
            return {
                "policy": render.PAIRED_PHOTOMETRY_EXPOSURE_POLICY,
                "before": 2,
                "applied": True,
                "paired_bias_search": True,
                "bias_ev": 0.0,
                "camera_override_mode": (
                    render.POST_PROCESS_BIAS_CAMERA_EXPOSURE_MODE
                ),
                "camera_override_applied": False,
                "camera_override_applied_view_count": 0,
                "saved_level": False,
            }

    with render.normalized_exposure(
        Bridge(), render.PAIRED_PHOTOMETRY_EXPOSURE_POLICY
    ) as audit:
        assert audit["bias_ev"] == 0.0
        assert audit["camera_override_applied"] is False

    assert audit["camera_override_mode"] == (
        render.POST_PROCESS_BIAS_CAMERA_EXPOSURE_MODE
    )
    assert any("r.EyeAdaptationQuality 0" in script for script in scripts)
    assert any("r.EyeAdaptationQuality 2" in script for script in scripts)


def test_manual_exposure_enables_processing_but_remains_fixed():
    scripts = []

    class Bridge:
        def exec_python(self, script, timeout=60.0):
            scripts.append(script)
            return {}

        def exec_python_result(self, script, key, timeout=120.0):
            scripts.append(script)
            assert key == "_SB_EXPOSURE_NORMALIZATION"
            assert "r.EyeAdaptationQuality 2" in script
            return {
                "policy": render.PAIRED_PHOTOMETRY_EXPOSURE_POLICY,
                "before": 0,
                "applied": True,
                "paired_bias_search": True,
                "bias_ev": 0.0,
                "camera_override_mode": (
                    render.PHOTOMETRY_CAMERA_EXPOSURE_MODE
                ),
                "camera_override_applied": False,
                "camera_override_applied_view_count": 0,
                "saved_level": False,
            }

    with render.normalized_exposure(
        Bridge(),
        render.PAIRED_PHOTOMETRY_EXPOSURE_POLICY,
        camera_exposure_mode=(
            render.PHOTOMETRY_CAMERA_EXPOSURE_MODE
        ),
    ) as audit:
        assert audit["camera_override_mode"] == (
            render.PHOTOMETRY_CAMERA_EXPOSURE_MODE
        )

    assert "r.EyeAdaptationQuality 0" in scripts[-1]


def test_manual_exposure_catalog_has_bounded_dark_rescue_headroom():
    levels = render.PHOTOMETRY_EXPOSURE_LEVELS
    assert max(float(level["bias_ev"]) for level in levels) == 8.0
    assert len({str(level["exposure_policy"]) for level in levels}) == len(
        levels
    )


def test_capture_applies_requested_bias_on_each_actual_camera(tmp_path):
    calls = []

    class Bridge:
        def command(self, name, args, timeout=300.0):
            calls.append((name, args))
            if name == "take_screenshot_batch":
                for shot in args["shots"]:
                    Path(shot["filepath"]).write_bytes(b"biased-frame")
                return {
                    "status": "success",
                    "camera_policy": "single-camera-slate-tick-batch",
                    "camera_exposure_policy": (
                        "camera-component-post-process-bias"
                    ),
                    "camera_exposure_bias_ev": -1.0,
                }
            raise AssertionError(name)

        def exec_python(self, script, timeout=60.0):
            return {}

        def exec_python_result(self, script, key, timeout=120.0):
            if key == "_SB_EXPOSURE_NORMALIZATION":
                return {
                    "policy": render.PHOTOMETRY_EXPOSURE_LEVELS[0][
                        "exposure_policy"
                    ],
                    "before": 2,
                    "applied": True,
                    "paired_bias_search": True,
                    "bias_ev": -1.0,
                    "camera_override_mode": (
                        "camera-component-post-process-bias"
                    ),
                    "camera_override_applied": False,
                    "camera_override_applied_view_count": 0,
                    "saved_level": False,
                }
            if key == "_SB_SHOTCAM":
                return {"removed": 1}
            raise AssertionError(key)

    views = render.ring(count=2, radius_cm=1000.0, height_cm=500.0)
    overrides = []
    produced = render.capture(
        Bridge(),
        tmp_path,
        views=views,
        exposure_policy=(
            render.PHOTOMETRY_EXPOSURE_LEVELS[0]["exposure_policy"]
        ),
        capture_overrides=overrides,
    )

    assert len(produced) == 2
    assert len(calls) == 1
    assert calls[0][0] == "take_screenshot_batch"
    assert {shot["exposure_bias_ev"] for shot in calls[0][1]["shots"]} == {
        -1.0
    }
    assert overrides[0]["camera_override_applied"] is True
    assert overrides[0]["camera_override_applied_view_count"] == 2


def test_single_rgb_view_batches_before_deferred_channels(
    tmp_path, monkeypatch,
):
    calls = []

    class Bridge:
        def command(self, name, args, timeout=300.0):
            calls.append("rgb_batch")
            assert name == "take_screenshot_batch"
            assert len(args["shots"]) == 1
            target = Path(args["shots"][0]["filepath"])
            target.write_bytes(b"rgb-frame")
            return {
                "status": "success",
                "camera_policy": "single-camera-slate-tick-batch",
            }

        def exec_python(self, script, timeout=60.0):
            return {}

        def exec_python_result(self, script, key, timeout=120.0):
            assert key == "_SB_SHOTCAM"
            return {"removed": 1}

    def fake_deferred(
        bridge, values, width, height, timeout,
    ):
        calls.append("deferred_batch")
        produced = set()
        for _, target, _ in values:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"deferred-frame")
            produced.add(target)
        return produced

    monkeypatch.setattr(render, "_capture_deferred_batch", fake_deferred)
    view = render.Viewpoint(
        "view_0", [1000.0, 0.0, 500.0], [0.0, -20.0, 180.0]
    )
    produced = render.capture(
        Bridge(),
        tmp_path,
        views=[view],
        channels=("rgb", "base_color", "scene_depth"),
    )

    assert len(produced) == 3
    assert calls[:2] == ["rgb_batch", "deferred_batch"]


def test_photometry_fill_installs_and_removes_the_selected_level(
    monkeypatch,
):
    level = next(
        value
        for value in render.PHOTOMETRY_FILL_LEVELS
        if value["id"] == "neutral_fill_weak"
    )
    policy = level["lighting_policy"]
    removals = []

    def remove(_bridge, timeout=300.0):
        removals.append(timeout)
        return 0 if len(removals) == 1 else len(level["rig_spec"])

    def add(_bridge, timeout=300.0, *, rig_spec, policy):
        assert rig_spec == level["rig_spec"]
        return {
            "policy": policy,
            "rig_sha256": render.PHOTOMETRY_FILL_RIG_SHA256[policy],
            "spawned": len(rig_spec),
            "configured": len(rig_spec),
            "errors": [],
        }

    monkeypatch.setattr(render, "remove_relight_rig", remove)
    monkeypatch.setattr(render, "add_deterministic_fill_rig", add)

    with render.normalized_lighting(object(), policy) as audit:
        assert audit["acquisition_setting"] == level["id"]

    assert audit["removed_after_capture"] == len(level["rig_spec"])
    assert len(removals) == 2


def test_only_images_that_appeared_are_reported(tmp_path):
    """A viewpoint that fails is skipped — a judge works from three as well as
    four — but what is reported has to be what exists."""
    made = {"n": 0}

    class Bridge:
        def command(self, name, args, timeout=300.0):
            made["n"] += 1
            if made["n"] != 2:                       # the second one fails
                from pathlib import Path

                Path(args["filepath"]).write_bytes(b"\x89PNG fake")
            return {"status": "success"}

        def exec_python(self, script, timeout=60.0):
            return {}

        def exec_python_result(self, script, key, timeout=120.0):
            return {"removed": 1}

    # The failing viewpoint never gets a file, and capture now waits for one
    # before giving up on it — so how long "never" takes is the timeout.
    produced = render.capture(Bridge(), tmp_path, half_extent_m=100, timeout=0.2)

    assert len(produced) == render.DEFAULT_VIEWS - 1
    assert all(p.endswith(".png") for p in produced)


def test_rgb_probe_can_stop_after_its_first_frame(tmp_path):
    calls = []

    class Bridge:
        def command(self, name, args, timeout=300.0):
            assert name == "take_screenshot"
            calls.append(args["filepath"])
            Path(args["filepath"]).write_bytes(b"\x89PNG probe")
            return {"status": "success"}

        def exec_python(self, script, timeout=60.0):
            return {}

        def exec_python_result(self, script, key, timeout=120.0):
            return {"removed": 1}

    views = render.ring(count=4, radius_cm=1000.0, height_cm=500.0)
    produced = render.capture(
        Bridge(),
        tmp_path,
        views=views,
        stop_after_rgb=lambda _view, _path: True,
    )

    assert len(calls) == 1
    assert produced == [str(tmp_path / "view_0.png")]


def test_a_pass_that_produced_nothing_raises(tmp_path):
    """Handing the judge an empty list makes it decline, and that would read
    as the scene's fault."""
    class Bridge:
        def command(self, name, args, timeout=300.0):
            return {"status": "success"}             # writes no file

        def exec_python(self, script, timeout=60.0):
            return {}

        def exec_python_result(self, script, key, timeout=120.0):
            return {"removed": 0}

    # A short timeout because this editor never writes: capture waits for a
    # file the real one finishes after the call returns, and "never" has to
    # be spelled as a deadline.
    with pytest.raises(render.RenderError, match="no viewpoint produced"):
        render.capture(Bridge(), tmp_path, timeout=0.2)


def test_the_photographer_is_taken_out_of_the_photograph(tmp_path):
    """The editor's screenshot tool destroys stale cameras at the START of a
    call and never removes the last one, and this pass runs BEFORE the
    verifiers — so the restoration scorer counted the camera as something the
    agent had added. Measuring a scene must not change it."""
    removed = {"n": 0}

    class Bridge:
        def command(self, name, args, timeout=300.0):
            from pathlib import Path

            Path(args["filepath"]).write_bytes(b"\x89PNG")
            return {"status": "success"}

        def exec_python(self, script, timeout=60.0):
            return {}

        def exec_python_result(self, script, key, timeout=120.0):
            assert render.SHOT_CAMERA_LABEL in script
            removed["n"] += 1
            return {"removed": 4}

    render.capture(Bridge(), tmp_path, half_extent_m=100)
    assert removed["n"] == 1, "cleaned up exactly once, after the shots"


def test_cleanup_runs_even_when_no_viewpoint_produced_an_image(tmp_path):
    """A viewpoint that failed can still have spawned its camera first."""
    seen = {"cleanup": False}

    class Bridge:
        def command(self, name, args, timeout=300.0):
            return {"status": "success"}          # writes nothing

        def exec_python(self, script, timeout=60.0):
            return {}

        def exec_python_result(self, script, key, timeout=120.0):
            seen["cleanup"] = True
            return {"removed": 1}

    with pytest.raises(render.RenderError):
        render.capture(Bridge(), tmp_path, timeout=0.2)   # never writes
    assert seen["cleanup"]


def test_session_provider_can_defer_camera_cleanup_until_close(tmp_path):
    removed = {"n": 0}

    class Bridge:
        def command(self, name, args, timeout=300.0):
            from pathlib import Path

            Path(args["filepath"]).write_bytes(b"\x89PNG")
            return {"status": "success"}

        def exec_python(self, script, timeout=60.0):
            return {}

        def exec_python_result(self, script, key, timeout=120.0):
            removed["n"] += 1
            return {"removed": 1}

    bridge = Bridge()
    render.capture(
        bridge,
        tmp_path,
        half_extent_m=100,
        retain_last_camera=True,
    )
    assert removed["n"] == 0
    render.remove_shot_cameras(bridge)
    assert removed["n"] == 1


def test_deferred_channels_share_one_camera_and_bridge_request(tmp_path):
    """Four addressed files must not require four SceneCapture2D requests."""
    views = render.ring(count=2, radius_cm=1000.0, height_cm=500.0)
    expected = {
        tmp_path / "base_color" / f"{view.name}.png"
        for view in views
    } | {
        tmp_path / "scene_depth" / f"{view.name}.exr"
        for view in views
    }
    calls = []

    class Bridge:
        def exec_python(self, script, timeout=60.0):
            return {}

        def exec_python_result(self, script, key, timeout=120.0):
            calls.append((key, script))
            if key == "_SB_RENDER_CHANNEL_BATCH":
                assert script.count("spawn_actor_from_class") == 1
                assert "for _spec in _specs:" in script
                assert "single-scene-capture-component-batch" in script
                for path in expected:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"deferred-channel")
                return {
                    "finished": True,
                    "camera_policy":
                    "single-scene-capture-component-batch",
                    "records": [],
                }
            if key == "_SB_SHOTCAM":
                return {"removed": 1}
            raise AssertionError(f"unexpected single-frame call {key}")

    produced = render.capture(
        Bridge(), tmp_path, views=views,
        channels=("base_color", "scene_depth"),
    )

    assert set(map(Path, produced)) == expected
    assert [key for key, _ in calls].count("_SB_RENDER_CHANNEL_BATCH") == 1
    assert all(key != "_SB_RENDER_CHANNEL" for key, _ in calls)


# ── the ring can stand somewhere other than the origin ────────────────────


def test_the_ring_aims_at_its_centre_not_at_the_origin():
    """Hangar's content is centred 79 m from the origin; an origin ring sized
    to its plate photographed sky with the scene as a distant silhouette.
    Third appearance of this defect (Stage 2 scouts, the plate measurement),
    same shape each time: a camera aimed at a coordinate system."""
    from code4scene.evaluation.render import ring

    centre = (-3430.0, -7090.0, -1500.0)
    views = ring(count=4, radius_cm=20000.0, center_cm=centre)

    xs = [v.location[0] for v in views]
    ys = [v.location[1] for v in views]
    assert abs(sum(xs) / 4 - centre[0]) < 1.0
    assert abs(sum(ys) / 4 - centre[1]) < 1.0, (
        "the ring must stand around the given centre; averaging its camera "
        "positions recovers it")
    for v in views:
        assert v.location[2] > centre[2], "cameras stand above the content floor"


def test_the_default_ring_is_unchanged():
    """Every existing caller framed the origin; the parameter must not move
    them."""
    from code4scene.evaluation.render import ring

    old = ring(count=4, radius_cm=10000.0)
    new = ring(count=4, radius_cm=10000.0, center_cm=(0.0, 0.0, 0.0))
    assert [v.location for v in old] == [v.location for v in new]
    assert [v.rotation for v in old] == [v.rotation for v in new]


def test_content_framing_is_the_one_placement_authority():
    """The origin-centred assumption was baked into three camera systems and
    failed the same way in each once cases derived from assembled
    environments. One implementation, imported by everyone, prevents the
    fourth occurrence — so it lives here, beside the ring it feeds."""
    from code4scene.evaluation.render import content_framing

    actors = [
        {"label": "SM_crate_1", "loc": [-7000.0, -11000.0, -150.0]},
        {"label": "SM_crate_2", "loc": [1000.0, -3000.0, 0.0]},
        {"label": "Sky Sphere", "loc": [0.0, 0.0, 0.0]},       # scenery: out
        {"label": "CameraActor"},                              # no loc: out
    ]
    center, reach = content_framing(actors)
    assert center == (-3000.0, -7000.0, -150.0)
    assert reach == 4000.0, "half the wider span, sky excluded"


def test_content_framing_of_nothing_is_a_usable_default():
    from code4scene.evaluation.render import content_framing

    center, reach = content_framing([{"label": "SkyLight", "loc": [0, 0, 0]}])
    assert center == (0.0, 0.0, 0.0) and reach > 0
