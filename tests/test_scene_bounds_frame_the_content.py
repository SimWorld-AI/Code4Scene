"""Scout cameras frame the scene, not the plate it is allowed to occupy.

Two different facts wear the same type here. The world plate is a CONSTRAINT —
the region an agent may build in, centred on the world origin because that is
what the bounds pass enforces. The scene bounds are an OBSERVATION — where the
content actually ended up. `_scene_bounds` overwrote the second with the first
whenever a task declared `size_m`.

On Hangar, whose content spans X -80..11 m and Y -118..-24 m under a 380 m
plate, that moved the framing target 79 m off the scene and put the four
eye-level scouts on the plate edge at ±190 m looking inward at empty ground.
Stage 2's judge reported "the provided frames show only sky, clouds, and a
horizon line" and refused every visual claim in the bundle.
"""

from __future__ import annotations

from code4scene.evaluation.requirement_graph.actions import generate_scout_poses
from code4scene.evaluation.requirement_graph.evidence_adapter import _scene_bounds


class _Descriptor:
    """The one attribute `_scene_bounds` reads."""

    def __init__(
        self,
        bounds,
        *,
        actor_class=None,
        asset_path=None,
        actor_label=None,
        unreal_name=None,
    ):
        self.bounds = bounds
        self.actor_class = actor_class
        self.asset_path = asset_path
        self.actor_label = actor_label
        self.unreal_name = unreal_name


class _Bounds:
    def __init__(self, min_cm, max_cm):
        self.min_cm, self.max_cm = min_cm, max_cm


#: Hangar, in centimetres, as the answer key records it.
HANGAR = _Descriptor(_Bounds((-7964.0, -11783.0, 0.0), (1112.0, -2399.0, 1500.0)))
PLATE_M = 190.0     # half of the size_m: 380 the generator measures for it


def test_a_plate_does_not_replace_the_measured_content_box():
    bounds = _scene_bounds([HANGAR], PLATE_M)

    assert (bounds.min_x, bounds.max_x) == (-7964.0, 1112.0)
    assert (bounds.min_y, bounds.max_y) == (-11783.0, -2399.0), (
        "the plate is where building is allowed; it is not where the scene is")


def test_the_scouts_look_at_the_scene_and_not_at_the_origin():
    bounds = _scene_bounds([HANGAR], PLATE_M)
    centre_x, centre_y, _ = bounds.center_cm

    # Within the content, not 79 m away at the plate's centre.
    assert -7964.0 <= centre_x <= 1112.0
    assert -11783.0 <= centre_y <= -2399.0

    for pose in generate_scout_poses(bounds):
        assert abs(pose.x) <= 9000.0 and abs(pose.y) <= 13000.0, (
            f"a scout at ({pose.x:.0f}, {pose.y:.0f}) cm is outside the scene "
            f"it is supposed to photograph")


def test_eye_level_scouts_step_inside_content_bounds():
    bounds = _scene_bounds(
        [_Descriptor(_Bounds((-400.0, -448.9, 0.0), (445.2, 429.0, 300.0)))],
        None,
    )
    eye_scouts = generate_scout_poses(bounds)[:4]

    assert all(bounds.min_x < pose.x < bounds.max_x for pose in eye_scouts)
    assert all(bounds.min_y < pose.y < bounds.max_y for pose in eye_scouts)
    assert eye_scouts[0].x > bounds.min_x + 200.0
    assert eye_scouts[1].x < bounds.max_x - 200.0
    assert eye_scouts[2].y > bounds.min_y + 200.0
    assert eye_scouts[3].y < bounds.max_y - 200.0


def test_an_empty_scene_falls_back_to_the_plate():
    """The one case where the plate IS the answer: there is no content to
    describe, and the canvas is all there is to say about the scene."""
    bounds = _scene_bounds([], PLATE_M)

    assert (bounds.min_x, bounds.max_x) == (-19000.0, 19000.0)
    assert (bounds.min_y, bounds.max_y) == (-19000.0, 19000.0)


def test_no_plate_and_no_content_is_still_a_usable_box():
    bounds = _scene_bounds([], None)
    assert bounds.max_x > bounds.min_x and bounds.max_z > bounds.min_z


def test_sky_sphere_and_camera_proxy_do_not_expand_the_content_box():
    room = _Descriptor(_Bounds((-340.0, -375.0, 0.0), (360.0, 400.0, 300.0)))
    sky = _Descriptor(
        _Bounds(
            (-1638400.0, -1638400.0, -1638400.0),
            (1638400.0, 1638400.0, 1638400.0),
        ),
        actor_class="/Engine/EngineSky/BP_Sky_Sphere.BP_Sky_Sphere_C",
        asset_path="/Engine/EngineSky/SM_SkySphere.SM_SkySphere",
    )
    camera = _Descriptor(
        _Bounds((-900.0, -700.0, -900.0), (1100.0, 1300.0, 1100.0)),
        actor_class="/Script/Engine.CameraActor",
        actor_label="CameraActor2",
    )

    bounds = _scene_bounds([room, sky, camera], None)

    assert bounds.min_cm == (-340.0, -375.0, 0.0)
    assert bounds.max_cm == (360.0, 400.0, 300.0)
