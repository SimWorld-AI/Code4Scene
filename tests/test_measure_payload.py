"""Current measurement rules, including fixed 5 cm inclusive support."""

import sys
from types import SimpleNamespace

import pytest

from code4scene.evaluation import measure_payload as mp


def row(name, ox=0.0, oy=0.0, oz=100.0, ex=100.0, ey=100.0, ez=100.0,
        lx=None, ly=None, cls="StaticMeshActor"):
    """Measurement row in cm; pivot defaults to the bounds centre."""
    return {"n": name, "ox": ox, "oy": oy, "oz": oz,
            "ex": ex, "ey": ey, "ez": ez,
            "lx": ox if lx is None else lx, "ly": oy if ly is None else ly,
            "bot": oz - ez, "top": oz + ez, "cls": cls}


def report(rows, gh=mp.DEFAULT_GH_CM):
    return mp.compute(rows, gh)["report"]


def test_structural_collision_and_partner_map():
    # Two big boxes overlapping well beyond TOUCH on every axis.
    rep = report([row("a"), row("b", ox=50.0)])
    assert rep["collision_pairs"] == 1
    assert rep["structural_collision_pairs"] == 1
    assert rep["structural_collision_actors"] == ["a", "b"]
    assert rep["colliding_with"] == {"a": ["b"], "b": ["a"]}


def test_touch_slack_is_not_a_collision():
    # X-overlap exactly TOUCH_CM (5): 2*ex=200 wide boxes, centres 195 apart.
    rep = report([row("a"), row("b", ox=195.0)])
    assert rep["collision_pairs"] == 0


def test_flat_actors_never_collide():
    # ez below FLAT_EZ_CM makes a ground sheet; overlaps with it are ignored.
    sheet = row("ground", ez=10.0, oz=10.0, ex=5000.0, ey=5000.0)
    rep = report([sheet, row("crate")])
    assert rep["collision_pairs"] == 0


def test_small_pair_collides_but_is_not_structural():
    # Both below CLUTTER_R (radius = hypot(50,50) ≈ 70.7 < 80): clutter only.
    rep = report([row("cup1", ex=50.0, ey=50.0), row("cup2", ox=30.0, ex=50.0, ey=50.0)])
    assert rep["collision_pairs"] == 1
    assert rep["structural_collision_pairs"] == 0
    assert rep["structural_collision_actors"] == []


def test_floating_and_support():
    ground = row("ground", ez=10.0, oz=10.0, ex=5000.0, ey=5000.0)
    # ground.top=20; the supported interval is bot in [20, 25].
    seated = row("seated", oz=125.0)
    floater = row("floater", oz=400.0)        # bot=300: gap 280 → unsupported
    rep = report([ground, seated, floater])
    assert rep["floating"] == ["floater"]


def test_floating_support_uses_pivot_not_bounds_centre():
    # Frozen quirk: horizontal containment compares the support's bounds
    # centre against THIS actor's PIVOT. Same bounds, pivot moved away →
    # floating flips on, even though the geometry never moved.
    support = row("support", oz=100.0)                    # top=200, ex=100
    box = row("box", oz=305.0, ex=10.0, ey=10.0)          # bot=205, gap 5 <= FT
    assert mp.compute([support, box])["report"]["floating"] == []
    far_pivot = row("box", oz=305.0, ex=10.0, ey=10.0, lx=500.0)
    assert mp.compute([support, far_pivot])["report"]["floating"] == ["box"]


def test_oob_is_pivot_based():
    inside_bounds_outside_pivot = row("sneaky", ox=0.0, lx=14000.0)
    outside_bounds_inside_pivot = row("hangover", ox=14000.0, lx=0.0)
    rep = report([inside_bounds_outside_pivot, outside_bounds_inside_pivot])
    assert rep["out_of_bounds"] == ["sneaky"]


def test_actor_rows_round_to_metres():
    result = mp.compute([row("a", ox=123.4, ex=250.0)])
    actor = result["actors"][0]
    assert actor["x_m"] == 1.234
    assert actor["w_m"] == 5.0
    assert actor["big"] is True and actor["flat"] is False


@pytest.mark.parametrize("gap", [-.01, 0, .5, 1, 1.01, 5, 5.01, 8, 12, 12.01])
def test_support_gap_includes_flush_contact_but_stops_at_five_cm(gap):
    support = row("platform", oz=50, ez=50)
    box = row("box", oz=110 + gap, ez=10, ex=10, ey=10)
    expected = ["box"] if gap < 0 or gap > 5 else []
    assert mp.floating_actors([support, box]) == expected
    assert report([support, box])["floating"] == expected


@pytest.mark.parametrize("gap", [-.01, 0, .5, 1, 5, 5.01, 8, 12])
def test_world_ground_uses_the_same_five_cm_tolerance(gap):
    box = row("box", oz=10 + gap, ez=10)
    assert mp.floating_actors([box]) == (["box"] if gap > 5 else [])


def test_fixed_rule_has_no_alternate_threshold_or_legacy_mode():
    assert mp.DEFAULT_FT_CM == 5.0
    with pytest.raises(TypeError):
        mp.compute([], ft_cm=12)
    with pytest.raises(TypeError):
        mp.floating_actors([], legacy=True)


def test_horizontal_support_boundary_remains_strict():
    support = row("platform", oz=50, ez=50)
    touching = row("box", oz=110, ez=10, ex=10, ey=10, lx=105)
    assert mp.floating_actors([support, touching]) == ["box"]
    touching['lx'] = 104.99
    assert mp.floating_actors([support, touching]) == []


def test_saved_scene_rows_share_editor_filter_and_preserve_vertical_extents():
    def actor(name, cls="/Script/Engine.StaticMeshActor", extent=(.5, 10, .5)):
        return dict(label=name, **{'class': cls},
                    bounds=dict(origin_cm=[0, 0, 10], extent_cm=list(extent)),
                    transform=dict(location_cm=[20, 30, 0]))
    kept = actor("crate")
    rows = mp.rows_from_scene({'actors': [kept, actor("floor_helper"),
        actor("helper", cls="/Script/Engine.CameraActor"),
        actor("tiny", extent=(.5, .5, .5))]})
    assert rows == [mp._measurement_row("crate", "StaticMeshActor",
                                      [0, 0, 10], [.5, 10, .5], [20, 30, 0])]
    assert rows[0]['ex'] == rows[0]['ez'] == 1
    assert rows[0]['bot'] == 9.5 and rows[0]['top'] == 10.5


def test_editor_collection_matches_exported_scene_collection(monkeypatch):
    actors = []
    for label, cls, extents in [
        ("crate", "StaticMeshActor", [.5, 10, .5]),
        ("floor_helper", "StaticMeshActor", [10, 10, 1]),
        ("helper", "CameraActor", [10, 10, 10]),
        ("tiny", "StaticMeshActor", [.5, .5, .5]),
    ]:
        actors.append(dict(label=label, **{'class': '/Script/Engine.' + cls},
            bounds=dict(origin_cm=[0, 0, 10], extent_cm=extents),
            transform=dict(location_cm=[20, 30, 0])))

    class Actor:
        def __init__(self, data):
            self.data = data

        def get_class(self):
            return SimpleNamespace(get_name=lambda: self.data['class'].rsplit('.', 1)[-1])

        def get_actor_label(self):
            return self.data['label']

        def get_actor_bounds(self, only_colliding):
            return tuple(SimpleNamespace(**dict(zip(('x', 'y', 'z'), self.data['bounds'][k], strict=True)))
                         for k in ('origin_cm', 'extent_cm'))

        def get_actor_location(self):
            return SimpleNamespace(**dict(zip(('x', 'y', 'z'), self.data['transform']['location_cm'], strict=True)))

    unreal = SimpleNamespace(EditorActorSubsystem=object,
        get_editor_subsystem=lambda _: SimpleNamespace(
            get_all_level_actors=lambda: [Actor(data) for data in actors]))
    monkeypatch.setitem(sys.modules, 'unreal', unreal)
    assert mp.collect_rows() == mp.rows_from_scene({'actors': actors})
