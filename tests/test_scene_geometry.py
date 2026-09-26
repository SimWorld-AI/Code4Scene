"""Shared geometry helpers the spatial verifiers all lean on."""

from __future__ import annotations

from code4scene.evaluation import scene_geometry


def box(x, *, key=None, extent=50.0):
    actor = {"bounds": {"origin_cm": [x, 0.0, 0.0],
                        "extent_cm": [extent, extent, extent]}}
    if key is not None:
        actor["label"] = key
    return actor


def test_unidentified_pairs_do_not_collapse_into_one():
    """Actors with no stable id, path or label used to share the identity
    str(None), so the first overlapping pair seen deduplicated all the rest
    out of the report — and an actor even "matched" itself across the two
    populations."""
    subjects = [box(0.0), box(1000.0)]
    others = [box(10.0), box(1010.0)]

    found = scene_geometry.aabb_overlaps_between(subjects, others)

    assert len(found) == 2, "two distinct overlapping pairs, both reported"


def test_the_same_actor_in_both_populations_is_not_its_own_collision():
    shared = box(0.0)
    found = scene_geometry.aabb_overlaps_between([shared], [shared, box(2000.0)])
    assert found == []


def test_identified_pairs_still_deduplicate_across_populations():
    a, b = box(0.0, key="a"), box(10.0, key="b")
    found = scene_geometry.aabb_overlaps_between([a, b], [a, b])
    assert len(found) == 1
