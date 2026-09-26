"""What a structured case declares about its own content, and what it refuses."""

from __future__ import annotations

from declared_cases import actor, case, check_by_id, context, scene

from code4scene.evaluation import contracts
from code4scene.evaluation.verifiers import (structure_additions,
                                             structure_concepts,
                                             structure_count)

TABLE = "/Game/Props/SM_DiningTable"
CHAIR = "/Game/Props/SM_Chair"

STRUCTURE = {
    "id": "dining-set",
    "primitive": "structure",
    "expected_raw_added_count": 5,
    "concepts": [
        {"concept": "dining table", "count": 1, "allowed_asset_paths": [TABLE]},
        {"concept": "chair", "count": 4, "allowed_asset_paths": [CHAIR]},
    ],
}


def dining_room(chairs: int = 4, extras: list = ()) -> tuple[dict, dict]:
    """(input, candidate) for a room the edit filled with a dining set."""
    floor = actor("floor", extent=(500.0, 500.0, 5.0))
    added = [actor("table", asset_path=TABLE, location=(0.0, 0.0, 40.0))]
    added += [actor(f"chair_{i}", asset_path=CHAIR,
                    location=(120.0 * (i + 1), 0.0, 40.0)) for i in range(chairs)]
    return scene([floor]), scene([floor, *added, *extras])


def test_the_declared_actor_count_is_what_the_edit_added(tmp_path):
    source, candidate = dining_room()
    report = structure_count.verify(context(
        tmp_path, candidate=candidate, input_scene=source,
        case_spec=case([STRUCTURE])))
    assert report["status"] == contracts.MEASURED
    assert report["score"] == 1.0
    assert check_by_id(report, "structure.actor_count")["observed"] == {
        "added_ue_actors": 5}


def test_one_chair_short_fails_the_count_and_the_concept(tmp_path):
    source, candidate = dining_room(chairs=3)
    counted = structure_count.verify(context(
        tmp_path, candidate=candidate, input_scene=source,
        case_spec=case([STRUCTURE])))
    assert counted["status"] == contracts.MEASURED
    assert counted["score"] == 0.0

    concepts = structure_concepts.verify(context(
        tmp_path, candidate=candidate, input_scene=source,
        case_spec=case([STRUCTURE])))
    # The table is still right, so the score is a share and not a boolean.
    assert concepts["status"] == contracts.MEASURED
    assert concepts["score"] == 0.5
    assert check_by_id(concepts, "structure.required_concept.chair")["observed"] == {
        "concept": "chair", "count": 3}
    assert check_by_id(
        concepts, "structure.required_concept.dining_table")["status"] == contracts.MEASURED


def test_an_addition_nothing_asked_for_is_unexpected(tmp_path):
    source, candidate = dining_room(extras=[actor("statue", asset_path="/Game/X")])
    report = structure_additions.verify(context(
        tmp_path, candidate=candidate, input_scene=source,
        case_spec=case([STRUCTURE])))
    assert report["status"] == contracts.MEASURED
    check = check_by_id(report, "structure.allowed_additions")
    assert check["observed"]["unexpected_added_actors"] == 1
    assert check["evidence"][0]["label"] == "statue"


def test_an_actor_matching_two_concepts_is_credited_to_neither(tmp_path):
    """Ambiguity is the case author's problem, not a coin flip."""
    ambiguous = {**STRUCTURE, "expected_raw_added_count": 1, "concepts": [
        {"concept": "seating", "count": 1, "allowed_asset_paths": [CHAIR]},
        {"concept": "chair", "count": 1, "allowed_asset_paths": [CHAIR]}]}
    source = scene([actor("floor")])
    candidate = scene([actor("floor"), actor("chair_1", asset_path=CHAIR)])
    report = structure_additions.verify(context(
        tmp_path, candidate=candidate, input_scene=source,
        case_spec=case([ambiguous])))
    assert check_by_id(
        report, "structure.allowed_additions")["observed"][
            "unexpected_added_actors"] == 1


def test_without_the_input_scene_the_verifier_refuses_rather_than_passes(tmp_path):
    """Additions cannot be known from the candidate alone."""
    _, candidate = dining_room()
    report = structure_count.verify(context(
        tmp_path, candidate=candidate, case_spec=case([STRUCTURE])))
    assert report["status"] == "not_evaluated"
    assert report["score"] is None
    assert "input_scene" in report["failure_reason"]


def test_a_case_declaring_no_structure_is_an_error_not_a_pass(tmp_path):
    source, candidate = dining_room()
    report = structure_concepts.verify(context(
        tmp_path, candidate=candidate, input_scene=source,
        case_spec=case([{"id": "p", "primitive": "no_overlap"}])))
    assert report["status"] == "not_applicable"
    assert "nothing to measure" in report["failure_reason"]


def test_compound_objects_need_exactly_the_declared_roles(tmp_path):
    compound = {
        "id": "compound", "primitive": "structure", "expected_raw_added_count": 2,
        "concepts": [],
        "compound_rules": [{"logical_object_id": "table_1",
                            "required_roles": ["top", "base"]}],
    }
    source = scene([actor("floor")])
    parts = [actor("top", logical_object_id="table_1", actor_role="top"),
             actor("base", logical_object_id="table_1", actor_role="base")]
    whole = structure_concepts.verify(context(
        tmp_path, candidate=scene([actor("floor"), *parts]), input_scene=source,
        case_spec=case([compound])))
    assert whole["status"] == contracts.MEASURED

    broken = structure_concepts.verify(context(
        tmp_path, candidate=scene([actor("floor"), parts[0]]), input_scene=source,
        case_spec=case([compound])))
    assert broken["status"] == contracts.MEASURED
    assert check_by_id(broken, "structure.compound_object.table_1")["observed"][
        "observed_roles"] == ["top"]
