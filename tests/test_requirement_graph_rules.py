"""Internal deterministic rules used only by ``semantic_requirements``."""

from __future__ import annotations

from declared_cases import actor, scene

from code4scene.evaluation import contracts
from code4scene.evaluation.requirement_graph.rules import (
    attribute,
    count,
    object_architecture,
    object_object,
)
from code4scene.evaluation.requirement_graph.rules.contracts import (
    RuleEvaluationContext,
)
from code4scene.evaluation.semantic_graph import build


def rule_context(candidate: dict) -> RuleEvaluationContext:
    return RuleEvaluationContext(
        graph=build(candidate["actors"]),
    )


def evaluate(module, candidate: dict, item: dict):
    return module.evaluate_item(rule_context(candidate), item)[0]


def test_count_uses_logical_objects_and_rejects_empty_bounds():
    parts = [
        actor(
            f"part_{index}",
            asset_category="table",
            logical_object_id="table_1",
            actor_role=f"part_{index}",
        )
        for index in range(5)
    ]
    item = {
        "id": "one-table",
        "type": "object_count",
        "subject": {"allowed_categories": ["table"]},
        "expected": {"exact_count": 1},
    }
    result = evaluate(count, scene(parts), item)
    assert result.status == contracts.PASS
    assert result.observed["count"] == 1

    invalid = {**item, "expected": {"count": 1}}
    result = evaluate(count, scene(parts), invalid)
    assert result.status == "not_evaluated"
    assert "asserts nothing" in result.failure_reason


def test_forbidden_object_and_incomplete_count_are_explicit():
    forbidden = {
        "id": "no-cars",
        "type": "forbidden_object",
        "subject": {"allowed_categories": ["car"]},
    }
    result = evaluate(
        count,
        scene([actor("car", asset_category="car")]),
        forbidden,
    )
    assert result.status == contracts.FAIL

    exact = {
        "id": "one-chair",
        "type": "object_count",
        "subject": {"allowed_categories": ["chair"]},
        "expected": {"exact_count": 1},
    }
    uncertain = scene(
        [actor("chair", asset_category="chair"), actor("uncategorised")]
    )
    assert evaluate(count, uncertain, exact).status == "not_evaluated"


def test_attributes_refuse_missing_or_ambiguous_evidence():
    item = {
        "id": "red-sofa",
        "type": "attribute_value",
        "subject": {"allowed_categories": ["sofa"]},
        "expected": {"field": "colour", "equals": "red"},
    }
    missing = scene([actor("sofa", asset_category="sofa")])
    assert evaluate(attribute, missing, item).status == "not_evaluated"

    compound = scene(
        [
            actor(
                "seat",
                asset_category="sofa",
                logical_object_id="sofa_1",
                colour="red",
            ),
            actor(
                "back",
                asset_category="sofa",
                logical_object_id="sofa_1",
                colour="blue",
            ),
        ]
    )
    assert evaluate(attribute, compound, item).status == "not_evaluated"


def test_attribute_quantification_refuses_an_incomplete_selection():
    """Uncategorised Actors never enter a category selection, so `all
    subjects` quantified over the survivors is vacuous — the count and
    relation families already refuse this evidence, and the attribute family
    used not to."""
    item = {
        "id": "all-sofas-red",
        "type": "attribute_value",
        "subject": {"allowed_categories": ["sofa"]},
        "expected": {"field": "colour", "equals": "red", "all_subjects": True},
    }
    incomplete = scene(
        [actor("sofa", asset_category="sofa", colour="red"),
         actor("uncategorised")]
    )
    result = evaluate(attribute, incomplete, item)
    assert result.status == "not_evaluated"
    assert "incomplete" in result.failure_reason


def test_attribute_lower_bound_already_met_survives_incompleteness():
    """A missing subject can only ADD subjects; it cannot break "at least one
    matches" once one does — the same carve-out the count family makes."""
    item = {
        "id": "a-red-sofa",
        "type": "attribute_value",
        "subject": {"allowed_categories": ["sofa"]},
        "expected": {"field": "colour", "equals": "red"},
    }
    incomplete = scene(
        [actor("sofa", asset_category="sofa", colour="red"),
         actor("uncategorised")]
    )
    result = evaluate(attribute, incomplete, item)
    assert result.status == contracts.PASS


def test_around_refuses_an_incomplete_selection_too():
    """The `around` branch used to return before the completeness check, so a
    ring judged over an incompletely categorised scene decided on subjects
    that were never all counted."""
    item = {
        "id": "chairs-around-table",
        "type": "around",
        "subject": {"allowed_categories": ["chair"]},
        "object": {"allowed_categories": ["table"]},
        "expected": {"minimum_subject_count": 1},
    }
    incomplete = scene(
        [actor("table", asset_category="table", extent=(80.0, 80.0, 40.0)),
         actor("chair", asset_category="chair", location=(150.0, 0.0, 40.0)),
         actor("uncategorised")]
    )
    result = evaluate(object_object, incomplete, item)
    assert result.status == "not_evaluated"
    assert "incomplete" in result.failure_reason


def test_around_requires_angular_coverage_not_only_distance():
    item = {
        "id": "chairs-around-table",
        "type": "around",
        "subject": {"allowed_categories": ["chair"]},
        "object": {"allowed_categories": ["table"]},
        "expected": {"minimum_subject_count": 4},
    }
    table = actor(
        "table",
        asset_category="table",
        location=(0.0, 0.0, 40.0),
        extent=(80.0, 80.0, 40.0),
    )
    chairs = [
        actor(
            f"chair_{index}",
            asset_category="chair",
            location=(150.0, 60.0 * index - 90.0, 40.0),
        )
        for index in range(4)
    ]
    result = evaluate(object_object, scene([table, *chairs]), item)
    assert result.status == contracts.FAIL
    assert result.observed["within_distance_count"] == 4
    assert result.observed["angular_coverage"] < 0.5


def test_architecture_rule_requires_an_explicit_architecture_selector():
    item = {
        "id": "shelf-on-wall",
        "type": "object_architecture_relation",
        "subject": {"allowed_categories": ["bookshelf"]},
        "expected": {"relation": "against"},
    }
    result = evaluate(
        object_architecture,
        scene([actor("shelf", asset_category="bookshelf")]),
        item,
    )
    assert result.status == "not_evaluated"
    assert "does not infer" in result.failure_reason
