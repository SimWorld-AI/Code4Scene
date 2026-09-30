"""Deterministic semantic v2 aggregation policy."""

from __future__ import annotations

import pytest

from code4scene.evaluation.semantic_scoring import (
    CASE_AGGREGATE_POLICY,
    PREDICATE_FAMILY,
    PREDICATE_WEIGHTS,
    SEMANTIC_FAMILY_ORDER,
    SEMANTIC_FAMILY_PREDICATE_TYPES,
    SEMANTIC_FAMILY_WEIGHTS,
    SEMANTIC_SCORE_POLICY,
    attach_semantic_families,
    macro_average_case_scores,
    score_semantic_case,
    semantic_exclusion_reasons,
)
from code4scene.evaluation.requirement_graph.contracts import PredicateType


def _requirement(
    node_id: str,
    text: str,
    span: tuple[int, int],
    predicate_type: str,
    status: str,
    *,
    score: float | None = None,
) -> dict:
    value = {
        "node_id": node_id,
        "requirement_id": node_id,
        "text": text,
        "source_span": list(span),
        "predicate_type": predicate_type,
        "evaluation_status": status,
    }
    if score is not None:
        value["score"] = score
    return value


def _span(prompt: str, text: str) -> tuple[int, int]:
    start = prompt.index(text)
    return (start, start + len(text))


def test_unknown_requirements_receive_zero_and_stay_in_denominator():
    prompt = "A foggy square with candles."
    result = score_semantic_case(
        prompt,
        [
            _requirement("fog", "foggy", _span(prompt, "foggy"), "environment", "MATCH"),
            _requirement(
                "square", "square", _span(prompt, "square"), "existence", "MATCH"
            ),
            _requirement(
                "candles",
                "candles",
                _span(prompt, "candles"),
                "existence",
                "NOT_EVALUATED",
            ),
        ],
    )

    assert result["score_policy"] == SEMANTIC_SCORE_POLICY
    assert result["score"] == 0.75
    assert result["score"] == 0.75
    assert "lower_bound" not in result and "upper_bound" not in result
    assert result["known_coverage"] == 0.75


def test_composite_duplicates_are_removed_and_parent_obeys_failed_child():
    prompt = "A foggy market with stalls and candles."
    set_text = "stalls and candles"
    relation_text = "market with stalls and candles"
    result = score_semantic_case(
        prompt,
        [
            _requirement(
                "identity", prompt, (0, len(prompt)), "scene_identity", "MATCH"
            ),
            _requirement("fog", "foggy", _span(prompt, "foggy"), "environment", "MATCH"),
            _requirement(
                "relation",
                relation_text,
                _span(prompt, relation_text),
                "spatial_relation",
                "MATCH",
            ),
            _requirement(
                "set", set_text, _span(prompt, set_text), "set", "MISMATCH"
            ),
            _requirement(
                "stalls", "stalls", _span(prompt, "stalls"), "existence", "MATCH"
            ),
            _requirement(
                "candles",
                "candles",
                _span(prompt, "candles"),
                "existence",
                "NOT_EVALUATED",
            ),
        ],
    )

    assert result["excluded_requirement_count"] == 2
    assert result["parent_adjustment_count"] == 1
    assert result["active_requirement_count"] == 4
    assert result["score"] == 0.5
    assert "lower_bound" not in result and "upper_bound" not in result
    by_id = {value["node_id"]: value for value in result["requirements"]}
    assert by_id["identity"]["exclusion_reason"] == "composite_replaced_by_children"
    assert by_id["set"]["exclusion_reason"] == "composite_replaced_by_children"
    assert by_id["relation"]["parent_constrained"] is True
    assert by_id["relation"]["effective_score"] == 0.0


def test_exclusion_plan_is_available_before_candidate_verdicts():
    prompt = "A foggy market with stalls and candles."
    set_text = "stalls and candles"
    rows = [
        _requirement(
            "identity",
            prompt,
            (0, len(prompt)),
            "scene_identity",
            "NOT_EVALUATED",
        ),
        _requirement(
            "fog",
            "foggy",
            _span(prompt, "foggy"),
            "environment",
            "NOT_EVALUATED",
        ),
        _requirement(
            "set",
            set_text,
            _span(prompt, set_text),
            "set",
            "NOT_EVALUATED",
        ),
        _requirement(
            "stalls",
            "stalls",
            _span(prompt, "stalls"),
            "existence",
            "NOT_EVALUATED",
        ),
        _requirement(
            "candles",
            "candles",
            _span(prompt, "candles"),
            "existence",
            "NOT_EVALUATED",
        ),
    ]
    rows[0]["semantic_replaced_by_children"] = True
    rows[2]["semantic_replaced_by_children"] = True

    assert semantic_exclusion_reasons(prompt, rows) == {
        "identity": "composite_replaced_by_children",
        "set": "composite_replaced_by_children",
    }


def test_extra_decomposition_inside_one_clause_does_not_add_case_weight():
    prompt = "A foggy square. Add barrels, crates, and carts."
    fog = _requirement(
        "fog", "A foggy square", _span(prompt, "A foggy square"), "environment", "MATCH"
    )
    one_detail = [_requirement("barrels", "barrels", _span(prompt, "barrels"), "existence", "MISMATCH")]
    three_details = [
        *one_detail,
        _requirement("crates", "crates", _span(prompt, "crates"), "existence", "MISMATCH"),
        _requirement("carts", "carts", _span(prompt, "carts"), "existence", "MISMATCH"),
    ]

    sparse = score_semantic_case(prompt, [fog, *one_detail])
    decomposed = score_semantic_case(prompt, [fog, *three_details])

    assert sparse["score"] == 0.5
    assert decomposed["score"] == 0.5
    assert len(sparse["clauses"]) == len(decomposed["clauses"]) == 2


def test_atomic_weights_are_equal_and_family_weights_own_importance():
    assert set(PREDICATE_WEIGHTS.values()) == {1.0}
    assert SEMANTIC_FAMILY_WEIGHTS == {
        family: 1.0 for family in SEMANTIC_FAMILY_ORDER
    }

    prompt = "Add a chair. Make it red."
    requirements = [
        _requirement(
            "chair", "Add a chair", _span(prompt, "Add a chair"), "existence", "MATCH"
        ),
        _requirement(
            "red", "Make it red", _span(prompt, "Make it red"), "attribute", "MISMATCH"
        ),
    ]
    equal = score_semantic_case(prompt, requirements)
    content_weighted = score_semantic_case(
        prompt,
        requirements,
        family_weights={"content_quantity": 3.0},
    )

    assert equal["score"] == 0.5
    assert equal["families"]["content_quantity"]["score"] == 1.0
    assert equal["families"]["attributes_materials"]["score"] == 0.0
    assert content_weighted["score"] == 0.75


def test_semantic_families_partition_every_non_logic_predicate_exactly_once():
    declared = [
        predicate_type
        for family in SEMANTIC_FAMILY_ORDER
        for predicate_type in SEMANTIC_FAMILY_PREDICATE_TYPES[family]
    ]
    expected = {
        predicate_type.value
        for predicate_type in PredicateType
        if predicate_type is not PredicateType.LOGIC
    }

    assert len(declared) == len(set(declared))
    assert set(declared) == expected
    assert set(PREDICATE_FAMILY) == expected


def test_logic_predicate_inherits_one_family_from_graph_scope_children():
    graph = {
        "nodes": [
            {"id": "choice", "predicate_type": "logic"},
            {"id": "red", "predicate_type": "attribute"},
            {"id": "blue", "predicate_type": "attribute"},
        ],
        "edges": [
            {"edge_type": "scope", "source_id": "choice", "target_id": "red"},
            {"edge_type": "scope", "source_id": "choice", "target_id": "blue"},
        ],
    }
    requirements = [
        _requirement("choice", "red or blue", (0, 11), "logic", "MATCH"),
        _requirement("red", "red or blue", (0, 11), "attribute", "MATCH"),
        _requirement("blue", "red or blue", (0, 11), "attribute", "MISMATCH"),
    ]
    requirements[1]["entity_scopes"] = {"red": "candidate_all"}
    requirements[2]["entity_scopes"] = {"blue": "candidate_all"}
    attached = attach_semantic_families(graph, requirements)

    assert attached[0]["semantic_family"] == "attributes_materials"
    result = score_semantic_case("red or blue", attached)
    by_id = {value["node_id"]: value for value in result["requirements"]}
    assert by_id["choice"]["exclusion_reason"] == "composite_replaced_by_children"
    assert result["families"]["attributes_materials"]["score"] == 0.5


def test_cross_family_logic_is_audited_but_only_children_are_scored():
    graph = {
        "nodes": [
            {"id": "choice", "predicate_type": "logic"},
            {"id": "chair", "predicate_type": "existence"},
            {"id": "red", "predicate_type": "attribute"},
        ],
        "edges": [
            {"edge_type": "scope", "source_id": "choice", "target_id": "chair"},
            {"edge_type": "scope", "source_id": "choice", "target_id": "red"},
        ],
    }
    requirements = [
        _requirement("choice", "chair or red", (0, 12), "logic", "MATCH"),
        _requirement("chair", "chair", (0, 5), "existence", "MATCH"),
        _requirement("red", "red", (9, 12), "attribute", "MISMATCH"),
    ]

    result = score_semantic_case(
        "chair or red",
        attach_semantic_families(graph, requirements),
    )
    by_id = {value["node_id"]: value for value in result["requirements"]}

    assert by_id["choice"]["semantic_family"] is None
    assert by_id["choice"]["exclusion_reason"] == "composite_replaced_by_children"
    assert result["families"]["content_quantity"]["score"] == 1.0
    assert result["families"]["attributes_materials"]["score"] == 0.0
    assert result["score"] == 0.5


def test_continuous_measured_scores_are_preserved():
    prompt = "A partially correct relation."
    result = score_semantic_case(
        prompt,
        [
            _requirement(
                "partial",
                "partially correct relation",
                _span(prompt, "partially correct relation"),
                "spatial_relation",
                "MISMATCH",
                score=0.25,
            )
        ],
    )

    assert result["score"] == 0.25
    assert result["score"] == 0.25


def test_rows_keep_the_unrounded_requirement_score():
    prompt = "A partially correct relation."
    result = score_semantic_case(
        prompt,
        [
            _requirement(
                "partial",
                "partially correct relation",
                _span(prompt, "partially correct relation"),
                "spatial_relation",
                "MISMATCH",
                score=1 / 3,
            )
        ],
    )

    (row,) = result["requirements"]
    assert row["effective_score"] == 0.3333
    assert row["unrounded_effective_score"] == 1 / 3


def test_same_source_sentence_with_different_predicates_is_not_a_duplicate():
    prompt = "Add three red chairs."
    requirements = [
        _requirement("count", prompt, (0, len(prompt)), "count", "MATCH"),
        _requirement("colour", prompt, (0, len(prompt)), "attribute", "MATCH"),
    ]

    result = score_semantic_case(prompt, requirements)

    assert result["active_requirement_count"] == 2
    assert result["excluded_requirement_count"] == 0


def test_same_source_and_predicate_with_different_targets_is_not_a_duplicate():
    prompt = "Add three chairs near one table."
    chair_count = _requirement(
        "chair_count", prompt, (0, len(prompt)), "count", "MATCH"
    )
    table_count = _requirement(
        "table_count", prompt, (0, len(prompt)), "count", "MATCH"
    )
    chair_count["entity_scopes"] = {"chairs": "additions"}
    table_count["entity_scopes"] = {"table": "candidate_all"}

    result = score_semantic_case(prompt, [chair_count, table_count])

    assert result["active_requirement_count"] == 2
    assert result["excluded_requirement_count"] == 0


def test_case_macro_average_never_uses_requirement_count_as_weight():
    cases = [
        {"case": "short", "score": 1.0, "source_requirement_count": 4},
        {"case": "long", "score": 0.0, "source_requirement_count": 22},
    ]

    equal = macro_average_case_scores(cases)
    weighted = macro_average_case_scores(
        cases,
        case_weights={"short": 1.0, "long": 3.0},
    )

    assert equal["score_policy"] == CASE_AGGREGATE_POLICY
    assert equal["score"] == 0.5
    assert equal["case_weight_sum"] == 2.0
    assert weighted["score"] == 0.25


def test_missing_case_is_not_silently_removed_from_macro_average():
    result = macro_average_case_scores(
        [
            {"case": "measured", "score": 1.0},
            {"case": "missing", "score": None},
        ]
    )

    assert result["score"] is None
    assert "lower_bound" not in result and "upper_bound" not in result
    assert result["complete_case_weight_coverage"] == 0.5


def test_case_macro_average_rejects_duplicate_case_ids():
    with pytest.raises(ValueError, match="duplicate case id"):
        macro_average_case_scores(
            [{"case": "same", "score": 1.0}, {"case": "same", "score": 0.0}]
        )


def _tower_rows(river_status: str) -> tuple[str, list[dict]]:
    prompt = "A tall stone tower beside a river."
    relation = "tall stone tower beside a river"
    rows = [
        _requirement("relation", relation, _span(prompt, relation), "spatial_relation", "MATCH"),
        _requirement("tall", "tall", _span(prompt, "tall"), "attribute", "MATCH"),
        _requirement("stone", "stone", _span(prompt, "stone"), "attribute", "MATCH"),
        # An exact duplicate is excluded before evidence and never judged.
        _requirement("stone_again", "stone", _span(prompt, "stone"), "attribute", "NOT_EVALUATED"),
        _requirement("tower", "tower", _span(prompt, "tower"), "existence", "MATCH"),
        _requirement("river", "river", _span(prompt, "river"), "existence", river_status),
    ]
    return prompt, rows


def test_a_requirement_that_is_never_scored_does_not_cap_its_parent():
    result = score_semantic_case(*_tower_rows("MATCH"))
    by_id = {value["node_id"]: value for value in result["requirements"]}
    assert by_id["stone_again"]["exclusion_reason"] == "exact_duplicate"
    assert by_id["relation"]["effective_score"] == 1.0
    assert result["score"] == 1.0


def test_a_scored_child_that_fails_still_caps_its_parent():
    result = score_semantic_case(*_tower_rows("MISMATCH"))
    by_id = {value["node_id"]: value for value in result["requirements"]}
    assert by_id["relation"]["effective_score"] == 0.0
    assert by_id["relation"]["parent_constrained"] is True
