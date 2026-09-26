"""Deterministic Stage 0 rule compiler behavior."""

from __future__ import annotations

from code4scene.evaluation.requirement_graph import rule_compiler as compiler


def by_id(document, identifier):
    return next(item for item in document["items"] if item["id"] == identifier)


def test_english_counts_are_atomic_deterministic_and_traceable():
    prompt = "Add one table and four chairs. Use no more than three lamps."
    first = compiler.compile_prompt(prompt)
    second = compiler.compile_prompt(prompt)

    assert first == second
    assert by_id(first, "clause_1_table_count")["expected"] == {"exact_count": 1}
    assert by_id(first, "clause_1_chair_count")["expected"] == {"exact_count": 4}
    assert by_id(first, "clause_2_lamp_count")["expected"] == {"max_count": 3}
    assert compiler.validate_draft(first)["can_freeze"]
    for item in first["items"]:
        start, end = item["traceability"]["source_span"]
        assert prompt[start:end] == item["traceability"]["source_text"]


def test_a_count_binds_only_to_the_following_category():
    draft = compiler.compile_prompt("Add one table and chairs.")
    assert by_id(draft, "clause_1_table_count")["expected"] == {"exact_count": 1}
    assert by_id(draft, "clause_1_chair_presence")["expected"] == {"min_count": 1}


def test_vague_or_unknown_language_routes_to_semantic_fallback():
    vague = compiler.compile_prompt("Add several chairs.")
    assert not compiler.validate_draft(vague)["can_freeze"]
    assert vague["unresolved"][0]["kind"] == "vague_quantity"

    unknown = compiler.compile_prompt("Build a cozy Gothic hall.")
    assert not compiler.validate_draft(unknown)["can_freeze"]
    diagnostics = [*unknown["unresolved"], *unknown["unsupported"]]
    assert diagnostics[0]["kind"] in {"unsupported_clause", "unsupported_language"}


def test_scale_dependent_relation_requires_a_reviewed_policy():
    blocked = compiler.compile_prompt("Place one chair near one table.")
    assert blocked["unresolved"][0]["kind"] == "missing_relation_policy"

    ready = compiler.compile_prompt(
        "Place four chairs around one table.",
        relation_policies={"around:chair:table": {
            "minimum_subject_count": 4,
            "minimum_angular_coverage": 0.5,
            "distance_factor": 3.0,
        }})
    relation = by_id(ready, "clause_1_chair_around_table")
    assert relation["type"] == "around"
    assert compiler.validate_draft(ready)["can_freeze"]


def test_conflicts_are_detected_before_review():
    draft = compiler.compile_prompt("Add one table.")
    draft["items"].append({**draft["items"][0],
                           "id": "contradictory_table_count",
                           "expected": {"min_count": 2}})
    validation = compiler.validate_draft(draft)
    assert not validation["can_freeze"]
    assert validation["conflicts"][0]["kind"] == "incompatible_count_constraints"
