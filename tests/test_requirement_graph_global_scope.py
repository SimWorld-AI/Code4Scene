"""Regression tests for entity-free global Stage0 v2 scopes."""

from code4scene.evaluation.requirement_graph.contracts import PredicateNode
from code4scene.evaluation.requirement_graph.stage0 import compile_semantic_draft
from code4scene.evaluation.requirement_graph.stage2_routing import (
    visual_claim_payload,
)
from code4scene.evaluation.requirement_graph.visual_claims import (
    sanitize_visual_claim_payload,
)


def test_global_scope_without_entity_is_valid_visual_claim_payload():
    prompt = "Light everything with a low, harsh sun."
    graph, _ = compile_semantic_draft(
        prompt,
        {
            "draft_version": "2.0",
            "entities": [],
            "requirements": [
                {
                    "key": "global_lighting",
                    "source_text": prompt,
                    "occurrence": 0,
                    "name": "global low harsh sun lighting",
                    "requirement_type": "environment",
                    "polarity": "affirmative",
                    "arguments": [],
                    "relation": None,
                    "quantity": None,
                    "scope": {
                        "quantifier": "global",
                        "entity_key": None,
                    },
                    "logic": None,
                    "qualifiers": [
                        {
                            "kind": "lighting",
                            "name": "low, harsh sun",
                            "source_text": "low, harsh sun",
                            "occurrence": 0,
                        }
                    ],
                }
            ],
        },
    )
    predicate = next(
        node for node in graph.nodes if isinstance(node, PredicateNode)
    )

    assert predicate.semantic_parameters["scope"]["entity_id"] is None
    payload = visual_claim_payload(graph, predicate.id)
    assert payload["semantic_dsl"]["scope"] == {
        "quantifier": "global",
        "entity_claim_text": None,
    }
    assert sanitize_visual_claim_payload(payload) == payload
