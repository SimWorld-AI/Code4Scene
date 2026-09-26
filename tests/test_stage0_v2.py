"""Stage0 v2 preserves prompt semantics before verifier capability planning."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from code4scene.evaluation.requirement_graph.capabilities import (
    GLOBAL_VLM_PLANNER_ROUTE,
    capability_for_node,
    capability_summary,
)
from code4scene.evaluation.requirement_graph.authoring import (
    _binding_for_node,
    _relocated_payload,
    _v2_grounding_diagnostics,
)
from code4scene.evaluation.requirement_graph.authoring import (
    compile_hybrid,
    freeze_compilation,
)
from code4scene.evaluation.requirement_graph.bundle import PopulationScope, TaskMode
from code4scene.evaluation.requirement_graph.contracts import (
    EntityGroundingMode,
    EntityNode,
    GraphValidationError,
    PredicateNode,
    PredicateType,
    RequirementGraph,
)
from code4scene.evaluation.requirement_graph.stage0 import (
    Stage0Compiler,
    compile_semantic_draft,
)
from code4scene.evaluation.requirement_graph.stage0_v2 import (
    SYSTEM_PROMPT,
    normalize_semantic_ir,
    tool_schema,
)
from code4scene.evaluation.requirement_graph.stage1 import Stage1Result
from code4scene.evaluation.requirement_graph.stage2_contracts import Stage2TaskKind
from code4scene.evaluation.requirement_graph.stage2_routing import visual_claim_payload
from code4scene.evaluation.requirement_graph.stage2_tasks import (
    build_stage2_queries,
    build_stage2_tasks,
)
from code4scene.evaluation.requirement_graph.visual_claims import (
    sanitize_visual_claim_payload,
)
from code4scene.evaluation.requirement_graph.existing_llm import LLMResponse, ToolCall
from code4scene.evaluation.verifiers import semantic_requirements


def _entity(
    key: str,
    source_text: str,
    name: str,
    *,
    entity_type: str = "object",
    referent_kind: str = "individual",
    grounding_mode: str | None = None,
    aliases=(),
    list_mode: str = "none",
    member_keys=(),
):
    if grounding_mode is None:
        if entity_type == "region":
            grounding_mode = "derived_region"
        elif entity_type == "surface" or referent_kind == "collection":
            grounding_mode = "actor_collection"
        else:
            grounding_mode = "actor"
    return {
        "key": key,
        "source_text": source_text,
        "occurrence": 0,
        "name": name,
        "aliases": list(aliases),
        "entity_type": entity_type,
        "referent_kind": referent_kind,
        "grounding_mode": grounding_mode,
        "list_mode": list_mode,
        "member_keys": list(member_keys),
    }


def _requirement(
    key: str,
    source_text: str,
    name: str,
    requirement_type: str,
    *,
    arguments=(),
    relation=None,
    quantity=None,
    scope=None,
    logic=None,
    qualifiers=(),
    polarity: str = "affirmative",
):
    return {
        "key": key,
        "source_text": source_text,
        "occurrence": 0,
        "name": name,
        "requirement_type": requirement_type,
        "polarity": polarity,
        "arguments": list(arguments),
        "relation": relation,
        "quantity": quantity,
        "scope": scope,
        "logic": logic,
        "qualifiers": list(qualifiers),
    }


def test_v2_represents_approximate_scoped_spatial_quantity():
    prompt = "Place approximately three benches around a fountain."
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity("fountain", "fountain", "fountain"),
            _entity(
                "benches",
                "benches",
                "bench",
                referent_kind="collection",
            ),
        ],
        "requirements": [
            _requirement(
                "rough_count",
                "approximately three benches",
                "approximately three benches",
                "quantity",
                arguments=[{"entity_key": "benches", "role": "thing"}],
                quantity={
                    "mode": "approximately",
                    "value": 3,
                    "lower": None,
                    "upper": None,
                    "qualitative": None,
                    "unit": "benches",
                },
            ),
            _requirement(
                "around",
                "benches around a fountain",
                "benches around fountain",
                "spatial_relation",
                arguments=[
                    {"entity_key": "benches", "role": "object"},
                    {"entity_key": "fountain", "role": "object"},
                ],
                relation="around",
            ),
        ],
    }

    graph, draft = compile_semantic_draft(prompt, value)

    assert graph.schema_version == "2.0"
    assert draft["coverage"]["coverage_ratio"] == 1.0
    quantity = next(
        node
        for node in graph.nodes
        if isinstance(node, PredicateNode)
        and node.predicate_type is PredicateType.QUANTITY
    )
    assert quantity.semantic_parameters["quantity"] == {
        "mode": "approximately",
        "value": 3,
        "lower": None,
        "upper": None,
        "qualitative": None,
        "unit": "benches",
    }
    spatial = next(
        node
        for node in graph.nodes
        if isinstance(node, PredicateNode)
        and node.predicate_type is PredicateType.SPATIAL_RELATION
    )
    assert [edge.role for edge in graph.arguments_for(spatial.id)] == [
        "subject",
        "reference",
    ]
    assert RequirementGraph.from_dict(graph.to_dict()) == graph
    assert (
        capability_for_node(graph, quantity).planner_route
        == GLOBAL_VLM_PLANNER_ROUTE
    )
    assert (
        capability_for_node(graph, spatial).planner_route
        == GLOBAL_VLM_PLANNER_ROUTE
    )
    visual_payload = sanitize_visual_claim_payload(
        visual_claim_payload(graph, quantity.id)
    )
    assert visual_payload["semantic_dsl"]["quantity"] == {
        "mode": "approximately",
        "value": 3,
        "lower": None,
        "upper": None,
        "qualitative": None,
        "unit": "benches",
    }
    assert "ir_key" not in visual_payload["semantic_dsl"]
    tasks = build_stage2_tasks(graph, Stage1Result((), "partial"))
    assert {task.kind for task in tasks} == {Stage2TaskKind.VISUAL_GLOBAL}
    assert {
        query.query_id: query.terms
        for query in build_stage2_queries(graph, tasks)
    } == {
        next(
            node.id
            for node in graph.nodes
            if isinstance(node, EntityNode) and node.name == "bench"
        ): ("bench",),
        next(
            node.id
            for node in graph.nodes
            if isinstance(node, EntityNode) and node.name == "fountain"
        ): ("fountain",),
    }
    assert capability_summary(graph) == {
        "matrix_version": "2.2.0",
        "requirement_count": 2,
        "supported_count": 2,
        "unplanned_count": 0,
        "by_semantic_type": {
            "quantity": {"supported": 1, "unplanned": 0},
            "spatial_relation": {"supported": 1, "unplanned": 0},
        },
    }


def test_v2_carries_style_environment_distribution_and_qualitative_quantity():
    prompt = (
        "Build dense blocks along narrow streets, with a Gothic style featuring "
        "stone façades and arched windows, under overcast light; almost no vehicles."
    )
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity("blocks", "blocks", "blocks", referent_kind="collection"),
            _entity("streets", "narrow streets", "narrow streets", referent_kind="collection"),
            _entity("vehicles", "vehicles", "vehicles", referent_kind="collection"),
        ],
        "requirements": [
            _requirement(
                "layout",
                "dense blocks along narrow streets",
                "dense blocks along streets",
                "distribution",
                arguments=[
                    {"entity_key": "blocks", "role": "subject"},
                    {"entity_key": "streets", "role": "path"},
                ],
                relation="dense_along",
                qualifiers=[
                    {
                        "source_text": "dense",
                        "occurrence": 0,
                        "kind": "density",
                        "name": "dense",
                    }
                ],
            ),
            _requirement(
                "style",
                "Gothic style featuring stone façades and arched windows",
                "Gothic architectural style",
                "style_bundle",
                arguments=[{"entity_key": "blocks", "role": "subject"}],
                qualifiers=[
                    {
                        "source_text": "Gothic style",
                        "occurrence": 0,
                        "kind": "style",
                        "name": "Gothic",
                    },
                    {
                        "source_text": "stone façades",
                        "occurrence": 0,
                        "kind": "material",
                        "name": "stone façades",
                    },
                    {
                        "source_text": "arched windows",
                        "occurrence": 0,
                        "kind": "shape",
                        "name": "arched windows",
                    },
                ],
            ),
            _requirement(
                "weather",
                "under overcast light",
                "overcast light",
                "environment",
                qualifiers=[
                    {
                        "source_text": "overcast light",
                        "occurrence": 0,
                        "kind": "lighting",
                        "name": "overcast",
                    }
                ],
            ),
            _requirement(
                "traffic",
                "almost no vehicles",
                "almost no vehicles",
                "quantity",
                arguments=[{"entity_key": "vehicles", "role": "collection"}],
                quantity={
                    "mode": "qualitative",
                    "value": None,
                    "lower": None,
                    "upper": None,
                    "qualitative": "almost_none",
                    "unit": "vehicles",
                },
            ),
        ],
    }

    graph, draft = compile_semantic_draft(prompt, value)
    types = {
        node.predicate_type
        for node in graph.nodes
        if isinstance(node, PredicateNode)
    }

    assert {
        PredicateType.DISTRIBUTION,
        PredicateType.STYLE_BUNDLE,
        PredicateType.ENVIRONMENT,
        PredicateType.QUANTITY,
    } <= types
    assert not any("unsupported" in key for key in draft)
    # The visible stone material remains a qualifier of the grouped style claim.
    assert len(graph.effective_weights()) == 4


def test_v2_normalizer_deduplicates_aliases_realigns_and_splits_tagged_lists():
    prompt = "Use Trapezoidal, triangular, and polygonal blocks."
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity(
                "bad id",
                "trapezoidal, triangular, and polygonal blocks",
                "block forms",
                referent_kind="collection",
                aliases=("Block forms", "blocks", "blocks"),
                list_mode="and",
            )
        ],
        "requirements": [
            _requirement(
                "list req",
                "Trapezoidal, triangular, and polygonal blocks",
                "three block forms",
                "set",
                arguments=[{"entity_key": "bad id", "role": "set"}],
            )
        ],
    }

    draft = normalize_semantic_ir(prompt, value)
    parent = next(value for value in draft["entities"] if value["name"] == "block forms")

    assert parent["source_text"].startswith("Trapezoidal")
    assert parent["aliases"] == ["blocks"]
    assert len(parent["member_keys"]) == 3
    assert len(draft["entities"]) == 4
    actions = {value["action"] for value in draft["normalization_actions"]}
    assert "source_span_realigned" in actions
    assert "duplicate_alias_removed" in actions
    assert "coordinated_list_split" in actions


def test_v2_logic_operands_are_dependencies_not_duplicate_scored_facets():
    prompt = "Use red or blue doors."
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity("red", "red", "red doors"),
            _entity("blue", "blue doors", "blue doors"),
        ],
        "requirements": [
            _requirement(
                "red_req",
                "red",
                "red doors",
                "presence",
                arguments=[{"entity_key": "red", "role": "subject"}],
            ),
            _requirement(
                "blue_req",
                "blue doors",
                "blue doors",
                "presence",
                arguments=[{"entity_key": "blue", "role": "subject"}],
            ),
            _requirement(
                "choice",
                "red or blue doors",
                "red or blue doors",
                "logic",
                logic={
                    "operator": "any_of",
                    "requirement_keys": ["red_req", "blue_req"],
                },
            ),
        ],
    }

    graph, draft = compile_semantic_draft(prompt, value)
    logic = next(
        node
        for node in graph.nodes
        if isinstance(node, PredicateNode)
        and node.predicate_type is PredicateType.LOGIC
    )

    assert set(graph.effective_weights()) == {logic.id}
    assert len(logic.semantic_parameters["logic"]["requirement_ids"]) == 2
    assert len(draft["requirements"]) == 3


def test_relocation_rewrites_v2_scope_logic_and_collection_references():
    scope_prompt = "Most structures are timber."
    scope_graph, _ = compile_semantic_draft(
        scope_prompt,
        {
            "draft_version": "2.0",
            "entities": [
                _entity(
                    "structures",
                    "structures",
                    "structures",
                    referent_kind="collection",
                )
            ],
            "requirements": [
                _requirement(
                    "timber",
                    scope_prompt,
                    "most structures are timber",
                    "attribute",
                    arguments=[
                        {"entity_key": "structures", "role": "subject"}
                    ],
                    scope={
                        "quantifier": "most",
                        "entity_key": "structures",
                    },
                )
            ],
        },
    )
    scope_nodes, _, _, scope_mapping = _relocated_payload(
        scope_graph,
        prefix="c04",
        offset=10,
    )
    relocated_scope = next(
        node["semantic_parameters"]["scope"]
        for node in scope_nodes
        if node.get("semantic_parameters", {}).get("scope") is not None
    )
    original_scope = next(
        node.semantic_parameters["scope"]
        for node in scope_graph.nodes
        if isinstance(node, PredicateNode)
    )
    assert relocated_scope["entity_id"] == scope_mapping[
        original_scope["entity_id"]
    ]

    logic_prompt = "Use red or blue doors."
    logic_graph, _ = compile_semantic_draft(
        logic_prompt,
        {
            "draft_version": "2.0",
            "entities": [
                _entity("red", "red", "red doors"),
                _entity("blue", "blue doors", "blue doors"),
            ],
            "requirements": [
                _requirement(
                    "red_req",
                    "red",
                    "red doors",
                    "presence",
                    arguments=[{"entity_key": "red", "role": "subject"}],
                ),
                _requirement(
                    "blue_req",
                    "blue doors",
                    "blue doors",
                    "presence",
                    arguments=[{"entity_key": "blue", "role": "subject"}],
                ),
                _requirement(
                    "choice",
                    "red or blue doors",
                    "red or blue doors",
                    "logic",
                    logic={
                        "operator": "any_of",
                        "requirement_keys": ["red_req", "blue_req"],
                    },
                ),
            ],
        },
    )
    logic_nodes, _, _, logic_mapping = _relocated_payload(
        logic_graph,
        prefix="c05",
        offset=20,
    )
    relocated_logic = next(
        node["semantic_parameters"]["logic"]
        for node in logic_nodes
        if node.get("semantic_parameters", {}).get("logic") is not None
    )
    original_logic = next(
        node.semantic_parameters["logic"]
        for node in logic_graph.nodes
        if isinstance(node, PredicateNode)
        and node.semantic_parameters.get("logic") is not None
    )
    assert relocated_logic["requirement_ids"] == [
        logic_mapping[value] for value in original_logic["requirement_ids"]
    ]

    collection_prompt = "Use trapezoidal, triangular, and polygonal blocks."
    collection_graph, _ = compile_semantic_draft(
        collection_prompt,
        {
            "draft_version": "2.0",
            "entities": [
                _entity(
                    "blocks",
                    "trapezoidal, triangular, and polygonal blocks",
                    "block forms",
                    referent_kind="collection",
                    list_mode="and",
                )
            ],
            "requirements": [
                _requirement(
                    "forms",
                    "trapezoidal, triangular, and polygonal blocks",
                    "three block forms",
                    "set",
                    arguments=[{"entity_key": "blocks", "role": "set"}],
                )
            ],
        },
    )
    collection_nodes, _, _, collection_mapping = _relocated_payload(
        collection_graph,
        prefix="c06",
        offset=30,
    )
    original_collection = next(
        node
        for node in collection_graph.nodes
        if isinstance(node, EntityNode) and node.member_ids
    )
    relocated_collection = next(
        node for node in collection_nodes if node.get("member_ids")
    )
    assert relocated_collection["member_ids"] == [
        collection_mapping[value] for value in original_collection.member_ids
    ]


def test_v2_semantic_references_are_validated_before_runtime_routing():
    prompt = "Most structures are timber."
    graph, _ = compile_semantic_draft(
        prompt,
        {
            "draft_version": "2.0",
            "entities": [
                _entity(
                    "structures",
                    "structures",
                    "structures",
                    referent_kind="collection",
                )
            ],
            "requirements": [
                _requirement(
                    "timber",
                    prompt,
                    "most structures are timber",
                    "attribute",
                    arguments=[
                        {"entity_key": "structures", "role": "subject"}
                    ],
                    scope={
                        "quantifier": "most",
                        "entity_key": "structures",
                    },
                )
            ],
        },
    )
    payload = graph.to_dict()
    predicate = next(
        node for node in payload["nodes"] if node["node_type"] == "predicate"
    )
    predicate["semantic_parameters"]["scope"]["entity_id"] = "missing_entity"

    with pytest.raises(GraphValidationError, match="dangling_semantic_reference"):
        RequirementGraph.from_dict(payload)


def test_v2_tool_schema_has_no_unsupported_or_scored_facet_escape_hatches():
    properties = tool_schema()["parameters"]["properties"]

    assert set(properties) == {"draft_version", "entities", "requirements"}
    assert "unsupported_semantics" not in properties
    assert "scored_facets" not in properties

    entity_schema = properties["entities"]["items"]
    assert "grounding_mode" in entity_schema["required"]
    assert set(entity_schema["properties"]["grounding_mode"]["enum"]) == {
        "actor",
        "actor_collection",
        "derived_region",
        "scene_global",
    }


def test_v2_grounding_contract_drives_identity_queries_and_global_routing():
    prompt = (
        "A marker and a tiled zone are present. "
        "The activity area is irregular."
    )
    graph, _ = compile_semantic_draft(
        prompt,
        {
            "draft_version": "2.0",
            "entities": [
                _entity(
                    "marker",
                    "marker",
                    "marker",
                    aliases=("indicator",),
                ),
                _entity(
                    "tiled_zone",
                    "tiled zone",
                    "tile",
                    entity_type="region",
                    referent_kind="collection",
                    grounding_mode="actor_collection",
                    aliases=("floor tile",),
                ),
                _entity(
                    "area",
                    "activity area",
                    "activity area",
                    entity_type="region",
                ),
            ],
            "requirements": [
                _requirement(
                    "marker_presence",
                    "A marker and a tiled zone are present.",
                    "marker presence",
                    "presence",
                    arguments=[{"entity_key": "marker", "role": "subject"}],
                ),
                _requirement(
                    "tiled_zone_presence",
                    "A marker and a tiled zone are present.",
                    "tiled zone presence",
                    "presence",
                    arguments=[
                        {"entity_key": "tiled_zone", "role": "subject"}
                    ],
                ),
                _requirement(
                    "area_shape",
                    "The activity area is irregular.",
                    "irregular activity area",
                    "attribute",
                    arguments=[{"entity_key": "area", "role": "subject"}],
                    qualifiers=[
                        {
                            "source_text": "irregular",
                            "occurrence": 0,
                            "kind": "shape",
                            "name": "irregular",
                        }
                    ],
                ),
            ],
        },
    )

    entities = {
        node.name: node
        for node in graph.nodes
        if isinstance(node, EntityNode)
    }
    assert entities["marker"].grounding_mode is EntityGroundingMode.ACTOR
    assert (
        entities["tile"].grounding_mode
        is EntityGroundingMode.ACTOR_COLLECTION
    )
    assert (
        entities["activity area"].grounding_mode
        is EntityGroundingMode.DERIVED_REGION
    )
    assert RequirementGraph.from_dict(graph.to_dict()) == graph

    tasks = build_stage2_tasks(graph, Stage1Result((), "partial"))
    queries = build_stage2_queries(graph, tasks)
    queries_by_id = {query.query_id: query.terms for query in queries}
    assert queries_by_id == {
        entities["marker"].id: ("marker", "indicator"),
        entities["tile"].id: ("tile", "floor tile"),
    }

    area_requirement = next(
        node
        for node in graph.nodes
        if isinstance(node, PredicateNode) and node.name == "irregular activity area"
    )
    assert (
        capability_for_node(graph, area_requirement).planner_route
        == GLOBAL_VLM_PLANNER_ROUTE
    )
    assert _v2_grounding_diagnostics(graph) == ()

    legacy_payload = graph.to_dict()
    marker_payload = next(
        node
        for node in legacy_payload["nodes"]
        if node.get("node_type") == "entity" and node.get("name") == "marker"
    )
    marker_payload.pop("grounding_mode")
    diagnostics = _v2_grounding_diagnostics(
        RequirementGraph.from_dict(legacy_payload)
    )
    assert [value["kind"] for value in diagnostics] == [
        "missing_entity_grounding_mode"
    ]


def test_v2_coordinated_actor_members_remain_one_grouped_claim():
    prompt = "Include lamps, stools, and planters."
    graph, draft = compile_semantic_draft(
        prompt,
        {
            "draft_version": "2.0",
            "entities": [
                _entity(
                    "items",
                    "lamps, stools, and planters",
                    "display items",
                    referent_kind="collection",
                    list_mode="and",
                    member_keys=("lamps", "stools", "planters"),
                ),
                _entity(
                    "lamps",
                    "lamps",
                    "lamp",
                    referent_kind="collection",
                ),
                _entity(
                    "stools",
                    "stools",
                    "stool",
                    referent_kind="collection",
                ),
                _entity(
                    "planters",
                    "planters",
                    "planter",
                    referent_kind="collection",
                ),
            ],
            "requirements": [
                _requirement(
                    "items_set",
                    "Include lamps, stools, and planters.",
                    "coordinated display items",
                    "set",
                    arguments=[{"entity_key": "items", "role": "collection"}],
                )
            ],
        },
    )

    generated = [
        value
        for value in draft["normalization_actions"]
        if value["action"] == "coordinated_atomic_presence_added"
    ]
    assert generated == []
    assert len(draft["requirements"]) == 1
    assert len(graph.effective_weights()) == 1
    collection = next(
        node
        for node in graph.nodes
        if isinstance(node, EntityNode) and node.name == "display items"
    )
    assert len(collection.member_ids) == 3
    tasks = build_stage2_tasks(graph, Stage1Result((), "partial"))
    queries = build_stage2_queries(graph, tasks)
    assert len(queries) == 1
    assert {"display item", "lamp", "stool", "planter"} <= set(
        queries[0].terms
    )
    assert _v2_grounding_diagnostics(graph) == ()


def test_v2_disjunction_operands_must_normalize_to_distinct_claims():
    prompt = "The outline is cross or Y shaped."
    _, draft = compile_semantic_draft(
        prompt,
        {
            "draft_version": "2.0",
            "entities": [
                _entity(
                    "outline",
                    "outline",
                    "outline",
                    entity_type="region",
                )
            ],
            "requirements": [
                _requirement(
                    "shape",
                    "outline is cross or Y shaped",
                    "cross or Y outline",
                    "attribute",
                    arguments=[{"entity_key": "outline", "role": "subject"}],
                    qualifiers=[
                        {
                            "source_text": "cross or Y shaped",
                            "occurrence": 0,
                            "kind": "shape",
                            "name": "cross or Y shaped",
                        }
                    ],
                )
            ],
        },
    )
    operands = [
        value
        for value in draft["requirements"]
        if value["requirement_type"] == "attribute"
    ]
    assert {
        tuple(qualifier["name"] for qualifier in value["qualifiers"])
        for value in operands
    } == {("cross",), ("Y",)}

    with pytest.raises(ValueError, match="duplicate_logic_operand_semantics"):
        compile_semantic_draft(
            "Use a bench or a chair.",
            {
                "draft_version": "2.0",
                "entities": [
                    _entity("bench", "bench", "bench"),
                    _entity("chair", "chair", "chair"),
                ],
                "requirements": [
                    _requirement(
                        "bench_presence",
                        "bench",
                        "bench presence",
                        "presence",
                        arguments=[
                            {"entity_key": "bench", "role": "subject"}
                        ],
                    ),
                    _requirement(
                        "choice",
                        "bench or a chair",
                        "bench or chair",
                        "logic",
                        logic={
                            "operator": "any_of",
                            "requirement_keys": [
                                "bench_presence",
                                "bench_presence",
                            ],
                        },
                    ),
                ],
            },
        )


def test_v2_coverage_allows_repeated_mentions_but_not_unique_missing_modifiers():
    repeated_prompt = "Place buildings near roads and include buildings."
    repeated = {
        "draft_version": "2.0",
        "entities": [
            _entity("buildings", "buildings", "buildings", referent_kind="collection"),
            _entity("roads", "roads", "roads", referent_kind="collection"),
        ],
        "requirements": [
            _requirement(
                "near",
                "buildings near roads",
                "buildings near roads",
                "spatial_relation",
                arguments=[
                    {"entity_key": "buildings", "role": "subject"},
                    {"entity_key": "roads", "role": "reference"},
                ],
                relation="near",
            ),
        ],
    }
    _, draft = compile_semantic_draft(repeated_prompt, repeated)
    assert draft["coverage"]["repeated_mention_count"] >= 1

    incomplete_prompt = "Place red doors beside blue walls."
    incomplete = {
        "draft_version": "2.0",
        "entities": [_entity("doors", "red doors", "red doors")],
        "requirements": [
            _requirement(
                "doors",
                "red doors",
                "red doors",
                "presence",
                arguments=[{"entity_key": "doors", "role": "subject"}],
            )
        ],
    }
    with pytest.raises(ValueError, match="blue.*walls"):
        compile_semantic_draft(incomplete_prompt, incomplete)


def test_v2_coverage_ignores_relative_clause_function_words():
    prompt = "Place a tower that is red."
    value = {
        "draft_version": "2.0",
        "entities": [_entity("tower", "tower", "tower")],
        "requirements": [
            _requirement(
                "tower",
                "tower",
                "tower presence",
                "presence",
                arguments=[{"entity_key": "tower", "role": "subject"}],
            ),
            _requirement(
                "red",
                "red",
                "red tower",
                "attribute",
                arguments=[{"entity_key": "tower", "role": "subject"}],
                qualifiers=[
                    {
                        "source_text": "red",
                        "occurrence": 0,
                        "kind": "color",
                        "name": "red",
                    }
                ],
            ),
        ],
    }

    _, draft = compile_semantic_draft(prompt, value)

    assert draft["coverage"]["coverage_ratio"] == 1.0


def test_v2_coverage_ignores_scene_form_instruction_but_not_relation():
    instruction_prompt = "Form the scene as an irregular island."
    instruction = {
        "draft_version": "2.0",
        "entities": [
            _entity(
                "island",
                "irregular island",
                "irregular island",
                entity_type="region",
            )
        ],
        "requirements": [
            _requirement(
                "island",
                "irregular island",
                "irregular island presence",
                "presence",
                arguments=[{"entity_key": "island", "role": "subject"}],
            )
        ],
    }

    _, draft = compile_semantic_draft(instruction_prompt, instruction)

    assert draft["coverage"]["coverage_ratio"] == 1.0

    relation_prompt = "Make the buildings form an L."
    relation = {
        "draft_version": "2.0",
        "entities": [
            _entity(
                "buildings",
                "buildings",
                "buildings",
                referent_kind="collection",
            )
        ],
        "requirements": [
            _requirement(
                "buildings",
                "buildings",
                "building presence",
                "presence",
                arguments=[{"entity_key": "buildings", "role": "subject"}],
            )
        ],
    }
    with pytest.raises(ValueError, match="form"):
        compile_semantic_draft(relation_prompt, relation)


def test_v2_coverage_includes_referenced_entity_spans_and_ignores_carriers():
    prompt = "Make a pale landscape, sunlit yet abandoned in mood."
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity(
                "landscape",
                "landscape",
                "landscape",
                entity_type="surface",
                referent_kind="mass",
            )
        ],
        "requirements": [
            _requirement(
                "pale",
                "pale",
                "pale landscape",
                "attribute",
                arguments=[{"entity_key": "landscape", "role": "subject"}],
                qualifiers=[
                    {"source_text": "pale", "occurrence": 0, "kind": "color", "name": "pale"}
                ],
            ),
            _requirement(
                "mood",
                "sunlit yet abandoned",
                "sunlit yet abandoned mood",
                "environment",
                arguments=[{"entity_key": "landscape", "role": "subject"}],
                qualifiers=[
                    {
                        "source_text": "sunlit yet abandoned",
                        "occurrence": 0,
                        "kind": "atmosphere",
                        "name": "sunlit yet abandoned",
                    }
                ],
            ),
        ],
    }

    _, draft = compile_semantic_draft(prompt, value)

    assert draft["coverage"]["coverage_ratio"] == 1.0


def test_v2_coverage_includes_prompt_mentions_of_referenced_entity_aliases():
    prompt = "Build a cemetery. Arrange the burial ground."
    value = {
        "draft_version": "2.0",
        "entities": [
            {
                **_entity("cemetery", "cemetery", "cemetery"),
                "aliases": ["burial ground"],
            }
        ],
        "requirements": [
            _requirement(
                "cemetery",
                "cemetery",
                "cemetery presence",
                "presence",
                arguments=[{"entity_key": "cemetery", "role": "subject"}],
            )
        ],
    }

    _, draft = compile_semantic_draft(prompt, value)

    assert draft["coverage"]["coverage_ratio"] == 1.0


def test_v2_normalizer_keeps_grouped_color_and_material_requirements():
    prompt = (
        "Place yellow-painted crosswalks beside weathered wood stalls with "
        "shutters in blue, red and purple."
    )
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity(
                "crosswalks",
                "crosswalks",
                "crosswalks",
                referent_kind="collection",
            ),
            _entity("stalls", "stalls", "stalls", referent_kind="collection"),
            _entity(
                "shutters",
                "shutters",
                "shutters",
                referent_kind="collection",
            ),
        ],
        "requirements": [
            _requirement(
                "crosswalk_relation",
                "yellow-painted crosswalks beside weathered wood stalls",
                "crosswalks beside stalls",
                "spatial_relation",
                arguments=[
                    {"entity_key": "crosswalks", "role": "subject"},
                    {"entity_key": "stalls", "role": "reference"},
                ],
                relation="beside",
                qualifiers=[
                    {
                        "source_text": "yellow-painted",
                        "occurrence": 0,
                        "kind": "color",
                        "name": "yellow-painted",
                    }
                ],
            ),
            _requirement(
                "stall_material",
                "weathered wood stalls",
                "weathered wood stalls",
                "material",
                arguments=[{"entity_key": "stalls", "role": "subject"}],
                qualifiers=[
                    {
                        "source_text": "weathered wood",
                        "occurrence": 0,
                        "kind": "material",
                        "name": "weathered wood",
                    }
                ],
            ),
            _requirement(
                "shutter_colors",
                "shutters in blue, red and purple",
                "shutter colors",
                "attribute",
                arguments=[{"entity_key": "shutters", "role": "subject"}],
                qualifiers=[
                    {
                        "source_text": "blue, red and purple",
                        "occurrence": 0,
                        "kind": "color",
                        "name": "blue, red and purple",
                    }
                ],
            ),
        ],
    }

    normalized = normalize_semantic_ir(prompt, value)
    assert len(normalized["requirements"]) == 3
    assert not any(
        value["action"] == "atomic_visible_qualifier_added"
        for value in normalized["normalization_actions"]
    )
    shutters = next(
        value
        for value in normalized["requirements"]
        if value["name"] == "shutter colors"
    )
    assert [value["name"] for value in shutters["qualifiers"]] == [
        "blue, red and purple"
    ]
    crosswalks = next(
        value
        for value in normalized["requirements"]
        if value["name"] == "crosswalks beside stalls"
    )
    assert [value["name"] for value in crosswalks["qualifiers"]] == [
        "yellow-painted"
    ]


def test_v2_normalizer_realigns_repeated_color_words_to_argument_entities():
    prompt = "blue shutters, red signs, red paper lanterns, blue sky"
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity("shutters", "blue shutters", "shutters"),
            _entity("signs", "red signs", "signs"),
            _entity("lanterns", "red paper lanterns", "lanterns"),
            _entity(
                "sky",
                "blue sky",
                "sky",
                entity_type="surface",
                grounding_mode="scene_global",
            ),
        ],
        "requirements": [
            _requirement(
                "shutters_present",
                "blue shutters",
                "shutters present",
                "presence",
                arguments=[{"entity_key": "shutters", "role": "subject"}],
            ),
            _requirement(
                "signs_present",
                "red signs",
                "signs present",
                "presence",
                arguments=[{"entity_key": "signs", "role": "subject"}],
            ),
            _requirement(
                "lanterns_present",
                "red paper lanterns",
                "lanterns present",
                "presence",
                arguments=[{"entity_key": "lanterns", "role": "subject"}],
            ),
            _requirement(
                "sky_present",
                "blue sky",
                "sky present",
                "presence",
                arguments=[{"entity_key": "sky", "role": "subject"}],
            ),
            _requirement(
                "lantern_red",
                "red",
                "lantern red",
                "attribute",
                arguments=[{"entity_key": "lanterns", "role": "subject"}],
                qualifiers=[
                    {
                        "source_text": "red",
                        "occurrence": 0,
                        "kind": "color",
                        "name": "red",
                    }
                ],
            ),
            _requirement(
                "sky_blue",
                "blue",
                "sky blue",
                "attribute",
                arguments=[{"entity_key": "sky", "role": "subject"}],
                qualifiers=[
                    {
                        "source_text": "blue",
                        "occurrence": 0,
                        "kind": "color",
                        "name": "blue",
                    }
                ],
            ),
        ],
    }

    normalized = normalize_semantic_ir(prompt, value)
    by_name = {item["name"]: item for item in normalized["requirements"]}

    assert by_name["lantern red"]["occurrence"] == 1
    assert by_name["lantern red"]["source_span"] == {"start": 26, "end": 29}
    assert by_name["sky blue"]["occurrence"] == 1
    assert by_name["sky blue"]["source_span"] == {"start": 46, "end": 50}


def test_v2_grouped_material_realigns_to_its_local_requirement():
    prompt = "A small paved plot. Scatter the paved street with traffic cones."
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity("plot", "paved plot", "plot", entity_type="surface"),
            _entity(
                "street",
                "paved street",
                "street",
                entity_type="surface",
                grounding_mode="derived_region",
            ),
            _entity(
                "cones",
                "traffic cones",
                "traffic cones",
                referent_kind="collection",
            ),
        ],
        "requirements": [
            _requirement(
                "plot_present",
                "A small paved plot",
                "plot present",
                "presence",
                arguments=[{"entity_key": "plot", "role": "subject"}],
            ),
            _requirement(
                "scatter",
                "Scatter the paved street with traffic cones",
                "scatter cones on street",
                "distribution",
                arguments=[
                    {"entity_key": "street", "role": "region"},
                    {"entity_key": "cones", "role": "subject"},
                ],
                qualifiers=[
                    {
                        "source_text": "paved",
                        "occurrence": 0,
                        "kind": "material",
                        "name": "paved",
                    }
                ],
            ),
        ],
    }

    normalized = normalize_semantic_ir(prompt, value)
    street_key = next(
        item["key"] for item in normalized["entities"] if item["name"] == "street"
    )
    grouped = next(
        item
        for item in normalized["requirements"]
        if item["name"] == "scatter cones on street"
    )
    qualifier = grouped["qualifiers"][0]

    assert qualifier["source_span"] == {"start": 32, "end": 37}
    assert qualifier["occurrence"] == 1
    assert any(
        argument["entity_key"] == street_key and argument["role"] == "region"
        for argument in grouped["arguments"]
    )


def test_v2_normalizer_expands_tagged_or_collection_into_any_of_logic():
    prompt = "Add a few benches or café chairs."
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity(
                "seating",
                "benches or café chairs",
                "seating choices",
                referent_kind="collection",
                list_mode="or",
            )
        ],
        "requirements": [
            _requirement(
                "few_seats",
                "a few benches or café chairs",
                "a few seating choices",
                "quantity",
                arguments=[{"entity_key": "seating", "role": "collection"}],
                quantity={
                    "mode": "qualitative",
                    "value": None,
                    "lower": None,
                    "upper": None,
                    "qualitative": "few",
                    "unit": "seats",
                },
            )
        ],
    }

    graph, draft = compile_semantic_draft(prompt, value)
    logic = [
        value for value in draft["requirements"] if value["requirement_type"] == "logic"
    ]
    quantities = [
        value
        for value in draft["requirements"]
        if value["requirement_type"] == "quantity"
    ]

    assert len(logic) == 1
    assert logic[0]["logic"]["operator"] == "any_of"
    assert len(logic[0]["logic"]["requirement_keys"]) == 2
    assert len(quantities) == 2
    assert len(graph.effective_weights()) == 1
    assert any(
        value["action"] == "disjunction_expanded"
        for value in draft["normalization_actions"]
    )


def test_v2_normalizer_accepts_declared_logic_with_shared_operand_span():
    prompt = "irregular L- or cross-shaped urban island"
    shared_qualifier = {
        "source_text": "L- or cross-shaped",
        "occurrence": 0,
        "kind": "shape",
    }
    graph, draft = compile_semantic_draft(
        prompt,
        {
            "draft_version": "2.0",
            "entities": [
                _entity(
                    "island",
                    "urban island",
                    "urban island",
                    entity_type="region",
                )
            ],
            "requirements": [
                _requirement(
                    "island_shape",
                    prompt,
                    "urban island shape",
                    "composition",
                    arguments=[{"entity_key": "island", "role": "subject"}],
                    logic={
                        "operator": "any_of",
                        "requirement_keys": ["island_shape_l", "island_shape_cross"],
                    },
                    qualifiers=[
                        {
                            "source_text": "irregular",
                            "occurrence": 0,
                            "kind": "shape",
                            "name": "irregular",
                        }
                    ],
                ),
                _requirement(
                    "island_shape_l",
                    "L- or cross-shaped",
                    "L-shaped island operand",
                    "attribute",
                    arguments=[{"entity_key": "island", "role": "subject"}],
                    qualifiers=[{**shared_qualifier, "name": "L-shaped"}],
                ),
                _requirement(
                    "island_shape_cross",
                    "L- or cross-shaped",
                    "cross-shaped island operand",
                    "attribute",
                    arguments=[{"entity_key": "island", "role": "subject"}],
                    qualifiers=[{**shared_qualifier, "name": "cross-shaped"}],
                ),
            ],
        },
    )

    assert len(draft["requirements"]) == 3
    logic = next(
        value
        for value in draft["requirements"]
        if value["requirement_type"] == "logic"
    )
    by_key = {value["key"]: value for value in draft["requirements"]}
    operands = [by_key[key] for key in logic["logic"]["requirement_keys"]]
    assert all(
        "irregular" in {qualifier["name"] for qualifier in operand["qualifiers"]}
        for operand in operands
    )
    assert len(graph.effective_weights()) == 1
    actions = {value["action"] for value in draft["normalization_actions"]}
    assert "declared_logic_type_normalized" in actions
    assert "shared_disjunction_span_operands_preserved" in actions


def test_v2_rejects_generic_whole_prompt_collapse_and_missing_scope():
    collapsed_prompt = "Build a village. Add a river. Place a bridge."
    collapsed = {
        "draft_version": "2.0",
        "entities": [],
        "requirements": [
            _requirement(
                "generic",
                collapsed_prompt,
                "generic scene",
                "scene_identity",
            )
        ],
    }
    with pytest.raises(ValueError, match="non_atomic_requirement"):
        compile_semantic_draft(collapsed_prompt, collapsed)

    scoped_prompt = "Place one lamp in every courtyard."
    scoped = {
        "draft_version": "2.0",
        "entities": [
            _entity("lamp", "lamp", "lamp"),
            _entity("courtyard", "courtyard", "courtyard"),
        ],
        "requirements": [
            _requirement(
                "placement",
                "one lamp in every courtyard",
                "lamp per courtyard",
                "spatial_relation",
                arguments=[
                    {"entity_key": "lamp", "role": "subject"},
                    {"entity_key": "courtyard", "role": "region"},
                ],
                relation="inside",
            )
        ],
    }
    with pytest.raises(ValueError, match="missing_scope"):
        compile_semantic_draft(scoped_prompt, scoped)


def test_v2_normalizer_expands_direct_arguments_and_alternative_qualifiers():
    prompt = "Add occasional arched blue or red doors and a few benches or café chairs."
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity("doors", "doors", "doors", referent_kind="collection"),
            _entity("benches", "benches", "benches", referent_kind="collection"),
            _entity("chairs", "café chairs", "café chairs", referent_kind="collection"),
        ],
        "requirements": [
            _requirement(
                "doors_req",
                "occasional arched blue or red doors",
                "arched colored doors",
                "presence",
                arguments=[{"entity_key": "doors", "role": "subject"}],
                qualifiers=[
                    {"source_text": "arched", "occurrence": 0, "kind": "shape", "name": "arched"},
                    {"source_text": "blue", "occurrence": 0, "kind": "color", "name": "blue"},
                    {"source_text": "red", "occurrence": 0, "kind": "color", "name": "red"},
                ],
            ),
            _requirement(
                "seat_req",
                "a few benches or café chairs",
                "a few seats",
                "quantity",
                arguments=[
                    {"entity_key": "benches", "role": "collection"},
                    {"entity_key": "chairs", "role": "collection"},
                ],
                quantity={
                    "mode": "qualitative",
                    "value": None,
                    "lower": None,
                    "upper": None,
                    "qualitative": "few",
                    "unit": "seats",
                },
            ),
        ],
    }

    graph, draft = compile_semantic_draft(prompt, value)
    logic = [
        value for value in draft["requirements"] if value["requirement_type"] == "logic"
    ]
    operands = [
        value for value in draft["requirements"] if value["requirement_type"] != "logic"
    ]

    assert len(logic) == 2
    assert len(operands) == 4
    assert len(graph.effective_weights()) == 2
    door_operands = [value for value in operands if value["requirement_type"] == "presence"]
    colors = {
        qualifier["name"]
        for value in door_operands
        for qualifier in value["qualifiers"]
        if qualifier["kind"] == "color"
    }
    assert colors == {"blue", "red"}


def test_v2_spatial_disjunction_preserves_common_subjects():
    prompt = "Industrial halls and sheds face yards or streets."
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity("halls", "Industrial halls", "halls", referent_kind="collection"),
            _entity("sheds", "sheds", "sheds", referent_kind="collection"),
            _entity("yards", "yards", "yards", referent_kind="collection"),
            _entity("streets", "streets", "streets", referent_kind="collection"),
        ],
        "requirements": [
            _requirement(
                "facing",
                "Industrial halls and sheds face yards or streets",
                "buildings face yards or streets",
                "spatial_relation",
                arguments=[
                    {"entity_key": "halls", "role": "subject"},
                    {"entity_key": "sheds", "role": "subject"},
                    {"entity_key": "yards", "role": "reference"},
                    {"entity_key": "streets", "role": "reference"},
                ],
                relation="facing",
            )
        ],
    }

    _, draft = compile_semantic_draft(prompt, value)
    logic = next(
        item for item in draft["requirements"] if item["requirement_type"] == "logic"
    )
    operands = [
        item
        for item in draft["requirements"]
        if item["key"] in logic["logic"]["requirement_keys"]
    ]

    assert len(operands) == 2
    assert all(len(item["arguments"]) == 3 for item in operands)
    assert all(
        [argument["role"] for argument in item["arguments"]].count("subject") == 2
        for item in operands
    )
    assert all(
        [argument["role"] for argument in item["arguments"]].count("reference") == 1
        for item in operands
    )


def test_v2_spatial_disjunction_falls_back_to_qualifier_alternatives():
    prompt = "Pale dirt or gravel roads divide sections."
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity("roads", "roads", "roads", referent_kind="collection"),
            _entity("sections", "sections", "sections", referent_kind="collection"),
        ],
        "requirements": [
            _requirement(
                "divide",
                "Pale dirt or gravel roads divide sections",
                "roads divide sections",
                "spatial_relation",
                arguments=[
                    {"entity_key": "roads", "role": "subject"},
                    {"entity_key": "sections", "role": "reference"},
                ],
                relation="divide",
                qualifiers=[
                    {
                        "source_text": "Pale",
                        "occurrence": 0,
                        "kind": "color",
                        "name": "pale",
                    }
                ],
            )
        ],
    }

    _, draft = compile_semantic_draft(prompt, value)
    logic = next(
        item for item in draft["requirements"] if item["requirement_type"] == "logic"
    )
    operands = [
        item
        for item in draft["requirements"]
        if item["key"] in logic["logic"]["requirement_keys"]
    ]

    assert len(operands) == 2
    assert all(len(item["arguments"]) == 2 for item in operands)
    assert {
        qualifier["name"]
        for item in operands
        for qualifier in item["qualifiers"]
        if qualifier["name"] in {"dirt", "gravel"}
    } == {"dirt", "gravel"}


def test_v2_missing_composition_qualifiers_inherit_exact_local_cues():
    prompt = "Markers create a conspicuous ordered patch."
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity("markers", "Markers", "markers", referent_kind="collection"),
            _entity("patch", "patch", "patch"),
        ],
        "requirements": [
            _requirement(
                "observable_patch",
                "Markers create a conspicuous ordered patch",
                "observable ordered patch",
                "distribution",
                arguments=[
                    {"entity_key": "markers", "role": "subject"},
                    {"entity_key": "patch", "role": "reference"},
                ],
                qualifiers=[
                    {
                        "source_text": "conspicuous",
                        "occurrence": 0,
                        "kind": "attribute",
                        "name": "conspicuous",
                    },
                    {
                        "source_text": "ordered",
                        "occurrence": 0,
                        "kind": "layout",
                        "name": "ordered",
                    },
                ],
            ),
            _requirement(
                "composition",
                "create a conspicuous ordered patch",
                "markers create ordered patch",
                "composition",
                arguments=[
                    {"entity_key": "markers", "role": "subject"},
                    {"entity_key": "patch", "role": "reference"},
                ],
                relation="creating",
            ),
        ],
    }

    _, draft = compile_semantic_draft(prompt, value)
    composition = next(
        item
        for item in draft["requirements"]
        if item["requirement_type"] == "composition"
    )

    assert [item["name"] for item in composition["qualifiers"]] == [
        "conspicuous",
        "ordered",
    ]


def test_v2_explicit_singleton_one_of_preserves_only_as_exact_quantity():
    prompt = "Let only a silhouette interrupt the horizon."
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity("silhouette", "silhouette", "silhouette"),
            _entity("horizon", "horizon", "horizon", entity_type="region"),
        ],
        "requirements": [
            _requirement(
                "only_silhouette",
                "only a silhouette interrupt the horizon",
                "only silhouette",
                "logic",
                arguments=[{"entity_key": "silhouette", "role": "subject"}],
                logic={
                    "operator": "one_of",
                    "requirement_keys": ["interrupt"],
                },
                qualifiers=[
                    {
                        "source_text": "only",
                        "occurrence": 0,
                        "kind": "condition",
                        "name": "only",
                    }
                ],
            ),
            _requirement(
                "interrupt",
                "a silhouette interrupt the horizon",
                "silhouette interrupts horizon",
                "spatial_relation",
                arguments=[
                    {"entity_key": "silhouette", "role": "subject"},
                    {"entity_key": "horizon", "role": "reference"},
                ],
                relation="interrupt",
            ),
        ],
    }

    _, draft = compile_semantic_draft(prompt, value)
    quantity = next(
        item
        for item in draft["requirements"]
        if item["requirement_type"] == "quantity"
    )
    spatial = next(
        item
        for item in draft["requirements"]
        if item["requirement_type"] == "spatial_relation"
    )

    assert quantity["quantity"]["mode"] == "exact"
    assert quantity["quantity"]["value"] == 1
    assert spatial["relation"] == "interrupt"


def test_generation_command_verbs_never_create_additions_only_scope():
    prompt = "Add a few benches around a fountain."
    value = {
        "draft_version": "2.0",
        "entities": [
            _entity("benches", "benches", "benches", referent_kind="collection"),
            _entity("fountain", "fountain", "fountain"),
        ],
        "requirements": [
            _requirement(
                "placement",
                "Add a few benches around a fountain",
                "benches around fountain",
                "spatial_relation",
                arguments=[
                    {"entity_key": "benches", "role": "subject"},
                    {"entity_key": "fountain", "role": "reference"},
                ],
                relation="around",
            )
        ],
    }
    graph, _ = compile_semantic_draft(prompt, value)
    node_id = next(iter(graph.effective_weights()))

    binding = _binding_for_node(graph, node_id, TaskMode.GENERATION)

    assert binding.population_scope is PopulationScope.CANDIDATE_ALL
    assert set(binding.entity_scopes.values()) == {PopulationScope.CANDIDATE_ALL}


def test_semantic_verifier_keeps_unplanned_requirement_in_v2_interval(monkeypatch):
    scene_evidence = SimpleNamespace(
        evidence=lambda: {},
        probes_used=lambda: (),
    )
    monkeypatch.setattr(
        semantic_requirements,
        "run_pipeline",
        lambda context: {
            "requirements": [
                {
                    "requirement_id": "atomic_presence",
                    "node_id": "atomic_presence",
                    "verdict": "match",
                    "evaluation_status": "MATCH",
                    "check": {"score": 1.0, "evidence": []},
                    "text": "a bench",
                    "source_span": [0, 7],
                    "population_scope": "candidate_all",
                    "entity_scopes": {},
                    "weight": 1.0,
                    "resolved_by": "stage2",
                    "unknown_reason": None,
                    "rationale": "visible",
                }
            ],
            "non_supported_requirements": [
                {
                    "requirement_id": "atomic_distribution",
                    "node_id": "predicate_dense",
                    "text": "dense benches",
                    "status": "unplanned",
                }
            ],
            "waived_requirements": [],
            "delegated_requirements": [],
            "scene_evidence": scene_evidence,
            "artifacts": {},
            "visual_error": None,
            "identity_grounding": None,
            "bundle": SimpleNamespace(
                bundle_id="bundle_v2",
                task_mode=SimpleNamespace(value="generation"),
                graph={
                    "nodes": [
                        {"id": "atomic_presence", "predicate_type": "existence"},
                        {"id": "predicate_dense", "predicate_type": "distribution"},
                    ],
                    "edges": [],
                },
            ),
        },
    )

    report = semantic_requirements.verify(
        SimpleNamespace(
            ids={"task_bundle_id": "task", "episode_id": "episode"},
            renders_for=lambda protocol: None,
        )
    )

    assert report["status"] == "measured"
    assert report["score"] == 0.5
    assert report["metrics"]["semantic_score"] == 0.5
    assert "semantic_score_upper_bound" not in report["metrics"]
    assert report["metrics"]["legacy_decided_only_semantic_score"] == 1.0
    assert report["metrics"]["capability_coverage"] == 0.5
    assert report["metrics"]["partial_score"] is True


def test_semantic_stress_policy_publishes_holistic_score_with_atomic_diagnostics(
    monkeypatch,
):
    scene_evidence = SimpleNamespace(evidence=lambda: {}, probes_used=lambda: ())
    requirement = {
        "requirement_id": "atomic_layout",
        "node_id": "atomic_layout",
        "verdict": "unknown",
        "evaluation_status": "NOT_EVALUATED",
        "check": None,
        "text": "dense blocks",
        "source_span": [0, 12],
        "population_scope": "candidate_all",
        "entity_scopes": {},
        "weight": 1.0,
        "resolved_by": None,
        "unknown_reason": "visual_evidence_incomplete",
        "rationale": "overview cannot resolve every street detail",
    }
    monkeypatch.setattr(
        semantic_requirements,
        "run_pipeline",
        lambda context: {
            "requirements": [requirement],
            "non_supported_requirements": [],
            "waived_requirements": [],
            "delegated_requirements": [],
            "scene_evidence": scene_evidence,
            "artifacts": {},
            "visual_error": None,
            "identity_grounding": None,
            "stage3": {
                "holistic": {
                    "status": "complete",
                    "overall_score": 0.61,
                    "dimensions": [{"name": "global_prompt_alignment", "score": 0.6}],
                    "summary": "partial but coherent reconstruction",
                    "evaluation_error": None,
                }
            },
            "bundle": SimpleNamespace(
                bundle_id="bundle_v2",
                task_mode=SimpleNamespace(value="generation"),
                graph={
                    "nodes": [
                        {"id": "atomic_layout", "predicate_type": "composition"}
                    ],
                    "edges": [],
                },
            ),
        },
    )

    report = semantic_requirements.verify(
        SimpleNamespace(
            ids={"task_bundle_id": "task", "episode_id": "episode"},
            spec={"semantic_stress_policy": "overview-holistic"},
            renders_for=lambda protocol: None,
        )
    )

    assert report["status"] == "measured"
    assert report["score"] == 0.61
    assert report["metrics"]["score_policy"] == "overview-holistic"
    assert report["metrics"]["atomic_coverage"] == 0.0
    assert report["evidence"]["atomic_results_are_diagnostic"] is True


def test_v2_authoring_prompt_requires_independently_falsifiable_claims():
    assert "one independently falsifiable visual proposition" in SYSTEM_PROMPT
    assert "evidence sharing is handled later" in SYSTEM_PROMPT
    assert "requires separate claims for colour" in SYSTEM_PROMPT
    assert "Use grouped scoring" not in SYSTEM_PROMPT


def test_v2_repair_prompt_makes_validator_errors_actionable():
    draft = {
        "draft_version": "2.0",
        "entities": [_entity("tower", "tower", "tower")],
        "requirements": [
            _requirement(
                "tower_presence",
                "tower",
                "tower present",
                "presence",
                arguments=[{"entity_key": "tower", "role": "subject"}],
            )
        ],
    }

    class Client:
        model = "stage0-v2-repair-test"
        max_tokens = 4096
        _strict_tool_calls = True
        _text_action_mode = False

        def chat(self, messages, tools, **kwargs):
            del tools, kwargs
            self.messages = messages
            return LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall("call", "return_requirement_graph_draft", draft)
                ],
                raw={},
            )

    client = Client()
    Stage0Compiler(client).compile(
        "tower",
        validation_feedback=(
            "Stage0DraftError: incomplete_semantic_coverage at prompt: "
            "uncovered content terms ['tower']"
        ),
        previous_draft=draft,
    )
    repair_text = "\n".join(
        str(block.get("text") or "")
        for block in client.messages[-1].content
        if block.get("type") == "text"
    )
    assert "every listed uncovered token" in repair_text
    assert "exact source_text/source_span" in repair_text
    assert "For ungrounded_text" in repair_text
    assert "coordinated ellipsis" in repair_text
    assert "settlement set in a basin" in repair_text
    assert "For missing_qualifier" in repair_text
    assert "For unmodeled_disjunction" in repair_text
    assert "For semantic_budget_exceeded" in repair_text


def test_v2_coverage_treats_plus_as_a_collection_connector():
    normalize_semantic_ir(
        "boxes plus crates",
        {
            "draft_version": "2.0",
            "entities": [
                _entity(
                    "boxes", "boxes", "boxes", grounding_mode="actor_collection"
                ),
                _entity(
                    "crates", "crates", "crates", grounding_mode="actor_collection"
                ),
            ],
            "requirements": [
                _requirement(
                    "boxes_present",
                    "boxes",
                    "boxes present",
                    "presence",
                    arguments=[{"entity_key": "boxes", "role": "subject"}],
                ),
                _requirement(
                    "crates_present",
                    "crates",
                    "crates present",
                    "presence",
                    arguments=[{"entity_key": "crates", "role": "subject"}],
                ),
            ],
        },
    )


def test_v2_rejects_more_than_96_top_level_scored_claims():
    prompt = "tower"
    value = {
        "draft_version": "2.0",
        "entities": [_entity("tower", "tower", "tower")],
        "requirements": [
            _requirement(
                f"tower_{index}",
                "tower",
                f"tower claim {index}",
                "presence",
                arguments=[{"entity_key": "tower", "role": "subject"}],
            )
            for index in range(97)
        ],
    }

    with pytest.raises(ValueError, match="semantic_budget_exceeded"):
        compile_semantic_draft(prompt, value)


def test_v2_canonical_ir_is_frozen_in_bundle_provenance():
    class Client:
        model = "stage0-v2-test"
        max_tokens = 4096
        _strict_tool_calls = True
        _text_action_mode = False

        def chat(self, messages, tools, **kwargs):
            del messages, tools, kwargs
            return LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(
                        "call",
                        "return_requirement_graph_draft",
                        {
                            "draft_version": "2.0",
                            "entities": [
                                _entity(
                                    "nook",
                                    "cozy reading nook",
                                    "cozy reading nook",
                                )
                            ],
                            "requirements": [
                                _requirement(
                                    "presence",
                                    "cozy reading nook",
                                    "cozy reading nook",
                                    "presence",
                                    arguments=[
                                        {"entity_key": "nook", "role": "subject"}
                                    ],
                                )
                            ],
                        },
                    )
                ],
                raw={},
            )

    compilation = compile_hybrid(
        "Create a cozy reading nook.",
        task_mode="generation",
        llm_client=Client(),
    )
    bundle = freeze_compilation(compilation)

    assert compilation.rule_draft["stage0_semantic_ir"]["draft_version"] == "2.0"
    assert bundle.provenance["stage0_semantic_ir"]["coverage"][
        "coverage_ratio"
    ] == 1.0
    assert len(bundle.provenance["stage0_semantic_ir"]["requirements"]) == 1
