from pathlib import Path

from code4scene.evaluation.requirement_graph.bundle import (
    EvidenceClass,
    FrozenVerificationBundle,
    PopulationScope,
    RequirementBinding,
    TaskMode,
    make_evaluation_binding,
)
from code4scene.evaluation.requirement_graph.contracts import (
    ArgumentEdge,
    ClaimVerdict,
    EntityNode,
    PredicateNode,
    ReferentKind,
    RequirementGraph,
    RequirementMemberEdge,
    RequirementNode,
    RootRequirement,
)
from code4scene.evaluation.requirement_graph.evidence_adapter import (
    build_scene_inventory,
)
from code4scene.evaluation.requirement_graph.pipeline import (
    _stage1_atomic,
    _stage2_skip_nodes,
)
from code4scene.evaluation.requirement_graph.stage1 import evaluate_stage1
from code4scene.evaluation.requirement_graph.stage2_tasks import (
    build_stage2_tasks,
)
from code4scene.evaluation.requirement_graph.structured_deterministic import (
    evaluate_structured_requirements,
)
from code4scene.evaluation.ue_evidence import SceneEvidence


def _actor(
    actor_id: str,
    category: str,
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 50.0),
    extent: tuple[float, float, float] = (25.0, 25.0, 50.0),
    material_path: str | None = None,
) -> dict:
    return {
        "stable_actor_id": actor_id,
        "label": actor_id,
        "name": actor_id,
        "class": "/Script/Engine.StaticMeshActor",
        "asset_path": f"/Game/Test/SM_{category.title()}",
        "asset_category": category,
        "asset_category_source": "asset_catalog",
        "component_asset_paths": [f"/Game/Test/SM_{category.title()}"],
        "material_paths": [material_path] if material_path else [],
        "component_material_slots": [],
        "actor_tags": [],
        "transform": {
            "location_cm": list(origin),
            "rotation_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "bounds": {
            "origin_cm": list(origin),
            "extent_cm": list(extent),
        },
        "properties": {},
        "export_diagnostics": {"hidden_in_editor": False},
        "export_errors": [],
    }


def _bundle(
    prompt: str,
    predicate: PredicateNode,
    entities: tuple[EntityNode, ...],
) -> FrozenVerificationBundle:
    requirement = RequirementNode(
        "requirement_root",
        prompt,
        {"start": 0, "end": len(prompt)},
    )
    edges = [
        RequirementMemberEdge(
            requirement.id,
            predicate.id,
            "scored_facet",
            1.0,
        ),
        *(
            RequirementMemberEdge(
                requirement.id,
                entity.id,
                "support_only",
            )
            for entity in entities
        ),
        *(
            ArgumentEdge(
                predicate.id,
                entity.id,
                "collection" if len(entities) == 1 else (
                    "subject" if index == 0 else "reference"
                ),
                index,
            )
            for index, entity in enumerate(entities)
        ),
    ]
    graph = RequirementGraph(
        prompt=prompt,
        nodes=(*entities, predicate, requirement),
        edges=tuple(edges),
        roots=(RootRequirement(requirement.id, 1.0),),
        schema_version="2.0",
    )
    scopes = {
        entity.id: PopulationScope.CANDIDATE_ALL for entity in entities
    }
    binding = RequirementBinding(
        requirement_id="semantic_requirement",
        node_id=predicate.id,
        source_span=(0, len(prompt)),
        source_text=prompt,
        primary_owner="requirement_graph",
        evidence_class=EvidenceClass.OPEN_ENDED,
        population_scope=PopulationScope.CANDIDATE_ALL,
        entity_scopes=scopes,
    )
    provenance = {
        "authoring_parser": "structured_test",
        "closed_categories": [entity.name for entity in entities],
    }
    return FrozenVerificationBundle(
        bundle_id="structured_test_bundle",
        task_mode=TaskMode.GENERATION,
        graph=graph,
        requirements=(binding,),
        evaluations=(
            make_evaluation_binding(binding, graph, provenance),
        ),
        provenance=provenance,
        review={"approved": True},
    )


def _evaluate(
    bundle: FrozenVerificationBundle,
    candidate: dict,
    *,
    asset_catalog: bool = True,
):
    evidence = SceneEvidence(
        candidate=candidate,
        candidate_path=Path("/tmp/structured-candidate.json"),
        case_spec=None,
        case_id="structured-test",
        root=Path("/tmp/structured-test"),
        independent=True,
        asset_catalog_id=("structured-test-catalog" if asset_catalog else None),
        asset_catalog_size=(10 if asset_catalog else 0),
        categories_resolved=(
            len(candidate.get("actors") or []) if asset_catalog else 0
        ),
    )
    scene = build_scene_inventory(
        evidence,
        closed_categories=tuple(bundle.provenance["closed_categories"]),
    )
    scoped_ids = {
        entity_id: scene.scope(scope).actor_ids
        for entity_id, scope in bundle.semantic_entity_scopes().items()
    }
    stage1 = evaluate_stage1(
        bundle.graph,
        scene.inventory,
        scene.scene_bounds,
        eligible_actor_ids_by_entity=scoped_ids,
    )
    structured = evaluate_structured_requirements(
        bundle,
        evidence,
        scene,
        stage1,
    )
    deterministic = structured.assessments
    atomic = _stage1_atomic(bundle, stage1, scene, deterministic)
    tasks = build_stage2_tasks(
        bundle.graph,
        stage1,
        skip_node_ids=_stage2_skip_nodes(bundle, atomic),
    )
    return structured, tasks


def test_exact_object_quantity_resolves_before_rgb_capture():
    prompt = "exactly two chairs"
    entity = EntityNode(
        "entity_chairs",
        "chairs",
        "chair",
        {"start": 12, "end": 18},
        referent_kind=ReferentKind.COLLECTION,
    )
    predicate = PredicateNode(
        "predicate_quantity",
        prompt,
        "exactly two chairs",
        "quantity",
        {"start": 0, "end": len(prompt)},
        semantic_parameters={
            "stage0_ir_version": "2.0",
            "requirement_type": "quantity",
            "quantity": {
                "mode": "exact",
                "value": 2,
                "unit": "objects",
            },
        },
    )
    bundle = _bundle(prompt, predicate, (entity,))
    candidate = {
        "actors": [
            _actor("chair_a", "chair", origin=(-50.0, 0.0, 50.0)),
            _actor("chair_b", "chair", origin=(50.0, 0.0, 50.0)),
        ],
        "export_metadata": {"status": "success"},
    }

    structured, tasks = _evaluate(bundle, candidate)

    assert len(structured.assessments) == 1
    assert structured.assessments[0].verdict is ClaimVerdict.MATCH
    assert structured.assessments[0].resolution_stage == "stage1_structured"
    assert structured.assessments[0].check["observed"]["count"] == 2
    assert tasks == ()


def test_name_grounded_lower_bound_count_resolves_without_asset_categories():
    prompt = "many trees"
    entity = EntityNode(
        "entity_trees",
        "trees",
        "trees",
        {"start": 5, "end": 10},
        referent_kind=ReferentKind.COLLECTION,
    )
    predicate = PredicateNode(
        "predicate_many_trees",
        prompt,
        prompt,
        "quantity",
        {"start": 0, "end": len(prompt)},
        semantic_parameters={
            "stage0_ir_version": "2.0",
            "requirement_type": "quantity",
            "quantity": {
                "mode": "qualitative",
                "qualitative": "many",
                "unit": "objects",
            },
        },
    )
    bundle = _bundle(prompt, predicate, (entity,))
    actors = [
        _actor(
            f"tree_{index}",
            "tree",
            origin=(float(index * 100), 0.0, 50.0),
        )
        for index in range(8)
    ]
    for actor in actors:
        actor.pop("asset_category", None)
        actor.pop("asset_category_source", None)
    candidate = {
        "actors": actors,
        "export_metadata": {"status": "success"},
    }

    structured, tasks = _evaluate(bundle, candidate, asset_catalog=False)

    assert len(structured.assessments) == 1
    assessment = structured.assessments[0]
    assert assessment.verdict is ClaimVerdict.MATCH
    assert assessment.check["measurement"] == (
        "ue_canonical_name_grounded_actor_lower_bound"
    )
    assert assessment.check["observed"]["count"] == 8
    assert tasks == ()


def test_name_grounding_does_not_claim_exact_count_without_population_closure():
    prompt = "exactly two trees"
    entity = EntityNode(
        "entity_trees",
        "trees",
        "trees",
        {"start": 12, "end": 17},
        referent_kind=ReferentKind.COLLECTION,
    )
    predicate = PredicateNode(
        "predicate_two_trees",
        prompt,
        prompt,
        "quantity",
        {"start": 0, "end": len(prompt)},
        semantic_parameters={
            "stage0_ir_version": "2.0",
            "requirement_type": "quantity",
            "quantity": {"mode": "exact", "value": 2, "unit": "objects"},
        },
    )
    bundle = _bundle(prompt, predicate, (entity,))
    actors = [_actor(f"tree_{index}", "tree") for index in range(2)]
    for actor in actors:
        actor.pop("asset_category", None)
        actor.pop("asset_category_source", None)
    candidate = {
        "actors": actors,
        "export_metadata": {"status": "success"},
    }

    structured, tasks = _evaluate(bundle, candidate, asset_catalog=False)

    assert structured.assessments == ()
    assert structured.attempts[0]["reason"] == "category_population_incomplete"
    assert [value.node_id for value in tasks] == [predicate.id]


def test_intrinsic_spatial_relation_resolves_from_actor_bounds():
    prompt = "vase on top of table"
    vase = EntityNode(
        "entity_vase",
        "vase",
        "vase",
        {"start": 0, "end": 4},
    )
    table = EntityNode(
        "entity_table",
        "table",
        "table",
        {"start": 15, "end": 20},
    )
    predicate = PredicateNode(
        "predicate_relation",
        prompt,
        "vase on top of table",
        "spatial_relation",
        {"start": 0, "end": len(prompt)},
        semantic_parameters={
            "stage0_ir_version": "2.0",
            "requirement_type": "spatial_relation",
            "relation": "on_top_of",
        },
    )
    bundle = _bundle(prompt, predicate, (vase, table))
    candidate = {
        "actors": [
            _actor(
                "vase_actor",
                "vase",
                origin=(0.0, 0.0, 125.0),
                extent=(20.0, 20.0, 25.0),
            ),
            _actor(
                "table_actor",
                "table",
                origin=(0.0, 0.0, 50.0),
                extent=(100.0, 100.0, 50.0),
            ),
        ],
        "export_metadata": {"status": "success"},
    }

    structured, tasks = _evaluate(bundle, candidate)

    assert len(structured.assessments) == 1
    assert structured.assessments[0].verdict is ClaimVerdict.MATCH
    assert structured.assessments[0].family == "STRUCTURED_SPATIAL_RELATION"
    assert tasks == ()


def test_texture_free_material_colour_resolves_before_rgb_capture():
    prompt = "a red chair"
    material_path = "/Game/Test/MI_Red"
    chair = EntityNode(
        "entity_chair",
        "chair",
        "chair",
        {"start": 6, "end": 11},
    )
    predicate = PredicateNode(
        "predicate_colour",
        prompt,
        "a red chair",
        "attribute",
        {"start": 0, "end": len(prompt)},
        semantic_parameters={
            "stage0_ir_version": "2.0",
            "requirement_type": "attribute",
            "qualifiers": [{"kind": "color", "name": "red"}],
        },
    )
    bundle = _bundle(prompt, predicate, (chair,))
    candidate = {
        "actors": [
            _actor("chair_actor", "chair", material_path=material_path),
        ],
        "material_parameter_catalog": {
            material_path: {
                "schema_version": "1.0",
                "probe_status": "success",
                "source": "ue_material_editing_library_resolved_parameters_v1",
                "resolved_vector_parameters": {
                    "BaseColor": [1.0, 0.0, 0.0, 1.0],
                },
                "used_texture_paths": [],
            },
        },
        "export_metadata": {"status": "success"},
    }

    structured, tasks = _evaluate(bundle, candidate)

    assert len(structured.assessments) == 1
    assert structured.assessments[0].verdict is ClaimVerdict.MATCH
    assert structured.assessments[0].family == "STRUCTURED_MATERIAL_COLOUR"
    assert tasks == ()


def test_texture_free_wrong_material_colour_is_structured_mismatch():
    prompt = "a red chair"
    material_path = "/Game/Test/MI_Blue"
    chair = EntityNode(
        "entity_chair",
        "chair",
        "chair",
        {"start": 6, "end": 11},
    )
    predicate = PredicateNode(
        "predicate_colour",
        prompt,
        "a red chair",
        "attribute",
        {"start": 0, "end": len(prompt)},
        semantic_parameters={
            "stage0_ir_version": "2.0",
            "requirement_type": "attribute",
            "qualifiers": [{"kind": "color", "name": "red"}],
        },
    )
    bundle = _bundle(prompt, predicate, (chair,))
    candidate = {
        "actors": [
            _actor("chair_actor", "chair", material_path=material_path),
        ],
        "material_parameter_catalog": {
            material_path: {
                "schema_version": "1.0",
                "probe_status": "success",
                "source": "ue_material_editing_library_resolved_parameters_v1",
                "resolved_vector_parameters": {
                    "BaseColor": [0.0, 0.03, 1.0, 1.0],
                },
                "used_texture_paths": [],
            },
        },
        "export_metadata": {"status": "success"},
    }

    structured, tasks = _evaluate(bundle, candidate)

    assert len(structured.assessments) == 1
    assert structured.assessments[0].verdict is ClaimVerdict.MISMATCH
    assert structured.assessments[0].check["observed"]["colour"] == "blue"
    assert tasks == ()


def test_textured_material_colour_falls_back_to_visual_capture():
    prompt = "a red chair"
    material_path = "/Game/Test/MI_Textured"
    chair = EntityNode(
        "entity_chair",
        "chair",
        "chair",
        {"start": 6, "end": 11},
    )
    predicate = PredicateNode(
        "predicate_colour",
        prompt,
        "a red chair",
        "attribute",
        {"start": 0, "end": len(prompt)},
        semantic_parameters={
            "stage0_ir_version": "2.0",
            "requirement_type": "attribute",
            "qualifiers": [{"kind": "color", "name": "red"}],
        },
    )
    bundle = _bundle(prompt, predicate, (chair,))
    candidate = {
        "actors": [
            _actor("chair_actor", "chair", material_path=material_path),
        ],
        "material_parameter_catalog": {
            material_path: {
                "schema_version": "1.0",
                "probe_status": "success",
                "source": "ue_material_editing_library_resolved_parameters_v1",
                "resolved_vector_parameters": {
                    "BaseColor": [1.0, 0.0, 0.0, 1.0],
                },
                "used_texture_paths": ["/Game/Test/T_Chair"],
            },
        },
        "export_metadata": {"status": "success"},
    }

    structured, tasks = _evaluate(bundle, candidate)

    assert structured.assessments == ()
    assert structured.attempts[0]["status"] == "visual_fallback"
    assert structured.attempts[0]["reason"] == (
        "material_texture_participation_is_not_empty"
    )
    assert len(tasks) == 1
