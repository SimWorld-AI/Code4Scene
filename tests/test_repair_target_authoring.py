"""Image-to-scene Stage 0 derives targets from GT/Input, never pixels."""

from __future__ import annotations

import copy

import pytest

from code4scene.evaluation.requirement_graph.actor_inventory import (
    ActorBounds,
    ActorInventorySnapshot,
    build_actor_descriptor,
)
from code4scene.evaluation.requirement_graph.bundle import (
    BindingSourceKind,
    EvidenceClass,
    FrozenVerificationBundle,
    UnknownReason,
    PopulationScope,
)
from code4scene.evaluation.requirement_graph.contracts import (
    ComparisonOperator,
    EntityEvaluationRoute,
    EntityNode,
    Polarity,
    PredicateNode,
    PredicateType,
    ReferentKind,
    SceneBounds,
)
from code4scene.evaluation.requirement_graph.repair_target_authoring import (
    compile_repair_target_bundle,
    derive_repair_targets,
)
from code4scene.evaluation.requirement_graph.evidence_adapter import (
    SceneInventoryEvidence,
    ScopeResolution,
)
from code4scene.evaluation.requirement_graph.deterministic import (
    evaluate_deterministic_rules,
    evaluate_repair_target_fields,
)
from code4scene.evaluation.requirement_graph.identity_grounding import (
    resolve_exact_identity_bindings,
)
from code4scene.evaluation.requirement_graph.pipeline import (
    _default_stage2_budget,
    _overlay_identity_grounding,
    _scope_actor_ids,
)
from code4scene.evaluation.requirement_graph.stage1 import evaluate_stage1
from code4scene.evaluation.requirement_graph.stage2_tasks import (
    build_stage2_queries,
    build_stage2_tasks,
)


SCENE_BOUNDS = SceneBounds(
    (-2000.0, -2000.0, -1000.0),
    (2000.0, 2000.0, 2000.0),
)


def _actor(
    stable_id: str,
    label: str,
    asset: str,
    *,
    location: tuple[float, float, float] = (0.0, 0.0, 100.0),
) -> dict:
    return {
        "stable_actor_id": stable_id,
        "actor_guid": stable_id.upper(),
        "actor_path": f"/Game/Test.Test:PersistentLevel.{label}",
        "label": label,
        "name": f"{label}_1",
        "class": "/Script/Engine.StaticMeshActor",
        "asset_path": f"/Game/Test/{asset}.{asset}",
        "actor_origin": "source",
        "actor_tags": [],
        "component_asset_paths": [f"/Game/Test/{asset}.{asset}"],
        "component_material_slots": [],
        "material_paths": [],
        "properties": {},
        "transform": {
            "location_cm": list(location),
            "rotation_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "bounds": {
            "origin_cm": list(location),
            "extent_cm": [50.0, 20.0, 80.0],
        },
        "export_diagnostics": {"hidden_in_editor": False},
        "export_errors": [],
    }


def _scene(*actors: dict) -> dict:
    return {
        "actors": list(actors),
        "actor_count": len(actors),
        "export_metadata": {"status": "success"},
    }


def test_image_stage0_api_rejects_a_prompt_argument():
    garage = _actor("garage", "GarageDoors", "SM_GarageDoors")

    with pytest.raises(TypeError):
        compile_repair_target_bundle(
            _scene(),
            _scene(garage),
            "this prompt must never reach image-to-scene Stage 0",
        )


def test_gt_only_actor_becomes_one_addition_scoped_visual_target():
    retained = _actor("keep", "Wall", "SM_Wall")
    garage = _actor(
        "garage-gt",
        "Modular_Environment_Garage_Doors",
        "SM_Modular_Environment_Garage_Doors",
    )

    bundle = compile_repair_target_bundle(
        _scene(retained),
        _scene(retained, garage),
        task_id="garage-case",
    )

    assert len(bundle.requirements) == 1
    assert len(bundle.graph.nodes) == 3
    assert bundle.provenance["reference_images_consumed"] is False
    assert bundle.provenance["candidate_consumed_during_authoring"] is False
    assert bundle.provenance["repair_target_count"] == 1
    target = bundle.provenance["repair_targets"][0]
    assert target["operation"] == "add"
    assert target["gt_actor"]["asset_path"].endswith(
        "SM_Modular_Environment_Garage_Doors.SM_Modular_Environment_Garage_Doors"
    )

    requirement = bundle.requirements[0]
    assert requirement.evidence_class is EvidenceClass.GT
    assert requirement.population_scope is PopulationScope.ADDITIONS
    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    assert entity.evaluation_route is EntityEvaluationRoute.STAGE2_VISUAL
    assert "Garage Doors" == entity.name
    assert "Engine" not in entity.aliases
    assert "designated restored Garage Doors" in entity.text
    assert bundle.provenance["authoring_inputs"] == [
        "input_scene_graph",
        "gt_scene_graph",
    ]
    assert "task_prompt" not in bundle.provenance

    evaluation = bundle.evaluations[0]
    assert evaluation.source_provenance["kind"] == BindingSourceKind.GT_INPUT_DIFF.value
    assert evaluation.source_provenance["operation"] == "add"
    assert evaluation.evaluator_id == "semantic_requirements.atomic_rule"

    restored = FrozenVerificationBundle.from_dict(bundle.to_dict())
    assert restored.to_dict() == bundle.to_dict()

    budget = _default_stage2_budget(bundle)
    assert budget.global_capture_limits_enabled is False


def test_identical_additions_become_one_collection_count_leaf():
    left_lamp = _actor("lamp-left", "BedLampLeft", "SM_Bed_Lamp")
    right_lamp = _actor(
        "lamp-right",
        "BedLampRight",
        "SM_Bed_Lamp",
        location=(200.0, 0.0, 100.0),
    )

    bundle = compile_repair_target_bundle(
        _scene(),
        _scene(left_lamp, right_lamp),
    )

    assert bundle.provenance["repair_target_count"] == 2
    assert bundle.provenance["semantic_repair_unit_count"] == 1
    assert len(bundle.requirements) == 1
    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    predicate = next(
        node for node in bundle.graph.nodes if isinstance(node, PredicateNode)
    )
    assert entity.referent_kind is ReferentKind.COLLECTION
    assert predicate.predicate_type is PredicateType.COUNT
    assert "at least 2 restored Bed Lamp objects" in entity.text
    assert predicate.constraint.operator is ComparisonOperator.GTE
    assert bundle.requirements[0].deterministic_rule_id is not None
    assert bundle.provenance["deterministic_rule_draft"]["items"][0][
        "type"
    ] == "object_count"
    audit = bundle.provenance["repair_target_by_node"][predicate.id]
    assert audit["group_size"] == 2
    assert audit["member_target_ids"] == [
        "repair_target_0001",
        "repair_target_0002",
    ]


def test_image_stage0_does_not_invent_prompt_only_anchor_relations():
    table = _actor("table", "DiningTable", "SM_Dining_Table")
    chairs = tuple(
        _actor(
            f"chair-{index}",
            f"DiningChair{index}",
            "SM_Dining_Chair",
            location=location,
        )
        for index, location in enumerate(
            (
                (200.0, 0.0, 100.0),
                (0.0, 200.0, 100.0),
                (-200.0, 0.0, 100.0),
                (0.0, -200.0, 100.0),
            )
        )
    )

    bundle = compile_repair_target_bundle(
        _scene(table),
        _scene(table, *chairs),
    )

    assert [
        item["type"]
        for item in bundle.provenance["deterministic_rule_draft"]["items"]
    ] == ["object_count"]
    assert len(bundle.requirements) == 1
    assert all(
        requirement.deterministic_rule_id
        for requirement in bundle.requirements
    )
    assert not any(
        isinstance(node, PredicateNode)
        and node.predicate_type is PredicateType.SPATIAL_RELATION
        for node in bundle.graph.nodes
    )


def test_image_stage0_keeps_gt_geometry_as_audit_not_prompt_relation():
    shelf = _actor("shelf", "StaticMeshActor_47", "SM_Shelf01")
    shelf["bounds"]["extent_cm"] = [200.0, 100.0, 100.0]
    boxes = tuple(
        _actor(
            f"box-{index}",
            f"Sm_Box{index + 11}",
            "Sm_Box02",
            location=(offset, 0.0, 100.0),
        )
        for index, offset in enumerate((-40.0, 40.0))
    )
    for box in boxes:
        box["bounds"]["extent_cm"] = [15.0, 15.0, 15.0]

    bundle = compile_repair_target_bundle(
        _scene(shelf),
        _scene(shelf, *boxes),
    )

    assert [
        item["type"]
        for item in bundle.provenance["deterministic_rule_draft"]["items"]
    ] == ["object_count"]
    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    geometry = bundle.provenance["repair_target_by_entity"][entity.id][
        "member_gt_geometry"
    ]
    assert len(geometry) == 2
    assert {item["actor_identity"] for item in geometry} == {
        "stable:box-0",
        "stable:box-1",
    }


def test_image_stage0_does_not_guess_a_support_from_geometry():
    bench = _actor("bench", "SM_Dungeon_bench", "SM_Dungeon_bench")
    bench["bounds"]["origin_cm"] = [0.0, 0.0, 0.0]
    bench["bounds"]["extent_cm"] = [100.0, 100.0, 50.0]
    misleading_table = _actor(
        "table",
        "SM_Table",
        "SM_Table",
        location=(109.0, 0.0, 0.0),
    )
    misleading_table["bounds"]["origin_cm"] = [109.0, 0.0, 0.0]
    misleading_table["bounds"]["extent_cm"] = [100.0, 100.0, 50.0]
    bowl = _actor(
        "bowl",
        "SM_Bowls",
        "SM_Bowls",
        location=(0.0, 0.0, 60.0),
    )
    bowl["bounds"]["extent_cm"] = [10.0, 10.0, 10.0]

    bundle = compile_repair_target_bundle(
        _scene(bench, misleading_table),
        _scene(bench, misleading_table, bowl),
    )

    items = bundle.provenance["deterministic_rule_draft"]["items"]
    assert [item["type"] for item in items] == ["object_presence"]
    assert all("object" not in item for item in items)


def test_image_stage0_does_not_create_unrequested_anchor_entities():
    walls = tuple(
        _actor(
            f"wall-{index}",
            f"MA_Wall{index}",
            "SM_Wall",
            location=(offset, 0.0, 100.0),
        )
        for index, offset in enumerate((-200.0, 0.0, 200.0), start=1)
    )
    hangers = tuple(
        _actor(
            f"hanger-{index}",
            f"SM_Swordhanger{index}",
            "SM_Swordhanger",
            location=(offset, 25.0, 100.0),
        )
        for index, offset in enumerate((-200.0, 0.0, 200.0), start=1)
    )
    for wall in walls:
        wall["bounds"]["extent_cm"] = [80.0, 1.0, 100.0]
    for hanger in hangers:
        hanger["bounds"]["extent_cm"] = [50.0, 20.0, 20.0]

    bundle = compile_repair_target_bundle(
        _scene(*walls),
        _scene(*walls, *hangers),
    )

    assert [
        item["type"]
        for item in bundle.provenance["deterministic_rule_draft"]["items"]
    ] == ["object_count"]
    assert not any(
        isinstance(node, EntityNode) and node.id.startswith("repair_anchor_")
        for node in bundle.graph.nodes
    )


def test_distinct_additions_receive_target_scoped_claims():
    box = _actor("box", "RestoredBox", "SM_Box01")
    vase = _actor("vase", "RestoredVase", "SM_Vase01")
    bundle = compile_repair_target_bundle(
        _scene(),
        _scene(box, vase),
    )

    assert len(bundle.requirements) == 2
    texts = {
        node.name: node.text
        for node in bundle.graph.nodes
        if isinstance(node, EntityNode) and node.id.startswith("repair_e_")
    }
    assert "designated restored Box" in texts["Box"]
    assert "designated restored Vase" in texts["Vase"]
    assert all("judging only this target" in value for value in texts.values())
    assert all("requested container" not in value for value in texts.values())


def test_distinct_gt_asset_identities_remain_distinct_without_prompt_aliases():
    binder_hole = _actor("binder-hole", "BinderHole", "SM_BinderHoleC")
    binder_plain = _actor(
        "binder-plain",
        "BinderPlain",
        "SM_Binder03",
        location=(200.0, 0.0, 100.0),
    )
    bundle = compile_repair_target_bundle(
        _scene(),
        _scene(binder_hole, binder_plain),
    )

    assert bundle.provenance["repair_target_count"] == 2
    assert bundle.provenance["semantic_repair_unit_count"] == 2
    entities = tuple(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    assert {node.name for node in entities} == {"Binder Hole C", "Binder"}
    assert all(node.referent_kind is ReferentKind.INDIVIDUAL for node in entities)


def test_image_claim_contains_no_prompt_only_layout_clause():
    box = _actor("box", "RestoredBox", "SM_Box01")
    bundle = compile_repair_target_bundle(
        _scene(),
        _scene(box),
    )

    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    assert "designated restored Box" in entity.text
    assert "requested container" not in entity.text
    assert "original" not in entity.text


def test_image_collection_claim_is_identity_and_count_only():
    left_box = _actor("left", "LeftBox", "SM_Box01")
    right_box = _actor(
        "right",
        "RightBox",
        "SM_Box02",
        location=(200.0, 0.0, 100.0),
    )
    bundle = compile_repair_target_bundle(
        _scene(),
        _scene(left_box, right_box),
    )

    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    assert "at least 2 restored Box objects" in entity.text
    assert "requested container" not in entity.text
    assert "spacing" not in entity.text


def test_authored_dining_chair_image_claim_does_not_read_prompt_relations():
    chairs = tuple(
        _actor(
            f"chair-{index}",
            f"RestoredDiningChair{index}",
            "SM_Chair",
            location=(index * 100.0, 0.0, 100.0),
        )
        for index in range(4)
    )
    bundle = compile_repair_target_bundle(
        _scene(),
        _scene(*chairs),
    )

    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    assert entity.name == "Chair"
    assert "at least 4 restored Chair objects" in entity.text
    assert "dining table" not in entity.text
    assert "facing" not in entity.text


def test_authored_place_setting_uses_only_gt_actor_identity():
    bowls = _actor("bowls", "RestoredBowls", "SM_Bowls")
    mug = _actor("mug", "RestoredDungeonMug", "SM_Dungeon_mug")
    bundle = compile_repair_target_bundle(
        _scene(),
        _scene(bowls, mug),
    )

    entities = tuple(
        node
        for node in bundle.graph.nodes
        if isinstance(node, EntityNode) and node.id.startswith("repair_e_")
    )
    assert {node.name.casefold() for node in entities} == {
        "bowls",
        "dungeon mug",
    }
    assert all("tabletop" not in node.text for node in entities)


def test_authored_taxonomy_uses_gt_asset_name_without_prompt_alias():
    left = _actor("sword-left", "LeftCinquedea", "SM_Medieval_Weapons_Cinquedea")
    right = _actor(
        "sword-right",
        "RightCinquedea",
        "SM_Medieval_Weapons_Cinquedea",
        location=(0.0, 1.0, 100.0),
    )
    bundle = compile_repair_target_bundle(
        _scene(),
        _scene(left, right),
    )

    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    assert entity.name == "Medieval Weapons Cinquedea"
    assert len(bundle.requirements) == 1
    assert "at least 2 restored Medieval Weapons Cinquedea objects" in entity.text
    assert "crossed" not in entity.text
    assert "door" not in entity.text


def test_image_stage0_does_not_infer_row_without_an_explicit_structured_fact():
    hangers = tuple(
        _actor(
            f"hanger-{index}",
            f"SwordHanger{index}",
            "SM_Swordhanger",
            location=(index * 200.0, 0.0, 100.0),
        )
        for index in range(3)
    )
    bundle = compile_repair_target_bundle(
        _scene(),
        _scene(*hangers),
    )

    predicates = {
        node.name: node
        for node in bundle.graph.nodes
        if isinstance(node, PredicateNode)
    }
    assert set(predicates) == {"count"}
    assert predicates["count"].predicate_type is PredicateType.COUNT


def test_authored_torch_group_preserves_gt_geometry_for_scene_diff_audit():
    torches = tuple(
        _actor(
            f"torch-{index}",
            f"RestoredTorch{index}",
            "SM_torch_02",
            location=(index * 300.0, 0.0, 100.0),
        )
        for index in range(3)
    )
    bundle = compile_repair_target_bundle(
        _scene(),
        _scene(*torches),
    )

    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    assert entity.name == "torch"
    assert "at least 3 restored torch objects" in entity.text
    assert "row" not in entity.text
    target = bundle.provenance["repair_target_by_entity"][entity.id]
    assert [
        value["location_cm"] for value in target["member_gt_geometry"]
    ] == [[0.0, 0.0, 100.0], [300.0, 0.0, 100.0], [600.0, 0.0, 100.0]]


def test_sparse_gt_schema_does_not_create_repairs_for_rich_input_fields():
    retained = _actor("keep", "Wall", "SM_Wall")
    sky = _actor("sky", "SkySphereBlueprint", "SM_SkySphere")
    sky["class"] = "/Engine/EngineSky/BP_Sky_Sphere.BP_Sky_Sphere_C"
    garage = _actor(
        "garage-gt",
        "Modular_Environment_Garage_Doors",
        "SM_Modular_Environment_Garage_Doors",
    )

    def answer_key_actor(actor: dict) -> dict:
        return {
            "stable_actor_id": actor["stable_actor_id"],
            "actor_path": actor["actor_path"],
            "actor_tags": actor["actor_tags"],
            "label": actor["label"],
            "class": actor["class"],
            "asset_path": actor["asset_path"],
            "transform": copy.deepcopy(actor["transform"]),
        }

    sparse_retained = answer_key_actor(retained)
    sparse_sky = answer_key_actor(sky)
    sparse_sky["class"] = "/Script/Engine.BP_Sky_Sphere_C"
    sparse_garage = answer_key_actor(garage)

    targets = derive_repair_targets(
        _scene(retained, sky),
        _scene(sparse_retained, sparse_sky, sparse_garage),
    )

    assert len(targets) == 1
    assert targets[0].operation == "add"
    assert targets[0].name == "Garage Doors"


def _descriptor(actor_id: str, asset: str):
    return build_actor_descriptor(
        live_actor_id=actor_id,
        asset_path=f"/Game/Test/{asset}.{asset}",
        bounds=ActorBounds((0.0, 0.0, 100.0), (50.0, 20.0, 80.0)),
        active=True,
        renderable=True,
        in_current_level=True,
    )


def test_exact_target_skips_rgb_while_similar_target_routes_to_visual_identity():
    garage = _actor(
        "garage-gt",
        "Garage_Doors",
        "SM_Garage_Doors",
    )
    bundle = compile_repair_target_bundle(
        _scene(),
        _scene(garage),
    )
    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    predicate = next(
        node for node in bundle.graph.nodes if isinstance(node, PredicateNode)
    )
    assert predicate.predicate_type is PredicateType.EXISTENCE

    exact = ActorInventorySnapshot(
        (_descriptor("candidate-exact", "SM_Garage_Doors"),)
    )
    exact_stage1 = evaluate_stage1(
        bundle.graph,
        exact,
        SCENE_BOUNDS,
        eligible_actor_ids_by_entity={entity.id: ("candidate-exact",)},
    )
    assert exact_stage1.for_entity(entity.id).verdict.value == "MATCH"
    assert exact_stage1.for_entity(entity.id).routed_to_stage2 is False
    assert exact_stage1.for_entity(entity.id).matched_actor_ids == (
        "candidate-exact",
    )
    exact_tasks = build_stage2_tasks(bundle.graph, exact_stage1)
    assert exact_tasks == ()
    assert build_stage2_queries(bundle.graph, exact_tasks) == ()

    similar = ActorInventorySnapshot(
        (_descriptor("candidate-similar", "SM_Industrial_Twin_Doors"),)
    )
    similar_stage1 = evaluate_stage1(
        bundle.graph,
        similar,
        SCENE_BOUNDS,
        eligible_actor_ids_by_entity={entity.id: ("candidate-similar",)},
    )
    assert similar_stage1.for_entity(entity.id).verdict.value == "UNKNOWN"
    assert similar_stage1.for_entity(entity.id).routed_to_stage2 is True
    tasks = build_stage2_tasks(bundle.graph, similar_stage1)
    assert [task.node_id for task in tasks] == ["repair_p_0001"]
    assert [query.query_id for query in build_stage2_queries(bundle.graph, tasks)] == [
        entity.id
    ]


def test_retained_repair_binds_only_its_stable_actor_not_every_same_asset():
    source = _actor("target", "AwningTarget", "SM_AwningRich")
    desired = copy.deepcopy(source)
    desired["transform"]["location_cm"] = [100.0, 0.0, 100.0]
    bundle = compile_repair_target_bundle(_scene(source), _scene(desired))
    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    candidates = (
        copy.deepcopy(desired),
        _actor("other-1", "AwningOther1", "SM_AwningRich"),
        _actor("other-2", "AwningOther2", "SM_AwningRich"),
    )

    class Evidence:
        def candidate_actors(self):
            return candidates

    descriptors = tuple(
        build_actor_descriptor(
            live_actor_id=f"stable:{actor['stable_actor_id']}",
            unreal_name=actor["name"],
            actor_class=actor["class"],
            asset_path=actor["asset_path"],
            actor_label=actor["label"],
            bounds=ActorBounds(
                tuple(actor["bounds"]["origin_cm"]),
                tuple(actor["bounds"]["extent_cm"]),
            ),
            active=True,
            renderable=True,
            in_current_level=True,
        )
        for actor in candidates
    )
    actor_ids = tuple(value.live_actor_id for value in descriptors)
    scene = SceneInventoryEvidence(
        inventory=ActorInventorySnapshot(descriptors),
        scene_bounds=SCENE_BOUNDS,
        scopes={
            PopulationScope.CANDIDATE_ALL: ScopeResolution(
                PopulationScope.CANDIDATE_ALL,
                actor_ids,
                source="candidate_scene_snapshot",
            ),
            PopulationScope.ADDITIONS: ScopeResolution(
                PopulationScope.ADDITIONS,
                actor_ids[1:],
                source="validated_before_after_diff",
            ),
            PopulationScope.GT_TARGETS: ScopeResolution(
                PopulationScope.GT_TARGETS,
                (actor_ids[0],),
                source="gt_target_identity",
            ),
        },
    )
    scoped_ids, unresolved = _scope_actor_ids(bundle, scene)
    assert unresolved == set()
    stage1 = evaluate_stage1(
        bundle.graph,
        scene.inventory,
        scene.scene_bounds,
        eligible_actor_ids_by_entity=scoped_ids,
    )

    grounding = resolve_exact_identity_bindings(
        bundle,
        stage1,
        Evidence(),
        scene,
    )

    assert grounding.actor_ids_by_entity == {
        entity.id: ("stable:target",)
    }


def test_resolved_empty_additions_prove_missing_identity_mismatch():
    pump = _actor(
        "pump-gt",
        "Vintage_American_Gas___Oil_Pumps",
        "SM_Vintage_American_Gas___Oil_Pumps",
    )
    bundle = compile_repair_target_bundle(_scene(), _scene(pump))
    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    predicate = next(
        node for node in bundle.graph.nodes if isinstance(node, PredicateNode)
    )
    scene = SceneInventoryEvidence(
        inventory=ActorInventorySnapshot(()),
        scene_bounds=SCENE_BOUNDS,
        scopes={
            PopulationScope.CANDIDATE_ALL: ScopeResolution(
                PopulationScope.CANDIDATE_ALL,
                (),
                source="candidate_scene_snapshot",
            ),
            PopulationScope.ADDITIONS: ScopeResolution(
                PopulationScope.ADDITIONS,
                (),
                source="validated_before_after_diff",
            ),
            PopulationScope.GT_TARGETS: ScopeResolution(
                PopulationScope.GT_TARGETS,
                unknown_reason=UnknownReason.SCOPE_UNRESOLVED,
            ),
        },
    )
    scoped_ids, unresolved = _scope_actor_ids(bundle, scene)
    assert unresolved == set()
    stage1 = evaluate_stage1(
        bundle.graph,
        scene.inventory,
        scene.scene_bounds,
        eligible_actor_ids_by_entity=scoped_ids,
    )

    class Evidence:
        def candidate_actors(self):
            return ()

    grounding = resolve_exact_identity_bindings(
        bundle,
        stage1,
        Evidence(),
        scene,
    )
    identity = grounding.for_entity(entity.id)
    assert identity is not None
    assert identity.complete is False
    assert identity.search_exhausted is True

    atomic = {predicate.id: {"verdict": "UNKNOWN"}}
    _overlay_identity_grounding(bundle, atomic, grounding)
    assert atomic[predicate.id]["verdict"] == "MISMATCH"
    assert atomic[predicate.id]["check"]["score"] == 0.0


def test_image_count_scores_only_after_exact_actor_binding_stage4():
    gt_books = tuple(
        _actor(
            f"gt-book-{index}",
            f"BookStack{index}",
            "SM_Stack_of_Books",
            location=(index * 100.0, 0.0, 100.0),
        )
        for index in range(3)
    )
    candidate_books = tuple(
        _actor(
            f"candidate-book-{index}",
            f"CandidateBookStack{index}",
            "SM_Stack_of_Books",
            location=(0.0, index * 200.0, 100.0),
        )
        for index in range(3)
    )
    bundle = compile_repair_target_bundle(_scene(), _scene(*gt_books))

    class Evidence:
        def candidate_actors(self):
            return candidate_books

    descriptors = tuple(
        build_actor_descriptor(
            live_actor_id=f"stable:{actor['stable_actor_id']}",
            unreal_name=actor["name"],
            actor_class=actor["class"],
            asset_path=actor["asset_path"],
            actor_label=actor["label"],
            bounds=ActorBounds(
                tuple(actor["bounds"]["origin_cm"]),
                tuple(actor["bounds"]["extent_cm"]),
            ),
            active=True,
            renderable=True,
            in_current_level=True,
        )
        for actor in candidate_books
    )
    actor_ids = tuple(value.live_actor_id for value in descriptors)
    scene = SceneInventoryEvidence(
        inventory=ActorInventorySnapshot(descriptors),
        scene_bounds=SCENE_BOUNDS,
        scopes={
            PopulationScope.CANDIDATE_ALL: ScopeResolution(
                PopulationScope.CANDIDATE_ALL,
                actor_ids,
                source="candidate_scene_snapshot",
            ),
            PopulationScope.ADDITIONS: ScopeResolution(
                PopulationScope.ADDITIONS,
                actor_ids,
                source="validated_before_after_diff",
            ),
            PopulationScope.GT_TARGETS: ScopeResolution(
                PopulationScope.GT_TARGETS,
                unknown_reason=UnknownReason.SCOPE_UNRESOLVED,
            ),
        },
    )
    scoped_ids, unresolved = _scope_actor_ids(bundle, scene)
    assert unresolved == set()
    stage1 = evaluate_stage1(
        bundle.graph,
        scene.inventory,
        scene.scene_bounds,
        eligible_actor_ids_by_entity=scoped_ids,
    )

    unbound = evaluate_deterministic_rules(
        bundle,
        Evidence(),
        scene,
        require_actor_bindings=True,
        resolution_stage="stage4_deterministic",
    )
    assert unbound[0].verdict.value == "UNKNOWN"
    assert "requires ActorBindings" in unbound[0].rationale

    grounding = resolve_exact_identity_bindings(
        bundle,
        stage1,
        Evidence(),
        scene,
    )
    assert grounding.actor_ids_by_entity == {
        next(
            node.id
            for node in bundle.graph.nodes
            if isinstance(node, EntityNode)
        ): actor_ids
    }
    measured = evaluate_deterministic_rules(
        bundle,
        Evidence(),
        scene,
        bound_actor_ids_by_entity=grounding.actor_ids_by_entity,
        require_actor_bindings=True,
        resolution_stage="stage4_deterministic",
    )
    assert measured[0].verdict.value == "MATCH"
    assert measured[0].check["score"] == pytest.approx(1.0)
    assert measured[0].to_dict()["resolved_by"] == "stage4_deterministic"
    stage4_record = measured[0].to_dict()
    atomic = {measured[0].node_id: copy.deepcopy(stage4_record)}
    _overlay_identity_grounding(bundle, atomic, grounding)
    assert atomic[measured[0].node_id] == stage4_record


def test_repair_replacement_addition_is_in_the_visual_search_scope():
    source = _actor("door", "GarageDoor", "SM_GarageDoor")
    desired = copy.deepcopy(source)
    desired["transform"]["location_cm"] = [200.0, 0.0, 100.0]
    bundle = compile_repair_target_bundle(
        _scene(source),
        _scene(desired),
    )
    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    replacement = _descriptor("candidate-replacement", "SM_Industrial_Twin_Doors")
    scene = SceneInventoryEvidence(
        inventory=ActorInventorySnapshot((replacement,)),
        scene_bounds=SCENE_BOUNDS,
        scopes={
            PopulationScope.CANDIDATE_ALL: ScopeResolution(
                PopulationScope.CANDIDATE_ALL,
                (replacement.live_actor_id,),
                source="candidate_scene_snapshot",
            ),
            PopulationScope.ADDITIONS: ScopeResolution(
                PopulationScope.ADDITIONS,
                (replacement.live_actor_id,),
                source="validated_before_after_diff",
            ),
            PopulationScope.GT_TARGETS: ScopeResolution(
                PopulationScope.GT_TARGETS,
                unknown_reason=UnknownReason.SCOPE_UNRESOLVED,
            ),
        },
    )

    scoped, unresolved = _scope_actor_ids(bundle, scene)

    assert unresolved == set()
    assert scoped[entity.id] == (replacement.live_actor_id,)


def test_move_ignores_non_visual_editor_label_change():
    source = _actor("door", "GarageDoor", "SM_GarageDoor")
    desired = copy.deepcopy(source)
    desired["transform"]["location_cm"] = [200.0, 0.0, 100.0]
    desired["label"] = "RestoredGarageDoor"

    targets = derive_repair_targets(_scene(source), _scene(desired))

    assert len(targets) == 1
    assert targets[0].operation == "repair"
    assert targets[0].changed_fields == ("transform.location",)


def test_non_visual_editor_metadata_does_not_create_semantic_target():
    source = _actor("door", "GarageDoor_Input", "SM_GarageDoor")
    desired = copy.deepcopy(source)
    desired["label"] = "GarageDoor_GT"
    desired["actor_origin"] = "canonical_authoring"
    desired["actor_role"] = "repair_target"
    desired["logical_object_id"] = "canonical-door"
    desired["actor_tags"] = ["canonical"]

    assert derive_repair_targets(_scene(source), _scene(desired)) == ()


def test_input_only_capture_camera_does_not_create_remove_requirement():
    pump = _actor(
        "pump-gt",
        "Vintage_American_Gas___Oil_Pumps",
        "SM_Vintage_American_Gas___Oil_Pumps",
    )
    camera = _actor(
        "capture-camera",
        "_SceneBenchOutdoorArcCamera_spatial_context",
        "MatineeCam_SM",
    )
    camera["class"] = "/Script/Engine.CameraActor"
    camera["asset_path"] = "/Engine/EditorMeshes/MatineeCam_SM.MatineeCam_SM"

    targets = derive_repair_targets(_scene(camera), _scene(pump))

    assert len(targets) == 1
    assert targets[0].operation == "add"
    assert targets[0].name == "Vintage American Gas Oil Pumps"


def test_rotation_repair_claim_uses_visible_yaw_instruction_not_asset_path():
    source = _actor("chair", "SM_Chair2", "SM_Chair")
    source["transform"]["rotation_deg"] = [0.0, -45.0, 0.0]
    desired = copy.deepcopy(source)
    desired["transform"]["rotation_deg"] = [0.0, -135.0, 0.0]
    bundle = compile_repair_target_bundle(
        _scene(source),
        _scene(desired),
    )

    target = bundle.provenance["repair_targets"][0]
    assert target["changed_fields"] == ["transform.rotation"]
    assert target["structured_expectation"]["transform"]["rotation_deg"] == [
        0.0,
        -135.0,
        0.0,
    ]
    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    predicate = next(
        node for node in bundle.graph.nodes if isinstance(node, PredicateNode)
    )
    assert predicate.predicate_type is PredicateType.ATTRIBUTE
    assert predicate.name == "orientation"
    assert entity.text == "visibly restore the Chair's orientation"
    assert "/Game/" not in entity.text

    class Evidence:
        def __init__(self, candidate):
            self.candidate = candidate

        def candidate_actors(self):
            return (self.candidate,)

    subject_entity_id = bundle.graph.arguments_for(predicate.id)[0].target_id
    binding = {subject_entity_id: ("stable:chair",)}
    positive = evaluate_repair_target_fields(
        bundle,
        Evidence(desired),
        bound_actor_ids_by_entity=binding,
    )
    negative = evaluate_repair_target_fields(
        bundle,
        Evidence(source),
        bound_actor_ids_by_entity=binding,
    )
    replacement = evaluate_repair_target_fields(
        bundle,
        Evidence(_actor("replacement", "SM_Chair2", "SM_Chair")),
    )

    assert positive[0].verdict.value == "MATCH"
    assert positive[0].check["score"] == 1.0
    assert negative[0].verdict.value == "MISMATCH"
    assert negative[0].check["score"] == 0.0
    assert negative[0].check["mismatched_fields"] == ["transform.rotation"]
    assert replacement[0].verdict.value == "UNKNOWN"
    assert replacement[0].unknown_reason is UnknownReason.VISUAL_EVIDENCE_INCOMPLETE


def test_material_repair_claim_uses_finish_and_slot_not_asset_path():
    source = _actor("chair", "SM_Chair2", "SM_Chair")
    source["material_paths"] = ["/Game/Test/M_Bad.M_Bad"]
    desired = copy.deepcopy(source)
    desired["material_paths"] = ["/Game/Test/M_Chair.M_Chair"]
    bundle = compile_repair_target_bundle(
        _scene(source),
        _scene(desired),
    )

    target = bundle.provenance["repair_targets"][0]
    assert target["changed_fields"] == ["property.material_paths"]
    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    predicate = next(
        node for node in bundle.graph.nodes if isinstance(node, PredicateNode)
    )
    assert predicate.predicate_type is PredicateType.MATERIAL
    assert predicate.name == "material"
    assert entity.text == "visibly restore the Chair's material appearance"
    assert "/Game/" not in entity.text


def test_transient_water_mids_do_not_expand_repair_target_camera_scope():
    water_input = _actor("water", "WaterBodyCustom6", "SM_Water_Volume_01")
    water_input["component_material_slots"] = [
        {
            "component_identity": (
                "CustomMeshComponent|/Script/Engine.StaticMeshComponent"
            ),
            "is_dynamic": True,
            "material_class_path": "/Script/Engine.MaterialInstanceDynamic",
            "material_path": "/Engine/Transient.WaterMID_277",
            "slot_index": 0,
        }
    ]
    water_input["material_paths"] = ["/Engine/Transient.WaterMID_277"]
    water_gt = copy.deepcopy(water_input)
    water_gt["component_material_slots"][0]["material_path"] = (
        "/Engine/Transient.WaterMID_293"
    )
    water_gt["material_paths"] = ["/Engine/Transient.WaterMID_293"]
    umbrella_input = _actor(
        "umbrella", "SM_Umbrella_01", "SM_Umbrella_01"
    )
    umbrella_gt = copy.deepcopy(umbrella_input)
    umbrella_gt["transform"]["location_cm"] = [300.0, 100.0, 100.0]
    umbrella_gt["transform"]["rotation_deg"] = [0.0, 0.0, 45.0]
    umbrella_gt["transform"]["scale"] = [1.2, 1.2, 1.2]

    targets = derive_repair_targets(
        _scene(water_input, umbrella_input),
        _scene(water_gt, umbrella_gt),
    )

    assert len(targets) == 1
    assert targets[0].desired_actor["stable_actor_id"] == "umbrella"
    assert targets[0].changed_fields == (
        "transform.location",
        "transform.rotation",
        "transform.scale",
    )


def test_image_addition_claim_is_identity_only_and_excludes_asset_metadata():
    books = _actor("books", "SM_Stack_of_Books", "SM_Stack_of_Books")
    bundle = compile_repair_target_bundle(
        _scene(),
        _scene(books),
    )

    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    assert "designated restored Stack of Books" in entity.text
    assert "shelf" not in entity.text
    assert "/Game/" not in entity.text


def test_input_only_actor_becomes_a_negated_remove_target():
    obsolete = _actor("obsolete", "TemporaryBarrier", "SM_Barrier")

    bundle = compile_repair_target_bundle(
        _scene(obsolete),
        _scene(),
    )

    target = bundle.provenance["repair_targets"][0]
    assert target["operation"] == "remove"
    predicate = next(
        node for node in bundle.graph.nodes if isinstance(node, PredicateNode)
    )
    assert predicate.polarity is Polarity.NEGATED
    assert predicate.name == "exists"
    entity = next(
        node for node in bundle.graph.nodes if isinstance(node, EntityNode)
    )
    assert entity.evaluation_route is EntityEvaluationRoute.STAGE2_VISUAL


def test_identical_input_and_gt_refuse_an_empty_semantic_graph():
    actor = _actor("same", "Wall", "SM_Wall")

    with pytest.raises(ValueError, match="no Actor-level repair targets"):
        compile_repair_target_bundle(_scene(actor), _scene(actor))


def test_failed_scene_export_is_not_treated_as_an_empty_scene():
    failed = _scene()
    failed["export_metadata"] = {"status": "error", "error": "load failed"}

    with pytest.raises(ValueError, match="load failed"):
        derive_repair_targets(failed, _scene())
