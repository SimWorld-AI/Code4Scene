"""Physical placement the closing measurement's three scene rates cannot see."""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from types import SimpleNamespace

import pytest

from declared_cases import actor, case, check_by_id, context, scene

from code4scene.evaluation import contracts
from code4scene.evaluation.assertions import Check
from code4scene.evaluation.evaluation_policy import FrozenEvaluationPolicy
from code4scene.evaluation.scene_diff import (
    diff_scenes,
    edited_candidate_actors,
)
from code4scene.evaluation.scene_semantics import (
    PHYSICS_ROLE_ENVIRONMENT_PROXY,
    PHYSICS_ROLE_NON_SOLID,
    PHYSICS_ROLE_SCORED_SOLID,
    PHYSICS_ROLE_SUPPORT_SURFACE,
    classify_physics_roles,
    ground_gap_support_surface_evidence,
    is_ground_support_ground_gap_target,
    is_non_solid,
    is_non_solid_ground_gap_target,
)
from code4scene.evaluation.ue_evidence import (
    DEFAULT_PHYSICS_CHUNK_SIZE,
    PHYSICS_SCRIPT,
    EvidenceError,
    _case_measurement_targets,
    _merge_physics_chunks,
    _physics_chunk_size,
    _solid_broad_phase_tolerance,
    _validate_physics_chunk,
)
from code4scene.evaluation.verifiers import (environment_consistency,
                                             floating,
                                             physics_regression,
                                             solid_penetration)

CANDIDATE_ALL = {"scope": "candidate_all"}
LAKE = case([{"id": "not-underwater", "primitive": "environment_consistency",
              "target_selector": {"scope": "candidate_all", "labels": ["bench"]}}])


class _FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def _ue_physics_symbols(*names: str) -> dict:
    """Load selected pure/reversible helpers without importing Unreal."""
    required = set(names)
    required.update({
        "_PENETRATION_DEPTH_PATTERN",
        "_actor_path",
        "_component_path",
        "_collision_response_token",
        "_overlap_components",
        "_unit_probe_directions",
        "_hit_result_object",
        "_native_mtd_from_hit",
    })
    tree = ast.parse(PHYSICS_SCRIPT.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in required:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in required
            for target in node.targets
        ):
            body.append(node)
    namespace = {
        "re": re,
        "unreal": SimpleNamespace(
            Vector=_FakeVector,
            PrimitiveComponent=object,
        ),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(PHYSICS_SCRIPT),
                 "exec"), namespace)
    return namespace


def lake(surface_z: float = 0.0) -> dict:
    """A wide, thin mesh named like water — the shape water actually has."""
    return actor("BP_WaterBody_Lake_01", location=(0.0, 0.0, surface_z),
                 extent=(1000.0, 1000.0, 5.0))


def test_a_bench_on_the_lakebed_is_a_defect_nothing_else_catches(tmp_path):
    drowned = actor("bench", location=(0.0, 0.0, -300.0), extent=(50.0, 25.0, 40.0))
    report = environment_consistency.verify(context(
        tmp_path, candidate=scene([lake(), drowned]), case_spec=LAKE))
    assert report["status"] == contracts.MEASURED
    check = check_by_id(report, "physics.environment_consistency")
    assert check["observed"]["fully_submerged_actor_count"] == 1
    assert check["observed"]["detected_water_surface_count"] == 1


def test_a_bench_on_the_shore_is_fine(tmp_path):
    dry = actor("bench", location=(0.0, 0.0, 40.0), extent=(50.0, 25.0, 40.0))
    report = environment_consistency.verify(context(
        tmp_path, candidate=scene([lake(), dry]), case_spec=LAKE))
    assert report["status"] == contracts.MEASURED
    assert report["score"] == 1.0


def test_a_prop_merely_named_water_does_not_flood_the_level(tmp_path):
    """Water must also be big: a bottle is not a lake."""
    bottle = actor("SM_WaterBottle", location=(0.0, 0.0, 100.0),
                   extent=(5.0, 5.0, 15.0))
    below = actor("bench", location=(0.0, 0.0, -300.0), extent=(50.0, 25.0, 40.0))
    report = environment_consistency.verify(context(
        tmp_path, candidate=scene([bottle, below]), case_spec=LAKE))
    assert report["status"] == contracts.MEASURED
    assert check_by_id(report, "physics.environment_consistency")["observed"][
        "detected_water_surface_count"] == 0


def test_water_material_inside_a_thick_blueprint_does_not_make_it_a_lake(tmp_path):
    """Component paths have no component-local bounds in the scene export.

    Treating the parent Blueprint's box as the water component's box makes a
    house containing a water tank define a waterline at its roof.
    """
    house = actor("BP_House", location=(0.0, 0.0, 500.0),
                  extent=(1000.0, 1000.0, 500.0))
    house["component_asset_paths"] = ["/Game/Props/SM_WaterTank"]
    bench = actor("bench", location=(0.0, 0.0, 40.0),
                  extent=(50.0, 25.0, 40.0))
    report = environment_consistency.verify(context(
        tmp_path, candidate=scene([house, bench]), case_spec=LAKE))
    assert report["status"] == contracts.MEASURED
    assert check_by_id(report, "physics.environment_consistency")["observed"][
        "detected_water_surface_count"] == 0


def test_water_material_can_identify_a_thin_surface(tmp_path):
    surface = actor("BP_Plane", location=(0.0, 0.0, 0.0),
                    extent=(1000.0, 1000.0, 5.0))
    surface["material_paths"] = ["/Game/Materials/M_Water_Ocean"]
    drowned = actor("bench", location=(0.0, 0.0, -300.0),
                    extent=(50.0, 25.0, 40.0))
    report = environment_consistency.verify(context(
        tmp_path, candidate=scene([surface, drowned]), case_spec=LAKE))
    assert report["status"] == contracts.MEASURED
    assert check_by_id(report, "physics.environment_consistency")["observed"][
        "detected_water_surface_count"] == 1


def test_thin_water_surface_is_non_solid():
    assert is_non_solid_ground_gap_target(lake())


def test_thick_collidable_actor_with_water_place_name_remains_solid():
    river_door = actor(
        "BPP_LI_River_Door_011",
        actor_class=(
            "/Game/MiddleEasternTown/Blueprints/PackedBlueprints/"
            "BPP_LI_River_Door_01.BPP_LI_River_Door_01_C"
        ),
        asset_path=(
            "/Game/MiddleEasternTown/Blueprints/PackedBlueprints/"
            "BPP_LI_River_Door_01.BPP_LI_River_Door_01_C"
        ),
        extent=(958.0, 784.0, 1281.0),
        collision={"collision_enabled": True},
    )

    # The shared V3 classifier remains frozen, but the I2S-only floating
    # population no longer mistakes a place-name token for a water surface.
    assert is_non_solid(river_door)
    assert not is_non_solid_ground_gap_target(river_door)


def test_thick_collidable_pool_mesh_remains_solid():
    pool_shell = actor(
        "SM_Pool_01",
        asset_path="/Game/Modular_house/Meshes/Outdoor/SM_Pool_01.SM_Pool_01",
        extent=(77.0, 400.0, 200.0),
        collision={"collision_enabled": True},
    )

    assert is_non_solid(pool_shell)
    assert not is_non_solid_ground_gap_target(pool_shell)


def test_i2s_ground_gap_excludes_actor_owned_broad_thin_support_surfaces():
    arena_ground = actor(
        "Arena_Env_Ground",
        asset_path="/Engine/BasicShapes/Plane.Plane",
        extent=(2634.0, 3063.0, 0.0),
    )
    sidewalk = actor(
        "SM_Sidewalk_21",
        asset_path=(
            "/Game/Modular_house/Meshes/Sidewalk/"
            "SM_Sidewalk_01.SM_Sidewalk_01"
        ),
        extent=(100.0001, 154.2613, 4.9999),
    )

    assert is_ground_support_ground_gap_target(arena_ground)
    assert is_ground_support_ground_gap_target(sidewalk)
    evidence = ground_gap_support_surface_evidence(sidewalk)
    assert evidence is not None
    assert evidence["semantic_evidence_source"] == "actor_identity"
    assert evidence["matched_support_words"] == ["sidewalk"]


def test_i2s_ground_gap_ignores_containing_package_names():
    taxi = actor(
        "Taxi_01_veiws7k92",
        actor_class=(
            "/Game/Street_NY/Models/Blueprint/"
            "BP_Sedan_Taxi_Blueprint.BP_Sedan_Taxi_Blueprint_C"
        ),
        asset_path=(
            "/Game/Street_NY/Models/Blueprint/"
            "BP_Sedan_Taxi_Blueprint.BP_Sedan_Taxi_Blueprint_C"
        ),
        extent=(268.28, 109.47, 82.56),
    )

    assert not is_ground_support_ground_gap_target(taxi)


def test_i2s_ground_gap_keeps_thick_cliffs_and_repair_slabs():
    cliff = actor(
        "Cliff_0",
        asset_path=(
            "/Game/Flying_Fantasy_Town/SkyTown/Meshes/"
            "SM_FlatCliff01_Mirrored.SM_FlatCliff01_Mirrored"
        ),
        extent=(1207.0, 1392.0, 1394.0),
    )
    repair_slab = actor(
        "Repair_Base_Slab",
        asset_path=(
            "/Game/RuralAustralia/StaticMeshes/Rocks/Cliff_L_01/"
            "SM_Cliff_L_01_A.SM_Cliff_L_01_A"
        ),
        extent=(3083.0, 7470.0, 1174.0),
    )

    assert not is_ground_support_ground_gap_target(cliff)
    assert not is_ground_support_ground_gap_target(repair_slab)


def test_i2s_ground_material_only_classifies_a_generic_basic_shape():
    plane = actor(
        "StaticMeshActor_1",
        asset_path="/Engine/BasicShapes/Plane.Plane",
        extent=(500.0, 500.0, 1.0),
        material_paths=["/Game/Materials/M_Cobblestone.M_Cobblestone"],
    )
    table = actor(
        "PatioTable",
        asset_path="/Game/Props/SM_PatioTable.SM_PatioTable",
        extent=(500.0, 500.0, 1.0),
        material_paths=["/Game/Materials/M_Cobblestone.M_Cobblestone"],
    )

    assert is_ground_support_ground_gap_target(plane)
    assert not is_ground_support_ground_gap_target(table)


def test_the_water_token_needs_a_word_boundary(tmp_path):
    """A known limit of the ported classifier, pinned rather than assumed.

    `BP_WaterBody_Lake` is detected; `WaterBodyLake` — the same Actor named
    without separators — is not, because the token must stand as a word. UE
    projects use both spellings. Left as ported: the token list was tuned
    against real imported levels and widening it here would silently reclassify
    every scene those numbers came from. It belongs in a `not_evaluated` path
    or a level-naming convention, not in a quiet regex edit.
    """
    unseparated = actor("WaterBodyLake_01", location=(0.0, 0.0, 0.0),
                        extent=(1000.0, 1000.0, 5.0))
    drowned = actor("bench", location=(0.0, 0.0, -300.0), extent=(50.0, 25.0, 40.0))
    report = environment_consistency.verify(context(
        tmp_path, candidate=scene([unseparated, drowned]), case_spec=LAKE))
    assert check_by_id(report, "physics.environment_consistency")["observed"][
        "detected_water_surface_count"] == 0


def test_an_actor_without_bounds_withholds_the_whole_check(tmp_path):
    boundless = {k: v for k, v in actor("bench").items() if k != "bounds"}
    report = environment_consistency.verify(context(
        tmp_path, candidate=scene([lake(), boundless]), case_spec=LAKE))
    assert report["status"] == "not_evaluated"
    assert "carry no bounds" in report["failure_reason"]


PENETRATION = case([{"id": "no-clipping", "primitive": "solid_penetration",
                     "target_selector": {"scope": "candidate_all",
                                         "labels": ["lamp"]}}])
PENETRATION_ALL = case([{
    "id": "no-clipping-anywhere",
    "primitive": "solid_penetration",
    "target_selector": {"scope": "candidate_all"},
}])
PENETRATION_TIGHT = case([{
    "id": "no-clipping-over-one-centimetre",
    "primitive": "solid_penetration",
    "target_selector": {"scope": "candidate_all", "labels": ["lamp"]},
    "maximum_penetration_cm": 1.0,
}])
ADAPTIVE_PENETRATION = case([{
    "id": "actor-scale-collision-tolerance",
    "primitive": "solid_penetration",
    "target_selector": {"scope": "candidate_all"},
    "maximum_penetration_cm": 5.0,
    "adaptive_penetration_tolerance": True,
    "relative_penetration_tolerance_fraction": 0.05,
    "maximum_adaptive_penetration_tolerance_cm": 50.0,
}])


def measurements(**records) -> dict:
    return {"actors": dict(records)}


def hospital_proxy_scene() -> tuple[dict, dict, dict, dict]:
    hill = actor(
        "Hill_00",
        location=(-6000.0, 9000.0, -200.0),
        extent=(200000.0, 150000.0, 125000.0),
        asset_path="/Engine/BasicShapes/Cube.Cube",
        scale=(4000.0, 3000.0, 2500.0),
    )
    castle = actor(
        "Large_Castle",
        location=(5000.0, 5000.0, 1000.0),
        extent=(200000.0, 150000.0, 125000.0),
        asset_path="/Game/Architecture/SM_Large_Castle.SM_Large_Castle",
    )
    wing = actor(
        "Hospital_Wing_Left",
        location=(-2200.0, 0.0, 500.0),
        extent=(827.0, 582.0, 500.0),
    )
    bench = actor("bench", location=(0.0, 0.0, 50.0))
    crate = actor("crate", location=(300.0, 0.0, 50.0))
    floor = actor(
        "floor", location=(0.0, 0.0, -5.0), extent=(5000.0, 5000.0, 5.0)
    )
    fog = actor(
        "Arena_Env_Fog",
        actor_class="/Script/Engine.ExponentialHeightFog",
    )
    return scene([hill, castle, wing, bench, crate, floor, fog]), hill, wing, bench


def test_physics_roles_separate_proxy_support_non_solid_and_large_building():
    candidate, hill, _wing, _bench = hospital_proxy_scene()
    roles = classify_physics_roles(candidate["actors"])

    by_label = {
        value["label"]: roles[value["actor_path"]]["role"]
        for value in candidate["actors"]
    }
    assert by_label[hill["label"]] == PHYSICS_ROLE_ENVIRONMENT_PROXY
    assert by_label["Large_Castle"] == PHYSICS_ROLE_SCORED_SOLID
    assert by_label["floor"] == PHYSICS_ROLE_SUPPORT_SURFACE
    assert by_label["Arena_Env_Fog"] == PHYSICS_ROLE_NON_SOLID
    reasons = roles[hill["actor_path"]]["reasons"]
    assert "generic_engine_basic_shape" in reasons
    assert "contains_many_independent_scene_actors" in reasons


def test_environment_proxy_mtd_is_one_actor_level_containment_failure(tmp_path):
    candidate, _hill, wing, bench = hospital_proxy_scene()
    contract = case([{
        "id": "hospital-proxy",
        "primitive": "solid_penetration",
        "target_selector": {
            "scope": "candidate_all",
            "labels": [wing["label"], bench["label"]],
        },
        "maximum_penetration_cm": 5.0,
    }])
    report = solid_penetration.verify(context(
        tmp_path,
        candidate=candidate,
        case_spec=contract,
        spec={"candidate_measurements": _write(
            tmp_path,
            "hospital-proxy-contacts.json",
            measurements(
                Hospital_Wing_Left={
                    "solid_penetration_evaluated": True,
                    "solid_penetrations": [{
                        "collider": "Hill_00",
                        "confirmed_overlap": True,
                        "penetration_depth_cm": 124491.492188,
                        "depth_method": "ue_fhitresult_initial_overlap_mtd",
                    }],
                },
                bench={
                    "solid_penetration_evaluated": True,
                    "solid_penetrations": [],
                },
            ),
        )},
    ))

    assert report["status"] == contracts.MEASURED
    assert report["score"] == 0.5
    check = check_by_id(report, "physics.solid_penetration")
    assert check["observed"]["environment_containment_actor_count"] == 1
    assert check["observed"]["ordinary_penetration_actor_count"] == 0
    assert check["observed"]["actor_failure_penalty_sum"] == 1.0
    assert check["observed"]["non_penetration_rate"] == 0.5
    assert check["observed"]["maximum_observed_penetration_cm"] == \
        pytest.approx(124491.492188)
    evidence = check["evidence"][0]
    assert evidence["actor_failure_penalty"] == 1.0
    assert evidence["verdict_basis"] == [
        "ue_collision_body_native_mtd",
        "environment_proxy_containment",
    ]
    assert evidence["environment_containment_failures"][0][
        "collider_physics_role"
    ] == PHYSICS_ROLE_ENVIRONMENT_PROXY


def test_fog_contact_is_filtered_and_never_becomes_containment(tmp_path):
    fog = actor(
        "Arena_Env_Fog",
        actor_class="/Script/Engine.ExponentialHeightFog",
    )
    wing = actor("Hospital_Wing_Left")
    contract = case([{
        "id": "fog-is-not-solid",
        "primitive": "solid_penetration",
        "target_selector": {
            "scope": "candidate_all", "labels": [wing["label"]]
        },
        "maximum_penetration_cm": 5.0,
    }])
    report = solid_penetration.verify(context(
        tmp_path,
        candidate=scene([wing, fog]),
        case_spec=contract,
        spec={"candidate_measurements": _write(
            tmp_path,
            "fog-contact.json",
            measurements(Hospital_Wing_Left={
                "solid_penetration_evaluated": True,
                "solid_penetrations": [{
                    "collider": "Arena_Env_Fog",
                    "confirmed_overlap": True,
                    "penetration_depth_cm": 999999.0,
                    "depth_method": "ue_fhitresult_initial_overlap_mtd",
                }],
            }),
        )},
    ))

    assert report["status"] == contracts.MEASURED
    assert report["score"] == 1.0
    check = check_by_id(report, "physics.solid_penetration")
    assert check["observed"]["environment_containment_actor_count"] == 0
    assert check["observed"]["penetrating_actor_count"] == 0
    assert check["observed"]["filtered_contact_count"] == 1


def test_ue_probe_keys_measurements_by_strong_actor_identity():
    """Duplicate display labels must not overwrite separate UE records."""
    source = PHYSICS_SCRIPT.read_text()
    path = source.index("_actor_path(actor)", source.index("measurements = {}"))
    stable = source.index('reference.get("stable_actor_id")', path)
    label = source.index('reference.get("label")', stable)
    assert path < stable < label


def test_ue_probe_does_not_filter_other_measurement_targets_from_collision():
    """Target-vs-target overlap is still a real UE collision to confirm."""
    source = PHYSICS_SCRIPT.read_text()
    broad_phase = source[source.index("def _solid_broad_phase_candidates"):
                         source.index("def _measure_solid_overlaps")]
    assert "if collider_path in ignored_actor_paths" not in broad_phase
    assert "if collider is actor or collider_path == actor_path" in broad_phase


def test_solid_only_collection_does_not_measure_ground_only_targets():
    ground = actor("ground-only")
    solid = actor("solid-only")
    contract = case([
        {"id": "ground", "primitive": "physics",
         "target_selector": {"scope": "candidate_all", "labels": ["ground-only"]}},
        {"id": "solid", "primitive": "solid_penetration",
         "target_selector": {"scope": "candidate_all", "labels": ["solid-only"]}},
    ])

    targets = _case_measurement_targets(
        contract, scene([ground, solid]), {"solid_penetration"}
    )

    assert targets == [{"actor_path": solid["actor_path"], "label": "solid-only"}]


def test_edited_actor_scope_selects_only_candidate_side_local_changes():
    unchanged = actor("unchanged")
    moved_before = actor("moved", location=(0.0, 0.0, 0.0))
    moved_after = actor("moved", location=(50.0, 0.0, 0.0))
    modified_before = actor("modified", asset_path="/Game/A.A")
    modified_after = actor("modified", asset_path="/Game/B.B")
    removed = actor("removed")
    added = actor("added")
    input_scene = scene([unchanged, moved_before, modified_before, removed])
    candidate = scene([unchanged, moved_after, modified_after, added])

    changed = edited_candidate_actors(diff_scenes(input_scene, candidate))

    assert [value["label"] for value in changed] == [
        "added",
        "moved",
        "modified",
    ]


def test_edited_actor_scope_limits_ue_targets_but_keeps_full_world_as_colliders():
    unchanged = actor("unchanged")
    moved_before = actor("moved", location=(0.0, 0.0, 0.0))
    moved_after = actor("moved", location=(50.0, 0.0, 0.0))
    added = actor("added")
    input_scene = scene([unchanged, moved_before])
    candidate = scene([unchanged, moved_after, added])
    contract = case([
        {
            "id": "local-ground",
            "primitive": "physics",
            "scope": "edited_actors",
            "target_selector": {"scope": "edited_actors"},
            "maximum_ground_gap_cm": 5.0,
            "maximum_penetration_cm": 5.0,
            "minimum_support_fraction": 0.05,
        },
        {
            "id": "local-solid",
            "primitive": "solid_penetration",
            "scope": "edited_actors",
            "target_selector": {"scope": "edited_actors"},
            "maximum_penetration_cm": 5.0,
        },
    ])

    targets = _case_measurement_targets(contract, candidate, input_scene=input_scene)

    assert targets == [
        {"actor_path": added["actor_path"], "label": "added"},
        {"actor_path": moved_after["actor_path"], "label": "moved"},
    ]
    # The target list is local, while the editor probe still builds its
    # collision cache from all_actors and therefore sees unchanged colliders.
    source = PHYSICS_SCRIPT.read_text()
    assert "measurements[key] = _measure_actor(" in source
    assert "measurement_target_paths" in source


def test_edited_actor_scope_refuses_to_run_without_input_scene():
    contract = case([{
        "id": "local-solid",
        "primitive": "solid_penetration",
        "scope": "edited_actors",
        "target_selector": {"scope": "edited_actors"},
        "maximum_penetration_cm": 5.0,
    }])

    with pytest.raises(EvidenceError, match="needs an Input scene"):
        _case_measurement_targets(contract, scene([actor("added")]))


def test_local_floating_uses_only_edited_actor_measurements(tmp_path):
    unchanged = actor("unchanged")
    moved_before = actor("moved", location=(0.0, 0.0, 0.0))
    moved_after = actor("moved", location=(50.0, 0.0, 0.0))
    added = actor("added")
    light_before = actor(
        "DirectionalLight",
        actor_class="/Script/Engine.DirectionalLight",
    )
    light_after = actor(
        "DirectionalLight",
        actor_class="/Script/Engine.DirectionalLight",
        location=(50.0, 0.0, 0.0),
    )
    fog_before = actor(
        "ExponentialHeightFog",
        actor_class="/Script/Engine.ExponentialHeightFog",
    )
    fog_after = actor(
        "ExponentialHeightFog",
        actor_class="/Script/Engine.ExponentialHeightFog",
        location=(0.0, 50.0, 0.0),
    )
    ground_before = actor(
        "Arena_Env_Ground",
        location=(0.0, 0.0, 0.0),
        extent=(2634.0, 3063.0, 0.0),
        asset_path="/Engine/BasicShapes/Plane.Plane",
    )
    ground_after = actor(
        "Arena_Env_Ground",
        location=(50.0, 0.0, 0.0),
        extent=(2634.0, 3063.0, 0.0),
        asset_path="/Engine/BasicShapes/Plane.Plane",
    )
    input_scene = scene([
        unchanged, moved_before, light_before, fog_before, ground_before,
    ])
    candidate = scene(
        [unchanged, moved_after, added, light_after, fog_after, ground_after]
    )
    profile = {
        **FrozenEvaluationPolicy(
            policy_id="base",
        ).physics_profile,
        "profile_id": "local-image-physics",
        "selector": {"scope": "edited_actors"},
    }
    policy = FrozenEvaluationPolicy(
        policy_id="local-image-policy",
        source_snapshot=input_scene,
        physics_profile=profile,
    )
    contract = policy.physics_contract("declared-case-test")
    configured = tmp_path / "physics.json"
    configured.write_text(json.dumps({"actors": {
        moved_after["actor_path"]: {
            "actor_path": moved_after["actor_path"],
            "actor_label": "moved",
            "grounded": False,
            "surface_detected": False,
            "ground_gap_cm": None,
            "measurement_method": "ue_vertical_collision_trace_3x3",
        },
        added["actor_path"]: {
            "actor_path": added["actor_path"],
            "actor_label": "added",
            "grounded": True,
            "surface_detected": True,
            "ground_gap_cm": 0.0,
            "measurement_method": "ue_vertical_collision_trace_3x3",
        },
    }}))
    ctx = context(
        tmp_path,
        candidate=candidate,
        input_scene=input_scene,
        case_spec=contract,
        spec={"candidate_measurements": str(configured)},
    )
    # A bad whole-scene rate must not leak into local repair Physics.
    ctx.record["metrics"] = {"actors": 1000, "floating_rate": 0.99}

    report = floating.verify(ctx)

    assert report["status"] == contracts.MEASURED
    assert report["score"] == pytest.approx(0.5)
    assert report["metadata"]["score_direction"] == "lower_is_better"
    assert report["evidence"]["population_scope"] == "edited_actors"
    # UE collection still records all five edited Actors. The local floating
    # scorer excludes two non-solid controls and the support surface.
    assert report["evidence"]["measurement_target_count"] == 5
    assert report["metrics"][
        "selected_actor_count_before_applicability_filter"
    ] == 5
    assert report["metrics"]["scored_actor_count_after_applicability_filter"] == 2
    assert report["metrics"]["scored_actor_count_after_non_solid_filter"] == 2
    assert report["metrics"]["filtered_non_solid_actor_count"] == 2
    assert report["metrics"]["filtered_support_surface_actor_count"] == 1
    assert report["evidence"]["applicability_filter"] == (
        "scene_semantics.i2s_local_ground_gap_applicability_v3"
    )
    assert report["evidence"]["non_solid_filter"] == (
        "scene_semantics.is_non_solid_ground_gap_target"
    )
    assert report["evidence"]["support_surface_filter"] == (
        "scene_semantics.is_ground_support_ground_gap_target"
    )
    assert {
        value["actor_label"]
        for value in report["evidence"]["filtered_non_solid_actors"]
    } == {"DirectionalLight", "ExponentialHeightFog"}
    assert [
        value["actor_label"]
        for value in report["evidence"]["filtered_support_surface_actors"]
    ] == ["Arena_Env_Ground"]
    assert {value["observed"].get("actor_label") for value in
            report["metrics"]["checks"]} == {"moved", "added"}


def test_local_floating_v3_uses_supported_and_preserves_ground_evidence(tmp_path):
    before = actor("picture", location=(0.0, 0.0, 100.0))
    after = actor("picture", location=(25.0, 0.0, 100.0))
    input_scene = scene([before])
    candidate = scene([after])
    profile = {
        **FrozenEvaluationPolicy(policy_id="base").physics_profile,
        "profile_id": "image-to-scene-edited-actors-physical-safety-v2",
        "selector": {"scope": "edited_actors"},
        "support_model": "ground_or_lateral_v1",
    }
    policy = FrozenEvaluationPolicy(
        policy_id="image-local-v3",
        source_snapshot=input_scene,
        physics_profile=profile,
    )
    configured = tmp_path / "physics-v3.json"
    configured.write_text(json.dumps({"actors": {
        after["actor_path"]: {
            "actor_path": after["actor_path"],
            "actor_label": "picture",
            "grounded": False,
            "supported": True,
            "support_mode": "lateral",
            "surface_detected": True,
            "ground_gap_cm": 120.0,
            "lateral_support_detected": True,
            "lateral_support_distance_cm": 1.5,
            "lateral_support_fraction": 0.8,
            "lateral_trace_count": 50,
            "lateral_supporting_colliders": [{"collider_label": "Wall"}],
            "measurement_method": "ue_lod0_vertex_terrain_trace",
        },
    }}))

    report = floating.verify(context(
        tmp_path,
        candidate=candidate,
        input_scene=input_scene,
        case_spec=policy.physics_contract("declared-case-test"),
        spec={"candidate_measurements": str(configured)},
    ))

    assert report["status"] == contracts.MEASURED
    assert report["score"] == 0.0
    check = check_by_id(
        report, f"physics.local_edit_floating:{after['actor_path']}"
    )
    assert check["status"] == contracts.MEASURED
    assert check["observed"]["grounded"] is False
    assert check["observed"]["supported"] is True
    assert check["observed"]["support_mode"] == "lateral"
    assert check["observed"]["lateral_trace_count"] == 50


def test_local_floating_v3_refuses_legacy_ground_only_evidence(tmp_path):
    before = actor("picture", location=(0.0, 0.0, 100.0))
    after = actor("picture", location=(25.0, 0.0, 100.0))
    input_scene = scene([before])
    candidate = scene([after])
    profile = {
        **FrozenEvaluationPolicy(policy_id="base").physics_profile,
        "profile_id": "image-to-scene-edited-actors-physical-safety-v2",
        "selector": {"scope": "edited_actors"},
        "support_model": "ground_or_lateral_v1",
    }
    policy = FrozenEvaluationPolicy(
        policy_id="image-local-v3",
        source_snapshot=input_scene,
        physics_profile=profile,
    )
    configured = tmp_path / "legacy-physics.json"
    configured.write_text(json.dumps({"actors": {
        after["actor_path"]: {
            "actor_path": after["actor_path"],
            "actor_label": "picture",
            "grounded": True,
            "surface_detected": True,
            "ground_gap_cm": 0.0,
            "measurement_method": "ue_vertical_collision_trace_3x3",
        },
    }}))

    report = floating.verify(context(
        tmp_path,
        candidate=candidate,
        input_scene=input_scene,
        case_spec=policy.physics_contract("declared-case-test"),
        spec={"candidate_measurements": str(configured)},
    ))

    assert report["status"] == "not_evaluated"
    assert report["score"] is None
    check = check_by_id(
        report, f"physics.local_edit_floating:{after['actor_path']}"
    )
    assert "no boolean supported verdict" in check["failure_reason"]


def test_local_floating_is_neutral_when_every_edit_is_non_solid(tmp_path):
    light_before = actor(
        "DirectionalLight",
        actor_class="/Script/Engine.DirectionalLight",
    )
    light_after = actor(
        "DirectionalLight",
        actor_class="/Script/Engine.DirectionalLight",
        location=(50.0, 0.0, 0.0),
    )
    input_scene = scene([light_before])
    candidate = scene([light_after])
    profile = {
        **FrozenEvaluationPolicy(policy_id="base").physics_profile,
        "profile_id": "local-image-physics",
        "selector": {"scope": "edited_actors"},
    }
    policy = FrozenEvaluationPolicy(
        policy_id="local-image-policy",
        source_snapshot=input_scene,
        physics_profile=profile,
    )
    configured = tmp_path / "physics.json"
    configured.write_text(json.dumps({"actors": {}}))
    report = floating.verify(context(
        tmp_path,
        candidate=candidate,
        input_scene=input_scene,
        case_spec=policy.physics_contract("declared-case-test"),
        spec={"candidate_measurements": str(configured)},
    ))

    assert report["status"] == contracts.MEASURED
    assert report["score"] == 0.0
    assert report["metrics"][
        "selected_actor_count_before_non_solid_filter"
    ] == 1
    assert report["metrics"]["scored_actor_count_after_non_solid_filter"] == 0
    assert report["metrics"]["filtered_non_solid_actor_count"] == 1
    check = check_by_id(report, "physics.local_edit_floating")
    assert check["observed"]["empty_scope_policy"] == (
        "neutral_no_introduced_physics"
    )


def test_candidate_all_floating_keeps_the_frozen_scene_rate_path(tmp_path):
    policy = FrozenEvaluationPolicy(policy_id="text-to-scene-policy")
    report_context = context(
        tmp_path,
        candidate=scene([
            actor(
                "DirectionalLight",
                actor_class="/Script/Engine.DirectionalLight",
            )
        ]),
        case_spec=policy.physics_contract("declared-case-test"),
    )
    report_context.record["metrics"] = {
        "actors": 121,
        "floating_rate": 0.0165,
    }

    report = floating.verify(report_context)

    assert report["status"] == contracts.MEASURED
    assert report["score"] == pytest.approx(0.0165)
    assert report["evidence"]["metric_id"] == "measure_v1"
    assert "filtered_non_solid_actor_count" not in report["metrics"]


def test_local_solid_penetration_requires_measurements_only_for_edited_actors(
    tmp_path,
):
    unchanged = actor("unchanged")
    moved_before = actor("moved", location=(0.0, 0.0, 0.0))
    moved_after = actor("moved", location=(50.0, 0.0, 0.0))
    added = actor("added")
    input_scene = scene([unchanged, moved_before])
    candidate = scene([unchanged, moved_after, added])
    contract = case([{
        "id": "local-solid",
        "primitive": "solid_penetration",
        "scope": "edited_actors",
        "target_selector": {"scope": "edited_actors"},
        "maximum_penetrating_actor_count": 0,
        "maximum_penetration_cm": 5.0,
    }])
    evidence = measurements(
        moved={
            "solid_penetration_evaluated": True,
            "solid_penetration_cm": 0.0,
            "solid_penetrations": [],
        },
        added={
            "solid_penetration_evaluated": True,
            "solid_penetration_cm": 0.0,
            "solid_penetrations": [],
        },
    )

    report = solid_penetration.verify(context(
        tmp_path,
        candidate=candidate,
        input_scene=input_scene,
        case_spec=contract,
        spec={"candidate_measurements": _write(
            tmp_path, "local-solid-clear.json", evidence
        )},
    ))

    assert report["status"] == contracts.MEASURED
    check = check_by_id(report, "physics.solid_penetration")
    assert check["observed"]["population_scope"] == "edited_actors"
    assert check["observed"]["target_actor_count"] == 2
    assert check["observed"]["trusted_measurement_actor_count"] == 2
    assert check["observed"]["unresolved_actor_count"] == 0


def test_ue_probe_uses_exact_spatial_index_and_skips_duplicate_ground_work():
    source = PHYSICS_SCRIPT.read_text()
    cache_builder = source[source.index("def _build_solid_actor_cache"):
                           source.index("def _cached_collidable_components")]
    solid_only = source[source.index("def _measure_actor_solid_only"):
                        source.index("def _hit_values")]

    assert "uniform_grid_exact" in source
    assert "overflow_entries" in source
    assert "_spatial_broad_phase_entries" in source
    assert "_cached_collidable_components" in source
    assert "def _native_sweep_penetrations" in source
    assert "native_depths = _native_sweep_penetrations" in source
    assert "O(k squared)" in source
    assert "_collidable_components(collider)" not in cache_builder
    assert "_measure_actor_mesh" not in solid_only
    assert 'options.get("solid_penetration_only") is True' in source
    assert 'collider_path in target_paths' in source
    assert '"maximum_lateral_trace_count_per_actor"' in source
    assert "2 * LATERAL_GRID_DIMENSION ** 2" in source
    assert "transform.transform_vector" not in source
    assert "world_endpoint[index] - world_zero_tuple[index]" in source
    assert DEFAULT_PHYSICS_CHUNK_SIZE == 4
    assert _physics_chunk_size({}, 999) == 4
    assert _physics_chunk_size({}, 1000) == 1
    assert _physics_chunk_size({"physics_chunk_size": 8}, 2000) == 8


def test_lateral_panel_geometry_is_rotation_safe_and_rejects_floor_and_pole():
    symbols = _ue_physics_symbols(
        "_panel_geometry",
        "_vector_length",
        "_normalized",
        "_dot",
        "LATERAL_PANEL_THINNESS_RATIO",
        "LATERAL_PANEL_MAX_NORMAL_Z",
        "LATERAL_PANEL_MIN_VERTICAL_AXIS_Z",
    )
    panel_geometry = symbols["_panel_geometry"]

    identity = ((1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0))
    panel, reason = panel_geometry((25.0, 1.0, 36.0), identity)
    assert reason is None
    assert panel["thin_axis"] == 1
    assert panel["normal"] == pytest.approx((0.0, 1.0, 0.0))

    yaw_rotated = ((0.0, 1.0, 0.0),
                   (-1.0, 0.0, 0.0),
                   (0.0, 0.0, 1.0))
    rotated, reason = panel_geometry((25.0, 1.0, 36.0), yaw_rotated)
    assert reason is None
    assert rotated["thin_axis"] == 1
    assert rotated["normal"] == pytest.approx((-1.0, 0.0, 0.0))

    floor, reason = panel_geometry((25.0, 36.0, 1.0), identity)
    assert floor is None
    assert reason == "panel_is_not_vertical"
    pole, reason = panel_geometry((1.0, 1.0, 50.0), identity)
    assert pole is None
    assert reason == "not_a_thin_panel"


def test_physics_chunks_merge_only_with_complete_unique_actor_coverage():
    shared = {
        "schema_version": "0.4.0",
        "measurement_type": "ue_editor_actor_physics",
        "map_path": "/Game/SavedScenes/scene",
        "capabilities": {"solid_penetration": {"broad_phase": "actor_world_aabb"}},
        "runtime_provenance": {"engine_version": "5.8"},
    }
    payloads = [
        {**shared, "actors": {"a": {"solid_penetrations": []}},
         "diagnostics": {"status": "success", "requested_actor_count": 1,
                         "measured_actor_count": 1}},
        {**shared, "actors": {"b": {"solid_penetrations": []}},
         "diagnostics": {"status": "success", "requested_actor_count": 1,
                         "measured_actor_count": 1}},
    ]

    merged = _merge_physics_chunks(
        payloads,
        requested_actor_count=2,
        options={"solid_penetration_only": True},
        chunk_size=1,
        chunk_timeout=30.0,
        chunk_records=[{"index": 0}, {"index": 1}],
    )

    assert set(merged["actors"]) == {"a", "b"}
    assert merged["diagnostics"]["status"] == "success"
    assert merged["diagnostics"]["chunking"]["complete_coverage_required"] is True

    duplicate = [payloads[0], {**payloads[1], "actors": payloads[0]["actors"]}]
    with pytest.raises(Exception, match="duplicate Actor identity"):
        _merge_physics_chunks(
            duplicate,
            requested_actor_count=2,
            options={},
            chunk_size=1,
            chunk_timeout=30.0,
            chunk_records=[],
        )


def test_physics_chunk_validation_rejects_same_count_wrong_actor_identity():
    payload = {
        "actors": {"/Game/Scene.Wrong": {}},
        "diagnostics": {
            "status": "success",
            "requested_actor_count": 1,
            "measured_actor_count": 1,
        },
    }

    with pytest.raises(EvidenceError, match="identities do not match"):
        _validate_physics_chunk(
            payload, [{"actor_path": "/Game/Scene.Expected", "label": "Expected"}]
        )


def test_solid_only_contract_still_collects_its_measurement_targets():
    lamp = actor("lamp")
    contract = case([{
        "id": "solid-only",
        "primitive": "solid_penetration",
        "target_selector": {"scope": "candidate_all", "labels": ["lamp"]},
        "maximum_penetration_cm": 5.0,
    }])

    targets = _case_measurement_targets(contract, scene([lamp]))

    assert targets == [{"actor_path": lamp["actor_path"], "label": "lamp"}]
    assert _solid_broad_phase_tolerance(contract) == 5.0


def test_broad_phase_gate_uses_the_strictest_declared_penetration_threshold():
    contract = case([
        {"id": "five", "primitive": "solid_penetration",
         "maximum_penetration_cm": 5.0},
        {"id": "two", "primitive": "solid_penetration",
         "maximum_penetration_cm": 2.0},
    ])

    assert _solid_broad_phase_tolerance(contract) == 2.0


def test_ue_probe_reads_native_mtd_instead_of_aabb_depth():
    source = PHYSICS_SCRIPT.read_text()
    assert "hit.export_text()" in source
    assert "PenetrationDepth=" in source
    assert "hit_component_path != collider_path" in source
    assert '"depth_method": "ue_fhitresult_initial_overlap_mtd"' in source
    assert '"aabb_role": "broad_phase_only"' in source
    assert "_component_pair_collision_response" in source
    assert '"solid_non_blocking_overlaps"' in source
    assert "_reciprocal_native_sweep_penetration" in source


def test_ue_probe_filters_pairwise_non_blocking_collision_responses():
    symbols = _ue_physics_symbols("_component_pair_collision_response")
    classify = symbols["_component_pair_collision_response"]

    class Component:
        def __init__(self, object_type, responses):
            self.object_type = object_type
            self.responses = responses

        def get_collision_object_type(self):
            return self.object_type

        def get_collision_response_to_channel(self, channel):
            return self.responses[channel]

    target = Component(
        "CollisionChannel.WORLD_DYNAMIC",
        {"CollisionChannel.WORLD_STATIC": "CollisionResponse.BLOCK"},
    )
    blocking = Component(
        "CollisionChannel.WORLD_STATIC",
        {"CollisionChannel.WORLD_DYNAMIC": "CollisionResponse.BLOCK"},
    )
    overlap_only = Component(
        "CollisionChannel.WORLD_STATIC",
        {"CollisionChannel.WORLD_DYNAMIC": "CollisionResponse.OVERLAP"},
    )
    ignored = Component(
        "CollisionChannel.WORLD_STATIC",
        {"CollisionChannel.WORLD_DYNAMIC": "CollisionResponse.IGNORE"},
    )

    assert classify(target, blocking)["effective_response"] == "block"
    assert classify(target, overlap_only)["effective_response"] == "overlap"
    assert classify(target, ignored)["effective_response"] == "ignore"


def test_ue_probe_tries_multiple_reversible_directions_for_native_mtd():
    symbols = _ue_physics_symbols("_native_sweep_penetration_prepared")
    measure = symbols["_native_sweep_penetration_prepared"]

    class Actor:
        def __init__(self, path, location):
            self.path = path
            self.location = location

        def get_path_name(self):
            return self.path

        def get_actor_location(self):
            return self.location

    class HitComponent:
        def get_path_name(self):
            return "/Collider.Component"

        def get_name(self):
            return "ColliderComponent"

    collider_actor = Actor("/Collider", _FakeVector(100.0, 0.0, 0.0))
    target_actor = Actor("/Target", _FakeVector(0.0, 0.0, 0.0))
    hit_component = HitComponent()

    class Hit:
        def __init__(self, initial_overlap):
            self.initial_overlap = initial_overlap

        def to_dict(self):
            return {
                "initial_overlap": self.initial_overlap,
                "hit_actor": collider_actor,
                "hit_component": hit_component,
                "normal": _FakeVector(1.0, 0.0, 0.0),
            }

        def export_text(self):
            return "(bStartPenetrating=True,PenetrationDepth=7.25)"

    class TargetComponent:
        def __init__(self):
            self.sweep_count = 0
            self.restore_count = 0

        def get_owner(self):
            return target_actor

        def set_world_location(self, _destination, sweep, _teleport):
            if not sweep:
                self.restore_count += 1
                return None
            self.sweep_count += 1
            hit = Hit(initial_overlap=self.sweep_count > 1)
            return (True, hit) if self.sweep_count > 1 else hit

    class ColliderComponent(HitComponent):
        def get_owner(self):
            return collider_actor

    target_component = TargetComponent()
    result = measure(
        target_component, ColliderComponent(), _FakeVector(0.0, 0.0, 0.0)
    )

    assert result["status"] == "measured"
    assert result["penetration_depth_cm"] == 7.25
    assert result["probe_attempt_count"] == 2
    assert result["probe_orientation"] == "target_to_collider"
    assert target_component.restore_count >= 2


def test_ue_probe_reciprocal_mtd_preserves_target_normal_convention():
    symbols = _ue_physics_symbols("_reciprocal_native_sweep_penetration")
    reciprocal = symbols["_reciprocal_native_sweep_penetration"]

    class Actor:
        def __init__(self, path):
            self.path = path

        def get_path_name(self):
            return self.path

    class Component:
        def __init__(self, path, owner):
            self.path = path
            self.owner = owner

        def get_path_name(self):
            return self.path

        def get_owner(self):
            return self.owner

        def get_world_transform(self):
            return object()

    target = Component("/Target.Component", Actor("/Target"))
    collider = Component("/Collider.Component", Actor("/Collider"))
    symbols["_native_sweep_penetrations"] = lambda *_args: {
        "/Target.Component": {
            "status": "measured",
            "penetration_depth_cm": 8.0,
            "penetration_normal": [1.0, -0.5, 0.25],
            "depth_method": "ue_fhitresult_initial_overlap_mtd",
        }
    }

    result = reciprocal(
        target,
        collider,
        lambda *_args: [target],
        ["WorldDynamic"],
    )

    assert result["status"] == "measured"
    assert result["penetration_depth_cm"] == 8.0
    assert result["penetration_normal"] == [-1.0, 0.5, -0.25]
    assert result["probe_orientation"] == "collider_to_target"


def test_a_box_overlap_alone_is_never_a_penetration(tmp_path):
    """The lamppost's box is inside the church's. Only the editor can say."""
    church = actor("church", location=(0.0, 0.0, 500.0), extent=(800.0, 800.0, 500.0))
    lamp = actor("lamp", location=(600.0, 0.0, 200.0), extent=(20.0, 20.0, 200.0))
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene([church, lamp]), case_spec=PENETRATION))
    assert report["status"] == "not_evaluated"
    assert "not a penetration" in report["failure_reason"]
    check = check_by_id(report, "physics.solid_penetration")
    assert check["status"] == "not_evaluated"
    assert check["evidence"][0]["method"] == "world_aabb_broad_phase_candidate"


def test_missing_measurement_never_becomes_a_clean_candidate_all_pass(tmp_path):
    isolated = actor("bench", location=(0.0, 0.0, 50.0),
                     extent=(50.0, 25.0, 50.0))
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene([isolated]), case_spec=PENETRATION_ALL))

    assert report["status"] == "not_evaluated"
    check = check_by_id(report, "physics.solid_penetration")
    assert check["status"] == "not_evaluated"
    assert check["observed"]["evaluated_actor_count"] == 0
    assert check["observed"]["trusted_measurement_actor_count"] == 0
    assert check["evidence"][0]["reason"] == \
        "trusted_collision_measurement_missing"


def test_candidate_all_does_not_hide_target_to_target_overlap(tmp_path):
    wall = actor("wall", location=(0.0, 0.0, 150.0),
                 extent=(500.0, 10.0, 150.0))
    pipe = actor("pipe", location=(0.0, 0.0, 150.0),
                 extent=(20.0, 20.0, 150.0))
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene([wall, pipe]), case_spec=PENETRATION_ALL))

    assert report["status"] == "not_evaluated"
    check = check_by_id(report, "physics.solid_penetration")
    assert check["observed"]["broad_phase_candidate_count"] == 2
    assert {item["collider_label"] for item in check["evidence"]} == {"wall", "pipe"}


def test_candidate_all_pass_requires_trusted_measurement_for_every_actor(tmp_path):
    bench = actor("bench", location=(0.0, 0.0, 50.0),
                  extent=(50.0, 25.0, 50.0))
    crate = actor("crate", location=(300.0, 0.0, 50.0),
                  extent=(50.0, 50.0, 50.0))
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene([bench, crate]), case_spec=PENETRATION_ALL,
        spec={"candidate_measurements": _write(
            tmp_path, "all-clear.json", measurements(
                bench={"solid_penetration_evaluated": True,
                       "solid_penetration_cm": 0.0,
                       "solid_penetrations": []},
                crate={"solid_penetration_evaluated": True,
                       "solid_penetration_cm": 0.0,
                       "solid_penetrations": []},
            ))}))

    assert report["status"] == contracts.MEASURED
    check = check_by_id(report, "physics.solid_penetration")
    assert check["observed"]["evaluated_actor_count"] == 2
    assert check["observed"]["unresolved_actor_count"] == 0


def test_the_editors_native_collision_measurement_decides_it(tmp_path):
    church = actor("church", location=(0.0, 0.0, 500.0), extent=(800.0, 800.0, 500.0))
    lamp = actor("lamp", location=(600.0, 0.0, 200.0), extent=(20.0, 20.0, 200.0))
    cleared = solid_penetration.verify(context(
        tmp_path, candidate=scene([church, lamp]), case_spec=PENETRATION,
        spec={"candidate_measurements": _write(tmp_path, "clear.json", measurements(
            lamp={"solid_penetration_evaluated": True,
                  "solid_penetrations": []}))}))
    assert cleared["status"] == contracts.MEASURED

    caught = solid_penetration.verify(context(
        tmp_path, candidate=scene([church, lamp]), case_spec=PENETRATION,
        spec={"candidate_measurements": _write(tmp_path, "deep.json", measurements(
            lamp={"solid_penetration_evaluated": True,
                  "solid_penetrations": [{"collider": "church",
                                          "penetration_depth_cm": 42.0,
                                          "depth_method":
                                          "ue_fhitresult_initial_overlap_mtd"}]}))}))
    assert caught["status"] == contracts.MEASURED
    assert check_by_id(caught, "physics.solid_penetration")["observed"][
        "penetrating_actor_count"] == 1


def test_broad_phase_clearance_is_valid_only_for_its_frozen_threshold(tmp_path):
    lamp = actor("lamp", location=(0.0, 0.0, 200.0),
                 extent=(20.0, 20.0, 200.0))
    record = measurements(lamp={
        "solid_penetration_evaluated": True,
        "solid_penetration_method": "ue_aabb_broad_phase_no_candidates",
        "solid_broad_phase_tolerance_cm": 5.0,
        "solid_penetrations": [],
    })

    canonical = solid_penetration.verify(context(
        tmp_path, candidate=scene([lamp]), case_spec=PENETRATION,
        spec={"candidate_measurements": _write(
            tmp_path, "broad-clear-5cm.json", record)}))
    tighter = solid_penetration.verify(context(
        tmp_path, candidate=scene([lamp]), case_spec=PENETRATION_TIGHT,
        spec={"candidate_measurements": _write(
            tmp_path, "broad-clear-insufficient.json", record)}))

    assert canonical["status"] == contracts.MEASURED
    assert tighter["status"] == "not_evaluated"
    assert check_by_id(tighter, "physics.solid_penetration")["status"] == \
        "not_evaluated"


def test_legacy_mesh_penetration_depth_is_not_formal_evidence(tmp_path):
    church = actor("church", location=(0.0, 0.0, 500.0),
                   extent=(800.0, 800.0, 500.0))
    lamp = actor("lamp", location=(600.0, 0.0, 200.0),
                 extent=(20.0, 20.0, 200.0))
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene([church, lamp]), case_spec=PENETRATION,
        spec={"candidate_measurements": _write(
            tmp_path, "legacy-depth.json", measurements(
                lamp={"solid_penetration_cm": 42.0,
                      "solid_penetrations": []}))}))

    assert report["status"] == "not_evaluated"
    assert check_by_id(report, "physics.solid_penetration")["status"] == \
        "not_evaluated"


def test_exact_collision_clearance_overrides_mesh_surface_depth(tmp_path):
    """A surface-depth diagnostic cannot overrule UE's exact collision test."""
    church = actor("church", location=(0.0, 0.0, 500.0),
                   extent=(800.0, 800.0, 500.0))
    lamp = actor("lamp", location=(600.0, 0.0, 200.0),
                 extent=(20.0, 20.0, 200.0))
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene([church, lamp]), case_spec=PENETRATION,
        spec={"candidate_measurements": _write(
            tmp_path, "exact-clear.json", measurements(lamp={
                "solid_penetration_evaluated": True,
                "solid_penetration_cm": 42.0,
                "solid_penetrations": [],
            }))}))
    assert report["status"] == contracts.MEASURED
    check = check_by_id(report, "physics.solid_penetration")
    assert check["observed"]["penetrating_actor_count"] == 0
    assert check["observed"]["trusted_measurement_actor_count"] == 1


def test_exact_collision_contact_gets_a_depth_weighted_score(tmp_path):
    church = actor("church", location=(0.0, 0.0, 500.0),
                   extent=(800.0, 800.0, 500.0))
    lamp = actor("lamp", location=(600.0, 0.0, 200.0),
                 extent=(20.0, 20.0, 200.0))
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene([church, lamp]), case_spec=PENETRATION,
        spec={"candidate_measurements": _write(
            tmp_path, "exact-hit.json", measurements(lamp={
                "solid_penetration_evaluated": True,
                "solid_penetration_cm": 0.0,
                "solid_penetrations": [{
                    "collider": "church",
                    "penetration_depth_cm": 42.0,
                    "depth_method": "ue_fhitresult_initial_overlap_mtd",
                    "confirmed_overlap": True,
                }],
            }))}))
    assert report["status"] == contracts.MEASURED
    expected_normalized_excess = 37.0 / 40.0
    expected_penalty = 1.0 + math.log1p(expected_normalized_excess)
    assert report["score"] == 0.0
    check = check_by_id(report, "physics.solid_penetration")
    assert check["status"] == contracts.MEASURED
    assert check["outcome"] == contracts.MEASURED
    assert check["observed"]["mean_penetration_depth_penalty"] == \
        pytest.approx(expected_penalty)
    assert check["observed"]["depth_weighted_score"] == 0.0
    evidence = check["evidence"][0]
    assert evidence["verdict_basis"] == ["ue_collision_body_native_mtd"]
    assert evidence["characteristic_size_cm"] == 40.0
    assert evidence["normalized_excess_penetration"] == \
        pytest.approx(expected_normalized_excess)
    assert evidence["penetration_depth_penalty"] == \
        pytest.approx(expected_penalty)


def test_native_collision_contact_within_tolerance_passes(tmp_path):
    church = actor("church", location=(0.0, 0.0, 500.0),
                   extent=(800.0, 800.0, 500.0))
    lamp = actor("lamp", location=(600.0, 0.0, 200.0),
                 extent=(20.0, 20.0, 200.0))
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene([church, lamp]), case_spec=PENETRATION,
        spec={"candidate_measurements": _write(
            tmp_path, "shallow-native-contact.json", measurements(lamp={
                "solid_penetration_evaluated": True,
                "solid_penetrations": [{
                    "collider": "church",
                    "penetration_depth_cm": 2.0,
                    "depth_method": "ue_fhitresult_initial_overlap_mtd",
                    "depth_status": "measured",
                    "confirmed_overlap": True,
                }],
            }))}))

    assert report["status"] == contracts.MEASURED
    check = check_by_id(report, "physics.solid_penetration")
    assert check["observed"]["maximum_observed_penetration_cm"] == 2.0
    assert check["observed"]["penetrating_actor_count"] == 0
    assert check["observed"]["mean_penetration_depth_penalty"] == 0.0
    assert check["observed"]["depth_weighted_score"] == 1.0
    assert report["score"] == 1.0


def test_default_actor_scale_tolerance_has_five_cm_floor_and_fifty_cm_cap(
    tmp_path,
):
    small = actor("small", extent=(20.0, 30.0, 50.0))
    medium = actor("medium", extent=(200.0, 300.0, 400.0))
    large = actor("large", extent=(1000.0, 1200.0, 1500.0))

    def native_contact(depth_cm: float) -> dict:
        return {
            "solid_penetration_evaluated": True,
            "solid_penetrations": [{
                "collider": "external-solid",
                "penetration_depth_cm": depth_cm,
                "depth_method": "ue_fhitresult_initial_overlap_mtd",
                "confirmed_overlap": True,
            }],
        }

    report = solid_penetration.verify(context(
        tmp_path,
        candidate=scene([small, medium, large]),
        case_spec=ADAPTIVE_PENETRATION,
        spec={"candidate_measurements": _write(
            tmp_path,
            "actor-scale-native-contacts.json",
            measurements(
                small=native_contact(6.0),
                medium=native_contact(15.0),
                large=native_contact(51.0),
            ),
        )},
    ))

    assert report["status"] == contracts.MEASURED
    check = check_by_id(report, "physics.solid_penetration")
    assert check["observed"]["penetrating_actor_count"] == 2
    assert check["observed"][
        "minimum_effective_penetration_tolerance_cm"
    ] == 5.0
    assert check["observed"][
        "maximum_effective_penetration_tolerance_cm"
    ] == 50.0
    evidence = {item["actor_label"]: item for item in check["evidence"]}
    assert set(evidence) == {"small", "large"}
    assert evidence["small"]["characteristic_size_cm"] == 40.0
    assert evidence["small"]["effective_penetration_tolerance_cm"] == 5.0
    assert evidence["large"]["characteristic_size_cm"] == 2000.0
    assert evidence["large"]["effective_penetration_tolerance_cm"] == 50.0


def test_actor_scale_tolerance_uses_five_percent_for_medium_actor(tmp_path):
    medium = actor("medium", extent=(200.0, 300.0, 400.0))
    report = solid_penetration.verify(context(
        tmp_path,
        candidate=scene([medium]),
        case_spec=ADAPTIVE_PENETRATION,
        spec={"candidate_measurements": _write(
            tmp_path,
            "medium-native-contact.json",
            measurements(medium={
                "solid_penetration_evaluated": True,
                "solid_penetrations": [{
                    "collider": "external-solid",
                    "penetration_depth_cm": 21.0,
                    "depth_method": "ue_fhitresult_initial_overlap_mtd",
                    "confirmed_overlap": True,
                }],
            }),
        )},
    ))

    check = check_by_id(report, "physics.solid_penetration")
    assert check["observed"]["penetrating_actor_count"] == 1
    evidence = check["evidence"][0]
    assert evidence["characteristic_size_cm"] == 400.0
    assert evidence["effective_penetration_tolerance_cm"] == 20.0
    assert evidence["normalized_excess_penetration"] == pytest.approx(1.0 / 400.0)


def test_exact_overlap_without_depth_withholds_instead_of_inventing_penetration(tmp_path):
    church = actor("church", location=(0.0, 0.0, 500.0),
                   extent=(800.0, 800.0, 500.0))
    lamp = actor("lamp", location=(600.0, 0.0, 200.0),
                 extent=(20.0, 20.0, 200.0))
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene([church, lamp]), case_spec=PENETRATION,
        spec={"candidate_measurements": _write(
            tmp_path, "exact-hit-no-depth.json", measurements(lamp={
                "solid_penetration_evaluated": True,
                "solid_penetration_cm": 0.0,
                "solid_penetrations": [{
                    "collider": "church",
                    "confirmed_overlap": True,
                    "penetration_depth_cm": None,
                }],
            }))}))

    assert report["status"] == "not_evaluated"
    check = check_by_id(report, "physics.solid_penetration")
    assert check["status"] == "not_evaluated"
    assert check["observed"]["penetrating_actor_count"] == 0
    assert check["evidence"][0]["reason"] == \
        "confirmed_collision_overlap_depth_unavailable"


def test_same_logical_object_contact_is_not_external_penetration(tmp_path):
    wall = actor("wall", logical_object_id="cathedral")
    arch = actor("arch", logical_object_id="cathedral")
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene([wall, arch]), case_spec=PENETRATION_ALL,
        spec={"candidate_measurements": _write(
            tmp_path, "same-object.json", measurements(
                wall={"solid_penetration_evaluated": True,
                      "solid_penetration_cm": 0.0,
                      "solid_penetrations": [{
                          "collider": "arch",
                          "confirmed_overlap": True,
                          "penetration_depth_cm": None,
                      }]},
                arch={"solid_penetration_evaluated": True,
                      "solid_penetration_cm": 0.0,
                      "solid_penetrations": [{
                          "collider": "wall",
                          "confirmed_overlap": True,
                          "penetration_depth_cm": None,
                      }]},
            ))}))

    assert report["status"] == contracts.MEASURED
    check = check_by_id(report, "physics.solid_penetration")
    assert check["observed"]["penetrating_actor_count"] == 0
    assert check["observed"]["unresolved_actor_count"] == 0


def test_a_light_is_not_penetrating_anything(tmp_path):
    """Non-solid Actors are filtered before the question is asked — the lamp
    inside the church's box neither fails the scene nor leaves it uncertain,
    while the solid actor beside it keeps the check answerable."""
    spec = case([{"id": "no-clipping", "primitive": "solid_penetration",
                  "target_selector": {"scope": "candidate_all",
                                      "labels": ["church", "lamp"]}}])
    church = actor("church", location=(0.0, 0.0, 500.0), extent=(800.0, 800.0, 500.0))
    lamp = actor("lamp", location=(0.0, 0.0, 200.0), extent=(20.0, 20.0, 200.0),
                 actor_class="/Script/Engine.PointLight")
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene([church, lamp]), case_spec=spec))
    assert report["status"] == "not_evaluated"
    check = check_by_id(report, "physics.solid_penetration")
    assert check["observed"]["filtered_non_solid_actor_count"] == 1
    assert check["observed"]["evaluated_actor_count"] == 0


def test_a_population_of_only_lights_is_not_a_perfect_score(tmp_path):
    """A selection the non-solid filter empties used to pass at 1.0 with an
    evaluated_actor_count of 0 — a check that cannot fail must not pass."""
    spec = case([{"id": "no-clipping", "primitive": "solid_penetration",
                  "target_selector": {"scope": "candidate_all",
                                      "labels": ["lamp_a", "lamp_b"]}}])
    lamps = [actor(name, location=(index * 100.0, 0.0, 200.0),
                   extent=(20.0, 20.0, 200.0),
                   actor_class="/Script/Engine.PointLight")
             for index, name in enumerate(["lamp_a", "lamp_b"])]
    report = solid_penetration.verify(context(
        tmp_path, candidate=scene(lamps), case_spec=spec))

    assert report["status"] == "not_applicable" and report["score"] is None
    check = check_by_id(report, "physics.solid_penetration")
    assert check["status"] == "not_applicable"
    assert check["observed"]["filtered_non_solid_actor_count"] == 2


def _write(tmp_path, name: str, document) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return str(path)


REGRESSION = case([{"id": "no-worse", "primitive": "physics_regression",
                    "scope": "candidate_all"}])


def crowded_room() -> dict:
    """An imported level that already contains one interpenetration."""
    return scene([
        actor("wall", location=(0.0, 0.0, 150.0), extent=(500.0, 10.0, 150.0)),
        actor("pipe", location=(0.0, 0.0, 150.0), extent=(20.0, 20.0, 150.0)),
        actor("floor", location=(0.0, 0.0, -5.0), extent=(500.0, 500.0, 5.0))])


def test_a_pre_existing_collision_is_not_the_agents_fault(tmp_path):
    source = crowded_room()
    candidate = copy.deepcopy(source)
    report = physics_regression.verify(context(
        tmp_path, candidate=candidate, input_scene=source, case_spec=REGRESSION))
    check = check_by_id(report, "physics.regression.collision")
    assert check["status"] == contracts.MEASURED
    assert check["observed"]["unchanged"] == 1
    assert check["observed"]["new_collision_pairs"] == 0


def test_a_collision_the_edit_introduced_is(tmp_path):
    source = crowded_room()
    candidate = copy.deepcopy(source)
    candidate["actors"].append(
        actor("crate", location=(200.0, 0.0, 150.0), extent=(60.0, 60.0, 60.0)))
    report = physics_regression.verify(context(
        tmp_path, candidate=candidate, input_scene=source, case_spec=REGRESSION))
    check = check_by_id(report, "physics.regression.collision")
    assert check["status"] == contracts.MEASURED
    assert check["observed"]["new_collision_pairs"] == 1
    assert check["observed"]["unchanged"] == 1
    assert check["score_role"] == "report_only"
    assert check["contributes_to_aggregate"] is False


def test_collision_regression_is_report_only_in_leaf_score():
    checks = [
        Check("physics.regression.collision", contracts.FAIL),
        Check("physics.regression.clearance", contracts.PASS),
        Check("physics.regression.ground_contact", contracts.PASS),
        Check("physics.regression.out_of_bounds", "not_applicable"),
    ]

    assert physics_regression._score_without_collision(checks) == 1.0


def test_fixing_a_pre_existing_collision_is_credited_not_ignored(tmp_path):
    source = crowded_room()
    candidate = copy.deepcopy(source)
    candidate["actors"][1]["bounds"]["origin_cm"] = [300.0, 300.0, 150.0]
    candidate["actors"][1]["transform"]["location_cm"] = [300.0, 300.0, 150.0]
    check = check_by_id(physics_regression.verify(context(
        tmp_path, candidate=candidate, input_scene=source, case_spec=REGRESSION)),
        "physics.regression.collision")
    assert check["observed"]["fixed"] == 1
    assert check["status"] == contracts.MEASURED


def test_one_measured_side_withholds_the_measured_halves(tmp_path):
    """A regression against an unmeasured input reads as "nothing new"."""
    source = crowded_room()
    candidate = copy.deepcopy(source)
    report = physics_regression.verify(context(
        tmp_path, candidate=candidate, input_scene=source, case_spec=REGRESSION,
        spec={"candidate_measurements": _write(tmp_path, "cand.json", measurements(
            wall={"grounded": True, "ground_gap_cm": 0.0, "penetration_cm": 0.0,
                  "support_fraction": 1.0, "out_of_bounds": False}))}))
    assert report["status"] == "not_evaluated"
    ground = check_by_id(report, "physics.regression.ground_contact")
    assert ground["status"] == "not_evaluated"
    assert "incomplete" in ground["failure_reason"]
