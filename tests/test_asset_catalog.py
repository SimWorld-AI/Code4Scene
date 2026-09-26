"""A Candidate's category comes from the asset, not from what it says it is."""

from __future__ import annotations

import json

import pytest
from declared_cases import IDS, StubTask, scene

from code4scene.evaluation import asset_catalog, ue_evidence
from code4scene.evaluation.context import Context
from code4scene.evaluation.selection import (category_from_actor,
                                             select_candidate_actors)

CATALOG = {
    "_meta": {"id": "test-palette-v1"},
    "market_goods": {"description": "stalls and their wares",
                     "items": ["/Game/Props/SM_MarketStall.SM_MarketStall"]},
    "trees": ["/Game/CityDatabase/blueprints/BP_Tree1.BP_Tree1"],
    "static_meshes": {"items": ["/Engine/BasicShapes/Cube.Cube"]},
}


def actor(label, asset_path=None, **extra):
    return {"label": label, "asset_path": asset_path, "actor_tags": [], **extra}


def test_a_blueprint_class_path_is_the_same_asset_as_its_object_path():
    """The export writes `..._C`; the catalog writes the object. One asset."""
    catalog = asset_catalog.load(CATALOG)
    assert catalog.category_for(
        "/Game/CityDatabase/blueprints/BP_Tree1.BP_Tree1_C") == "trees"


def test_the_catalog_id_travels_so_two_scores_can_be_compared():
    assert asset_catalog.load(CATALOG).id == "test-palette-v1"
    assert asset_catalog.load({"trees": ["/Game/A.A"]}, "fallback").id == "fallback"


def test_a_flat_asset_list_is_also_a_catalog():
    catalog = asset_catalog.load(
        {"assets": [{"path": "/Game/Props/SM_Bench.SM_Bench", "category": "seating"}]})
    assert catalog.category_for("/Game/Props/SM_Bench.SM_Bench") == "seating"


def test_an_empty_catalog_is_refused_rather_than_reporting_every_object_unknown():
    with pytest.raises(asset_catalog.AssetCatalogError):
        asset_catalog.load({"_meta": {"id": "empty"}})


def test_a_cube_named_cathedral_does_not_satisfy_a_rubric_asking_for_one():
    """The failure this module exists to prevent, stated as a test."""
    catalog = asset_catalog.load(CATALOG)
    grey_box = actor("Cathedral_Central", "/Engine/BasicShapes/Cube.Cube")
    catalog.apply([grey_box])
    assert grey_box["asset_category"] == "static_meshes"
    assert select_candidate_actors(
        [grey_box], {"allowed_categories": ["cathedral"]}) == []


def test_a_category_the_candidate_claimed_is_replaced_not_believed():
    """An agent can write Actor metadata; it cannot write the asset path."""
    catalog = asset_catalog.load(CATALOG)
    forged = actor("Stall_01", "/Engine/BasicShapes/Cube.Cube",
                   asset_category="market_goods")
    counts = catalog.apply([forged])
    assert forged["asset_category"] == "static_meshes"
    assert counts == {"categories_resolved": 1, "categories_unknown": 0}
    assert select_candidate_actors(
        [forged], {"allowed_categories": ["market_goods"]}) == []


def test_the_fallback_category_keys_cannot_answer_after_the_catalog_refused():
    """`category_from_actor` falls back through semantic_category, category
    and semantic_concept when asset_category is empty — all writable by the
    agent. Applying a catalog used to leave them in place, so a refused
    category could be re-asserted through the side door."""
    catalog = asset_catalog.load(CATALOG)
    forged = actor("Mystery", "/Game/Somewhere/SM_Unlisted.SM_Unlisted",
                   semantic_category="cathedral", category="cathedral",
                   semantic_concept="cathedral")
    catalog.apply([forged])

    assert category_from_actor(forged) is None
    assert select_candidate_actors(
        [forged], {"allowed_categories": ["cathedral"]}) == []


def test_an_asset_the_catalog_does_not_know_has_no_category():
    catalog = asset_catalog.load(CATALOG)
    unknown = actor("Mystery", "/Game/Somewhere/SM_Unlisted.SM_Unlisted",
                    asset_category="market_goods")
    counts = catalog.apply([unknown])
    assert unknown["asset_category"] is None
    assert counts == {"categories_resolved": 0, "categories_unknown": 1}


def test_the_real_asset_is_found_and_counted():
    catalog = asset_catalog.load(CATALOG)
    real = actor("Stall_01", "/Game/Props/SM_MarketStall.SM_MarketStall")
    catalog.apply([real])
    assert real["asset_category"] == "market_goods"
    assert real[asset_catalog.SOURCE_KEY] == asset_catalog.FROM_CATALOG
    assert len(select_candidate_actors(
        [real], {"allowed_categories": ["market_goods"]})) == 1


def test_a_multi_mesh_actor_falls_back_to_its_components():
    catalog = asset_catalog.load(CATALOG)
    compound = actor("Stall_02", None)
    compound["component_asset_paths"] = [
        "/Game/Unlisted/SM_Cloth.SM_Cloth",
        "/Game/Props/SM_MarketStall.SM_MarketStall"]
    catalog.apply([compound])
    assert compound["asset_category"] == "market_goods"


def test_selection_no_longer_reads_a_category_out_of_actor_tags():
    """`execute_python_script` can set any tag; it must not set the score."""
    tagged = actor("Cube_7", "/Engine/BasicShapes/Cube.Cube")
    tagged["actor_tags"] = ["simcodearena.asset_category=cathedral",
                            "asset_category=cathedral"]
    assert category_from_actor(tagged) is None
    assert select_candidate_actors(
        [tagged], {"allowed_categories": ["cathedral"]}) == []


def test_an_authored_category_field_is_still_read_when_no_catalog_ran():
    """Scene-repair levels are authored offline; the exporter lifts their tags."""
    authored = actor("Pew_3", "/Game/Props/SM_Pew.SM_Pew",
                     asset_category="seating")
    assert category_from_actor(authored) == "seating"


# ── end to end: the catalog reaches unified scene evidence ────────────────

def _context(tmp_path, actors):
    task_path = tmp_path / "task.yaml"
    task_path.write_text("id: catalog-test\n")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(scene(actors)))
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(CATALOG))
    spec = {
        "candidate_scene": str(candidate),
        "asset_catalog": str(path),
        "case_id": "catalog-test",
    }
    return Context(record={"metrics": {}}, task=StubTask(task_path),
                   ids=dict(IDS), out_dir=tmp_path / "out", spec=spec)


def test_catalog_categories_reach_the_unified_semantic_evidence(tmp_path):
    cube = actor(
        "Cathedral_Central",
        "/Engine/BasicShapes/Cube.Cube",
        asset_category="cathedral",
        bounds={"origin_cm": [0, 0, 50], "extent_cm": [50, 50, 50]},
    )
    evidence = ue_evidence.collect(_context(tmp_path, [cube]))

    assert evidence.candidate["actors"][0]["asset_category"] == "static_meshes"
    assert evidence.asset_catalog_id == "test-palette-v1"
    assert evidence.categories_resolved == 1
    assert evidence.categories_unknown == 0
    assert "asset_catalog" in evidence.probes_used()
