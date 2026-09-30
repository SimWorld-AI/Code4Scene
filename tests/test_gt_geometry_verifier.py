"""Integration boundary tests for the public ``gt_geometry`` verifier."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

from code4scene.core import inventory
from code4scene.evaluation import contracts
from code4scene.evaluation.pairwise_identity_locator import (
    LLMPairwiseIdentityLocatorBackend,
)
from code4scene.evaluation.requirement_graph.existing_llm import (
    LLMResponse,
    ToolCall,
)
from code4scene.evaluation.requirement_graph import vlm_client as vlm_client_module
from code4scene.evaluation.verifiers import gt_geometry


IDS = {"task_bundle_id": "synthetic-town", "episode_id": "candidate"}


def actor(
    label: str,
    x: float,
    *,
    y: float = 0.0,
    yaw: float = 0.0,
    extent: tuple[float, float, float] = (50.0, 50.0, 50.0),
    asset: str | None = None,
    guid: str | None = None,
    logical_id: str | None = None,
    stable_id: str | None = None,
) -> dict:
    return {
        "label": label,
        "actor_guid": guid,
        "stable_actor_id": stable_id,
        "class": "/Script/Engine.StaticMeshActor",
        "asset_path": asset or f"/Game/ExamplePack/{label}.{label}",
        "logical_object_id": logical_id,
        "transform": {
            "location_cm": [x, y, 0.0],
            "rotation_deg": [0.0, yaw, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "bounds": {
            "origin_cm": [x, y, 0.0],
            "extent_cm": list(extent),
        },
    }


def context(*, scoring=None, bridge=None, out_dir=None, spec=None):
    return SimpleNamespace(
        ids=IDS,
        task=SimpleNamespace(path="synthetic-town.yaml"),
        scoring=scoring,
        bridge=bridge,
        out_dir=out_dir,
        spec=spec or {},
    )


class PairwiseLocatorClient:
    name = "pairwise-locator-test-client"
    model = "pairwise-locator-test-model"
    max_tokens = 2048
    _strict_tool_calls = True
    _text_action_mode = False

    def __init__(self, *, status="plausible_locator", fail=False):
        self.status = status
        self.fail = fail
        self.calls = []

    def chat(self, messages, tools, **kwargs):
        self.calls.append((messages, tools, kwargs))
        if self.fail:
            raise RuntimeError("pairwise locator unavailable")
        payload = json.loads(messages[1].content[0]["text"].split(":\n", 1)[1])
        candidate_group = payload["candidate_identity_groups"][0]["group_id"]
        matches = [
            {
                "canonical_group_id": canonical["group_id"],
                "candidates": [
                    {
                        "candidate_group_id": candidate_group,
                        "confidence": 0.9,
                        "status": self.status,
                        "reason": "same long-tail logical object kind",
                    }
                ],
            }
            for canonical in payload["canonical_identity_groups"]
        ]
        return LLMResponse(
            text=None,
            tool_calls=[
                ToolCall(
                    "pairwise-locator-call",
                    "return_pairwise_identity_edges",
                    {"schema_version": "1.0", "matches": matches},
                )
            ],
            usage={"prompt_tokens": 50},
        )


def test_authored_gt_graph_needs_no_stable_ids(monkeypatch, tmp_path):
    """Geometric assignment, not repair ids, establishes correspondence."""

    canonical = [actor("Church", 0.0, guid="GT-CHURCH")]
    candidate = [actor("Church", 125.0)]
    clean_bridge = object()
    monkeypatch.setattr(gt_geometry, "read_label", lambda _context: {
        "gt_id": "synthetic-town-demo",
        "canonical_actors": canonical,
    })
    monkeypatch.setattr(
        gt_geometry.ue_evidence,
        "collect",
        lambda _context: SimpleNamespace(
            candidate_actors=lambda: candidate,
            exported_from="independent_scoring_editor",
        ),
    )

    report = gt_geometry.verify(context(
        scoring=SimpleNamespace(bridge=clean_bridge),
        out_dir=tmp_path,
    ))

    assert report["status"] == contracts.MEASURED
    assert report["metrics"]["global_yaw_error_deg"] is None
    assert report["score"] is None
    assert report["metrics"]["matched_actor_count"] == 1
    assert report["metrics"]["metric_version"] == "gt-geometry-v5"
    assert report["metrics"]["global_translation_error_cm"] == 125.0
    assert report["metrics"]["aligned_position_rmse_cm"] == 0.0
    assert report["evidence"]["candidate_exported_from"] == (
        "independent_scoring_editor"
    )
    assert report["evidence"]["visual_responsibility"] == (
        "scene_diff.visual_semantic_diff"
    )
    assert report["probes_used"] == ("scene_graph", "answer_key")
    audit_path = report["artifacts"]["correspondence"]
    audit = json.loads(open(audit_path).read())
    assert audit["schema_version"] == "gt-geometry-correspondence.v5"
    assert audit["identity_locator"]["role"] == (
        "locator_only_not_semantic_verdict"
    )


def test_missing_canonical_actor_list_is_refused(monkeypatch):
    monkeypatch.setattr(gt_geometry, "read_label", lambda _context: {
        "gt_id": "synthetic-town-demo",
        "canonical_actors": [],
    })

    report = gt_geometry.verify(context(bridge=object()))

    assert report["status"] == contracts.ERROR
    assert report["score"] is None
    assert "no canonical scene" in report["failure_reason"]


def test_missing_candidate_editor_is_refused(monkeypatch):
    monkeypatch.setattr(gt_geometry, "read_label", lambda _context: {
        "gt_id": "synthetic-town-demo",
        "canonical_actors": [actor("Church", 0.0, guid="GT-CHURCH")],
    })

    report = gt_geometry.verify(context())

    assert report["status"] == contracts.ERROR
    assert report["score"] is None
    assert "independent scoring editor" in report["failure_reason"]


def test_global_translation_is_penalized_then_removed_from_local_metrics():
    canonical = [actor("Church", 0.0), actor("Bank", 1000.0)]
    candidate = [actor("Church", 5000.0), actor("Bank", 6000.0)]

    metrics = gt_geometry.compare(candidate, canonical)

    assert metrics["global_translation_vector_cm"] == [-5000.0, 0.0, 0.0]
    assert metrics["global_translation_error_cm"] == 5000.0
    assert metrics["global_yaw_error_deg"] == 0.0
    assert metrics["assignment_cost_matrix_cell_count"] == 2
    assert metrics["assignment_cost_matrix_reduction_ratio"] == 0.5
    assert metrics["aligned_position_rmse_cm"] == 0.0
    assert metrics["pairwise_layout_error"] == 0.0


def test_global_yaw_is_penalized_without_becoming_translation_or_local_error():
    canonical = [actor("Church", 0.0), actor("Bank", 1000.0)]
    candidate = [
        actor("Church", 0.0, yaw=-90.0),
        actor("Bank", 0.0, y=-1000.0, yaw=-90.0),
    ]

    metrics = gt_geometry.compare(candidate, canonical)

    assert metrics["global_yaw_error_deg"] == 90.0
    assert metrics["global_translation_error_cm"] == 0.0
    assert metrics["aligned_position_rmse_cm"] == 0.0
    assert metrics["aligned_rotation_mean_deg"] == 0.0


def test_local_layout_error_remains_after_global_alignment():
    canonical = [actor("Church", 0.0), actor("Bank", 1000.0)]
    candidate = [actor("Church", 5000.0), actor("Bank", 6200.0)]

    metrics = gt_geometry.compare(candidate, canonical)

    assert metrics["global_translation_error_cm"] == 5100.0
    assert metrics["aligned_position_rmse_cm"] == 100.0
    assert metrics["pairwise_layout_error"] == 2.0
    assert metrics["mean_footprint_iou"] == 0.0


def test_scale_log_rmse_is_root_mean_square_not_mean_of_actor_errors():
    canonical = [actor("Church", 0.0), actor("Bank", 1000.0)]
    candidate = [actor("Church", 0.0), actor("Bank", 1000.0)]
    candidate[0]["transform"]["scale"] = [math.e, 1.0, 1.0]
    candidate[1]["transform"]["scale"] = [math.exp(2.0), 1.0, 1.0]

    metrics = gt_geometry.compare(candidate, canonical)

    # Per-Actor log RMSE values are sqrt(1/3) and sqrt(4/3), so their
    # cross-Actor RMSE is sqrt(5/6), not their arithmetic mean.
    assert metrics["scale_log_rmse"] == round(math.sqrt(5.0 / 6.0), 6)
    assert metrics["scale_log_rmse"] != metrics["scale_log_error"]["mean"]


def test_connected_floor_parts_are_penalized_then_compared_as_one_object():
    floor_asset = "/Game/ExamplePack/SM_Floor.SM_Floor"
    canonical = [
        actor("Floor_A", -50.0, extent=(50.0, 100.0, 1.0), asset=floor_asset),
        actor("Floor_B", 50.0, extent=(50.0, 100.0, 1.0), asset=floor_asset),
    ]
    candidate = [
        actor("Floor_1", -75.0, extent=(25.0, 100.0, 1.0), asset=floor_asset),
        actor("Floor_2", -25.0, extent=(25.0, 100.0, 1.0), asset=floor_asset),
        actor("Floor_3", 25.0, extent=(25.0, 100.0, 1.0), asset=floor_asset),
        actor("Floor_4", 75.0, extent=(25.0, 100.0, 1.0), asset=floor_asset),
    ]

    metrics = gt_geometry.compare(candidate, canonical)

    assert metrics["candidate_logical_object_count"] == 1
    assert metrics["canonical_logical_object_count"] == 1
    assert metrics["matched_logical_object_count"] == 1
    assert metrics["extra_logical_object_count"] == 0
    assert metrics["matched_actor_count"] == 2
    assert metrics["missing_actor_count"] == 0
    assert metrics["extra_actor_count"] == 2
    assert metrics["over_fragmented_actor_count"] == 2
    assert metrics["under_segmented_actor_count"] == 0
    assert metrics["fragmentation_error_rate"] == 1.0
    assert metrics["aligned_position_rmse_cm"] == 0.0
    assert metrics["mean_footprint_iou"] == 1.0


def test_large_modular_parts_use_scale_aware_connectivity():
    rock_asset = "/Game/ExamplePack/SM_Mountain.SM_Mountain"
    perimeter = [
        actor("Mountain_A", 0.0, extent=(500.0, 400.0, 150.0), asset=rock_asset),
        actor(
            "Mountain_B",
            1500.0,
            extent=(500.0, 400.0, 150.0),
            asset=rock_asset,
        ),
    ]

    metrics = gt_geometry.compare(perimeter, perimeter)

    assert metrics["candidate_logical_object_count"] == 1
    assert metrics["canonical_logical_object_count"] == 1


def test_connected_compact_props_remain_distinct_logical_objects():
    barrel_asset = "/Game/ExamplePack/SM_Barrel.SM_Barrel"
    canonical = [
        actor("Barrel_A", 0.0, extent=(50.0, 50.0, 50.0), asset=barrel_asset),
        actor("Barrel_B", 100.0, extent=(50.0, 50.0, 50.0), asset=barrel_asset),
    ]

    metrics = gt_geometry.compare(canonical, canonical)

    assert metrics["candidate_logical_object_count"] == 2
    assert metrics["canonical_logical_object_count"] == 2
    assert metrics["over_fragmented_actor_count"] == 0


def test_vertical_props_remain_distinct_logical_objects():
    lamp_asset = "/Game/ExamplePack/SM_Lamp.SM_Lamp"
    canonical = [
        actor("Lamp_A", 0.0, extent=(30.0, 30.0, 300.0), asset=lamp_asset),
        actor("Lamp_B", 100.0, extent=(30.0, 30.0, 300.0), asset=lamp_asset),
    ]

    metrics = gt_geometry.compare(canonical, canonical)

    assert metrics["candidate_logical_object_count"] == 2
    assert metrics["over_fragmented_actor_count"] == 0


def test_mixed_asset_groups_only_its_modular_subset():
    shared_asset = "/Game/ExamplePack/SM_Mixed.SM_Mixed"
    canonical = [
        actor("Floor_A", -50.0, extent=(50.0, 100.0, 1.0), asset=shared_asset),
        actor("Floor_B", 50.0, extent=(50.0, 100.0, 1.0), asset=shared_asset),
        actor(
            "Tower",
            1000.0,
            extent=(30.0, 30.0, 300.0),
            asset=shared_asset,
        ),
    ]

    metrics = gt_geometry.compare(canonical, canonical)

    assert metrics["candidate_logical_object_count"] == 2
    assert metrics["canonical_logical_object_count"] == 2


def test_class_only_assignment_is_rejected_as_missing_and_extra():
    canonical = [actor("Church", 0.0)]
    candidate = [actor("Barrel", 0.0)]

    metrics = gt_geometry.compare(candidate, canonical)

    assert metrics["matched_logical_object_count"] == 0
    assert metrics["missing_logical_object_count"] == 1
    assert metrics["extra_logical_object_count"] == 1
    assert metrics["aligned_position_rmse_cm"] is None
    assert metrics["global_yaw_error_deg"] is None


def test_pairwise_llm_locator_opens_only_a_candidate_assignment_edge():
    canonical = [
        actor(
            "WaterTower",
            0.0,
            asset="/Game/GT/SM_WaterTower.SM_WaterTower",
        )
    ]
    candidate = [
        actor(
            "WaterTank",
            0.0,
            asset="/Game/Candidate/SM_WaterTank.SM_WaterTank",
        )
    ]
    client = PairwiseLocatorClient()

    result = gt_geometry.compare_detailed(
        candidate,
        canonical,
        identity_locator=LLMPairwiseIdentityLocatorBackend(client),
    )

    assert len(client.calls) == 1
    request = client.calls[0][0][1].content[0]["text"]
    assert "candidate:object:" not in request
    assert "canonical:object:" not in request
    payload = json.loads(request.split(":\n", 1)[1])
    assert payload["canonical_identity_groups"][0][
        "compatible_candidate_group_ids"
    ] == [payload["candidate_identity_groups"][0]["group_id"]]
    assert result["metrics"]["matched_logical_object_count"] == 1
    assert result["audit"]["matches"][0]["identity_source"] == (
        "llm_pairwise_locator"
    )
    locator = result["audit"]["identity_locator"]
    assert locator["role"] == "locator_only_not_semantic_verdict"
    assert locator["usable_edge_count"] == 1


def test_uncertain_pairwise_locator_record_never_opens_assignment_edge():
    canonical = [
        actor("WaterTower", 0.0, asset="/Game/GT/SM_WaterTower.SM_WaterTower")
    ]
    candidate = [
        actor("WaterTank", 0.0, asset="/Game/Candidate/SM_WaterTank.SM_WaterTank")
    ]
    result = gt_geometry.compare_detailed(
        candidate,
        canonical,
        identity_locator=LLMPairwiseIdentityLocatorBackend(
            PairwiseLocatorClient(status="uncertain")
        ),
    )

    assert result["metrics"]["matched_logical_object_count"] == 0
    assert result["audit"]["identity_locator"]["edge_count"] == 1
    assert result["audit"]["identity_locator"]["usable_edge_count"] == 0


def test_pairwise_locator_failure_falls_back_without_fabricating_match():
    canonical = [
        actor("WaterTower", 0.0, asset="/Game/GT/SM_WaterTower.SM_WaterTower")
    ]
    candidate = [
        actor("WaterTank", 0.0, asset="/Game/Candidate/SM_WaterTank.SM_WaterTank")
    ]
    result = gt_geometry.compare_detailed(
        candidate,
        canonical,
        identity_locator=LLMPairwiseIdentityLocatorBackend(
            PairwiseLocatorClient(fail=True)
        ),
    )

    assert result["metrics"]["matched_logical_object_count"] == 0
    assert "pairwise locator unavailable" in result["audit"][
        "identity_locator"
    ]["backend_error"]


def test_exact_asset_identity_skips_pairwise_llm_call():
    canonical = [actor("Church", 0.0)]
    client = PairwiseLocatorClient()

    result = gt_geometry.compare_detailed(
        canonical,
        canonical,
        identity_locator=LLMPairwiseIdentityLocatorBackend(client),
    )

    assert client.calls == []
    assert result["metrics"]["matched_logical_object_count"] == 1
    assert result["audit"]["matches"][0]["identity_source"] == (
        "exact_asset_path"
    )


def test_stable_ids_prevent_repeated_asset_swap_from_hiding_bad_repair():
    shared_asset = "/Game/Test/SM_Repeated.SM_Repeated"
    canonical = [
        actor("Retained", 0.0, asset=shared_asset, stable_id="source-retained"),
        actor("DeletedTarget", 1000.0, asset=shared_asset, stable_id="source-target"),
    ]
    candidate = [
        actor("Retained", 1000.0, asset=shared_asset, stable_id="source-retained"),
        actor("GeneratedRepair", 0.0, asset=shared_asset, stable_id="generated"),
    ]

    result = gt_geometry.compare_detailed(candidate, canonical)

    assert result["metrics"]["matched_logical_object_count"] == 2
    assert result["metrics"]["absolute_location_error_cm"]["max"] == 1000.0
    stable_match = next(
        value
        for value in result["audit"]["actor_matches"]
        if value["candidate"]["stable_actor_ids"] == ["source-retained"]
    )
    assert stable_match["canonical"]["stable_actor_ids"] == ["source-retained"]
    assert stable_match["identity_source"] == "exact_stable_actor_id"
    assert result["audit"]["alignment"]["method"] == (
        "exact_stable_actor_id_least_squares"
    )


def test_stable_id_correspondence_reports_structured_identity_mismatch():
    canonical = [
        actor(
            "Chair",
            0.0,
            asset="/Game/Test/SM_Chair.SM_Chair",
            stable_id="source-target",
        )
    ]
    candidate = [
        actor(
            "Table",
            0.0,
            asset="/Game/Test/SM_Table.SM_Table",
            stable_id="source-target",
        )
    ]

    result = gt_geometry.compare_detailed(candidate, canonical)

    assert result["metrics"]["matched_logical_object_count"] == 1
    assert result["metrics"]["identity_comparison_pair_count"] == 1
    assert result["metrics"]["identity_mismatch_count"] == 1
    assert result["metrics"]["identity_match_rate"] == 0.0
    assert (
        result["audit"]["actor_matches"][0]["structured_identity_match"]
        is False
    )


def test_incompatible_long_tail_pair_is_blocked_before_pairwise_llm_call():
    anchor_asset = "/Game/Shared/SM_Anchor.SM_Anchor"
    canonical = [
        actor("Anchor", 0.0, asset=anchor_asset),
        actor(
            "WaterTower",
            1000.0,
            asset="/Game/GT/SM_WaterTower.SM_WaterTower",
        ),
    ]
    candidate = [
        actor("Anchor", 0.0, asset=anchor_asset),
        actor(
            "WaterTank",
            100000.0,
            asset="/Game/Candidate/SM_WaterTank.SM_WaterTank",
        ),
    ]
    client = PairwiseLocatorClient()

    result = gt_geometry.compare_detailed(
        candidate,
        canonical,
        identity_locator=LLMPairwiseIdentityLocatorBackend(client),
    )

    assert client.calls == []
    locator = result["audit"]["identity_locator"]
    assert locator["long_tail_candidate_object_count"] == 1
    assert locator["long_tail_canonical_object_count"] == 1
    assert locator["compatible_object_pair_count"] == 0
    assert locator["compatible_group_pair_count"] == 0
    assert locator["candidate_group_count"] == 0
    assert locator["canonical_group_count"] == 0
    assert result["metrics"]["matched_logical_object_count"] == 1
    assert result["metrics"]["missing_logical_object_count"] == 1
    assert result["metrics"]["extra_logical_object_count"] == 1


def test_gt_pairwise_retrieval_reuses_configured_vlm_client(monkeypatch):
    client = PairwiseLocatorClient()
    monkeypatch.setattr(vlm_client_module, "tool_client_from_env", lambda: client)

    backend, mode = gt_geometry._identity_locator_options(
        context(
            spec={
                "identity_retrieval": {
                    "mode": "llm",
                    "top_k": 7,
                    "max_tokens": 1536,
                }
            }
        )
    )

    assert isinstance(backend, LLMPairwiseIdentityLocatorBackend)
    assert backend.client is client
    assert backend.top_k == 7
    assert backend.max_tokens == 1536
    assert mode == "llm_pairwise_locator"
    assert client.calls == []


def test_repeated_exact_assets_are_split_into_spatial_assignment_blocks():
    shared_asset = "/Game/ExamplePack/SM_Barrel.SM_Barrel"
    canonical = [
        actor(f"Barrel_{index}", index * 10000.0, asset=shared_asset)
        for index in range(4)
    ]

    result = gt_geometry.compare_detailed(canonical, canonical)
    metrics = result["metrics"]
    assignment = result["audit"]["assignment"]

    assert metrics["matched_logical_object_count"] == 4
    assert metrics["assignment_cost_matrix_cell_count"] == 4
    assert metrics["assignment_cost_matrix_reduction_ratio"] == 0.75
    assert assignment["strategy"] == (
        "identity_size_spatial_blocks_then_bipartite_components"
    )
    assert assignment["largest_assignment_block"] == {
        "candidate_count": 1,
        "canonical_count": 1,
    }


def test_size_incompatible_exact_asset_is_not_forced_into_a_match():
    shared_asset = "/Game/ExamplePack/SM_Block.SM_Block"
    canonical = [
        actor(
            "Canonical_Block",
            0.0,
            extent=(50.0, 50.0, 50.0),
            asset=shared_asset,
        )
    ]
    candidate = [
        actor(
            "Candidate_Block",
            0.0,
            extent=(5000.0, 5000.0, 5000.0),
            asset=shared_asset,
        )
    ]

    metrics = gt_geometry.compare(candidate, canonical)

    assert metrics["matched_logical_object_count"] == 0
    assert metrics["missing_logical_object_count"] == 1
    assert metrics["extra_logical_object_count"] == 1


def test_rotated_exact_asset_world_bounds_are_still_corresponded():
    shared_asset = "/Game/Dungeon/SM_Sword.SM_Sword"
    canonical = [
        actor(
            "CanonicalSword",
            0.0,
            extent=(24.7, 2.8, 27.1),
            asset=shared_asset,
            stable_id="canonical-sword",
        )
    ]
    candidate = [
        actor(
            "RestoredSword",
            0.0,
            extent=(5.5, 49.4, 54.2),
            asset=shared_asset,
            stable_id="generated-sword",
        )
    ]

    result = gt_geometry.compare_detailed(candidate, canonical)

    assert result["metrics"]["matched_actor_count"] == 1
    assert result["audit"]["actor_matches"][0]["identity_source"] == (
        "exact_asset_path"
    )


def test_candidate_declared_logical_ids_cannot_hide_raw_actor_topology():
    canonical = [actor("Church", 0.0), actor("Barrel", 500.0)]
    candidate = [
        actor("Church", 0.0, logical_id="pretend_everything_is_one"),
        actor("Barrel", 500.0, logical_id="pretend_everything_is_one"),
    ]

    result = gt_geometry.compare_detailed(candidate, canonical)

    assert result["metrics"]["candidate_logical_object_count"] == 2
    assert result["audit"]["population"]["candidate"][
        "ignored_untrusted_declared_actor_count"
    ] == 2


def test_scene_graph_export_carries_canonical_bounds_and_class_evidence():
    """GT authoring consumes the same schema as runtime scene capture."""

    script = inventory.scene_graph_script()

    assert "get_actor_bounds(False)" in script
    assert '"bounds": {' in script
    assert '"class": class_path' in script
    assert '"schema_version": "0.3.0"' in script
    assert inventory.SNAPSHOT_EXPORTER.read_text() in script
