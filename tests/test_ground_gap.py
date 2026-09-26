"""One-property contract tests for ``physics.ground_gap``."""

from __future__ import annotations

import math

import pytest

from code4scene.evaluation.contracts import MetricResult, RawMeasurement
from code4scene.evaluation.verifiers.ground_gap import (
    GroundGapEvidence,
    GroundGapPolicy,
    GroundGapVerifier,
    evaluate_ground_gap,
)


def actor(label: str, *, actor_path: str | None = None) -> dict:
    return {
        "label": label,
        "actor_path": actor_path or f"/Game/Test.Test:PersistentLevel.{label}",
        "class": "/Script/Engine.StaticMeshActor",
        "actor_tags": [],
        "transform": {
            "location_cm": [0.0, 0.0, 0.0],
            "rotation_deg": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "bounds": {
            "origin_cm": [0.0, 0.0, 50.0],
            "extent_cm": [25.0, 25.0, 50.0],
        },
        "properties": {},
    }


def candidate_scene(actors: list[dict], **extra) -> dict:
    return {"actor_count": len(actors), "actors": actors, **extra}


def measurement(
    gap: float | None,
    *,
    surface: bool = True,
    method: str | None = "ue_lod0_vertex_terrain_trace",
    **other,
) -> dict:
    return {
        "surface_detected": surface,
        "ground_gap_cm": gap,
        "measurement_method": method,
        **other,
    }


def physics_case(
    *,
    labels: list[str] | None = None,
    maximum: float = 5.0,
    scope: str = "candidate_all",
) -> dict:
    selector = {"scope": scope}
    if labels is not None:
        selector["labels"] = labels
    return {
        "schema_version": "0.2.0",
        "case_id": "atomic-ground-gap-test",
        "assertions": [{
            "id": "physics-grounded",
            "primitive": "physics",
            "scope": scope,
            "target_selector": selector,
            "maximum_ground_gap_cm": maximum,
            # These belong to other future metrics.
            "maximum_penetration_cm": 5.0,
            "minimum_support_fraction": 0.5,
        }],
    }


def measurement_payload(measured: dict) -> dict:
    return {
        "schema_version": "0.4.0",
        "measurement_type": "ue_editor_actor_physics",
        "map_path": "/Game/Test",
        "capabilities": {"solid_penetration": {
            "broad_phase": "actor_world_aabb",
            "overlap_method": "ue_component_overlap_components",
            "depth_method": "ue_fhitresult_initial_overlap_mtd",
            "aabb_role": "broad_phase_only",
        }},
        "runtime_provenance": {
            "project_file_path": "/project/Test.uproject",
            "engine_version": "5.8",
            "map_path": "/Game/Test",
        },
        "actors": measured,
        "diagnostics": {
            "status": "success",
            "requested_actor_count": len(measured),
            "measured_actor_count": len(measured),
            "unresolved_targets": [],
            "measurement_errors": [],
        },
    }


def evaluate(actors: list[dict], measured: dict, case: dict | None = None) -> dict:
    return evaluate_ground_gap(
        candidate_scene(actors),
        measurement_payload(measured),
        case or physics_case(),
    )[0]


def test_complete_ground_gap_pass_is_typed_and_versioned():
    result = evaluate([actor("A")], {"A": measurement(0.0)})

    assert result["id"] == "physics.ground_gap"
    assert result["instance_id"] == "physics.ground_gap:physics-grounded"
    assert result["metric_version"] == "ground-gap-v1"
    assert result["status"] == "pass" and result["applicable"] is True
    assert result["coverage"] == 1.0 and result["score"] == 1.0
    assert result["raw"] == {
        "unit": "cm",
        "target_actor_count": 1,
        "observed_actor_count": 1,
        "mean": 0.0,
        "p95": 0.0,
        "max": 0.0,
        "affected_actor_count": 0,
        "affected_actor_fraction": 0.0,
    }
    assert result["contributes_to_aggregate"] is False
    assert result["calibration_status"] == "not_human_validated_diagnostic"
    assert result["dimension"] == "actor-to-ground surface separation"
    assert result["required_evidence"] == [
        "candidate actor population",
        "surface_detected",
        "ground_gap_cm",
        "measurement_method",
        "physics measurement provenance",
    ]
    producer = result["evidence"][0]
    assert producer["evidence_type"] == "physics_measurement_producer"
    assert producer["schema_version"] == "0.4.0"
    assert producer["runtime_provenance"]["engine_version"] == "5.8"


def test_ground_gap_score_preserves_continuous_severity():
    result = evaluate(
        [actor("A"), actor("B")],
        {"A": measurement(0.0), "B": measurement(50.0)},
    )

    assert result["status"] == "fail" and result["score"] == 0.5
    assert result["raw"] == {
        "unit": "cm",
        "target_actor_count": 2,
        "observed_actor_count": 2,
        "mean": 25.0,
        "p95": 50.0,
        "max": 50.0,
        "affected_actor_count": 1,
        "affected_actor_fraction": 0.5,
    }
    assert "1/2 target actor(s) exceed 5 cm" in result["failure_reason"]


def test_gap_equal_to_tolerance_is_not_counted_as_affected():
    result = evaluate([actor("A")], {"A": measurement(5.0)})

    assert result["status"] == "pass"
    assert result["raw"]["affected_actor_count"] == 0


def test_missing_actor_withholds_normalization_instead_of_imputing_zero():
    result = evaluate(
        [actor("A"), actor("B")],
        {"A": measurement(2.0)},
    )

    assert result["status"] == "not_evaluated"
    assert result["coverage"] == 0.5 and result["score"] is None
    assert result["raw"]["mean"] == 2.0
    assert result["raw"]["affected_actor_count"] is None
    assert result["raw"]["affected_actor_fraction"] is None
    missing = [item for item in result["evidence"] if item.get("missing_reason")]
    assert missing == [{
        "actor_id": "/Game/Test.Test:PersistentLevel.B",
        "label": "B",
        "surface_detected": None,
        "ground_gap_cm": None,
        "measurement_method": None,
        "measurement_key": None,
        "missing_reason": "measurement_missing",
    }]


def test_no_detected_surface_is_missing_evidence_not_a_zero_gap():
    result = evaluate([actor("A")], {"A": measurement(None, surface=False)})

    assert result["status"] == "not_evaluated"
    assert result["coverage"] == 0.0 and result["score"] is None
    assert result["raw"]["mean"] is None
    assert result["evidence"][-1]["missing_reason"] == "surface_not_detected"


def test_candidate_all_keeps_the_original_unfiltered_population():
    crate = actor("GroundedCrate")
    floor = actor("SB_Ground")
    floor["bounds"] = {
        "origin_cm": [0.0, 0.0, -50.0],
        "extent_cm": [1000.0, 1000.0, 50.0],
    }
    light = actor("KeyLight")
    light["class"] = "/Script/Engine.DirectionalLight"
    result = evaluate(
        [floor, crate, light],
        {
            "SB_Ground": measurement(None, surface=False),
            "GroundedCrate": measurement(0.0),
            "KeyLight": measurement(100.0),
        },
    )

    assert result["status"] == "not_evaluated"
    assert result["raw"]["target_actor_count"] == 3
    assert result["coverage"] == pytest.approx(2.0 / 3.0)


def test_candidate_all_does_not_semantically_filter_named_ground_meshes():
    square = actor("Wet_Cobblestone_Square")
    square["bounds"] = {
        "origin_cm": [0.0, 0.0, 2.0],
        "extent_cm": [6500.0, 6500.0, 0.0],
    }
    barrel = actor("Barrel")
    result = evaluate(
        [square, barrel],
        {
            "Wet_Cobblestone_Square": measurement(None, surface=False),
            "Barrel": measurement(0.0),
        },
    )

    assert result["status"] == "not_evaluated"
    assert result["raw"]["target_actor_count"] == 2
    assert result["coverage"] == 0.5


def test_candidate_all_with_only_infrastructure_remains_declared_but_unmeasured():
    floor = actor("Floor")
    floor["bounds"]["extent_cm"] = [1000.0, 1000.0, 10.0]
    result = evaluate(
        [floor],
        {"Floor": measurement(None, surface=False)},
    )

    assert result["status"] == "not_evaluated"
    assert result["applicable"] is True
    assert result["coverage"] == 0.0
    assert result["score"] is None
    assert result["evidence"][-1]["missing_reason"] == "surface_not_detected"


def test_applicability_is_explicit_for_unconfigured_and_empty_populations():
    unconfigured = evaluate_ground_gap(
        candidate_scene([actor("A")]),
        measurement_payload({"A": measurement(0.0)}),
        {
            "schema_version": "0.2.0",
            "case_id": "atomic-ground-gap-test",
            "assertions": [{
                "id": "geometry-only",
                "primitive": "no_overlap",
                "scope": "candidate_all",
            }],
        },
    )[0]
    empty = evaluate(
        [actor("A")],
        {"A": measurement(0.0)},
        physics_case(labels=["absent"]),
    )

    for result in (unconfigured, empty):
        assert result["status"] == "not_applicable"
        assert result["applicable"] is False
        assert result["coverage"] is None and result["score"] is None
    assert unconfigured["instance_id"].endswith(":not-configured")
    assert empty["instance_id"].endswith(":physics-grounded")
    assert unconfigured["evidence"] == [{
        "applicability_reason": "physics_assertion_not_configured",
    }]
    assert empty["evidence"] == [{
        "applicability_reason": "selector_matched_no_candidate_actors",
    }]


def test_penetration_support_and_grounded_flags_cannot_change_this_property():
    independent = evaluate(
        [actor("A")],
        {"A": measurement(
            0.0,
            grounded=False,
            penetration_cm=500.0,
            support_fraction=0.0,
        )},
    )
    contradictory_legacy_flag = evaluate(
        [actor("A")],
        {"A": measurement(
            50.0,
            grounded=True,
            penetration_cm=0.0,
            support_fraction=1.0,
        )},
    )

    assert independent["status"] == "pass"
    assert contradictory_legacy_flag["status"] == "fail"
    assert independent["score"] == 1.0
    assert contradictory_legacy_flag["score"] == 0.0


def test_duplicate_label_fallback_fails_closed_instead_of_reusing_evidence():
    actors = [
        actor("Same", actor_path="/Game/Test.Test:PersistentLevel.Same_1"),
        actor("Same", actor_path="/Game/Test.Test:PersistentLevel.Same_2"),
    ]
    result = evaluate(
        actors,
        {"Same": measurement(0.0)},
        physics_case(labels=["Same"]),
    )

    assert result["status"] == "not_evaluated" and result["coverage"] == 0.0
    assert [item["missing_reason"] for item in result["evidence"]
            if item.get("missing_reason")] == [
        "ambiguous_label", "ambiguous_label",
    ]


def test_nearest_rank_p95_is_frozen_by_metric_version():
    actors = [actor(f"A{index}") for index in range(1, 21)]
    measured = {
        f"A{index}": measurement(float(index)) for index in range(1, 21)
    }
    result = evaluate(actors, measured, physics_case(maximum=100.0))

    assert result["status"] == "pass"
    assert result["raw"]["mean"] == 10.5
    assert result["raw"]["p95"] == 19.0
    assert result["raw"]["max"] == 20.0


def test_unsupported_scope_is_visible_not_silently_widened():
    result = evaluate(
        [actor("A")],
        {"A": measurement(0.0)},
        physics_case(scope="primary_additions"),
    )

    assert result["status"] == "not_evaluated"
    assert result["applicable"] is True
    assert result["coverage"] is None and result["raw"] is None
    assert result["evidence"] == [{"unsupported_scope": "primary_additions"}]


def test_atomic_contract_rejects_non_json_finite_values():
    with pytest.raises(ValueError, match="non-finite"):
        MetricResult(
            id="physics.ground_gap",
            instance_id="physics.ground_gap:test",
            metric_version="ground-gap-v1",
            dimension="actor-to-ground surface separation",
            applicability_policy="test",
            required_evidence=("ground_gap_cm",),
            applicable=True,
            status="pass",
            coverage=1.0,
            raw={"max": math.nan},
            score=1.0,
            normalization_policy="test",
            normalization_parameters={},
            calibration_status="diagnostic",
            contributes_to_aggregate=False,
        )


def test_invalid_measurement_envelope_cannot_produce_a_pass():
    payload = measurement_payload({"A": measurement(0.0)})
    payload.update({
        "schema_version": "99.0.0",
        "measurement_type": "untrusted_numbers",
        "map_path": "/Game/Wrong",
    })
    payload["diagnostics"].update({
        "status": "error",
        "measurement_errors": [{"actor": "A", "error": "probe failed"}],
    })
    result = evaluate_ground_gap(
        candidate_scene([actor("A")], map_path="/Game/Test"),
        payload,
        physics_case(),
    )[0]

    assert result["status"] == "not_evaluated"
    assert result["coverage"] is None and result["score"] is None
    errors = result["evidence"][0]["evidence_validation_errors"]
    assert any("schema_version" in error for error in errors)
    assert any("measurement_type" in error for error in errors)
    assert any("diagnostics.status" in error for error in errors)
    assert any("candidate map_path" in error for error in errors)


def test_pre_native_mtd_evidence_schema_is_rejected():
    payload = measurement_payload({"A": measurement(0.0)})
    payload["schema_version"] = "0.3.0"

    result = evaluate_ground_gap(
        candidate_scene([actor("A")], map_path="/Game/Test"),
        payload,
        physics_case(),
    )[0]

    assert result["status"] == "not_evaluated"
    assert any(
        "schema_version must be a 0.4.x or 0.5.x string" in error
        for error in result["evidence"][0]["evidence_validation_errors"]
    )


def test_native_mtd_capability_declaration_is_required():
    payload = measurement_payload({"A": measurement(0.0)})
    payload["capabilities"] = {}

    result = evaluate_ground_gap(
        candidate_scene([actor("A")], map_path="/Game/Test"),
        payload,
        physics_case(),
    )[0]

    assert result["status"] == "not_evaluated"
    assert any(
        "capabilities.solid_penetration" in error
        for error in result["evidence"][0]["evidence_validation_errors"]
    )


def test_conflicting_strong_actor_identity_fails_closed():
    result = evaluate(
        [actor("A")],
        {"A": measurement(
            0.0,
            actor_path="/Game/Test.Test:PersistentLevel.DefinitelyNotA",
        )},
    )

    assert result["status"] == "not_evaluated" and result["coverage"] == 0.0
    assert result["evidence"][-1]["missing_reason"] == "conflicting_strong_identity"


def test_malformed_selector_cannot_fail_open_to_all_actors():
    case = physics_case()
    case["assertions"][0]["target_selector"]["labels"] = "DefinitelyAbsent"

    result = evaluate_ground_gap(
        candidate_scene([actor("A")]),
        measurement_payload({"A": measurement(0.0)}),
        case,
    )[0]

    assert result["instance_id"].endswith(":invalid-contract")
    assert result["status"] == "not_evaluated" and result["score"] is None
    assert "labels must be an array" in result["failure_reason"]


def test_duplicate_assertion_ids_invalidate_the_case_before_measurement():
    case = physics_case()
    case["assertions"].append({**case["assertions"][0]})

    results = evaluate_ground_gap(
        candidate_scene([actor("A")]),
        measurement_payload({"A": measurement(0.0)}),
        case,
    )

    assert len(results) == 1
    assert results[0]["status"] == "not_evaluated"
    assert results[0]["score"] is None
    assert "duplicate assertion id" in results[0]["failure_reason"]


def test_invalid_candidate_scene_cannot_produce_an_atomic_pass():
    broken = actor("A")
    broken.pop("bounds")

    result = evaluate_ground_gap(
        candidate_scene([broken]),
        measurement_payload({"A": measurement(0.0)}),
        physics_case(),
    )[0]

    assert result["status"] == "not_evaluated" and result["score"] is None
    assert "candidate.actors[0].bounds is required" in result["failure_reason"]


def test_independent_candidate_rejects_non_independent_measurements():
    result = evaluate_ground_gap(
        candidate_scene([actor("A")]),
        measurement_payload({"A": measurement(0.0)}),
        physics_case(),
        {
            "independent_scoring_editor": True,
            "measurements_from_independent_editor": False,
        },
    )[0]

    assert result["status"] == "not_evaluated" and result["score"] is None
    assert "same independent scoring editor" in result["failure_reason"]


def test_successful_producer_counts_must_describe_complete_collection():
    payload = measurement_payload({"A": measurement(0.0)})
    payload["diagnostics"]["requested_actor_count"] = 0

    result = evaluate_ground_gap(
        candidate_scene([actor("A")]), payload, physics_case(),
    )[0]

    assert result["status"] == "not_evaluated" and result["score"] is None
    assert "equal requested_actor_count" in result["failure_reason"]


@pytest.mark.parametrize("invalid", [True, None, "5", math.nan])
def test_invalid_threshold_is_dataset_policy_qa_not_a_model_score(invalid):
    case = physics_case()
    case["assertions"][0]["maximum_ground_gap_cm"] = invalid
    result = evaluate([actor("A")], {"A": measurement(0.0)}, case)

    assert result["status"] == "not_evaluated"
    assert result["score"] is None and result["contributes_to_aggregate"] is False
    assert result["normalization_policy"] is None
    assert "invalid ground-gap normalization policy" in result["failure_reason"]


def test_normalizer_refuses_a_policy_from_another_requirement():
    verifier = GroundGapVerifier()
    raw = verifier.measure(GroundGapEvidence(
        requirement_id="requirement-a",
        target_actor_ids=(),
        actors=(),
        missing=(),
    ))

    with pytest.raises(ValueError, match="requirement ids do not match"):
        verifier.normalize(raw, GroundGapPolicy("requirement-b", 5.0))


def test_each_physics_assertion_gets_a_distinct_atomic_instance():
    case = physics_case(maximum=5.0)
    second = {**case["assertions"][0], "id": "lenient", "maximum_ground_gap_cm": 100.0}
    case["assertions"].append(second)
    results = evaluate_ground_gap(
        candidate_scene([actor("A")]),
        measurement_payload({"A": measurement(50.0)}),
        case,
    )

    assert [item["instance_id"] for item in results] == [
        "physics.ground_gap:physics-grounded",
        "physics.ground_gap:lenient",
    ]
    assert [item["status"] for item in results] == ["fail", "pass"]
    assert [item["score"] for item in results] == [0.0, 1.0]


def test_atomic_contract_rejects_internally_inconsistent_states():
    with pytest.raises(ValueError, match="complete coverage"):
        MetricResult(
            id="physics.ground_gap",
            instance_id="physics.ground_gap:test",
            metric_version="ground-gap-v1",
            dimension="gap",
            applicability_policy="configured",
            required_evidence=("gap",),
            applicable=True,
            status="pass",
            coverage=None,
            raw=None,
            score=None,
            normalization_policy=None,
            normalization_parameters={},
            calibration_status="diagnostic",
            contributes_to_aggregate=False,
        )
    with pytest.raises(ValueError, match="raw evidence"):
        RawMeasurement(
            id="physics.ground_gap",
            instance_id="physics.ground_gap:test",
            metric_version="ground-gap-v1",
            applicable=True,
            status="measured",
            coverage=1.0,
            raw=None,
        )
