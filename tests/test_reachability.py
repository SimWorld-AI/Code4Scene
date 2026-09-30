"""One-property contract tests for ``navigation.reachability``.

Stored payloads, no editor: the probe's output is the input to scoring, so
everything below is CI-safe and a threshold change is re-scored here rather
than re-run against a rebuilt navmesh.
"""

from __future__ import annotations

import pytest

from code4scene.evaluation.verifiers.reachability import (
    MINIMUM_USEFUL_ACTOR_COUNT,
    ReachabilityPolicy,
    ReachabilityVerifier,
    embodied_utility,
    evaluate_reachability,
    walk_validation_rate,
)


def actor_record(
    label: str,
    *,
    eligible: bool = True,
    projected: bool = True,
    connected: bool | None = True,
    ineligible_reason: str | None = None,
    actor_class: str = "StaticMeshActor",
) -> dict:
    return {
        "actor_path": f"/Game/Test.Test:PersistentLevel.{label}",
        "label": label,
        "class": actor_class,
        "eligible": eligible,
        "ineligible_reason": ineligible_reason,
        "projected_point_cm": [0.0, 0.0, 0.0] if projected else None,
        "projection_offset_index": 0 if projected else None,
        "connected": connected if projected else None,
    }


#: One scene measured twice. Left: a pass whose union-find was charged against
#: a 6000-PAIR cap and truncated — 56 components at a 0.475 largest share.
#: Right: the same scene measured without truncation, 52 at 0.495, inside the
#: 49-52 / 0.495-0.535 band that repeated measurements give.
TRUNCATED_SIZES = (95,) + (2,) * 50 + (1,) * 5
RECORDED_SIZES = (99,) + (2,) * 50 + (1,)


def payload(
    records: list[dict],
    *,
    settled: bool = True,
    sample_count: int = 200,
    component_sizes: tuple[int, ...] = (200,),
    navmesh_actor: str | None = "/Game/Test.Test:PersistentLevel.RecastNavMesh_0",
    status: str = "success",
    settle_path: str | None = "witnessed_drain",
    spread_passes: int = 1,
    query_batches: int = 199,
    path_queries: int = 199,
    query_budget: int = 200,
    truncated: bool = False,
    **extra,
) -> dict:
    eligible = [record for record in records if record["eligible"]]
    largest = max(component_sizes) if component_sizes else 0
    settle_evidence = (
        {"witness": {"witnessed_build_tasks": 498, "zero_reads_required": 2,
                     "build_task_readings": [498, 237, 0, 0], "drain_s": 1.8}}
        if settle_path == "witnessed_drain" else {})
    return {
        "schema_version": "0.1.0",
        "measurement_type": "ue_editor_reachability",
        "map_path": "/Game/Test",
        "runtime_provenance": {
            "project_file_path": "/project/Test.uproject",
            "engine_version": "5.8",
            "map_path": "/Game/Test",
        },
        "navmesh": {
            "actor_path": navmesh_actor,
            "class": "RecastNavMesh",
            "agent_radius_requested_cm": 50.0,
            "agent_radius_properties_set": ["nav_data_config.agent_radius"],
            "agent_radius_errors": [],
            "rebuild": [{"method": "NavigationSystemV1.build_navigation",
                         "status": "called"}],
            "parameters_after_rebuild": {
                "agent_radius": 50.0,
                "cell_size": 19.0,
                "tile_size_uu": 1000.0,
            },
            "bounds_min_cm": [-5000.0, -5000.0, -200.0],
            "bounds_max_cm": [5000.0, 5000.0, 800.0],
            "bounds_source": "nav_bounds_volumes",
            "settle": {
                "settled": settled,
                "method": ("witnessed_drain_then_resample_equality"
                           if settle_path == "witnessed_drain" else "resample_equality"),
                "probe_count": 32,
                "wait_s": 5.0,
                "remaining_build_tasks_reported": 0,
                "passes": [{"attempt": 0, "sample_count": 32},
                           {"attempt": 1, "sample_count": 32}],
                **({"path": settle_path} if settle_path is not None else {}),
                **settle_evidence,
            },
        },
        "samples": {
            "seed": 1234,
            "requested_count": 200,
            "accepted_count": sample_count,
            "projection_attempts": 240,
            "projection_extent_cm": 500.0,
            "points_cm": [],
        },
        "components": {
            "count": len(component_sizes),
            "sizes": list(component_sizes),
            "largest_size": largest,
            "largest_representative_cm": [10.0, 20.0, 0.0] if component_sizes else None,
            "path_endpoint_tolerance_cm": 100.0,
            "cost": {"query_batches": query_batches, "path_queries": path_queries,
                     "query_budget": query_budget,
                     "query_budget_unit": "sample_batches",
                     "query_budget_truncated": truncated},
            "spread": [{"settled": settled, "settle_path": settle_path,
                        "count": len(component_sizes), "largest_size": largest}
                       for _ in range(spread_passes)],
        },
        "actors": {record["actor_path"]: record for record in records},
        "diagnostics": {
            "status": status,
            "level_actor_count": len(records),
            "eligible_actor_count": len(eligible),
            "projected_actor_count": sum(
                1 for record in eligible if record["projected_point_cm"]),
            "connected_actor_count": sum(
                1 for record in eligible if record["connected"] is True),
            "unresolved": [],
            "measurement_errors": [],
            "options": {"agent_radius_cm": 50.0, "sample_count": 200},
        },
        **extra,
    }


def evaluate(records: list[dict], *, floor: float = 0.5, **kwargs) -> dict:
    return evaluate_reachability(payload(records, **kwargs), "scene-under-test", floor)


def utility(records: list[dict], *, floor: float = 0.5, **kwargs) -> dict:
    """The scene-level EUS off a stored payload, W read from it as verify() does."""
    body = payload(records, **kwargs)
    verifier = ReachabilityVerifier()
    normalized = verifier.normalize(
        verifier.measure(body, "scene-under-test"),
        ReachabilityPolicy("scene-under-test", floor),
    )
    return embodied_utility(normalized, walk_validation_rate(body))


def populated(count: int, *, unreachable: int = 0) -> list[dict]:
    """A level above the empty-world floor, with `unreachable` of them cut off."""
    return [actor_record(f"A{index}", connected=index >= unreachable)
            for index in range(count)]


# ── the number ────────────────────────────────────────────────────────────

def test_a_fully_reachable_scene_scores_one_and_is_typed():
    result = evaluate([actor_record("A"), actor_record("B")])

    assert result["id"] == "navigation.reachability"
    assert result["instance_id"] == "navigation.reachability:scene-under-test"
    assert result["metric_version"] == "reachability-v1"
    assert result["status"] == "pass" and result["applicable"] is True
    assert result["coverage"] == 1.0 and result["score"] == 1.0
    assert result["dimension"] == (
        "share of placed actors an embodied agent can walk up to")
    assert result["contributes_to_aggregate"] is False
    assert result["raw"]["unreachable_actor_count"] == 0
    assert result["raw"]["agent_radius_cm"] == 50.0
    producer = result["evidence"][0]
    assert producer["evidence_type"] == "reachability_measurement_producer"
    assert producer["schema_version"] == "0.1.0"
    assert producer["runtime_provenance"]["engine_version"] == "5.8"
    assert producer["navmesh_parameters_after_rebuild"]["agent_radius"] == 50.0


def test_the_score_is_connected_over_eligible():
    result = evaluate([
        actor_record("A"),
        actor_record("B"),
        actor_record("C"),
        actor_record("D", connected=False),
    ])

    assert result["status"] == "fail" and result["score"] == 0.75
    assert result["raw"]["eligible_actor_count"] == 4
    assert result["raw"]["connected_actor_count"] == 3
    assert result["raw"]["unreachable_actor_count"] == 1
    assert "1/4" in result["failure_reason"]


def test_an_eligible_actor_that_never_projected_counts_as_unreachable():
    """It is in the denominator: nothing walked up to it, which is the question."""
    result = evaluate([
        actor_record("A"), actor_record("B"), actor_record("C"),
        actor_record("D", projected=False, connected=None),
    ])

    assert result["status"] == "fail" and result["score"] == 0.75
    assert result["raw"]["projected_actor_count"] == 3
    assert result["evidence"][1]["unprojected_actor_paths"] == [
        "/Game/Test.Test:PersistentLevel.D"]


# ── who is in the population ──────────────────────────────────────────────

def test_actors_without_collision_are_not_destinations():
    """The BoxReflectionCapture lesson: excluded, not counted as unreachable."""
    result = evaluate([
        actor_record("Sofa"),
        actor_record("Capture", eligible=False, projected=False, connected=None,
                     ineligible_reason="no_collision_class",
                     actor_class="BoxReflectionCapture"),
        actor_record("KeyLight", eligible=False, projected=False, connected=None,
                     ineligible_reason="no_collision_class",
                     actor_class="DirectionalLight"),
    ])

    assert result["status"] == "pass" and result["score"] == 1.0
    assert result["raw"]["eligible_actor_count"] == 1
    assert result["raw"]["ineligible_actor_count"] == 2
    assert result["evidence"][1]["ineligible_reasons"] == ["no_collision_class"]


def test_a_level_with_no_collidable_actor_is_not_applicable():
    result = evaluate([
        actor_record("Capture", eligible=False, projected=False, connected=None,
                     ineligible_reason="no_collision_class"),
    ])

    assert result["applicable"] is False and result["status"] == "not_applicable"
    assert result["score"] is None


# ── the three withholding conditions: absent, not zero ────────────────────

def test_an_empty_navmesh_withholds_the_score():
    result = evaluate(
        [actor_record("A", connected=False), actor_record("B", connected=False)],
        sample_count=0, component_sizes=(),
    )

    assert result["status"] == "not_evaluated" and result["score"] is None
    assert "empty or degenerate" in result["failure_reason"]


def test_a_missing_navmesh_actor_withholds_the_score():
    result = evaluate([actor_record("A", connected=False)], navmesh_actor=None)

    assert result["status"] == "not_evaluated" and result["score"] is None
    assert "unbuilt navmesh is not an unreachable scene" in result["failure_reason"]


def test_an_unsettled_rebuild_withholds_the_score():
    result = evaluate([actor_record("A"), actor_record("B")], settled=False)

    assert result["status"] == "not_evaluated" and result["score"] is None
    assert "never settled" in result["failure_reason"]


def test_a_scene_whose_actors_mostly_do_not_project_withholds_the_score():
    """The multi-level guard: a navmesh on one floor of three is not evidence."""
    result = evaluate([
        actor_record("A"),
        actor_record("B", projected=False, connected=None),
        actor_record("C", projected=False, connected=None),
    ])

    assert result["status"] == "not_evaluated" and result["score"] is None
    assert "1/3" in result["failure_reason"]
    assert result["raw"]["projected_actor_count"] == 1


def test_the_projection_floor_is_inclusive():
    """Exactly at the floor still scores — only strictly below withholds."""
    result = evaluate([
        actor_record("A"),
        actor_record("B", projected=False, connected=None),
    ])

    assert result["status"] == "fail" and result["score"] == 0.5


def test_a_projected_actor_with_no_path_answer_withholds_rather_than_guesses():
    result = evaluate([actor_record("A"), actor_record("B", connected=None)])

    assert result["status"] == "not_evaluated" and result["score"] is None
    assert result["coverage"] == 0.5
    assert "no path-query answer" in result["failure_reason"]


@pytest.mark.parametrize("bad", [
    {"schema_version": "0.9.0"},
    {"measurement_type": "ue_editor_actor_physics"},
    {"actors": []},
])
def test_a_payload_that_is_not_this_probes_output_withholds(bad):
    document = payload([actor_record("A")])
    document.update(bad)

    result = evaluate_reachability(document, "scene-under-test")

    assert result["status"] == "not_evaluated" and result["score"] is None
    assert "invalid reachability evidence" in result["failure_reason"]


def test_a_probe_that_reported_an_error_withholds():
    document = payload([actor_record("A")], status="error")
    document["diagnostics"]["error"] = "UNavigationSystemV1 is not available"

    result = evaluate_reachability(document, "scene-under-test")

    assert result["status"] == "not_evaluated" and result["score"] is None
    assert "UNavigationSystemV1" in result["failure_reason"]


# ── the certification split: which settle ran, and what it may claim ──────

def test_the_result_records_which_settle_path_produced_the_navmesh():
    result = evaluate([actor_record("A")])

    assert result["raw"]["settle_path"] == "witnessed_drain"
    settle = result["evidence"][0]["navmesh_settle"]
    assert settle["path"] == "witnessed_drain"
    assert settle["witness"]["witnessed_build_tasks"] == 498, (
        "the pre-wait build-task reading is the evidence a rebuild started; "
        "without it on the payload nobody can tell the fast settle from a guess")


def test_the_fast_settle_scores_but_withholds_the_component_structure():
    """Certified for the share, not certified for the connectivity numbers."""
    result = evaluate([actor_record("A"), actor_record("B")],
                      component_sizes=(120, 80))

    assert result["status"] == "pass" and result["score"] == 1.0
    assert result["raw"]["component_count"] is None
    assert result["raw"]["largest_component_size"] is None
    assert "2.75x" in result["raw"]["component_structure_withheld_reason"]


def test_the_conservative_settle_reports_the_component_structure():
    result = evaluate([actor_record("A"), actor_record("B")],
                      component_sizes=(120, 80), settle_path="conservative")

    assert result["status"] == "pass" and result["score"] == 1.0
    assert result["raw"]["component_count"] == 2
    assert result["raw"]["largest_component_size"] == 120
    assert result["raw"]["component_structure_withheld_reason"] is None


def test_a_multi_pass_spread_lets_a_fast_pass_report_its_components():
    """The other way to earn it: a spread instead of a bare point estimate."""
    result = evaluate([actor_record("A")], component_sizes=(120, 80),
                      spread_passes=3)

    assert result["raw"]["settle_path"] == "witnessed_drain"
    assert result["raw"]["settled_component_passes"] == 3
    assert result["raw"]["component_count"] == 2
    assert len(result["evidence"][0]["component_spread"]) == 3


def test_a_payload_from_before_the_fast_path_reads_as_the_conservative_one():
    """A 0.1.0 payload still scores: no `path` on its settle, because the
    conservative one was the only settle there was when it was written."""
    document = payload([actor_record("A")], component_sizes=(120, 80),
                       settle_path=None)
    document["schema_version"] = "0.1.0"
    document["components"].pop("cost")
    document["components"].pop("spread")

    result = evaluate_reachability(document, "scene-under-test")

    assert result["status"] == "pass" and result["score"] == 1.0
    assert result["raw"]["settle_path"] == "conservative"
    assert result["raw"]["component_count"] == 2


# ── the query budget, and the truncation it used to allow ─────────────────

def test_the_query_budget_is_denominated_in_batches_and_not_in_pairs():
    """A truncated union-find is a different answer, not a slower one.

    A batch of path queries has no early exit — a sample is tested against
    every component even after one answers yes — so the PAIRS it spends run
    2-3x what a sequential loop spends (5576 against 2277 on this scene).
    Charged against the 6000-pair cap that was right when a pair WAS a round
    trip, the loop stopped partway through the samples and manufactured
    components out of the ones it never tested: 56 components at a 0.475
    largest share, against the 49-52 at 0.495-0.535 that repeated measurements
    give. Truncation is not a slower answer, it is a different one, so the
    score goes with it.

    Budgeting batches instead — one per sample — makes truncation structurally
    impossible at the default however many pairs the batches cost.
    """
    truncated = evaluate(
        [actor_record("A"), actor_record("B")], component_sizes=TRUNCATED_SIZES,
        query_batches=176, path_queries=6000, query_budget=6000, truncated=True)

    assert truncated["status"] == "not_evaluated" and truncated["score"] is None
    assert "truncat" in truncated["failure_reason"]
    assert truncated["raw"]["component_query_budget_truncated"] is True
    assert truncated["raw"]["component_count"] is None

    intact = evaluate(
        [actor_record("A"), actor_record("B")], component_sizes=RECORDED_SIZES,
        settle_path="conservative",
        query_batches=199, path_queries=5576, query_budget=200)

    assert intact["status"] == "pass" and intact["score"] == 1.0
    assert intact["raw"]["component_count"] == 52
    cost = intact["evidence"][0]["component_query_cost"]
    assert cost["query_budget_unit"] == "sample_batches"
    assert cost["query_batches"] == 199 and cost["query_batches"] < cost["query_budget"]
    assert cost["path_queries"] > cost["query_budget"], (
        "the pairs outrun a budget the batches never reach — which is the whole "
        "reason the budget counts batches")


# ── measure and normalize stay apart ──────────────────────────────────────

def test_a_stored_measurement_is_rescored_without_touching_an_editor():
    """The atomic split, checked: one measure(), two policies, two answers."""
    verifier = ReachabilityVerifier()
    measurement = verifier.measure(
        payload([actor_record("A"), actor_record("B", projected=False, connected=None)]),
        "scene-under-test",
    )

    lenient = verifier.normalize(measurement, ReachabilityPolicy("scene-under-test", 0.5))
    strict = verifier.normalize(measurement, ReachabilityPolicy("scene-under-test", 0.9))

    assert measurement.status == "measured" and measurement.raw is not None
    assert lenient.status == "fail" and lenient.score == 0.5
    assert strict.status == "not_evaluated" and strict.score is None
    assert strict.normalization_parameters == {"minimum_projection_fraction": 0.9}


def test_normalize_refuses_a_measurement_from_another_scene():
    verifier = ReachabilityVerifier()
    measurement = verifier.measure(payload([actor_record("A")]), "scene-under-test")

    with pytest.raises(ValueError):
        verifier.normalize(measurement, ReachabilityPolicy("another-scene"))


def test_the_policy_refuses_a_floor_outside_the_unit_interval():
    with pytest.raises(ValueError):
        ReachabilityPolicy("scene-under-test", 1.5)


# ── the embodied utility score: EUS = A x W ───────────────────────────────

def test_the_utility_is_the_approachability_share_times_the_walk_rate():
    result = utility(populated(5, unreachable=2),
                     walk_validation={"completed_route_fraction": 0.5})

    assert result["basis"] == "approachability_x_walk_validation"
    assert result["walk_validation_rate"] == 0.5
    assert result["embodied_utility_score"] == pytest.approx(0.6 * 0.5)


def test_a_missing_walk_rate_is_labelled_provisional_and_never_read_as_one():
    """No walker ran, so the number is A and says so. A silent W=1.0 would
    publish an unwalked scene at the same value as a walked one."""
    result = utility(populated(5, unreachable=2))

    assert result["status"] == "scored"
    assert result["basis"] == "approachability_only"
    assert result["walk_validation_rate"] is None
    assert result["embodied_utility_score"] == pytest.approx(0.6)
    assert "walk-validation pending" in result["note"]


def test_an_unreadable_walk_rate_is_refused_rather_than_dropped():
    """Dropping it would read as "no walker ran" and score HIGHER than the bad
    walk it actually described."""
    result = utility(populated(5), walk_validation={"completed_route_fraction": 1.5})

    assert result["status"] == "absent"
    assert result["embodied_utility_score"] is None


# ── the empty-world rule: zero, and scored ────────────────────────────────

def test_a_level_with_nothing_in_it_scores_zero_rather_than_withholding():
    """Utility requires something to use. Not applicable to the SHARE, which is
    a share of nothing; a hard zero for the UTILITY, which is a fact."""
    result = utility([actor_record("Decor", eligible=False,
                                   ineligible_reason="no_collision")])

    assert result["status"] == "scored"
    assert result["embodied_utility_score"] == 0.0
    assert result["basis"] == "empty_world"


def test_the_empty_world_floor_is_exclusive_and_the_first_useful_scene_scores():
    """The boundary, both sides, off the constant rather than a literal 5."""
    below = utility(populated(MINIMUM_USEFUL_ACTOR_COUNT - 1))
    at = utility(populated(MINIMUM_USEFUL_ACTOR_COUNT))

    assert below["embodied_utility_score"] == 0.0 and below["basis"] == "empty_world"
    assert at["basis"] == "approachability_only"
    assert at["embodied_utility_score"] == pytest.approx(1.0)


def test_an_empty_world_stays_zero_even_when_the_measurement_failed():
    """How many actors a level holds is a fact about the SCENE. No navmesh
    failure can add some, so the rule runs ahead of every withholding."""
    result = utility(populated(2), settled=False)

    assert result["embodied_utility_score"] == 0.0
    assert result["basis"] == "empty_world"


# ── the absent line: withheld, never zero ─────────────────────────────────

@pytest.mark.parametrize("kwargs,condition", [
    ({"navmesh_actor": None}, "a navmesh that never built"),
    ({"sample_count": 0, "component_sizes": ()}, "a degenerate navmesh"),
    ({"settled": False}, "a rebuild that never settled"),
    ({"truncated": True}, "a union-find that truncated on its budget"),
])
def test_a_measurement_failure_withholds_the_utility(kwargs, condition):
    result = utility(populated(5), **kwargs)

    assert result["embodied_utility_score"] is None, condition
    assert result["status"] == "absent" and result["basis"] == "withheld"
    assert result["note"], "a withheld cell has to say which condition fired"


def test_a_scene_below_the_projection_floor_withholds_the_utility():
    """The multi-level case: the navmesh covers one floor and the actors stand
    on two. We could not see, which is not the same as nothing being there."""
    records = populated(2) + [actor_record(f"U{index}", projected=False, connected=None)
                              for index in range(3)]

    result = utility(records)

    assert result["embodied_utility_score"] is None
    assert result["status"] == "absent"


def test_the_two_zeroes_are_distinguishable_without_opening_the_payload():
    """The line the whole metric turns on, asserted as one comparison."""
    nothing_there = utility(populated(1))
    could_not_see = utility(populated(5), settled=False)

    assert nothing_there["embodied_utility_score"] == 0.0
    assert could_not_see["embodied_utility_score"] is None
    assert nothing_there["basis"] != could_not_see["basis"]
