"""Regression coverage for locator-only actor focus and camera health."""

from __future__ import annotations
import json

from inspect import signature
from types import SimpleNamespace

import numpy as np

from code4scene.evaluation.requirement_graph.actions import generate_scout_poses
from code4scene.evaluation.requirement_graph import pipeline as pipeline_module
from code4scene.evaluation.requirement_graph.actor_inventory import (
    ActorBounds,
    ActorInventorySnapshot,
    build_actor_descriptor,
    estimate_camera_content_floor_z,
)
from code4scene.evaluation.requirement_graph.asset_candidates import (
    assess_bounds_camera_pose,
    plan_bounds_camera_poses,
)
from code4scene.evaluation.requirement_graph.contracts import (
    CameraPose,
    EntityNode,
    PredicateNode,
    ReferentKind,
    RequirementGraph,
    RequirementMemberEdge,
    RequirementNode,
    RootRequirement,
    SceneBounds,
)
from code4scene.evaluation.requirement_graph.existing_llm import (
    LLMResponse,
    ToolCall,
)
from code4scene.evaluation.requirement_graph.llm_identity_locator import LLMIdentityLocatorBackend
from code4scene.evaluation.requirement_graph.semantic_retrieval import (
    CandidateUse,
    EntityIdentityQuery,
    retrieve_scene_identities,
)
from code4scene.evaluation.requirement_graph.stage2_capture import (
    CaptureAttemptStatus,
    cluster_stage2_targets,
    execute_stage2_capture_plan,
    plan_stage2_capture_groups,
    preferred_shot_roles_for_task,
)
from code4scene.evaluation.requirement_graph.stage2 import (
    _evidence_precondition_reason,
    evaluate_stage2_detailed,
)
from code4scene.evaluation.requirement_graph.identity_grounding import (
    resolve_identity_bindings,
)
from code4scene.evaluation.requirement_graph.stage1 import evaluate_stage1
from code4scene.evaluation.requirement_graph.stage2_artifacts import (
    _capture_metadata_payloads,
)
from code4scene.evaluation.requirement_graph.stage2_contracts import (
    CaptureProgram,
    CaptureRequest,
    CaptureShotRole,
    Stage2Budget,
    Stage2CapturePlan,
    Stage2Task,
    Stage2TaskArgument,
    Stage2TaskKind,
)
from code4scene.evaluation.requirement_graph.stage2_frames import FrameStore
from code4scene.evaluation.requirement_graph.stage2_routing import (
    Stage2ActorTarget,
    Stage2EntityRoute,
    Stage2RouteSource,
    Stage2RoutingPlan,
)
from code4scene.evaluation.requirement_graph.runtime import CapturedFrame
from code4scene.evaluation.requirement_graph.stage3_contracts import Stage3Budget
from code4scene.evaluation.requirement_graph.stage3_artifacts import _prepare_store
from code4scene.evaluation.requirement_graph.stage3_explorer import Stage3Explorer
from code4scene.evaluation.requirement_graph.stage3_judge import (
    IdentityDecision,
    IdentityVerdict,
)


SCENE_BOUNDS = SceneBounds((-2000.0, -2000.0, -1000.0), (2000.0, 2000.0, 2000.0))


def _actor(
    actor_id: str,
    asset_name: str,
    center: tuple[float, float, float],
    extent: tuple[float, float, float] = (40.0, 40.0, 40.0),
):
    return build_actor_descriptor(
        live_actor_id=actor_id,
        asset_path=f"/Game/Test/SM_{asset_name}.SM_{asset_name}",
        bounds=ActorBounds(center, extent),
        active=True,
        renderable=True,
        in_current_level=True,
    )


def _rgb(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, 256, size=(24, 32, 3), dtype=np.uint8
    )


def test_broad_alias_is_locator_only_and_cannot_become_an_exact_match():
    ac = _actor("actor-ac", "AC", (0.0, 0.0, 200.0))
    result = retrieve_scene_identities(
        (EntityIdentityQuery("entity-ac", ("AC unit",)),),
        (ac,),
        scene_bounds=SCENE_BOUNDS,
    ).for_query("entity-ac")

    assert result is not None
    assert result.exact_matches == ()
    assert [(value.identity_term, value.matched_query_term) for value in result.locator_candidates] == [
        ("ac", "ac")
    ]
    assert result.locator_candidates[0].use is CandidateUse.LOCATOR_ONLY


class _IdentityLocatorClient:
    name = "identity-locator-test-client"
    model = "identity-locator-test-model"
    max_tokens = 2048
    _strict_tool_calls = True
    _text_action_mode = False

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def chat(self, messages, tools, **kwargs):
        self.calls.append((messages, tools, kwargs))
        if self.fail:
            raise RuntimeError("locator unavailable")
        payload = json.loads(messages[1].content[0]["text"].split(":\n", 1)[1])
        groups = payload["scene_identity_groups"]

        def group_with(field, token):
            return next(
                group["group_id"]
                for group in groups
                if token.casefold() in str(group[field]).casefold()
            )

        downpipe = group_with("asset_terms", "downpipe")
        rollup = group_with("asset_terms", "rollup")
        ordinary_door = next(
            group["group_id"]
            for group in groups
            if "door" in str(group["asset_terms"]).casefold()
            and group["group_id"] != rollup
        )
        return LLMResponse(
            text=None,
            tool_calls=[
                ToolCall(
                    "locator-call",
                    "return_scene_identity_matches",
                    {
                        "schema_version": "1.0",
                        "matches": [
                            {
                                "entity_id": "entity-cable",
                                "candidates": [
                                    {
                                        "group_id": downpipe,
                                        "confidence": 0.15,
                                        "status": "identity_conflict",
                                        "reason": "label says cable but the asset says downpipe",
                                    }
                                ],
                            },
                            {
                                "entity_id": "entity-rollup",
                                "candidates": [
                                    {
                                        "group_id": rollup,
                                        "confidence": 0.92,
                                        "status": "plausible_locator",
                                        "reason": "specific rollup-door asset identity",
                                    },
                                    {
                                        "group_id": ordinary_door,
                                        "confidence": 0.1,
                                        "status": "uncertain",
                                        "reason": "generic door is not a specific rollup door",
                                    },
                                ],
                            },
                        ],
                    },
                )
            ],
            usage={"prompt_tokens": 100},
            raw={},
        )


def test_llm_locator_groups_identities_once_and_maps_opaque_ids_back_to_actors():
    cable = build_actor_descriptor(
        live_actor_id="internal-cable-actor",
        asset_path="/Game/Test/SM_Downpipe3_01.SM_Downpipe3_01",
        actor_label="Cable_X-50_C0",
        bounds=ActorBounds((-500.0, 0.0, 200.0), (40.0, 40.0, 80.0)),
        active=True,
        renderable=True,
        in_current_level=True,
    )
    rollup = build_actor_descriptor(
        live_actor_id="internal-rollup-actor",
        asset_path="/Game/Test/SM_Rollup1_X10.SM_Rollup1_X10",
        actor_label="Rollup1_X10",
        bounds=ActorBounds((0.0, 0.0, 200.0), (80.0, 20.0, 100.0)),
        active=True,
        renderable=True,
        in_current_level=True,
    )
    ordinary = build_actor_descriptor(
        live_actor_id="internal-ordinary-door",
        asset_path="/Game/Test/SM_Enter_T1_DoorT1.SM_Enter_T1_DoorT1",
        actor_label="Door_S0_X-60",
        bounds=ActorBounds((500.0, 0.0, 200.0), (80.0, 20.0, 100.0)),
        active=True,
        renderable=True,
        in_current_level=True,
    )
    client = _IdentityLocatorClient()
    result = LLMIdentityLocatorBackend(client).retrieve_scene_identities(
        (
            EntityIdentityQuery("entity-cable", ("cable", "wire")),
            EntityIdentityQuery(
                "entity-rollup", ("rollup door", "roller shutter")
            ),
        ),
        (cable, rollup, ordinary),
        scene_bounds=SCENE_BOUNDS,
        top_k=3,
    )

    assert len(client.calls) == 1
    request_text = client.calls[0][0][1].content[0]["text"]
    assert "internal-cable-actor" not in request_text
    assert "internal-rollup-actor" not in request_text
    assert result.identity_group_count == 3
    assert len(result.semantic_request_manifest) == 1
    cable_result = result.for_query("entity-cable")
    rollup_result = result.for_query("entity-rollup")
    assert cable_result.locator_candidates[0].actor_ids == (
        "internal-cable-actor",
    )
    assert cable_result.locator_candidates[0].locator_status == "identity_conflict"
    assert cable_result.locator_candidates[0].can_defer_absence is False
    assert rollup_result.locator_candidates[0].actor_ids == (
        "internal-rollup-actor",
    )
    assert rollup_result.locator_candidates[0].locator_status == "plausible_locator"
    assert all(
        candidate.use is CandidateUse.LOCATOR_ONLY
        for candidate in (*cable_result.locator_candidates, *rollup_result.locator_candidates)
    )


def test_llm_locator_failure_falls_back_to_lexical_hints_without_fabrication():
    cable = build_actor_descriptor(
        live_actor_id="internal-cable-actor",
        asset_path="/Game/Test/SM_Downpipe3_01.SM_Downpipe3_01",
        actor_label="Cable_X-50_C0",
        bounds=ActorBounds((0.0, 0.0, 200.0), (40.0, 40.0, 80.0)),
        active=True,
        renderable=True,
        in_current_level=True,
    )
    client = _IdentityLocatorClient(fail=True)
    result = LLMIdentityLocatorBackend(client).retrieve_scene_identities(
        (EntityIdentityQuery("entity-cable", ("cable",)),),
        (cable,),
        scene_bounds=SCENE_BOUNDS,
    )

    assert len(client.calls) == 1
    assert "locator unavailable" in result.semantic_backend_error
    query = result.for_query("entity-cable")
    assert query.exact_matches == ()
    assert query.locator_candidates[0].lexical_relation.value == "query_token_substring"
    assert query.locator_candidates[0].locator_group_id is None



def test_pipeline_llm_retrieval_mode_reuses_the_configured_vlm_client(monkeypatch):
    client = _IdentityLocatorClient()
    monkeypatch.setattr(pipeline_module, "tool_client_from_env", lambda: client)
    context = SimpleNamespace(
        spec={
            "semantic_retrieval": {
                "mode": "llm",
                "top_k": 7,
                "max_tokens": 1536,
            }
        }
    )

    backend, top_k = pipeline_module._retrieval_options(context)

    assert isinstance(backend, LLMIdentityLocatorBackend)
    assert backend.client is client
    assert backend.max_tokens == 1536
    assert top_k == 7
    assert client.calls == []

def test_inventory_floor_prevents_one_low_actor_from_burying_scout_cameras():
    actors = [_actor("low", "Pipe", (0.0, 0.0, -950.0), (20.0, 20.0, 50.0))]
    actors.extend(
        _actor(f"prop-{index}", "Bench", (index * 50.0, 0.0, 50.0))
        for index in range(9)
    )
    inventory = ActorInventorySnapshot(tuple(actors))

    floor_z = estimate_camera_content_floor_z(inventory, SCENE_BOUNDS)
    scouts = generate_scout_poses(SCENE_BOUNDS, floor_z=floor_z)

    assert floor_z == 10.0
    assert {value.z for value in scouts[:4]} == {230.0}


def test_actor_focus_planner_passes_frustum_occupancy_gate():
    bounds = ActorBounds((0.0, 0.0, 200.0), (40.0, 50.0, 80.0))
    planned = plan_bounds_camera_poses(
        bounds.center_cm,
        bounds.extent_cm,
        SCENE_BOUNDS,
    )

    assert all(
        assess_bounds_camera_pose(
            bounds.center_cm,
            bounds.extent_cm,
            pose,
        ).healthy
        for pose in planned
    )
    assert planned[1].pitch <= -12.0
    elevated = plan_bounds_camera_poses(
        bounds.center_cm,
        bounds.extent_cm,
        SCENE_BOUNDS,
        minimum_context_elevation_degrees=12.0,
    )
    assert elevated[0].pitch <= -12.0
    assert elevated[2].pitch <= -12.0
    assert not assess_bounds_camera_pose(
        bounds.center_cm,
        bounds.extent_cm,
        CameraPose(500.0, 0.0, 200.0, 0.0, 0.0),
    ).healthy


def test_count_identity_capture_starts_with_one_overlapping_actor_candidate():
    graph = RequirementGraph(
        prompt="two crossed swords",
        nodes=(
            EntityNode(
                "entity_swords",
                "two crossed swords",
                "Short Sword",
                {"start": 0, "end": 18},
                referent_kind=ReferentKind.COLLECTION,
            ),
            RequirementNode(
                "requirement_swords",
                "two crossed swords",
                {"start": 0, "end": 18},
            ),
        ),
        edges=(
            RequirementMemberEdge(
                "requirement_swords",
                "entity_swords",
                "scored_facet",
                1.0,
            ),
        ),
        roots=(RootRequirement("requirement_swords", 1.0),),
    )
    shared_bounds = ActorBounds((300.0, -75.0, 140.0), (6.0, 50.0, 55.0))
    targets = (
        Stage2ActorTarget("sword-left", shared_bounds),
        Stage2ActorTarget("sword-right", shared_bounds),
    )
    assert len(cluster_stage2_targets(targets, limit=8)) == 1
    assert len(
        cluster_stage2_targets(
            targets,
            limit=8,
            deduplicate_spatial_identity=False,
        )
    ) == 2
    routes = Stage2RoutingPlan(
        (
            Stage2EntityRoute(
                "entity_swords",
                Stage2RouteSource.LOCATOR_ONLY,
                targets,
                stage2_visual_only=True,
            ),
        )
    )
    task = Stage2Task(
        "stage2:requirement_swords",
        "requirement_swords",
        Stage2TaskKind.COUNT,
        1.0,
        arguments=(Stage2TaskArgument("entity_swords", "collection"),),
    )

    plan = plan_stage2_capture_groups(
        graph,
        (task,),
        routes,
        SCENE_BOUNDS,
        budget=Stage2Budget(
            overview_frames=0,
            max_overview_frames=0,
            max_targeted_frames=2,
            max_recovery_frames=0,
            max_valid_frames=2,
            max_evidence_frames_per_task=2,
        ),
    )

    assert plan.programs
    assert [
        request.actor_ids
        for program in plan.programs
        for request in program.requests
    ] == [
        ("sword-left",),
        ("sword-left", "sword-right"),
    ]


def test_locator_only_existence_starts_close_then_adds_context_when_budgeted():
    graph = RequirementGraph(
        prompt="AC unit",
        nodes=(
            EntityNode(
                "entity_ac",
                "AC unit",
                "AC unit",
                {"start": 0, "end": 7},
                referent_kind=ReferentKind.COLLECTION,
            ),
            RequirementNode("requirement_ac", "AC unit", {"start": 0, "end": 7}),
        ),
        edges=(
            RequirementMemberEdge(
                "requirement_ac",
                "entity_ac",
                "scored_facet",
                1.0,
            ),
        ),
        roots=(RootRequirement("requirement_ac", 1.0),),
    )
    target = Stage2ActorTarget(
        "actor-ac",
        ActorBounds((0.0, 0.0, 200.0), (40.0, 50.0, 80.0)),
    )
    routes = Stage2RoutingPlan(
        (
            Stage2EntityRoute(
                "entity_ac",
                Stage2RouteSource.LOCATOR_ONLY,
                (target,),
                stage2_visual_only=True,
            ),
        )
    )
    task = Stage2Task(
        "stage2:entity_ac",
        "entity_ac",
        Stage2TaskKind.OBJECT_EXISTENCE,
        1.0,
        arguments=(Stage2TaskArgument("entity_ac", "subject"),),
    )
    plan = plan_stage2_capture_groups(
        graph,
        (task,),
        routes,
        SCENE_BOUNDS,
        budget=Stage2Budget(
            overview_frames=0,
            max_overview_frames=0,
            max_targeted_frames=1,
            max_recovery_frames=0,
            max_valid_frames=1,
            max_evidence_frames_per_task=1,
            max_targeted_views_per_entity_group=1,
        ),
    )

    request = plan.programs[0].requests[0]
    assessment = assess_bounds_camera_pose(
        target.bounds.center_cm,
        target.bounds.extent_cm,
        request.pose,
    )
    assert request.shot_role is CaptureShotRole.CLOSE
    assert request.actor_ids == ("actor-ac",)
    assert assessment.healthy
    assert assessment.longest_viewport_fraction is not None
    assert 0.45 <= assessment.longest_viewport_fraction <= 0.75

    two_view_plan = plan_stage2_capture_groups(
        graph,
        (task,),
        routes,
        SCENE_BOUNDS,
        budget=Stage2Budget(
            overview_frames=0,
            max_overview_frames=0,
            max_targeted_frames=2,
            max_recovery_frames=0,
            max_valid_frames=2,
            max_evidence_frames_per_task=2,
            max_targeted_views_per_entity_group=2,
        ),
    )
    two_views = two_view_plan.programs[0].requests
    assert [value.shot_role for value in two_views] == [
        CaptureShotRole.CLOSE,
        CaptureShotRole.CONTEXT,
    ]

    unresolved_attribute = Stage2Task(
        "stage2:entity_ac:attribute",
        "entity_ac",
        Stage2TaskKind.ATTRIBUTE,
        1.0,
        arguments=(Stage2TaskArgument("entity_ac", "subject"),),
    )
    attribute_plan = plan_stage2_capture_groups(
        graph,
        (unresolved_attribute,),
        routes,
        SCENE_BOUNDS,
        budget=Stage2Budget(
            overview_frames=0,
            max_overview_frames=0,
            max_targeted_frames=2,
            max_recovery_frames=0,
            max_valid_frames=2,
            max_evidence_frames_per_task=2,
            max_targeted_views_per_entity_group=2,
        ),
    )
    assert [value.shot_role for value in attribute_plan.programs[0].requests] == [
        CaptureShotRole.CLOSE,
        CaptureShotRole.DETAIL,
    ]


def _lazy_identity_fixture():
    graph = RequirementGraph(
        prompt="Add a chair",
        nodes=(
            EntityNode(
                "entity_chair",
                "a chair",
                "chair",
                {"start": 4, "end": 11},
            ),
            RequirementNode(
                "requirement_chair",
                "Add a chair",
                {"start": 0, "end": 11},
            ),
        ),
        edges=(
            RequirementMemberEdge(
                "requirement_chair",
                "entity_chair",
                "scored_facet",
                1.0,
            ),
        ),
        roots=(RootRequirement("requirement_chair", 1.0),),
    )
    inventory = ActorInventorySnapshot(
        (
            _actor("candidate-a", "ChairAlpha", (-200.0, 0.0, 100.0)),
            _actor("candidate-b", "ChairBeta", (200.0, 0.0, 100.0)),
            _actor("candidate-c", "ChairGamma", (0.0, 300.0, 100.0)),
        )
    )
    stage1 = evaluate_stage1(graph, inventory, SCENE_BOUNDS)

    class Provider:
        def __init__(self):
            self.calls = []
            self.resolutions = []

        def resolve_camera_poses(self, poses, *, actor_ids_by_pose):
            requested = tuple(poses)
            self.resolutions.append(
                (requested, tuple(tuple(value) for value in actor_ids_by_pose))
            )
            return requested

        def capture(self, poses, **kwargs):
            requested = tuple(poses)
            self.calls.append((requested, kwargs))
            return tuple(
                CapturedFrame(
                    f"captured-{len(self.calls)}-{index}",
                    pose,
                    _rgb(len(self.calls) * 10 + index),
                )
                for index, pose in enumerate(requested, start=1)
            )

    provider = Provider()
    stage2 = evaluate_stage2_detailed(
        graph,
        stage1,
        inventory,
        SCENE_BOUNDS,
        provider,
        budget=Stage2Budget(
            overview_frames=0,
            max_overview_frames=0,
            max_targeted_frames=1,
            max_recovery_frames=0,
            max_valid_frames=1,
            max_evidence_frames_per_task=1,
            max_targeted_views_per_entity_group=1,
        ),
        locator_top_k=3,
    )
    return graph, stage1, stage2, provider


class _IdentitySequenceJudge:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.call_count = 0
        self.frame_counts = []

    def judge_identity(self, claim, frames):
        del claim
        self.call_count += 1
        self.frame_counts.append(len(frames))
        verdict = self.verdicts.pop(0)
        return IdentityDecision(
            verdict=verdict,
            confidence=0.9 if verdict is not IdentityVerdict.UNKNOWN else 0.0,
            evidence_frame_ids=(frames[-1].frame_id,)
            if verdict is not IdentityVerdict.UNKNOWN
            else (),
            rationale="test identity decision",
            transport_status="success",
            parse_status="valid",
        )


def test_lazy_identity_moves_to_top2_after_top1_mismatch_then_stops():
    graph, stage1, stage2, provider = _lazy_identity_fixture()
    judge = _IdentitySequenceJudge(
        (IdentityVerdict.MISMATCH, IdentityVerdict.MATCH)
    )

    result = resolve_identity_bindings(
        graph,
        stage1,
        stage2,
        SCENE_BOUNDS,
        provider,
        judge,
    )

    route = stage2.routing_plan.for_entity("entity_chair")
    assert route is not None
    binding = result.for_entity("entity_chair")
    assert binding is not None
    assert binding.visual_actor_ids == (route.locator_actor_ids[1],)
    assert [value.actor_id for value in binding.candidate_assessments] == list(
        route.locator_actor_ids[:2]
    )
    assert result.capture_count == 1
    assert result.judge_call_count == 2
    assert judge.frame_counts == [1, 1]


def test_lazy_identity_adds_second_angle_only_after_unknown():
    graph, stage1, stage2, provider = _lazy_identity_fixture()
    judge = _IdentitySequenceJudge((IdentityVerdict.UNKNOWN, IdentityVerdict.MATCH))

    result = resolve_identity_bindings(
        graph,
        stage1,
        stage2,
        SCENE_BOUNDS,
        provider,
        judge,
    )

    binding = result.for_entity("entity_chair")
    assert binding is not None
    assert len(binding.candidate_assessments) == 1
    assert binding.visual_actor_ids == (
        stage2.routing_plan.for_entity("entity_chair").locator_actor_ids[0],
    )
    assert result.capture_count == 1
    assert result.judge_call_count == 2
    assert judge.frame_counts == [1, 2]


def test_object_task_does_not_schedule_generic_overviews_under_default_budget():
    graph = RequirementGraph(
        prompt="a chair",
        nodes=(
            EntityNode("entity_chair", "a chair", "chair", {"start": 0, "end": 7}),
            RequirementNode("requirement_chair", "a chair", {"start": 0, "end": 7}),
        ),
        edges=(
            RequirementMemberEdge(
                "requirement_chair", "entity_chair", "scored_facet", 1.0
            ),
        ),
        roots=(RootRequirement("requirement_chair", 1.0),),
    )
    target = Stage2ActorTarget(
        "actor-chair",
        ActorBounds((0.0, 0.0, 100.0), (45.0, 45.0, 100.0)),
    )
    task = Stage2Task(
        "stage2:chair",
        "entity_chair",
        Stage2TaskKind.OBJECT_EXISTENCE,
        1.0,
        arguments=(Stage2TaskArgument("entity_chair", "subject"),),
    )
    plan = plan_stage2_capture_groups(
        graph,
        (task,),
        Stage2RoutingPlan(
            (
                Stage2EntityRoute(
                    "entity_chair", Stage2RouteSource.STRICT_MATCH, (target,)
                ),
            )
        ),
        SCENE_BOUNDS,
        budget=Stage2Budget.graph_default(),
    )

    assert plan.requests
    assert all(
        request.shot_role is not CaptureShotRole.OVERVIEW
        for request in plan.requests
    )
    assert all(request.task_ids == (task.task_id,) for request in plan.requests)


def test_overviews_are_owned_by_global_and_visual_feature_tasks():
    graph = RequirementGraph(
        prompt="a cozy dining room with a chair",
        nodes=(
            EntityNode("entity_chair", "chair", "chair", {"start": 26, "end": 31}),
            RequirementNode("requirement_chair", "a chair", {"start": 24, "end": 31}),
            PredicateNode(
                "predicate_scene",
                "a cozy dining room",
                "a cozy dining room",
                "scene_identity",
                {"start": 0, "end": 18},
            ),
            RequirementNode(
                "requirement_scene", "a cozy dining room", {"start": 0, "end": 18}
            ),
        ),
        edges=(
            RequirementMemberEdge(
                "requirement_chair", "entity_chair", "scored_facet", 1.0
            ),
            RequirementMemberEdge(
                "requirement_scene", "predicate_scene", "scored_facet", 1.0
            ),
        ),
        roots=(
            RootRequirement("requirement_chair", 0.5),
            RootRequirement("requirement_scene", 0.5),
        ),
    )
    target = Stage2ActorTarget(
        "actor-chair",
        ActorBounds((0.0, 0.0, 100.0), (45.0, 45.0, 100.0)),
    )
    object_task = Stage2Task(
        "stage2:chair",
        "entity_chair",
        Stage2TaskKind.OBJECT_EXISTENCE,
        0.5,
        arguments=(Stage2TaskArgument("entity_chair", "subject"),),
    )
    scene_task = Stage2Task(
        "stage2:scene",
        "predicate_scene",
        Stage2TaskKind.SCENE_IDENTITY,
        0.5,
    )
    feature_task = Stage2Task(
        "stage2:chair-finish",
        "predicate_scene",
        Stage2TaskKind.ATTRIBUTE,
        0.5,
        arguments=(
            Stage2TaskArgument("entity_chair", "subject"),
            Stage2TaskArgument("entity_chair", "feature"),
        ),
    )
    plan = plan_stage2_capture_groups(
        graph,
        (object_task, scene_task, feature_task),
        Stage2RoutingPlan(
            (
                Stage2EntityRoute(
                    "entity_chair", Stage2RouteSource.STRICT_MATCH, (target,)
                ),
            )
        ),
        SCENE_BOUNDS,
        budget=Stage2Budget.graph_default(),
    )

    overviews = tuple(
        request
        for request in plan.requests
        if request.shot_role is CaptureShotRole.OVERVIEW
    )
    assert len(overviews) == 6
    assert all(
        request.task_ids == (scene_task.task_id, feature_task.task_id)
        for request in overviews
    )
    assert all(
        object_task.task_id not in request.task_ids for request in overviews
    )
    assert feature_task.allows_overview_fallback is True


def test_unbounded_stage2_reuses_formal_overview_as_global_fallback():
    graph = RequirementGraph(
        prompt="a brooding Gothic cathedral complex",
        nodes=(
            PredicateNode(
                "predicate_scene",
                "a brooding Gothic cathedral complex",
                "a brooding Gothic cathedral complex",
                "scene_identity",
                {"start": 0, "end": 35},
            ),
            RequirementNode(
                "requirement_scene",
                "a brooding Gothic cathedral complex",
                {"start": 0, "end": 35},
            ),
        ),
        edges=(
            RequirementMemberEdge(
                "requirement_scene",
                "predicate_scene",
                "scored_facet",
                1.0,
            ),
        ),
        roots=(RootRequirement("requirement_scene", 1.0),),
    )
    inventory = ActorInventorySnapshot(())
    stage1 = evaluate_stage1(graph, inventory, SCENE_BOUNDS)
    overview = CapturedFrame(
        "formal_view_0",
        CameraPose(-1200.0, -900.0, 500.0, -12.0, 35.0),
        _rgb(91),
        phase="formal_overview_reuse",
    )

    class Provider:
        def capture(self, *args, **kwargs):
            raise AssertionError("the preloaded overview should avoid a new capture")

    stage2 = evaluate_stage2_detailed(
        graph,
        stage1,
        inventory,
        SCENE_BOUNDS,
        Provider(),
        budget=Stage2Budget(
            overview_frames=1,
            max_overview_frames=1,
            global_capture_limits_enabled=False,
        ),
        initial_overview_frames=(overview,),
    )

    assert stage2.capture_plan.requests == ()
    assert len(stage2.frame_store.records) == 1
    record = stage2.frame_store.records[0]
    assert record.phase == "formal_overview_reuse"
    assert record.shot_role is CaptureShotRole.OVERVIEW
    assert record.task_ids == (stage2.tasks[0].task_id,)
    assert stage2.result.assessments[0].unknown_reason == (
        "visual_evidence_incomplete"
    )
    assert stage2.result.assessments[0].evidence_frame_ids == (record.frame_id,)


def test_separate_graph_entities_resolved_to_same_actor_share_capture_program():
    graph = RequirementGraph(
        prompt="a red metal garage door",
        nodes=(
            EntityNode(
                "entity_door_color",
                "red metal garage door",
                "garage door",
                {"start": 2, "end": 23},
            ),
            EntityNode(
                "entity_door_material",
                "red metal garage door",
                "garage door",
                {"start": 2, "end": 23},
            ),
            RequirementNode(
                "requirement_color",
                "red metal garage door",
                {"start": 2, "end": 23},
            ),
            RequirementNode(
                "requirement_material",
                "red metal garage door",
                {"start": 2, "end": 23},
            ),
        ),
        edges=(
            RequirementMemberEdge(
                "requirement_color", "entity_door_color", "scored_facet", 1.0
            ),
            RequirementMemberEdge(
                "requirement_material",
                "entity_door_material",
                "scored_facet",
                1.0,
            ),
        ),
        roots=(
            RootRequirement("requirement_color", 0.5),
            RootRequirement("requirement_material", 0.5),
        ),
    )
    target = Stage2ActorTarget(
        "candidate-garage-door",
        ActorBounds((0.0, 0.0, 200.0), (180.0, 25.0, 120.0)),
    )
    routes = Stage2RoutingPlan(
        (
            Stage2EntityRoute(
                "entity_door_color", Stage2RouteSource.STRICT_MATCH, (target,)
            ),
            Stage2EntityRoute(
                "entity_door_material", Stage2RouteSource.STRICT_MATCH, (target,)
            ),
        )
    )
    tasks = (
        Stage2Task(
            "stage2:color",
            "entity_door_color",
            Stage2TaskKind.ATTRIBUTE,
            0.5,
            arguments=(Stage2TaskArgument("entity_door_color", "subject"),),
        ),
        Stage2Task(
            "stage2:material",
            "entity_door_material",
            Stage2TaskKind.MATERIAL,
            0.5,
            arguments=(Stage2TaskArgument("entity_door_material", "subject"),),
        ),
    )

    plan = plan_stage2_capture_groups(
        graph,
        tasks,
        routes,
        SCENE_BOUNDS,
        budget=Stage2Budget(
            overview_frames=0,
            max_overview_frames=0,
            max_targeted_frames=12,
            max_recovery_frames=2,
            max_valid_frames=18,
            max_targeted_views_per_entity_group=2,
            global_capture_limits_enabled=False,
        ),
    )

    assert len(plan.programs) == 1
    program = plan.programs[0]
    assert set(program.task_ids) == {"stage2:color", "stage2:material"}
    assert len([request for request in program.requests if not request.recovery_step]) == 2
    assert all(
        set(request.task_ids) == {"stage2:color", "stage2:material"}
        for request in program.requests
    )
    assert all(
        request.actor_ids == ("candidate-garage-door",)
        for request in program.requests
    )


def test_multi_entity_task_sharing_one_actor_is_added_to_capture_group_once():
    graph = RequirementGraph(
        prompt="a fortified village in woodland",
        nodes=(
            EntityNode(
                "entity_village",
                "fortified village",
                "village",
                {"start": 2, "end": 19},
            ),
            EntityNode(
                "entity_woodland",
                "woodland",
                "woodland",
                {"start": 23, "end": 31},
            ),
            RequirementNode(
                "requirement_relation",
                "a fortified village in woodland",
                {"start": 0, "end": 31},
            ),
        ),
        edges=(
            RequirementMemberEdge(
                "requirement_relation", "entity_village", "scored_facet", 0.5
            ),
            RequirementMemberEdge(
                "requirement_relation", "entity_woodland", "scored_facet", 0.5
            ),
        ),
        roots=(RootRequirement("requirement_relation", 1.0),),
    )
    target = Stage2ActorTarget(
        "candidate-scene-anchor",
        ActorBounds((0.0, 0.0, 200.0), (180.0, 180.0, 120.0)),
    )
    routes = Stage2RoutingPlan(
        (
            Stage2EntityRoute(
                "entity_village", Stage2RouteSource.LOCATOR_ONLY, (target,)
            ),
            Stage2EntityRoute(
                "entity_woodland", Stage2RouteSource.LOCATOR_ONLY, (target,)
            ),
        )
    )
    task = Stage2Task(
        "stage2:village-in-woodland",
        "requirement_relation",
        Stage2TaskKind.SPATIAL_RELATION,
        1.0,
        arguments=(
            Stage2TaskArgument("entity_village", "subject"),
            Stage2TaskArgument("entity_woodland", "reference"),
        ),
    )

    plan = plan_stage2_capture_groups(
        graph,
        (task,),
        routes,
        SCENE_BOUNDS,
        budget=Stage2Budget.graph_default(),
    )

    programs = tuple(
        program for program in plan.programs if task.task_id in program.task_ids
    )
    assert programs
    assert all(program.task_ids == (task.task_id,) for program in programs)
    assert all(
        request.task_ids == (task.task_id,)
        for program in programs
        for request in program.requests
    )

def test_stage2_rejects_underground_global_pose_before_calling_provider():
    request = CaptureRequest(
        "capture-1",
        ("task-1",),
        CaptureShotRole.OVERVIEW,
        CameraPose(0.0, 0.0, -100.0, 0.0, 0.0),
    )
    plan = Stage2CapturePlan(
        (CaptureProgram("program-1", ("task-1",), (request,)),),
        Stage2Budget(overview_frames=1),
    )

    class Provider:
        def capture(self, *args, **kwargs):
            raise AssertionError("an underground camera must not reach UE")

    result = execute_stage2_capture_plan(
        plan,
        Provider(),
        FrameStore(),
        content_floor_z=0.0,
    )

    assert result.attempts[0].status is CaptureAttemptStatus.REJECTED
    assert result.attempts[0].reason == "camera_below_content_floor"


def test_graph_profile_is_complete_and_not_order_capped():
    budget = Stage2Budget.graph_default()

    assert budget.global_capture_limits_enabled is False
    assert budget.overview_frames == 6
    assert budget.max_overview_frames == 6
    assert budget.max_evidence_frames_per_task == 8
    assert budget.max_targeted_views_per_entity_group == 2


def test_graph_profile_targets_every_localized_entity_beyond_old_global_cap():
    entity_count = 16
    labels = tuple(f"object {index}" for index in range(entity_count))
    prompt = " | ".join(labels)
    spans = []
    cursor = 0
    for label in labels:
        spans.append({"start": cursor, "end": cursor + len(label)})
        cursor += len(label) + 3
    nodes = []
    edges = []
    roots = []
    tasks = []
    routes = []
    for index in range(entity_count):
        entity_id = f"entity_{index:02d}"
        requirement_id = f"requirement_{index:02d}"
        task_id = f"stage2:{index:02d}"
        nodes.extend(
            (
                EntityNode(
                    entity_id,
                    labels[index],
                    labels[index],
                    spans[index],
                ),
                RequirementNode(
                    requirement_id,
                    labels[index],
                    spans[index],
                ),
            )
        )
        edges.append(
            RequirementMemberEdge(requirement_id, entity_id, "scored_facet", 1.0)
        )
        roots.append(RootRequirement(requirement_id, 1.0 / entity_count))
        tasks.append(
            Stage2Task(
                task_id,
                entity_id,
                Stage2TaskKind.OBJECT_EXISTENCE,
                1.0 / entity_count,
                arguments=(Stage2TaskArgument(entity_id, "subject"),),
            )
        )
        x = -1500.0 + float(index % 4) * 1000.0
        y = -1500.0 + float(index // 4) * 1000.0
        routes.append(
            Stage2EntityRoute(
                entity_id,
                Stage2RouteSource.STRICT_MATCH,
                (
                    Stage2ActorTarget(
                        f"actor-{index:02d}",
                        ActorBounds((x, y, 120.0), (40.0, 40.0, 80.0)),
                    ),
                ),
            )
        )
    graph = RequirementGraph(
        prompt=prompt,
        nodes=tuple(nodes),
        edges=tuple(edges),
        roots=tuple(roots),
    )

    plan = plan_stage2_capture_groups(
        graph,
        tuple(tasks),
        Stage2RoutingPlan(tuple(routes)),
        SCENE_BOUNDS,
        budget=Stage2Budget.graph_default(),
    )

    targeted_task_ids = {
        task_id
        for request in plan.requests
        if request.recovery_step == 0
        and request.shot_role
        not in {CaptureShotRole.OVERVIEW, CaptureShotRole.GRID}
        for task_id in request.task_ids
    }
    assert targeted_task_ids == {task.task_id for task in tasks}
    assert len(targeted_task_ids) > 12


def test_locator_hypotheses_receive_collection_wide_and_joint_views():
    graph = RequirementGraph(
        prompt="three lanterns around a gate",
        nodes=(
            EntityNode(
                "entity_lanterns",
                "three lanterns",
                "lanterns",
                {"start": 0, "end": 14},
                referent_kind=ReferentKind.COLLECTION,
            ),
            EntityNode(
                "entity_lanterns_relation",
                "lanterns",
                "lanterns",
                {"start": 6, "end": 14},
                referent_kind=ReferentKind.COLLECTION,
            ),
            EntityNode(
                "entity_gate",
                "gate",
                "gate",
                {"start": 24, "end": 28},
            ),
            RequirementNode(
                "requirement_count",
                "three lanterns",
                {"start": 0, "end": 14},
            ),
            RequirementNode(
                "requirement_relation",
                "lanterns around a gate",
                {"start": 6, "end": 28},
            ),
        ),
        edges=(
            RequirementMemberEdge(
                "requirement_count", "entity_lanterns", "scored_facet", 1.0
            ),
            RequirementMemberEdge(
                "requirement_relation",
                "entity_lanterns_relation",
                "scored_facet",
                0.5,
            ),
            RequirementMemberEdge(
                "requirement_relation", "entity_gate", "scored_facet", 0.5
            ),
        ),
        roots=(
            RootRequirement("requirement_count", 0.5),
            RootRequirement("requirement_relation", 0.5),
        ),
    )
    lanterns = tuple(
        Stage2ActorTarget(
            f"lantern-{index}",
            ActorBounds((float(index - 1) * 300.0, 0.0, 180.0), (35.0, 35.0, 90.0)),
        )
        for index in range(3)
    )
    gate = Stage2ActorTarget(
        "gate-actor",
        ActorBounds((0.0, 350.0, 200.0), (180.0, 45.0, 200.0)),
    )
    routes = Stage2RoutingPlan(
        (
            Stage2EntityRoute(
                "entity_lanterns", Stage2RouteSource.LOCATOR_ONLY, lanterns
            ),
            Stage2EntityRoute(
                "entity_lanterns_relation",
                Stage2RouteSource.LOCATOR_ONLY,
                lanterns,
            ),
            Stage2EntityRoute(
                "entity_gate", Stage2RouteSource.LOCATOR_ONLY, (gate,)
            ),
        )
    )
    count_task = Stage2Task(
        "stage2:count",
        "requirement_count",
        Stage2TaskKind.COUNT,
        0.5,
        arguments=(Stage2TaskArgument("entity_lanterns", "subject"),),
    )
    relation_task = Stage2Task(
        "stage2:relation",
        "requirement_relation",
        Stage2TaskKind.SPATIAL_RELATION,
        0.5,
        arguments=(
            Stage2TaskArgument("entity_lanterns_relation", "subject"),
            Stage2TaskArgument("entity_gate", "reference"),
        ),
    )
    second_relation_task = Stage2Task(
        "stage2:relation-second-atomic-facet",
        "requirement_relation",
        Stage2TaskKind.SPATIAL_RELATION,
        0.5,
        arguments=relation_task.arguments,
    )

    plan = plan_stage2_capture_groups(
        graph,
        (count_task, relation_task, second_relation_task),
        routes,
        SCENE_BOUNDS,
        budget=Stage2Budget.graph_default(),
    )

    assert any(
        request.shot_role is CaptureShotRole.COLLECTION_WIDE
        and count_task.task_id in request.task_ids
        and len(request.actor_ids) >= 2
        for request in plan.requests
    )
    assert any(
        request.shot_role is CaptureShotRole.JOINT
        and {
            relation_task.task_id,
            second_relation_task.task_id,
        }.issubset(request.task_ids)
        and len(request.actor_ids) == 2
        for request in plan.requests
    )
    shared_relation_programs = [
        program
        for program in plan.programs
        if any(
            request.shot_role is CaptureShotRole.JOINT
            for request in program.requests
        )
        and (
            relation_task.task_id in program.task_ids
            or second_relation_task.task_id in program.task_ids
        )
    ]
    assert len(shared_relation_programs) == 1
    assert set(shared_relation_programs[0].task_ids) == {
        relation_task.task_id,
        second_relation_task.task_id,
    }


def test_explicit_frame_evidence_budget_is_not_silently_truncated():
    assert FrameStore(evidence_frames_per_task=8).evidence_frames_per_task == 8


def test_stage3_graph_budget_inherits_all_seed_frames_and_scales_attempts():
    budget = Stage3Budget.for_requirement_graph(
        seed_frame_count=27,
        focus_actor_count=20,
        grid_task_count=3,
        require_global_context=True,
    )

    assert budget.max_seed_frames == 27
    assert budget.max_new_capture_attempts == 29
    assert budget.max_valid_exploration_frames == 76
    assert budget.max_portfolio_frames == 12


def test_stage2_sends_one_program_as_one_provider_camera_sweep():
    poses = (
        CameraPose(-1000.0, 0.0, 300.0, 0.0, 0.0),
        CameraPose(0.0, -1000.0, 300.0, 0.0, 90.0),
        CameraPose(1000.0, 0.0, 300.0, 0.0, 180.0),
    )
    requests = tuple(
        CaptureRequest(
            f"capture-{index}",
            ("task-1",),
            CaptureShotRole.OVERVIEW,
            pose,
        )
        for index, pose in enumerate(poses, start=1)
    )
    plan = Stage2CapturePlan(
        tuple(
            CaptureProgram(
                f"program-{index}",
                ("task-1",),
                (request,),
            )
            for index, request in enumerate(requests, start=1)
        ),
        Stage2Budget(overview_frames=3),
    )

    class Provider:
        def __init__(self):
            self.calls = []

        def capture(self, requested, **kwargs):
            self.calls.append((tuple(requested), kwargs))
            return [
                CapturedFrame(f"provider-{index}", pose, _rgb(index))
                for index, pose in enumerate(requested, start=10)
            ]

    provider = Provider()
    result = execute_stage2_capture_plan(plan, provider, FrameStore())

    assert len(provider.calls) == 1
    assert provider.calls[0][0] == poses
    assert [value.status for value in result.attempts] == [
        CaptureAttemptStatus.ACCEPTED,
        CaptureAttemptStatus.ACCEPTED,
        CaptureAttemptStatus.ACCEPTED,
    ]


def test_stage2_unresolved_actor_pose_recovers_from_inventory_bounds():
    initial_pose = CameraPose(0.0, -500.0, 200.0, 0.0, 90.0)
    plan = Stage2CapturePlan(
        (
            CaptureProgram(
                "program-target",
                ("task-target",),
                (
                    CaptureRequest(
                        "capture-target",
                        ("task-target",),
                        CaptureShotRole.CLOSE,
                        initial_pose,
                        actor_ids=("thin-actor",),
                    ),
                ),
            ),
        ),
        Stage2Budget(overview_frames=0),
    )

    class Provider:
        def __init__(self):
            self.scene = SimpleNamespace(
                inventory=ActorInventorySnapshot(
                    (
                        _actor(
                            "thin-actor",
                            "ThinActor",
                            (0.0, 0.0, 250.0),
                            (5.0, 300.0, 20.0),
                        ),
                    )
                ),
                scene_bounds=SCENE_BOUNDS,
            )
            self.resolutions = []

        def resolve_camera_poses(self, poses, *, actor_ids_by_pose):
            requested = tuple(poses)
            self.resolutions.append(
                (requested, tuple(tuple(value) for value in actor_ids_by_pose))
            )
            if len(self.resolutions) == 1:
                return (None,)
            return (requested[0], *(None for _ in requested[1:]))

        def capture(self, poses, **kwargs):
            del kwargs
            return tuple(
                CapturedFrame(f"recovered-{index}", pose, _rgb(index + 70))
                for index, pose in enumerate(poses)
            )

    provider = Provider()
    result = execute_stage2_capture_plan(plan, provider, FrameStore())

    assert result.runtime_error is None
    assert len(provider.resolutions) == 2
    assert len(provider.resolutions[1][0]) >= 8
    assert all(
        actor_ids == ("thin-actor",)
        for actor_ids in provider.resolutions[1][1]
    )
    assert result.attempts[0].status is CaptureAttemptStatus.ACCEPTED


def test_unlocalized_task_does_not_borrow_another_requirements_focus_frame():
    store = FrameStore(valid_frame_hard_cap=4)
    store.admit(
        _rgb(1),
        pose=CameraPose(-1500.0, 0.0, 220.0, -5.0, 0.0),
        phase="stage2_capture",
        shot_role=CaptureShotRole.OVERVIEW,
        task_ids=("unknown-task",),
    )
    focused = store.admit(
        _rgb(2),
        pose=CameraPose(300.0, 0.0, 200.0, 0.0, 180.0),
        phase="stage2_capture",
        shot_role=CaptureShotRole.CONTEXT,
        task_ids=("other-task",),
        actor_ids=("actor-ac",),
    ).record

    selected = store.select_records_for_task(
        "unknown-task",
        preferred_shot_roles=(
            CaptureShotRole.CONTEXT,
            CaptureShotRole.CLOSE,
            CaptureShotRole.OVERVIEW,
        ),
    )

    assert focused is not None
    assert [value.frame_id for value in selected] == ["s2f_000001"]


def test_stage2_public_evaluator_has_no_judge_parameter():
    assert "judge" not in signature(evaluate_stage2_detailed).parameters


def test_object_requirement_needs_target_or_its_own_grid_before_stage3():
    task = Stage2Task(
        "unknown-task",
        "unknown-node",
        Stage2TaskKind.OBJECT_EXISTENCE,
        1.0,
    )
    store = FrameStore(valid_frame_hard_cap=3)
    store.admit(
        _rgb(10),
        pose=CameraPose(-1500.0, 0.0, 220.0, -5.0, 0.0),
        phase="stage2_capture",
        shot_role=CaptureShotRole.OVERVIEW,
        task_ids=(task.task_id,),
    )
    overview_only = store.select_records_for_task(task.task_id)

    assert _evidence_precondition_reason(
        task,
        Stage2RoutingPlan(()),
        overview_only,
    ) == "target_not_localized"

    store.admit(
        _rgb(11),
        pose=CameraPose(0.0, 0.0, 300.0, -10.0, 45.0),
        phase="stage2_capture",
        shot_role=CaptureShotRole.GRID,
        task_ids=(task.task_id,),
    )
    requirement_search = store.select_records_for_task(task.task_id)
    assert _evidence_precondition_reason(
        task,
        Stage2RoutingPlan(()),
        requirement_search,
    ) is None

def test_requirement_grid_survives_four_overviews_and_the_four_frame_cap():
    task = Stage2Task(
        "grid-task",
        "grid-node",
        Stage2TaskKind.OBJECT_EXISTENCE,
        1.0,
    )
    store = FrameStore(valid_frame_hard_cap=5, evidence_frames_per_task=4)
    for index in range(4):
        store.admit(
            _rgb(120 + index),
            pose=CameraPose(
                -1500.0 + index * 500.0,
                0.0,
                220.0,
                -5.0,
                float(index * 45),
            ),
            phase="stage2_capture",
            shot_role=CaptureShotRole.OVERVIEW,
            task_ids=(task.task_id,),
        )
    grid = store.admit(
        _rgb(130),
        pose=CameraPose(0.0, 800.0, 300.0, -10.0, -90.0),
        phase="stage2_capture",
        shot_role=CaptureShotRole.GRID,
        task_ids=(task.task_id,),
    ).record

    selected = store.select_records_for_task(
        task.task_id,
        preferred_shot_roles=preferred_shot_roles_for_task(task),
    )

    assert grid is not None
    assert grid.frame_id in {value.frame_id for value in selected}
    assert len(selected) == 4
    assert _evidence_precondition_reason(
        task,
        Stage2RoutingPlan(()),
        selected,
    ) is None



def test_stage2_splits_controller_audit_from_judge_visible_metadata():
    bounds = ActorBounds((20.0, 30.0, 200.0), (40.0, 50.0, 80.0))
    store = FrameStore(valid_frame_hard_cap=2)
    admitted = store.admit(
        _rgb(12),
        pose=CameraPose(300.0, 30.0, 220.0, 0.0, 180.0),
        phase="stage2_capture",
        shot_role=CaptureShotRole.CONTEXT,
        task_ids=("task-neon",),
        actor_ids=("BP_NeonSign_03",),
    )
    assert admitted.record is not None
    audit, visible = _capture_metadata_payloads(
        frame_store=store,
        routing_plan=Stage2RoutingPlan(
            (
                Stage2EntityRoute(
                    "entity-neon",
                    Stage2RouteSource.STRICT_MATCH,
                    (Stage2ActorTarget("BP_NeonSign_03", bounds),),
                ),
            )
        ),
        requirement_ids_by_task={"task-neon": ("req-07",)},
    )

    assert audit["frames"][0]["requirement_ids"] == ["req-07"]
    assert audit["frames"][0]["capture_target_actor_ids"] == ["BP_NeonSign_03"]
    assert audit["frames"][0]["capture_target_bounds"]["BP_NeonSign_03"] == {
        "center_cm": [20.0, 30.0, 200.0],
        "extent_cm": [40.0, 50.0, 80.0],
    }
    assert visible == {
        "schema_version": "1.0",
        "frames": [
            {
                "frame_id": admitted.record.frame_id,
                "channel": "rgb",
                "view_type": "targeted",
            }
        ],
    }
    assert "NeonSign" not in str(visible)


def test_stage3_reuses_stage2_focus_by_actor_route_without_recapture():
    bounds = ActorBounds((0.0, 0.0, 200.0), (40.0, 50.0, 80.0))
    pose = plan_bounds_camera_poses(
        bounds.center_cm,
        bounds.extent_cm,
        SCENE_BOUNDS,
    )[0]
    stage2_store = FrameStore(valid_frame_hard_cap=2)
    stage2_store.admit(
        _rgb(3),
        pose=pose,
        phase="stage2_capture",
        shot_role=CaptureShotRole.CONTEXT,
        task_ids=("other-task",),
        actor_ids=("actor-ac",),
    )

    class Provider:
        def capture(self, *args, **kwargs):
            raise AssertionError("the valid Stage 2 focus frame should be reused")

    result = Stage3Explorer(
        Stage3Budget(
            max_seed_frames=1,
            max_new_capture_attempts=0,
            max_valid_exploration_frames=1,
            min_portfolio_frames=1,
            max_portfolio_frames=1,
            max_recovery_attempts=0,
        )
    ).explore(
        SCENE_BOUNDS,
        Provider(),
        stage2_frame_store=stage2_store,
        focus_targets_by_task={
            "unknown-task": (Stage2ActorTarget("actor-ac", bounds),)
        },
        content_floor_z=0.0,
    )

    records = result.frame_store.records
    assert len(records) == 1
    assert records[0].task_ids == ()
    assert records[0].actor_ids == ()
    assert result.task_frame_ids == {
        "unknown-task": (records[0].frame_id,)
    }
    assert result.frame_actor_ids == {records[0].frame_id: ("actor-ac",)}
    assert "task_frame_ids" not in result.to_dict()
    assert "frame_actor_ids" not in result.to_dict()
    assert len(result.judge_frames_for_task("unknown-task")) == 1
    frame_index, metadata, png_payloads = _prepare_store(result.frame_store)
    assert frame_index["valid_frame_count"] == 1
    assert tuple(metadata) == (records[0].frame_id,)
    assert tuple(png_payloads) == (records[0].frame_id,)


def test_stage3_batches_missing_actor_focus_and_scout_poses():
    left = ActorBounds((-500.0, 0.0, 200.0), (40.0, 50.0, 80.0))
    right = ActorBounds((500.0, 0.0, 200.0), (40.0, 50.0, 80.0))

    class Provider:
        def __init__(self):
            self.calls = []
            self.resolutions = []

        def resolve_camera_poses(self, poses, *, actor_ids_by_pose):
            requested = tuple(poses)
            self.resolutions.append(
                (requested, tuple(tuple(value) for value in actor_ids_by_pose))
            )
            return requested

        def capture(self, poses, **kwargs):
            self.calls.append((tuple(poses), kwargs))
            return [
                CapturedFrame(f"provider-{index}", pose, _rgb(index + 30))
                for index, pose in enumerate(poses)
            ]

    provider = Provider()
    result = Stage3Explorer(
        Stage3Budget(
            max_seed_frames=0,
            max_new_capture_attempts=4,
            max_valid_exploration_frames=4,
            min_portfolio_frames=1,
            max_portfolio_frames=4,
            max_recovery_attempts=0,
        )
    ).explore(
        SCENE_BOUNDS,
        provider,
        focus_targets_by_task={
            "left-task": (Stage2ActorTarget("left-actor", left),),
            "right-task": (Stage2ActorTarget("right-actor", right),),
        },
        content_floor_z=0.0,
    )

    assert len(provider.calls) == 1
    assert len(provider.resolutions) == 1
    assert {
        actor_ids for actor_ids in provider.resolutions[0][1] if actor_ids
    } == {("left-actor",), ("right-actor",)}
    assert len(provider.calls[0][0]) == 4
    assert provider.calls[0][1]["phase"] == "stage3_batch"
    assert result.coverage.focus_frames == 2
    assert result.coverage.valid_new_frames == 4


def test_stage3_targeted_mode_does_not_capture_scouts_or_generic_grid():
    left = ActorBounds((-500.0, 0.0, 200.0), (40.0, 50.0, 80.0))
    right = ActorBounds((500.0, 0.0, 200.0), (40.0, 50.0, 80.0))

    class Provider:
        def __init__(self):
            self.calls = []

        def capture(self, poses, **kwargs):
            self.calls.append((tuple(poses), kwargs))
            return [
                CapturedFrame(f"target-{index}", pose, _rgb(index + 140))
                for index, pose in enumerate(poses)
            ]

    provider = Provider()
    result = Stage3Explorer(
        Stage3Budget(
            max_seed_frames=0,
            max_new_capture_attempts=4,
            max_valid_exploration_frames=4,
            min_portfolio_frames=1,
            max_portfolio_frames=4,
            max_recovery_attempts=0,
        )
    ).explore(
        SCENE_BOUNDS,
        provider,
        focus_targets_by_task={
            "left-task": (Stage2ActorTarget("left-actor", left),),
            "right-task": (Stage2ActorTarget("right-actor", right),),
        },
        require_global_context=False,
        content_floor_z=0.0,
    )

    assert len(provider.calls) == 1
    assert len(provider.calls[0][0]) == 2
    assert result.global_context_required is False
    assert result.coverage.constraints_met is True
    assert result.coverage.global_frames == 0
    assert {
        value.shot_role for value in result.frame_store.records
    } == {CaptureShotRole.CONTEXT}


def test_stage3_assigns_distinct_grid_frames_to_unlocalized_requirements():
    class Provider:
        def __init__(self):
            self.calls = []

        def capture(self, poses, **kwargs):
            self.calls.append((tuple(poses), kwargs))
            return [
                CapturedFrame(f"grid-{index}", pose, _rgb(index + 80))
                for index, pose in enumerate(poses)
            ]

    provider = Provider()
    result = Stage3Explorer(
        Stage3Budget(
            max_seed_frames=0,
            max_new_capture_attempts=2,
            max_valid_exploration_frames=2,
            min_portfolio_frames=1,
            max_portfolio_frames=2,
            max_recovery_attempts=0,
        )
    ).explore(
        SCENE_BOUNDS,
        provider,
        focus_targets_by_task={"missing-a": (), "missing-b": ()},
        grid_search_task_ids=("missing-a", "missing-b"),
        content_floor_z=0.0,
    )

    assert len(provider.calls) == 1
    assert len(provider.calls[0][0]) == 2
    assert set(result.task_frame_ids) == {"missing-a", "missing-b"}
    left = result.task_frame_ids["missing-a"]
    right = result.task_frame_ids["missing-b"]
    assert len(left) == len(right) == 1
    assert left != right
    assert {result.frame_store.get(value[0]).shot_role for value in (left, right)} == {
        CaptureShotRole.GRID
    }
