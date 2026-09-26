"""RequirementGraph Stage 2 evidence acquisition without owning an Unreal session.

The controller in this module deliberately keeps four kinds of information on
separate sides of the pipeline:

* RequirementGraph semantics create scored tasks;
* retrieval and actor geometry are acquisition hints only;
* :class:`FrameStore` is the sole owner of valid RGB and controller metadata;
* requirement-linked evidence is handed to Stage 3 through opaque frame ids.

``evaluate_stage2_detailed`` is used by the graph runner and artifact writer.
The smaller ``evaluate_stage2`` API returns only its evidence-status result.
Neither function enters, closes, or otherwise owns the supplied frame provider.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .actor_inventory import (
    ActorInventorySnapshot,
    InventoryStatus,
    estimate_camera_content_floor_z,
)
from .contracts import RequirementGraph, SceneBounds
from .overview_planning import plan_overview_poses
from .runtime import CapturedFrame, FrameProvider
from .semantic_retrieval import (
    SceneIdentityRetrievalResult,
    SemanticRetrievalBackend,
    retrieve_scene_identities,
)
from .stage1 import Stage1Result
from .stage2_capture import (
    Stage2CaptureExecution,
    execute_stage2_capture_plan,
    plan_stage2_capture_groups,
    preferred_shot_roles_for_task,
)
from .stage2_contracts import (
    CaptureShotRole,
    SearchCoverage,
    Stage2Assessment,
    Stage2Budget,
    Stage2CapturePlan,
    Stage2EvidenceBasis,
    Stage2Result,
    Stage2Task,
    Stage2Verdict,
    VisualGrounding,
)
from .stage2_frames import FrameRecord, FrameStore
from .stage2_routing import (
    Stage2RoutingPlan,
    plan_stage2_entity_routes,
)
from .stage2_tasks import build_stage2_queries, build_stage2_tasks


@dataclass(frozen=True, slots=True)
class Stage2Evaluation:
    """Full controller product used to write a reproducible Stage 2 run."""

    tasks: tuple[Stage2Task, ...]
    identity_retrieval: SceneIdentityRetrievalResult
    routing_plan: Stage2RoutingPlan
    capture_plan: Stage2CapturePlan
    capture_execution: Stage2CaptureExecution
    frame_store: FrameStore
    evidence_selection: Mapping[str, Any]
    request_manifest: tuple[Mapping[str, Any], ...]
    raw_records: tuple[Mapping[str, Any], ...]
    result: Stage2Result

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "evidence_selection", dict(self.evidence_selection))
        object.__setattr__(
            self,
            "request_manifest",
            tuple(dict(value) for value in self.request_manifest),
        )
        object.__setattr__(
            self,
            "raw_records",
            tuple(dict(value) for value in self.raw_records),
        )

    @property
    def capture_artifact(self) -> dict[str, Any]:
        """Plan plus acquisition outcomes for the controller-only artifact."""

        return {
            "schema_version": "1.0",
            "plan": self.capture_plan.to_dict(),
            "execution": self.capture_execution.to_dict(),
        }


class _CountingSemanticBackend:
    """Transparent one-physical-call guard for either retrieval backend API."""

    def __init__(self, backend: SemanticRetrievalBackend) -> None:
        self._backend = backend
        self.call_count = 0

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._backend, name)
        if name not in {
            "score",
            "score_with_absence_gate",
            "retrieve_scene_identities",
        } or not callable(attribute):
            return attribute

        def counted(*args: Any, **kwargs: Any) -> Any:
            if self.call_count >= 1:
                raise RuntimeError(
                    "Stage 2 semantic retrieval backend call budget exceeded"
                )
            self.call_count += 1
            return attribute(*args, **kwargs)

        return counted


def _unknown_assessment(
    task: Stage2Task,
    coverage: SearchCoverage,
    reason: str,
    *,
    rationale: str | None = None,
    evidence_frame_ids: Sequence[str] = (),
    confidence: float = 0.0,
    mismatch_downgraded: bool = False,
) -> Stage2Assessment:
    return Stage2Assessment(
        task_id=task.task_id,
        node_id=task.node_id,
        verdict=Stage2Verdict.UNKNOWN,
        confidence=confidence,
        evidence_frame_ids=tuple(evidence_frame_ids),
        rationale=rationale or reason.replace("_", " "),
        evidence_basis=Stage2EvidenceBasis.INSUFFICIENT,
        coverage=coverage,
        grounding=VisualGrounding(),
        unknown_reason=reason,
        mismatch_downgraded=mismatch_downgraded,
    )


def _route_target_ids(
    routing: Stage2RoutingPlan,
    entity_id: str,
) -> tuple[str, ...]:
    route = routing.for_entity(entity_id)
    return route.actor_ids if route is not None else ()


def _evidence_precondition_reason(
    task: Stage2Task,
    routing: Stage2RoutingPlan,
    evidence: Sequence[FrameRecord],
) -> str | None:
    """Return why this task lacks legal, requirement-linked Stage 3 evidence.

    Scene-level atmosphere/identity claims legitimately use overviews.  An
    object, attribute, material, count, or relation claim instead needs either
    an actor-associated target view or a grid frame explicitly acquired for
    that task.  Generic overview pixels and another requirement's close-up do
    not establish that the requested subject was inspected.
    """

    if not evidence:
        return "no_valid_evidence"
    if task.allows_overview_fallback:
        return None

    global_roles = {CaptureShotRole.OVERVIEW, CaptureShotRole.GRID}
    targeted = tuple(
        record
        for record in evidence
        if task.task_id in record.task_ids
        and bool(record.actor_ids)
        and record.shot_role not in global_roles
    )
    requirement_search = tuple(
        record
        for record in evidence
        if task.task_id in record.task_ids
        and record.shot_role is CaptureShotRole.GRID
    )
    if targeted or requirement_search:
        return None

    localized = any(
        bool(_route_target_ids(routing, entity_id))
        for entity_id in task.dependency_entity_ids
    )
    if not localized:
        return "target_not_localized"
    return "targeted_evidence_missing"


def _empty_capture_execution(
    tasks: Sequence[Stage2Task],
    *,
    runtime_error: str | None = None,
) -> Stage2CaptureExecution:
    return Stage2CaptureExecution(
        (),
        {task.task_id: SearchCoverage() for task in tasks},
        runtime_error,
    )


def evaluate_stage2_detailed(
    graph: RequirementGraph,
    stage1_result: Stage1Result,
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
    frame_provider: FrameProvider,
    *,
    retrieval_backend: SemanticRetrievalBackend | None = None,
    frame_store: FrameStore | None = None,
    budget: Stage2Budget | None = None,
    locator_top_k: int = 5,
    skip_node_ids: Collection[str] = (),
    eligible_actor_ids_by_entity: Mapping[str, Sequence[str]] | None = None,
    initial_overview_frames: Sequence[CapturedFrame] = (),
) -> Stage2Evaluation:
    """Run one policy-bounded Stage 2 evaluation on an active provider.

    All active object queries are passed to one invocation of
    :func:`retrieve_scene_identities`; that function itself performs at most
    one semantic-backend batch.  Independent primary poses are submitted as
    one ordered UE camera sweep; only health-driven recovery is adaptive.
    Legal targeted evidence is retained as visual-incomplete for Stage 3.
    Stage 2 never calls a VLM.
    """

    if not isinstance(graph, RequirementGraph):
        raise TypeError("graph must be a RequirementGraph")
    if not isinstance(stage1_result, Stage1Result):
        raise TypeError("stage1_result must be a Stage1Result")
    if not isinstance(inventory, ActorInventorySnapshot):
        raise TypeError("inventory must be an ActorInventorySnapshot")
    if not isinstance(scene_bounds, SceneBounds):
        raise TypeError("scene_bounds must be a SceneBounds")
    if isinstance(locator_top_k, bool) or not isinstance(locator_top_k, int):
        raise TypeError("locator_top_k must be an integer")
    if locator_top_k < 1:
        raise ValueError("locator_top_k must be positive")
    selected_budget = Stage2Budget() if budget is None else budget
    if not isinstance(selected_budget, Stage2Budget):
        raise TypeError("budget must be a Stage2Budget")
    if frame_store is not None and not isinstance(frame_store, FrameStore):
        raise TypeError("frame_store must be a FrameStore")
    initial_overviews = tuple(initial_overview_frames)
    if any(not isinstance(value, CapturedFrame) for value in initial_overviews):
        raise TypeError("initial_overview_frames must contain CapturedFrame values")

    tasks = build_stage2_tasks(
        graph, stage1_result, skip_node_ids=skip_node_ids
    )
    queries = build_stage2_queries(graph, tasks)
    retrieval_actors = inventory.actors
    if eligible_actor_ids_by_entity is not None:
        allowed_actor_ids = {
            str(actor_id).casefold()
            for actor_ids in eligible_actor_ids_by_entity.values()
            for actor_id in actor_ids
        }
        retrieval_actors = tuple(
            actor
            for actor in inventory.actors
            if actor.live_actor_id.casefold() in allowed_actor_ids
        )
    counted_backend = (
        _CountingSemanticBackend(retrieval_backend)
        if retrieval_backend is not None
        else None
    )
    structured_retriever = (
        getattr(counted_backend, "retrieve_scene_identities", None)
        if counted_backend is not None
        else None
    )
    if callable(structured_retriever) and queries:
        retrieval = structured_retriever(
            queries,
            retrieval_actors,
            scene_bounds=scene_bounds,
            top_k=locator_top_k,
            include_resolved_locators=True,
        )
    else:
        retrieval = retrieve_scene_identities(
            queries,
            retrieval_actors,
            scene_bounds=scene_bounds,
            semantic_backend=counted_backend,
            top_k=locator_top_k,
            include_resolved_locators=True,
        )
    routing = plan_stage2_entity_routes(
        graph,
        stage1_result,
        inventory,
        scene_bounds,
        locator_retrieval=retrieval,
        max_locator_actors_per_entity=locator_top_k,
        eligible_actor_ids_by_entity=eligible_actor_ids_by_entity,
    )
    content_floor_z = estimate_camera_content_floor_z(inventory, scene_bounds)
    overview_poses = plan_overview_poses(
        inventory,
        scene_bounds,
        count=selected_budget.max_overview_frames,
    )
    overview_task_ids = tuple(
        value.task_id
        for value in tasks
        if value.allows_overview_fallback
    )
    reusable_overviews = initial_overviews[: selected_budget.max_overview_frames]
    if selected_budget.global_capture_limits_enabled:
        store = (
            frame_store
            if frame_store is not None
            else FrameStore(selected_budget)
        )
        if overview_task_ids:
            for captured in reusable_overviews:
                store.admit_captured_frame(
                    captured,
                    shot_role=CaptureShotRole.OVERVIEW,
                    task_ids=overview_task_ids,
                )
        preloaded_overview_count = sum(
            value.shot_role is CaptureShotRole.OVERVIEW
            and bool(set(value.task_ids).intersection(overview_task_ids))
            for value in store.records
        )
        capture_plan = plan_stage2_capture_groups(
            graph,
            tasks,
            routing,
            scene_bounds,
            budget=selected_budget,
            locator_retrieval=retrieval,
            content_floor_z=content_floor_z,
            preloaded_overview_count=preloaded_overview_count,
            overview_poses=overview_poses,
        )
    else:
        # Retain the legacy unconstrained direct-call behavior without letting
        # it affect the production RequirementGraph profile.
        provisional_capture_plan = plan_stage2_capture_groups(
            graph,
            tasks,
            routing,
            scene_bounds,
            budget=selected_budget,
            locator_retrieval=retrieval,
            content_floor_z=content_floor_z,
            overview_poses=overview_poses,
        )
        # With global capture limits disabled, the plan itself is still finite
        # (bounded representatives/views per task).  Size the store to that
        # complete plan so it cannot silently reintroduce the old 18-frame
        # first-come-first-served cap.  An explicitly supplied store remains an
        # intentional caller-owned constraint.
        valid_frame_cap = (
            selected_budget.max_valid_frames
            if selected_budget.global_capture_limits_enabled
            else max(
                1,
                len(provisional_capture_plan.requests) + len(reusable_overviews),
            )
        )
        store = (
            frame_store
            if frame_store is not None
            else FrameStore(
                selected_budget,
                valid_frame_hard_cap=valid_frame_cap,
            )
        )
        if overview_task_ids:
            for captured in reusable_overviews:
                store.admit_captured_frame(
                    captured,
                    shot_role=CaptureShotRole.OVERVIEW,
                    task_ids=overview_task_ids,
                )
        preloaded_overview_count = sum(
            value.shot_role is CaptureShotRole.OVERVIEW
            and bool(set(value.task_ids).intersection(overview_task_ids))
            for value in store.records
        )
        capture_plan = (
            plan_stage2_capture_groups(
                graph,
                tasks,
                routing,
                scene_bounds,
                budget=selected_budget,
                locator_retrieval=retrieval,
                content_floor_z=content_floor_z,
                preloaded_overview_count=preloaded_overview_count,
                overview_poses=overview_poses,
            )
            if preloaded_overview_count
            else provisional_capture_plan
        )

    fatal_errors: list[str] = []
    if tasks and inventory.status in {
        InventoryStatus.ERROR,
        InventoryStatus.UNAVAILABLE,
    }:
        message = f"inventory_{inventory.status.value}"
        capture_execution = _empty_capture_execution(tasks, runtime_error=message)
        fatal_errors.append(message)
    else:
        capture_execution = execute_stage2_capture_plan(
            capture_plan,
            frame_provider,
            store,
            content_floor_z=content_floor_z,
        )
        if capture_execution.runtime_error is not None:
            fatal_errors.append(f"capture: {capture_execution.runtime_error}")

    assessments: list[Stage2Assessment] = []
    selection_rows: list[dict[str, Any]] = []
    request_manifest: tuple[Mapping[str, Any], ...] = ()
    raw_records: tuple[Mapping[str, Any], ...] = ()

    for task in tasks:
        coverage = capture_execution.coverage_by_task.get(
            task.task_id, SearchCoverage()
        )
        preferred_roles = preferred_shot_roles_for_task(task)
        evidence = store.select_records_for_task(
            task.task_id,
            preferred_shot_roles=preferred_roles,
        )
        selected_ids = tuple(record.frame_id for record in evidence)
        row: dict[str, Any] = {
            "task_id": task.task_id,
            "node_id": task.node_id,
            "preferred_shot_roles": [value.value for value in preferred_roles],
            "selected_frame_ids": list(selected_ids),
            "coverage": coverage.to_dict(),
            "judge_attempted": False,
            "skip_reason": None,
        }

        if fatal_errors:
            reason = "evaluation_runtime_error"
            assessment = _unknown_assessment(
                task,
                coverage,
                reason,
                rationale="; ".join(fatal_errors),
                evidence_frame_ids=selected_ids,
            )
            row["skip_reason"] = reason
            assessments.append(assessment)
            selection_rows.append(row)
            continue

        precondition = _evidence_precondition_reason(
            task,
            routing,
            evidence,
        )
        if precondition is not None:
            assessment = _unknown_assessment(
                task,
                coverage,
                precondition,
                evidence_frame_ids=selected_ids,
            )
            row["skip_reason"] = precondition
            assessments.append(assessment)
            selection_rows.append(row)
            continue

        reason = "visual_evidence_incomplete"
        assessment = _unknown_assessment(
            task,
            coverage,
            reason,
            rationale=(
                "Requirement-linked visual evidence was captured; semantic "
                "interpretation is delegated to Stage 3."
            ),
            evidence_frame_ids=selected_ids,
        )
        row["skip_reason"] = "stage3_adjudication"
        assessments.append(assessment)
        selection_rows.append(row)

    evidence_selection = {
        "schema_version": "1.0",
        "max_evidence_frames_per_task": selected_budget.max_evidence_frames_per_task,
        "tasks": selection_rows,
    }
    result = Stage2Result(
        tuple(assessments),
        evaluation_error="; ".join(dict.fromkeys(fatal_errors)) or None,
        diagnostics={
            "scheduled_task_count": len(tasks),
            "retrieval_query_count": len(queries),
            "retrieval_actor_count": len(retrieval_actors),
            "retrieval_batch_count": int(bool(queries)),
            "semantic_backend_call_count": (
                counted_backend.call_count if counted_backend is not None else 0
            ),
            "semantic_backend": retrieval.semantic_backend,
            "semantic_backend_error": retrieval.semantic_backend_error,
            "capture_attempt_count": capture_execution.attempted_capture_count,
            "recovery_attempt_count": capture_execution.recovery_attempt_count,
            "valid_frame_count": store.valid_frame_count,
            "rejected_frame_count": len(store.rejections),
            "judge_attempt_count": 0,
            "judge_attempts_by_task": {
                task.task_id: 0 for task in tasks
            },
            "request_manifest_count": len(request_manifest),
            "raw_record_count": len(raw_records),
            "camera_content_floor_z": content_floor_z,
        },
    )
    return Stage2Evaluation(
        tasks,
        retrieval,
        routing,
        capture_plan,
        capture_execution,
        store,
        evidence_selection,
        request_manifest,
        raw_records,
        result,
    )


def evaluate_stage2(
    graph: RequirementGraph,
    stage1_result: Stage1Result,
    inventory: ActorInventorySnapshot,
    scene_bounds: SceneBounds,
    frame_provider: FrameProvider,
    *,
    retrieval_backend: SemanticRetrievalBackend | None = None,
    frame_store: FrameStore | None = None,
    budget: Stage2Budget | None = None,
    locator_top_k: int = 5,
    skip_node_ids: Collection[str] = (),
    eligible_actor_ids_by_entity: Mapping[str, Sequence[str]] | None = None,
) -> Stage2Result:
    """Return the semantic result of one Stage 2 evaluation."""

    return evaluate_stage2_detailed(
        graph,
        stage1_result,
        inventory,
        scene_bounds,
        frame_provider,
        retrieval_backend=retrieval_backend,
        frame_store=frame_store,
        budget=budget,
        locator_top_k=locator_top_k,
        skip_node_ids=skip_node_ids,
        eligible_actor_ids_by_entity=eligible_actor_ids_by_entity,
    ).result


__all__ = ["Stage2Evaluation", "evaluate_stage2", "evaluate_stage2_detailed"]
