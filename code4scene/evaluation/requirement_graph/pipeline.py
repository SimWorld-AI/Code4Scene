"""Code4Scene-owned Stage 1--3 controller for one frozen graph bundle."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from code4scene.evaluation import ue_evidence
from code4scene.evaluation.context import Context, read_label
from code4scene.evaluation.evaluation_policy import load_evaluation_policy
from code4scene.evaluation.semantic_scoring import (
    attach_semantic_families,
    score_semantic_case,
    semantic_exclusion_reasons,
)
from code4scene.evaluation.vlm_concurrency import (
    parallel_map as parallel_vlm_map,
    runtime_config as vlm_runtime_config,
    runtime_snapshot as vlm_runtime_snapshot,
)

from .bundle import (
    FrozenVerificationBundle,
    PopulationScope,
    RequirementEvaluationStatus,
    RequirementStatus,
    UnknownReason,
)
from .capabilities import capability_for_node
from .contracts import (
    ClaimVerdict,
    EntityGroundingMode,
    EntityNode,
    PredicateNode,
    PredicateType,
)
from .deterministic import (
    DeterministicAssessment,
    evaluate_deterministic_rules,
    evaluate_repair_collection_geometry,
    evaluate_repair_target_fields,
)
from .evidence_adapter import SceneInventoryEvidence, build_scene_inventory
from .legacy_atomic import evaluate_legacy_contract_bindings
from .repair_target_authoring import compile_repair_target_bundle
from .stage1 import Stage1Result, evaluate_stage1
from .stage2_contracts import (
    GLOBAL_CONTEXT_TASK_KINDS,
    Stage2Budget,
    Stage2TaskKind,
    Stage2Verdict,
)
from .stage2_tasks import build_stage2_tasks
from .stage3_contracts import Stage3BatchPolicy, Stage3Budget, Stage3Verdict
from .structured_deterministic import (
    StructuredEvaluation,
    evaluate_structured_requirements,
)

if TYPE_CHECKING:
    from .stage2 import Stage2Evaluation


def _ordered_concurrent_map(
    values: Sequence[Any],
    function: Callable[[Any], Any],
    *,
    max_workers: int,
) -> tuple[Any, ...]:
    """Evaluate independent I/O work concurrently and retain input order."""

    return parallel_vlm_map(
        values,
        function,
        max_workers=max_workers,
        thread_name_prefix="code4scene-stage3-vlm",
    )


def tool_client_from_env() -> Any:
    """Construct the visual client only when an unresolved claim needs it.

    Keeping this small module-level seam preserves the historical test and
    integration hook without reintroducing eager Pillow/NumPy imports during
    verifier registry loading.
    """

    from .vlm_client import tool_client_from_env as build_client

    return build_client()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return str(value)


def _path_map(values: Any) -> dict[str, str]:
    """Stable artifact-name mapping for a writer's returned paths."""

    return {Path(value).name: str(Path(value)) for value in values}


def _frame_evidence_refs(value: Mapping[str, Any]) -> list[dict[str, str]]:
    """Keep a retained frame id attached to the stage that captured it."""

    stage = value.get("evidence_stage")
    if stage not in {"stage2", "stage3"}:
        return []
    return [
        {"stage": str(stage), "evidence_id": str(frame_id), "kind": "rgb"}
        for frame_id in value.get("evidence_frame_ids", ())
    ]


def _stage3_capture_metadata(
    bundle: FrozenVerificationBundle,
    *,
    exploration: Any,
    task_by_id: Mapping[str, Any],
    focus_targets_by_task: Mapping[str, tuple[Any, ...]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split controller acquisition provenance from neutral judge metadata."""

    frame_tasks: dict[str, list[str]] = {}
    for task_id, frame_ids in exploration.task_frame_ids.items():
        for frame_id in frame_ids:
            frame_tasks.setdefault(frame_id, []).append(task_id)
    target_index: dict[str, tuple[str, Any]] = {}
    for targets in focus_targets_by_task.values():
        for target in targets:
            target_index.setdefault(
                target.actor_id.casefold(),
                (target.actor_id, target.bounds),
            )

    audit_frames: list[dict[str, Any]] = []
    visible_frames: list[dict[str, str]] = []
    for record in exploration.frame_store.records:
        task_ids = tuple(dict.fromkeys(frame_tasks.get(record.frame_id, ())))
        requirement_ids = tuple(
            dict.fromkeys(
                bundle.binding_for(task_by_id[task_id].node_id).requirement_id
                for task_id in task_ids
                if task_id in task_by_id
            )
        )
        actor_ids: list[str] = []
        target_bounds: dict[str, Any] = {}
        for actor_key in exploration.frame_actor_ids.get(record.frame_id, ()):
            canonical, bounds = target_index.get(actor_key.casefold(), (actor_key, None))
            actor_ids.append(canonical)
            if bounds is not None:
                target_bounds[canonical] = _jsonable(bounds)
        if actor_ids:
            capture_mode = "actor_focus"
            view_type = "targeted"
        elif record.shot_role.value == "grid" and task_ids:
            capture_mode = "requirement_grid"
            view_type = "grid"
        elif record.shot_role.value == "overview":
            capture_mode = "scene_overview"
            view_type = "overview"
        else:
            capture_mode = "scene_context"
            view_type = "context"
        audit_frames.append(
            {
                "frame_id": record.frame_id,
                "requirement_ids": list(requirement_ids),
                "task_ids": list(task_ids),
                "capture_target_actor_ids": actor_ids,
                "capture_target_bounds": target_bounds,
                "capture_mode": capture_mode,
                "shot_role": record.shot_role.value,
            }
        )
        visible_frames.append(
            {
                "frame_id": record.frame_id,
                "channel": "rgb",
                "view_type": view_type,
            }
        )
    return (
        {"schema_version": "1.0", "frames": audit_frames},
        {"schema_version": "1.0", "frames": visible_frames},
    )


def _production_max_tokens(client: Any, fallback: int) -> int:
    """Use the packaged deployment limit on the production graph path."""

    value = getattr(client, "max_tokens", fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("the VLM client's max_tokens must be a positive integer")
    return value


def _publish_stage2(
    output_dir: Path,
    evaluation: Any,
    bundle: FrozenVerificationBundle,
) -> dict[str, str]:
    """Publish the complete Stage 2 handoff contract."""

    from .stage2_artifacts import write_stage2_artifacts

    written = write_stage2_artifacts(
        output_dir,
        tasks=evaluation.tasks,
        identity_retrieval=evaluation.identity_retrieval,
        routing_plan=evaluation.routing_plan,
        capture_plan=evaluation.capture_artifact,
        frame_store=evaluation.frame_store,
        evidence_selection=evaluation.evidence_selection,
        request_manifest=evaluation.request_manifest,
        raw_records=evaluation.raw_records,
        result=evaluation.result,
        requirement_ids_by_task={
            task.task_id: (
                bundle.binding_for(task.node_id).requirement_id,
            )
            for task in evaluation.tasks
        },
    )
    return _path_map(written)


def _context_with_source_snapshot(context: Context) -> Context:
    """Expose the frozen Input snapshot to deterministic repair authoring."""

    if getattr(context.task, "case_type", "prompt_to_scene") != "image_to_scene":
        return context
    policy = load_evaluation_policy(context.task)
    return replace(
        context,
        spec=policy.leaf_spec(context.task, "semantic_requirements", context.spec),
    )


def _derive_image_repair_bundle(
    context: Context,
    evidence: ue_evidence.SceneEvidence,
) -> FrozenVerificationBundle:
    if evidence.input_scene is None:
        raise ValueError(
            "image_to_scene semantic verification needs the frozen Input "
            "scene snapshot in evaluation_policy.source_snapshot"
        )
    label = read_label(context)
    if label is None:
        raise ValueError(
            "image_to_scene semantic verification needs the scene_diff "
            "answer-key label containing canonical_actors"
        )
    canonical = label.get("canonical_actors")
    if not isinstance(canonical, list) or not canonical or any(
        not isinstance(actor, Mapping) for actor in canonical
    ):
        raise ValueError(
            "image_to_scene answer key must contain canonical_actors"
        )
    gt_scene = {
        "actors": canonical,
        "actor_count": len(canonical),
        "export_metadata": {
            "status": "success",
            "source": "scene_diff_answer_key",
        },
    }
    return compile_repair_target_bundle(
        evidence.input_scene,
        gt_scene,
        task_id=str(context.task.id),
    )


def load_bundle(
    context: Context,
    evidence: ue_evidence.SceneEvidence | None = None,
) -> FrozenVerificationBundle:
    if getattr(context.task, "case_type", "prompt_to_scene") == "image_to_scene":
        if evidence is None:
            raise ValueError(
                "image_to_scene bundle derivation needs collected Input evidence"
            )
        return _derive_image_repair_bundle(context, evidence)

    configured = context.spec.get("verification_bundle")
    if configured is None:
        configured = context.spec.get("requirement_graph_bundle")
    if configured is None:
        raise ValueError(
            "semantic_requirements needs a frozen verification_bundle; "
            "runtime prompt interpretation is not permitted"
        )
    document = ue_evidence.document(context, configured, "verification bundle")
    if not isinstance(document, Mapping):
        raise ValueError("verification_bundle must be a JSON object")
    bundle = FrozenVerificationBundle.from_dict(document)
    prompt = str(context.task.prompt)
    if bundle.prompt != prompt:
        raise ValueError("frozen bundle prompt does not equal the task prompt")
    return bundle


def _budget(value: Any, cls: type[Any], name: str) -> Any:
    if value is None:
        return cls()
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cls(**dict(value))


def _default_stage2_budget(bundle: FrozenVerificationBundle) -> Stage2Budget:
    """Keep every graph entity eligible for its finite evidence shape."""

    if bundle.provenance.get("authoring_parser") == (
        "deterministic_gt_input_scene_diff"
    ):
        return Stage2Budget(global_capture_limits_enabled=False)
    return Stage2Budget.graph_default()


def _stage2_unknown_reason(reason: str | None) -> UnknownReason:
    normalized = str(reason or "").casefold()
    if any(
        token in normalized
        for token in ("judge_error", "runtime_error", "inventory_error")
    ):
        return UnknownReason.EVALUATION_ERROR
    if "target_not_localized" in normalized:
        return UnknownReason.TARGET_NOT_LOCALIZED
    if normalized in {"no_valid_evidence", "targeted_evidence_missing"}:
        return UnknownReason.TARGETED_EVIDENCE_MISSING
    return UnknownReason.VISUAL_EVIDENCE_INCOMPLETE


def _requirement_evaluation_status(value: Mapping[str, Any]) -> str:
    """Lower internal routing state to a public terminal result."""

    verdict = ClaimVerdict.coerce(value["verdict"])
    if verdict is ClaimVerdict.MATCH:
        return RequirementEvaluationStatus.MATCH.value
    if verdict is ClaimVerdict.MISMATCH:
        return RequirementEvaluationStatus.MISMATCH.value
    reason = str(value.get("unknown_reason") or "")
    if reason == UnknownReason.EVALUATION_ERROR.value:
        return RequirementEvaluationStatus.ERROR.value
    if reason in {
        UnknownReason.TARGET_NOT_LOCALIZED.value,
        UnknownReason.TARGETED_EVIDENCE_MISSING.value,
        UnknownReason.SCOPE_UNRESOLVED.value,
        UnknownReason.PROVENANCE_UNTRUSTED.value,
        UnknownReason.UNSUPPORTED.value,
        UnknownReason.NON_VISUAL.value,
        UnknownReason.VISUAL_EVIDENCE_INCOMPLETE.value,
    }:
        return RequirementEvaluationStatus.NOT_EVALUATED.value
    return RequirementEvaluationStatus.NOT_EVALUATED.value


def _atomic_score(value: Mapping[str, Any]) -> float | None:
    verdict = ClaimVerdict.coerce(value["verdict"])
    if verdict is ClaimVerdict.UNKNOWN:
        return None
    check = value.get("check")
    raw = check.get("score") if isinstance(check, Mapping) else None
    if not isinstance(raw, bool) and isinstance(raw, (int, float)):
        score = float(raw)
        if math.isfinite(score) and 0.0 <= score <= 1.0:
            return score
    return 1.0 if verdict is ClaimVerdict.MATCH else 0.0


def _retrieval_options(context: Context) -> tuple[Any | None, int]:
    """Build the configured locator-only backend without changing the rubric."""

    raw = context.spec.get("semantic_retrieval")
    if raw is None:
        return None, 5
    if not isinstance(raw, Mapping):
        raise ValueError("semantic_retrieval must be an object")
    mode = str(raw.get("mode", "lexical")).strip().casefold()
    top_k = raw.get("top_k", 5)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("semantic_retrieval.top_k must be a positive integer")
    if mode == "lexical":
        return None, top_k
    if mode == "llm":
        from .llm_identity_locator import LLMIdentityLocatorBackend

        max_tokens = raw.get("max_tokens")
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError(
                "semantic_retrieval.max_tokens must be a positive integer"
            )
        return LLMIdentityLocatorBackend(
            tool_client_from_env(),
            max_tokens=max_tokens,
        ), top_k
    if mode != "local_embedding":
        raise ValueError(
            "semantic_retrieval.mode must be lexical, llm, or local_embedding"
        )
    from .semantic_retrieval import LocalEmbeddingServiceBackend

    endpoint = str(
        raw.get("endpoint", "http://127.0.0.1:7777/embed")
    ).strip()
    timeout = raw.get("timeout_seconds", 30.0)
    return LocalEmbeddingServiceBackend(
        endpoint=endpoint,
        timeout_seconds=float(timeout),
    ), top_k


def _stage1_atomic(
    bundle: FrozenVerificationBundle,
    stage1: Stage1Result,
    scene: SceneInventoryEvidence,
    deterministic: tuple[DeterministicAssessment, ...],
) -> dict[str, dict[str, Any]]:
    results = {value.node_id: value.to_dict() for value in deterministic}
    for binding in bundle.requirements:
        if (
            binding.primary_owner != "requirement_graph"
            or binding.status is not RequirementStatus.SUPPORTED
            or binding.node_id in results
        ):
            continue
        unresolved = [
            scene.scope(scope)
            for scope in set(binding.entity_scopes.values())
            if not scene.scope(scope).resolved
        ]
        node = bundle.graph.node(binding.node_id)
        capability_route = capability_for_node(bundle.graph, node).planner_route
        assessment = None
        if isinstance(node, EntityNode):
            assessment = stage1.for_entity(node.id)
        elif isinstance(node, PredicateNode) and node.predicate_type is PredicateType.EXISTENCE:
            assessment = stage1.for_predicate(node.id)
        if unresolved:
            first = unresolved[0]
            results[binding.node_id] = {
                "node_id": binding.node_id,
                "requirement_id": binding.requirement_id,
                "verdict": ClaimVerdict.UNKNOWN.value,
                "resolved_by": None,
                "unknown_reason": (
                    first.unknown_reason or UnknownReason.SCOPE_UNRESOLVED
                ).value,
                "rationale": first.detail,
                "capability_route": capability_route,
            }
        elif assessment is not None:
            results[binding.node_id] = {
                "node_id": binding.node_id,
                "requirement_id": binding.requirement_id,
                "verdict": assessment.verdict.value,
                "resolved_by": (
                    "stage1" if assessment.verdict is not ClaimVerdict.UNKNOWN else None
                ),
                "unknown_reason": (
                    None
                    if assessment.verdict is not ClaimVerdict.UNKNOWN
                    else UnknownReason.VISUAL_EVIDENCE_INCOMPLETE.value
                ),
                "rationale": assessment.rationale,
                "stage1_assessment_id": assessment.assessment_id,
                "actor_ids": list(assessment.actor_ids),
                "grounded_actor_ids": list(assessment.grounded_actor_ids),
                "name_grounding_audit": [
                    match.to_dict() for match in assessment.grounding_matches
                ],
                "capability_route": capability_route,
            }
        else:
            results[binding.node_id] = {
                "node_id": binding.node_id,
                "requirement_id": binding.requirement_id,
                "verdict": ClaimVerdict.UNKNOWN.value,
                "resolved_by": None,
                "unknown_reason": UnknownReason.VISUAL_EVIDENCE_INCOMPLETE.value,
                "rationale": "this visual predicate is not decidable from structured metadata",
                "capability_route": capability_route,
            }
    return results


def _scope_actor_ids(
    bundle: FrozenVerificationBundle,
    scene: SceneInventoryEvidence,
) -> tuple[dict[str, tuple[str, ...]], set[str]]:
    values: dict[str, tuple[str, ...]] = {}
    unresolved: set[str] = set()
    repair_targets = bundle.provenance.get("repair_target_by_entity")
    repair_targets = repair_targets if isinstance(repair_targets, Mapping) else {}
    live_actor_ids = {
        actor.live_actor_id.casefold(): actor.live_actor_id
        for actor in scene.inventory.actors
    }
    for entity_id, scope in bundle.semantic_entity_scopes().items():
        resolution = scene.scope(scope)
        actor_ids = resolution.actor_ids if resolution.resolved else ()
        target = repair_targets.get(entity_id)
        if isinstance(target, Mapping) and target.get("operation") != "add":
            # A repair agent may retain the original Actor or replace it with
            # a semantically equivalent Actor. Keep the stable original when
            # present and also search Candidate additions; Stage 2/3, rather
            # than asset-name equality, decides visible substitution quality.
            identity = str(
                target.get("gt_actor_identity")
                or target.get("input_actor_identity")
                or ""
            ).strip()
            matched = live_actor_ids.get(identity.casefold())
            additions = scene.scope(PopulationScope.ADDITIONS)
            actor_ids = tuple(
                dict.fromkeys(
                    (
                        *((matched,) if matched is not None else ()),
                        *(additions.actor_ids if additions.resolved else ()),
                    )
                )
            )
        values[entity_id] = actor_ids
        if not resolution.resolved:
            unresolved.add(entity_id)
    return values, unresolved


def _stage2_skip_nodes(
    bundle: FrozenVerificationBundle,
    atomic: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    semantic = set(bundle.semantic_nodes())
    skip = set(bundle.graph.effective_weights()) - semantic
    for node_id, value in atomic.items():
        verdict = ClaimVerdict.coerce(value["verdict"])
        reason = value.get("unknown_reason")
        if verdict is not ClaimVerdict.UNKNOWN:
            skip.add(node_id)
        elif reason != UnknownReason.VISUAL_EVIDENCE_INCOMPLETE.value:
            skip.add(node_id)
    return skip


def _semantic_pre_evidence_exclusions(
    bundle: FrozenVerificationBundle,
    atomic: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Apply the final score policy's static exclusions before visual work."""

    rows: list[dict[str, Any]] = []
    for binding in bundle.requirements:
        if (
            binding.status is not RequirementStatus.SUPPORTED
            or binding.node_id not in atomic
        ):
            continue
        rows.append(
            {
                **atomic[binding.node_id],
                "text": binding.source_text,
                "source_span": list(binding.source_span),
                "population_scope": binding.population_scope.value,
                "entity_scopes": {
                    key: value.value
                    for key, value in binding.entity_scopes.items()
                },
                "evaluation_binding": bundle.evaluation_for(
                    binding.node_id
                ).to_dict(),
            }
        )
    scoring_rows = attach_semantic_families(bundle.graph, rows)
    return semantic_exclusion_reasons(bundle.prompt, scoring_rows)


def _overlay_stage2(
    atomic: dict[str, dict[str, Any]], evaluation: Stage2Evaluation
) -> None:
    for assessment in evaluation.result.assessments:
        reason = (
            _stage2_unknown_reason(assessment.unknown_reason)
            if assessment.verdict is Stage2Verdict.UNKNOWN
            else None
        )
        atomic[assessment.node_id].update(
            {
                "verdict": assessment.verdict.value,
                "resolved_by": (
                    "stage2"
                    if assessment.verdict is not Stage2Verdict.UNKNOWN
                    else None
                ),
                "unknown_reason": reason.value if reason is not None else None,
                "rationale": assessment.rationale,
                "confidence": assessment.confidence,
                "evidence_frame_ids": list(assessment.evidence_frame_ids),
                "evidence_stage": (
                    "stage2" if assessment.evidence_frame_ids else None
                ),
                "stage2_task_id": assessment.task_id,
            }
        )


def _identity_subject_id(bundle: FrozenVerificationBundle, node_id: str) -> str | None:
    node = bundle.graph.node(node_id)
    if isinstance(node, EntityNode):
        return node.id
    if not isinstance(node, PredicateNode) or node.predicate_type not in {
        PredicateType.EXISTENCE,
        PredicateType.COUNT,
    }:
        return None
    arguments = bundle.graph.arguments_for(node.id)
    return arguments[0].target_id if arguments else None


def _overlay_identity_grounding(
    bundle: FrozenVerificationBundle,
    atomic: dict[str, dict[str, Any]],
    grounding: Any,
) -> None:
    """Resolve existence identity only; Stage 4 owns every measured facet."""

    for binding in bundle.requirements:
        entity_id = _identity_subject_id(bundle, binding.node_id)
        if entity_id is None or binding.node_id not in atomic:
            continue
        node = bundle.graph.node(binding.node_id)
        if not (
            isinstance(node, PredicateNode)
            and node.predicate_type is PredicateType.EXISTENCE
        ):
            # Binding an Actor proves only its semantic identity. It must not
            # turn count, attribute, material, orientation, or relation into
            # MATCH. Those facets are measured after binding in Stage 4.
            continue
        identity = grounding.for_entity(entity_id)
        if identity is None:
            continue
        if identity.evaluation_error is not None:
            atomic[binding.node_id].update(
                verdict=ClaimVerdict.UNKNOWN.value,
                resolved_by=None,
                unknown_reason=UnknownReason.EVALUATION_ERROR.value,
                rationale=identity.evaluation_error,
            )
            continue
        if not identity.complete and not identity.search_exhausted:
            atomic[binding.node_id].update(
                verdict=ClaimVerdict.UNKNOWN.value,
                resolved_by=None,
                unknown_reason=UnknownReason.TARGET_NOT_LOCALIZED.value,
                rationale=(
                    "semantic identity search could not establish a complete "
                    "Candidate Actor population"
                ),
            )
            continue
        score = float(identity.matched_count > 0)
        verdict = (
            ClaimVerdict.MATCH
            if math.isclose(score, 1.0, rel_tol=0.0, abs_tol=1e-9)
            else ClaimVerdict.MISMATCH
        )
        atomic[binding.node_id].update(
            verdict=verdict.value,
            resolved_by="identity_grounding",
            unknown_reason=None,
            rationale=(
                f"confirmed {identity.matched_count} distinct Actor(s) for "
                f"semantic entity {identity.expected_name!r}"
            ),
            confidence=(
                min(
                    (value.confidence for value in identity.candidate_assessments),
                    default=1.0,
                )
            ),
            actor_ids=list(identity.matched_actor_ids),
            check={
                "measurement": "semantic_identity_population",
                "score": score,
                "matched_count": identity.matched_count,
                "required_count": identity.required_count,
                "exact_actor_ids": list(identity.exact_actor_ids),
                "visual_actor_ids": list(identity.visual_actor_ids),
            },
        )


def _overlay_grounded_deterministic(
    bundle: FrozenVerificationBundle,
    atomic: dict[str, dict[str, Any]],
    assessments: tuple[DeterministicAssessment, ...],
    bound_entity_ids: set[str],
) -> None:
    """Replace only leaves whose geometry now has identity-bound populations."""

    for assessment in assessments:
        if assessment.node_id not in atomic:
            continue
        dependencies = {
            edge.target_id
            for edge in bundle.graph.arguments_for(assessment.node_id)
        }
        if not dependencies.intersection(bound_entity_ids):
            continue
        value = assessment.to_dict()
        if assessment.verdict is not ClaimVerdict.UNKNOWN:
            value["resolved_by"] = "identity_bound_scene_graph"
        atomic[assessment.node_id].update(value)


def _run_stage3(
    bundle: FrozenVerificationBundle,
    atomic: dict[str, dict[str, Any]],
    stage2: Stage2Evaluation,
    provider: Any,
    client: Any,
    artifacts: Path,
    budget: Stage3Budget | None,
    batch_policy: Stage3BatchPolicy,
    identity_grounding: Any | None = None,
    holistic_stress_policy: str | None = None,
) -> dict[str, Any] | None:
    # These imports own NumPy/Pillow.  Keeping them behind the visual-stage
    # branch lets a base Code4Scene installation import and run Stage 1.
    from .stage3 import (
        _coerce_holistic_result,
        aggregate_stage3_batch_passes,
        evaluate_stage3_binary_arbitration,
        evaluate_stage3_unknown_batches,
    )
    from .stage2_capture import cluster_stage2_targets
    from .stage3_artifacts import write_stage3_artifacts
    from .stage3_explorer import (
        Stage3Explorer,
        capture_stage3_actor_reframes,
    )
    from .stage3_judge import LLMHolisticJudge, LLMStage3UnknownJudge

    runtime_limit = vlm_runtime_config().max_concurrency
    if batch_policy.max_concurrent_claims != runtime_limit:
        batch_policy = replace(
            batch_policy,
            max_concurrent_claims=runtime_limit,
        )

    task_by_id = {value.task_id: value for value in stage2.tasks}
    eligible = []
    for assessment in stage2.result.assessments:
        task = task_by_id[assessment.task_id]
        production_kinds = {
            Stage2TaskKind.ATTRIBUTE,
            Stage2TaskKind.MATERIAL,
            Stage2TaskKind.ATMOSPHERE,
            Stage2TaskKind.SCENE_IDENTITY,
        }
        if holistic_stress_policy is None:
            production_kinds.add(Stage2TaskKind.VISUAL_GLOBAL)
        if identity_grounding is None:
            # Compatibility for direct legacy callers. The production
            # run_pipeline always supplies identity_grounding and therefore
            # never lets the whole-claim judge own identity/count/geometry.
            production_kinds.update(
                {
                    Stage2TaskKind.OBJECT_EXISTENCE,
                    Stage2TaskKind.COUNT,
                    Stage2TaskKind.SPATIAL_RELATION,
                }
            )
        if task.kind not in production_kinds:
            continue
        incomplete_identity = (
            identity_grounding is not None
            and any(
                (value := identity_grounding.for_entity(entity_id)) is not None
                and not value.complete
                for entity_id in task.dependency_entity_ids
            )
        )
        unknown_reason = atomic[assessment.node_id].get("unknown_reason")
        has_legal_overview_fallback = (
            unknown_reason == UnknownReason.TARGETED_EVIDENCE_MISSING.value
            and task.allows_overview_fallback
        )
        if (
            not incomplete_identity
            and assessment.verdict is Stage2Verdict.UNKNOWN
            and (
                unknown_reason == UnknownReason.VISUAL_EVIDENCE_INCOMPLETE.value
                or has_legal_overview_fallback
            )
        ):
            eligible.append(assessment)
    if not eligible and holistic_stress_policy is None:
        return None
    if client is None:
        client = tool_client_from_env()
    bound_actor_ids = (
        identity_grounding.actor_ids_by_entity
        if identity_grounding is not None
        else {}
    )
    stage2_capture_plan = getattr(stage2, "capture_plan", None)
    stage2_focus_budget = getattr(
        stage2_capture_plan,
        "budget",
        Stage2Budget.graph_default(),
    )
    focus_targets_by_task: dict[str, tuple[Any, ...]] = {}
    routing_plan = getattr(stage2, "routing_plan", None)
    for assessment in eligible:
        task = task_by_id[assessment.task_id]
        selected: dict[str, Any] = {}
        for entity_id in task.dependency_entity_ids:
            route = (
                routing_plan.for_entity(entity_id)
                if routing_plan is not None
                else None
            )
            if route is None:
                continue
            allowed_ids = (
                {value.casefold() for value in bound_actor_ids[entity_id]}
                if entity_id in bound_actor_ids
                else None
            )
            eligible_targets = tuple(
                target
                for target in route.targets
                if allowed_ids is None
                or target.actor_id.casefold() in allowed_ids
            )
            entity = bundle.graph.node(entity_id)
            target_limit = (
                stage2_focus_budget.max_collection_representatives
                if isinstance(entity, EntityNode)
                and entity.effective_grounding_mode
                is EntityGroundingMode.ACTOR_COLLECTION
                else stage2_focus_budget.max_individual_representatives
            )
            representative_targets = (
                cluster_stage2_targets(
                    eligible_targets,
                    limit=target_limit,
                )
                if eligible_targets
                else ()
            )
            for target in representative_targets:
                selected.setdefault(target.actor_id.casefold(), target)
        focus_targets_by_task[task.task_id] = tuple(selected.values())
    exploration_options: dict[str, Any] = {
        "focus_targets_by_task": focus_targets_by_task,
        "require_global_context": any(
            task_by_id[assessment.task_id].kind in GLOBAL_CONTEXT_TASK_KINDS
            for assessment in eligible
        )
        or holistic_stress_policy is not None,
        "grid_search_task_ids": tuple(
            assessment.task_id
            for assessment in eligible
            if not focus_targets_by_task.get(assessment.task_id)
            and task_by_id[assessment.task_id].kind
            not in GLOBAL_CONTEXT_TASK_KINDS
        ),
    }
    from .actor_inventory import (
        ActorInventorySnapshot,
        estimate_camera_content_floor_z,
    )

    inventory = getattr(provider.scene, "inventory", None)
    if isinstance(inventory, ActorInventorySnapshot):
        exploration_options["content_floor_z"] = estimate_camera_content_floor_z(
            inventory,
            provider.scene.scene_bounds,
        )
    unique_focus_actor_count = len(
        {
            target.actor_id.casefold()
            for targets in focus_targets_by_task.values()
            for target in targets
        }
    )
    if budget is None:
        budget = Stage3Budget.for_requirement_graph(
            seed_frame_count=len(stage2.frame_store.records),
            focus_actor_count=unique_focus_actor_count,
            grid_task_count=len(exploration_options["grid_search_task_ids"]),
            require_global_context=bool(
                exploration_options["require_global_context"]
            ),
        )
    explorer = Stage3Explorer(budget=budget)
    adaptive_reframe_enabled = batch_policy.max_batches_per_claim >= 2
    automatic_reframe_reserve = min(
        unique_focus_actor_count,
        budget.max_valid_exploration_frames - budget.min_portfolio_frames,
    )
    exploration = explorer.explore(
        provider.scene.scene_bounds,
        provider,
        stage2_frame_store=stage2.frame_store,
        adaptive_reframe_reserve=(
            automatic_reframe_reserve if adaptive_reframe_enabled else 0
        ),
        **exploration_options,
    )
    resolutions: list[Any] = []
    batch_evaluations: list[dict[str, Any]] = []
    judge = LLMStage3UnknownJudge(
        client,
        max_tokens=_production_max_tokens(client, 640),
        max_frames_per_request=batch_policy.max_frames_per_request,
    )

    def update_atomic_from_resolution(task: Any, resolution: Any) -> None:
        if resolution.evaluation_error is not None:
            atomic[task.node_id].update(
                verdict=ClaimVerdict.UNKNOWN.value,
                resolved_by=None,
                unknown_reason=UnknownReason.EVALUATION_ERROR.value,
                rationale=resolution.evaluation_error,
            )
            return
        final = resolution.final_verdict or Stage3Verdict.UNKNOWN
        atomic[task.node_id].update(
            verdict=final.value,
            resolved_by=("stage3" if final is not Stage3Verdict.UNKNOWN else None),
            unknown_reason=(
                UnknownReason.VISUAL_EVIDENCE_INCOMPLETE.value
                if final is Stage3Verdict.UNKNOWN
                else None
            ),
            rationale=resolution.rationale,
            confidence=resolution.confidence,
            evidence_frame_ids=[
                value.evidence_id for value in resolution.evidence_refs
            ],
            evidence_stage=("stage3" if resolution.evidence_refs else None),
            forced_mismatch=resolution.forced_mismatch,
            forced_reason=resolution.forced_reason,
        )

    if exploration.runtime_error:
        for assessment in eligible:
            atomic[assessment.node_id].update(
                unknown_reason=UnknownReason.EVALUATION_ERROR.value,
                rationale=f"Stage 3 capture failed: {exploration.runtime_error}",
            )
    else:
        single_batch_policy = Stage3BatchPolicy(
            policy_id=batch_policy.policy_id,
            max_frames_per_request=batch_policy.max_frames_per_request,
            max_batches_per_claim=1,
            max_concurrent_claims=batch_policy.max_concurrent_claims,
            schema_version=batch_policy.schema_version,
        )
        completed: dict[str, tuple[Any, dict[str, Any]]] = {}
        scheduled: dict[str, dict[str, Any]] = {}
        first_pass_jobs: list[tuple[str, Any, tuple[Any, ...], Any, bool]] = []
        for assessment in eligible:
            task = task_by_id[assessment.task_id]
            routed_frame_ids = tuple(
                exploration.task_frame_ids.get(task.task_id, ())
            )
            if (
                task.kind not in GLOBAL_CONTEXT_TASK_KINDS
                and not routed_frame_ids
            ):
                first_pass_jobs.append(
                    (task.task_id, task, (), batch_policy, False)
                )
                continue
            frames = exploration.judge_frames_for_task(task.task_id)
            if not (
                adaptive_reframe_enabled
                and focus_targets_by_task.get(task.task_id)
            ):
                first_pass_jobs.append(
                    (task.task_id, task, tuple(frames), batch_policy, False)
                )
                continue

            target_only_first_pass = (
                task.kind
                in {
                    Stage2TaskKind.OBJECT_EXISTENCE,
                    Stage2TaskKind.ATTRIBUTE,
                    Stage2TaskKind.MATERIAL,
                    Stage2TaskKind.COUNT,
                }
                and bool(focus_targets_by_task.get(task.task_id))
            )
            if target_only_first_pass:
                frame_by_id = {
                    str(value.frame_id): value for value in frames
                }
                routed_focus_frames = tuple(
                    frame_by_id[frame_id]
                    for frame_id in routed_frame_ids
                    if frame_id in frame_by_id
                )
            else:
                routed_focus_frames = ()
            first_frames = tuple(
                (routed_focus_frames or frames)[
                    : batch_policy.max_frames_per_request
                ]
            )
            scheduled[task.task_id] = {
                "task": task,
                "initial_frames": tuple(frames),
                "route_ids_before": routed_frame_ids,
                "target_only_first_pass": target_only_first_pass,
                "passes": [],
                "reframe_recommended": False,
            }
            first_pass_jobs.append(
                (
                    task.task_id,
                    task,
                    first_frames,
                    single_batch_policy,
                    True,
                )
            )

        first_pass_results = _ordered_concurrent_map(
            first_pass_jobs,
            lambda job: evaluate_stage3_unknown_batches(
                bundle.graph,
                job[1],
                job[2],
                judge,
                job[3],
            ),
            max_workers=batch_policy.max_concurrent_claims,
        )
        for job, first_pass in zip(
            first_pass_jobs,
            first_pass_results,
            strict=True,
        ):
            task_id, _, _, _, is_scheduled = job
            if not is_scheduled:
                completed[task_id] = first_pass
                continue
            state = scheduled[task_id]
            state["passes"].append(first_pass)
            state["reframe_recommended"] = any(
                bool(value.get("reframe_recommended"))
                for value in first_pass[1].get("batches", ())
            )

        reframe_targets: dict[str, tuple[Any, ...]] = {}
        followup_jobs: list[tuple[str, Any, tuple[Any, ...], Any]] = []
        for task_id, state in scheduled.items():
            first_resolution = state["passes"][0][0]
            if (
                first_resolution.evaluation_error is not None
                or first_resolution.final_verdict is not Stage3Verdict.UNKNOWN
                or not state["reframe_recommended"]
            ):
                continue
            first_frame_ids = {
                str(value.frame_id)
                for value in state["initial_frames"][
                    : batch_policy.max_frames_per_request
                ]
            }
            visible_actor_ids = {
                actor_id.casefold()
                for frame_id in first_frame_ids
                for actor_id in exploration.frame_actor_ids.get(frame_id, ())
            }
            targets = tuple(
                value
                for value in focus_targets_by_task[task_id]
                if not visible_actor_ids
                or value.actor_id.casefold() in visible_actor_ids
            )
            reframe_targets[task_id] = (
                targets if targets else focus_targets_by_task[task_id]
            )

        if reframe_targets:
            exploration = capture_stage3_actor_reframes(
                exploration,
                provider.scene.scene_bounds,
                provider,
                targets_by_task=reframe_targets,
                content_floor_z=exploration_options.get("content_floor_z"),
            )

        for task_id, state in scheduled.items():
            task = state["task"]
            first_resolution = state["passes"][0][0]
            second_frames: tuple[Any, ...] = ()
            if first_resolution.evaluation_error is None:
                consumed_ids = set(
                    state["passes"][0][1].get("selected_frame_ids", ())
                )
                if (
                    first_resolution.final_verdict is Stage3Verdict.UNKNOWN
                    and state["reframe_recommended"]
                ):
                    refreshed_frames = exploration.judge_frames_for_task(task_id)
                    frame_by_id = {
                        str(value.frame_id): value for value in refreshed_frames
                    }
                    route_ids_after = tuple(
                        exploration.task_frame_ids.get(task_id, ())
                    )
                    route_ids_before = state["route_ids_before"]
                    new_ids = tuple(
                        value
                        for value in route_ids_after
                        if value not in set(route_ids_before)
                    )
                    if new_ids:
                        trailing_ids = (
                            ()
                            if state["target_only_first_pass"]
                            else tuple(
                                str(value.frame_id)
                                for value in refreshed_frames
                            )
                        )
                        ordered_ids = tuple(
                            dict.fromkeys(
                                (
                                    *new_ids,
                                    *route_ids_before,
                                    *trailing_ids,
                                )
                            )
                        )
                        second_frames = tuple(
                            frame_by_id[value]
                            for value in ordered_ids
                            if value in frame_by_id and value not in consumed_ids
                        )
                if not second_frames:
                    refreshed_frames = exploration.judge_frames_for_task(task_id)
                    second_frames = tuple(
                        value
                        for value in refreshed_frames
                        if str(value.frame_id) not in consumed_ids
                    )
            if second_frames:
                used_batches = sum(
                    len(value[1].get("batches", ())) for value in state["passes"]
                )
                remaining_batches = batch_policy.max_batches_per_claim - used_batches
                followup_policy = Stage3BatchPolicy(
                    policy_id=batch_policy.policy_id,
                    max_frames_per_request=batch_policy.max_frames_per_request,
                    max_batches_per_claim=max(1, remaining_batches),
                    max_concurrent_claims=batch_policy.max_concurrent_claims,
                    schema_version=batch_policy.schema_version,
                )
                followup_jobs.append(
                    (task_id, task, second_frames, followup_policy)
                )

        followup_results = _ordered_concurrent_map(
            followup_jobs,
            lambda job: evaluate_stage3_unknown_batches(
                bundle.graph,
                job[1],
                job[2],
                judge,
                job[3],
            ),
            max_workers=batch_policy.max_concurrent_claims,
        )
        for job, followup in zip(
            followup_jobs,
            followup_results,
            strict=True,
        ):
            scheduled[job[0]]["passes"].append(followup)

        for task_id, state in scheduled.items():
            task = state["task"]
            available_frames = exploration.judge_frames_for_task(task_id)
            completed[task_id] = aggregate_stage3_batch_passes(
                task,
                state["passes"],
                batch_policy,
                available_frame_ids=tuple(
                    str(value.frame_id) for value in available_frames
                ),
            )

        binary_job_values = []
        for assessment in eligible:
            result = completed.get(assessment.task_id)
            if (
                result is not None
                and result[0].final_verdict is Stage3Verdict.UNKNOWN
            ):
                binary_job_values.append(task_by_id[assessment.task_id])
        binary_jobs = tuple(binary_job_values)
        binary_results = _ordered_concurrent_map(
            binary_jobs,
            lambda task: evaluate_stage3_binary_arbitration(
                bundle.graph,
                task,
                exploration.judge_frames_for_task(task.task_id),
                judge,
            ),
            max_workers=batch_policy.max_concurrent_claims,
        )
        binary_by_task_id = dict(
            zip(
                (task.task_id for task in binary_jobs),
                binary_results,
                strict=True,
            )
        )

        for assessment in eligible:
            task = task_by_id[assessment.task_id]
            result = completed.get(task.task_id)
            if result is None:
                continue
            resolution, batch_evaluation = result
            batch_evaluation = dict(batch_evaluation)
            if task.task_id in binary_by_task_id:
                resolution, binary_audit = binary_by_task_id[task.task_id]
                batch_evaluation["binary_arbitration"] = binary_audit
            batch_evaluation["aggregation"] = resolution.to_dict()
            resolutions.append(resolution)
            batch_evaluations.append(batch_evaluation)
            update_atomic_from_resolution(task, resolution)

    holistic_adapter = None
    holistic_owner = "vlm_as_judge"
    holistic_result: Any = {
        "status": "delegated",
        "primary_owner": holistic_owner,
        "overall_score": None,
        "evaluation_error": None,
    }
    if holistic_stress_policy is not None and exploration.runtime_error is None:
        prompt = bundle.graph.prompt
        semantic_view = bundle.provenance.get("semantic_prompt_view")
        if isinstance(semantic_view, Mapping):
            source_span = semantic_view.get("source_span")
            if (
                isinstance(source_span, Mapping)
                and isinstance(source_span.get("start"), int)
                and isinstance(source_span.get("end"), int)
            ):
                start = int(source_span["start"])
                end = int(source_span["end"])
                if 0 <= start < end <= len(prompt):
                    prompt = prompt[start:end]
            elif (
                isinstance(source_span, (list, tuple))
                and len(source_span) == 2
                and all(isinstance(value, int) for value in source_span)
            ):
                start, end = source_span
                if 0 <= start < end <= len(prompt):
                    prompt = prompt[start:end]
        holistic_adapter = LLMHolisticJudge(
            client,
            max_tokens=_production_max_tokens(client, 2048),
        )
        holistic_result = _coerce_holistic_result(
            holistic_adapter.judge(prompt, exploration.judge_frames())
        )
        holistic_owner = f"semantic_requirements.{holistic_stress_policy}"

    weights = bundle.graph.effective_weights()
    final_assessments = []
    for node_id in bundle.semantic_nodes():
        value = atomic[node_id]
        final_assessments.append(
            {
                "node_id": node_id,
                "weight": weights[node_id],
                "verdict": value["verdict"],
                "score": _atomic_score(value),
                "evaluation_status": _requirement_evaluation_status(value),
                "decision_source": value.get("resolved_by") or "unresolved",
                "confidence": value.get("confidence"),
                "evidence_refs": _frame_evidence_refs(value),
                "rationale": value.get("rationale"),
                "forced_mismatch": bool(value.get("forced_mismatch", False)),
                "forced_reason": value.get("forced_reason"),
            }
        )
    decided = tuple(
        value
        for value in final_assessments
        if value["evaluation_status"]
        in {
            RequirementEvaluationStatus.MATCH.value,
            RequirementEvaluationStatus.MISMATCH.value,
        }
    )
    legacy_semantic_score = (
        sum(float(value["score"]) for value in decided) / len(decided)
        if decided
        else None
    )
    legacy_semantic_coverage = (
        len(decided) / len(final_assessments) if final_assessments else 0.0
    )
    total_weight = sum(float(value["weight"]) for value in final_assessments)
    scored_weight = sum(
        float(value["weight"]) * float(value["score"])
        for value in decided
    )
    semantic_rows = []
    for value in final_assessments:
        binding = bundle.binding_for(value["node_id"])
        semantic_rows.append(
            {
                **value,
                "text": binding.source_text,
                "source_span": list(binding.source_span),
                "evaluation_binding": bundle.evaluation_for(
                    value["node_id"]
                ).to_dict(),
            }
        )
    semantic_aggregation = score_semantic_case(
        bundle.prompt,
        attach_semantic_families(bundle.graph, semantic_rows),
    )
    semantic_score = semantic_aggregation["score"]
    semantic_coverage = semantic_aggregation["known_coverage"]
    resolution_errors = tuple(
        value.evaluation_error
        for value in resolutions
        if value.evaluation_error is not None
    )
    stage3_evaluation_error = exploration.runtime_error
    if resolution_errors:
        joined = "; ".join(dict.fromkeys(resolution_errors))
        stage3_evaluation_error = (
            f"{stage3_evaluation_error}; {joined}"
            if stage3_evaluation_error
            else joined
        )
    holistic_score = getattr(holistic_result, "overall_score", None)
    if isinstance(holistic_result, Mapping):
        holistic_score = holistic_result.get("overall_score")
    final_result = {
        "schema_version": "1.0",
        "status": (
            "evaluation_error"
            if stage3_evaluation_error
            else "not_evaluated" if semantic_coverage < 1.0 else "complete"
        ),
        "evaluation_error": stage3_evaluation_error,
        "requirements_score": semantic_score,
        "semantic_score": semantic_score,
        "coverage": semantic_coverage,
        "decided_requirement_count": len(decided),
        "total_requirement_count": len(final_assessments),
        "weighted_semantic_score": semantic_score,
        "weighted_coverage": semantic_coverage,
        "score_policy": semantic_aggregation["score_policy"],
        "known_coverage": semantic_coverage,
        "legacy_decided_only_semantic_score": legacy_semantic_score,
        "legacy_decided_only_coverage": legacy_semantic_coverage,
        "legacy_weighted_decided_score": (
            scored_weight
            / sum(float(value["weight"]) for value in decided)
            if decided
            else None
        ),
        "legacy_weighted_decided_coverage": (
            sum(float(value["weight"]) for value in decided) / total_weight
            if total_weight
            else 0.0
        ),
        "semantic_aggregation": semantic_aggregation,
        "assessments": final_assessments,
        "holistic_overall_score": holistic_score,
        "holistic_primary_owner": holistic_owner,
        "holistic_stress_policy": holistic_stress_policy,
    }
    capture_audit, judge_visible_frames = _stage3_capture_metadata(
        bundle,
        exploration=exploration,
        task_by_id=task_by_id,
        focus_targets_by_task=focus_targets_by_task,
    )
    written = write_stage3_artifacts(
        artifacts / "stage3",
        exploration=exploration,
        frame_store=exploration.frame_store,
        unknown_resolutions=resolutions,
        batch_evaluations=batch_evaluations,
        batch_policy=batch_policy,
        holistic_result=holistic_result,
        final_result=final_result,
        request_manifest=[
            *judge.manifest_records,
            *(holistic_adapter.manifest_records if holistic_adapter else ()),
        ],
        raw_records=[
            *judge.raw_records,
            *(holistic_adapter.raw_records if holistic_adapter else ()),
        ],
        stage0_summary={
            "bundle_id": bundle.bundle_id,
            "holistic_primary_owner": holistic_owner,
            "holistic_stress_policy": holistic_stress_policy,
            "vlm_runtime": vlm_runtime_snapshot(),
        },
        capture_audit=capture_audit,
        judge_visible_frames=judge_visible_frames,
    )
    return {
        "exploration": exploration.to_dict(),
        "resolutions": [_jsonable(value) for value in resolutions],
        "batch_policy": batch_policy.to_dict(),
        "vlm_runtime": vlm_runtime_snapshot(),
        "batch_evaluations": batch_evaluations,
        "request_manifest_count": len(judge.manifest_records)
        + (len(holistic_adapter.manifest_records) if holistic_adapter else 0),
        "raw_record_count": len(judge.raw_records)
        + (len(holistic_adapter.raw_records) if holistic_adapter else 0),
        "final_result": final_result,
        "holistic": _jsonable(holistic_result),
        "artifact_paths": _path_map(written),
    }


def run_pipeline(context: Context) -> dict[str, Any]:
    context = _context_with_source_snapshot(context)
    stress_value = context.spec.get("semantic_stress_policy")
    holistic_stress_policy = (
        str(stress_value).strip().casefold() if stress_value is not None else None
    )
    if holistic_stress_policy not in {None, "overview-holistic"}:
        raise ValueError(
            "semantic_stress_policy must be overview-holistic when configured"
        )
    image_to_scene = (
        getattr(context.task, "case_type", "prompt_to_scene")
        == "image_to_scene"
    )
    evidence = ue_evidence.collect(context)
    bundle = load_bundle(context, evidence)
    operation_provenance = context.spec.get("operation_provenance")
    if operation_provenance is not None:
        operation_provenance = ue_evidence.document(
            context, operation_provenance, "operation provenance"
        )
    closed = tuple(bundle.provenance.get("closed_categories") or ())
    canonical_closed = tuple(
        bundle.provenance.get("canonical_identity_closed_categories") or ()
    )
    scene = build_scene_inventory(
        evidence,
        half_extent_m=getattr(context.task, "half_extent_m", None),
        operation_provenance=operation_provenance,
        closed_categories=closed,
        canonical_identity_closed_categories=canonical_closed,
    )
    scoped_ids, _ = _scope_actor_ids(bundle, scene)
    stage1 = evaluate_stage1(
        bundle.graph,
        scene.inventory,
        scene.scene_bounds,
        eligible_actor_ids_by_entity=scoped_ids,
    )
    identity_grounding = None
    bound_actor_ids: Mapping[str, tuple[str, ...]] = {}
    if image_to_scene:
        from .identity_grounding import resolve_exact_identity_bindings

        identity_grounding = resolve_exact_identity_bindings(
            bundle,
            stage1,
            evidence,
            scene,
        )
        bound_actor_ids = identity_grounding.actor_ids_by_entity
    image_deterministic = (
        (
            *evaluate_repair_target_fields(
                bundle,
                evidence,
                bound_actor_ids_by_entity=bound_actor_ids,
            ),
            *evaluate_repair_collection_geometry(
                bundle,
                evidence,
                scene,
                bound_actor_ids_by_entity=bound_actor_ids,
            ),
        )
        if image_to_scene
        else ()
    )
    structured = (
        StructuredEvaluation()
        if image_to_scene
        else evaluate_structured_requirements(
            bundle,
            evidence,
            scene,
            stage1,
        )
    )
    deterministic = (
        *evaluate_deterministic_rules(
            bundle,
            evidence,
            scene,
            bound_actor_ids_by_entity=(
                bound_actor_ids if image_to_scene else None
            ),
            require_actor_bindings=image_to_scene,
            resolution_stage=(
                "stage4_deterministic" if image_to_scene else "stage1"
            ),
        ),
        *evaluate_legacy_contract_bindings(context, bundle),
        *image_deterministic,
        *structured.assessments,
    )
    atomic = _stage1_atomic(bundle, stage1, scene, deterministic)
    if identity_grounding is not None:
        _overlay_identity_grounding(bundle, atomic, identity_grounding)
    pre_evidence_exclusions = _semantic_pre_evidence_exclusions(bundle, atomic)
    for node_id, reason in pre_evidence_exclusions.items():
        if node_id not in atomic:
            continue
        atomic[node_id].update(
            pre_evidence_exclusion_reason=reason,
            rationale=(
                "Visual evidence acquisition skipped because the frozen "
                f"semantic score policy excludes this row: {reason}."
            ),
        )
    skip_nodes = (
        _stage2_skip_nodes(bundle, atomic)
        | set(pre_evidence_exclusions)
    )
    tasks = build_stage2_tasks(
        bundle.graph, stage1, skip_node_ids=skip_nodes
    )

    artifacts = evidence.root / "requirement_graph"
    artifacts.mkdir(parents=True, exist_ok=True)
    stage2: Stage2Evaluation | None = None
    stage2_paths: dict[str, str] = {}
    stage3: dict[str, Any] | None = None
    grounded_deterministic: tuple[DeterministicAssessment, ...] = ()
    camera_visibility_audit: tuple[dict[str, Any], ...] = ()
    visual_error: str | None = None
    if tasks:
        try:
            # Optional visual dependencies are required only after Stage 1 has
            # proved that at least one claim actually needs pixels.
            from .frame_provider import (
                frame_provider_from_context,
                reusable_overview_frames_from_context,
            )
            from .stage2 import evaluate_stage2_detailed
            provider = frame_provider_from_context(context, scene, artifacts / "frames")
            retrieval_backend, locator_top_k = _retrieval_options(context)
            try:
                stage2 = evaluate_stage2_detailed(
                    bundle.graph,
                    stage1,
                    scene.inventory,
                    scene.scene_bounds,
                    provider,
                    retrieval_backend=retrieval_backend,
                    locator_top_k=locator_top_k,
                    budget=_budget(
                        context.spec.get("stage2_budget"),
                        Stage2Budget,
                        "stage2_budget",
                    ) if context.spec.get("stage2_budget") is not None
                    else _default_stage2_budget(bundle),
                    skip_node_ids=skip_nodes,
                    eligible_actor_ids_by_entity=scoped_ids,
                    initial_overview_frames=reusable_overview_frames_from_context(
                        context
                    ),
                )
                stage2_paths = _publish_stage2(
                    artifacts / "stage2", stage2, bundle
                )
                _overlay_stage2(atomic, stage2)
                client = None
                if image_to_scene:
                    from .identity_grounding import (
                        requires_visual_identity_grounding,
                        resolve_identity_bindings,
                    )

                    identity_judge = None
                    if requires_visual_identity_grounding(
                        bundle.graph,
                        stage1,
                        stage2,
                        identity_grounding,
                    ):
                        from .stage3_judge import LLMStage3IdentityJudge

                        client = tool_client_from_env()
                        identity_judge = LLMStage3IdentityJudge(
                            client,
                            max_tokens=min(
                                _production_max_tokens(client, 384), 384
                            ),
                            max_frames_per_request=2,
                        )
                    identity_grounding = resolve_identity_bindings(
                        bundle.graph,
                        stage1,
                        stage2,
                        scene.scene_bounds,
                        provider,
                        identity_judge,
                        identity_grounding,
                    )
                    _overlay_identity_grounding(
                        bundle,
                        atomic,
                        identity_grounding,
                    )
                    bound_actor_ids = identity_grounding.actor_ids_by_entity
                    if bound_actor_ids:
                        grounded_deterministic = (
                            *evaluate_deterministic_rules(
                                bundle,
                                evidence,
                                scene,
                                bound_actor_ids_by_entity=bound_actor_ids,
                                require_actor_bindings=True,
                                resolution_stage="stage4_deterministic",
                            ),
                            *evaluate_repair_collection_geometry(
                                bundle,
                                evidence,
                                scene,
                                bound_actor_ids_by_entity=bound_actor_ids,
                            ),
                            *evaluate_repair_target_fields(
                                bundle,
                                evidence,
                                bound_actor_ids_by_entity=bound_actor_ids,
                            ),
                        )
                        _overlay_grounded_deterministic(
                            bundle,
                            atomic,
                            grounded_deterministic,
                            set(bound_actor_ids),
                        )
                stage3 = _run_stage3(
                    bundle,
                    atomic,
                    stage2,
                    provider,
                    client,
                    artifacts,
                    _budget(
                        context.spec.get("stage3_budget"),
                        Stage3Budget,
                        "stage3_budget",
                    )
                    if context.spec.get("stage3_budget") is not None
                    else None,
                    _budget(
                        context.spec.get("stage3_batch_policy"),
                        Stage3BatchPolicy,
                        "stage3_batch_policy",
                    ),
                    identity_grounding=(
                        identity_grounding if image_to_scene else None
                    ),
                    holistic_stress_policy=holistic_stress_policy,
                )
                if stage3 is not None:
                    # Preserve recorded failures as public verifier errors so
                    # completed earlier measurements cannot hide a failed judge.
                    visual_error = stage3["final_result"].get("evaluation_error")
            finally:
                camera_visibility_audit = tuple(
                    dict(value)
                    for value in getattr(
                        provider,
                        "camera_visibility_audits",
                        (),
                    )
                )
                provider.close()
        except Exception as error:  # noqa: BLE001 - report missing evaluation
            visual_error = f"{type(error).__name__}: {error}"
            # Keep the public terminal reason compact, but retain the full
            # diagnostic stack.  Without it every controller defect collapses
            # to an unactionable value such as ``KeyError: ''`` after this
            # fail-closed boundary converts all affected requirements to
            # evaluation errors.
            import traceback

            try:
                (artifacts / "visual_error_traceback.txt").write_text(
                    traceback.format_exc(),
                    encoding="utf-8",
                )
            except OSError:
                pass
            for task in tasks:
                atomic[task.node_id].update(
                    verdict=ClaimVerdict.UNKNOWN.value,
                    resolved_by=None,
                    unknown_reason=UnknownReason.EVALUATION_ERROR.value,
                    rationale=visual_error,
                )

    requirement_rows = []
    for binding in bundle.requirements:
        # Non-SUPPORTED bindings stay out of the scored rows — but not out of
        # the record: they are listed in `waived_requirements` below, because
        # a requirement that vanishes without trace cannot be told from one
        # that was never in the bundle.
        if binding.status is not RequirementStatus.SUPPORTED:
            continue
        if binding.node_id not in atomic:
            continue
        value = atomic[binding.node_id]
        requirement_rows.append(
            {
                **value,
                "evaluation_status": _requirement_evaluation_status(value),
                "text": binding.source_text,
                "source_span": list(binding.source_span),
                "population_scope": binding.population_scope.value,
                "entity_scopes": {
                    key: value.value for key, value in binding.entity_scopes.items()
                },
                "weight": bundle.graph.effective_weights()[binding.node_id],
                "evaluation_binding": bundle.evaluation_for(binding.node_id).to_dict(),
            }
        )
    delegated = []
    for value in bundle.requirements:
        evaluation = bundle.evaluation_for(value.node_id)
        if evaluation.evaluator_id.startswith("semantic_requirements."):
            continue
        delegated.append(
            {
                "requirement_id": value.requirement_id,
                "node_id": value.node_id,
                "text": value.source_text,
                "evaluator_id": evaluation.evaluator_id,
                "evaluator_version": evaluation.evaluator_version,
                "evaluation_binding_id": evaluation.binding_id,
            }
        )
    waived = [
        {
            "requirement_id": value.requirement_id,
            "node_id": value.node_id,
            "text": value.source_text,
            "status": value.status.value,
        }
        for value in bundle.requirements
        if value.status is not RequirementStatus.SUPPORTED
    ]
    artifact_payloads = {
        "bundle": bundle.to_dict(),
        "inventory": scene.inventory.to_dict(),
        "scopes": {
            key.value: _jsonable(value) for key, value in scene.scopes.items()
        },
        "stage1": stage1.to_dict(),
        "structured": structured.to_dict(),
        "deterministic": [value.to_dict() for value in deterministic],
        "grounded_deterministic": [
            value.to_dict() for value in grounded_deterministic
        ],
        "pre_evidence_exclusions": pre_evidence_exclusions,
        "stage2": (
            {
                "result": stage2.result.to_dict(),
                "capture": stage2.capture_artifact,
                "routing": stage2.routing_plan.to_dict(),
                "evidence_selection": dict(stage2.evidence_selection),
                "artifact_paths": stage2_paths,
            }
            if stage2 is not None
            else None
        ),
        "stage3": stage3,
        "camera_visibility": camera_visibility_audit,
        "identity_grounding": (
            identity_grounding.to_dict()
            if identity_grounding is not None
            else None
        ),
        "visual_error": visual_error,
        "requirements": requirement_rows,
        "delegated_requirements": delegated,
        "non_supported_requirements": waived,
        "waived_requirements": waived,
    }
    paths: dict[str, str] = {}
    for name, payload in artifact_payloads.items():
        path = artifacts / f"{name}.json"
        ue_evidence.write_json(path, _jsonable(payload))
        paths[name] = str(path)
    paths.update({f"stage2/{name}": path for name, path in stage2_paths.items()})
    if stage3 is not None:
        paths.update(
            {
                f"stage3/{name}": path
                for name, path in stage3.get("artifact_paths", {}).items()
            }
        )
    return {
        "bundle": bundle,
        "requirements": requirement_rows,
        "delegated_requirements": delegated,
        "non_supported_requirements": waived,
        "waived_requirements": waived,
        "scene_evidence": evidence,
        "artifacts": paths,
        "visual_error": visual_error,
        "identity_grounding": identity_grounding,
        "structured": structured,
        "stage3": stage3,
        "pre_evidence_exclusions": pre_evidence_exclusions,
    }


__all__ = ["load_bundle", "run_pipeline"]
