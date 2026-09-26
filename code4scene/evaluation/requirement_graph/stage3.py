"""RequirementGraph Stage 3 orchestration and final graph reconciliation.

Stage 3 borrows the active frame provider but never owns its lifecycle.  It
keeps exploration, per-claim UNKNOWN resolution, and holistic judging on
separate RGB-only boundaries, then overlays their products on the immutable
Stage 1 and Stage 2 results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    ClaimVerdict,
    ComparisonOperator,
    EntityNode,
    PredicateNode,
    PredicateType,
    RequirementGraph,
    SceneBounds,
)
from .stage1 import Stage1Result
from .stage2 import Stage2Evaluation
from .stage2_contracts import (
    GLOBAL_CONTEXT_TASK_KINDS,
    Stage2Assessment,
    Stage2Task,
    Stage2TaskKind,
    Stage2Verdict,
)
from .stage2_routing import visual_claim_payload
from .stage3_contracts import (
    EvidenceRef,
    FinalGraphResult,
    FinalLeafAssessment,
    HolisticDimensionScore,
    HolisticResult,
    Stage3BatchPolicy,
    Stage3ClaimResolution,
    Stage3Verdict,
)


class Stage3EvaluationError(RuntimeError):
    """An evaluator/runtime failure which must not become a scene mismatch."""


@dataclass(frozen=True, slots=True)
class Stage3VlmComponents:
    """Fresh-context adapters sharing one underlying RGB-capable client."""

    selector: Any
    unknown_judge: Any
    holistic_judge: Any


@dataclass(frozen=True, slots=True)
class Stage3Evaluation:
    """Complete Stage 3 controller product used by the runner and writer."""

    exploration: Any
    unknown_resolutions: tuple[Stage3ClaimResolution, ...]
    holistic_result: HolisticResult
    final_result: FinalGraphResult
    batch_evaluations: tuple[Mapping[str, Any], ...] = ()
    request_manifest: tuple[Mapping[str, Any], ...] = ()
    raw_records: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "unknown_resolutions", tuple(self.unknown_resolutions))
        object.__setattr__(
            self,
            "batch_evaluations",
            tuple(dict(value) for value in self.batch_evaluations),
        )
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
    def frame_store(self) -> Any:
        return self.exploration.frame_store


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value)).strip()


def _verdict(value: Any) -> Stage3Verdict:
    normalized = _enum_text(value).upper()
    if normalized == "PARTIAL_MATCH":
        normalized = "UNKNOWN"
    return Stage3Verdict(normalized)


def _adapter_is_healthy(value: Any) -> bool:
    transport = _enum_text(getattr(value, "transport_status", "success")).casefold()
    parsed = _enum_text(getattr(value, "parse_status", "success")).casefold()
    return transport in {"success", "ok"} and parsed in {
        "success",
        "valid",
        "ok",
    }


def _adapter_error(value: Any) -> str:
    error = getattr(value, "error", None) or getattr(value, "evaluation_error", None)
    if error:
        return str(error).strip()
    return (
        f"transport_status={_enum_text(getattr(value, 'transport_status', 'unknown'))}, "
        f"parse_status={_enum_text(getattr(value, 'parse_status', 'unknown'))}"
    )


def _transport_status(value: Any) -> str:
    raw = _enum_text(getattr(value, "transport_status", "success")).casefold()
    return "success" if raw in {"success", "ok"} else "error"


def _parse_status(value: Any) -> str:
    transport = _transport_status(value)
    if transport == "error":
        return "not_attempted"
    raw = _enum_text(getattr(value, "parse_status", "success")).casefold()
    return "success" if raw in {"success", "valid", "ok"} else "error"


def _evidence_refs(stage: str, frame_ids: Sequence[str]) -> tuple[EvidenceRef, ...]:
    return tuple(
        EvidenceRef(stage=stage, evidence_id=str(frame_id), kind="rgb")
        for frame_id in dict.fromkeys(str(value) for value in frame_ids)
    )


def _visible_count_satisfies(
    graph: RequirementGraph,
    task: Stage2Task,
    decision: Any,
) -> bool:
    node = graph.node(task.node_id)
    grounding = getattr(decision, "grounding", None)
    visible_count = getattr(grounding, "visible_instance_count", None)
    if not isinstance(node, PredicateNode) or node.constraint is None:
        return False
    if isinstance(visible_count, bool) or not isinstance(visible_count, int):
        return False
    constraint = node.constraint
    if constraint.operator is ComparisonOperator.EQ:
        return visible_count == constraint.value
    if constraint.operator is ComparisonOperator.GTE:
        return visible_count >= constraint.value
    if constraint.operator is ComparisonOperator.LTE:
        return visible_count <= constraint.value
    return (
        constraint.operator is ComparisonOperator.BETWEEN
        and constraint.upper_value is not None
        and constraint.value <= visible_count <= constraint.upper_value
    )


def _visual_resolution_gate(
    graph: RequirementGraph,
    task: Stage2Task,
    decision: Any,
    verdict: Stage3Verdict,
) -> str | None:
    """Validate a model resolution without using Stage 2 route/shot metadata."""

    evidence = tuple(getattr(decision, "evidence_frame_ids", ()) or ())
    if verdict is Stage3Verdict.UNKNOWN:
        return "stage3_semantic_unknown"
    if not evidence:
        return "stage3_resolved_verdict_missing_rgb_citation"

    grounding = getattr(decision, "grounding", None)
    if verdict is Stage3Verdict.MATCH:
        if task.kind in {
            Stage2TaskKind.OBJECT_EXISTENCE,
            Stage2TaskKind.ATTRIBUTE,
            Stage2TaskKind.MATERIAL,
        } and not bool(getattr(grounding, "subject_confirmed", False)):
            return "stage3_subject_not_visually_confirmed"
        if task.kind is Stage2TaskKind.SPATIAL_RELATION and not (
            bool(getattr(grounding, "participants_confirmed", False))
            and bool(getattr(grounding, "relation_scope_covered", False))
        ):
            return "stage3_relation_participants_or_scope_not_confirmed"
        if task.kind is Stage2TaskKind.COUNT and not (
            bool(getattr(grounding, "collection_complete", False))
            and bool(getattr(grounding, "instances_countable", False))
            and _visible_count_satisfies(graph, task, decision)
        ):
            return "stage3_count_match_not_complete_or_inconsistent"
        if task.kind in GLOBAL_CONTEXT_TASK_KINDS and len(set(evidence)) < 2:
            return "stage3_scene_match_requires_two_views"
        return None

    # A finite holistic search cannot honestly promote affirmative object
    # absence to a visual contradiction.  The externally requested closed-world
    # policy is applied below with explicit provenance instead.
    if (
        task.kind is Stage2TaskKind.OBJECT_EXISTENCE
        and _enum_text(task.polarity).casefold() == "affirmative"
    ):
        return "stage3_affirmative_existence_absence_is_not_visual_proof"
    if task.kind is Stage2TaskKind.OBJECT_EXISTENCE:
        if not bool(getattr(grounding, "subject_confirmed", False)):
            return "stage3_forbidden_subject_not_visually_confirmed"
    elif task.kind in {Stage2TaskKind.ATTRIBUTE, Stage2TaskKind.MATERIAL}:
        if not bool(getattr(grounding, "subject_confirmed", False)):
            return "stage3_subject_not_visually_confirmed"
    elif task.kind is Stage2TaskKind.SPATIAL_RELATION:
        if not (
            bool(getattr(grounding, "participants_confirmed", False))
            and bool(getattr(grounding, "relation_scope_covered", False))
        ):
            return "stage3_relation_participants_or_scope_not_confirmed"
    elif task.kind is Stage2TaskKind.COUNT:
        if not (
            bool(getattr(grounding, "collection_complete", False))
            and bool(getattr(grounding, "instances_countable", False))
            and not _visible_count_satisfies(graph, task, decision)
        ):
            return "stage3_count_mismatch_not_complete_or_inconsistent"
    elif task.kind in GLOBAL_CONTEXT_TASK_KINDS and len(set(evidence)) < 2:
        return "stage3_scene_mismatch_requires_two_views"
    return None


def finalize_stage3_unknown_decision(
    graph: RequirementGraph,
    task: Stage2Task,
    decision: Any,
) -> Stage3ClaimResolution:
    """Resolve one Stage 2 UNKNOWN or apply the explicit policy fallback."""

    if not _adapter_is_healthy(decision):
        error = _adapter_error(decision)
        return Stage3ClaimResolution(
            task_id=task.task_id,
            node_id=task.node_id,
            judge_verdict=None,
            final_verdict=None,
            confidence=0.0,
            rationale="Stage 3 UNKNOWN resolver failed.",
            evaluation_error=error,
        )

    raw_verdict = _verdict(getattr(decision, "verdict", "UNKNOWN"))
    reason = _visual_resolution_gate(graph, task, decision, raw_verdict)
    evidence_ids = tuple(getattr(decision, "evidence_frame_ids", ()) or ())
    confidence = float(getattr(decision, "confidence", 0.0) or 0.0)
    rationale = str(getattr(decision, "rationale", "")).strip()
    if reason is None:
        return Stage3ClaimResolution(
            task_id=task.task_id,
            node_id=task.node_id,
            judge_verdict=raw_verdict,
            final_verdict=raw_verdict,
            confidence=confidence,
            evidence_refs=_evidence_refs("stage3", evidence_ids),
            rationale=rationale,
        )
    return Stage3ClaimResolution(
        task_id=task.task_id,
        node_id=task.node_id,
        # ``judge_verdict`` is the controller-safe semantic verdict.  The raw
        # adapter response remains preserved in stage3_vlm_raw.jsonl.
        judge_verdict=Stage3Verdict.UNKNOWN,
        final_verdict=Stage3Verdict.UNKNOWN,
        confidence=0.0,
        evidence_refs=(),
        rationale=(
            rationale
            or "The exploration portfolio did not resolve this claim visually."
        ),
        forced_mismatch=False,
        forced_reason=None,
    )


def finalize_stage3_binary_decision(
    task: Stage2Task,
    decision: Any,
) -> Stage3ClaimResolution:
    """Accept one final best-evidence binary VLM decision.

    This pass deliberately does not re-apply the conservative completeness
    gates used while deciding whether more capture is needed. The model must
    still return a healthy structured MATCH/MISMATCH and cite supplied RGB.
    Invocation and parsing failures carry an evaluation error and no verdict.
    """

    if not _adapter_is_healthy(decision):
        detail = _adapter_error(decision)
        return Stage3ClaimResolution(
            task_id=task.task_id,
            node_id=task.node_id,
            judge_verdict=None,
            final_verdict=None,
            confidence=0.0,
            rationale=f"Final binary VLM arbitration failed: {detail}",
            evaluation_error=detail,
        )
    verdict = _verdict(getattr(decision, "verdict", "UNKNOWN"))
    evidence_ids = tuple(getattr(decision, "evidence_frame_ids", ()) or ())
    if verdict not in {Stage3Verdict.MATCH, Stage3Verdict.MISMATCH} or not evidence_ids:
        return Stage3ClaimResolution(
            task_id=task.task_id,
            node_id=task.node_id,
            judge_verdict=Stage3Verdict.UNKNOWN,
            final_verdict=Stage3Verdict.UNKNOWN,
            confidence=0.0,
            rationale=(
                "Final binary VLM arbitration produced no valid RGB-cited "
                "MATCH/MISMATCH decision."
            ),
        )
    return Stage3ClaimResolution(
        task_id=task.task_id,
        node_id=task.node_id,
        judge_verdict=verdict,
        final_verdict=verdict,
        confidence=float(getattr(decision, "confidence", 0.0) or 0.0),
        evidence_refs=_evidence_refs("stage3", evidence_ids),
        rationale=str(getattr(decision, "rationale", "")).strip(),
    )


def finalize_stage3_binary_resolution(
    resolution: Stage3ClaimResolution,
) -> Stage3ClaimResolution:
    """Deprecated compatibility shim; blind UNKNOWN-to-MISMATCH is disabled."""

    if not isinstance(resolution, Stage3ClaimResolution):
        raise TypeError("resolution must be a Stage3ClaimResolution")
    return resolution


def stage3_actor_reframe_recommended(
    task: Stage2Task,
    decision: Any,
    resolution: Stage3ClaimResolution,
) -> bool:
    """Whether a healthy structured VLM result asks for another actor angle."""

    if resolution.final_verdict is not Stage3Verdict.UNKNOWN:
        return False
    if resolution.evaluation_error is not None or not _adapter_is_healthy(decision):
        return False
    if task.kind not in {
        Stage2TaskKind.OBJECT_EXISTENCE,
        Stage2TaskKind.ATTRIBUTE,
        Stage2TaskKind.MATERIAL,
        Stage2TaskKind.COUNT,
    }:
        return False
    grounding = getattr(decision, "grounding", None)
    if task.kind is Stage2TaskKind.COUNT:
        return grounding is not None and not (
            bool(getattr(grounding, "collection_complete", False))
            and bool(getattr(grounding, "instances_countable", False))
        )
    return grounding is not None and not bool(
        getattr(grounding, "subject_confirmed", False)
    )



def _batch_request_id(judge: Any, previous_count: int) -> str | None:
    # Production judges expose the exact request started by this worker thread.
    # Falling back to the ledger count keeps simple test/dummy judges compatible,
    # but must not be used for concurrent production calls: another worker may
    # append a later manifest record before this worker returns.
    current = getattr(judge, "current_thread_request_id", None)
    if current is not None:
        request_id = str(current).strip()
        if request_id:
            return request_id
    records = _records(judge, "manifest_records")
    if len(records) <= previous_count:
        return None
    request_id = str(records[-1].get("request_id") or "").strip()
    return request_id or None


def _batch_request_ids(judge: Any, previous_count: int) -> tuple[str, ...]:
    current = getattr(judge, "current_thread_request_ids", ())
    request_ids = tuple(
        value
        for value in (str(item).strip() for item in current)
        if value
    )
    if request_ids:
        return request_ids
    request_id = _batch_request_id(judge, previous_count)
    return (request_id,) if request_id is not None else ()


def _batch_audit_record(
    *,
    task: Stage2Task,
    batch_index: int,
    frame_ids: tuple[str, ...],
    request_id: str | None,
    attempt_request_ids: tuple[str, ...],
    decision: Any,
    resolution: Stage3ClaimResolution,
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "frame_ids": list(frame_ids),
        "request_id": request_id,
        "attempt_request_ids": list(attempt_request_ids),
        "transport_status": _enum_text(
            getattr(decision, "transport_status", "unknown")
        ),
        "parse_status": _enum_text(getattr(decision, "parse_status", "unknown")),
        "judge_verdict": (
            _enum_text(getattr(decision, "verdict", "UNKNOWN")).upper()
            if resolution.evaluation_error is None
            else None
        ),
        "accepted_verdict": (
            resolution.final_verdict.value
            if resolution.final_verdict is not None
            else None
        ),
        "confidence": resolution.confidence,
        "evidence_frame_ids": [
            value.evidence_id for value in resolution.evidence_refs
        ],
        "reframe_recommended": stage3_actor_reframe_recommended(
            task,
            decision,
            resolution,
        ),
        "rationale": resolution.rationale,
        "evaluation_error": resolution.evaluation_error,
    }


def _aggregate_stage3_batches(
    task: Stage2Task,
    resolutions: Sequence[Stage3ClaimResolution],
    *,
    omitted_frame_ids: Sequence[str],
) -> Stage3ClaimResolution:
    """Combine type-gated batch resolutions without voting or count addition."""

    errors = [value.evaluation_error for value in resolutions if value.evaluation_error]
    if errors:
        return Stage3ClaimResolution(
            task_id=task.task_id,
            node_id=task.node_id,
            judge_verdict=None,
            final_verdict=None,
            confidence=0.0,
            rationale="Stage 3 UNKNOWN batch evaluation failed.",
            evaluation_error="; ".join(dict.fromkeys(errors)),
        )

    decided = [
        value
        for value in resolutions
        if value.final_verdict in {Stage3Verdict.MATCH, Stage3Verdict.MISMATCH}
    ]
    decided_verdicts = {value.final_verdict for value in decided}
    if len(decided_verdicts) > 1:
        return Stage3ClaimResolution(
            task_id=task.task_id,
            node_id=task.node_id,
            judge_verdict=Stage3Verdict.UNKNOWN,
            final_verdict=Stage3Verdict.UNKNOWN,
            confidence=0.0,
            rationale=(
                "Stage 3 batches produced conflicting, individually grounded "
                "verdicts; the claim remains unresolved."
            ),
        )
    if decided:
        # ``finalize_stage3_unknown_decision`` already applied the claim-kind
        # gates: affirmative existence cannot be disproved by finite search,
        # count decisions require a complete countable collection, and
        # relation/attribute decisions require their specific grounding.
        return max(decided, key=lambda value: value.confidence)

    rationales = tuple(
        dict.fromkeys(value.rationale for value in resolutions if value.rationale)
    )
    detail = " ".join(rationales[:2]) or (
        "The exploration evidence did not resolve this claim visually."
    )
    if omitted_frame_ids:
        detail += (
            " The configured Stage 3 batch budget omitted "
            f"{len(tuple(omitted_frame_ids))} lower-priority frame(s)."
        )
    return Stage3ClaimResolution(
        task_id=task.task_id,
        node_id=task.node_id,
        judge_verdict=Stage3Verdict.UNKNOWN,
        final_verdict=Stage3Verdict.UNKNOWN,
        confidence=0.0,
        rationale=detail,
    )

def aggregate_stage3_batch_passes(
    task: Stage2Task,
    passes: Sequence[tuple[Stage3ClaimResolution, Mapping[str, Any]]],
    policy: Stage3BatchPolicy,
    *,
    available_frame_ids: Sequence[str],
) -> tuple[Stage3ClaimResolution, dict[str, Any]]:
    """Join separately scheduled first/reframe passes under one batch policy."""

    if not isinstance(task, Stage2Task):
        raise TypeError("task must be a Stage2Task")
    if not isinstance(policy, Stage3BatchPolicy):
        raise TypeError("policy must be a Stage3BatchPolicy")
    normalized_passes = tuple(passes)
    if not normalized_passes:
        raise ValueError("passes must contain at least one Stage 3 batch")
    if len(normalized_passes) > policy.max_batches_per_claim:
        raise ValueError("passes exceed max_batches_per_claim")
    batches: list[dict[str, Any]] = []
    resolutions: list[Stage3ClaimResolution] = []
    for resolution, audit in normalized_passes:
        if not isinstance(resolution, Stage3ClaimResolution):
            raise TypeError("passes must contain Stage3ClaimResolution values")
        resolutions.append(resolution)
        for value in tuple(audit.get("batches") or ()):
            record = dict(value)
            record["batch_index"] = len(batches) + 1
            batches.append(record)
    selected_ids = tuple(
        dict.fromkeys(
            str(frame_id)
            for batch in batches
            for frame_id in tuple(batch.get("frame_ids") or ())
        )
    )
    reported_available_ids = tuple(
        dict.fromkeys(
            (
                *(str(value) for value in available_frame_ids),
                *selected_ids,
            )
        )
    )
    selected_id_set = set(selected_ids)
    # The artifact contract records the actual request priority: selected
    # frames first, followed by evidence omitted by the bounded VLM policy.
    available_ids = (
        *selected_ids,
        *(value for value in reported_available_ids if value not in selected_id_set),
    )
    omitted_ids = tuple(
        value for value in available_ids if value not in selected_id_set
    )
    aggregate = _aggregate_stage3_batches(
        task,
        resolutions,
        omitted_frame_ids=omitted_ids,
    )
    return aggregate, {
        "schema_version": "1.0",
        "task_id": task.task_id,
        "node_id": task.node_id,
        "claim_kind": task.kind.value,
        "policy_id": policy.policy_id,
        "available_frame_ids": list(available_ids),
        "selected_frame_ids": list(selected_ids),
        "omitted_frame_ids": list(omitted_ids),
        "batches": batches,
        "aggregation": aggregate.to_dict(),
    }



def evaluate_stage3_unknown_batches(
    graph: RequirementGraph,
    task: Stage2Task,
    frames: Sequence[Any],
    judge: Any,
    policy: Stage3BatchPolicy,
) -> tuple[Stage3ClaimResolution, dict[str, Any]]:
    """Resolve one UNKNOWN claim across deterministic, bounded RGB batches."""

    if not isinstance(graph, RequirementGraph):
        raise TypeError("graph must be a RequirementGraph")
    if not isinstance(task, Stage2Task):
        raise TypeError("task must be a Stage2Task")
    if not isinstance(policy, Stage3BatchPolicy):
        raise TypeError("policy must be a Stage3BatchPolicy")
    ordered_frames = tuple(frames)
    if not ordered_frames:
        resolution = Stage3ClaimResolution(
            task_id=task.task_id,
            node_id=task.node_id,
            judge_verdict=Stage3Verdict.UNKNOWN,
            final_verdict=Stage3Verdict.UNKNOWN,
            confidence=0.0,
            rationale="Stage 3 had no RGB evidence for this claim.",
        )
        return resolution, {
            "schema_version": "1.0",
            "task_id": task.task_id,
            "node_id": task.node_id,
            "claim_kind": task.kind.value,
            "policy_id": policy.policy_id,
            "available_frame_ids": [],
            "selected_frame_ids": [],
            "omitted_frame_ids": [],
            "batches": [],
            "aggregation": resolution.to_dict(),
        }

    available_ids = tuple(str(value.frame_id) for value in ordered_frames)
    if len(available_ids) != len(set(available_ids)):
        raise ValueError("Stage 3 claim frames must have unique frame ids")
    within_safety_cap = ordered_frames[: policy.maximum_frames_per_claim]
    batches = tuple(
        within_safety_cap[index : index + policy.max_frames_per_request]
        for index in range(0, len(within_safety_cap), policy.max_frames_per_request)
    )
    resolutions: list[Stage3ClaimResolution] = []
    audit_batches: list[dict[str, Any]] = []
    processed_ids: list[str] = []
    stable_verdict: Stage3Verdict | None = None
    stable_count = 0
    saw_judge_error = False
    stop_reason = "evidence_exhausted"
    payload = visual_claim_payload(graph, task.node_id)
    for batch_index, batch in enumerate(batches, start=1):
        previous_manifest_count = len(_records(judge, "manifest_records"))
        decision = judge.judge(payload, batch)
        resolution = finalize_stage3_unknown_decision(graph, task, decision)
        resolutions.append(resolution)
        processed_ids.extend(str(value.frame_id) for value in batch)
        audit_batches.append(
            _batch_audit_record(
                task=task,
                batch_index=batch_index,
                frame_ids=tuple(str(value.frame_id) for value in batch),
                request_id=_batch_request_id(judge, previous_manifest_count),
                attempt_request_ids=_batch_request_ids(
                    judge,
                    previous_manifest_count,
                ),
                decision=decision,
                resolution=resolution,
            )
        )
        if resolution.evaluation_error is not None:
            saw_judge_error = True
            stable_verdict = None
            stable_count = 0
            continue
        verdict = resolution.final_verdict
        if verdict in {Stage3Verdict.MATCH, Stage3Verdict.MISMATCH}:
            if verdict is stable_verdict:
                stable_count += 1
            else:
                stable_verdict = verdict
                stable_count = 1
            if stable_count >= 2:
                stop_reason = "stable_verdict"
                break
        else:
            stable_verdict = None
            stable_count = 0

    processed_id_set = set(processed_ids)
    omitted_ids = tuple(
        frame_id for frame_id in available_ids if frame_id not in processed_id_set
    )
    if stop_reason == "evidence_exhausted" and saw_judge_error:
        stop_reason = "judge_error"
    if (
        stop_reason == "evidence_exhausted"
        and len(ordered_frames) > policy.maximum_frames_per_claim
    ):
        stop_reason = "safety_cap"
    aggregate = _aggregate_stage3_batches(
        task,
        resolutions,
        omitted_frame_ids=omitted_ids,
    )
    return aggregate, {
        "schema_version": "1.0",
        "task_id": task.task_id,
        "node_id": task.node_id,
        "claim_kind": task.kind.value,
        "policy_id": policy.policy_id,
        "available_frame_ids": list(available_ids),
        "selected_frame_ids": processed_ids,
        "omitted_frame_ids": list(omitted_ids),
        "batches": audit_batches,
        "stop_reason": stop_reason,
        "stable_verdict": (
            stable_verdict.value if stable_count >= 2 and stable_verdict else None
        ),
        "aggregation": aggregate.to_dict(),
    }


def evaluate_stage3_binary_arbitration(
    graph: RequirementGraph,
    task: Stage2Task,
    frames: Sequence[Any],
    judge: Any,
) -> tuple[Stage3ClaimResolution, dict[str, Any]]:
    """Make one final binary VLM decision from the best available RGB."""

    ordered_frames = tuple(frames)
    maximum = int(getattr(judge, "max_binary_frames_per_request", 10))
    selected = ordered_frames[:maximum]
    base_audit: dict[str, Any] = {
        "attempted": False,
        "available_frame_ids": [
            str(value.frame_id) for value in ordered_frames
        ],
        "selected_frame_ids": [str(value.frame_id) for value in selected],
        "omitted_frame_ids": [
            str(value.frame_id) for value in ordered_frames[maximum:]
        ],
        "request_id": None,
    }
    if not selected:
        resolution = Stage3ClaimResolution(
            task_id=task.task_id,
            node_id=task.node_id,
            judge_verdict=Stage3Verdict.UNKNOWN,
            final_verdict=Stage3Verdict.UNKNOWN,
            confidence=0.0,
            rationale="No valid RGB was available for final binary arbitration.",
        )
        return resolution, {**base_audit, "resolution": resolution.to_dict()}
    binary_judge = getattr(judge, "judge_binary", None)
    if not callable(binary_judge):
        resolution = Stage3ClaimResolution(
            task_id=task.task_id,
            node_id=task.node_id,
            judge_verdict=Stage3Verdict.UNKNOWN,
            final_verdict=Stage3Verdict.UNKNOWN,
            confidence=0.0,
            rationale="The configured Stage 3 judge has no binary arbitration API.",
        )
        return resolution, {**base_audit, "resolution": resolution.to_dict()}
    previous_manifest_count = len(_records(judge, "manifest_records"))
    decision = binary_judge(visual_claim_payload(graph, task.node_id), selected)
    resolution = finalize_stage3_binary_decision(task, decision)
    attempt_request_ids = _batch_request_ids(judge, previous_manifest_count)
    return resolution, {
        **base_audit,
        "attempted": True,
        "request_id": _batch_request_id(judge, previous_manifest_count),
        "attempt_request_ids": list(attempt_request_ids),
        "transport_status": _enum_text(
            getattr(decision, "transport_status", "unknown")
        ),
        "parse_status": _enum_text(getattr(decision, "parse_status", "unknown")),
        "raw_verdict": _enum_text(getattr(decision, "verdict", "UNKNOWN")).upper(),
        "resolution": resolution.to_dict(),
    }


_HOLISTIC_WEIGHTS = {
    "global_prompt_alignment": 0.40,
    "composition_and_layout": 0.25,
    "style_atmosphere_coherence": 0.20,
    "completeness_and_polish": 0.15,
}


def _coerce_holistic_result(value: Any) -> HolisticResult:
    if isinstance(value, HolisticResult):
        return value
    if not _adapter_is_healthy(value):
        return HolisticResult(
            status="evaluation_error",
            transport_status=_transport_status(value),
            parse_status=_parse_status(value),
            evaluation_error=_adapter_error(value),
        )

    raw_dimensions = getattr(value, "dimensions", ())
    if isinstance(raw_dimensions, Mapping):
        items = tuple(raw_dimensions.values())
    else:
        items = tuple(raw_dimensions or ())
    dimensions: list[HolisticDimensionScore] = []
    for raw in items:
        name = _enum_text(getattr(raw, "name", "")).casefold()
        if name not in _HOLISTIC_WEIGHTS:
            continue
        frame_ids = tuple(getattr(raw, "evidence_frame_ids", ()) or ())
        dimensions.append(
            HolisticDimensionScore(
                name=name,
                score=float(getattr(raw, "score", 0.0)),
                weight=_HOLISTIC_WEIGHTS[name],
                rationale=str(getattr(raw, "rationale", "")).strip(),
                evidence_refs=_evidence_refs("stage3", frame_ids),
            )
        )
    by_name = {_enum_text(value.name).casefold(): value for value in dimensions}
    if set(by_name) != set(_HOLISTIC_WEIGHTS):
        return HolisticResult(
            dimensions=tuple(dimensions),
            status="evaluation_error",
            transport_status="success",
            parse_status="error",
            evaluation_error="Holistic judge omitted one or more required dimensions.",
        )
    overall = sum(
        by_name[name].score * weight for name, weight in _HOLISTIC_WEIGHTS.items()
    )
    return HolisticResult(
        dimensions=tuple(by_name[name] for name in _HOLISTIC_WEIGHTS),
        overall_score=overall,
        summary=str(getattr(value, "summary", "")).strip(),
        status="complete",
        transport_status=getattr(value, "transport_status", "success"),
        parse_status=getattr(value, "parse_status", "success"),
    )


def _stage1_for_scored_node(
    graph: RequirementGraph,
    stage1_result: Stage1Result,
    node_id: str,
) -> Any | None:
    node = graph.node(node_id)
    if isinstance(node, EntityNode):
        assessment = stage1_result.for_entity(node_id)
    elif isinstance(node, PredicateNode) and node.predicate_type is PredicateType.EXISTENCE:
        assessment = stage1_result.for_predicate(node_id)
    else:
        return None
    if assessment is None or not assessment.stage1_decision_applicable:
        return None
    if assessment.verdict not in {ClaimVerdict.MATCH, ClaimVerdict.MISMATCH}:
        return None
    return assessment


def reconcile_graph_results(
    graph: RequirementGraph,
    stage1_result: Stage1Result,
    stage2_evaluation: Stage2Evaluation,
    resolutions: Sequence[Stage3ClaimResolution],
    holistic_result: HolisticResult,
) -> FinalGraphResult:
    """Overlay stage products without mutating any earlier assessment."""

    weights = graph.effective_weights()
    stage2_by_node: dict[str, Stage2Assessment] = {}
    for assessment in stage2_evaluation.result.assessments:
        if assessment.node_id in stage2_by_node:
            raise Stage3EvaluationError(
                f"duplicate Stage 2 assessment for {assessment.node_id}"
            )
        stage2_by_node[assessment.node_id] = assessment
    resolutions_by_node: dict[str, Stage3ClaimResolution] = {}
    for resolution in resolutions:
        if resolution.node_id in resolutions_by_node:
            raise Stage3EvaluationError(
                f"duplicate Stage 3 resolution for {resolution.node_id}"
            )
        resolutions_by_node[resolution.node_id] = resolution

    stage1_resolved_nodes = {
        node_id
        for node_id in weights
        if _stage1_for_scored_node(graph, stage1_result, node_id) is not None
    }
    expected_stage2_nodes = set(weights) - stage1_resolved_nodes
    missing_stage2 = sorted(expected_stage2_nodes - set(stage2_by_node))
    extra_stage2 = sorted(set(stage2_by_node) - expected_stage2_nodes)
    if missing_stage2 or extra_stage2:
        raise Stage3EvaluationError(
            "Stage 2 assessment leaf set does not match unresolved scored leaves: "
            f"missing={missing_stage2!r}, extra={extra_stage2!r}"
        )

    task_by_id: dict[str, Stage2Task] = {}
    for task in stage2_evaluation.tasks:
        if task.task_id in task_by_id:
            raise Stage3EvaluationError(f"duplicate Stage 2 task {task.task_id}")
        task_by_id[task.task_id] = task
    for node_id, assessment in stage2_by_node.items():
        task = task_by_id.get(assessment.task_id)
        if task is None or task.node_id != node_id:
            raise Stage3EvaluationError(
                f"Stage 2 assessment/task identity mismatch for {node_id}"
            )

    expected_stage3_nodes = {
        node_id
        for node_id, assessment in stage2_by_node.items()
        if assessment.verdict is Stage2Verdict.UNKNOWN
    }
    missing_stage3 = sorted(expected_stage3_nodes - set(resolutions_by_node))
    extra_stage3 = sorted(set(resolutions_by_node) - expected_stage3_nodes)
    if missing_stage3 or extra_stage3:
        raise Stage3EvaluationError(
            "Stage 3 resolution leaf set does not match Stage 2 UNKNOWN leaves: "
            f"missing={missing_stage3!r}, extra={extra_stage3!r}"
        )
    for node_id, resolution in resolutions_by_node.items():
        if resolution.task_id != stage2_by_node[node_id].task_id:
            raise Stage3EvaluationError(
                f"Stage 3 resolution/task identity mismatch for {node_id}"
            )

    leaves: list[FinalLeafAssessment] = []
    for node_id, weight in sorted(weights.items()):
        stage1 = _stage1_for_scored_node(graph, stage1_result, node_id)
        if stage1 is not None:
            verdict = Stage3Verdict(_enum_text(stage1.verdict).upper())
            leaves.append(
                FinalLeafAssessment(
                    node_id=node_id,
                    weight=weight,
                    verdict=verdict,
                    decision_source="stage1_metadata",
                    confidence=1.0,
                    evidence_refs=(
                        EvidenceRef(
                            stage="stage1",
                            evidence_id=stage1.assessment_id,
                            kind="metadata",
                        ),
                    ),
                    rationale=stage1.rationale,
                )
            )
            continue

        stage2 = stage2_by_node.get(node_id)
        if stage2 is None:
            raise Stage3EvaluationError(
                f"missing Stage 2 assessment for unresolved scored leaf {node_id}"
            )
        if stage2.verdict in {Stage2Verdict.MATCH, Stage2Verdict.MISMATCH}:
            leaves.append(
                FinalLeafAssessment(
                    node_id=node_id,
                    task_id=stage2.task_id,
                    weight=weight,
                    verdict=Stage3Verdict(_enum_text(stage2.verdict).upper()),
                    decision_source="stage2_visual",
                    confidence=stage2.confidence,
                    evidence_refs=_evidence_refs(
                        "stage2", stage2.evidence_frame_ids
                    ),
                    rationale=stage2.rationale,
                )
            )
            continue

        resolution = resolutions_by_node.get(node_id)
        if resolution is None:
            raise Stage3EvaluationError(
                f"missing Stage 3 resolution for UNKNOWN leaf {node_id}"
            )
        if resolution.evaluation_error is not None or resolution.final_verdict is None:
            raise Stage3EvaluationError(
                f"Stage 3 resolver failed for {node_id}: {resolution.evaluation_error}"
            )
        leaves.append(
            FinalLeafAssessment(
                node_id=node_id,
                task_id=stage2.task_id,
                weight=weight,
                verdict=resolution.final_verdict,
                decision_source=(
                    "stage3_unknown_fallback"
                    if resolution.forced_mismatch
                    else (
                        "stage3_unresolved"
                        if resolution.final_verdict is Stage3Verdict.UNKNOWN
                        else "stage3_visual"
                    )
                ),
                confidence=resolution.confidence,
                evidence_refs=resolution.evidence_refs,
                rationale=resolution.rationale,
                stage3_raw_verdict=resolution.judge_verdict,
                forced_mismatch=resolution.forced_mismatch,
                forced_reason=resolution.forced_reason,
            )
        )

    total_weight = sum(value.weight for value in leaves)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise Stage3EvaluationError("final graph has no positive scoring weight")
    has_unknown = any(value.verdict is Stage3Verdict.UNKNOWN for value in leaves)
    matched_weight = sum(
        value.weight for value in leaves if value.verdict is Stage3Verdict.MATCH
    )
    return FinalGraphResult(
        assessments=tuple(leaves),
        requirements_score=None if has_unknown else matched_weight / total_weight,
        holistic_result=holistic_result,
    )


def _records(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    raw = getattr(value, name, ())
    try:
        return tuple(item for item in raw if isinstance(item, Mapping))
    except TypeError:
        return ()


def evaluate_stage3_detailed(
    graph: RequirementGraph,
    stage1_result: Stage1Result,
    stage2_evaluation: Stage2Evaluation,
    scene_bounds: SceneBounds,
    frame_provider: Any,
    explorer: Any,
    unknown_judge: Any,
    holistic_judge: Any,
    batch_policy: Stage3BatchPolicy | None = None,
) -> Stage3Evaluation:
    """Run Stage 3 on an already-entered provider and return an auditable result."""

    if not isinstance(graph, RequirementGraph):
        raise TypeError("graph must be a RequirementGraph")
    if not isinstance(stage1_result, Stage1Result):
        raise TypeError("stage1_result must be a Stage1Result")
    if not isinstance(stage2_evaluation, Stage2Evaluation):
        raise TypeError("stage2_evaluation must be a Stage2Evaluation")
    if not isinstance(scene_bounds, SceneBounds):
        raise TypeError("scene_bounds must be a SceneBounds")
    if stage2_evaluation.result.evaluation_error is not None:
        raise Stage3EvaluationError(
            "Stage 2 evaluation_error prevents trustworthy Stage 3 closure: "
            + stage2_evaluation.result.evaluation_error
        )

    exploration = explorer.explore(
        scene_bounds,
        frame_provider,
        stage2_frame_store=stage2_evaluation.frame_store,
        require_global_context=any(
            task.kind in GLOBAL_CONTEXT_TASK_KINDS
            for task in stage2_evaluation.tasks
        ),
    )
    if getattr(exploration, "runtime_error", None):
        raise Stage3EvaluationError(
            f"Stage 3 exploration failed: {exploration.runtime_error}"
        )
    portfolio_ids = tuple(getattr(exploration, "portfolio_frame_ids", ()) or ())
    global_context_required = bool(
        getattr(exploration, "global_context_required", True)
    )
    minimum = int(getattr(exploration.budget, "min_portfolio_frames", 4))
    if global_context_required and len(portfolio_ids) < minimum:
        raise Stage3EvaluationError(
            f"Stage 3 captured only {len(portfolio_ids)} usable portfolio frames; "
            f"minimum is {minimum}"
        )
    if global_context_required and not bool(
        getattr(exploration.coverage, "constraints_met", False)
    ):
        missing = tuple(
            _enum_text(value)
            for value in getattr(
                exploration.coverage,
                "missing_requirements",
                (),
            )
        )
        raise Stage3EvaluationError(
            "Stage 3 exploration portfolio did not meet controller coverage "
            f"requirements: {missing!r}"
        )
    portfolio_frames = exploration.judge_frames()
    policy = batch_policy or Stage3BatchPolicy(
        max_frames_per_request=int(
            getattr(unknown_judge, "max_frames_per_request", 6)
        )
    )
    judge_limit = getattr(unknown_judge, "max_frames_per_request", None)
    if judge_limit is not None and judge_limit != policy.max_frames_per_request:
        raise Stage3EvaluationError(
            "Stage 3 batch policy and UNKNOWN judge disagree on "
            "max_frames_per_request"
        )

    task_by_id = {task.task_id: task for task in stage2_evaluation.tasks}
    resolutions: list[Stage3ClaimResolution] = []
    batch_evaluations: list[Mapping[str, Any]] = []
    for assessment in stage2_evaluation.result.assessments:
        if assessment.verdict is not Stage2Verdict.UNKNOWN:
            continue
        task = task_by_id.get(assessment.task_id)
        if task is None:
            raise Stage3EvaluationError(
                f"UNKNOWN assessment has no Stage 2 task: {assessment.task_id}"
            )
        resolution, batch_evaluation = evaluate_stage3_unknown_batches(
            graph,
            task,
            exploration.judge_frames_for_task(task.task_id),
            unknown_judge,
            policy,
        )
        binary_audit = None
        if resolution.final_verdict is Stage3Verdict.UNKNOWN:
            resolution, binary_audit = evaluate_stage3_binary_arbitration(
                graph,
                task,
                exploration.judge_frames_for_task(task.task_id),
                unknown_judge,
            )
        batch_evaluation = dict(batch_evaluation)
        if binary_audit is not None:
            batch_evaluation["binary_arbitration"] = binary_audit
        batch_evaluation["aggregation"] = resolution.to_dict()
        resolutions.append(resolution)
        batch_evaluations.append(batch_evaluation)

    holistic_decision = holistic_judge.judge(graph.prompt, portfolio_frames)
    holistic_result = _coerce_holistic_result(holistic_decision)
    if holistic_result.evaluation_error is not None:
        raise Stage3EvaluationError(
            "Holistic judge failed: " + holistic_result.evaluation_error
        )
    final_result = reconcile_graph_results(
        graph,
        stage1_result,
        stage2_evaluation,
        resolutions,
        holistic_result,
    )

    components = (
        getattr(explorer, "selector", None),
        unknown_judge,
        holistic_judge,
    )
    manifest = tuple(
        record
        for component in components
        if component is not None
        for record in _records(component, "manifest_records")
    )
    raw = tuple(
        record
        for component in components
        if component is not None
        for record in _records(component, "raw_records")
    )
    return Stage3Evaluation(
        exploration=exploration,
        unknown_resolutions=tuple(resolutions),
        holistic_result=holistic_result,
        final_result=final_result,
        batch_evaluations=tuple(batch_evaluations),
        request_manifest=manifest,
        raw_records=raw,
    )


__all__ = [
    "Stage3Evaluation",
    "Stage3EvaluationError",
    "Stage3VlmComponents",
    "aggregate_stage3_batch_passes",
    "evaluate_stage3_binary_arbitration",
    "evaluate_stage3_unknown_batches",
    "evaluate_stage3_detailed",
    "finalize_stage3_binary_decision",
    "finalize_stage3_binary_resolution",
    "finalize_stage3_unknown_decision",
    "reconcile_graph_results",
    "stage3_actor_reframe_recommended",
]
