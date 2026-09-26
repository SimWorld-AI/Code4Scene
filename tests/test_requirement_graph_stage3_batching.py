"""Regression coverage for bounded multi-request Stage 3 claim judging."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from code4scene.evaluation.requirement_graph.actor_inventory import ActorBounds
from code4scene.evaluation.requirement_graph.asset_candidates import (
    assess_bounds_camera_pose,
    plan_bounds_camera_recovery_poses,
    plan_bounds_camera_poses,
)
from code4scene.evaluation.requirement_graph.bundle import (
    FrozenVerificationBundle,
    UnknownReason,
)
from code4scene.evaluation.requirement_graph.contracts import (
    CameraPose,
    ClaimVerdict,
    EntityNode,
    PredicateNode,
    SceneBounds,
)
from code4scene.evaluation.requirement_graph.existing_llm import LLMResponse, ToolCall
from code4scene.evaluation.requirement_graph.pipeline import (
    _ordered_concurrent_map,
    _run_stage3,
)
from code4scene.evaluation.requirement_graph.stage2_contracts import (
    CaptureShotRole,
    Stage2Assessment,
    Stage2EvidenceBasis,
    Stage2Task,
    Stage2TaskArgument,
    Stage2TaskKind,
    Stage2Verdict,
    VisualGrounding,
)
from code4scene.evaluation.requirement_graph.stage2_frames import FrameStore
from code4scene.evaluation.requirement_graph.stage2_routing import (
    Stage2ActorTarget,
    Stage2EntityRoute,
    Stage2RouteSource,
    Stage2RoutingPlan,
)
from code4scene.evaluation.requirement_graph.runtime import CapturedFrame
from code4scene.evaluation.requirement_graph.stage3 import (
    evaluate_stage3_binary_arbitration,
    evaluate_stage3_unknown_batches,
    finalize_stage3_unknown_decision,
    stage3_actor_reframe_recommended,
)
from code4scene.evaluation.requirement_graph.stage3_artifacts import (
    write_stage3_artifacts,
)
from code4scene.evaluation.requirement_graph.stage3_contracts import (
    Stage3BatchPolicy,
    Stage3Budget,
    Stage3Coverage,
    Stage3ExplorationResult,
    Stage3Verdict,
)
from code4scene.evaluation.requirement_graph.stage3_explorer import (
    capture_stage3_actor_reframes,
)
from code4scene.evaluation.requirement_graph.stage3_judge import (
    LLMStage3UnknownJudge,
    Stage3JudgeFrame,
    Stage3UnknownDecision,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stage3_policy_bounds_concurrency():
    assert Stage3BatchPolicy().policy_id == "stage3-dynamic-batching-v3"
    assert Stage3BatchPolicy().max_concurrent_claims == 32
    with pytest.raises(ValueError, match="max_concurrent_claims"):
        Stage3BatchPolicy(max_concurrent_claims=0)
    with pytest.raises(ValueError, match="max_concurrent_claims"):
        Stage3BatchPolicy(max_concurrent_claims=33)


def test_ordered_concurrent_map_is_parallel_but_returns_input_order():
    worker_count = 32
    barrier = threading.Barrier(worker_count)

    def work(value):
        barrier.wait(timeout=5.0)
        return value * 10

    values = tuple(reversed(range(worker_count)))
    assert _ordered_concurrent_map(
        values,
        work,
        max_workers=worker_count,
    ) == tuple(
        value * 10 for value in values
    )


def test_concurrent_batch_audits_keep_their_own_request_ids():
    bundle = _bundle()
    task = _task(bundle)
    barrier = threading.Barrier(2)

    class Client:
        model = "concurrent-request-id-test"
        _strict_tool_calls = True
        _text_action_mode = False

        def __init__(self):
            self._lock = threading.Lock()
            self._calls = 0

        def chat(self, messages, tools, **kwargs):
            del messages, tools, kwargs
            with self._lock:
                self._calls += 1
                call_number = self._calls
            # Both manifest entries must exist before either worker tries to
            # attach a request id to its batch audit.
            barrier.wait(timeout=2.0)
            return LLMResponse(
                text=None,
                tool_calls=(
                    ToolCall(
                        f"concurrent-{call_number}",
                        "record_stage3_unknown_verdict",
                        {
                            "verdict": "UNKNOWN",
                            "confidence": 0.0,
                            "evidence_frame_ids": [],
                            "rationale": "the subject is not sufficiently visible",
                            "evidence_basis": "insufficient",
                            "grounding": {
                                "subject_confirmed": False,
                                "participants_confirmed": False,
                                "relation_scope_covered": False,
                                "collection_complete": False,
                                "instances_countable": False,
                                "visible_instance_count": None,
                            },
                        },
                    ),
                ),
                raw={},
            )

    judge = LLMStage3UnknownJudge(Client())
    policy = Stage3BatchPolicy(
        max_batches_per_claim=1,
        max_concurrent_claims=2,
    )
    frame_sets = tuple((frame,) for frame in _judge_frames(2))
    results = _ordered_concurrent_map(
        frame_sets,
        lambda frames: evaluate_stage3_unknown_batches(
            bundle.graph,
            task,
            frames,
            judge,
            policy,
        ),
        max_workers=2,
    )

    manifest_by_id = {
        record["request_id"]: record for record in judge.manifest_records
    }
    request_ids = []
    for _, audit in results:
        batch = audit["batches"][0]
        request_ids.append(batch["request_id"])
        assert manifest_by_id[batch["request_id"]]["frame_ids"] == batch["frame_ids"]
    assert len(set(request_ids)) == 2


def _bundle() -> FrozenVerificationBundle:
    path = Path(__file__).resolve().parent / "fixtures/synthetic-alley.bundle.json"
    return FrozenVerificationBundle.from_dict(json.loads(path.read_text()))


def _task(bundle: FrozenVerificationBundle) -> Stage2Task:
    entity_id = next(
        node.id
        for node in bundle.graph.nodes
        if isinstance(node, EntityNode) and node.name.casefold() == "rollup doors"
    )
    node_id = next(
        node.id
        for node in bundle.graph.nodes
        if isinstance(node, PredicateNode)
        and node.name.casefold() == "rollup doors present"
    )
    return Stage2Task(
        task_id=f"stage2:{node_id}",
        node_id=node_id,
        kind=Stage2TaskKind.OBJECT_EXISTENCE,
        weight=bundle.graph.effective_weights()[node_id],
        arguments=(Stage2TaskArgument(entity_id, "subject"),),
    )


def _rgb(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0,
        256,
        size=(24, 32, 3),
        dtype=np.uint8,
    )


def _judge_frames(count: int) -> tuple[Stage3JudgeFrame, ...]:
    return tuple(
        Stage3JudgeFrame(f"s3f_{index:06d}", _rgb(index))
        for index in range(1, count + 1)
    )


def _unknown(rationale: str = "not visible") -> Stage3UnknownDecision:
    return Stage3UnknownDecision(
        verdict=Stage3Verdict.UNKNOWN,
        confidence=0.0,
        rationale=rationale,
        evidence_basis=Stage2EvidenceBasis.INSUFFICIENT,
        grounding=VisualGrounding(),
        transport_status="success",
        parse_status="valid",
    )


def _existence_claim() -> dict:
    return {
        "node_type": "predicate",
        "claim_text": "carpets are present",
        "predicate_name": "carpets present",
        "predicate_type": "existence",
        "polarity": "affirmative",
        "arguments": [
            {
                "role": "subject",
                "ordinal": 0,
                "claim_text": "carpets",
                "entity_name": "carpets",
            }
        ],
    }


def test_unknown_judge_declares_focus_first_target_subject_protocol():
    class Client:
        system_text = ""

        def chat(self, messages, tools, **kwargs):
            del tools, kwargs
            self.system_text = messages[0].content[0]["text"]
            return LLMResponse(text=None, tool_calls=(), raw={})

    client = Client()
    judge = LLMStage3UnknownJudge(client)
    judge.judge(
        {
            "node_type": "entity",
            "claim_text": "The retained dining chair faces the table.",
            "entity_name": "retained dining chair",
        },
        (Stage3JudgeFrame("s3f_000001", _rgb(1)),),
    )

    assert "focus-first order" in client.system_text
    assert "dominant centered instance" in client.system_text
    assert "designated subject" in client.system_text
    assert "same-category instances" in client.system_text
    assert "different routed Candidate Actors" in client.system_text
    assert "different surrounding scene" in client.system_text
    assert "positions, treat those centered targets" in client.system_text
    assert "routing cue establishes distinct observation targets" in client.system_text
    assert "unseen canonical/reference image" in client.system_text
    assert "seat's open/front direction" in client.system_text
    assert "backrest should lie on the side away" in client.system_text


def test_count_judge_canonicalizes_verdict_from_complete_visible_count():
    class Client:
        model = "count-test-model"
        _strict_tool_calls = True
        _text_action_mode = False

        def chat(self, messages, tools, **kwargs):
            del messages, tools, kwargs
            return LLMResponse(
                text=None,
                tool_calls=(
                    ToolCall(
                        "count-call",
                        "record_stage3_unknown_verdict",
                        {
                            "verdict": "MISMATCH",
                            "confidence": 0.78,
                            "evidence_frame_ids": ["s3f_000001"],
                            "rationale": (
                                "Four binders are visible, but the collection is not "
                                "cleanly limited to two."
                            ),
                            "evidence_basis": "sufficient_count_view",
                            "grounding": {
                                "subject_confirmed": True,
                                "participants_confirmed": True,
                                "relation_scope_covered": True,
                                "collection_complete": True,
                                "instances_countable": True,
                                "visible_instance_count": 4,
                            },
                        },
                    ),
                ),
                raw={},
            )

    decision = LLMStage3UnknownJudge(Client()).judge(
        {
            "node_type": "predicate",
            "claim_text": "At least two binders are visible inside the cupboard.",
            "predicate_name": "binder count",
            "predicate_type": "count",
            "polarity": "affirmative",
            "arguments": [
                {
                    "role": "collection",
                    "ordinal": 0,
                    "claim_text": "the restored Binder objects",
                    "entity_name": "Binders",
                }
            ],
            "constraint": {"operator": "gte", "value": 2, "upper_value": None},
        },
        (Stage3JudgeFrame("s3f_000001", _rgb(1)),),
    )

    assert decision.parse_status == "valid"
    assert decision.error is None
    assert decision.verdict is Stage3Verdict.MATCH
    assert decision.evidence_basis is Stage2EvidenceBasis.SUPPORT
    assert decision.grounding.visible_instance_count == 4
    assert "Structured count normalization" in decision.rationale


def test_unknown_judge_repairs_malformed_schema_on_bounded_retry():
    class Client:
        model = "validation-retry-test-model"
        _strict_tool_calls = True
        _text_action_mode = False

        def __init__(self):
            self.calls = 0
            self.system_texts = []

        def chat(self, messages, tools, **kwargs):
            del tools, kwargs
            self.calls += 1
            self.system_texts.append(messages[0].content[0]["text"])
            arguments = {
                "verdict": "MATCH",
                "confidence": 0.91,
                "evidence_frame_ids": ["s3f_000001"],
                "rationale": "a carpet is visible",
                "evidence_basis": "support",
                "grounding": {
                    "subject_confirmed": True,
                    "participants_confirmed": False,
                    "relation_scope_covered": False,
                    "collection_complete": False,
                    "instances_countable": False,
                    "visible_instance_count": None,
                },
            }
            if self.calls == 1:
                arguments["verdict\n=MATCH\n</parameter"] = arguments.pop("verdict")
            return LLMResponse(
                text=None,
                tool_calls=(
                    ToolCall(
                        f"call-{self.calls}",
                        "record_stage3_unknown_verdict",
                        arguments,
                    ),
                ),
                raw={},
            )

    client = Client()
    judge = LLMStage3UnknownJudge(client)
    decision = judge.judge(_existence_claim(), _judge_frames(1))

    assert client.calls == 2
    assert decision.verdict is Stage3Verdict.MATCH
    assert decision.parse_status == "valid"
    assert decision.error is None
    assert len(judge.manifest_records) == 2
    assert len(judge.raw_records) == 2
    assert judge.raw_records[0]["parse_status"] == "invalid"
    assert judge.raw_records[0]["validation_error"] == (
        "Judge tool arguments did not match the strict schema."
    )
    assert judge.raw_records[1]["parse_status"] == "valid"
    assert judge.raw_records[1]["retry_of"] == "s3r_000001"
    assert "previous structured response was rejected" in client.system_texts[1]


@pytest.mark.parametrize(
    ("grounding", "expected_error"),
    (
        (
            (
                '{"subject_confirmed": False, "participants_confirmed": False, '
                '"relation_scope_covered": True, "collection_complete": True, '
                '"instances_countable": True, "visible_instance_count": 0}'
            ),
            "Judge returned invalid visual grounding flags.",
        ),
        (
            {
                "subject_confirmed": False,
                "participants_confirmed": False,
                "relation_scope_covered": True,
                "collection_complete": False,
                "instances_countable": False,
                "visible_instance_count": 0,
            },
            "Searched-absence verdict requires complete collection grounding.",
        ),
    ),
)
def test_unknown_judge_exhausted_validation_retries_downgrade_to_unknown(
    grounding,
    expected_error,
):
    class Client:
        model = "validation-fallback-test-model"
        _strict_tool_calls = True
        _text_action_mode = False

        def __init__(self):
            self.calls = 0
            self.system_texts = []

        def chat(self, messages, tools, **kwargs):
            del tools, kwargs
            self.calls += 1
            self.system_texts.append(messages[0].content[0]["text"])
            return LLMResponse(
                text=None,
                tool_calls=(
                    ToolCall(
                        f"call-{self.calls}",
                        "record_stage3_unknown_verdict",
                        {
                            "verdict": "MISMATCH",
                            "confidence": 0.72,
                            "evidence_frame_ids": ["s3f_000001"],
                            "rationale": "no carpet is visible",
                            "evidence_basis": "searched_absence",
                            "grounding": grounding,
                        },
                    ),
                ),
                raw={},
            )

    client = Client()
    judge = LLMStage3UnknownJudge(client, max_validation_retries=2)
    decision = judge.judge(_existence_claim(), _judge_frames(1))

    assert client.calls == 3
    assert decision.verdict is Stage3Verdict.UNKNOWN
    assert decision.confidence == 0.0
    assert decision.evidence_frame_ids == ()
    assert decision.parse_status == "valid"
    assert decision.error is None
    assert "downgraded to UNKNOWN" in decision.rationale
    assert expected_error in decision.rationale
    assert len(judge.manifest_records) == 3
    assert len(judge.raw_records) == 3
    assert all(value["parse_status"] == "invalid" for value in judge.raw_records)
    assert all(
        value["validation_error"] == expected_error
        for value in judge.raw_records
    )
    assert "return UNKNOWN" in client.system_texts[1]


def test_binary_judge_closes_rgb_claim_without_conservative_completeness_gate():
    class Client:
        model = "binary-test-model"
        _strict_tool_calls = True
        _text_action_mode = False

        def __init__(self):
            self.system_text = ""

        def chat(self, messages, tools, **kwargs):
            del kwargs
            self.system_text = messages[0].content[0]["text"]
            assert tools[0]["name"] == "record_stage3_binary_verdict"
            return LLMResponse(
                text=None,
                tool_calls=(
                    ToolCall(
                        "binary-call",
                        "record_stage3_binary_verdict",
                        {
                            "verdict": "MATCH",
                            "confidence": 0.68,
                            "evidence_frame_ids": ["s3f_000001"],
                            "rationale": "the visible surface supports the claim",
                            "evidence_basis": "support",
                            "grounding": {
                                "subject_confirmed": False,
                                "participants_confirmed": False,
                                "relation_scope_covered": False,
                                "collection_complete": False,
                                "instances_countable": False,
                                "visible_instance_count": 0,
                            },
                        },
                    ),
                ),
                raw={},
            )

    client = Client()
    judge = LLMStage3UnknownJudge(client)
    rgb = _rgb(1)
    decision = judge.judge_binary(
        {
            "node_type": "entity",
            "claim_text": "muted blue painted shutters",
            "entity_name": "painted shutters",
        },
        (Stage3JudgeFrame("s3f_000001", rgb),),
    )

    assert decision.verdict is Stage3Verdict.MATCH
    assert decision.parse_status == "valid"
    assert decision.grounding.visible_instance_count is None
    assert "For colour" in client.system_text
    assert "For material" in client.system_text

def test_binary_arbitration_without_images_remains_inconclusive():
    bundle = _bundle()
    task = _task(bundle)

    class Judge:
        def judge_binary(self, *args):
            raise AssertionError("the judge must not be called without images")

    resolution, audit = evaluate_stage3_binary_arbitration(bundle.graph, task, (), Judge())

    assert resolution.final_verdict is Stage3Verdict.UNKNOWN
    assert resolution.evaluation_error is None
    assert audit["attempted"] is False


def test_incomplete_count_collection_requests_actor_reframe():
    task = Stage2Task(
        task_id="stage2:count",
        node_id="count",
        kind=Stage2TaskKind.COUNT,
        weight=1.0,
        arguments=(Stage2TaskArgument("boxes", "collection"),),
    )
    decision = Stage3UnknownDecision(
        verdict=Stage3Verdict.UNKNOWN,
        confidence=0.0,
        rationale="the collection is not fully countable",
        evidence_basis=Stage2EvidenceBasis.INSUFFICIENT,
        grounding=VisualGrounding(
            collection_complete=False,
            instances_countable=False,
        ),
        transport_status="success",
        parse_status="valid",
    )
    resolution = SimpleNamespace(
        final_verdict=Stage3Verdict.UNKNOWN,
        evaluation_error=None,
    )

    assert stage3_actor_reframe_recommended(task, decision, resolution) is True

    complete = Stage3UnknownDecision(
        verdict=Stage3Verdict.UNKNOWN,
        confidence=0.0,
        rationale="countable but another relation is unclear",
        evidence_basis=Stage2EvidenceBasis.INSUFFICIENT,
        grounding=VisualGrounding(
            collection_complete=True,
            instances_countable=True,
            visible_instance_count=2,
        ),
        transport_status="success",
        parse_status="valid",
    )
    assert stage3_actor_reframe_recommended(task, complete, resolution) is False


def _match(frame_id: str) -> Stage3UnknownDecision:
    return Stage3UnknownDecision(
        verdict=Stage3Verdict.MATCH,
        confidence=0.92,
        evidence_frame_ids=(frame_id,),
        rationale="the requested object is clearly visible",
        evidence_basis=Stage2EvidenceBasis.SUPPORT,
        grounding=VisualGrounding(subject_confirmed=True),
        transport_status="success",
        parse_status="valid",
    )


class _SequenceJudge:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []
        self.manifest_records = ()
        self.raw_records = ()

    def judge(self, payload, frames):
        del payload
        self.calls.append(tuple(value.frame_id for value in frames))
        return self.decisions.pop(0)


def test_stage3_claim_evidence_is_sent_in_two_bounded_batches():
    bundle = _bundle()
    task = _task(bundle)
    judge = _SequenceJudge((_unknown(), _match("s3f_000009")))

    resolution, audit = evaluate_stage3_unknown_batches(
        bundle.graph,
        task,
        _judge_frames(9),
        judge,
        Stage3BatchPolicy(max_frames_per_request=6, max_batches_per_claim=2),
    )

    assert tuple(map(len, judge.calls)) == (6, 3)
    assert judge.calls[0] == tuple(f"s3f_{value:06d}" for value in range(1, 7))
    assert judge.calls[1] == tuple(f"s3f_{value:06d}" for value in range(7, 10))
    assert resolution.final_verdict is Stage3Verdict.MATCH
    assert audit["selected_frame_ids"] == [
        f"s3f_{value:06d}" for value in range(1, 10)
    ]
    assert audit["omitted_frame_ids"] == []
    assert [len(value["frame_ids"]) for value in audit["batches"]] == [6, 3]


def test_stage3_batch_transport_error_is_not_hidden_by_another_match():
    bundle = _bundle()
    task = _task(bundle)
    failed = Stage3UnknownDecision(
        verdict=Stage3Verdict.UNKNOWN,
        confidence=0.0,
        rationale="request failed",
        evidence_basis=Stage2EvidenceBasis.INSUFFICIENT,
        grounding=VisualGrounding(),
        transport_status="error",
        parse_status="not_attempted",
        error="socket timeout",
    )
    judge = _SequenceJudge((failed, _match("s3f_000009")))

    resolution, audit = evaluate_stage3_unknown_batches(
        bundle.graph,
        task,
        _judge_frames(9),
        judge,
        Stage3BatchPolicy(),
    )

    assert resolution.final_verdict is None
    assert resolution.evaluation_error == "socket timeout"
    assert audit["aggregation"]["evaluation_error"] == "socket timeout"


def test_stage3_all_unknown_batches_remain_not_evaluated_not_error():
    bundle = _bundle()
    task = _task(bundle)
    judge = _SequenceJudge((_unknown("first view ambiguous"), _unknown("still hidden")))

    resolution, audit = evaluate_stage3_unknown_batches(
        bundle.graph,
        task,
        _judge_frames(9),
        judge,
        Stage3BatchPolicy(),
    )

    assert resolution.final_verdict is Stage3Verdict.UNKNOWN
    assert resolution.evaluation_error is None
    assert audit["aggregation"]["final_verdict"] == "UNKNOWN"


def test_stage3_dynamic_batches_stop_after_two_stable_verdicts():
    bundle = _bundle()
    task = _task(bundle)
    judge = _SequenceJudge((_match("s3f_000001"), _match("s3f_000007")))

    resolution, audit = evaluate_stage3_unknown_batches(
        bundle.graph,
        task,
        _judge_frames(30),
        judge,
        Stage3BatchPolicy(),
    )

    assert resolution.final_verdict is Stage3Verdict.MATCH
    assert len(judge.calls) == 2
    assert audit["stop_reason"] == "stable_verdict"
    assert audit["stable_verdict"] == "MATCH"
    assert len(audit["omitted_frame_ids"]) == 18


def test_stage3_dynamic_batches_process_more_than_old_eighteen_frame_cap():
    bundle = _bundle()
    task = _task(bundle)
    judge = _SequenceJudge(tuple(_unknown() for _ in range(5)))

    resolution, audit = evaluate_stage3_unknown_batches(
        bundle.graph,
        task,
        _judge_frames(25),
        judge,
        Stage3BatchPolicy(),
    )

    assert resolution.final_verdict is Stage3Verdict.UNKNOWN
    assert tuple(map(len, judge.calls)) == (6, 6, 6, 6, 1)
    assert audit["stop_reason"] == "evidence_exhausted"
    assert audit["omitted_frame_ids"] == []


def test_thin_actor_camera_recovery_prioritizes_diverse_broadside_views():
    bounds = SceneBounds((-5000.0, -5000.0, 0.0), (5000.0, 5000.0, 3000.0))
    poses = plan_bounds_camera_recovery_poses(
        (0.0, 0.0, 800.0),
        (1600.0, 15.0, 20.0),
        bounds,
    )

    assert len(poses) >= 6
    assert len({round(value.yaw / 20.0) for value in poses}) >= 4
    assert len({round(value.z / 100.0) for value in poses}) >= 2


def test_stage3_cannot_match_an_object_it_did_not_confirm_visually():
    bundle = _bundle()
    task = _task(bundle)
    decision = Stage3UnknownDecision(
        verdict=Stage3Verdict.MATCH,
        confidence=0.95,
        evidence_frame_ids=("s3f_000001",),
        rationale="the image seems consistent",
        evidence_basis=Stage2EvidenceBasis.SUPPORT,
        grounding=VisualGrounding(subject_confirmed=False),
        transport_status="success",
        parse_status="valid",
    )

    resolution = finalize_stage3_unknown_decision(bundle.graph, task, decision)

    assert resolution.final_verdict is Stage3Verdict.UNKNOWN
    assert resolution.confidence == 0.0
    assert resolution.evidence_refs == ()


def test_stage3_writer_publishes_and_joins_batch_audit(tmp_path):
    bundle = _bundle()
    task = _task(bundle)
    store = FrameStore(valid_frame_hard_cap=10, frame_id_prefix="s3f")
    frames = []
    for index in range(1, 10):
        admitted = store.admit(
            _rgb(index),
            pose=CameraPose(float(index * 100), 0.0, 200.0, 0.0, 0.0),
            phase="stage3_test",
            shot_role="context",
        )
        assert admitted.record is not None
        frames.append(
            Stage3JudgeFrame(
                admitted.record.frame_id,
                admitted.record.rgb,
            )
        )

    class Client:
        model = "batch-test-model"
        _strict_tool_calls = True
        _text_action_mode = False

        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools, **kwargs):
            del messages, tools, kwargs
            self.calls += 1
            if self.calls == 1:
                verdict = "UNKNOWN"
                confidence = 0.0
                evidence_ids = []
                rationale = "the first batch is ambiguous"
                basis = "insufficient"
                subject_confirmed = False
            else:
                verdict = "MATCH"
                confidence = 0.92
                evidence_ids = ["s3f_000009"]
                rationale = "the second batch clearly shows the object"
                basis = "support"
                subject_confirmed = True
            return LLMResponse(
                text=None,
                tool_calls=(
                    ToolCall(
                        f"call-{self.calls}",
                        "record_stage3_unknown_verdict",
                        {
                            "verdict": verdict,
                            "confidence": confidence,
                            "evidence_frame_ids": evidence_ids,
                            "rationale": rationale,
                            "evidence_basis": basis,
                            "grounding": {
                                "subject_confirmed": subject_confirmed,
                                "participants_confirmed": False,
                                "relation_scope_covered": False,
                                "collection_complete": False,
                                "instances_countable": False,
                                "visible_instance_count": None,
                            },
                        },
                    ),
                ),
                raw={},
            )

    policy = Stage3BatchPolicy()
    judge = LLMStage3UnknownJudge(
        Client(),
        max_frames_per_request=policy.max_frames_per_request,
    )
    resolution, audit = evaluate_stage3_unknown_batches(
        bundle.graph,
        task,
        tuple(frames),
        judge,
        policy,
    )
    final_result = {
        "schema_version": "1.0",
        "status": "complete",
        "evaluation_error": None,
        "requirements_score": 1.0,
        "assessments": [
            {
                "node_id": task.node_id,
                "verdict": "MATCH",
                "decision_source": "stage3",
                "forced_mismatch": False,
            }
        ],
    }
    written = write_stage3_artifacts(
        tmp_path / "stage3",
        exploration={
            "schema_version": "1.0",
            "portfolio_frame_ids": [value.frame_id for value in frames[:6]],
        },
        frame_store=store,
        unknown_resolutions=(resolution,),
        batch_evaluations=(audit,),
        batch_policy=policy,
        holistic_result={
            "status": "delegated",
            "overall_score": None,
            "evaluation_error": None,
        },
        final_result=final_result,
        request_manifest=judge.manifest_records,
        raw_records=judge.raw_records,
    )

    assert any(value.name == "stage3_batch_results.json" for value in written)
    batch_artifact = json.loads(
        (tmp_path / "stage3/stage3_batch_results.json").read_text()
    )
    summary = json.loads((tmp_path / "stage3/graph_summary.json").read_text())
    assert [
        len(value["frame_ids"])
        for value in batch_artifact["claims"][0]["batches"]
    ] == [6, 3]
    assert summary["unknown_resolution_batch_count"] == 2
    assert summary["evaluation_error"] is None


def test_production_stage3_path_batches_nine_task_frames(monkeypatch, tmp_path):
    bundle = _bundle()
    task = _task(bundle)
    frames = _judge_frames(9)

    store = FrameStore(valid_frame_hard_cap=10, frame_id_prefix="s3f")
    for index, frame in enumerate(frames):
        admitted = store.admit(
            frame.rgb,
            pose=CameraPose(float(index * 100), 0.0, 200.0, 0.0, 0.0),
            phase="stage3_test",
            shot_role="context",
        )
        assert admitted.record is not None

    class Exploration:
        runtime_error = None
        frame_store = store
        portfolio_frame_ids = tuple(value.frame_id for value in frames[:6])
        task_frame_ids = {task.task_id: tuple(value.frame_id for value in frames)}
        frame_actor_ids = {}

        def judge_frames_for_task(self, task_id):
            assert task_id == task.task_id
            return frames

        def to_dict(self):
            return {
                "schema_version": "1.0",
                "portfolio_frame_ids": list(self.portfolio_frame_ids),
                "runtime_error": None,
            }

    class Explorer:
        def __init__(self, *, budget):
            self.budget = budget

        def explore(self, *args, **kwargs):
            del args, kwargs
            return Exploration()

    judge_instances = []

    class Judge(_SequenceJudge):
        def __init__(self, client, *, max_tokens, max_frames_per_request):
            del client, max_tokens
            assert max_frames_per_request == 6
            failed = Stage3UnknownDecision(
                verdict=Stage3Verdict.UNKNOWN,
                confidence=0.0,
                rationale="request failed",
                evidence_basis=Stage2EvidenceBasis.INSUFFICIENT,
                grounding=VisualGrounding(),
                transport_status="error",
                parse_status="not_attempted",
                error="socket timeout",
            )
            super().__init__((failed, _match("s3f_000009")))
            judge_instances.append(self)

    written = {}

    def write_artifacts(output_dir, **kwargs):
        written.update(kwargs)
        return (Path(output_dir) / "stage3_batch_results.json",)

    from code4scene.evaluation.requirement_graph import stage3_artifacts
    from code4scene.evaluation.requirement_graph import stage3_explorer
    from code4scene.evaluation.requirement_graph import stage3_judge

    monkeypatch.setattr(stage3_explorer, "Stage3Explorer", Explorer)
    monkeypatch.setattr(stage3_judge, "LLMStage3UnknownJudge", Judge)
    monkeypatch.setattr(stage3_artifacts, "write_stage3_artifacts", write_artifacts)

    assessment = Stage2Assessment(
        task_id=task.task_id,
        node_id=task.node_id,
        verdict=Stage2Verdict.UNKNOWN,
        rationale="Stage 2 could not decide",
        unknown_reason="visual_evidence_incomplete",
    )
    stage2 = SimpleNamespace(
        tasks=(task,),
        result=SimpleNamespace(assessments=(assessment,)),
        routing_plan=Stage2RoutingPlan(()),
        frame_store=FrameStore(),
    )
    atomic = {
        node_id: {
            "node_id": node_id,
            "verdict": ClaimVerdict.MATCH.value,
            "resolved_by": "stage1",
            "unknown_reason": None,
            "rationale": "pre-resolved test leaf",
        }
        for node_id in bundle.semantic_nodes()
    }
    atomic[task.node_id].update(
        verdict=ClaimVerdict.UNKNOWN.value,
        resolved_by=None,
        unknown_reason=UnknownReason.VISUAL_EVIDENCE_INCOMPLETE.value,
    )
    provider = SimpleNamespace(
        scene=SimpleNamespace(
            scene_bounds=SimpleNamespace(),
            inventory=None,
        )
    )

    result = _run_stage3(
        bundle,
        atomic,
        stage2,
        provider,
        SimpleNamespace(max_tokens=640),
        tmp_path,
        Stage3Budget(),
        Stage3BatchPolicy(),
    )

    assert result is not None
    assert tuple(map(len, judge_instances[0].calls)) == (6, 3)
    assert atomic[task.node_id]["verdict"] == ClaimVerdict.UNKNOWN.value
    assert atomic[task.node_id]["unknown_reason"] == "evaluation_error"
    assert result["final_result"]["evaluation_error"] == "socket timeout"
    assert result["final_result"]["status"] == "evaluation_error"
    assert len(written["batch_evaluations"][0]["batches"]) == 2
    assert written["batch_policy"].policy_id == "stage3-dynamic-batching-v3"
    audit_frames = written["capture_audit"]["frames"]
    assert {task.task_id} == set(audit_frames[0]["task_ids"])
    assert audit_frames[0]["requirement_ids"] == [
        bundle.binding_for(task.node_id).requirement_id
    ]
    visible_text = json.dumps(written["judge_visible_frames"], sort_keys=True)
    assert "actor" not in visible_text.casefold()
    assert "bounds" not in visible_text.casefold()


def test_adaptive_reframe_shares_one_new_view_for_same_actor_tasks():
    from code4scene.evaluation.requirement_graph.contracts import SceneBounds

    scene_bounds = SceneBounds(
        (-2000.0, -2000.0, -1000.0),
        (2000.0, 2000.0, 2000.0),
    )
    bounds = ActorBounds((0.0, 0.0, 220.0), (60.0, 45.0, 100.0))
    target = Stage2ActorTarget("shared-actor", bounds)
    budget = Stage3Budget(
        max_seed_frames=1,
        max_new_capture_attempts=0,
        max_valid_exploration_frames=3,
        min_portfolio_frames=1,
        max_portfolio_frames=1,
        max_recovery_attempts=0,
    )
    store = FrameStore(budget, frame_id_prefix="s3f")
    initial_pose = plan_bounds_camera_poses(
        bounds.center_cm,
        bounds.extent_cm,
        scene_bounds,
    )[1]
    admitted = store.admit(
        _rgb(70),
        pose=initial_pose,
        phase="stage3_seed",
        shot_role=CaptureShotRole.CLOSE,
        task_ids=("task:a", "task:b"),
        actor_ids=(target.actor_id,),
    )
    assert admitted.record is not None
    initial_id = admitted.record.frame_id
    exploration = Stage3ExplorationResult(
        budget=budget,
        attempts=(),
        coverage=Stage3Coverage(),
        portfolio_frame_ids=(initial_id,),
        frame_store=store,
        task_frame_ids={
            "task:a": (initial_id,),
            "task:b": (initial_id,),
        },
        frame_actor_ids={initial_id: (target.actor_id,)},
    )

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
            del kwargs
            requested = tuple(poses)
            self.calls.append(requested)
            return [
                CapturedFrame(
                    f"shared-reframe-{index}",
                    pose,
                    _rgb(70 + index),
                )
                for index, pose in enumerate(requested, start=1)
            ]

    provider = Provider()
    reframed = capture_stage3_actor_reframes(
        exploration,
        scene_bounds,
        provider,
        targets_by_task={
            "task:a": (target,),
            "task:b": (target,),
        },
    )

    assert len(provider.calls) == 1
    assert len(provider.resolutions) == 1
    assert provider.resolutions[0][1] == (("shared-actor",),)
    assert len(provider.calls[0]) == 1
    a_new = set(reframed.task_frame_ids["task:a"]) - {initial_id}
    b_new = set(reframed.task_frame_ids["task:b"]) - {initial_id}
    assert len(a_new) == 1
    assert a_new == b_new
    new_id = next(iter(a_new))
    assert reframed.frame_actor_ids[new_id] == (target.actor_id,)
    assert len(reframed.attempts) == 1


def test_adaptive_reframe_routes_collection_recovery_before_actor_views():
    from code4scene.evaluation.requirement_graph.contracts import SceneBounds

    scene_bounds = SceneBounds(
        (-2000.0, -2000.0, -1000.0),
        (2000.0, 2000.0, 2000.0),
    )
    left = Stage2ActorTarget(
        "torch-left",
        ActorBounds((-300.0, 0.0, 100.0), (20.0, 20.0, 90.0)),
    )
    right = Stage2ActorTarget(
        "torch-right",
        ActorBounds((300.0, 0.0, 100.0), (20.0, 20.0, 90.0)),
    )
    budget = Stage3Budget(
        max_seed_frames=1,
        max_new_capture_attempts=0,
        max_valid_exploration_frames=4,
        min_portfolio_frames=1,
        max_portfolio_frames=1,
        max_recovery_attempts=0,
    )
    store = FrameStore(budget, frame_id_prefix="s3f")
    initial_pose = plan_bounds_camera_poses(
        left.bounds.center_cm,
        left.bounds.extent_cm,
        scene_bounds,
    )[0]
    admitted = store.admit(
        _rgb(80),
        pose=initial_pose,
        phase="stage3_seed",
        shot_role=CaptureShotRole.COLLECTION_WIDE,
        task_ids=("task:count",),
        actor_ids=(left.actor_id, right.actor_id),
    )
    assert admitted.record is not None
    initial_id = admitted.record.frame_id
    exploration = Stage3ExplorationResult(
        budget=budget,
        attempts=(),
        coverage=Stage3Coverage(),
        portfolio_frame_ids=(initial_id,),
        frame_store=store,
        task_frame_ids={"task:count": (initial_id,)},
        frame_actor_ids={initial_id: (left.actor_id, right.actor_id)},
    )

    class Provider:
        def __init__(self):
            self.calls = []

        def capture(self, poses, **kwargs):
            del kwargs
            requested = tuple(poses)
            self.calls.append(requested)
            return [
                CapturedFrame(
                    f"union-reframe-{index}",
                    pose,
                    _rgb(80 + index),
                )
                for index, pose in enumerate(requested, start=1)
            ]

    provider = Provider()
    reframed = capture_stage3_actor_reframes(
        exploration,
        scene_bounds,
        provider,
        targets_by_task={"task:count": (left, right)},
    )

    assert len(provider.calls) == 1
    assert len(provider.calls[0]) == 3
    union_bounds = ActorBounds.from_min_max(
        tuple(
            min(left.bounds.min_cm[axis], right.bounds.min_cm[axis])
            for axis in range(3)
        ),
        tuple(
            max(left.bounds.max_cm[axis], right.bounds.max_cm[axis])
            for axis in range(3)
        ),
    )
    assert assess_bounds_camera_pose(
        union_bounds.center_cm,
        union_bounds.extent_cm,
        provider.calls[0][0],
    ).healthy
    assert all(
        assess_bounds_camera_pose(
            target.bounds.center_cm,
            target.bounds.extent_cm,
            pose,
        ).healthy
        for target, pose in zip((left, right), provider.calls[0][1:], strict=True)
    )
    new_ids = set(reframed.task_frame_ids["task:count"]) - {initial_id}
    assert len(new_ids) == 3
    assert {
        reframed.frame_actor_ids[new_id] for new_id in new_ids
    } == {
        (left.actor_id, right.actor_id),
        (left.actor_id,),
        (right.actor_id,),
    }


@pytest.mark.parametrize("binary_failure", [None, "transport", "parse"])
def test_production_stage3_reframes_then_uses_binary_vlm_after_unknown(
    tmp_path,
    binary_failure,
):
    bundle = _bundle()
    task = _task(bundle)
    bounds = ActorBounds((0.0, 0.0, 220.0), (60.0, 45.0, 100.0))
    scene_bounds = SimpleNamespace(
        min_cm=(-2000.0, -2000.0, -1000.0),
        max_cm=(2000.0, 2000.0, 2000.0),
    )
    from code4scene.evaluation.requirement_graph.contracts import SceneBounds

    scene_bounds = SceneBounds(scene_bounds.min_cm, scene_bounds.max_cm)
    initial_pose = plan_bounds_camera_poses(
        bounds.center_cm,
        bounds.extent_cm,
        scene_bounds,
    )[1]
    stage2_store = FrameStore(valid_frame_hard_cap=3)
    admitted = stage2_store.admit(
        _rgb(41),
        pose=initial_pose,
        phase="stage2_capture",
        shot_role=CaptureShotRole.CLOSE,
        task_ids=(task.task_id,),
        actor_ids=("Rollup1_X-58",),
    )
    assert admitted.record is not None
    overview = stage2_store.admit(
        _rgb(42),
        pose=CameraPose(-1200.0, -900.0, 500.0, -12.0, 35.0),
        phase="stage2_capture",
        shot_role=CaptureShotRole.OVERVIEW,
    )
    assert overview.record is not None

    class Provider:
        def __init__(self):
            self.scene = SimpleNamespace(scene_bounds=scene_bounds, inventory=None)
            self.calls = []

        def capture(self, poses, **kwargs):
            requested = tuple(poses)
            self.calls.append((requested, kwargs))
            return [
                CapturedFrame(
                    f"adaptive-{index}",
                    pose,
                    _rgb(50 + index),
                )
                for index, pose in enumerate(requested, start=1)
            ]

    class Client:
        model = "adaptive-test-model"
        max_tokens = 640
        _strict_tool_calls = True
        _text_action_mode = False

        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools, **kwargs):
            del messages, kwargs
            self.calls += 1
            if tools[0]["name"] == "record_stage3_binary_verdict":
                if binary_failure == "transport":
                    raise TimeoutError("binary judge timed out")
                if binary_failure == "parse":
                    return LLMResponse(text="malformed binary reply", tool_calls=(), raw={})
                return LLMResponse(
                    text=None,
                    tool_calls=(
                        ToolCall(
                            f"call-{self.calls}",
                            "record_stage3_binary_verdict",
                            {
                                "verdict": "MISMATCH",
                                "confidence": 0.63,
                                "evidence_frame_ids": ["s3f_000002"],
                                "rationale": (
                                    "the best available views do not support "
                                    "the requested rollup doors"
                                ),
                                "evidence_basis": "searched_absence",
                                "grounding": {
                                    "subject_confirmed": False,
                                    "participants_confirmed": False,
                                    "relation_scope_covered": False,
                                    "collection_complete": False,
                                    "instances_countable": False,
                                    "visible_instance_count": None,
                                },
                            },
                        ),
                    ),
                    raw={},
                )
            return LLMResponse(
                text=None,
                tool_calls=(
                    ToolCall(
                        f"call-{self.calls}",
                        "record_stage3_unknown_verdict",
                        {
                            "verdict": "UNKNOWN",
                            "confidence": 0.0,
                            "evidence_frame_ids": [],
                            "rationale": (
                                "the requested actor is occluded in this view"
                                if self.calls == 1
                                else "the second angle still does not confirm the requirement"
                            ),
                            "evidence_basis": "insufficient",
                            "grounding": {
                                "subject_confirmed": False,
                                "participants_confirmed": False,
                                "relation_scope_covered": False,
                                "collection_complete": False,
                                "instances_countable": False,
                                "visible_instance_count": None,
                            },
                        },
                    ),
                ),
                raw={},
            )

    assessment = Stage2Assessment(
        task_id=task.task_id,
        node_id=task.node_id,
        verdict=Stage2Verdict.UNKNOWN,
        rationale="Stage 2 needs RGB confirmation",
        unknown_reason="visual_evidence_incomplete",
    )
    stage2 = SimpleNamespace(
        tasks=(task,),
        result=SimpleNamespace(assessments=(assessment,)),
        routing_plan=Stage2RoutingPlan(
            (
                Stage2EntityRoute(
                    task.arguments[0].entity_id,
                    Stage2RouteSource.LOCATOR_ONLY,
                    (Stage2ActorTarget("Rollup1_X-58", bounds),),
                    stage2_visual_only=True,
                ),
            )
        ),
        frame_store=stage2_store,
    )
    atomic = {
        node_id: {
            "node_id": node_id,
            "verdict": ClaimVerdict.MATCH.value,
            "resolved_by": "stage1",
            "unknown_reason": None,
            "rationale": "pre-resolved test leaf",
        }
        for node_id in bundle.semantic_nodes()
    }
    atomic[task.node_id].update(
        verdict=ClaimVerdict.UNKNOWN.value,
        resolved_by=None,
        unknown_reason=UnknownReason.VISUAL_EVIDENCE_INCOMPLETE.value,
    )
    provider = Provider()

    client = Client()
    result = _run_stage3(
        bundle,
        atomic,
        stage2,
        provider,
        client,
        tmp_path,
        Stage3Budget(
            max_seed_frames=2,
            max_new_capture_attempts=0,
            max_valid_exploration_frames=3,
            min_portfolio_frames=1,
            max_portfolio_frames=2,
            max_recovery_attempts=0,
        ),
        Stage3BatchPolicy(),
    )

    assert result is not None
    assert len(provider.calls) == 1
    poses, capture_options = provider.calls[0]
    assert capture_options["phase"] == "stage3_adaptive_reframe"
    assert len(poses) == 1
    assert poses[0] != initial_pose
    assert assess_bounds_camera_pose(
        bounds.center_cm,
        bounds.extent_cm,
        poses[0],
    ).healthy
    assert client.calls == (5 if binary_failure == "parse" else 3)
    manifest = json.loads(
        Path(result["artifact_paths"]["stage3_vlm_request_manifest.json"])
        .read_text()
    )
    expected_binary_attempts = 3 if binary_failure == "parse" else 1
    assert [value["frame_ids"] for value in manifest] == [
        ["s3f_000001"],
        ["s3f_000002"],
        *(
            [["s3f_000001", "s3f_000002"]]
            * expected_binary_attempts
        ),
    ]
    assert [value["call_kind"] for value in manifest] == [
        "unknown_resolution",
        "unknown_resolution",
        *(["binary_resolution"] * expected_binary_attempts),
    ]
    summary = json.loads(
        Path(result["artifact_paths"]["graph_summary.json"]).read_text()
    )
    assert summary["vlm_call_counts"]["binary_resolution"] == expected_binary_attempts
    if binary_failure is None:
        assert atomic[task.node_id]["verdict"] == ClaimVerdict.MISMATCH.value
        assert atomic[task.node_id]["unknown_reason"] is None
        assert atomic[task.node_id]["forced_mismatch"] is False
        assert result["final_result"]["evaluation_error"] is None
    elif binary_failure == "transport":
        assert atomic[task.node_id]["verdict"] == ClaimVerdict.UNKNOWN.value
        assert atomic[task.node_id]["unknown_reason"] == "evaluation_error"
        assert result["final_result"]["status"] == "evaluation_error"
        assert result["final_result"]["evaluation_error"]
        assert summary["status"] == "evaluation_error"
        assert summary["evaluation_error"] == result["final_result"]["evaluation_error"]
    else:
        assert atomic[task.node_id]["verdict"] == ClaimVerdict.UNKNOWN.value
        assert atomic[task.node_id]["unknown_reason"] == (
            UnknownReason.VISUAL_EVIDENCE_INCOMPLETE.value
        )
        assert result["final_result"]["status"] == "not_evaluated"
        assert result["final_result"]["evaluation_error"] is None
        assert summary["status"] == "not_evaluated"
        assert summary["evaluation_error"] is None
    batch_artifact = json.loads(
        Path(result["artifact_paths"]["stage3_batch_results.json"]).read_text()
    )
    batch_audit = batch_artifact["claims"][0]
    batches = batch_audit["batches"]
    assert batches[0]["reframe_recommended"] is True
    assert [value["batch_index"] for value in batches] == [1, 2]
    assert (
        batch_audit["selected_frame_ids"] + batch_audit["omitted_frame_ids"]
        == batch_audit["available_frame_ids"]
    )
    expected_verdict = (
        "MISMATCH"
        if binary_failure is None
        else ("UNKNOWN" if binary_failure == "parse" else None)
    )
    assert batch_audit["aggregation"]["judge_verdict"] == expected_verdict
    assert batch_audit["aggregation"]["final_verdict"] == expected_verdict
    assert batch_audit["aggregation"]["forced_mismatch"] is False
    assert batch_audit["binary_arbitration"]["attempted"] is True
    if binary_failure == "transport":
        assert batch_audit["aggregation"]["evaluation_error"]
        binary = batch_audit["binary_arbitration"]
        assert (binary["transport_status"], binary["parse_status"]) == (
            "error",
            "not_attempted",
        )
    elif binary_failure == "parse":
        assert batch_audit["aggregation"]["evaluation_error"] is None
        binary = batch_audit["binary_arbitration"]
        assert (binary["transport_status"], binary["parse_status"]) == (
            "success",
            "valid",
        )
        assert binary["resolution"]["judge_verdict"] == "UNKNOWN"
        assert len(binary["attempt_request_ids"]) == expected_binary_attempts
    capture_artifact = json.loads(
        Path(result["artifact_paths"]["stage3_capture_audit.json"]).read_text()
    )
    reframe = next(
        value
        for value in capture_artifact["frames"]
        if value["frame_id"] == "s3f_000002"
    )
    assert reframe["capture_target_actor_ids"] == ["Rollup1_X-58"]
    assert reframe["shot_role"] == CaptureShotRole.OBLIQUE.value


@pytest.mark.parametrize(
    "unknown_reason",
    (
        UnknownReason.TARGET_NOT_LOCALIZED.value,
        UnknownReason.TARGETED_EVIDENCE_MISSING.value,
    ),
)
def test_stage3_does_not_construct_a_vlm_client_without_eligible_claims(
    monkeypatch, tmp_path, unknown_reason
):
    bundle = _bundle()
    task = _task(bundle)
    assessment = Stage2Assessment(
        task_id=task.task_id,
        node_id=task.node_id,
        verdict=Stage2Verdict.UNKNOWN,
        rationale="no requirement-linked evidence",
        unknown_reason=unknown_reason,
    )
    stage2 = SimpleNamespace(
        tasks=(task,),
        result=SimpleNamespace(assessments=(assessment,)),
        routing_plan=Stage2RoutingPlan(()),
        frame_store=FrameStore(),
    )
    atomic = {
        task.node_id: {
            "node_id": task.node_id,
            "verdict": ClaimVerdict.UNKNOWN.value,
            "resolved_by": None,
            "unknown_reason": unknown_reason,
        }
    }

    from code4scene.evaluation.requirement_graph import pipeline

    monkeypatch.setattr(
        pipeline,
        "tool_client_from_env",
        lambda: (_ for _ in ()).throw(AssertionError("VLM client was constructed")),
    )
    result = _run_stage3(
        bundle,
        atomic,
        stage2,
        SimpleNamespace(scene=SimpleNamespace()),
        None,
        tmp_path,
        Stage3Budget(),
        Stage3BatchPolicy(),
    )

    assert result is None
    assert atomic[task.node_id]["verdict"] == ClaimVerdict.UNKNOWN.value
    assert atomic[task.node_id]["resolved_by"] is None
    assert atomic[task.node_id]["unknown_reason"] == unknown_reason


def test_stage3_recovers_targeted_evidence_missing_for_overview_legal_task(
    monkeypatch, tmp_path
):
    bundle = _bundle()
    source_task = _task(bundle)
    task = Stage2Task(
        task_id=source_task.task_id,
        node_id=source_task.node_id,
        kind=Stage2TaskKind.SCENE_IDENTITY,
        weight=source_task.weight,
    )
    stage3_store = FrameStore(valid_frame_hard_cap=2, frame_id_prefix="s3f")
    frames = []
    for value, pose in (
        (94, CameraPose(-1200.0, -900.0, 500.0, -12.0, 35.0)),
        (111, CameraPose(1100.0, 850.0, 560.0, -10.0, -145.0)),
    ):
        admitted = stage3_store.admit(
            _rgb(value),
            pose=pose,
            phase="formal_overview_reuse",
            shot_role=CaptureShotRole.OVERVIEW,
            task_ids=(task.task_id,),
        )
        assert admitted.record is not None
        frames.append(Stage3JudgeFrame(admitted.record.frame_id, admitted.record.rgb))

    class Exploration:
        runtime_error = None
        frame_store = stage3_store
        portfolio_frame_ids = tuple(frame.frame_id for frame in frames)
        task_frame_ids = {
            task.task_id: tuple(frame.frame_id for frame in frames)
        }
        frame_actor_ids = {}

        def judge_frames_for_task(self, task_id):
            assert task_id == task.task_id
            return tuple(frames)

        def to_dict(self):
            return {
                "schema_version": "1.0",
                "portfolio_frame_ids": [frame.frame_id for frame in frames],
                "runtime_error": None,
            }

    class Explorer:
        def __init__(self, *, budget):
            self.budget = budget

        def explore(self, *args, **kwargs):
            del args, kwargs
            return Exploration()

    judge_instances = []

    class Judge(_SequenceJudge):
        def __init__(self, client, *, max_tokens, max_frames_per_request):
            del client, max_tokens, max_frames_per_request
            super().__init__(
                (
                    Stage3UnknownDecision(
                        verdict=Stage3Verdict.MATCH,
                        confidence=0.92,
                        evidence_frame_ids=tuple(
                            frame.frame_id for frame in frames
                        ),
                        rationale="the requested scene identity is clearly visible",
                        evidence_basis=Stage2EvidenceBasis.SUPPORT,
                        grounding=VisualGrounding(),
                        transport_status="success",
                        parse_status="valid",
                    ),
                )
            )
            judge_instances.append(self)

    def write_artifacts(output_dir, **kwargs):
        del kwargs
        return (Path(output_dir) / "stage3_batch_results.json",)

    from code4scene.evaluation.requirement_graph import stage3_artifacts
    from code4scene.evaluation.requirement_graph import stage3_explorer
    from code4scene.evaluation.requirement_graph import stage3_judge

    monkeypatch.setattr(stage3_explorer, "Stage3Explorer", Explorer)
    monkeypatch.setattr(stage3_judge, "LLMStage3UnknownJudge", Judge)
    monkeypatch.setattr(stage3_artifacts, "write_stage3_artifacts", write_artifacts)

    assessment = Stage2Assessment(
        task_id=task.task_id,
        node_id=task.node_id,
        verdict=Stage2Verdict.UNKNOWN,
        rationale="targeted capture failed",
        unknown_reason="no_valid_evidence",
    )
    stage2 = SimpleNamespace(
        tasks=(task,),
        result=SimpleNamespace(assessments=(assessment,)),
        routing_plan=Stage2RoutingPlan(()),
        frame_store=FrameStore(),
    )
    atomic = {
        node_id: {
            "node_id": node_id,
            "verdict": ClaimVerdict.MATCH.value,
            "resolved_by": "stage1",
            "unknown_reason": None,
            "rationale": "pre-resolved test leaf",
        }
        for node_id in bundle.semantic_nodes()
    }
    atomic[task.node_id].update(
        verdict=ClaimVerdict.UNKNOWN.value,
        resolved_by=None,
        unknown_reason=UnknownReason.TARGETED_EVIDENCE_MISSING.value,
    )
    provider = SimpleNamespace(
        scene=SimpleNamespace(scene_bounds=SimpleNamespace(), inventory=None)
    )

    result = _run_stage3(
        bundle,
        atomic,
        stage2,
        provider,
        SimpleNamespace(max_tokens=640),
        tmp_path,
        Stage3Budget(),
        Stage3BatchPolicy(),
    )

    assert result is not None
    assert judge_instances[0].calls == [
        tuple(frame.frame_id for frame in frames)
    ]
    assert atomic[task.node_id]["verdict"] == ClaimVerdict.MATCH.value
    assert atomic[task.node_id]["resolved_by"] == "stage3"
    assert atomic[task.node_id]["unknown_reason"] is None
