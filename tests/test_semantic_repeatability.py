"""Regression tests for fresh calls, fixed inputs, failure handling and summaries."""
import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from code4scene.evaluation import semantic_repeatability as replay
from code4scene.evaluation.requirement_graph.bundle import FrozenVerificationBundle
from code4scene.evaluation.requirement_graph.contracts import EntityNode, PredicateNode
from code4scene.evaluation.requirement_graph.existing_llm import LLMResponse, ToolCall
from code4scene.evaluation.requirement_graph.stage2_contracts import (
    Stage2Task, Stage2TaskArgument,
)
from code4scene.evaluation.requirement_graph.stage2_routing import visual_claim_payload


class Client:
    model = "test-judge"
    _strict_tool_calls = True

    def __init__(self, binary="MATCH", unknown="UNKNOWN", invalid=False):
        self.binary = binary
        self.unknown = unknown
        self.invalid = invalid
        self.requests = []

    def chat(self, messages, tools, **kwargs):
        self.requests.append((messages, tools, kwargs))
        if self.invalid:
            return LLMResponse(None, [ToolCall("bad", tools[0]["name"], {})])
        verdict = self.binary if "binary" in tools[0]["name"] else self.unknown
        args = {
            "verdict": verdict, "confidence": 0.0 if verdict == "UNKNOWN" else 0.9,
            "evidence_frame_ids": [] if verdict == "UNKNOWN" else ["s3f_000001"],
            "rationale": "controlled test response",
            "evidence_basis": {"UNKNOWN": "insufficient", "MATCH": "support",
                               "MISMATCH": "searched_absence"}[verdict],
            "grounding": {
                "subject_confirmed": verdict != "UNKNOWN",
                "participants_confirmed": False, "relation_scope_covered": False,
                "collection_complete": False, "instances_countable": False,
                "visible_instance_count": None,
            },
        }
        return LLMResponse(None, [ToolCall("test", tools[0]["name"], args)],
                           raw={"test_response": verdict})


@pytest.fixture
def candidate(tmp_path):
    bundle = FrozenVerificationBundle.from_dict(replay.read(
        Path(__file__).resolve().parent / "fixtures/synthetic-alley.bundle.json"))
    entity = next(n for n in bundle.graph.nodes
                  if isinstance(n, EntityNode) and n.name.casefold() == "rollup doors")
    node = next(n for n in bundle.graph.nodes
                if isinstance(n, PredicateNode) and n.name.casefold() == "rollup doors present")
    task = Stage2Task(task_id=f"stage2:{node.id}", node_id=node.id,
                      kind="object_existence", weight=bundle.graph.effective_weights()[node.id],
                      arguments=(Stage2TaskArgument(entity.id, "subject"),))
    payload = visual_claim_payload(bundle.graph, node.id)
    rgb = np.random.default_rng(9).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    path = tmp_path / "frame.png"
    Image.fromarray(rgb).save(path)
    spec = {"frame_id": "s3f_000001", "path": str(path),
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "png_sha256": replay.file_hash(path)}
    claim = {
        "node_id": node.id, "task": task.to_dict(), "payload": payload,
        "unknown_template": replay.compile_template(payload),
        "binary_template": replay.compile_template(payload, binary=True),
        "calls": [
            {"kind": "unknown_resolution", "historical_request_id": "s3r_000001",
             "frame_ids": ["s3f_000001"]},
            {"kind": "unknown_resolution", "historical_request_id": "s3r_000002",
             "frame_ids": ["s3f_000001"]},
            {"kind": "binary_resolution", "historical_request_id": "s3r_000003",
             "frame_ids": ["s3f_000001"]},
        ],
    }
    plan = {
        "model_id": "test-model", "case_id": "test-case",
        "bundle": {"graph": bundle.graph.to_dict()},
        "family_weights": {"identity_environment": 0.25, "content_quantity": 0.4,
                           "spatial_composition": 0.2, "attributes_materials": 0.15},
        "frames": {"s3f_000001": spec}, "claims": [claim],
        "frozen_score_rows": [
            {"node_id": node.id, "text": "Doors exist", "source_span": [0, 11],
             "predicate_type": "existence", "semantic_family": "content_quantity",
             "evaluation_status": "MATCH", "score": 1.0},
        ],
    }
    return plan


def frames(plan):
    return {k: replay.load_frame(v) for k, v in plan["frames"].items()}


def experiment(tmp_path, monkeypatch, plan):
    monkeypatch.setattr(replay, "code_hashes", lambda: {"fixed.py": "fixed"})
    prepared = tmp_path / "prepared"
    relative = "candidates/model/case.json"
    replay.write(prepared / relative, plan)
    meta = {
        "protocol": replay.PROTOCOL, "code_hashes": replay.code_hashes(),
        "judge": replay.judge_config(), "repeats": 3, "workers": 1,
        "model_ids": [plan["model_id"]], "case_ids": [plan["case_id"]],
        "scope": "test fixed inputs",
        "candidates": [{"model_id": plan["model_id"], "case_id": plan["case_id"],
                        "plan": relative, "sha256": replay.file_hash(prepared / relative)}],
    }
    meta["fingerprint"] = replay.json_hash(meta)
    replay.write(prepared / "protocol.json", meta)
    return prepared, tmp_path / "runs"


def test_every_saved_call_is_sent_even_after_two_matching_batches(candidate):
    client = Client(unknown="MATCH", binary="MISMATCH")
    outcome = replay.replay_claim(candidate, candidate["claims"][0], frames(candidate),
                                  lambda: client)
    assert outcome["status"] == "complete"
    assert len(client.requests) == 3
    assert outcome["resolution"]["final_verdict"] == "MATCH"
    assert outcome["binary_used"] is False
    # Even the unused binary request was made; none is selected using the new result.
    assert outcome["calls"][2]["decision"]["verdict"] == "MISMATCH"


def test_binary_resolves_unknown_and_only_replayed_score_rows_change(candidate):
    original = copy.deepcopy(candidate)
    candidate["frozen_score_rows"].append({
        "node_id": "fixed", "text": "A fixed identity", "source_span": [25, 41],
        "predicate_type": "scene_identity", "semantic_family": "identity_environment",
        "evaluation_status": "MATCH", "score": 1.0,
    })
    outcome = replay.replay_claim(candidate, candidate["claims"][0], frames(candidate),
                                  lambda: Client(binary="MISMATCH"))
    aggregate = replay.score_replayed(candidate, [outcome])
    assert outcome["binary_used"] is True
    assert outcome["resolution"]["final_verdict"] == "MISMATCH"
    assert aggregate["families"]["identity_environment"]["score"] == 1.0
    assert aggregate["families"]["content_quantity"]["score"] == 0.0
    assert candidate["frozen_score_rows"][0] == original["frozen_score_rows"][0]


def test_production_unknown_fallback_is_scored_and_counted(candidate):
    client = Client(invalid=True)
    outcome = replay.replay_claim(candidate, candidate["claims"][0], frames(candidate),
                                  lambda: client)
    assert outcome["status"] == "complete"
    assert outcome["resolution"]["final_verdict"] == "UNKNOWN"
    assert outcome["validation_fallback_count"] == 3
    assert len(client.requests) == 3  # No validation repair prompt or extra call.
    assert all(c["raw_records"][0].get("validation_error") for c in outcome["calls"])


def test_template_change_is_rejected_before_network(candidate):
    candidate["claims"][0]["unknown_template"]["system_text"] += " changed"
    client = Client()
    with pytest.raises(ValueError, match="template"):
        replay.replay_claim(candidate, candidate["claims"][0], frames(candidate), lambda: client)
    assert not client.requests


def test_changed_pixels_rejected_before_creating_client(tmp_path, monkeypatch, candidate):
    prepared, output = experiment(tmp_path, monkeypatch, candidate)
    image_path = Path(candidate["frames"]["s3f_000001"]["path"])
    Image.fromarray(np.zeros((24, 32, 3), dtype=np.uint8)).save(image_path)

    def forbidden():
        pytest.fail("client must not be constructed for changed evidence")

    with pytest.raises(ValueError, match="frame file changed"):
        replay.run(prepared, output, client_factory=forbidden)
    assert not (output / ".running.lock").exists()


def test_three_fresh_runs_resume_and_sample_sd(tmp_path, monkeypatch, candidate):
    prepared, output = experiment(tmp_path, monkeypatch, candidate)
    clients = []
    verdicts = iter(["MATCH", "MISMATCH", "MATCH"])

    def factory():
        client = Client(binary=next(verdicts))
        clients.append(client)
        return client

    result = replay.run(prepared, output, client_factory=factory)
    assert result["complete_scores"] == 3 and result["error_scores"] == 0
    assert len(clients) == 3 and sum(len(c.requests) for c in clients) == 9
    rows = list(__import__("csv").DictReader((output / "case_statistics.csv").open()))
    assert float(rows[0]["mean"]) == pytest.approx(2 / 3)
    assert float(rows[0]["sd"]) == pytest.approx(1 / np.sqrt(3))
    fingerprints = []
    for run in [1, 2, 3]:
        p = output / f"run-{run}/test-model/test-case/claims/0000.json"
        fingerprints.append([v["request_fingerprint"] for v in replay.read(p)["calls"]])
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]
    replay.run(prepared, output, resume=True,
               client_factory=lambda: pytest.fail("resume must not rerun completed claims"))
    with pytest.raises(ValueError, match="requires --resume"):
        replay.run(prepared, output, client_factory=factory)


def test_pending_runs_do_not_produce_sd_or_ranking(tmp_path, monkeypatch, candidate):
    prepared, output = experiment(tmp_path, monkeypatch, candidate)
    result = replay.run(prepared, output, run_ids=[1], client_factory=Client)
    assert result["complete_scores"] == 1
    assert all(v["tau_b"] is None for v in result["ranking_comparisons"])
    text = (output / "case_statistics.csv").read_text()
    assert "test-model,test-case,1,,\n" in text


def test_failed_case_has_no_published_score(tmp_path, monkeypatch, candidate):
    prepared, output = experiment(tmp_path, monkeypatch, candidate)
    class BrokenClient(Client):
        def chat(self, *args, **kwargs):
            raise ValueError("controlled transport failure")

    result = replay.run(prepared, output, run_ids=[1],
                        client_factory=BrokenClient)
    assert result["error_scores"] == 1
    saved = replay.read(output / "run-1/test-model/test-case/result.json")
    assert saved["score"] is None
    assert saved["diagnostic_aggregation"]["score"] is not None


def test_unknown_fallback_does_not_remove_case_from_statistics(tmp_path, monkeypatch, candidate):
    prepared, output = experiment(tmp_path, monkeypatch, candidate)
    result = replay.run(prepared, output,
                        client_factory=lambda: Client(invalid=True))
    assert result["complete_scores"] == 3 and result["error_scores"] == 0
    assert result["validation_fallback_count"] == 9
    saved = replay.read(output / "run-1/test-model/test-case/result.json")
    assert saved["status"] == "complete" and saved["score"] is not None
    assert saved["validation_fallback_count"] == 3


def test_changed_judge_code_plan_or_fingerprint_rejected(tmp_path, monkeypatch, candidate):
    prepared, output = experiment(tmp_path, monkeypatch, candidate)
    monkeypatch.setattr(replay, "code_hashes", lambda: {"fixed.py": "changed"})
    with pytest.raises(ValueError, match="code changed"):
        replay.run(prepared, output)
    monkeypatch.setattr(replay, "code_hashes", lambda: {"fixed.py": "fixed"})
    monkeypatch.setattr(replay.config, "SEED", 17)
    with pytest.raises(ValueError, match="configuration changed"):
        replay.run(prepared, output)


def test_tau_b_handles_reversal_ties_and_undefined():
    assert replay.tau_b([1, 2, 3], [3, 2, 1]) == -1
    assert replay.tau_b([1, 1, 2], [1, 1, 2]) == 1
    assert replay.tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(2 / np.sqrt(6))
    assert replay.tau_b([1, 1], [2, 2]) is None


def test_recovery_overrides_only_filled_nodes(tmp_path, candidate):
    graph = candidate["bundle"]

    def source(root, names):
        replay.write(root / "bundle.json", graph)
        claims, manifests, tasks = [], [], []
        for i, name in enumerate(names, 1):
            rid = f"s3r_{i:06d}"
            claims.append({"node_id": name, "batches": [
                {"request_id": rid, "frame_ids": ["s3f_000001"]}]})
            manifests.append({"request_id": rid})
            tasks.append({"node_id": name})
        replay.write(root / "stage3/stage3_batch_results.json", {"claims": claims})
        replay.write(root / "stage2/stage2_tasks.json", tasks)
        replay.write(root / "stage3/stage3_vlm_request_manifest.json", manifests)
        replay.write(root / "stage3/stage3_frame_index.json",
                     {"frames": [{"frame_id": "s3f_000001", "image_hash": str(root)}]})

    primary, recovered = tmp_path / "primary", tmp_path / "recovered"
    source(primary, ["kept", "filled"])
    source(recovered, ["kept", "filled"])
    note = tmp_path / "recovery.json"
    replay.write(note, {"fills": [{"node_id": "filled"}]})
    report = {
        "evidence": {"incremental_unknown_recovery": {"filled_node_ids": ["filled"]}},
        "artifacts": {"incremental_unknown_recovery": str(note),
                      "incremental_unknown_replacement_graph": str(recovered)},
    }
    selected, provenance = replay.schedule_sources(report, primary, graph)
    mapped = {v["claim"]["node_id"]: v for v in selected}
    assert mapped["kept"]["source_id"] == "primary"
    assert mapped["filled"]["source_id"] == "recovery"
    assert len(provenance) == 2


def test_namespaced_storage_keys_do_not_enter_judge_request(candidate):
    spec = candidate["frames"].pop("s3f_000001")
    candidate["frames"]["recovery:s3f_000001"] = spec
    for call in candidate["claims"][0]["calls"]:
        call["frame_keys"] = ["recovery:s3f_000001"]
    client = Client()
    result = replay.replay_claim(candidate, candidate["claims"][0], frames(candidate),
                                 lambda: client)
    assert result["status"] == "complete"
    for messages, _, _ in client.requests:
        captions = [b["text"] for b in messages[1].content if b["type"] == "text"]
        assert "frame_id=s3f_000001" in captions
        assert all("recovery:" not in text for text in captions)
