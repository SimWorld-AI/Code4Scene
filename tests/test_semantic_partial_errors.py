"""A failed judgement costs only the requirements it failed."""

from __future__ import annotations

from types import SimpleNamespace

from code4scene.evaluation.verifiers import semantic_requirements


def _row(node_id, text, span, status):
    decided = status in ("MATCH", "MISMATCH")
    return {
        "requirement_id": node_id,
        "node_id": node_id,
        "verdict": status.lower() if decided else "unknown",
        "evaluation_status": status,
        "check": {"score": 1.0 if status == "MATCH" else 0.0, "evidence": []} if decided else None,
        "text": text,
        "source_span": list(span),
        "population_scope": "candidate_all",
        "entity_scopes": {},
        "weight": 1.0,
        "resolved_by": "stage3" if decided else None,
        "unknown_reason": None if decided else "evaluation_error",
        "rationale": "judge timeout" if status == "ERROR" else "visible",
    }


def _report(monkeypatch, rows, visual_error):
    nodes = [{"id": r["node_id"], "predicate_type": "existence"} for r in rows]
    monkeypatch.setattr(semantic_requirements, "run_pipeline", lambda context: {
        "requirements": rows,
        "non_supported_requirements": [],
        "waived_requirements": [],
        "delegated_requirements": [],
        "scene_evidence": SimpleNamespace(evidence=lambda: {}, probes_used=lambda: ()),
        "artifacts": {},
        "visual_error": visual_error,
        "identity_grounding": None,
        "bundle": SimpleNamespace(bundle_id="b", task_mode=SimpleNamespace(value="generation"),
                                  graph={"nodes": nodes, "edges": []}),
    })
    return semantic_requirements.verify(SimpleNamespace(
        ids={"task_bundle_id": "task", "episode_id": "episode"}, renders_for=lambda protocol: None))


def test_one_failed_judgement_zeroes_only_that_requirement(monkeypatch):
    rows = [_row("bench", "a bench", (0, 7), "MATCH"),
            _row("lamp", "a lamp", (9, 15), "MATCH"),
            _row("fountain", "a fountain", (17, 27), "ERROR")]
    report = _report(monkeypatch, rows, "fountain: RuntimeError: judge timeout")
    assert report["status"] == "measured"
    assert report["score"] is not None and 0.0 < report["score"] < 1.0
    assert report["score"] == report["metrics"]["semantic_score"]
    assert report["metrics"]["partial_score"] is True
    assert report["evidence"]["visual_error"] == "fountain: RuntimeError: judge timeout"


def test_nothing_decided_is_still_an_error(monkeypatch):
    rows = [_row("bench", "a bench", (0, 7), "ERROR"), _row("lamp", "a lamp", (9, 15), "ERROR")]
    report = _report(monkeypatch, rows, "RuntimeError: judge unreachable")
    assert report["status"] == "error"
    assert report["score"] is None
    assert "judge unreachable" in report["failure_reason"]
