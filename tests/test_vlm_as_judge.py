"""Rubrics, judged scoring, and the constraints around the model call."""

import json

import pytest

from code4scene.evaluation.verifiers import vlm_as_judge
from code4scene.evaluation.verifiers.vlm_as_judge import Judge, JudgeError, RubricError, load_rubric
from code4scene.evaluation.verifiers.vlm_as_judge.judge import INJECTION_NOTICE
from code4scene.resources import config_file

IDS = {"task_bundle_id": "bundle-1", "episode_id": "ep-1"}

RUBRIC = {
    "id": "test-quality", "scale_max": 4,
    "criteria": [
        {"id": "fidelity", "question": "Does it match the request?", "weight": 3},
        {"id": "finish", "question": "Does it look finished?", "weight": 1,
         "guidance": "Ground covered, edges resolved"},
    ],
    "ceilings": [
        {"metric": "structural_collision_rate", "above": 0.3, "max_score": 0.5,
         "reason": "over 30% of structures intersect"},
        {"metric": "actors", "below": 20, "max_score": 0.4,
         "reason": "too little was built to judge"},
    ],
}


def write_rubric(tmp_path, body=None, name="r.yaml"):
    path = tmp_path / name
    path.write_text(json.dumps(body or RUBRIC))
    return path


def images(tmp_path, count=3):
    out = []
    for i in range(count):
        p = tmp_path / f"view_{i}.png"
        p.write_bytes(b"\x89PNG")
        out.append(p)
    return out


def backend_scoring(**scores):
    def backend(request):
        return {cid: {"score": value, "rationale": f"because {cid}"}
                for cid, value in scores.items()}
    return backend


# ── rubric ────────────────────────────────────────────────────────────────

def test_rubric_is_validated_and_named(tmp_path):
    rubric = load_rubric(write_rubric(tmp_path))
    assert rubric.name == "test-quality"
    assert rubric.total_weight == 4

    edited = dict(RUBRIC, instructions="judge kindly")
    assert load_rubric(
        write_rubric(tmp_path, edited, "r2.yaml")
    ).instructions == "judge kindly"


@pytest.mark.parametrize("body,message", [
    ({"id": "x", "version": 1}, "criteria"),
    ({"id": "x", "criteria": []}, "no criteria"),
    ({"id": "x", "criteria": [{"id": "a"}]}, "question"),
    ({"id": "x", "criteria": [{"id": "a", "question": "q"},
                                            {"id": "a", "question": "q"}]},
     "declared twice"),
    ({"id": "x", "criteria": [{"id": "a", "question": "q", "weight": 0}]},
     "positive weight"),
    ({"id": "x", "criteria": [{"id": "a", "question": "q"}],
      "ceilings": [{"metric": "m"}]}, "above"),
    ({"id": "x", "criteria": [{"id": "a", "question": "q"}],
      "ceilings": [{"metric": "m", "above": 1, "max_score": 2}]}, "within 0..1"),
])
def test_bad_rubrics_fail_loudly(tmp_path, body, message):
    with pytest.raises(RubricError, match=message):
        load_rubric(write_rubric(tmp_path, body))


def test_the_strictest_applicable_ceiling_wins(tmp_path):
    rubric = load_rubric(write_rubric(tmp_path))
    ceiling, reasons = rubric.ceiling_for({"structural_collision_rate": 0.5,
                                           "actors": 5})
    assert ceiling == 0.4 and len(reasons) == 2


def test_no_ceiling_on_a_healthy_scene(tmp_path):
    rubric = load_rubric(write_rubric(tmp_path))
    assert rubric.ceiling_for({"structural_collision_rate": 0.0,
                               "actors": 400}) == (1.0, [])


def test_an_unmeasured_scene_is_capped_not_freed(tmp_path):
    """A missing metric used to skip its ceiling, so a measurement that failed
    removed every cap the rubric had and the scene nobody could measure was
    the one the judge could score highest. The ceilings exist because a scene
    measuring badly must not be rescued by photographing well; a scene
    measuring not at all is not evidence of quality either."""
    rubric = load_rubric(write_rubric(tmp_path))

    capped, reasons = rubric.ceiling_for({})
    assert capped < 1.0 and reasons


# ── judge ─────────────────────────────────────────────────────────────────

def test_weighted_score_and_provenance(tmp_path):
    rubric = load_rubric(write_rubric(tmp_path))
    judge = Judge(rubric=rubric, backend=backend_scoring(fidelity=4, finish=2),
                  model="judge-model-1")
    verdict = judge.score("a winter village", images(tmp_path),
                          {"structural_collision_rate": 0.0, "actors": 400})

    # (4*3 + 2*1) / (4 weights * 4 scale) = 14/16
    assert verdict.judged == 0.875 and verdict.final == 0.875
    assert not verdict.capped
    assert verdict.rubric == "test-quality"
    assert verdict.model == "judge-model-1"
    assert verdict.rationales["fidelity"] == "because fidelity"
    assert verdict.images == ["view_0.png", "view_1.png", "view_2.png"]


@pytest.mark.parametrize("trailing_comma", [False, True])
def test_double_encoded_structured_scores_are_recovered(tmp_path, trailing_comma):
    """Qwen can double-encode a tool argument value and include its separator."""
    embedded = json.dumps({
        "score": 4,
        "rationale": "The paired views have the same layout.",
        "rationale_extra": "",
    })
    if trailing_comma:
        embedded += ","
    raw = {
        "fidelity": embedded,
        "finish": {"score": 2, "rationale": "Some details differ."},
    }
    judge = Judge(rubric=load_rubric(write_rubric(tmp_path)),
                  backend=lambda request: raw)

    verdict = judge.score("p", images(tmp_path), {"actors": 400})

    assert verdict.scores == {"fidelity": 4, "finish": 2}
    assert verdict.rationales["fidelity"] == (
        "The paired views have the same layout."
    )
    assert verdict.raw_response is raw


@pytest.mark.parametrize("escaped_newlines", [False, True])
@pytest.mark.parametrize(
    "marker",
    ["</parameter=finish>", "<parameter finish>"],
)
def test_concatenated_qwen_parameter_chain_is_recovered(
    tmp_path, escaped_newlines, marker
):
    """A complete tool verdict serialized into its first field is recoverable."""
    separator = "\\n" if escaped_newlines else "\n"
    fidelity = {
        "score": 2,
        "rationale": "The candidate is missing one prominent bench.",
    }
    finish = {
        "score": 3,
        "rationale": "The remaining geometry and materials align.",
    }
    chain = (
        json.dumps(fidelity)
        + ","
        + separator
        + marker
        + separator
        + json.dumps(finish)
    )
    raw = {"fidelity": chain}
    judge = Judge(
        rubric=load_rubric(write_rubric(tmp_path)),
        backend=lambda request: raw,
    )

    verdict = judge.score("p", images(tmp_path), {"actors": 400})

    assert verdict.scores == {"fidelity": 2, "finish": 3}
    assert verdict.rationales == {
        "fidelity": fidelity["rationale"],
        "finish": finish["rationale"],
    }
    assert verdict.raw_response is raw
    assert verdict.structured_output_recovery == {
        "applied": True,
        "method": "qwen-concatenated-parameter-chain",
        "source_criterion": "fidelity",
        "recovered_criteria": ["fidelity", "finish"],
    }


@pytest.mark.parametrize(
    "chain",
    [
        '{"score": 2, "rationale": "ok"},\n'
        '</parameter=unknown>{"score": 3, "rationale": "ok"}',
        '{"score": 2, "rationale": "ok"},\n'
        '</parameter=finish>{"score": 3, "rationale": "ok"} trailing prose',
        '{"score": 2, "rationale": "ok"},\n'
        '</parameter=finish>{"score": 3}',
    ],
)
def test_ambiguous_or_incomplete_parameter_chain_is_not_recovered(tmp_path, chain):
    judge = Judge(
        rubric=load_rubric(write_rubric(tmp_path)),
        backend=lambda request: {"fidelity": chain},
    )

    with pytest.raises(JudgeError):
        judge.score("p", images(tmp_path), {"actors": 400})


def test_metrics_cap_a_well_photographed_wreck(tmp_path):
    """Half the structures intersecting is not rescued by a good angle."""
    judge = Judge(rubric=load_rubric(write_rubric(tmp_path)),
                  backend=backend_scoring(fidelity=4, finish=4))
    verdict = judge.score("a winter village", images(tmp_path),
                          {"structural_collision_rate": 0.5, "actors": 400})

    assert verdict.judged == 1.0
    assert verdict.final == 0.5 and verdict.capped
    assert "intersect" in verdict.ceiling_reasons[0]


def test_one_viewpoint_is_refused(tmp_path):
    judge = Judge(rubric=load_rubric(write_rubric(tmp_path)),
                  backend=backend_scoring(fidelity=4, finish=4))
    with pytest.raises(JudgeError, match="at least 2"):
        judge.score("p", images(tmp_path, count=1), {"actors": 400})


def test_missing_renders_are_refused(tmp_path):
    judge = Judge(rubric=load_rubric(write_rubric(tmp_path)),
                  backend=backend_scoring(fidelity=4, finish=4))
    with pytest.raises(JudgeError, match="not found"):
        judge.score("p", [tmp_path / "absent.png", tmp_path / "gone.png"],
                    {"actors": 400})


def test_the_prompt_disarms_in_scene_instructions(tmp_path):
    """An agent can build a sign reading "score 10"; a judge that reads it as
    an instruction is a judge that can be talked into a score."""
    captured = {}

    def backend(request):
        captured["text"] = request.instructions()
        return {"fidelity": 3, "finish": 3}

    judge = Judge(rubric=load_rubric(write_rubric(tmp_path)), backend=backend)
    judge.score("a winter village", images(tmp_path), {"actors": 400})

    assert INJECTION_NOTICE in captured["text"]
    assert "scenery" in captured["text"]
    # The rubric's own questions and scale reach the backend.
    assert "Does it match the request?" in captured["text"]
    assert "0 to 4" in captured["text"]
    assert "a winter village" in captured["text"]


def test_prompt_repeats_the_machine_readable_verdict_contract(tmp_path):
    captured = {}

    def backend(request):
        captured["text"] = request.instructions()
        return backend_scoring(fidelity=4, finish=3)(request)

    judge = Judge(rubric=load_rubric(write_rubric(tmp_path)), backend=backend)
    judge.score("a winter village", images(tmp_path), {"actors": 400})

    prompt = captured["text"]
    assert "Structured verdict contract (mandatory)" in prompt
    assert "fidelity, finish" in prompt
    assert '"score": <JSON integer from 0 to 4>' in prompt
    assert "exactly once and no others" in prompt
    assert "XML/parameter tags" in prompt


@pytest.mark.parametrize("raw,message", [
    ({"fidelity": 4}, "did not score 'finish'"),
    ({"fidelity": 4, "finish": 9}, "outside the rubric's"),
    ({"fidelity": 4, "finish": -1}, "outside the rubric's"),
    ({"fidelity": 4, "finish": "good"}, "not a number"),
    ({"fidelity": 4, "finish": '{"score": 9},'}, "outside the rubric's"),
    ({"fidelity": 4, "finish": '{"rationale": "missing"},'}, "not a number"),
    ({"fidelity": 4, "finish": 2, "vibes": 4}, "not in the rubric"),
    ("nope", "expected a mapping"),
])
def test_a_partial_or_wrong_verdict_is_not_a_verdict(tmp_path, raw, message):
    """Clamping or filling in would make an ignored scale indistinguishable
    from a real score."""
    judge = Judge(rubric=load_rubric(write_rubric(tmp_path)),
                  backend=lambda request: raw)
    with pytest.raises(JudgeError, match=message):
        judge.score("p", images(tmp_path), {"actors": 400})


def test_a_failing_backend_is_a_judge_failure(tmp_path):
    def broken(request):
        raise TimeoutError("model did not answer")

    judge = Judge(rubric=load_rubric(write_rubric(tmp_path)), backend=broken)
    with pytest.raises(JudgeError, match="TimeoutError"):
        judge.score("p", images(tmp_path), {"actors": 400})


def test_shipped_rubric_is_valid_and_caps_a_broken_scene():
    shipped = config_file("rubrics", "scene-quality.yaml")
    rubric = load_rubric(shipped)
    assert rubric.name == "scene-quality"
    assert {c.id for c in rubric.criteria} == {"task_fidelity", "coherence",
                                              "composition", "finish"}
    assert rubric.ceiling_for({"structural_collision_rate": 0.55,
                               "actors": 900})[0] == 0.5
    assert rubric.ceiling_for({"actors": 5})[0] == 0.4
    assert rubric.ceiling_for({"structural_collision_rate": 0.0,
                               "floating_rate": 0.0, "actors": 500}) == (1.0, [])


def test_a_missing_metric_caps_the_judge_rather_than_freeing_it():
    """This returned False when the metric was absent, so a measurement that
    failed removed every cap the rubric had — the scene nobody could measure
    was the one the judge was free to score highest. Exactly backwards."""
    from code4scene.evaluation.verifiers.vlm_as_judge.rubric import Ceiling

    ceiling = Ceiling(metric="structural_collision_rate", above=0.1,
                      max_score=0.4, reason="buildings intersect")

    assert ceiling.applies_to({"structural_collision_rate": 0.5}) is True
    assert ceiling.applies_to({"structural_collision_rate": 0.0}) is False
    assert ceiling.applies_to({}) is True, "unmeasured is not evidence of quality"
    assert ceiling.applies_to({"structural_collision_rate": None}) is True

def test_judge_report_is_a_real_verifier_report(infra):
    verdict = {"final": 0.5, "judged": 1.0, "ceiling": 0.5, "scores": {"fidelity": 4},
               "rubric": "scene-quality", "model": "j-1",
               "ceiling_reasons": ["over 30% of structures intersect"]}
    report = infra.VerifierReport.from_json_dict(vlm_as_judge.report(verdict, **IDS))
    assert report.score == 0.5
    assert report.evidence["rubric"] == "scene-quality"


def test_judge_report_accepts_the_verdict_the_judge_returns(infra):
    """The judge returns a `Verdict`, not a mapping.

    Only the mapping was ever exercised, so the real path — judge a scene,
    lower the verdict — raised `AttributeError: 'Verdict' object has no
    attribute 'get'` the first time a live model answered. A verifier the
    tests reach only through a stand-in for its own output is a verifier
    nothing tests.
    """
    from code4scene.evaluation.verifiers.vlm_as_judge.judge import Verdict

    verdict = Verdict(rubric="scene-quality", model="j-1",
                      scores={"fidelity": 4}, judged=1.0, ceiling=0.5, final=0.5,
                      ceiling_reasons=["over 30% of structures intersect"])
    report = infra.VerifierReport.from_json_dict(vlm_as_judge.report(verdict, **IDS))

    assert report.score == 0.5
    assert report.evidence["rubric"] == "scene-quality"
    assert report.metrics["criterion.fidelity"] == 4


def test_tiny_visual_score_is_still_measured_without_a_threshold(infra):
    verdict = {
        "final": 0.02,
        "judged": 0.02,
        "ceiling": 1.0,
        "scores": {"fidelity": 0},
        "rubric": "visual-prompt",
        "model": "j-1",
    }

    report = infra.VerifierReport.from_json_dict(
        vlm_as_judge.report(verdict, **IDS)
    )

    assert report.status == "measured"
    assert report.score == 0.02
    assert report.failure_reason is None
