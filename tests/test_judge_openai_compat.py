"""The OpenAI-compatible judge backend.

No network: what is worth checking here is the request shape and the failure
attribution, both of which the harness cannot check for itself. The live
behaviour was verified separately against a real endpoint.
"""

from __future__ import annotations

import base64
import json

import pytest

from code4scene.evaluation.verifiers.vlm_as_judge import (
    JudgeError,
    JudgeRequest,
    backend_from_env,
    load_rubric,
    model_config,
    verdict_schema,
)
from code4scene.evaluation.verifiers.vlm_as_judge import openai_compat as oc
from code4scene.evaluation.requirement_graph.vlm_client import tool_client_from_env
from code4scene.resources import config_file

_RUBRIC = config_file("rubrics", "scene-quality.yaml")
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def request_(tmp_path):
    shots = []
    for name in ("aerial.png", "eye_level.png"):
        path = tmp_path / name
        path.write_bytes(_PNG)
        shots.append(path)
    return JudgeRequest(
        prompt="a small winter village",
        rubric=load_rubric(_RUBRIC),
        images=shots,
        metrics={"actors": 400},
    )


def test_images_travel_as_data_urls_with_the_instructions_last(request_):
    parts = oc.content_parts(request_)
    images = [p for p in parts if p["type"] == "image_url"]
    assert len(images) == 2, "a judge that sees one view judges one view"
    url = images[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == _PNG
    assert "Score each criterion" in parts[-1]["text"]


def test_the_schema_is_the_rubric_s_not_the_backend_s(request_):
    """Both backends ask for the same shape, so a rubric change reaches every
    provider at once."""
    schema = verdict_schema(request_)
    assert set(schema["properties"]) == {c.id for c in request_.rubric.criteria}
    assert schema["properties"]["task_fidelity"]["properties"]["score"]["enum"] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_a_fenced_verdict_is_accepted():
    """A model that wrapped a correct answer in ```json has answered; failing
    the run over punctuation would be the wrong call."""
    body = {"task_fidelity": {"score": 3, "rationale": "ok"}}
    assert oc.parse_verdict("```json\n" + json.dumps(body) + "\n```") == body
    assert oc.parse_verdict(json.dumps(body)) == body


def test_prose_is_an_error_not_a_guess():
    with pytest.raises(JudgeError, match="not JSON"):
        oc.parse_verdict("I would score this about a 3 out of 4.")


class FakeOpener:
    def __init__(self, payload):
        self.payload = payload
        self.sent = []

    def __call__(self, path, body):
        self.sent.append(body)
        return self.payload


def judge_with(payload):
    judge = oc.OpenAICompatJudge(model="m-1", api_key="k")
    judge._post = FakeOpener(payload)
    return judge


def test_a_verdict_comes_back_as_the_backend_contract(request_):
    verdict = {c.id: {"score": 2, "rationale": "because"} for c in request_.rubric.criteria}
    judge = judge_with(
        {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(verdict)}}
            ]
        }
    )
    assert judge(request_) == verdict
    sent = judge._post.sent[0]
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["model"] == "m-1"


def test_a_truncated_verdict_is_not_a_partial_verdict(request_):
    """Observed for real: a reasoning model spent its whole budget thinking."""
    judge = judge_with(
        {"choices": [{"finish_reason": "length", "message": {"content": '{"task_fid'}}]}
    )
    with pytest.raises(JudgeError, match="cut off at max_tokens"):
        judge(request_)


def test_a_filtered_request_is_an_error_not_a_zero(request_):
    judge = judge_with(
        {"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]}
    )
    with pytest.raises(JudgeError, match="declined to score"):
        judge(request_)


def test_structured_output_can_be_turned_off_for_models_that_reject_it(request_):
    verdict = {c.id: {"score": 1, "rationale": "r"} for c in request_.rubric.criteria}
    judge = oc.OpenAICompatJudge(model="m-1", api_key="k", structured=False)
    judge._post = FakeOpener(
        {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(verdict)}}
            ]
        }
    )
    judge(request_)
    assert "response_format" not in judge._post.sent[0]


def test_a_ceiling_reason_carries_no_yaml_line_break():
    """Reasons render straight into a verdict; a folded YAML block leaves a
    trailing newline that showed up in a real run's output."""
    rubric = load_rubric(_RUBRIC)
    _, reasons = rubric.ceiling_for(
        {"structural_collision_rate": 0.45, "floating_rate": 0.40, "oob_rate": 0.2}
    )
    assert reasons and all(r == r.strip() for r in reasons)


def test_every_vlm_path_uses_the_one_frozen_qwen_deployment(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    judge = backend_from_env()

    assert isinstance(judge, oc.OpenAICompatJudge)
    assert judge.base_url == model_config.BASE_URL
    assert judge.model == model_config.MODEL
    assert judge.max_tokens == model_config.MAX_TOKENS
    assert judge.timeout_s == model_config.TIMEOUT_S
    assert judge.temperature == model_config.TEMPERATURE
    assert judge.seed == model_config.SEED
    assert judge.enable_thinking is False
    assert judge.api_key == ""

    requirement_graph_client = tool_client_from_env()
    assert requirement_graph_client.model == model_config.MODEL
    assert requirement_graph_client.backend.base_url == model_config.BASE_URL
