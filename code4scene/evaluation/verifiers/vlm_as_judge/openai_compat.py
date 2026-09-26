"""A judge backend for any OpenAI-compatible chat endpoint.

Written against the wire format rather than a vendor: OpenRouter, a local
vLLM server, or anything else that serves ``/chat/completions`` works by
changing ``base_url``. That matters here because judging is the one part of
this benchmark that costs money per episode, and a lab already running an open
model should be able to point at it instead.

Code4Scene fixes one judge model (Qwen3.8-27B) and its decoding parameters
for every VLM path; the endpoint that serves it is supplied by the user. This
module owns the OpenAI-compatible wire protocol; the rubric schema remains in
``judge.py`` because it is evaluation policy rather than transport behavior.

Structured output is requested through ``response_format``. Support varies by
model, so the parser also tolerates a fenced code block: a model that returns
```json … ``` has answered correctly and rejecting it would fail a run over
punctuation.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ...vlm_concurrency import (
    request_with_adaptive_concurrency,
    retryable_http_error,
)
from .judge import CHANNEL_MAJOR_BLOCKS, JudgeError, JudgeRequest, verdict_schema
from .model_config import BASE_URL, require_base_url

#: What the editor's screenshot path produces.
_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

JSON_SCHEMA_RESPONSE_FORMAT = "json_schema_response_format"


def content_parts(request: JudgeRequest) -> list[dict[str, Any]]:
    """The images, then the instructions.

    Images first because the instructions refer to them; a model asked to
    score before it has seen anything reads the criteria without a subject.
    """
    parts: list[dict[str, Any]] = []
    previous_channel: str | None = None
    for index, item in enumerate(request.image_evidence(), start=1):
        if (
            request.evidence_layout == CHANNEL_MAJOR_BLOCKS
            and item.channel != previous_channel
            and request.evidence
        ):
            channel_number = request.rubric.channels.index(item.channel) + 1
            parts.append(
                {
                    "type": "text",
                    "text": (
                        f"BEGIN CHANNEL BLOCK {channel_number}/"
                        f"{len(request.rubric.channels)}: "
                        f"CHANNEL={item.channel.upper()}. The following images "
                        "belong only to this channel until the next block."
                    ),
                }
            )
            previous_channel = item.channel
        path = item.path
        media_type = _MEDIA_TYPES.get(path.suffix.lower())
        if media_type is None:
            raise JudgeError(
                f"{path.name}: unsupported image format for judging "
                f"(expected one of {', '.join(sorted(_MEDIA_TYPES))})"
            )
        try:
            data = base64.standard_b64encode(path.read_bytes()).decode()
        except OSError as e:
            raise JudgeError(f"cannot read viewpoint {path}: {e}") from e
        if request.evidence_layout == CHANNEL_MAJOR_BLOCKS and request.evidence:
            if request.comparison_mode == "gt_paired":
                label = (
                    f"SCENE={item.scene.upper()} / CHANNEL={item.channel.upper()} "
                    f"/ VIEW={item.view}: {path.name}"
                )
            else:
                label = f"CHANNEL={item.channel.upper()} / VIEW={item.view}: {path.name}"
        else:
            label = item.label if request.evidence else f"Viewpoint {index}: {path.name}"
        if item.description:
            label += f" — {item.description}"
        parts.append({"type": "text", "text": label})
        parts.append(
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
        )
    parts.append({"type": "text", "text": request.instructions()})
    return parts


def parse_verdict(text: str) -> dict[str, Any]:
    """The model's answer as a mapping.

    Accepts a fenced block as well as bare JSON: not every model honours
    ``response_format``, and one that wrapped a correct answer in ```json has
    answered the question. Anything else is an error rather than a guess — a
    verdict recovered by pattern-matching prose is not a verdict.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1]
        if candidate.lstrip().lower().startswith("json"):
            candidate = candidate.lstrip()[4:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise JudgeError(
            f"the judge's verdict was not JSON: {e}. First 200 characters: {text[:200]!r}"
        ) from e


@dataclass
class OpenAICompatJudge:
    """Scores viewpoints through an OpenAI-compatible chat endpoint."""

    model: str
    base_url: str = BASE_URL
    api_key: str = ""
    #: Generous because a reasoning model spends most of its budget thinking
    #: before it writes the verdict: qwen3.6-35b-a3b exhausted 4000 tokens on a
    #: four-criterion rubric and returned nothing usable.
    max_tokens: int = 16000
    timeout_s: float = 180.0
    #: Judging is measurement, not creative sampling.  Make repeated runs over
    #: identical evidence as stable as the serving stack permits.
    temperature: float = 0.0
    seed: int | None = 0
    #: vLLM/Qwen can spend the whole output budget on hidden reasoning.  Leave
    #: False for the frozen Qwen deployment so structured verdicts do not lose
    #: their output budget to hidden reasoning.
    enable_thinking: bool | None = None
    #: Sent verbatim as headers. OpenRouter asks callers to identify
    #: themselves; other endpoints ignore these.
    headers: dict[str, str] = field(default_factory=dict)
    #: Set False for a model that rejects response_format. The schema is then
    #: only described in the prompt, so the verdict is checked, not trusted —
    #: which the harness does anyway.
    structured: bool = True
    #: The canonical endpoint mechanism used to enforce the verdict schema.
    structured_output_method: str = JSON_SCHEMA_RESPONSE_FORMAT

    def __call__(self, request: JudgeRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content_parts(request)}],
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking,
            }
        if self.structured and self.structured_output_method == JSON_SCHEMA_RESPONSE_FORMAT:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "scene_verdict",
                    "strict": True,
                    "schema": verdict_schema(request),
                },
            }
        elif self.structured:
            raise JudgeError(
                f"unknown structured output method {self.structured_output_method!r}"
            )
        payload = self._post("/chat/completions", body)

        choices = payload.get("choices") or []
        if not choices:
            raise JudgeError(
                f"the endpoint returned no choices: {json.dumps(payload)[:300]}"
            )
        choice = choices[0]
        reason = choice.get("finish_reason")
        if reason == "length":
            raise JudgeError(
                f"the verdict was cut off at max_tokens={self.max_tokens}; a "
                f"truncated verdict is not a partial verdict"
            )
        if reason == "content_filter":
            # Not a zero: a filtered request is not a badly-built scene, and
            # recording them alike would rank an episode that was never judged.
            raise JudgeError("the judge model declined to score this scene")

        text = (choice.get("message") or {}).get("content")
        if not text:
            raise JudgeError(f"the judge returned no text (finish_reason={reason})")
        return parse_verdict(text)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = require_base_url(self.base_url)
        url = f"{endpoint}{path}"
        payload = json.dumps(body).encode()

        def post() -> dict[str, Any]:
            request = urllib.request.Request(
                url,
                data=payload,
                headers=headers,
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode())

        try:
            return request_with_adaptive_concurrency(
                endpoint,
                post,
                retryable=retryable_http_error,
            )
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            # The body carries the reason (unknown model, no credit, image
            # unsupported); the status alone sends people to the wrong place.
            raise JudgeError(f"judge endpoint returned {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise JudgeError(
                f"cannot reach the judge endpoint at {self.base_url}: {e}"
            ) from e


__all__ = [
    "JSON_SCHEMA_RESPONSE_FORMAT",
    "OpenAICompatJudge",
    "content_parts",
    "parse_verdict",
]
