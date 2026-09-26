"""``gt_caption_similarity`` — semantic distance between two scene captions.

Candidate and canonical are photographed with the strict four-view GT paired
ring.  Each scene is captioned in a separate, stateless Qwen request using the
same prompt and view order.  A Qwen text-embedding model then embeds both
captions in one request, and this verifier reports their cosine similarity.

The scenes are deliberately never shown together to the caption model.  A
pairwise prompt could make the model optimise its prose for agreement instead
of independently describing what is visible.  The only comparison is the
deterministic cosine over the two returned vectors.

The clipped cosine is published directly as a continuous score; it is never
thresholded into PASS/FAIL.  It complements, rather than replaces, geometric
and pixel/VLM paired verifiers: captions intentionally discard exact
transforms, counts, materials and small objects.

Serves component ``semantic.gt_caption_similarity``.
"""

from __future__ import annotations

import base64
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

from .. import contracts, paired_views, render, scene_graph_capture, vlm_model_config
from ..context import Context, read_label
from ..vlm_concurrency import (
    parallel_map as parallel_vlm_map,
    request_with_adaptive_concurrency,
    retryable_http_error,
    runtime_snapshot as vlm_runtime_snapshot,
)

METRIC_PROTOCOL = "gt-caption-cosine"
EVALUATOR_ID = "caption.independent_caption_embedding_distance"
CAPTION_PROTOCOL = "independent-four-overview-scene-caption"
EMBEDDING_PROTOCOL = "paired-caption-openai-embedding"

# Endpoints are user-supplied (CODE4SCENE_VLM_BASE_URL and
# CODE4SCENE_EMBED_BASE_URL); nothing is shipped. This metric is report-only.
CAPTION_BASE_URL = vlm_model_config.BASE_URL
CAPTION_MODEL = vlm_model_config.MODEL
EMBEDDING_BASE_URL = vlm_model_config.embed_base_url()
EMBEDDING_MODEL = vlm_model_config.embed_model()

VIEW_COUNT = 4
CAPTION_MAX_TOKENS = 1200
CAPTION_TIMEOUT_S = 600.0
CAPTION_MAX_ATTEMPTS = 3
EMBEDDING_TIMEOUT_S = 120.0
_CAPTION_TOOL = "record_scene_caption"
_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def evidence_requests(task: Any, spec: dict[str, Any]):
    # Imported lazily because the canonical wrapper imports this atomic module.
    from .caption_similarity import evidence_requests as canonical_requests

    return canonical_requests(task, spec)


class CaptionSimilarityError(Exception):
    """The paired captions or embeddings could not be measured."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        self.diagnostics = tuple(dict(value) for value in diagnostics)


@dataclass(frozen=True)
class CaptionResult:
    caption: str
    request_id: str | None
    usage: Mapping[str, Any]
    attempt_count: int = 1
    structured_mode: str = "tool_call"


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], tuple[float, ...]]
    request_id: str | None
    usage: Mapping[str, Any]


def _post_json(
    base_url: str,
    path: str,
    body: Mapping[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    api_key = vlm_model_config.api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    endpoint = str(base_url or "").strip().rstrip("/")
    if not endpoint:
        raise CaptionSimilarityError(
            "caption similarity needs CODE4SCENE_VLM_BASE_URL and "
            "CODE4SCENE_EMBED_BASE_URL (report-only metric)"
        )
    url = f"{endpoint}{path}"
    encoded = json.dumps(body).encode("utf-8")

    def post() -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=encoded,
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        payload = request_with_adaptive_concurrency(
            endpoint,
            post,
            retryable=retryable_http_error,
        )
    except urllib.error.HTTPError as exception:
        detail = exception.read().decode("utf-8", errors="replace")[:500]
        raise CaptionSimilarityError(
            f"endpoint {base_url} returned {exception.code}: {detail}"
        ) from exception
    except (urllib.error.URLError, TimeoutError, OSError) as exception:
        raise CaptionSimilarityError(
            f"cannot reach endpoint {base_url}: {exception}"
        ) from exception
    except (UnicodeDecodeError, json.JSONDecodeError) as exception:
        raise CaptionSimilarityError(
            f"endpoint {base_url} returned invalid JSON: {exception}"
        ) from exception
    if not isinstance(payload, dict):
        raise CaptionSimilarityError(f"endpoint {base_url} returned a non-object")
    return payload


def _image_part(path: Path) -> dict[str, Any]:
    media_type = _MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise CaptionSimilarityError(
            f"{path.name}: unsupported image type; expected "
            f"{', '.join(sorted(_MEDIA_TYPES))}"
        )
    try:
        encoded = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    except OSError as exception:
        raise CaptionSimilarityError(f"cannot read overview {path}: {exception}") from exception
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }


def _response_diagnostic(payload: Mapping[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str) and len(content) > 4000:
        content = content[:4000] + "...[truncated]"
    return {
        "response_id": payload.get("id"),
        "finish_reason": (
            choice.get("finish_reason") if isinstance(choice, Mapping) else None
        ),
        "content": content,
        "tool_calls": (
            message.get("tool_calls") if isinstance(message, Mapping) else None
        ),
        "reasoning_character_count": (
            len(message.get("reasoning") or "")
            if isinstance(message, Mapping)
            and isinstance(message.get("reasoning"), str)
            else 0
        ),
        "usage": (
            dict(payload.get("usage"))
            if isinstance(payload.get("usage"), Mapping)
            else {}
        ),
    }


def _caption_from_response(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise CaptionSimilarityError("caption endpoint returned no unique choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise CaptionSimilarityError("caption endpoint returned a malformed choice")
    if choice.get("finish_reason") == "length":
        raise CaptionSimilarityError("caption was truncated at the output token limit")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise CaptionSimilarityError("caption endpoint returned a malformed message")

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        try:
            structured = json.loads(content)
        except json.JSONDecodeError:
            structured = None
        if isinstance(structured, Mapping) and set(structured) == {"caption"}:
            caption = structured.get("caption")
            if isinstance(caption, str) and caption.strip():
                return caption.strip()

    calls = message.get("tool_calls")
    if isinstance(calls, list) and len(calls) == 1:
        function = (
            calls[0].get("function") if isinstance(calls[0], Mapping) else None
        )
        if not isinstance(function, Mapping) or function.get("name") != _CAPTION_TOOL:
            raise CaptionSimilarityError("caption endpoint called an unexpected tool")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exception:
                raise CaptionSimilarityError(
                    f"caption tool arguments were invalid JSON: {exception}"
                ) from exception
        if not isinstance(arguments, Mapping) or set(arguments) != {"caption"}:
            raise CaptionSimilarityError(
                "caption tool must return only the caption field"
            )
        caption = arguments.get("caption")
        if isinstance(caption, str) and caption.strip():
            return caption.strip()
        raise CaptionSimilarityError("caption endpoint returned an empty caption")

    call_count = len(calls) if isinstance(calls, list) else 0
    raise CaptionSimilarityError(
        "caption endpoint returned neither schema-conformant content nor exactly "
        f"one structured tool call (tool_call_count={call_count})"
    )


class QwenCaptionClient:
    """One independent four-view scene-caption request."""

    def caption(self, images: Sequence[Path]) -> CaptionResult:
        if len(images) != VIEW_COUNT:
            raise CaptionSimilarityError(
                f"caption protocol needs exactly {VIEW_COUNT} views, got {len(images)}"
            )
        content: list[dict[str, Any]] = []
        for index, image in enumerate(images):
            content.append(
                {
                    "type": "text",
                    "text": f"OVERVIEW VIEW {index + 1}/{VIEW_COUNT}",
                }
            )
            content.append(_image_part(Path(image)))
        content.append(
            {
                "type": "text",
                "text": (
                    "Independently describe the single 3D environment visible across "
                    "these four overview views. Do not compare it with another scene, "
                    "do not mention image quality, and do not infer hidden objects. "
                    "Write one factual English caption covering the setting, major "
                    "structures and landmarks, terrain or ground, spatial organization, "
                    "recurring visible props, and overall visual atmosphere. Merge "
                    "duplicate observations across views. Prefer concrete visible facts "
                    "over evaluative language. Target 100-180 words."
                ),
            }
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"caption": {"type": "string"}},
            "required": ["caption"],
        }
        common = {
            "model": CAPTION_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": CAPTION_MAX_TOKENS,
            "temperature": 0.0,
            "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        attempts: list[dict[str, Any]] = []
        last_error = "unknown caption response error"
        for attempt in range(1, CAPTION_MAX_ATTEMPTS + 1):
            if attempt == 1:
                structured_mode = "tool_call"
                body = {
                    **common,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": _CAPTION_TOOL,
                                "description": (
                                    "Record the independent scene caption."
                                ),
                                "parameters": schema,
                            },
                        }
                    ],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": _CAPTION_TOOL},
                    },
                }
            else:
                structured_mode = "json_schema"
                body = {
                    **common,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "scene_caption",
                            "schema": schema,
                        },
                    },
                }
            try:
                payload = _post_json(
                    CAPTION_BASE_URL,
                    "/chat/completions",
                    body,
                    timeout_s=CAPTION_TIMEOUT_S,
                )
            except CaptionSimilarityError as exception:
                last_error = str(exception)
                attempts.append(
                    {
                        "attempt": attempt,
                        "structured_mode": structured_mode,
                        "error": last_error,
                    }
                )
                continue
            try:
                caption = _caption_from_response(payload)
            except CaptionSimilarityError as exception:
                last_error = str(exception)
                attempts.append(
                    {
                        "attempt": attempt,
                        "structured_mode": structured_mode,
                        "error": last_error,
                        "response": _response_diagnostic(payload),
                    }
                )
                continue
            return CaptionResult(
                caption=caption,
                request_id=str(payload["id"]) if payload.get("id") else None,
                usage=(
                    payload.get("usage")
                    if isinstance(payload.get("usage"), Mapping)
                    else {}
                ),
                attempt_count=attempt,
                structured_mode=structured_mode,
            )
        raise CaptionSimilarityError(
            f"caption endpoint failed after {CAPTION_MAX_ATTEMPTS} attempts: "
            f"{last_error}",
            diagnostics=attempts,
        )


class QwenEmbeddingClient:
    """Embed both captions together so they share one model invocation."""

    def embed_pair(self, captions: Sequence[str]) -> EmbeddingResult:
        if len(captions) != 2 or any(not isinstance(value, str) or not value for value in captions):
            raise CaptionSimilarityError("embedding input must contain two non-empty captions")
        payload = _post_json(
            EMBEDDING_BASE_URL,
            "/embeddings",
            {"model": EMBEDDING_MODEL, "input": list(captions)},
            timeout_s=EMBEDDING_TIMEOUT_S,
        )
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != 2:
            raise CaptionSimilarityError("embedding endpoint did not return two vectors")
        ordered: list[tuple[float, ...] | None] = [None, None]
        for item in data:
            if not isinstance(item, Mapping):
                raise CaptionSimilarityError("embedding endpoint returned a malformed row")
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index not in (0, 1):
                raise CaptionSimilarityError("embedding row has an invalid index")
            raw = item.get("embedding")
            if not isinstance(raw, list) or not raw:
                raise CaptionSimilarityError("embedding row contains no vector")
            try:
                vector = tuple(float(value) for value in raw)
            except (TypeError, ValueError, OverflowError) as exception:
                raise CaptionSimilarityError("embedding vector must be numeric") from exception
            if any(not math.isfinite(value) for value in vector):
                raise CaptionSimilarityError("embedding vector contains non-finite values")
            ordered[index] = vector
        if ordered[0] is None or ordered[1] is None:
            raise CaptionSimilarityError("embedding response has missing or duplicate indices")
        left, right = ordered
        if len(left) != len(right):
            raise CaptionSimilarityError("embedding vectors have inconsistent dimensions")
        if not any(left) or not any(right):
            raise CaptionSimilarityError("embedding endpoint returned a zero vector")
        return EmbeddingResult(
            vectors=(left, right),
            request_id=str(payload["id"]) if payload.get("id") else None,
            usage=payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {},
        )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise CaptionSimilarityError("cosine needs two non-empty equal-length vectors")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise CaptionSimilarityError("cosine is undefined for a zero vector")
    value = dot / (left_norm * right_norm)
    if not math.isfinite(value):
        raise CaptionSimilarityError("cosine similarity is not finite")
    return max(-1.0, min(1.0, value))


def _write_artifact(
    context: Context,
    payload: Mapping[str, Any],
    report_id: str,
) -> str | None:
    if context.out_dir is None:
        return None
    root = Path(context.out_dir) / report_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / "comparison.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return str(path)


def _caption_client() -> QwenCaptionClient:
    return QwenCaptionClient()


def _embedding_client() -> QwenEmbeddingClient:
    return QwenEmbeddingClient()


def _caption_scene(
    client: QwenCaptionClient,
    images: Sequence[Path],
    *,
    scene: str,
) -> CaptionResult:
    try:
        return client.caption(images)
    except CaptionSimilarityError as exception:
        diagnostics = [
            {**dict(value), "scene": scene}
            for value in exception.diagnostics
        ]
        raise CaptionSimilarityError(
            f"{scene} caption failed: {exception}",
            diagnostics=diagnostics,
        ) from exception


def measure(context: Context) -> contracts.RawMeasurement[dict[str, Any]]:
    """Caption the two independently captured scenes and measure cosine."""

    instance_id = f"{EVALUATOR_ID}:{context.ids.get('episode_id', 'episode')}"
    from .caption_similarity import render_protocol as task_render_protocol

    evidence_protocol = task_render_protocol(context.task)
    try:
        label = read_label(context)
        if label is None:
            raise CaptionSimilarityError(
                "caption similarity needs an answer key naming a canonical scene"
            )
        renders = context.renders_for(evidence_protocol)
        if renders is None:
            raise CaptionSimilarityError(
                f"required RenderSet {evidence_protocol!r} was not captured"
            )
        settings = getattr(renders, "capture_settings", None) or {}
        protocols = {
            value.get("camera_protocol")
            for value in settings.values() if isinstance(value, Mapping)
        }
        if protocols != {render.GT_CAPTION_ENVIRONMENT_SCENE_GRAPH_PROTOCOL}:
            raise CaptionSimilarityError(
                f"unsupported or inconsistent paired camera protocols: {protocols}")
        pairing = paired_views.strict_pair(
            renders,
            ("rgb",),
            expected_view_count=VIEW_COUNT,
            camera_protocol=render.GT_CAPTION_ENVIRONMENT_SCENE_GRAPH_PROTOCOL,
            camera_plan_policy=scene_graph_capture.CAPTION_CAMERA_PLAN,
            lighting_policy=render.LIGHTING_NORMALIZATION_POLICY,
            frame_quality_policy=scene_graph_capture.CAPTION_FRAME_QUALITY,
        )
        rows = pairing.pairs["rgb"]
        gt_images = tuple(Path(reference) for _view, reference, _candidate in rows)
        candidate_images = tuple(Path(candidate) for _view, _reference, candidate in rows)
        overview_luma = {
            "gt": [render.mean_luma(path) for path in gt_images],
            "candidate": [render.mean_luma(path) for path in candidate_images],
        }
        blank = {
            scene: [index for index, value in enumerate(values)
                    if value is not None
                    and value < scene_graph_capture.NEAR_BLACK_LUMA]
            for scene, values in overview_luma.items()
        }
        blank = {scene: values for scene, values in blank.items() if values}
        if blank:
            raise CaptionSimilarityError(
                "caption evidence contains blank overview / near-black views after capture "
                f"fallback: {blank}")

        caption_client = _caption_client()
        caption_jobs = (
            ("gt", gt_images),
            ("candidate", candidate_images),
        )
        caption_results = parallel_vlm_map(
            caption_jobs,
            lambda job: (
                job[0],
                _caption_scene(caption_client, job[1], scene=job[0]),
            ),
            thread_name_prefix="code4scene-caption-vlm",
        )
        captions_by_scene = dict(caption_results)
        gt_caption = captions_by_scene["gt"]
        candidate_caption = captions_by_scene["candidate"]
        embedded = _embedding_client().embed_pair(
            (gt_caption.caption, candidate_caption.caption)
        )
        similarity = _cosine(*embedded.vectors)
        artifact_payload = {
            "schema_version": "gt-caption-similarity.v1",
            "metric_version": METRIC_PROTOCOL,
            "render_protocol": evidence_protocol,
            "caption_protocol": CAPTION_PROTOCOL,
            "embedding_protocol": EMBEDDING_PROTOCOL,
            "caption_model": {
                "base_url": CAPTION_BASE_URL,
                "model": CAPTION_MODEL,
            },
            "embedding_model": {
                "base_url": EMBEDDING_BASE_URL,
                "model": EMBEDDING_MODEL,
            },
            "captions": {
                "gt": gt_caption.caption,
                "candidate": candidate_caption.caption,
            },
            "caption_request_ids": {
                "gt": gt_caption.request_id,
                "candidate": candidate_caption.request_id,
            },
            "caption_usage": {
                "gt": dict(gt_caption.usage),
                "candidate": dict(candidate_caption.usage),
            },
            "caption_attempts": {
                "gt": gt_caption.attempt_count,
                "candidate": candidate_caption.attempt_count,
            },
            "caption_structured_modes": {
                "gt": gt_caption.structured_mode,
                "candidate": candidate_caption.structured_mode,
            },
            "embedding_request_id": embedded.request_id,
            "embedding_usage": dict(embedded.usage),
            "embedding_dimension": len(embedded.vectors[0]),
            "cosine_similarity": similarity,
            "cosine_distance": 1.0 - similarity,
            "overview_mean_luma": overview_luma,
            "vlm_runtime": vlm_runtime_snapshot(),
        }
    except Exception as exception:  # noqa: BLE001 - evaluator failures are withheld
        diagnostics = tuple(
            dict(value)
            for value in getattr(exception, "diagnostics", ())
            if isinstance(value, Mapping)
        )
        failure_artifact = _write_artifact(
            context,
            {
                "schema_version": "gt-caption-failure.v1",
                "exception_type": type(exception).__name__,
                "failure_reason": str(exception),
                "attempts": list(diagnostics),
            },
            "gt_caption_similarity_failure",
        )
        failure_evidence = {
            "render_protocol": evidence_protocol,
            "attempt_count": len(diagnostics),
            "failure_response_artifact": failure_artifact,
        }
        return contracts.RawMeasurement(
            id=EVALUATOR_ID,
            instance_id=instance_id,
            metric_version=METRIC_PROTOCOL,
            applicable=True,
            status="error",
            coverage=None,
            raw=None,
            evidence=(failure_evidence,),
            failure_reason=f"{type(exception).__name__}: {exception}",
        )

    rounded_similarity = round(similarity, 6)
    rounded_distance = round(1.0 - similarity, 6)
    evidence = {
        **pairing.evidence(),
        "gt_id": label.get("gt_id"),
        "scoring_policy": "direct_continuous_cosine_similarity",
        "caption_protocol": CAPTION_PROTOCOL,
        "render_protocol": evidence_protocol,
        "embedding_protocol": EMBEDDING_PROTOCOL,
        "caption_model": CAPTION_MODEL,
        "caption_base_url": CAPTION_BASE_URL,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_base_url": EMBEDDING_BASE_URL,
        "independent_caption_requests": True,
        "vlm_runtime": vlm_runtime_snapshot(),
        "capture_overrides": list(
            getattr(renders, "capture_overrides", None) or []
        ),
    }
    metrics = {
        "metric_version": METRIC_PROTOCOL,
        "cosine_similarity": rounded_similarity,
        "cosine_distance": rounded_distance,
        "embedding_dimension": len(embedded.vectors[0]),
        "caption_character_count": {
            "gt": len(gt_caption.caption),
            "candidate": len(candidate_caption.caption),
        },
        "caption_attempt_count": {
            "gt": gt_caption.attempt_count,
            "candidate": candidate_caption.attempt_count,
        },
        "caption_structured_modes": {
            "gt": gt_caption.structured_mode,
            "candidate": candidate_caption.structured_mode,
        },
        "model_call_count": (
            gt_caption.attempt_count + candidate_caption.attempt_count + 1
        ),
    }
    return contracts.RawMeasurement(
        id=EVALUATOR_ID,
        instance_id=instance_id,
        metric_version=METRIC_PROTOCOL,
        applicable=True,
        status="measured",
        coverage=1.0,
        raw={
            "metrics": metrics,
            "evidence": evidence,
            "artifact_payload": artifact_payload,
        },
        evidence=(evidence,),
    )


def normalize(
    measurement: contracts.RawMeasurement[dict[str, Any]],
    _policy: Mapping[str, Any],
) -> contracts.MetricResult[dict[str, Any]]:
    """Expose clipped cosine similarity as a continuous atomic score."""

    if measurement.id != EVALUATOR_ID or measurement.metric_version != METRIC_PROTOCOL:
        raise ValueError("caption normalizer received another evaluator's measurement")
    render_protocol = next(
        (
            str(value["render_protocol"])
            for value in measurement.evidence
            if isinstance(value, Mapping) and value.get("render_protocol")
        ),
        "caption_similarity.environment_visibility_gallery_rgb",
    )
    common = {
        "id": EVALUATOR_ID,
        "instance_id": measurement.instance_id,
        "metric_version": METRIC_PROTOCOL,
        "dimension": "independent_caption_embedding_distance",
        "applicability_policy": "gt_paired_caption_continuous",
        "required_evidence": (
            f"paired_render:{render_protocol}",
            "caption_model",
            "text_embedding_model",
        ),
        "normalization_parameters": {},
        "calibration_status": "direct_metric_not_human_calibrated",
        "contributes_to_aggregate": True,
        "evidence": measurement.evidence,
    }
    if measurement.status != "measured":
        return contracts.MetricResult(
            **common,
            applicable=True,
            status=measurement.status,
            coverage=None,
            raw=None,
            score=None,
            normalization_policy=None,
            failure_reason=measurement.failure_reason,
        )
    raw = measurement.raw or {}
    metrics = dict(raw.get("metrics") or {})
    similarity = max(0.0, min(1.0, float(metrics["cosine_similarity"])))
    return contracts.MetricResult(
        **common,
        applicable=True,
        status="measured",
        coverage=1.0,
        raw={
            "unit": "cosine_distance",
            "cosine_similarity": metrics["cosine_similarity"],
            "cosine_distance": metrics["cosine_distance"],
        },
        score=similarity,
        normalization_policy="clip_cosine_similarity_to_unit_interval",
    )


def report_from_atomic(
    context: Context,
    measurement: contracts.RawMeasurement[dict[str, Any]],
    result: contracts.MetricResult[dict[str, Any]],
    *,
    report_id: str = "gt_caption_similarity",
) -> dict[str, Any]:
    """Lower the atomic contracts into the public verifier report envelope."""

    from .caption_similarity import render_protocol as task_render_protocol

    if measurement.status != "measured":
        failure_evidence = (
            dict(measurement.evidence[0])
            if measurement.evidence
            and isinstance(measurement.evidence[0], Mapping)
            else {"render_protocol": task_render_protocol(context.task)}
        )
        failure_artifact = failure_evidence.get("failure_response_artifact")
        return {
            **contracts.base(report_id, context.ids),
            "status": result.status,
            "score": None,
            "failure_reason": result.failure_reason,
            "metrics": {"atomic_result": result.to_json_dict()},
            "evidence": failure_evidence,
            "artifacts": (
                {"failure_response": str(failure_artifact)}
                if failure_artifact
                else {}
            ),
            "probes_used": ("answer_key", "render", "vlm_caption", "text_embedding"),
        }
    raw = measurement.raw or {}
    artifact_path = _write_artifact(context, raw["artifact_payload"], report_id)
    return {
        **contracts.base(report_id, context.ids),
        "status": result.status,
        "score": result.score,
        "metrics": {
            **dict(raw.get("metrics") or {}),
            "atomic_result": result.to_json_dict(),
        },
        "evidence": dict(raw.get("evidence") or {}),
        "artifacts": ({"comparison": artifact_path} if artifact_path else {}),
        "probes_used": ("answer_key", "render", "vlm_caption", "text_embedding"),
    }


def verify(context: Context) -> dict[str, Any]:
    """0.1.x entrypoint, backed by the canonical atomic implementation."""

    measurement = measure(context)
    result = normalize(measurement, {})
    return report_from_atomic(context, measurement, result)


__all__ = [
    "CAPTION_BASE_URL",
    "CAPTION_MODEL",
    "CAPTION_PROTOCOL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_PROTOCOL",
    "EVALUATOR_ID",
    "METRIC_PROTOCOL",
    "QwenCaptionClient",
    "QwenEmbeddingClient",
    "VIEW_COUNT",
    "evidence_requests",
    "measure",
    "normalize",
    "report_from_atomic",
    "verify",
]
