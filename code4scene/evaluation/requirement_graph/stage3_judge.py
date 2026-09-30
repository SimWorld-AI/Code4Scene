"""Stateless, RGB-only model adapters used by RequirementGraph Stage 3.

The three adapters in this module intentionally have narrow and different
semantic boundaries:

* :class:`LLMHolisticFrameSelector` receives only opaque frame ids and pixels;
* :class:`LLMStage3UnknownJudge` receives one sanitized visual claim plus RGB;
* :class:`LLMHolisticJudge` receives the original user prompt plus RGB.

Every request attempt creates a fresh two-message context.  The UNKNOWN judge
uses bounded, instruction-guided retries for malformed structured output and
claim-grounding contradictions; exhausted validation failures become an
audited UNKNOWN rather than an evaluation error.  Provider errors remain
explicit failures, and there is no text-response fallback in this layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import JsonSerializable
from .existing_llm import LLMClient, LLMMessage
from .stage2_contracts import Stage2EvidenceBasis, VisualGrounding
from .stage3_contracts import Stage3Verdict
from .visual_claims import sanitize_visual_claim_payload

JSON = dict[str, Any]

_FRAME_ID = re.compile(r"^s3f_[0-9]{6}$", re.ASCII)
_SELECT_TOOL = "record_holistic_frame_selection"
_IDENTITY_TOOL = "record_actor_identity_verdict"
_UNKNOWN_TOOL = "record_stage3_unknown_verdict"
_BINARY_TOOL = "record_stage3_binary_verdict"
_HOLISTIC_TOOL = "record_stage3_holistic_scores"
_DIMENSION_NAMES = (
    "global_prompt_alignment",
    "composition_and_layout",
    "style_atmosphere_coherence",
    "completeness_and_polish",
)
_DIMENSION_WEIGHTS = {
    "global_prompt_alignment": 0.40,
    "composition_and_layout": 0.25,
    "style_atmosphere_coherence": 0.20,
    "completeness_and_polish": 0.15,
}
_TRANSPORT_STATUSES = frozenset({"not_attempted", "success", "error"})
_PARSE_STATUSES = frozenset({"not_attempted", "valid", "invalid"})


@dataclass(frozen=True, slots=True)
class Stage3JudgeFrame:
    """The only frame projection accepted across a Stage 3 VLM boundary."""

    frame_id: str
    rgb: np.ndarray

    def __post_init__(self) -> None:
        frame_id = str(self.frame_id).strip()
        if not _FRAME_ID.fullmatch(frame_id) or int(frame_id.removeprefix("s3f_")) < 1:
            raise ValueError("frame_id must be an opaque Stage 3 id (s3f_NNNNNN)")
        array = np.asarray(self.rgb)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("rgb must have shape HxWx3")
        if array.dtype != np.uint8:
            raise ValueError("rgb must have dtype uint8")
        owned = np.array(array, dtype=np.uint8, order="C", copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "rgb", owned)


@dataclass(frozen=True, slots=True)
class HolisticFrameSelection(JsonSerializable):
    """Prompt-blind ordering and usability classification for exploration RGB."""

    ordered_frame_ids: tuple[str, ...] = ()
    unusable_frame_ids: tuple[str, ...] = ()
    rationale: str = ""
    transport_status: str = "not_attempted"
    parse_status: str = "not_attempted"
    error: str | None = None

    def __post_init__(self) -> None:
        ordered = _unique_ids(self.ordered_frame_ids, name="ordered_frame_ids")
        unusable = _unique_ids(self.unusable_frame_ids, name="unusable_frame_ids")
        if set(ordered) & set(unusable):
            raise ValueError("usable and unusable frame ids must be disjoint")
        _validate_statuses(self.transport_status, self.parse_status)
        error = str(self.error).strip() if self.error is not None else None
        if error == "":
            error = None
        object.__setattr__(self, "ordered_frame_ids", ordered)
        object.__setattr__(self, "unusable_frame_ids", unusable)
        object.__setattr__(self, "rationale", str(self.rationale).strip())
        object.__setattr__(self, "error", error)


@dataclass(frozen=True, slots=True)
class Stage3UnknownDecision(JsonSerializable):
    """Untrusted Stage 3 result for one formerly-UNKNOWN visual claim."""

    verdict: Stage3Verdict | str = Stage3Verdict.UNKNOWN
    confidence: float = 0.0
    evidence_frame_ids: tuple[str, ...] = ()
    rationale: str = ""
    evidence_basis: Stage2EvidenceBasis | str = Stage2EvidenceBasis.INSUFFICIENT
    grounding: VisualGrounding = field(default_factory=VisualGrounding)
    transport_status: str = "not_attempted"
    parse_status: str = "not_attempted"
    error: str | None = None

    def __post_init__(self) -> None:
        verdict = (
            self.verdict
            if isinstance(self.verdict, Stage3Verdict)
            else Stage3Verdict(str(self.verdict).strip().upper())
        )
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("confidence must be numeric") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        evidence = _unique_ids(self.evidence_frame_ids, name="evidence_frame_ids")
        basis = (
            self.evidence_basis
            if isinstance(self.evidence_basis, Stage2EvidenceBasis)
            else Stage2EvidenceBasis(str(self.evidence_basis).strip().casefold())
        )
        if not isinstance(self.grounding, VisualGrounding):
            raise TypeError("grounding must be VisualGrounding")
        _validate_statuses(self.transport_status, self.parse_status)
        error = str(self.error).strip() if self.error is not None else None
        if error == "":
            error = None
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_frame_ids", evidence)
        object.__setattr__(self, "rationale", str(self.rationale).strip())
        object.__setattr__(self, "evidence_basis", basis)
        object.__setattr__(self, "error", error)


class IdentityVerdict(str, Enum):
    """Pixels-only semantic identity of one designated Candidate Actor."""

    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class IdentityDecision(JsonSerializable):
    """One candidate-specific identity decision, never a constraint verdict."""

    verdict: IdentityVerdict | str = IdentityVerdict.UNKNOWN
    confidence: float = 0.0
    evidence_frame_ids: tuple[str, ...] = ()
    rationale: str = ""
    transport_status: str = "not_attempted"
    parse_status: str = "not_attempted"
    error: str | None = None

    def __post_init__(self) -> None:
        verdict = (
            self.verdict
            if isinstance(self.verdict, IdentityVerdict)
            else IdentityVerdict(str(self.verdict).strip().upper())
        )
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("confidence must be numeric") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        evidence = _unique_ids(self.evidence_frame_ids, name="evidence_frame_ids")
        if verdict is not IdentityVerdict.UNKNOWN and not evidence:
            raise ValueError("resolved identity decisions must cite RGB evidence")
        _validate_statuses(self.transport_status, self.parse_status)
        error = str(self.error).strip() if self.error is not None else None
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_frame_ids", evidence)
        object.__setattr__(self, "rationale", str(self.rationale).strip())
        object.__setattr__(self, "error", error or None)


@dataclass(frozen=True, slots=True)
class HolisticJudgeDimension(JsonSerializable):
    """One validated, RGB-cited holistic score returned by the adapter."""

    name: str
    score: float
    evidence_frame_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if name not in _DIMENSION_WEIGHTS:
            raise ValueError(f"unknown holistic dimension: {name!r}")
        try:
            score = float(self.score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("score must be numeric") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be finite and between 0 and 1")
        evidence = _unique_ids(self.evidence_frame_ids, name="evidence_frame_ids")
        if not evidence:
            raise ValueError("a holistic dimension must cite RGB evidence")
        rationale = str(self.rationale).strip()
        if not rationale:
            raise ValueError("a holistic dimension rationale must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "evidence_frame_ids", evidence)
        object.__setattr__(self, "rationale", rationale)


@dataclass(frozen=True, slots=True)
class HolisticJudgeDecision(JsonSerializable):
    """Validated four-dimensional holistic model result.

    ``overall_score`` is always controller-computed from the four dimension
    scores, never copied from provider output.
    """

    dimensions: tuple[HolisticJudgeDimension, ...] = ()
    overall_score: float | None = None
    summary: str = ""
    transport_status: str = "not_attempted"
    parse_status: str = "not_attempted"
    error: str | None = None

    def __post_init__(self) -> None:
        dimensions = tuple(self.dimensions)
        if any(not isinstance(item, HolisticJudgeDimension) for item in dimensions):
            raise TypeError("dimensions must contain HolisticJudgeDimension values")
        names = tuple(item.name for item in dimensions)
        if names and (len(names) != len(set(names)) or set(names) != set(_DIMENSION_NAMES)):
            raise ValueError("dimensions must contain each fixed dimension exactly once")
        overall = self.overall_score
        if overall is not None:
            overall = float(overall)
            if not math.isfinite(overall) or not 0.0 <= overall <= 1.0:
                raise ValueError("overall_score must be between 0 and 1 or None")
        _validate_statuses(self.transport_status, self.parse_status)
        error = str(self.error).strip() if self.error is not None else None
        if error == "":
            error = None
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "overall_score", overall)
        object.__setattr__(self, "summary", str(self.summary).strip())
        object.__setattr__(self, "error", error)


def _validate_statuses(transport_status: str, parse_status: str) -> None:
    if transport_status not in _TRANSPORT_STATUSES:
        raise ValueError("invalid transport_status")
    if parse_status not in _PARSE_STATUSES:
        raise ValueError("invalid parse_status")


def _unique_ids(values: Any, *, name: str) -> tuple[str, ...]:
    try:
        result = tuple(str(value).strip() for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be iterable") from exc
    if any(
        not _FRAME_ID.fullmatch(value)
        or int(value.removeprefix("s3f_")) < 1
        for value in result
    ):
        raise ValueError(f"{name} must contain only opaque Stage 3 frame ids")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _normalize_frames(
    frames: Sequence[Any],
    *,
    maximum: int,
) -> tuple[Stage3JudgeFrame, ...]:
    try:
        values = tuple(frames)
    except TypeError as exc:
        raise TypeError("frames must be a sequence") from exc
    if not values:
        raise ValueError("at least one RGB frame is required")
    if len(values) > maximum:
        raise ValueError(f"at most {maximum} RGB frames may be supplied")
    if any(type(item) is not Stage3JudgeFrame for item in values):
        bad = next(item for item in values if type(item) is not Stage3JudgeFrame)
        raise TypeError(
            "frames must contain exact Stage3JudgeFrame projections; "
            f"metadata-bearing {type(bad).__name__} values are forbidden"
        )
    ids = tuple(item.frame_id for item in values)
    if len(ids) != len(set(ids)):
        raise ValueError("frames must have unique frame ids")
    # Stage3JudgeFrame already owns and freezes its pixels.  Copy once more at
    # the method boundary so even malicious object mutation cannot alias a call.
    return tuple(Stage3JudgeFrame(item.frame_id, item.rgb) for item in values)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and "data:image" in value.casefold():
            return "<redacted-image-data>"
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview, np.ndarray)):
        return f"<{type(value).__name__} redacted>"
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _safe_provider_raw(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: JSON = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z]", "", str(key).casefold())
            if normalized in {
                "request",
                "requestbody",
                "requestmessages",
                "messages",
                "prompt",
                "prompttext",
                "prompttokenids",
                "input",
                "image",
                "images",
                "imageurl",
            }:
                result[str(key)] = "<redacted-request-data>"
            else:
                result[str(key)] = _safe_provider_raw(item)
        return result
    if isinstance(value, list):
        return [_safe_provider_raw(item) for item in value]
    return _json_safe(value)


def _safe_error(exc: BaseException) -> str:
    message = re.sub(
        r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=_-]+",
        "<redacted-image-data>",
        str(exc),
        flags=re.IGNORECASE,
    )
    return f"{type(exc).__name__}: {message[:1000]}"


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    temporary.replace(path)


def _response_record(
    request_id: str,
    call_kind: str,
    status: str,
    *,
    response: Any = None,
    error: str | None = None,
) -> JSON:
    record: JSON = {
        "request_id": request_id,
        "call_kind": call_kind,
        "status": status,
    }
    if error is not None:
        record["error"] = error[:1000]
    if response is not None:
        record["response"] = {
            "text": _json_safe(getattr(response, "text", None)),
            "reasoning": _json_safe(getattr(response, "reasoning", None)),
            "tool_calls": [
                {
                    "name": str(getattr(call, "name", "")),
                    "arguments": _json_safe(getattr(call, "arguments", {})),
                }
                for call in (getattr(response, "tool_calls", ()) or ())
            ],
            "usage": _json_safe(getattr(response, "usage", {}) or {}),
            "provider_raw": _safe_provider_raw(getattr(response, "raw", None)),
        }
    return record


class _OneShotAdapter:
    """Shared request ledger; deliberately stores no conversation history."""

    request_prefix = ""
    call_kind = ""
    tool_name = ""

    def __init__(
        self,
        client: LLMClient,
        *,
        max_tokens: int,
        manifest_path: str | Path | None,
        raw_path: str | Path | None,
    ) -> None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if bool(getattr(client, "_text_action_mode", False)):
            raise ValueError("Stage 3 RGB judging requires native structured tool calls")
        if hasattr(client, "_strict_tool_calls") and not bool(client._strict_tool_calls):
            raise ValueError(
                "OpenAI-compatible Stage 3 clients require strict_tool_calls=True"
            )
        self.client = client
        self.max_tokens = max_tokens
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.raw_path = Path(raw_path) if raw_path is not None else None
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self._request_counter = 0
        self._manifest_records: list[JSON] = []
        self._raw_records: list[JSON] = []

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._request_counter

    @property
    def current_thread_request_id(self) -> str | None:
        """Return the request most recently started by the calling thread."""

        value = getattr(self._thread_local, "request_id", None)
        return str(value) if value is not None else None

    @property
    def current_thread_request_ids(self) -> tuple[str, ...]:
        """Return the logical operation's request attempts for this thread."""

        values = getattr(self._thread_local, "request_ids", ())
        return tuple(str(value) for value in values)

    @property
    def manifest_records(self) -> tuple[JSON, ...]:
        with self._lock:
            return tuple(
                json.loads(json.dumps(item))
                for item in sorted(
                    self._manifest_records,
                    key=lambda value: str(value.get("request_id", "")),
                )
            )

    @property
    def raw_records(self) -> tuple[JSON, ...]:
        with self._lock:
            return tuple(
                json.loads(json.dumps(item))
                for item in sorted(
                    self._raw_records,
                    key=lambda value: str(value.get("request_id", "")),
                )
            )

    def _begin_request(
        self,
        frames: Sequence[Stage3JudgeFrame],
        *,
        call_kind: str | None = None,
        tool_name: str | None = None,
    ) -> str:
        with self._lock:
            self._request_counter += 1
            request_id = f"{self.request_prefix}{self._request_counter:06d}"
        self._thread_local.request_id = request_id
        record: JSON = {
            "request_id": request_id,
            "call_kind": call_kind or self.call_kind,
            "frame_ids": [frame.frame_id for frame in frames],
            "images": [
                {
                    "frame_id": frame.frame_id,
                    "hash_algorithm": "sha256_rgb_u8_c_order_v1",
                    "sha256": _sha256(frame.rgb.tobytes(order="C")),
                    "height": int(frame.rgb.shape[0]),
                    "width": int(frame.rgb.shape[1]),
                    "channels": 3,
                }
                for frame in frames
            ],
            "message_roles": ["system", "user"],
            "model": str(getattr(self.client, "model", type(self.client).__name__)),
            "tool_name": tool_name or self.tool_name,
            "schema_version": "1.0",
        }
        with self._lock:
            self._manifest_records.append(record)
            self._manifest_records.sort(
                key=lambda value: str(value.get("request_id", ""))
            )
            if self.manifest_path is not None:
                _write_json_atomic(self.manifest_path, self._manifest_records)
        return request_id

    def _append_raw(self, record: JSON) -> None:
        with self._lock:
            self._raw_records.append(record)
            self._raw_records.sort(
                key=lambda value: str(value.get("request_id", ""))
            )
            if self.raw_path is not None:
                _write_jsonl_atomic(self.raw_path, self._raw_records)


def _image_message(text: str, frames: Sequence[Stage3JudgeFrame]) -> LLMMessage:
    return LLMMessage.user_with_images(
        text,
        [frame.rgb for frame in frames],
        [f"frame_id={frame.frame_id}" for frame in frames],
    )


_SELECT_SYSTEM = """Select a small representative portfolio using only the supplied RGB pixels.
No scene prompt, claim, camera pose, object identity, or acquisition metadata is available.
Classify every supplied frame as usable or unusable. A frame is unusable when it is
blank, corrupt, severely clipped, inside geometry, or so occluded that it cannot
represent the scene. Order up to six usable opaque frame ids by portfolio value,
favoring complementary views, broad visual coverage, clarity, and viewpoint diversity.
Use only identifiers supplied with the images."""


def _select_schema() -> JSON:
    return {
        "name": _SELECT_TOOL,
        "description": "Record prompt-blind RGB frame usability and portfolio order.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ordered_frame_ids": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {"type": "string"},
                },
                "unusable_frame_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "rationale": {"type": "string"},
            },
            "required": ["ordered_frame_ids", "unusable_frame_ids", "rationale"],
        },
    }


class LLMHolisticFrameSelector(_OneShotAdapter):
    """Prompt-blind, one-shot selector over at most ten exploration frames."""

    request_prefix = "s3s_"
    call_kind = "frame_selection"
    tool_name = _SELECT_TOOL

    def __init__(
        self,
        client: LLMClient,
        *,
        max_tokens: int = 512,
        manifest_path: str | Path | None = None,
        raw_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            client,
            max_tokens=max_tokens,
            manifest_path=manifest_path,
            raw_path=raw_path,
        )

    def select(self, frames: Sequence[Stage3JudgeFrame]) -> HolisticFrameSelection:
        try:
            normalized = _normalize_frames(frames, maximum=10)
        except (TypeError, ValueError, OverflowError) as exc:
            return HolisticFrameSelection(error=f"Selector input rejected: {exc}")
        request_id = self._begin_request(normalized)
        response: Any = None
        try:
            response = self.client.chat(
                [
                    LLMMessage.text("system", _SELECT_SYSTEM),
                    _image_message(
                        "Assess these opaque RGB frames. No semantic task text is provided.",
                        normalized,
                    ),
                ],
                [_select_schema()],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            result = _parse_selection(
                response,
                available_ids=tuple(frame.frame_id for frame in normalized),
            )
            raw = _response_record(
                request_id, self.call_kind, "success", response=response
            )
        except Exception as exc:  # noqa: BLE001 - fail closed at provider boundary
            error = _safe_error(exc)
            result = HolisticFrameSelection(
                transport_status="error",
                parse_status="not_attempted",
                error=error,
            )
            raw = _response_record(
                request_id, self.call_kind, "error", response=response, error=error
            )
        self._append_raw(raw)
        return result


def _parse_selection(response: Any, *, available_ids: tuple[str, ...]) -> HolisticFrameSelection:
    calls = list(getattr(response, "tool_calls", ()) or ())
    if len(calls) != 1 or str(getattr(calls[0], "name", "")) != _SELECT_TOOL:
        return HolisticFrameSelection(
            transport_status="success",
            parse_status="invalid",
            error="Selector returned no unique frame-selection tool call.",
        )
    raw = getattr(calls[0], "arguments", None)
    expected = {"ordered_frame_ids", "unusable_frame_ids", "rationale"}
    if not isinstance(raw, Mapping) or set(raw) != expected:
        return HolisticFrameSelection(
            transport_status="success",
            parse_status="invalid",
            error="Selector tool arguments did not match the strict schema.",
        )
    ordered = raw["ordered_frame_ids"]
    unusable = raw["unusable_frame_ids"]
    rationale = raw["rationale"]
    if (
        not isinstance(ordered, list)
        or not isinstance(unusable, list)
        or any(not isinstance(item, str) for item in [*ordered, *unusable])
        or not isinstance(rationale, str)
        or not rationale.strip()
    ):
        return HolisticFrameSelection(
            transport_status="success",
            parse_status="invalid",
            error="Selector returned invalid field types.",
        )
    if len(ordered) > 6 or len(ordered) != len(set(ordered)) or len(unusable) != len(set(unusable)):
        return HolisticFrameSelection(
            transport_status="success",
            parse_status="invalid",
            error="Selector returned duplicate ids or exceeded the six-frame cap.",
        )
    supplied = set(available_ids)
    if (
        set(ordered) & set(unusable)
        or not set(ordered).issubset(supplied)
        or not set(unusable).issubset(supplied)
    ):
        return HolisticFrameSelection(
            transport_status="success",
            parse_status="invalid",
            error="Selector returned unsupplied or conflicting frame ids.",
        )
    return HolisticFrameSelection(
        ordered_frame_ids=tuple(ordered),
        unusable_frame_ids=tuple(unusable),
        rationale=rationale,
        transport_status="success",
        parse_status="valid",
    )


@dataclass(frozen=True, slots=True)
class _BasisContract:
    claim_kind: str
    match: Stage2EvidenceBasis
    mismatch: Stage2EvidenceBasis

    def expected(self, verdict: Stage3Verdict) -> Stage2EvidenceBasis:
        if verdict is Stage3Verdict.MATCH:
            return self.match
        if verdict is Stage3Verdict.MISMATCH:
            return self.mismatch
        return Stage2EvidenceBasis.INSUFFICIENT

    @property
    def allowed(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (self.match.value, self.mismatch.value, Stage2EvidenceBasis.INSUFFICIENT.value)
            )
        )

    @property
    def instruction(self) -> str:
        return (
            f"For this {self.claim_kind} claim use exactly: MATCH -> '{self.match.value}'; "
            f"MISMATCH -> '{self.mismatch.value}'; UNKNOWN -> 'insufficient'."
        )


def _basis_contract(payload: Mapping[str, Any]) -> _BasisContract:
    if payload["node_type"] == "entity":
        return _BasisContract(
            "affirmative object-existence",
            Stage2EvidenceBasis.SUPPORT,
            Stage2EvidenceBasis.SEARCHED_ABSENCE,
        )
    kind = str(payload["predicate_type"])
    if kind == "existence":
        if payload["polarity"] == "negated":
            return _BasisContract(
                "negated object-existence",
                Stage2EvidenceBasis.SEARCHED_ABSENCE,
                Stage2EvidenceBasis.VISIBLE_CONTRADICTION,
            )
        return _BasisContract(
            "affirmative object-existence",
            Stage2EvidenceBasis.SUPPORT,
            Stage2EvidenceBasis.SEARCHED_ABSENCE,
        )
    mismatch = {
        "attribute": Stage2EvidenceBasis.ATTRIBUTE_CONTRADICTION,
        "material": Stage2EvidenceBasis.MATERIAL_CONTRADICTION,
        "spatial_relation": Stage2EvidenceBasis.SUFFICIENT_RELATION_VIEW,
        "count": Stage2EvidenceBasis.SUFFICIENT_COUNT_VIEW,
        "atmosphere": Stage2EvidenceBasis.SCENE_CONTRADICTION,
        "scene_identity": Stage2EvidenceBasis.SCENE_CONTRADICTION,
        "quantity": Stage2EvidenceBasis.SCENE_CONTRADICTION,
        "set": Stage2EvidenceBasis.SCENE_CONTRADICTION,
        "distribution": Stage2EvidenceBasis.SCENE_CONTRADICTION,
        "composition": Stage2EvidenceBasis.SCENE_CONTRADICTION,
        "style_bundle": Stage2EvidenceBasis.SCENE_CONTRADICTION,
        "environment": Stage2EvidenceBasis.SCENE_CONTRADICTION,
        "logic": Stage2EvidenceBasis.SCENE_CONTRADICTION,
        "boundary": Stage2EvidenceBasis.SCENE_CONTRADICTION,
    }.get(kind)
    if mismatch is None:
        raise ValueError(f"unsupported visual claim kind: {kind!r}")
    if kind == "count":
        constraint = payload.get("constraint")
        if not isinstance(constraint, Mapping):
            raise ValueError("count claims require a visual count constraint")
        if constraint.get("operator") == "between" and constraint.get("upper_value") is None:
            raise ValueError("between count claims require upper_value")
    return _BasisContract(kind, Stage2EvidenceBasis.SUPPORT, mismatch)


_GROUNDING_PROPERTIES = {
    "subject_confirmed": {"type": "boolean"},
    "participants_confirmed": {"type": "boolean"},
    "relation_scope_covered": {"type": "boolean"},
    "collection_complete": {"type": "boolean"},
    "instances_countable": {"type": "boolean"},
    "visible_instance_count": {
        "anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]
    },
}


def _unknown_schema(
    payload: Mapping[str, Any],
    *,
    maximum_evidence_frames: int,
) -> JSON:
    contract = _basis_contract(payload)
    return {
        "name": _UNKNOWN_TOOL,
        "description": "Record one pixels-only Stage 3 claim verdict. " + contract.instruction,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["MATCH", "MISMATCH", "UNKNOWN"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_frame_ids": {
                    "type": "array",
                    "maxItems": maximum_evidence_frames,
                    "items": {"type": "string"},
                },
                "rationale": {"type": "string"},
                "evidence_basis": {"type": "string", "enum": list(contract.allowed)},
                "grounding": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": _GROUNDING_PROPERTIES,
                    "required": list(_GROUNDING_PROPERTIES),
                },
            },
            "required": [
                "verdict",
                "confidence",
                "evidence_frame_ids",
                "rationale",
                "evidence_basis",
                "grounding",
            ],
        },
    }


def _binary_schema(
    payload: Mapping[str, Any],
    *,
    maximum_evidence_frames: int,
) -> JSON:
    schema = _unknown_schema(
        payload,
        maximum_evidence_frames=maximum_evidence_frames,
    )
    schema["name"] = _BINARY_TOOL
    schema["description"] = (
        "Record one best-evidence binary Stage 3 claim verdict. "
        + _basis_contract(payload).instruction
    )
    schema["parameters"]["properties"]["verdict"]["enum"] = [
        "MATCH",
        "MISMATCH",
    ]
    return schema


_UNKNOWN_SYSTEM = """Re-evaluate exactly one visual claim using only the supplied exploration RGB.
The frame labels are opaque identifiers, not scene metadata. Return MATCH or MISMATCH
only when the cited pixels and claim-specific grounding establish the result. Return
UNKNOWN when the subject, participants, full relation scope, or countable collection
is not sufficiently visible. Do not infer engine metadata, hidden objects, camera pose,
or undeclared acquisition history. Frames are supplied in focus-first order. For a
target-specific individual claim, the dominant centered instance in the first frame is
the designated subject, even when same-category instances appear in later context
frames. Do not reject the claim merely because RGB cannot distinguish that designated
subject from those other instances; use later frames only to confirm its visible
attribute or surrounding relation. This focus cue identifies what to inspect; it does
not establish that the claim is true. Judge a stated visible relation directly and do
not demand an unseen canonical/reference image. For an object claim, MATCH requires an
unambiguous instance of
the exact named visual category: do not broaden the noun or substitute a functionally
related object, marking, arrow, symbol, container, or piece of industrial equipment.
If the category remains ambiguous, return UNKNOWN. Every resolved verdict must cite
supplied RGB ids. For counts, report a visible_instance_count only with a complete,
reliably de-duplicated collection. Count-claim frames are also supplied in focus-first
order. Multiple early focus frames may center different routed Candidate Actors. When
their pixels unambiguously show the requested category at different surrounding scene
positions, treat those centered targets as distinct instances rather than possible
duplicate views; use later wider frames to verify the collection-level spatial relation.
The routing cue establishes distinct observation targets, not their category or whether
the claim is true. For a Stage0 v2 quantity, set/list, distribution, composition,
style, environment, logic, or boundary claim, judge the complete claim and its
semantic_dsl from the multi-view scene overview. Qualitative words such as roughly,
few, dense, sparse, nearly, most, or almost none are visual tolerances, not exact
deterministic counts. Return UNKNOWN when the supplied views do not cover the relevant
scene extent or visible detail. Use no identifier other than those supplied."""


_BINARY_SYSTEM = """Make the final best-evidence binary decision for exactly one visual claim
using only the supplied Candidate RGB frames. You must return MATCH or MISMATCH; UNKNOWN and
abstention are not allowed. Weigh all visible evidence together and express residual uncertainty
through confidence and rationale. MATCH means the cited pixels support the complete claim more
strongly than they contradict it; otherwise return MISMATCH. For colour, judge visible surface
colour while accounting for ordinary illumination and shadow. For material, judge visible material
appearance such as wood, metal, glass, stone, fabric, or concrete without inventing hidden physical
composition. For counts, layouts, collections, and relations, use the widest relevant frames as
well as targeted views. Cite only supplied RGB frame ids and never infer engine metadata, filenames,
camera metadata, or unseen content."""


def _claim_precision_instruction(payload: Mapping[str, Any]) -> str:
    """Add a narrow visual distinction only when the claim actually needs it."""

    claim_text = _canonical_json(payload).casefold()
    if "warning sign" in claim_text or "hazard sign" in claim_text:
        return (
            "\nFor this sign claim, a warning sign requires a visibly recognizable "
            "warning or hazard sign, not a generic directional or decorative marking."
        )
    if "chair" in claim_text and "fac" in claim_text and "table" in claim_text:
        return (
            "\nFor this chair-facing-table claim, infer facing from visible furniture "
            "geometry: the chair seat's open/front direction should point toward the "
            "table and its backrest should lie on the side away from the table. Seeing "
            "the backrest from the camera does not by itself mean the chair faces away "
            "from the table."
        )
    return ""


class LLMStage3UnknownJudge(_OneShotAdapter):
    """Stateless resolver used by the controller's per-claim batch loop."""

    request_prefix = "s3r_"
    call_kind = "unknown_resolution"
    tool_name = _UNKNOWN_TOOL

    def __init__(
        self,
        client: LLMClient,
        *,
        max_tokens: int = 640,
        max_frames_per_request: int = 6,
        max_binary_frames_per_request: int = 10,
        max_validation_retries: int = 2,
        manifest_path: str | Path | None = None,
        raw_path: str | Path | None = None,
    ) -> None:
        if (
            isinstance(max_frames_per_request, bool)
            or not isinstance(max_frames_per_request, int)
            or not 1 <= max_frames_per_request <= 10
        ):
            raise ValueError("max_frames_per_request must be between 1 and 10")
        if (
            isinstance(max_binary_frames_per_request, bool)
            or not isinstance(max_binary_frames_per_request, int)
            or not 1 <= max_binary_frames_per_request <= 10
        ):
            raise ValueError(
                "max_binary_frames_per_request must be between 1 and 10"
            )
        if (
            isinstance(max_validation_retries, bool)
            or not isinstance(max_validation_retries, int)
            or not 0 <= max_validation_retries <= 8
        ):
            raise ValueError("max_validation_retries must be between 0 and 8")
        super().__init__(
            client,
            max_tokens=max_tokens,
            manifest_path=manifest_path,
            raw_path=raw_path,
        )
        self.max_frames_per_request = max_frames_per_request
        self.max_binary_frames_per_request = max_binary_frames_per_request
        self.max_validation_retries = max_validation_retries

    @staticmethod
    def _repair_instruction(error: str, *, allow_unknown: bool) -> str:
        detail = " ".join(str(error).split())[:500]
        instruction = (
            "\n\nYour previous structured response was rejected by the verifier: "
            f"{detail} Return exactly one tool call whose arguments match the "
            "provided schema. Do not rename keys, add keys, omit keys, or encode "
            "the grounding object as a string. Use JSON booleans and null with the "
            "declared field types."
        )
        if allow_unknown:
            instruction += (
                " If the RGB does not establish the claim-specific grounding "
                "required for MATCH or MISMATCH, return UNKNOWN with confidence "
                "0, an empty evidence_frame_ids list, evidence_basis=insufficient, "
                "all grounding flags false, and visible_instance_count=null."
            )
        return instruction

    @staticmethod
    def _validation_fallback(
        error: str,
        *,
        attempt_count: int,
    ) -> Stage3UnknownDecision:
        return Stage3UnknownDecision(
            verdict=Stage3Verdict.UNKNOWN,
            confidence=0.0,
            evidence_frame_ids=(),
            rationale=(
                "Stage 3 judge output remained invalid after "
                f"{attempt_count} attempts and was downgraded to UNKNOWN. "
                f"Last validation error: {error}"
            ),
            evidence_basis=Stage2EvidenceBasis.INSUFFICIENT,
            grounding=VisualGrounding(),
            transport_status="success",
            parse_status="valid",
        )

    def _judge_prepared(
        self,
        *,
        payload: Mapping[str, Any],
        normalized: Sequence[Stage3JudgeFrame],
        contract: _BasisContract,
        system_text: str,
        payload_text: str,
        schema: JSON,
        call_kind: str,
        tool_name: str,
        maximum_evidence_frames: int,
        allowed_verdicts: frozenset[Stage3Verdict],
        enforce_claim_grounding: bool,
    ) -> Stage3UnknownDecision:
        self._thread_local.request_ids = []
        last_invalid: Stage3UnknownDecision | None = None
        first_request_id: str | None = None
        attempt_count = self.max_validation_retries + 1
        for attempt_index in range(attempt_count):
            repair = (
                ""
                if last_invalid is None
                else self._repair_instruction(
                    last_invalid.error or "structured output validation failed",
                    allow_unknown=Stage3Verdict.UNKNOWN in allowed_verdicts,
                )
            )
            request_id = self._begin_request(
                normalized,
                call_kind=call_kind,
                tool_name=tool_name,
            )
            self._thread_local.request_ids.append(request_id)
            if first_request_id is None:
                first_request_id = request_id
            response: Any = None
            try:
                response = self.client.chat(
                    [
                        LLMMessage.text("system", system_text + repair),
                        _image_message(
                            "Visual claim payload:\n" + payload_text,
                            normalized,
                        ),
                    ],
                    [schema],
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
                result = _parse_unknown(
                    response,
                    payload=payload,
                    available_ids={frame.frame_id for frame in normalized},
                    contract=contract,
                    maximum_evidence_frames=maximum_evidence_frames,
                    tool_name=tool_name,
                    allowed_verdicts=allowed_verdicts,
                    enforce_claim_grounding=enforce_claim_grounding,
                )
                raw = _response_record(
                    request_id,
                    call_kind,
                    "success",
                    response=response,
                )
                raw["parse_status"] = result.parse_status
                if result.error is not None:
                    raw["validation_error"] = result.error[:1000]
            except Exception as exc:  # noqa: BLE001 - provider boundary
                error = _safe_error(exc)
                result = _unknown_failure(
                    f"Judge call failed: {error}",
                    transport_status="error",
                    parse_status="not_attempted",
                    error=error,
                )
                raw = _response_record(
                    request_id,
                    call_kind,
                    "error",
                    response=response,
                    error=error,
                )
                raw["parse_status"] = result.parse_status
            raw["attempt_index"] = attempt_index + 1
            if request_id != first_request_id:
                raw["retry_of"] = first_request_id
            self._append_raw(raw)
            if result.transport_status != "success":
                return result
            if result.parse_status == "valid":
                return result
            last_invalid = result
        assert last_invalid is not None
        return self._validation_fallback(
            last_invalid.error or "structured output validation failed",
            attempt_count=attempt_count,
        )

    def judge(
        self,
        claim_payload: Mapping[str, Any],
        frames: Sequence[Stage3JudgeFrame],
    ) -> Stage3UnknownDecision:
        try:
            payload = sanitize_visual_claim_payload(claim_payload)
            normalized = _normalize_frames(
                frames,
                maximum=self.max_frames_per_request,
            )
            contract = _basis_contract(payload)
        except (TypeError, ValueError, OverflowError) as exc:
            return _unknown_failure(f"Judge input rejected: {type(exc).__name__}: {exc}")
        payload_text = _canonical_json(payload)
        return self._judge_prepared(
            payload=payload,
            normalized=normalized,
            contract=contract,
            system_text=(
                _UNKNOWN_SYSTEM
                + _claim_precision_instruction(payload)
                + "\n\n"
                + contract.instruction
            ),
            payload_text=payload_text,
            schema=_unknown_schema(
                payload,
                maximum_evidence_frames=self.max_frames_per_request,
            ),
            call_kind=self.call_kind,
            tool_name=self.tool_name,
            maximum_evidence_frames=self.max_frames_per_request,
            allowed_verdicts=frozenset(Stage3Verdict),
            enforce_claim_grounding=True,
        )

    def judge_binary(
        self,
        claim_payload: Mapping[str, Any],
        frames: Sequence[Stage3JudgeFrame],
    ) -> Stage3UnknownDecision:
        """Return a final RGB-cited MATCH/MISMATCH best-evidence decision."""

        try:
            payload = sanitize_visual_claim_payload(claim_payload)
            normalized = _normalize_frames(
                frames,
                maximum=self.max_binary_frames_per_request,
            )
            contract = _basis_contract(payload)
        except (TypeError, ValueError, OverflowError) as exc:
            return _unknown_failure(
                f"Binary judge input rejected: {type(exc).__name__}: {exc}"
            )
        payload_text = _canonical_json(payload)
        return self._judge_prepared(
            payload=payload,
            normalized=normalized,
            contract=contract,
            system_text=(
                _BINARY_SYSTEM
                + _claim_precision_instruction(payload)
                + "\n\n"
                + contract.instruction
            ),
            payload_text=payload_text,
            schema=_binary_schema(
                payload,
                maximum_evidence_frames=self.max_binary_frames_per_request,
            ),
            call_kind="binary_resolution",
            tool_name=_BINARY_TOOL,
            maximum_evidence_frames=self.max_binary_frames_per_request,
            allowed_verdicts=frozenset(
                {Stage3Verdict.MATCH, Stage3Verdict.MISMATCH}
            ),
            enforce_claim_grounding=False,
        )


_IDENTITY_SYSTEM = """Judge the semantic identity of exactly one designated Candidate Actor.
All supplied frames show the same centered Actor from one or more camera angles. Compare
only the Actor's visible object category with expected_entity_name and its aliases.
MATCH means it is visibly the requested kind of object, including a semantically
equivalent asset. MISMATCH means it is clearly a different kind of object. UNKNOWN
means visibility or category evidence is insufficient. Do not judge count, position,
distance, orientation, spatial relations, material, colour, style, preservation, or
overall scene quality. Do not infer identity from filenames, retrieval rank, camera
metadata, or hidden engine state. Every resolved decision must cite supplied frame ids."""


def _identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "expected_entity_name",
        "aliases",
    }:
        raise ValueError(
            "identity payload must contain exactly expected_entity_name and aliases"
        )
    name = str(value["expected_entity_name"]).strip()
    aliases_raw = value["aliases"]
    if (
        not name
        or not isinstance(aliases_raw, Sequence)
        or isinstance(aliases_raw, (str, bytes, bytearray))
    ):
        raise ValueError("identity payload contains invalid name or aliases")
    aliases = tuple(
        dict.fromkeys(str(alias).strip() for alias in aliases_raw if str(alias).strip())
    )
    if len(aliases) > 16:
        raise ValueError("identity payload may contain at most 16 aliases")
    return {"expected_entity_name": name, "aliases": list(aliases)}


def _identity_schema(*, maximum_evidence_frames: int) -> JSON:
    return {
        "name": _IDENTITY_TOOL,
        "description": "Record only the semantic identity of one Candidate Actor.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["MATCH", "MISMATCH", "UNKNOWN"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_frame_ids": {
                    "type": "array",
                    "maxItems": maximum_evidence_frames,
                    "items": {"type": "string"},
                },
                "rationale": {"type": "string"},
            },
            "required": [
                "verdict",
                "confidence",
                "evidence_frame_ids",
                "rationale",
            ],
        },
    }


def _identity_failure(
    rationale: str,
    *,
    transport_status: str = "not_attempted",
    parse_status: str = "not_attempted",
    error: str | None = None,
) -> IdentityDecision:
    return IdentityDecision(
        verdict=IdentityVerdict.UNKNOWN,
        confidence=0.0,
        evidence_frame_ids=(),
        rationale=rationale,
        transport_status=transport_status,
        parse_status=parse_status,
        error=error or rationale,
    )


def _parse_identity(
    response: Any,
    *,
    available_ids: set[str],
    maximum_evidence_frames: int,
) -> IdentityDecision:
    def invalid(reason: str) -> IdentityDecision:
        return _identity_failure(
            reason,
            transport_status="success",
            parse_status="invalid",
            error=reason,
        )

    calls = list(getattr(response, "tool_calls", ()) or ())
    if len(calls) != 1 or str(getattr(calls[0], "name", "")) != _IDENTITY_TOOL:
        return invalid("Identity judge returned no unique identity tool call.")
    raw = getattr(calls[0], "arguments", None)
    expected = {"verdict", "confidence", "evidence_frame_ids", "rationale"}
    if not isinstance(raw, Mapping) or set(raw) != expected:
        return invalid("Identity tool arguments did not match the strict schema.")
    try:
        verdict = IdentityVerdict(str(raw["verdict"]).strip().upper())
    except (TypeError, ValueError):
        return invalid("Identity judge returned an invalid verdict.")
    confidence_raw = raw["confidence"]
    if isinstance(confidence_raw, bool) or not isinstance(
        confidence_raw, (int, float)
    ):
        return invalid("Identity judge returned invalid confidence.")
    confidence = float(confidence_raw)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return invalid("Identity judge returned invalid confidence.")
    raw_ids = raw["evidence_frame_ids"]
    if (
        not isinstance(raw_ids, list)
        or len(raw_ids) > maximum_evidence_frames
        or any(not isinstance(item, str) for item in raw_ids)
        or len(raw_ids) != len(set(raw_ids))
        or not set(raw_ids).issubset(available_ids)
    ):
        return invalid("Identity judge returned invalid evidence frame ids.")
    if verdict is not IdentityVerdict.UNKNOWN and not raw_ids:
        return invalid("Resolved identity verdict did not cite supplied RGB evidence.")
    rationale = raw["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        return invalid("Identity judge returned an invalid rationale.")
    if verdict is IdentityVerdict.UNKNOWN:
        confidence = 0.0
        raw_ids = []
    return IdentityDecision(
        verdict=verdict,
        confidence=confidence,
        evidence_frame_ids=tuple(raw_ids),
        rationale=rationale,
        transport_status="success",
        parse_status="valid",
    )


class LLMStage3IdentityJudge(_OneShotAdapter):
    """One-request, candidate-specific semantic identity judge."""

    request_prefix = "s3i_"
    call_kind = "actor_identity"
    tool_name = _IDENTITY_TOOL

    def __init__(
        self,
        client: LLMClient,
        *,
        max_tokens: int = 384,
        max_frames_per_request: int = 2,
        manifest_path: str | Path | None = None,
        raw_path: str | Path | None = None,
    ) -> None:
        if (
            isinstance(max_frames_per_request, bool)
            or not isinstance(max_frames_per_request, int)
            or not 1 <= max_frames_per_request <= 2
        ):
            raise ValueError("identity judge accepts one or two frames")
        super().__init__(
            client,
            max_tokens=max_tokens,
            manifest_path=manifest_path,
            raw_path=raw_path,
        )
        self.max_frames_per_request = max_frames_per_request

    def judge_identity(
        self,
        identity_payload: Mapping[str, Any],
        frames: Sequence[Stage3JudgeFrame],
    ) -> IdentityDecision:
        try:
            payload = _identity_payload(identity_payload)
            normalized = _normalize_frames(
                frames,
                maximum=self.max_frames_per_request,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return _identity_failure(
                f"Identity judge input rejected: {type(exc).__name__}: {exc}"
            )
        request_id = self._begin_request(normalized)
        response: Any = None
        try:
            response = self.client.chat(
                [
                    LLMMessage.text("system", _IDENTITY_SYSTEM),
                    _image_message(
                        "Expected Actor identity:\n" + _canonical_json(payload),
                        normalized,
                    ),
                ],
                [
                    _identity_schema(
                        maximum_evidence_frames=self.max_frames_per_request
                    )
                ],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            result = _parse_identity(
                response,
                available_ids={frame.frame_id for frame in normalized},
                maximum_evidence_frames=self.max_frames_per_request,
            )
            raw = _response_record(
                request_id, self.call_kind, "success", response=response
            )
        except Exception as exc:  # noqa: BLE001 - provider boundary
            error = _safe_error(exc)
            result = _identity_failure(
                f"Identity judge call failed: {error}",
                transport_status="error",
                parse_status="not_attempted",
                error=error,
            )
            raw = _response_record(
                request_id, self.call_kind, "error", response=response, error=error
            )
        self._append_raw(raw)
        return result


def _unknown_failure(
    rationale: str,
    *,
    transport_status: str = "not_attempted",
    parse_status: str = "not_attempted",
    error: str | None = None,
) -> Stage3UnknownDecision:
    return Stage3UnknownDecision(
        verdict=Stage3Verdict.UNKNOWN,
        confidence=0.0,
        evidence_frame_ids=(),
        rationale=rationale,
        evidence_basis=Stage2EvidenceBasis.INSUFFICIENT,
        grounding=VisualGrounding(),
        transport_status=transport_status,
        parse_status=parse_status,
        error=error or rationale,
    )


def _parse_unknown(
    response: Any,
    *,
    payload: Mapping[str, Any],
    available_ids: set[str],
    contract: _BasisContract,
    maximum_evidence_frames: int,
    tool_name: str = _UNKNOWN_TOOL,
    allowed_verdicts: frozenset[Stage3Verdict] = frozenset(Stage3Verdict),
    enforce_claim_grounding: bool = True,
) -> Stage3UnknownDecision:
    def invalid(reason: str) -> Stage3UnknownDecision:
        return _unknown_failure(
            reason,
            transport_status="success",
            parse_status="invalid",
            error=reason,
        )

    calls = list(getattr(response, "tool_calls", ()) or ())
    if len(calls) != 1 or str(getattr(calls[0], "name", "")) != tool_name:
        return invalid("Judge returned no unique Stage 3 verdict tool call.")
    raw = getattr(calls[0], "arguments", None)
    expected = {
        "verdict",
        "confidence",
        "evidence_frame_ids",
        "rationale",
        "evidence_basis",
        "grounding",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        return invalid("Judge tool arguments did not match the strict schema.")
    verdict_text = raw["verdict"]
    if not isinstance(verdict_text, str):
        return invalid("Judge returned an invalid verdict.")
    try:
        verdict = Stage3Verdict(verdict_text.strip().upper())
    except ValueError:
        return invalid("Judge returned an invalid verdict.")
    if verdict not in allowed_verdicts:
        return invalid("Judge returned a verdict forbidden by this decision mode.")
    confidence_raw = raw["confidence"]
    if isinstance(confidence_raw, bool) or not isinstance(confidence_raw, (int, float)):
        return invalid("Judge returned invalid confidence.")
    confidence = float(confidence_raw)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return invalid("Judge returned invalid confidence.")
    raw_ids = raw["evidence_frame_ids"]
    if (
        not isinstance(raw_ids, list)
        or len(raw_ids) > maximum_evidence_frames
        or any(not isinstance(item, str) for item in raw_ids)
        or len(raw_ids) != len(set(raw_ids))
        or not set(raw_ids).issubset(available_ids)
    ):
        return invalid("Judge returned invalid or unsupplied evidence frame ids.")
    if verdict is not Stage3Verdict.UNKNOWN and not raw_ids:
        return invalid("Resolved verdict did not cite supplied RGB evidence.")
    rationale = raw["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        return invalid("Judge returned invalid rationale.")
    basis_text = raw["evidence_basis"]
    try:
        basis = Stage2EvidenceBasis(str(basis_text).strip().casefold())
    except (TypeError, ValueError):
        return invalid("Judge returned invalid evidence basis.")
    if basis is not contract.expected(verdict):
        return invalid(
            "Judge returned an evidence basis incompatible with the claim-specific verdict."
        )
    grounding_raw = raw["grounding"]
    bool_keys = {
        "subject_confirmed",
        "participants_confirmed",
        "relation_scope_covered",
        "collection_complete",
        "instances_countable",
    }
    expected_grounding = bool_keys | {"visible_instance_count"}
    if (
        not isinstance(grounding_raw, Mapping)
        or set(grounding_raw) != expected_grounding
        or any(type(grounding_raw[key]) is not bool for key in bool_keys)
    ):
        return invalid("Judge returned invalid visual grounding flags.")
    visible_count = grounding_raw["visible_instance_count"]
    if visible_count is not None and (
        isinstance(visible_count, bool)
        or not isinstance(visible_count, int)
        or visible_count < 0
    ):
        return invalid("Judge returned invalid visible instance count.")
    # UNKNOWN is the model's explicit admission that the pixels do not settle
    # the claim.  Auxiliary grounding cannot make that semantic result more
    # authoritative, and vision models often populate schema-required flags
    # even though they are irrelevant to an unresolved claim.  Canonicalize
    # those fields before the controller applies the required UNKNOWN policy;
    # malformed field shapes, illegal citations, and non-insufficient bases
    # were still rejected above.
    if verdict is Stage3Verdict.UNKNOWN:
        return Stage3UnknownDecision(
            verdict=Stage3Verdict.UNKNOWN,
            confidence=0.0,
            evidence_frame_ids=(),
            rationale=rationale,
            evidence_basis=Stage2EvidenceBasis.INSUFFICIENT,
            grounding=VisualGrounding(),
            transport_status="success",
            parse_status="valid",
        )
    is_count_claim = (
        payload["node_type"] == "predicate"
        and payload["predicate_type"] == "count"
    )
    normalized_grounding = dict(grounding_raw)
    if not is_count_claim:
        # The shared grounding schema includes count-only fields for every
        # claim kind.  Some VLMs use visible_instance_count=0 as shorthand for
        # "the subject is not visible" even on attributes and relations.  That
        # field is irrelevant outside count claims and must not invalidate an
        # otherwise well-formed RGB-cited verdict.
        normalized_grounding["visible_instance_count"] = None
        visible_count = None
    elif visible_count is not None and not grounding_raw["instances_countable"]:
        return invalid("Judge reported a count without countable RGB evidence.")
    grounding = VisualGrounding(**normalized_grounding)
    # A resolved count answer already contains all inputs needed for the
    # semantic verdict. Vision models occasionally report a correct, complete
    # visible count but attach the opposite free-form verdict (for example,
    # visible_instance_count=4 with a GTE 2 constraint and verdict=MISMATCH).
    # Preserve that raw response in the request audit, but canonicalize the
    # accepted decision from the structured count fields so a contradictory
    # label cannot turn an otherwise measurable leaf into a pipeline ERROR.
    if (
        payload["node_type"] == "predicate"
        and payload["predicate_type"] == "count"
        and grounding.collection_complete
        and grounding.instances_countable
        and grounding.visible_instance_count is not None
    ):
        canonical_verdict = (
            Stage3Verdict.MATCH
            if _count_satisfies(
                payload["constraint"], grounding.visible_instance_count
            )
            else Stage3Verdict.MISMATCH
        )
        if verdict is not canonical_verdict:
            rationale = (
                f"{rationale} Structured count normalization used visible_instance_count="
                f"{grounding.visible_instance_count} and the declared constraint to "
                f"canonicalize the verdict as {canonical_verdict.value}."
            )
            verdict = canonical_verdict
            basis = contract.expected(verdict)
    if enforce_claim_grounding:
        grounding_error = _claim_grounding_error(payload, verdict, grounding)
        if grounding_error is not None:
            return invalid(grounding_error)
    return Stage3UnknownDecision(
        verdict=verdict,
        confidence=confidence,
        evidence_frame_ids=tuple(raw_ids),
        rationale=rationale,
        evidence_basis=basis,
        grounding=grounding,
        transport_status="success",
        parse_status="valid",
    )


def _claim_grounding_error(
    payload: Mapping[str, Any],
    verdict: Stage3Verdict,
    grounding: VisualGrounding,
) -> str | None:
    if verdict is Stage3Verdict.UNKNOWN:
        return None
    if payload["node_type"] == "entity":
        if verdict is Stage3Verdict.MATCH and not grounding.subject_confirmed:
            return "Object MATCH requires visually confirmed subject grounding."
        if verdict is Stage3Verdict.MISMATCH and not grounding.collection_complete:
            return "Object-absence MISMATCH requires complete collection grounding."
        return None
    kind = str(payload["predicate_type"])
    polarity = str(payload["polarity"])
    if kind == "existence":
        visible_subject_verdict = (
            Stage3Verdict.MISMATCH if polarity == "negated" else Stage3Verdict.MATCH
        )
        if verdict is visible_subject_verdict and not grounding.subject_confirmed:
            return "Visible existence verdict requires confirmed subject grounding."
        if verdict is not visible_subject_verdict and not grounding.collection_complete:
            return "Searched-absence verdict requires complete collection grounding."
    elif kind in {"attribute", "material"} and not grounding.subject_confirmed:
        return f"{kind} verdict requires confirmed subject grounding."
    elif kind == "spatial_relation" and not (
        grounding.participants_confirmed and grounding.relation_scope_covered
    ):
        return "Relation verdict requires confirmed participants and relation scope."
    elif kind == "count":
        if not (
            grounding.collection_complete
            and grounding.instances_countable
            and grounding.visible_instance_count is not None
        ):
            return "Count verdict requires a complete, countable collection and visible count."
        satisfies = _count_satisfies(payload["constraint"], grounding.visible_instance_count)
        if (verdict is Stage3Verdict.MATCH) != satisfies:
            return "Count verdict is inconsistent with visible count and claim constraint."
    return None


def _count_satisfies(constraint: Mapping[str, Any], count: int) -> bool:
    operator = str(constraint["operator"])
    value = float(constraint["value"])
    if operator == "eq":
        return count == value
    if operator == "gte":
        return count >= value
    if operator == "lte":
        return count <= value
    if operator == "between":
        upper = constraint["upper_value"]
        if upper is None:
            return False
        return value <= count <= float(upper)
    return False


_HOLISTIC_SYSTEM = """Evaluate the overall visible scene against the original user prompt using
only the supplied RGB exploration portfolio. Score exactly four dimensions from 0 to 1:
global_prompt_alignment, composition_and_layout, style_atmosphere_coherence, and
completeness_and_polish. Cite one or more supplied opaque frame ids for every dimension.
Judge only visible pixels; do not infer camera pose, engine metadata, asset identity, or
unseen content. Require unambiguous pixel evidence for prompt-listed objects and do not
credit a functionally related or visually different category as the requested object;
missing or ambiguous elements must reduce alignment and completeness. Return the fixed
structured tool call once."""


def _holistic_schema() -> JSON:
    dimension = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_frame_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {"type": "string"},
            },
            "rationale": {"type": "string"},
        },
        "required": ["score", "evidence_frame_ids", "rationale"],
    }
    return {
        "name": _HOLISTIC_TOOL,
        "description": "Record four RGB-cited holistic scene scores.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "dimensions": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {name: dimension for name in _DIMENSION_NAMES},
                    "required": list(_DIMENSION_NAMES),
                },
                "summary": {"type": "string"},
            },
            "required": ["dimensions", "summary"],
        },
    }


class LLMHolisticJudge(_OneShotAdapter):
    """One-shot original-prompt + RGB holistic scorer."""

    request_prefix = "s3h_"
    call_kind = "holistic_judgment"
    tool_name = _HOLISTIC_TOOL

    def __init__(
        self,
        client: LLMClient,
        *,
        max_tokens: int = 768,
        manifest_path: str | Path | None = None,
        raw_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            client,
            max_tokens=max_tokens,
            manifest_path=manifest_path,
            raw_path=raw_path,
        )

    def judge(
        self,
        prompt: str,
        frames: Sequence[Stage3JudgeFrame],
    ) -> HolisticJudgeDecision:
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a non-empty string")
            original_prompt = prompt
            normalized = _normalize_frames(frames, maximum=6)
        except (TypeError, ValueError, OverflowError) as exc:
            return HolisticJudgeDecision(error=f"Holistic input rejected: {exc}")
        request_id = self._begin_request(normalized)
        response: Any = None
        try:
            response = self.client.chat(
                [
                    LLMMessage.text("system", _HOLISTIC_SYSTEM),
                    _image_message("Original scene prompt:\n" + original_prompt, normalized),
                ],
                [_holistic_schema()],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            result = _parse_holistic(
                response,
                available_ids={frame.frame_id for frame in normalized},
            )
            raw = _response_record(
                request_id, self.call_kind, "success", response=response
            )
        except Exception as exc:  # noqa: BLE001 - fail closed at provider boundary
            error = _safe_error(exc)
            result = HolisticJudgeDecision(
                transport_status="error",
                parse_status="not_attempted",
                error=error,
            )
            raw = _response_record(
                request_id, self.call_kind, "error", response=response, error=error
            )
        self._append_raw(raw)
        return result


def _parse_holistic(response: Any, *, available_ids: set[str]) -> HolisticJudgeDecision:
    def invalid(reason: str) -> HolisticJudgeDecision:
        return HolisticJudgeDecision(
            transport_status="success",
            parse_status="invalid",
            error=reason,
        )

    calls = list(getattr(response, "tool_calls", ()) or ())
    if len(calls) != 1 or str(getattr(calls[0], "name", "")) != _HOLISTIC_TOOL:
        return invalid("Holistic judge returned no unique score tool call.")
    raw = getattr(calls[0], "arguments", None)
    if not isinstance(raw, Mapping) or set(raw) != {"dimensions", "summary"}:
        return invalid("Holistic tool arguments did not match the strict schema.")
    dimensions_raw = raw["dimensions"]
    summary = raw["summary"]
    if (
        not isinstance(dimensions_raw, Mapping)
        or set(dimensions_raw) != set(_DIMENSION_NAMES)
        or not isinstance(summary, str)
        or not summary.strip()
    ):
        return invalid("Holistic judge returned invalid dimensions or summary.")
    dimensions: list[HolisticJudgeDimension] = []
    for name in _DIMENSION_NAMES:
        item = dimensions_raw[name]
        if not isinstance(item, Mapping) or set(item) != {
            "score",
            "evidence_frame_ids",
            "rationale",
        }:
            return invalid(f"Holistic dimension {name} did not match the strict schema.")
        score = item["score"]
        ids = item["evidence_frame_ids"]
        rationale = item["rationale"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or not isinstance(ids, list)
            or not 1 <= len(ids) <= 6
            or any(not isinstance(frame_id, str) for frame_id in ids)
            or len(ids) != len(set(ids))
            or not set(ids).issubset(available_ids)
            or not isinstance(rationale, str)
            or not rationale.strip()
        ):
            return invalid(f"Holistic dimension {name} contained invalid score or citation data.")
        dimensions.append(
            HolisticJudgeDimension(
                name=name,
                score=float(score),
                evidence_frame_ids=tuple(ids),
                rationale=rationale,
            )
        )
    overall = math.fsum(
        dimension.score * _DIMENSION_WEIGHTS[dimension.name]
        for dimension in dimensions
    )
    return HolisticJudgeDecision(
        dimensions=tuple(dimensions),
        overall_score=overall,
        summary=summary,
        transport_status="success",
        parse_status="valid",
    )


__all__ = [
    "HolisticFrameSelection",
    "HolisticJudgeDecision",
    "HolisticJudgeDimension",
    "IdentityDecision",
    "IdentityVerdict",
    "LLMHolisticFrameSelector",
    "LLMHolisticJudge",
    "LLMStage3IdentityJudge",
    "LLMStage3UnknownJudge",
    "Stage3JudgeFrame",
    "Stage3UnknownDecision",
]
