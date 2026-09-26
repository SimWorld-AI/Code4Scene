"""Atomic, metadata-safe publication for RequirementGraph Stage 3.

Stage 3 keeps controller exploration metadata, the explicit capture audit,
neutral judge-visible frame metadata, RGB files, metadata-free VLM request
audits, and semantic results in separate artifacts. ``graph_summary`` is
written last and acts as the commit marker for a complete publication.

The implementation intentionally uses the public, duck-typed ``FrameStore``
surface (``records`` and ``frame_index()``).  This keeps the writer usable by
the Stage 3 namespaced store without coupling it to judge or orchestration
contracts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from .stage2_frames import RGB_HASH_ALGORITHM

_OPAQUE_FRAME_ID = re.compile(r"\As3f_[0-9]{6}\Z", re.ASCII)
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z", re.ASCII)
_CALL_KINDS = frozenset(
    {
        "binary_resolution",
        "frame_selection",
        "unknown_resolution",
        "holistic_judgment",
    }
)
_BINARY_RESOLUTION_IMAGE_CAP = 10
_MANIFEST_KEYS = frozenset(
    {
        "request_id",
        "call_id",
        "call_kind",
        "frame_ids",
        "images",
        "message_roles",
        "model",
        "tool_name",
        "schema_version",
    }
)
_MANIFEST_REQUIRED_KEYS = frozenset(
    {
        "request_id",
        "call_kind",
        "frame_ids",
        "images",
        "message_roles",
        "model",
        "tool_name",
        "schema_version",
    }
)
_IMAGE_MANIFEST_KEYS = frozenset(
    {"frame_id", "hash_algorithm", "sha256", "height", "width", "channels"}
)
_RAW_KEYS = frozenset(
    {
        "request_id",
        "call_id",
        "call_kind",
        "status",
        "response",
        "error",
        "attempt_index",
        "retry_of",
        "parse_status",
        "validation_error",
    }
)
_BATCH_CLAIM_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "node_id",
        "claim_kind",
        "policy_id",
        "available_frame_ids",
        "selected_frame_ids",
        "omitted_frame_ids",
        "batches",
        "aggregation",
    }
)
_BATCH_CLAIM_OPTIONAL_KEYS = frozenset(
    {"binary_arbitration", "stable_verdict", "stop_reason"}
)
_BATCH_CLAIM_KEYS = _BATCH_CLAIM_REQUIRED_KEYS | _BATCH_CLAIM_OPTIONAL_KEYS
_BATCH_RECORD_REQUIRED_KEYS = frozenset(
    {
        "batch_index",
        "frame_ids",
        "request_id",
        "transport_status",
        "parse_status",
        "judge_verdict",
        "accepted_verdict",
        "confidence",
        "evidence_frame_ids",
        "reframe_recommended",
        "rationale",
        "evaluation_error",
    }
)
_BATCH_RECORD_KEYS = _BATCH_RECORD_REQUIRED_KEYS | {"attempt_request_ids"}
_BINARY_AUDIT_BASE_KEYS = frozenset(
    {
        "attempted",
        "available_frame_ids",
        "selected_frame_ids",
        "omitted_frame_ids",
        "request_id",
        "resolution",
    }
)
_BINARY_AUDIT_ATTEMPT_KEYS = frozenset(
    {"parse_status", "raw_verdict", "transport_status"}
)
_BINARY_AUDIT_OPTIONAL_KEYS = frozenset({"attempt_request_ids"})
_BATCH_STOP_REASONS = frozenset(
    {"evidence_exhausted", "judge_error", "safety_cap", "stable_verdict"}
)
_PIXEL_KEY_TOKENS = frozenset(
    {"rgb", "pixels", "pixeldata", "encodedimage", "imagebase64", "base64"}
)
_JUDGE_METADATA_KEY_TOKENS = (
    frozenset(
        {
            "actorid",
            "actorids",
            "actorlabel",
            "assetpath",
            "bounds",
            "camerapose",
            "image",
            "images",
            "input",
            "messages",
            "pose",
            "prompt",
            "requestbody",
            "requestmessages",
            "requestpayload",
            "retrieval",
            "routing",
            "routingplan",
        }
    )
    | _PIXEL_KEY_TOKENS
)


def _jsonable(value: Any, *, path: str = "$", seen: set[int] | None = None) -> Any:
    """Convert evaluator values to strict JSON without silently accepting NaN."""

    if isinstance(value, Enum):
        return _jsonable(value.value, path=path, seen=seen)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item(), path=path, seen=seen)

    if seen is None:
        seen = set()
    track_identity = dataclasses.is_dataclass(value) or isinstance(
        value,
        (Mapping, Sequence, set, frozenset, np.ndarray, SimpleNamespace),
    )
    value_id = id(value)
    if track_identity:
        if value_id in seen:
            raise ValueError(f"{path} contains a recursive value")
        seen.add(value_id)
    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: _jsonable(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}",
                    seen=seen,
                )
                for field in dataclasses.fields(value)
            }
        if isinstance(value, SimpleNamespace):
            return _jsonable(vars(value), path=path, seen=seen)
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                raw_key = key.value if isinstance(key, Enum) else key
                if isinstance(raw_key, np.generic):
                    raw_key = raw_key.item()
                text_key = str(raw_key)
                if text_key in result:
                    raise ValueError(
                        f"{path} contains mapping keys that collide as JSON strings"
                    )
                result[text_key] = _jsonable(
                    item,
                    path=f"{path}.{text_key}",
                    seen=seen,
                )
            return result
        if isinstance(value, np.ndarray):
            return _jsonable(value.tolist(), path=path, seen=seen)
        if isinstance(value, (set, frozenset)):
            converted = [_jsonable(item, path=f"{path}[]", seen=seen) for item in value]
            return sorted(
                converted,
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                ),
            )
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray, memoryview)
        ):
            return [
                _jsonable(item, path=f"{path}[{index}]", seen=seen)
                for index, item in enumerate(value)
            ]
        if isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(f"{path} contains bytes; binary data must not enter JSON")
        raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")
    finally:
        if track_identity:
            seen.remove(value_id)


def _artifact_payload(
    value: Any,
    *,
    path: str,
    exclude_keys: frozenset[str] = frozenset(),
) -> Any:
    converter = getattr(value, "to_dict", None)
    candidate = converter() if callable(converter) else value
    if isinstance(candidate, SimpleNamespace):
        candidate = vars(candidate)
    if isinstance(candidate, Mapping) and exclude_keys:
        candidate = {
            key: item for key, item in candidate.items() if str(key) not in exclude_keys
        }
    return _jsonable(candidate, path=path)


def _json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=not pretty,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalized_key_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _reject_keys(value: Any, *, path: str, forbidden: frozenset[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _normalized_key_token(str(key)) in forbidden:
                # Provider response envelopes may retain the name of a
                # sensitive field solely to prove that it was redacted.
                if item == "<redacted-request-data>":
                    continue
                raise ValueError(f"{path} exposes forbidden field {key!r}")
            _reject_keys(item, path=f"{path}.{key}", forbidden=forbidden)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_keys(item, path=f"{path}[{index}]", forbidden=forbidden)


def _reject_encoded_images(value: Any, *, path: str) -> None:
    serialized = _json_bytes(value).decode("utf-8").casefold()
    if (
        "data:image" in serialized
        or ";base64," in serialized
        or "base64," in serialized
    ):
        raise ValueError(f"{path} must not contain image payloads or base64")


def _opaque_frame_id(value: Any, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or _OPAQUE_FRAME_ID.fullmatch(value) is None
        or int(value.removeprefix("s3f_")) < 1
    ):
        raise ValueError(f"{path} must be an opaque generated s3f_XXXXXX frame id")
    return value


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _required_text(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _record_id(record: Mapping[str, Any], *, path: str) -> str:
    if "request_id" not in record:
        raise ValueError(f"{path} must contain request_id")
    request_id = _required_text(record["request_id"], path=f"{path}.request_id")
    if "call_id" in record:
        call_id = _required_text(record["call_id"], path=f"{path}.call_id")
        if call_id != request_id:
            raise ValueError(f"{path}.call_id must equal request_id")
    return request_id


def _rgb_array(value: Any, *, frame_id: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1] < 1
        or array.shape[2] != 3
    ):
        raise ValueError(f"frame {frame_id} must contain non-empty uint8 HxWx3 RGB")
    return np.array(array, dtype=np.uint8, order="C", copy=True)


def _png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _prepare_store(
    frame_store: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, bytes]]:
    records_value = getattr(frame_store, "records", None)
    if records_value is None:
        raise TypeError("frame_store must expose records")
    try:
        records = tuple(records_value)
    except TypeError as exc:
        raise TypeError("frame_store.records must be iterable") from exc
    index_method = getattr(frame_store, "frame_index", None)
    if not callable(index_method):
        raise TypeError("frame_store must expose frame_index()")
    frame_index = _jsonable(index_method(), path="frame_index")
    if not isinstance(frame_index, dict):
        raise TypeError("frame_store.frame_index() must return an object")
    _reject_keys(frame_index, path="frame_index", forbidden=_PIXEL_KEY_TOKENS)
    _reject_encoded_images(frame_index, path="frame_index")

    metadata: dict[str, dict[str, Any]] = {}
    png_payloads: dict[str, bytes] = {}
    for index, record in enumerate(records):
        path = f"frame_store.records[{index}]"
        frame_id = _opaque_frame_id(_field(record, "frame_id"), path=f"{path}.frame_id")
        if frame_id in metadata:
            raise ValueError(f"duplicate frame_store frame id {frame_id!r}")
        task_ids = tuple(_field(record, "task_ids", ()) or ())
        actor_ids = tuple(_field(record, "actor_ids", ()) or ())
        if task_ids or actor_ids:
            raise ValueError(
                f"{path} must not retain Stage 2 task or actor associations"
            )
        rgb = _rgb_array(_field(record, "rgb"), frame_id=frame_id)
        image_hash = hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
        stored_hash = _field(record, "image_hash")
        if not isinstance(stored_hash, str) or stored_hash != image_hash:
            raise ValueError(f"{path}.image_hash does not match stored RGB")
        metadata[frame_id] = {
            "frame_id": frame_id,
            "hash_algorithm": RGB_HASH_ALGORITHM,
            "sha256": image_hash,
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "channels": 3,
        }
        png_payloads[frame_id] = _png_bytes(rgb)

    raw_index_records = frame_index.get("frames")
    if not isinstance(raw_index_records, list):
        raise TypeError("frame_index.frames must be a list")
    index_ids: list[str] = []
    for index, record in enumerate(raw_index_records):
        path = f"frame_index.frames[{index}]"
        if not isinstance(record, dict):
            raise TypeError(f"{path} must be an object")
        frame_id = _opaque_frame_id(record.get("frame_id"), path=f"{path}.frame_id")
        index_ids.append(frame_id)
        if record.get("image_hash") != metadata.get(frame_id, {}).get("sha256"):
            raise ValueError(f"{path}.image_hash does not match stored RGB")
        if record.get("hash_algorithm") != RGB_HASH_ALGORITHM:
            raise ValueError(f"{path}.hash_algorithm is unsupported")
        if tuple(record.get("task_ids", ()) or ()) or tuple(
            record.get("actor_ids", ()) or ()
        ):
            raise ValueError(f"{path} exposes Stage 2 task or actor associations")
    if index_ids != list(metadata):
        raise ValueError("frame_index.frames must match frame_store.records in order")
    if frame_index.get("valid_frame_count") != len(metadata):
        raise ValueError("frame_index.valid_frame_count does not match records")
    if frame_index.get("hash_algorithm") != RGB_HASH_ALGORITHM:
        raise ValueError("frame_index.hash_algorithm is unsupported")
    return frame_index, metadata, png_payloads


def _validate_manifest(
    value: Any,
    *,
    known_frames: Mapping[str, Mapping[str, Any]],
    unknown_image_cap: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...], dict[str, str]]:
    payload = _jsonable(value, path="request_manifest")
    if not isinstance(payload, list):
        raise TypeError("request_manifest must be a sequence of records")
    records: list[dict[str, Any]] = []
    request_ids: list[str] = []
    kinds_by_request: dict[str, str] = {}
    for index, record in enumerate(payload):
        path = f"request_manifest[{index}]"
        if not isinstance(record, dict):
            raise TypeError(f"{path} must be an object")
        unknown = set(record) - _MANIFEST_KEYS
        missing = _MANIFEST_REQUIRED_KEYS - set(record)
        if unknown:
            raise ValueError(f"{path} contains forbidden keys {sorted(unknown)!r}")
        if missing:
            raise ValueError(f"{path} is missing keys {sorted(missing)!r}")
        request_id = _record_id(record, path=path)
        request_ids.append(request_id)
        call_kind = _required_text(record["call_kind"], path=f"{path}.call_kind")
        if call_kind not in _CALL_KINDS:
            raise ValueError(f"{path}.call_kind must be one of {sorted(_CALL_KINDS)!r}")
        kinds_by_request[request_id] = call_kind

        frame_ids = record["frame_ids"]
        images = record["images"]
        if not isinstance(frame_ids, list) or not isinstance(images, list):
            raise TypeError(f"{path}.frame_ids and images must be lists")
        normalized_ids = [
            _opaque_frame_id(frame_id, path=f"{path}.frame_ids[{frame_index}]")
            for frame_index, frame_id in enumerate(frame_ids)
        ]
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError(f"{path}.frame_ids must be unique")
        image_cap = {
            "binary_resolution": _BINARY_RESOLUTION_IMAGE_CAP,
            "frame_selection": 10,
            "unknown_resolution": unknown_image_cap,
            "holistic_judgment": 6,
        }[call_kind]
        if len(normalized_ids) > image_cap:
            raise ValueError(f"{path}.frame_ids exceeds the Stage 3 image cap")
        image_ids: list[str] = []
        for image_index, image in enumerate(images):
            image_path = f"{path}.images[{image_index}]"
            if not isinstance(image, dict) or set(image) != _IMAGE_MANIFEST_KEYS:
                raise ValueError(
                    f"{image_path} must contain exactly "
                    f"{sorted(_IMAGE_MANIFEST_KEYS)!r}"
                )
            frame_id = _opaque_frame_id(
                image["frame_id"], path=f"{image_path}.frame_id"
            )
            image_ids.append(frame_id)
            expected = known_frames.get(frame_id)
            if expected is None:
                raise ValueError(f"{image_path}.frame_id does not exist in frame_store")
            if image["hash_algorithm"] != RGB_HASH_ALGORITHM:
                raise ValueError(f"{image_path}.hash_algorithm is unsupported")
            if image["sha256"] != expected["sha256"]:
                raise ValueError(f"{image_path}.sha256 does not match stored RGB")
            if (
                _positive_int(image["height"], path=f"{image_path}.height")
                != expected["height"]
            ):
                raise ValueError(f"{image_path}.height does not match stored RGB")
            if (
                _positive_int(image["width"], path=f"{image_path}.width")
                != expected["width"]
            ):
                raise ValueError(f"{image_path}.width does not match stored RGB")
            if image["channels"] != 3:
                raise ValueError(f"{image_path}.channels must be 3")
        if image_ids != normalized_ids:
            raise ValueError(f"{path}.images must match frame_ids in order")
        if record["message_roles"] != ["system", "user"]:
            raise ValueError(f"{path}.message_roles must be ['system', 'user']")
        _required_text(record["model"], path=f"{path}.model")
        _required_text(record["tool_name"], path=f"{path}.tool_name")
        if record["schema_version"] != "1.0":
            raise ValueError(f"{path}.schema_version must be '1.0'")
        records.append(record)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_manifest request ids must be unique")
    _reject_encoded_images(records, path="request_manifest")
    return records, tuple(request_ids), kinds_by_request


def _validate_raw_records(
    value: Any,
    *,
    manifest_request_ids: tuple[str, ...],
    kinds_by_request: Mapping[str, str],
) -> bytes:
    payload = _jsonable(value, path="raw_records")
    if not isinstance(payload, list):
        raise TypeError("raw_records must be a sequence of records")
    request_ids: list[str] = []
    records_by_request: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(payload):
        path = f"raw_records[{index}]"
        if not isinstance(record, dict):
            raise TypeError(f"{path} must be an object")
        unknown = set(record) - _RAW_KEYS
        if unknown:
            raise ValueError(f"{path} contains forbidden keys {sorted(unknown)!r}")
        if record.get("status") not in {"success", "error"}:
            raise ValueError(f"{path}.status must be 'success' or 'error'")
        request_id = _record_id(record, path=path)
        request_ids.append(request_id)
        call_kind = _required_text(record.get("call_kind"), path=f"{path}.call_kind")
        expected_kind = kinds_by_request.get(request_id)
        if expected_kind is not None and call_kind != expected_kind:
            raise ValueError(f"{path}.call_kind must match request_manifest")
        if "attempt_index" in record:
            _positive_int(record["attempt_index"], path=f"{path}.attempt_index")
        if "retry_of" in record:
            retry_of = _required_text(record["retry_of"], path=f"{path}.retry_of")
            if retry_of == request_id:
                raise ValueError(f"{path}.retry_of must name an earlier request")
        if "parse_status" in record and record["parse_status"] not in {
            "not_attempted",
            "valid",
            "invalid",
        }:
            raise ValueError(f"{path}.parse_status is invalid")
        if "validation_error" in record:
            _required_text(
                record["validation_error"],
                path=f"{path}.validation_error",
            )
        _reject_keys(record, path=path, forbidden=_JUDGE_METADATA_KEY_TOKENS)
        _reject_encoded_images(record, path=path)
        records_by_request[request_id] = record
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("raw_records request ids must be unique")
    if set(request_ids) != set(manifest_request_ids):
        raise ValueError("raw_records must join request_manifest one-to-one")
    ordered = [records_by_request[request_id] for request_id in manifest_request_ids]
    return b"".join(_json_bytes(record, pretty=False) for record in ordered)


def _validate_batch_evaluations(
    value: Any,
    *,
    policy: Mapping[str, Any],
    known_frames: Mapping[str, Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    payload = _jsonable(value, path="batch_evaluations")
    if not isinstance(payload, list):
        raise TypeError("batch_evaluations must be a list")
    if policy.get("schema_version") != "1.0":
        raise ValueError("batch_policy.schema_version must be '1.0'")
    policy_id = _required_text(policy.get("policy_id"), path="batch_policy.policy_id")
    max_frames = _positive_int(
        policy.get("max_frames_per_request"),
        path="batch_policy.max_frames_per_request",
    )
    max_batches = _positive_int(
        policy.get("max_batches_per_claim"),
        path="batch_policy.max_batches_per_claim",
    )
    if max_frames > 10 or max_batches > 12:
        raise ValueError("batch_policy exceeds the Stage 3 safety contract")
    manifest_by_id = {
        str(record["request_id"]): record
        for record in manifest
        if record.get("call_kind") == "unknown_resolution"
    }
    binary_manifest_by_id = {
        str(record["request_id"]): record
        for record in manifest
        if record.get("call_kind") == "binary_resolution"
    }
    referenced_requests: list[str] = []
    referenced_binary_requests: list[str] = []
    task_ids: list[str] = []

    def frame_ids(raw: Any, *, path: str) -> list[str]:
        if not isinstance(raw, list):
            raise TypeError(f"{path} must be a list")
        result = [
            _opaque_frame_id(value, path=f"{path}[{index}]")
            for index, value in enumerate(raw)
        ]
        if len(result) != len(set(result)):
            raise ValueError(f"{path} must not contain duplicates")
        if any(value not in known_frames for value in result):
            raise ValueError(f"{path} references a frame outside frame_store")
        return result

    def attempt_request_ids(
        raw: Any,
        *,
        final_request_id: str,
        path: str,
    ) -> list[str]:
        if raw is None:
            return [final_request_id]
        if not isinstance(raw, list) or not raw:
            raise TypeError(f"{path} must be a non-empty list")
        result = [
            _required_text(value, path=f"{path}[{index}]")
            for index, value in enumerate(raw)
        ]
        if len(result) != len(set(result)):
            raise ValueError(f"{path} must not contain duplicates")
        if result[-1] != final_request_id:
            raise ValueError(f"{path} must end with request_id")
        return result

    for claim_index, claim in enumerate(payload):
        path = f"batch_evaluations[{claim_index}]"
        if not isinstance(claim, dict):
            raise TypeError(f"{path} must be an object")
        unknown_claim_keys = set(claim) - _BATCH_CLAIM_KEYS
        missing_claim_keys = _BATCH_CLAIM_REQUIRED_KEYS - set(claim)
        if unknown_claim_keys or missing_claim_keys:
            raise ValueError(
                f"{path} has invalid keys; missing={sorted(missing_claim_keys)!r}, "
                f"unknown={sorted(unknown_claim_keys)!r}"
            )
        if claim.get("schema_version") != "1.0":
            raise ValueError(f"{path}.schema_version must be '1.0'")
        task_id = _required_text(claim.get("task_id"), path=f"{path}.task_id")
        task_ids.append(task_id)
        _required_text(claim.get("node_id"), path=f"{path}.node_id")
        _required_text(claim.get("claim_kind"), path=f"{path}.claim_kind")
        if claim.get("policy_id") != policy_id:
            raise ValueError(f"{path}.policy_id must match batch_policy")
        available = frame_ids(
            claim.get("available_frame_ids"),
            path=f"{path}.available_frame_ids",
        )
        selected = frame_ids(
            claim.get("selected_frame_ids"),
            path=f"{path}.selected_frame_ids",
        )
        omitted = frame_ids(
            claim.get("omitted_frame_ids"),
            path=f"{path}.omitted_frame_ids",
        )
        if selected + omitted != available:
            raise ValueError(
                f"{path} selected and omitted frames must partition available frames"
            )
        raw_batches = claim.get("batches")
        if not isinstance(raw_batches, list) or len(raw_batches) > max_batches:
            raise ValueError(f"{path}.batches exceeds the configured batch count")
        flattened: list[str] = []
        for batch_offset, batch in enumerate(raw_batches, start=1):
            batch_path = f"{path}.batches[{batch_offset - 1}]"
            if not isinstance(batch, dict):
                raise ValueError(f"{batch_path} must be an object")
            unknown_batch_keys = set(batch) - _BATCH_RECORD_KEYS
            missing_batch_keys = _BATCH_RECORD_REQUIRED_KEYS - set(batch)
            if unknown_batch_keys or missing_batch_keys:
                raise ValueError(
                    f"{batch_path} has invalid keys; "
                    f"missing={sorted(missing_batch_keys)!r}, "
                    f"unknown={sorted(unknown_batch_keys)!r}"
                )
            if batch.get("batch_index") != batch_offset:
                raise ValueError(f"{batch_path}.batch_index must be contiguous")
            ids = frame_ids(batch.get("frame_ids"), path=f"{batch_path}.frame_ids")
            if not ids or len(ids) > max_frames:
                raise ValueError(f"{batch_path}.frame_ids exceeds the request cap")
            if not isinstance(batch.get("reframe_recommended"), bool):
                raise TypeError(
                    f"{batch_path}.reframe_recommended must be a bool"
                )
            flattened.extend(ids)
            evidence_ids = frame_ids(
                batch.get("evidence_frame_ids"),
                path=f"{batch_path}.evidence_frame_ids",
            )
            if not set(evidence_ids).issubset(ids):
                raise ValueError(f"{batch_path} cites evidence outside its request")
            request_id = _required_text(
                batch.get("request_id"),
                path=f"{batch_path}.request_id",
            )
            attempt_ids = attempt_request_ids(
                batch.get("attempt_request_ids"),
                final_request_id=request_id,
                path=f"{batch_path}.attempt_request_ids",
            )
            for attempt_id in attempt_ids:
                request = manifest_by_id.get(attempt_id)
                if request is None or request.get("frame_ids") != ids:
                    raise ValueError(
                        f"{batch_path}.attempt_request_ids must join matching "
                        "UNKNOWN requests"
                    )
            referenced_requests.extend(attempt_ids)
        # Adaptive reframe batches deliberately resend the original actor view
        # beside the new angle.  ``selected_frame_ids`` is therefore the
        # stable ordered union of request frames, not a disjoint partition.
        if list(dict.fromkeys(flattened)) != selected:
            raise ValueError(
                f"{path}.batches must cover selected_frame_ids in stable order"
            )
        if not isinstance(claim.get("aggregation"), Mapping):
            raise TypeError(f"{path}.aggregation must be an object")
        if "stop_reason" in claim:
            stop_reason = claim.get("stop_reason")
            if stop_reason not in _BATCH_STOP_REASONS:
                raise ValueError(
                    f"{path}.stop_reason must be one of "
                    f"{sorted(_BATCH_STOP_REASONS)!r}"
                )
        if "stable_verdict" in claim and claim.get("stable_verdict") not in {
            None,
            "MATCH",
            "MISMATCH",
        }:
            raise ValueError(
                f"{path}.stable_verdict must be MATCH, MISMATCH, or null"
            )
        binary = claim.get("binary_arbitration")
        if binary is not None:
            binary_path = f"{path}.binary_arbitration"
            if not isinstance(binary, dict):
                raise TypeError(f"{binary_path} must be an object or null")
            attempted = binary.get("attempted")
            if not isinstance(attempted, bool):
                raise TypeError(f"{binary_path}.attempted must be a bool")
            required_keys = _BINARY_AUDIT_BASE_KEYS | (
                _BINARY_AUDIT_ATTEMPT_KEYS if attempted else frozenset()
            )
            allowed_keys = required_keys | (
                _BINARY_AUDIT_OPTIONAL_KEYS if attempted else frozenset()
            )
            unknown_binary_keys = set(binary) - allowed_keys
            missing_binary_keys = required_keys - set(binary)
            if unknown_binary_keys or missing_binary_keys:
                raise ValueError(
                    f"{binary_path} has invalid keys; "
                    f"missing={sorted(missing_binary_keys)!r}, "
                    f"unknown={sorted(unknown_binary_keys)!r}"
                )
            binary_available = frame_ids(
                binary.get("available_frame_ids"),
                path=f"{binary_path}.available_frame_ids",
            )
            binary_selected = frame_ids(
                binary.get("selected_frame_ids"),
                path=f"{binary_path}.selected_frame_ids",
            )
            binary_omitted = frame_ids(
                binary.get("omitted_frame_ids"),
                path=f"{binary_path}.omitted_frame_ids",
            )
            if binary_selected + binary_omitted != binary_available:
                raise ValueError(
                    f"{binary_path} selected and omitted frames must partition "
                    "available frames"
                )
            if len(binary_selected) > _BINARY_RESOLUTION_IMAGE_CAP:
                raise ValueError(f"{binary_path}.selected_frame_ids exceeds the cap")
            if not isinstance(binary.get("resolution"), Mapping):
                raise TypeError(f"{binary_path}.resolution must be an object")
            if attempted:
                request_id = _required_text(
                    binary.get("request_id"),
                    path=f"{binary_path}.request_id",
                )
                attempt_ids = attempt_request_ids(
                    binary.get("attempt_request_ids"),
                    final_request_id=request_id,
                    path=f"{binary_path}.attempt_request_ids",
                )
                for attempt_id in attempt_ids:
                    request = binary_manifest_by_id.get(attempt_id)
                    if request is None or request.get("frame_ids") != binary_selected:
                        raise ValueError(
                            f"{binary_path}.attempt_request_ids must join matching "
                            "binary requests"
                        )
                referenced_binary_requests.extend(attempt_ids)
                if binary.get("raw_verdict") not in {
                    "MATCH",
                    "MISMATCH",
                    "UNKNOWN",
                }:
                    raise ValueError(f"{binary_path}.raw_verdict is invalid")
                _required_text(
                    binary.get("transport_status"),
                    path=f"{binary_path}.transport_status",
                )
                _required_text(
                    binary.get("parse_status"),
                    path=f"{binary_path}.parse_status",
                )
            elif binary.get("request_id") is not None:
                raise ValueError(
                    f"{binary_path}.request_id must be null when not attempted"
                )
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("batch_evaluations task ids must be unique")
    if len(referenced_requests) != len(set(referenced_requests)):
        raise ValueError("UNKNOWN request ids must join exactly one batch")
    if set(referenced_requests) != set(manifest_by_id):
        raise ValueError("every UNKNOWN request must join one batch evaluation")
    if len(referenced_binary_requests) != len(set(referenced_binary_requests)):
        raise ValueError("binary request ids must join exactly one arbitration")
    if set(referenced_binary_requests) != set(binary_manifest_by_id):
        raise ValueError("every binary request must join one binary arbitration")
    return payload


def _items(value: Any, *names: str) -> tuple[Any, ...]:
    candidate = value
    for name in names:
        selected = _field(value, name, None)
        if selected is not None:
            candidate = selected
            break
    if candidate is None:
        return ()
    if isinstance(candidate, Mapping):
        return (candidate,)
    if isinstance(candidate, Sequence) and not isinstance(
        candidate, (str, bytes, bytearray)
    ):
        return tuple(candidate)
    try:
        return tuple(candidate)
    except TypeError:
        return ()


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value))


def _summary(
    *,
    exploration: Any,
    frame_index: Mapping[str, Any],
    unknown_resolutions: Any,
    batch_evaluations: Sequence[Mapping[str, Any]],
    batch_policy: Mapping[str, Any],
    holistic_result: Any,
    final_result: Any,
    request_manifest: Sequence[Mapping[str, Any]],
    stage0_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolutions = _items(unknown_resolutions, "resolutions", "decisions", "assessments")
    final_assessments = _items(final_result, "leaf_assessments", "assessments")
    resolution_counts = Counter(
        _enum_text(
            _field(item, "final_verdict", _field(item, "verdict", "UNKNOWN"))
        ).upper()
        for item in resolutions
    )
    final_counts = Counter(
        _enum_text(
            _field(item, "final_verdict", _field(item, "verdict", "UNKNOWN"))
        ).upper()
        for item in final_assessments
    )
    for counter in (resolution_counts, final_counts):
        for verdict in ("MATCH", "MISMATCH", "UNKNOWN"):
            counter.setdefault(verdict, 0)
    sources = Counter(
        _enum_text(_field(item, "decision_source"))
        for item in final_assessments
        if _field(item, "decision_source") is not None
    )
    call_counts = Counter(str(record["call_kind"]) for record in request_manifest)
    for call_kind in sorted(_CALL_KINDS):
        call_counts.setdefault(call_kind, 0)

    requirements_score = _field(final_result, "requirements_score")
    holistic_score = _field(final_result, "holistic_overall_score")
    if holistic_score is None:
        holistic_score = _field(holistic_result, "overall_score")
    evaluation_error = _field(final_result, "evaluation_error")
    status = _field(final_result, "status")
    if status is None:
        status = "complete" if evaluation_error is None else "evaluation_error"
    result = {
        "schema_version": "1.0",
        "status": status,
        "evaluation_error": evaluation_error,
        "valid_frame_count": frame_index.get("valid_frame_count", 0),
        "rejected_frame_count": frame_index.get("rejected_frame_count", 0),
        "portfolio_frame_count": len(
            tuple(_field(exploration, "portfolio_frame_ids", ()) or ())
        ),
        "vlm_call_count": len(request_manifest),
        "vlm_call_counts": dict(sorted(call_counts.items())),
        "unknown_resolution_count": len(resolutions),
        "unknown_resolution_batch_count": sum(
            len(tuple(_field(value, "batches", ()) or ()))
            for value in batch_evaluations
        ),
        "batch_policy": dict(batch_policy),
        "unknown_resolution_verdict_counts": dict(sorted(resolution_counts.items())),
        "final_leaf_count": len(final_assessments),
        "final_verdict_counts": dict(sorted(final_counts.items())),
        "decision_source_counts": dict(sorted(sources.items())),
        "forced_mismatch_count": sum(
            bool(_field(item, "forced_mismatch", False)) for item in final_assessments
        ),
        "requirements_score": requirements_score,
        "holistic_overall_score": holistic_score,
        "holistic_status": _field(holistic_result, "status"),
        "holistic_error": _field(
            holistic_result,
            "evaluation_error",
            _field(holistic_result, "error"),
        ),
    }
    if stage0_summary is not None:
        result["stage0"] = dict(stage0_summary)
    return result


def _prepare_frames_directory(frames_dir: Path, expected_names: set[str]) -> None:
    if frames_dir.is_symlink():
        raise ValueError("stage3 frames directory must not be a symbolic link")
    frames_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name for path in frames_dir.iterdir() if path.name not in expected_names
    )
    if unexpected:
        raise FileExistsError(
            "stage3 frames directory contains files not owned by this FrameStore: "
            f"{unexpected!r}"
        )
    unsafe = sorted(
        path.name
        for path in frames_dir.iterdir()
        if path.is_symlink() or not path.is_file()
    )
    if unsafe:
        raise ValueError(f"stage3 frame targets must be regular files: {unsafe!r}")


def write_stage3_artifacts(
    output_dir: str | Path,
    *,
    exploration: Any,
    frame_store: Any,
    unknown_resolutions: Any,
    batch_evaluations: Sequence[Mapping[str, Any]],
    batch_policy: Any,
    holistic_result: Any,
    final_result: Any,
    request_manifest: Sequence[Mapping[str, Any]],
    raw_records: Sequence[Mapping[str, Any]],
    stage0_summary: Mapping[str, Any] | None = None,
    capture_audit: Mapping[str, Any] | None = None,
    judge_visible_frames: Mapping[str, Any] | None = None,
) -> tuple[Path, ...]:
    """Publish the complete Stage 3 contract and return paths in stable order."""

    root = Path(output_dir).expanduser()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(str(root))

    exploration_payload = _artifact_payload(
        exploration,
        path="exploration",
        exclude_keys=frozenset({"frame_store"}),
    )
    frame_index, known_frames, frame_payloads = _prepare_store(frame_store)
    audit_payload = _artifact_payload(
        capture_audit or {"schema_version": "1.0", "frames": []},
        path="capture_audit",
    )
    visible_payload = _artifact_payload(
        judge_visible_frames or {"schema_version": "1.0", "frames": []},
        path="judge_visible_frames",
    )
    if not isinstance(audit_payload, Mapping):
        raise TypeError("capture_audit must serialize to an object")
    if not isinstance(visible_payload, Mapping):
        raise TypeError("judge_visible_frames must serialize to an object")
    _reject_keys(audit_payload, path="capture_audit", forbidden=_PIXEL_KEY_TOKENS)
    _reject_encoded_images(audit_payload, path="capture_audit")
    _reject_keys(
        visible_payload,
        path="judge_visible_frames",
        forbidden=_JUDGE_METADATA_KEY_TOKENS,
    )
    _reject_encoded_images(visible_payload, path="judge_visible_frames")
    unknown_payload = _artifact_payload(unknown_resolutions, path="unknown_resolutions")
    policy_payload = _artifact_payload(batch_policy, path="batch_policy")
    if not isinstance(policy_payload, Mapping):
        raise TypeError("batch_policy must serialize to an object")
    unknown_image_cap = _positive_int(
        policy_payload.get("max_frames_per_request"),
        path="batch_policy.max_frames_per_request",
    )
    batch_payload = {
        "schema_version": "1.0",
        "policy": dict(policy_payload),
        "claims": _artifact_payload(
            batch_evaluations,
            path="batch_evaluations",
        ),
    }
    holistic_payload = _artifact_payload(holistic_result, path="holistic_result")
    final_payload = _artifact_payload(final_result, path="final_result")
    for path, payload in (
        ("exploration", exploration_payload),
        ("unknown_resolutions", unknown_payload),
        ("batch_evaluations", batch_payload),
        ("holistic_result", holistic_payload),
        ("final_result", final_payload),
    ):
        _reject_keys(payload, path=path, forbidden=_PIXEL_KEY_TOKENS)
        _reject_encoded_images(payload, path=path)
    for path, payload in (
        ("unknown_resolutions", unknown_payload),
        ("batch_evaluations", batch_payload),
        ("holistic_result", holistic_payload),
        ("final_result", final_payload),
    ):
        _reject_keys(payload, path=path, forbidden=_JUDGE_METADATA_KEY_TOKENS)

    manifest_payload, request_ids, kinds_by_request = _validate_manifest(
        request_manifest,
        known_frames=known_frames,
        unknown_image_cap=unknown_image_cap,
    )
    batch_payload["claims"] = _validate_batch_evaluations(
        batch_payload["claims"],
        policy=policy_payload,
        known_frames=known_frames,
        manifest=manifest_payload,
    )
    raw_payload = _validate_raw_records(
        raw_records,
        manifest_request_ids=request_ids,
        kinds_by_request=kinds_by_request,
    )
    normalized_stage0_summary: Mapping[str, Any] | None = None
    if stage0_summary is not None:
        candidate = _jsonable(stage0_summary, path="stage0_summary")
        if not isinstance(candidate, Mapping):
            raise TypeError("stage0_summary must be a mapping")
        _reject_keys(candidate, path="stage0_summary", forbidden=_PIXEL_KEY_TOKENS)
        _reject_encoded_images(candidate, path="stage0_summary")
        normalized_stage0_summary = candidate
    summary_payload = _jsonable(
        _summary(
            exploration=exploration,
            frame_index=frame_index,
            unknown_resolutions=unknown_resolutions,
            batch_evaluations=tuple(batch_payload["claims"]),
            batch_policy=policy_payload,
            holistic_result=holistic_result,
            final_result=final_result,
            request_manifest=manifest_payload,
            stage0_summary=normalized_stage0_summary,
        ),
        path="graph_summary",
    )

    paths = {
        "exploration": root / "stage3_exploration.json",
        "frame_index": root / "stage3_frame_index.json",
        "audit": root / "stage3_capture_audit.json",
        "judge_visible": root / "stage3_judge_visible_frames.json",
        "frames": root / "stage3_frames",
        "unknown": root / "stage3_unknown_resolutions.json",
        "batches": root / "stage3_batch_results.json",
        "holistic": root / "stage3_holistic_result.json",
        "manifest": root / "stage3_vlm_request_manifest.json",
        "raw": root / "stage3_vlm_raw.jsonl",
        "final": root / "final_graph_result.json",
        "summary": root / "graph_summary.json",
    }
    expected_frame_names = {f"{frame_id}.png" for frame_id in frame_payloads}
    _prepare_frames_directory(paths["frames"], expected_frame_names)

    for frame_id, payload in frame_payloads.items():
        _atomic_write_bytes(paths["frames"] / f"{frame_id}.png", payload)
    fixed_payloads = (
        (paths["exploration"], _json_bytes(exploration_payload)),
        (paths["frame_index"], _json_bytes(frame_index)),
        (paths["audit"], _json_bytes(audit_payload)),
        (paths["judge_visible"], _json_bytes(visible_payload)),
        (paths["unknown"], _json_bytes(unknown_payload)),
        (paths["batches"], _json_bytes(batch_payload)),
        (paths["holistic"], _json_bytes(holistic_payload)),
        (paths["manifest"], _json_bytes(manifest_payload)),
        (paths["raw"], raw_payload),
        (paths["final"], _json_bytes(final_payload)),
        # Commit marker: consumers should only trust a run with this file.
        (paths["summary"], _json_bytes(summary_payload)),
    )
    for path, payload in fixed_payloads:
        _atomic_write_bytes(path, payload)

    return tuple(paths.values())


__all__ = ["write_stage3_artifacts"]
