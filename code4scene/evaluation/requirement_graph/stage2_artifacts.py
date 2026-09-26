"""Atomic, auditable artifact publication for RequirementGraph Stage 2.

The writer deliberately keeps the Stage 2 trust domains separate:

* controller acquisition metadata is written to the capture index and the
  explicit ``stage2_capture_audit.json``;
* neutral frame metadata is written to ``stage2_judge_visible_frames.json``;
* visual evidence is written as opaque-id PNG files;
* semantic decisions and the metadata-free VLM request audit are written to
  their own files.

In particular, this module never serializes a ``FrameRecord`` together with
its RGB: the ``FrameStore`` supplies a pixel-free controller index, while PNG
bytes are published via an independent, opaque filename path.
"""

from __future__ import annotations

import dataclasses
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
from typing import Any

import numpy as np
from PIL import Image

from .stage2_frames import RGB_HASH_ALGORITHM, FrameStore

_OPAQUE_FRAME_ID = re.compile(r"\As2f_[0-9]{6}\Z", re.ASCII)
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z", re.ASCII)
_MANIFEST_KEYS = frozenset(
    {
        "request_id",
        "call_id",
        "node_id",
        "task_id",
        "claim_payload_keys",
        "claim_payload_key_paths",
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
        "claim_payload_keys",
        "claim_payload_key_paths",
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
_FORBIDDEN_CLAIM_KEY_TOKENS = frozenset(
    {
        "actorid",
        "actorids",
        "actorlabel",
        "assetpath",
        "bounds",
        "camerapose",
        "classname",
        "metadata",
        "pose",
        "rank",
        "route",
        "routingplan",
        "score",
        "stage1",
        "unrealname",
    }
)
_VISUAL_CLAIM_KEYS = frozenset(
    {
        "arguments",
        "claim_text",
        "constraint",
        "entity_name",
        "node_type",
        "operator",
        "ordinal",
        "polarity",
        "predicate_name",
        "predicate_type",
        "role",
        "scopes",
        "upper_value",
        "value",
    }
)
_PIXEL_KEYS = frozenset({"rgb", "pixels", "pixeldata", "encodedimage", "imagebase64"})


def _jsonable(value: Any, *, path: str = "$", seen: set[int] | None = None) -> Any:
    """Convert evaluator, Enum, Path, and NumPy values to strict JSON values."""

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
        value, (Mapping, Sequence, set, frozenset, np.ndarray)
    )
    value_id = id(value)
    if track_identity:
        if value_id in seen:
            raise ValueError(f"{path} contains a recursive value")
        seen.add(value_id)
    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                item.name: _jsonable(
                    getattr(value, item.name),
                    path=f"{path}.{item.name}",
                    seen=seen,
                )
                for item in dataclasses.fields(value)
            }
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


def _json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    return (
        json.dumps(
            value,
            # JSONL must be physically one record per line. Escaping non-ASCII
            # in compact mode prevents U+0085/U+2028/U+2029 from acting as
            # additional line boundaries in downstream tooling.
            ensure_ascii=not pretty,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace one file atomically, leaving any previous target intact on error."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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


def _opaque_frame_id(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _OPAQUE_FRAME_ID.fullmatch(value) is None:
        raise ValueError(f"{path} must be an opaque s2f_XXXXXX frame id")
    if int(value.removeprefix("s2f_")) < 1:
        raise ValueError(f"{path} must be a generated non-zero frame id")
    return value


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _record_id(record: Mapping[str, Any], *, path: str) -> str:
    if "request_id" not in record:
        raise ValueError(f"{path} must contain request_id")
    value = record["request_id"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.request_id must be a non-empty string")
    request_id = value.strip()
    if "call_id" in record:
        call_id = record["call_id"]
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError(f"{path}.call_id must be a non-empty string")
        if call_id.strip() != request_id:
            raise ValueError(f"{path}.call_id must equal request_id")
    return request_id


def _normalized_key_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _validate_claim_key(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    segments = [item for item in re.split(r"[/.[\]]+", value) if item]
    for segment in segments:
        token = _normalized_key_token(segment)
        if token in _FORBIDDEN_CLAIM_KEY_TOKENS:
            raise ValueError(f"{path} exposes forbidden metadata key {segment!r}")
        if not segment.isdigit() and segment not in _VISUAL_CLAIM_KEYS:
            raise ValueError(f"{path} contains unknown visual claim key {segment!r}")
    return value


def _validate_manifest(
    value: Any,
    *,
    frame_store: FrameStore,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    payload = _jsonable(value, path="request_manifest")
    if not isinstance(payload, list):
        raise TypeError("request_manifest must be a sequence of records")
    known_frames = {record.frame_id: record for record in frame_store.records}
    records: list[dict[str, Any]] = []
    request_ids: list[str] = []
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
        for identifier_name in ("node_id", "task_id"):
            if identifier_name in record and (
                not isinstance(record[identifier_name], str)
                or not record[identifier_name].strip()
            ):
                raise ValueError(f"{path}.{identifier_name} must be a non-empty string")

        claim_keys = record["claim_payload_keys"]
        claim_paths = record["claim_payload_key_paths"]
        if not isinstance(claim_keys, list) or not isinstance(claim_paths, list):
            raise TypeError(f"{path} claim payload key audit values must be lists")
        for key_index, key in enumerate(claim_keys):
            _validate_claim_key(key, path=f"{path}.claim_payload_keys[{key_index}]")
        for key_index, key_path in enumerate(claim_paths):
            _validate_claim_key(
                key_path,
                path=f"{path}.claim_payload_key_paths[{key_index}]",
            )
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
        if len(normalized_ids) > 4:
            raise ValueError(f"{path}.frame_ids exceeds the Stage 2 evidence cap")
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
            try:
                stored = known_frames[frame_id]
            except KeyError:
                raise ValueError(
                    f"{image_path}.frame_id does not exist in frame_store"
                ) from None
            if image["hash_algorithm"] != RGB_HASH_ALGORITHM:
                raise ValueError(f"{image_path}.hash_algorithm is unsupported")
            if image["sha256"] != stored.image_hash:
                raise ValueError(f"{image_path}.sha256 does not match stored RGB")
            if _positive_int(image["height"], path=f"{image_path}.height") != int(
                stored.rgb.shape[0]
            ):
                raise ValueError(f"{image_path}.height does not match stored RGB")
            if _positive_int(image["width"], path=f"{image_path}.width") != int(
                stored.rgb.shape[1]
            ):
                raise ValueError(f"{image_path}.width does not match stored RGB")
            if image["channels"] != 3:
                raise ValueError(f"{image_path}.channels must be 3")
        if image_ids != normalized_ids:
            raise ValueError(f"{path}.images must match frame_ids in order")

        if record["message_roles"] != ["system", "user"]:
            raise ValueError(f"{path}.message_roles must be ['system', 'user']")
        if not isinstance(record["model"], str) or not record["model"].strip():
            raise ValueError(f"{path}.model must be a non-empty string")
        if not isinstance(record["tool_name"], str) or not record["tool_name"].strip():
            raise ValueError(f"{path}.tool_name must be a non-empty string")
        if record["schema_version"] != "1.0":
            raise ValueError(f"{path}.schema_version must be '1.0'")
        records.append(record)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request_manifest request ids must be unique")

    serialized = _json_bytes(records).decode("utf-8").casefold()
    if "data:image" in serialized or "base64," in serialized:
        raise ValueError("request_manifest must not contain image payloads or base64")
    return records, tuple(request_ids)


def _validate_raw_records(
    value: Any,
    *,
    manifest_request_ids: tuple[str, ...],
) -> tuple[list[dict[str, Any]], bytes]:
    payload = _jsonable(value, path="raw_records")
    if not isinstance(payload, list):
        raise TypeError("raw_records must be a sequence of records")
    request_ids: list[str] = []
    records_by_request: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(payload):
        path = f"raw_records[{index}]"
        if not isinstance(record, dict):
            raise TypeError(f"{path} must be an object")
        if record.get("status") not in {"success", "error"}:
            raise ValueError(f"{path}.status must be 'success' or 'error'")
        request_id = _record_id(record, path=path)
        request_ids.append(request_id)
        records_by_request[request_id] = record
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("raw_records request ids must be unique")
    if set(request_ids) != set(manifest_request_ids):
        raise ValueError("raw_records must join request_manifest one-to-one")
    records = [records_by_request[request_id] for request_id in manifest_request_ids]
    lines = [_json_bytes(record, pretty=False) for record in records]
    return records, b"".join(lines)


def _reject_pixel_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            token = _normalized_key_token(str(key))
            if token in _PIXEL_KEYS:
                raise ValueError(f"{path} must not embed pixel field {key!r}")
            _reject_pixel_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_pixel_keys(item, path=f"{path}[{index}]")


def _png_bytes(rgb: np.ndarray, *, frame_id: str) -> bytes:
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"frame {frame_id} must contain uint8 HxWx3 RGB")
    owned = np.array(array, dtype=np.uint8, order="C", copy=True)
    buffer = io.BytesIO()
    Image.fromarray(owned, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)) and int(value) >= 0:
        return int(value)
    return None


def _retrieval_call_count(identity_retrieval: Any, result: Any) -> int:
    diagnostics = _field(result, "diagnostics", {})
    for container in (diagnostics, identity_retrieval):
        for name in (
            "retrieval_backend_call_count",
            "semantic_backend_call_count",
            "backend_call_count",
        ):
            count = _nonnegative_int(_field(container, name))
            if count is not None:
                return count
    return int(
        bool(
            _field(identity_retrieval, "semantic_backend")
            or _field(identity_retrieval, "semantic_backend_error")
        )
    )


def _summary(
    *,
    tasks: Sequence[Any],
    identity_retrieval: Any,
    capture_plan: Any,
    frame_index: Mapping[str, Any],
    request_manifest: Sequence[Mapping[str, Any]],
    result: Any,
) -> dict[str, Any]:
    assessments = tuple(_field(result, "assessments", ()) or ())
    verdict_counts = Counter(
        str(
            getattr(
                _field(item, "verdict", "UNKNOWN"),
                "value",
                _field(item, "verdict", "UNKNOWN"),
            )
        ).upper()
        for item in assessments
    )
    for verdict in ("MATCH", "MISMATCH", "UNKNOWN"):
        verdict_counts.setdefault(verdict, 0)
    unknown_reason_counts = Counter(
        str(_field(item, "unknown_reason"))
        for item in assessments
        if str(
            getattr(
                _field(item, "verdict", "UNKNOWN"),
                "value",
                _field(item, "verdict", "UNKNOWN"),
            )
        ).upper()
        == "UNKNOWN"
        and _field(item, "unknown_reason")
    )
    mismatch_downgraded_count = sum(
        bool(_field(item, "mismatch_downgraded", False)) for item in assessments
    )

    # The same-session runner publishes the controller plan together with its
    # execution outcomes.  Keep accepting a bare Stage2CapturePlan for the
    # public writer API and unwrap the richer runner artifact when present.
    plan = _field(capture_plan, "plan", capture_plan)
    programs = tuple(_field(plan, "programs", ()) or ())
    requests = tuple(
        request
        for program in programs
        for request in (_field(program, "requests", ()) or ())
    )
    planned_recovery = sum(
        int(_field(request, "recovery_step", 0) or 0) > 0 for request in requests
    )
    planned_overview = sum(
        str(
            getattr(
                _field(request, "shot_role", ""),
                "value",
                _field(request, "shot_role", ""),
            )
        )
        in {"overview", "grid"}
        and int(_field(request, "recovery_step", 0) or 0) == 0
        for request in requests
    )
    planned_targeted = len(requests) - planned_overview - planned_recovery
    budget = _jsonable(_field(plan, "budget", {}), path="capture_plan.budget")
    if not isinstance(budget, dict):
        budget = {}

    rejection_counts = dict(frame_index.get("rejection_counts", {}))
    duplicate_count = int(rejection_counts.get("near_duplicate", 0))
    rejected_count = int(frame_index.get("rejected_frame_count", 0))
    valid_count = int(frame_index.get("valid_frame_count", 0))
    invalid_count = max(0, rejected_count - duplicate_count)
    valid_cap = int(frame_index.get("valid_frame_hard_cap", 0))
    query_count = len(tuple(_field(identity_retrieval, "queries", ()) or ()))
    retrieval_calls = _retrieval_call_count(identity_retrieval, result)
    vlm_calls = len(request_manifest)
    diagnostics = _field(result, "diagnostics", {})
    capture_attempt_count = _nonnegative_int(
        _field(diagnostics, "capture_attempt_count")
    )
    recovery_attempt_count = _nonnegative_int(
        _field(diagnostics, "recovery_attempt_count")
    )
    judge_attempt_count = _nonnegative_int(_field(diagnostics, "judge_attempt_count"))
    return {
        "schema_version": "1.0",
        "task_count": len(tasks),
        "assessment_count": len(assessments),
        "identity_query_count": query_count,
        "retrieval_backend_call_count": retrieval_calls,
        "vlm_call_count": vlm_calls,
        "capture_attempt_count": capture_attempt_count,
        "recovery_attempt_count": recovery_attempt_count,
        "judge_attempt_count": judge_attempt_count,
        "valid_frame_count": valid_count,
        "rejected_frame_count": rejected_count,
        "invalid_frame_count": invalid_count,
        "duplicate_frame_count": duplicate_count,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "mismatch_downgraded_count": mismatch_downgraded_count,
        "unknown_reason_counts": dict(sorted(unknown_reason_counts.items())),
        "budget_usage": {
            "limits": budget,
            "planned_overview_frames": planned_overview,
            "planned_targeted_frames": planned_targeted,
            "planned_recovery_frames": planned_recovery,
            "valid_frames": valid_count,
            "remaining_valid_frame_capacity": max(0, valid_cap - valid_count),
        },
        "evaluation_error": _field(result, "evaluation_error"),
    }


def _prepare_frames_directory(frames_dir: Path, expected_names: set[str]) -> None:
    if frames_dir.is_symlink():
        raise ValueError("frames directory must not be a symbolic link")
    frames_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name for path in frames_dir.iterdir() if path.name not in expected_names
    )
    if unexpected:
        raise FileExistsError(
            "frames directory contains files not owned by this FrameStore: "
            f"{unexpected!r}"
        )


def _capture_metadata_payloads(
    *,
    frame_store: FrameStore,
    routing_plan: Any,
    requirement_ids_by_task: Mapping[str, Sequence[str]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build separate controller-audit and metadata-minimal judge indexes."""

    requirement_map: dict[str, tuple[str, ...]] = {}
    for raw_task_id, raw_requirement_ids in (
        requirement_ids_by_task or {}
    ).items():
        task_id = str(raw_task_id).strip()
        if not task_id:
            raise ValueError("requirement_ids_by_task keys must be non-empty")
        requirement_ids = tuple(
            dict.fromkeys(str(value).strip() for value in raw_requirement_ids)
        )
        if any(not value for value in requirement_ids):
            raise ValueError(
                "requirement_ids_by_task values must contain non-empty ids"
            )
        requirement_map[task_id] = requirement_ids

    targets: dict[str, tuple[str, Any]] = {}
    for route in tuple(_field(routing_plan, "routes", ()) or ()):
        for target in tuple(_field(route, "targets", ()) or ()):
            actor_id = str(_field(target, "actor_id", "")).strip()
            bounds = _field(target, "bounds")
            if actor_id and bounds is not None:
                targets.setdefault(actor_id.casefold(), (actor_id, bounds))

    audit_frames: list[dict[str, Any]] = []
    visible_frames: list[dict[str, str]] = []
    for record in frame_store.records:
        requirement_ids = tuple(
            dict.fromkeys(
                requirement_id
                for task_id in record.task_ids
                for requirement_id in requirement_map.get(task_id, ())
            )
        )
        canonical_actor_ids: list[str] = []
        target_bounds: dict[str, Any] = {}
        for raw_actor_id in record.actor_ids:
            canonical, bounds = targets.get(
                raw_actor_id.casefold(), (raw_actor_id, None)
            )
            canonical_actor_ids.append(canonical)
            if bounds is not None:
                target_bounds[canonical] = _jsonable(
                    bounds, path=f"capture_audit.bounds.{canonical}"
                )
        if canonical_actor_ids:
            capture_mode = "actor_focus"
            view_type = "targeted"
        elif record.shot_role.value == "grid" and record.task_ids:
            capture_mode = "requirement_grid"
            view_type = "grid"
        elif record.shot_role.value == "overview":
            capture_mode = "scene_overview"
            view_type = "overview"
        else:
            capture_mode = "scene_context"
            view_type = "context"
        audit_frames.append(
            {
                "frame_id": record.frame_id,
                "requirement_ids": list(requirement_ids),
                "task_ids": list(record.task_ids),
                "capture_target_actor_ids": canonical_actor_ids,
                "capture_target_bounds": target_bounds,
                "capture_mode": capture_mode,
                "shot_role": record.shot_role.value,
            }
        )
        visible_frames.append(
            {
                "frame_id": record.frame_id,
                "channel": "rgb",
                "view_type": view_type,
            }
        )

    audit_payload = {"schema_version": "1.0", "frames": audit_frames}
    visible_payload = {"schema_version": "1.0", "frames": visible_frames}
    _reject_pixel_keys(audit_payload, path="capture_audit")
    _reject_pixel_keys(visible_payload, path="judge_visible_frames")
    return audit_payload, visible_payload


def write_stage2_artifacts(
    output_dir: str | Path,
    *,
    tasks: Sequence[Any],
    identity_retrieval: Any,
    routing_plan: Any,
    capture_plan: Any,
    frame_store: FrameStore,
    evidence_selection: Any,
    request_manifest: Sequence[Mapping[str, Any]],
    raw_records: Sequence[Mapping[str, Any]],
    result: Any,
    requirement_ids_by_task: Mapping[str, Sequence[str]] | None = None,
) -> tuple[Path, ...]:
    """Atomically publish the complete Stage 2 artifact contract.

    The returned tuple has a stable order. Individual PNGs are named only by
    store-generated opaque frame IDs; controller audit metadata and the
    metadata-minimal judge-visible frame index are separate JSON artifacts.
    """

    if not isinstance(frame_store, FrameStore):
        raise TypeError("frame_store must be a FrameStore")
    try:
        task_values = tuple(tasks)
    except TypeError as exc:
        raise TypeError("tasks must be a sequence") from exc
    root = Path(output_dir).expanduser()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(str(root))

    task_payload = _jsonable(task_values, path="tasks")
    retrieval_payload = _jsonable(identity_retrieval, path="identity_retrieval")
    routing_payload = _jsonable(routing_plan, path="routing_plan")
    capture_payload = _jsonable(capture_plan, path="capture_plan")
    frame_index = _jsonable(frame_store.frame_index(), path="frame_index")
    evidence_payload = _jsonable(evidence_selection, path="evidence_selection")
    result_payload = _jsonable(result, path="result")
    audit_payload, visible_payload = _capture_metadata_payloads(
        frame_store=frame_store,
        routing_plan=routing_plan,
        requirement_ids_by_task=requirement_ids_by_task,
    )
    _reject_pixel_keys(frame_index, path="frame_index")
    _reject_pixel_keys(evidence_payload, path="evidence_selection")
    manifest_payload, request_ids = _validate_manifest(
        request_manifest,
        frame_store=frame_store,
    )
    _, raw_payload = _validate_raw_records(
        raw_records,
        manifest_request_ids=request_ids,
    )
    summary_payload = _jsonable(
        _summary(
            tasks=task_values,
            identity_retrieval=identity_retrieval,
            capture_plan=capture_plan,
            frame_index=frame_index,
            request_manifest=manifest_payload,
            result=result,
        ),
        path="summary",
    )

    frame_payloads: dict[str, bytes] = {}
    for record in frame_store.records:
        frame_id = _opaque_frame_id(
            record.frame_id, path="frame_store.records.frame_id"
        )
        if frame_id in frame_payloads:
            raise ValueError(f"duplicate FrameStore frame id {frame_id!r}")
        frame_payloads[frame_id] = _png_bytes(record.rgb, frame_id=frame_id)

    paths = {
        "tasks": root / "stage2_tasks.json",
        "retrieval": root / "stage2_identity_retrieval.json",
        "routing": root / "stage2_routing_plan.json",
        "capture": root / "stage2_capture_plan.json",
        "frame_index": root / "frame_index.json",
        "audit": root / "stage2_capture_audit.json",
        "judge_visible": root / "stage2_judge_visible_frames.json",
        "frames": root / "frames",
        "evidence": root / "stage2_evidence_selection.json",
        "manifest": root / "stage2_vlm_request_manifest.json",
        "raw": root / "stage2_vlm_raw.jsonl",
        "result": root / "stage2_result.json",
        "summary": root / "stage2_summary.json",
    }
    expected_frame_names = {f"{frame_id}.png" for frame_id in frame_payloads}
    _prepare_frames_directory(paths["frames"], expected_frame_names)

    # Frames precede their index, and summary is the final commit marker.  Each
    # individual path is nevertheless published via same-directory replace.
    for frame_id, payload in frame_payloads.items():
        _atomic_write_bytes(paths["frames"] / f"{frame_id}.png", payload)
    fixed_payloads = (
        (paths["tasks"], _json_bytes(task_payload)),
        (paths["retrieval"], _json_bytes(retrieval_payload)),
        (paths["routing"], _json_bytes(routing_payload)),
        (paths["capture"], _json_bytes(capture_payload)),
        (paths["frame_index"], _json_bytes(frame_index)),
        (paths["audit"], _json_bytes(audit_payload)),
        (paths["judge_visible"], _json_bytes(visible_payload)),
        (paths["evidence"], _json_bytes(evidence_payload)),
        (paths["manifest"], _json_bytes(manifest_payload)),
        (paths["raw"], raw_payload),
        (paths["result"], _json_bytes(result_payload)),
        (paths["summary"], _json_bytes(summary_payload)),
    )
    for path, payload in fixed_payloads:
        _atomic_write_bytes(path, payload)

    return tuple(paths.values())


__all__ = ["write_stage2_artifacts"]
