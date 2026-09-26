"""Atomic, metadata-safe publication for RequirementGraph Stage 0.

Stage 0 may either compile a graph with one structured model request or accept
an already-provided graph.  The artifact boundary deliberately consumes a
small duck-typed evaluation object so the compiler contracts can evolve
without teaching the runner how to serialize provider records.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_FORBIDDEN_LEDGER_KEYS = frozenset(
    {
        "request",
        "requests",
        "requestbody",
        "requestmessages",
        "messages",
        "authorization",
        "authorizationheader",
        "apikey",
        "base64",
        "imagebase64",
    }
)
_BASE64_BLOB = re.compile(
    r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{256,}={0,2}(?![A-Za-z0-9+/])"
)


def _jsonable(value: Any, *, path: str = "$", seen: set[int] | None = None) -> Any:
    """Convert values to strict JSON while rejecting recursion and NaN/Inf."""

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

    scalar = getattr(value, "item", None)
    if callable(scalar) and not isinstance(value, type):
        try:
            converted = scalar()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return _jsonable(converted, path=path, seen=seen)

    if seen is None:
        seen = set()
    track_identity = dataclasses.is_dataclass(value) or isinstance(
        value, (Mapping, Sequence, set, frozenset, SimpleNamespace)
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


def _to_dict_payload(value: Any, *, path: str) -> dict[str, Any]:
    converter = getattr(value, "to_dict", None)
    if not callable(converter):
        raise TypeError(f"{path} must expose to_dict()")
    payload = _jsonable(converter(), path=path)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}.to_dict() must return an object")
    return payload


def _to_mapping_payload(value: Any, *, path: str) -> dict[str, Any]:
    payload = _jsonable(value, path=path)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must be an object")
    return payload


def _json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    return (
        json.dumps(
            value,
            # JSONL must remain one physical record per line.  ASCII escaping
            # also protects U+0085/U+2028/U+2029 from downstream line readers.
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


def _normalized_key_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _is_forbidden_ledger_key(value: Any) -> bool:
    token = _normalized_key_token(value)
    return (
        token in _FORBIDDEN_LEDGER_KEYS
        or (token.startswith("request") and token != "requestid")
        or token.startswith("authorization")
        or "apikey" in token
        or "base64" in token
    )


def _reject_sensitive_ledger_data(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _is_forbidden_ledger_key(key):
                raise ValueError(f"{path} exposes forbidden field {key!r}")
            _reject_sensitive_ledger_data(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_sensitive_ledger_data(item, path=f"{path}[{index}]")

    if isinstance(value, str):
        lowered = value.casefold()
        if (
            "data:image" in lowered
            or ";base64," in lowered
            or _BASE64_BLOB.search(value)
        ):
            raise ValueError(f"{path} must not contain base64 or embedded payloads")


def _required_request_id(record: Mapping[str, Any], *, path: str) -> str:
    request_id = record.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError(f"{path}.request_id must be a non-empty string")
    normalized = request_id.strip()
    if "call_id" in record:
        call_id = record["call_id"]
        if not isinstance(call_id, str) or call_id.strip() != normalized:
            raise ValueError(f"{path}.call_id must equal request_id")
    return normalized


def _records(value: Any, *, path: str) -> list[dict[str, Any]]:
    payload = _jsonable(value, path=path)
    if not isinstance(payload, list):
        raise TypeError(f"{path} must be a sequence of records")
    records: list[dict[str, Any]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise TypeError(f"{path}[{index}] must be an object")
        records.append(record)
    return records


def _validate_ledgers(
    request_manifest: Any,
    raw_records: Any,
) -> tuple[list[dict[str, Any]], bytes]:
    manifest = _records(request_manifest, path="request_manifest")
    raw = _records(raw_records, path="raw_records")

    manifest_ids = [
        _required_request_id(record, path=f"request_manifest[{index}]")
        for index, record in enumerate(manifest)
    ]
    raw_ids = [
        _required_request_id(record, path=f"raw_records[{index}]")
        for index, record in enumerate(raw)
    ]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("request_manifest request ids must be unique")
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("raw_records request ids must be unique")
    if set(raw_ids) != set(manifest_ids):
        raise ValueError("raw_records must join request_manifest one-to-one")

    for index, record in enumerate(manifest):
        _reject_sensitive_ledger_data(record, path=f"request_manifest[{index}]")
    raw_by_id: dict[str, dict[str, Any]] = {}
    for index, (request_id, record) in enumerate(zip(raw_ids, raw, strict=True)):
        _reject_sensitive_ledger_data(record, path=f"raw_records[{index}]")
        raw_by_id[request_id] = record

    ordered_raw = [raw_by_id[request_id] for request_id in manifest_ids]
    raw_payload = b"".join(_json_bytes(record, pretty=False) for record in ordered_raw)
    return manifest, raw_payload


def _mode(result: Mapping[str, Any]) -> str:
    for key in (
        "input_mode",
        "mode",
        "compilation_mode",
        "graph_source",
        "source",
    ):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold().replace("-", "_")
    return ""


def write_stage0_artifacts(
    output_dir: str | Path,
    evaluation: Any,
) -> tuple[Path, ...]:
    """Validate and atomically publish one Stage 0 evaluation.

    Returned paths are stable: result, optional compiled semantic draft,
    request manifest, raw JSONL, followed by the optional RequirementGraph.
    The graph is written last so consumers can treat it as the successful
    compilation product.
    """

    root = Path(output_dir).expanduser()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(str(root))

    if not hasattr(evaluation, "result"):
        raise TypeError("evaluation must expose result")
    if not hasattr(evaluation, "graph"):
        raise TypeError("evaluation must expose graph")
    if not hasattr(evaluation, "draft"):
        raise TypeError("evaluation must expose draft")
    if not hasattr(evaluation, "request_manifest"):
        raise TypeError("evaluation must expose request_manifest")
    if not hasattr(evaluation, "raw_records"):
        raise TypeError("evaluation must expose raw_records")

    result_payload = _to_dict_payload(evaluation.result, path="result")
    manifest_payload, raw_payload = _validate_ledgers(
        evaluation.request_manifest,
        evaluation.raw_records,
    )
    vlm_call_count = result_payload.get("vlm_call_count")
    if (
        isinstance(vlm_call_count, bool)
        or not isinstance(vlm_call_count, int)
        or vlm_call_count < 0
    ):
        raise ValueError("result.vlm_call_count must be a non-negative integer")
    if len(manifest_payload) != vlm_call_count:
        raise ValueError(
            "Stage 0 VLM ledger count must equal result.vlm_call_count"
        )
    input_mode = _mode(result_payload)
    provided_mode = input_mode in {"provided", "provided_graph"}
    if provided_mode and manifest_payload:
        raise ValueError("provided Stage 0 mode must have empty VLM ledgers")

    graph = evaluation.graph
    graph_payload = (
        None if graph is None else _to_dict_payload(graph, path="graph")
    )

    status = str(result_payload.get("status", "")).strip().casefold()
    draft = evaluation.draft
    draft_payload: dict[str, Any] | None = None
    if status == "compiled":
        if provided_mode:
            raise ValueError("provided Stage 0 mode cannot publish a compiled draft")
        if draft is None:
            raise ValueError("compiled Stage 0 outcome requires draft")
        draft_payload = _to_mapping_payload(draft, path="draft")
    elif (status == "bypassed" or provided_mode) and draft is not None:
        raise ValueError("bypassed Stage 0 outcome cannot contain draft")

    result_path = root / "stage0_result.json"
    draft_path = root / "stage0_draft.json"
    manifest_path = root / "stage0_vlm_request_manifest.json"
    raw_path = root / "stage0_vlm_raw.jsonl"
    paths = [result_path]
    fixed_payloads = [(result_path, _json_bytes(result_payload))]
    if draft_payload is not None:
        paths.append(draft_path)
        fixed_payloads.append((draft_path, _json_bytes(draft_payload)))
    paths.extend((manifest_path, raw_path))
    fixed_payloads.extend(
        (
            (manifest_path, _json_bytes(manifest_payload)),
            (raw_path, raw_payload),
        )
    )
    for path, payload in fixed_payloads:
        _atomic_write_bytes(path, payload)

    if graph_payload is None:
        return tuple(paths)

    graph_path = root / "requirement_graph.json"
    _atomic_write_bytes(graph_path, _json_bytes(graph_payload))
    return (*paths, graph_path)


__all__ = ["write_stage0_artifacts"]
