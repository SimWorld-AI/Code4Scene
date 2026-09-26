"""Strict, provider-independent contracts for RequirementGraph Stage 0.

Stage 0 either compiles a semantic prompt into a validated
:class:`RequirementGraph`, validates an explicitly supplied graph without a
compiler call, or fails without publishing a graph.  The contracts in this
module keep those three outcomes distinct so a malformed compiler response can
never be mistaken for a usable evaluation rubric.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .contracts import (
    EntityEvaluationRoute,
    EntityNode,
    JsonSerializable,
    RequirementGraph,
)

_VALIDATION_STATUSES = frozenset({"not_attempted", "success", "error"})
_WEIGHT_POLICY = "equal_atomic_facets_v1"


class Stage0CompilationError(RuntimeError):
    """Raised when Stage 0 cannot safely publish a RequirementGraph."""

    def __init__(self, message: str, *, code: str = "stage0_compilation_error") -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Stage 0 error code must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Stage 0 error message must be a non-empty string")
        self.code = code.strip()
        self.message = message.strip()
        super().__init__(f"{self.code}: {self.message}")


class _CoercibleEnum(str, Enum):
    @classmethod
    def coerce(cls, value: Any):
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(f"{cls.__name__} must be a string or {cls.__name__}")
        return cls(value.strip().casefold())


class Stage0Status(_CoercibleEnum):
    COMPILED = "compiled"
    FAILED = "failed"
    BYPASSED = "bypassed"


class Stage0InputMode(_CoercibleEnum):
    PROMPT = "prompt"
    REQUEST_JSON = "request_json"
    PROVIDED_GRAPH = "provided_graph"


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _elapsed(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("elapsed_s must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("elapsed_s must be finite and non-negative")
    return result


def _validate_json_value(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string object key")
            _validate_json_value(item, path=f"{path}/{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}/{index}")
        return
    raise TypeError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    _validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class Stage0CompilationResult(JsonSerializable):
    """Auditable summary of one Stage 0 terminal outcome."""

    status: Stage0Status | str
    input_mode: Stage0InputMode | str
    compiler_used: bool
    model: str | None = None
    tool_name: str = "return_requirement_graph_draft"
    weight_policy: str = _WEIGHT_POLICY
    validation_status: str = "not_attempted"
    node_count: int = 0
    edge_count: int = 0
    root_count: int = 0
    scored_leaf_count: int = 0
    inventory_existence_entity_count: int = 0
    stage2_visual_entity_count: int = 0
    vlm_call_count: int = 0
    elapsed_s: float = 0.0
    evaluation_error: str | None = None
    schema_version: str = "2.0"

    def __post_init__(self) -> None:
        status = Stage0Status.coerce(self.status)
        input_mode = Stage0InputMode.coerce(self.input_mode)
        if not isinstance(self.compiler_used, bool):
            raise TypeError("compiler_used must be a bool")

        model = _optional_text(self.model, "model")
        tool_name = _required_text(self.tool_name, "tool_name")
        if not isinstance(self.weight_policy, str) or self.weight_policy != _WEIGHT_POLICY:
            raise ValueError(f"weight_policy must be {_WEIGHT_POLICY!r}")
        if not isinstance(self.validation_status, str):
            raise TypeError("validation_status must be a string")
        validation_status = self.validation_status.strip().casefold()
        if validation_status not in _VALIDATION_STATUSES:
            raise ValueError(
                "validation_status must be not_attempted, success, or error"
            )

        count_names = (
            "node_count",
            "edge_count",
            "root_count",
            "scored_leaf_count",
            "inventory_existence_entity_count",
            "stage2_visual_entity_count",
            "vlm_call_count",
        )
        counts = {name: _count(getattr(self, name), name) for name in count_names}
        elapsed = _elapsed(self.elapsed_s)
        error = _optional_text(self.evaluation_error, "evaluation_error")
        if not isinstance(self.schema_version, str) or self.schema_version not in {
            "1.0",
            "2.0",
        }:
            raise ValueError("unsupported Stage 0 result schema version")

        if input_mode is Stage0InputMode.PROVIDED_GRAPH:
            if self.compiler_used:
                raise ValueError("provided_graph input cannot use the compiler")
        elif not self.compiler_used:
            raise ValueError("prompt and request_json inputs require compiler_used=True")

        graph_counts = tuple(counts[name] for name in count_names[:6])
        if counts["root_count"] > counts["node_count"]:
            raise ValueError("root_count cannot exceed node_count")
        if counts["scored_leaf_count"] > counts["node_count"]:
            raise ValueError("scored_leaf_count cannot exceed node_count")
        if (
            counts["inventory_existence_entity_count"]
            + counts["stage2_visual_entity_count"]
            > counts["node_count"]
        ):
            raise ValueError("entity route counts cannot exceed node_count")

        if status is Stage0Status.COMPILED:
            if input_mode is Stage0InputMode.PROVIDED_GRAPH:
                raise ValueError("compiled status requires prompt or request_json input")
            if model is None:
                raise ValueError("compiled status requires model")
            if validation_status != "success":
                raise ValueError("compiled status requires successful validation")
            if error is not None:
                raise ValueError("compiled status cannot contain evaluation_error")
            if any(value < 1 for value in graph_counts[:4]):
                raise ValueError("compiled status requires non-empty graph counts")
            if counts["vlm_call_count"] < 1:
                raise ValueError("compiled status requires at least one VLM call")
        elif status is Stage0Status.BYPASSED:
            if input_mode is not Stage0InputMode.PROVIDED_GRAPH:
                raise ValueError("bypassed status requires provided_graph input")
            if model is not None:
                raise ValueError("bypassed status cannot contain compiler draft/model data")
            if validation_status != "success":
                raise ValueError("bypassed status requires successful validation")
            if error is not None:
                raise ValueError("bypassed status cannot contain evaluation_error")
            if any(value < 1 for value in graph_counts[:4]):
                raise ValueError("bypassed status requires non-empty graph counts")
            if counts["vlm_call_count"] != 0:
                raise ValueError("bypassed status cannot contain VLM calls")
        else:
            if validation_status == "success":
                raise ValueError("failed status cannot report successful validation")
            if error is None:
                raise ValueError("failed status requires evaluation_error")
            if any(graph_counts):
                raise ValueError("failed status cannot publish graph counts")
            if input_mode is Stage0InputMode.PROVIDED_GRAPH:
                if model is not None or counts["vlm_call_count"]:
                    raise ValueError(
                        "failed provided_graph input cannot contain compiler data"
                    )
            elif counts["vlm_call_count"] and model is None:
                raise ValueError("failed VLM calls require model")

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "input_mode", input_mode)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "validation_status", validation_status)
        for name, value in counts.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "elapsed_s", elapsed)
        object.__setattr__(self, "evaluation_error", error)


def _json_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _validate_json_value(value, path=name)
    copied = json.loads(_canonical_json_bytes(value).decode("utf-8"))
    if not isinstance(copied, dict):  # pragma: no cover - Mapping guarantees object
        raise TypeError(f"{name} must encode a JSON object")
    return copied


def _ledger_records(
    values: Sequence[Mapping[str, Any]], name: str
) -> tuple[dict[str, Any], ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of mappings")
    records = tuple(
        _json_mapping(value, f"{name}[{index}]") for index, value in enumerate(values)
    )
    request_ids: list[str] = []
    for index, record in enumerate(records):
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError(f"{name}[{index}].request_id must be a non-empty string")
        request_ids.append(request_id.strip())
        record["request_id"] = request_id.strip()
    if len(request_ids) != len(set(request_ids)):
        raise ValueError(f"{name} request_id values must be unique")
    return records


@dataclass(frozen=True, slots=True)
class Stage0Evaluation:
    """Complete in-memory Stage 0 product, including auditable call ledgers."""

    graph: RequirementGraph | None
    draft: Mapping[str, Any] | None
    result: Stage0CompilationResult
    request_manifest: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    raw_records: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.graph is not None and not isinstance(self.graph, RequirementGraph):
            raise TypeError("graph must be a RequirementGraph or None")
        if not isinstance(self.result, Stage0CompilationResult):
            raise TypeError("result must be a Stage0CompilationResult")
        draft = (
            _json_mapping(self.draft, "draft") if self.draft is not None else None
        )
        manifest = _ledger_records(self.request_manifest, "request_manifest")
        raw = _ledger_records(self.raw_records, "raw_records")
        manifest_ids = tuple(value["request_id"] for value in manifest)
        raw_ids = tuple(value["request_id"] for value in raw)
        if manifest_ids != raw_ids:
            raise ValueError("request_manifest and raw_records request ids must match in order")
        if len(manifest) != self.result.vlm_call_count:
            raise ValueError("ledger record count must equal result.vlm_call_count")

        if self.result.status in {Stage0Status.COMPILED, Stage0Status.BYPASSED}:
            if self.graph is None:
                raise ValueError("successful Stage 0 outcomes require graph")
            graph_counts = _graph_counts(self.graph)
            for name, expected in graph_counts.items():
                if getattr(self.result, name) != expected:
                    raise ValueError(f"result {name} does not match graph")
        elif self.graph is not None:
            raise ValueError("failed Stage 0 outcomes cannot publish graph")

        if self.result.status is Stage0Status.COMPILED:
            if draft is None:
                raise ValueError("compiled Stage 0 outcome requires draft")
        elif self.result.status is Stage0Status.BYPASSED and draft is not None:
            raise ValueError("bypassed Stage 0 outcome cannot contain draft")
        object.__setattr__(self, "draft", draft)
        object.__setattr__(self, "request_manifest", manifest)
        object.__setattr__(self, "raw_records", raw)

    def to_summary(self) -> dict[str, Any]:
        """Return the compact Stage 0 section embedded in graph-run summaries."""

        summary = self.result.to_dict()
        summary.update(
            {
                "graph_available": self.graph is not None,
                "draft_available": self.draft is not None,
                "request_manifest_count": len(self.request_manifest),
                "raw_record_count": len(self.raw_records),
            }
        )
        return summary


def _graph_counts(graph: RequirementGraph) -> dict[str, int]:
    entities = tuple(node for node in graph.nodes if isinstance(node, EntityNode))
    return {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "root_count": len(graph.roots),
        "scored_leaf_count": len(graph.effective_weights()),
        "inventory_existence_entity_count": sum(
            node.evaluation_route is EntityEvaluationRoute.INVENTORY_EXISTENCE
            for node in entities
        ),
        "stage2_visual_entity_count": sum(
            node.evaluation_route is EntityEvaluationRoute.STAGE2_VISUAL
            for node in entities
        ),
    }


def provided_graph_evaluation(
    graph: RequirementGraph,
    *,
    elapsed_s: float = 0.0,
) -> Stage0Evaluation:
    """Wrap an already validated graph as an explicit Stage 0 bypass result."""

    if not isinstance(graph, RequirementGraph):
        raise TypeError("graph must be a RequirementGraph")
    result = Stage0CompilationResult(
        status=Stage0Status.BYPASSED,
        input_mode=Stage0InputMode.PROVIDED_GRAPH,
        compiler_used=False,
        validation_status="success",
        elapsed_s=elapsed_s,
        **_graph_counts(graph),
    )
    return Stage0Evaluation(graph=graph, draft=None, result=result)


__all__ = [
    "Stage0CompilationError",
    "Stage0CompilationResult",
    "Stage0Evaluation",
    "Stage0InputMode",
    "Stage0Status",
    "provided_graph_evaluation",
]
