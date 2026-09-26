"""The verifier report contract every Code4Scene verifier conforms to.

A verifier report is a pure data contract: a ``status`` from
:class:`ReportStatus`, a score in [0, 1] only when the status carries a
measurement, and a failure reason for every withheld outcome. Withheld
outcomes (``not_evaluated``, ``not_applicable``, ``error``, ``invalid``,
``timeout``) are never aliases for a zero score; the scoring protocol decides
separately how an unavailable component is scored (see
:mod:`code4scene.protocol`).

Vendored from the platform's shared schema module so the package validates
its own reports without that dependency.
"""
from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeVar


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class ReportStatus(_StrEnum):
    MEASURED = "measured"
    VALID = "valid"
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"
    NOT_APPLICABLE = "not_applicable"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    ERROR = "error"


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class VerifierReport:
    """Structured verifier result with enough evidence for audit and RL reports."""

    report_id: str
    task_bundle_id: str
    episode_id: str
    status: ReportStatus
    # None means the verifier did not produce a quality judgement.  In
    # particular NOT_EVALUATED/NOT_APPLICABLE/ERROR/INVALID/TIMEOUT are
    # withheld outcomes, never aliases for a scene score of zero. PASS/FAIL
    # reports may also omit a score when they are explicitly report-only
    # diagnostics. MEASURED carries a continuous diagnostic score without a
    # pass/fail decision.
    score: float | None = None
    metrics: MappingABC[str, Any] = None  # type: ignore[assignment]
    evidence: MappingABC[str, Any] = None  # type: ignore[assignment]
    probes_used: tuple[str, ...] = ()
    failure_reason: str | None = None
    frame_tokens: tuple[MappingABC[str, Any], ...] = ()
    artifacts: MappingABC[str, str] = None  # type: ignore[assignment]
    metadata: MappingABC[str, Any] = None  # type: ignore[assignment]

    @classmethod
    def from_json_dict(cls, data: MappingABC[str, Any]) -> VerifierReport:
        return cls(**dict(data))

    def __post_init__(self) -> None:
        _require_non_empty(self.report_id, "VerifierReport.report_id")
        _require_non_empty(self.task_bundle_id, "VerifierReport.task_bundle_id")
        _require_non_empty(self.episode_id, "VerifierReport.episode_id")
        object.__setattr__(self, "status", _coerce_enum(ReportStatus, self.status, "VerifierReport.status"))
        if (
            self.status not in {
                ReportStatus.MEASURED,
                ReportStatus.PASS,
                ReportStatus.VALID,
            }
            and not self.failure_reason
        ):
            raise ValueError(
                "VerifierReport.failure_reason is required for failed or withheld statuses"
            )
        if self.status not in {
            ReportStatus.MEASURED,
            ReportStatus.PASS,
            ReportStatus.FAIL,
        } and self.score is not None:
            raise ValueError(
                "VerifierReport.score must be None for withheld statuses"
            )
        if self.status is ReportStatus.MEASURED and self.score is None:
            raise ValueError("VerifierReport.score is required when status is measured")
        if self.status is ReportStatus.MEASURED and self.failure_reason is not None:
            raise ValueError(
                "VerifierReport.failure_reason must be None when status is measured"
            )
        if self.score is not None:
            if not 0.0 <= float(self.score) <= 1.0:
                raise ValueError("VerifierReport.score must be in [0, 1]")
            object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metrics", _mapping(self.metrics, "VerifierReport.metrics"))
        object.__setattr__(self, "evidence", _mapping(self.evidence, "VerifierReport.evidence"))
        object.__setattr__(self, "probes_used", _string_tuple(self.probes_used, "VerifierReport.probes_used"))
        object.__setattr__(
            self,
            "frame_tokens",
            tuple(_mapping(item, "VerifierReport.frame_tokens.item") for item in self.frame_tokens or ()),
        )
        object.__setattr__(self, "artifacts", _string_mapping(self.artifacts, "VerifierReport.artifacts"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "VerifierReport.metadata"))


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _coerce_enum(enum_cls: type[_T], value: Any, field_name: str) -> _T:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)  # type: ignore[call-arg]
    except Exception as exc:
        valid = [item.value for item in enum_cls]  # type: ignore[attr-defined]
        raise ValueError(f"{field_name} must be one of {valid}, got {value!r}") from exc



def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, SequenceABC):
        raise ValueError(f"{field_name} must be a sequence of strings")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise ValueError(f"{field_name} cannot contain empty strings")
    return result


def _mapping(value: Any, field_name: str) -> MappingABC[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, MappingABC):
        raise ValueError(f"{field_name} must be a mapping")
    return MappingProxyType({
        str(key): _freeze_jsonish_value(item)
        for key, item in value.items()
    })


def _string_mapping(value: Any, field_name: str) -> MappingABC[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, MappingABC):
        raise ValueError(f"{field_name} must be a mapping")
    return MappingProxyType({str(key): str(item) for key, item in value.items()})


def _freeze_jsonish_value(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType({
            str(key): _freeze_jsonish_value(item)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_jsonish_value(item) for item in value)
    return value


__all__ = ["ReportStatus", "VerifierReport"]
