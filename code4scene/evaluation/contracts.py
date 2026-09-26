"""The shape a verifier's answer takes, and which runs may be published.

The platform already had a typed result vocabulary — the ``VerifierReport``
vendored in :mod:`code4scene.evaluation.report_schema`, used to evaluate agents
*acting* in scenes. This module evaluates agents *generating* them and grew its
own record, so the run record stays the harness's working format — it carries
what the shared schema has no place for, like the round loop and the plate
corrections — and a report is lowered into the shared shape on its way out.
Nothing imports the other package here: the shapes are plain dicts, so this
works in a standalone install, and a test constructs the real dataclasses from
them so "conforms to the schema" is checked rather than claimed.

What is NOT here any more: how a verifier decides its number. ``measure_report``,
``bounds_report``, ``judge_report`` and the scene-score formula lived in this
file, which left three verifier files as one-line shims importing their own
algorithm from a shared module two directories up — opening the file named for
a verifier told you nothing about it. Each verifier owns its scoring now. This
module owns the envelope and the publication rule, and nothing else.

The shared schema's rules that shape this code:

* ``score`` is 0..1. Normalized quality is higher-is-better, while a leaf may
  explicitly publish a direct lower-is-better measurement in
  ``metadata.score_direction``; aggregation orients it exactly once;
* new scene-quality reports use continuous ``measured`` status and make no
  pass/fail decision;
* ``artifacts`` maps string to string, so paths only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, ClassVar, Generic, Literal, Protocol, TypeVar

PASS = "pass"
FAIL = "fail"
ERROR = "error"
MEASURED = "measured"
VALID = "valid"
INVALID = "invalid"

#: Report statuses that describe an unanswered question rather than scene
#: quality.  ``not_evaluated`` and ``not_applicable`` normally live on typed
#: metric results; keeping them here too makes the aggregation boundary fail
#: closed if a verifier ever surfaces one at report level.
WITHHELD_STATUSES = (
    ERROR,
    "not_evaluated",
    "not_applicable",
    VALID,
    INVALID,
)

#: Exit reasons whose record must never be scored, ranked or published.
#:
#: ``tampered`` was in the record and in nothing else: the harness detected a
#: rewound budget counter, wrote the reason, and every consumer went on asking
#: only whether the reason was ``infra_error``. So the one run the benchmark
#: had caught cheating was stored as done, scored, and averaged into the
#: leaderboard — the opposite of the thing detecting it was for. It is a single
#: predicate now precisely because four call sites each deciding this
#: separately is how the fifth one gets it wrong.
UNPUBLISHABLE_EXIT_REASONS = ("infra_error", "tampered")


def unpublishable(record: dict[str, Any]) -> str | None:
    """The reason this record must not be published, or None."""
    reason = record.get("exit_reason")
    return reason if reason in UNPUBLISHABLE_EXIT_REASONS else None


#: What the caller must supply, because the shared schema identifies a report
#: by the bundle and episode it belongs to and the harness's record does not
#: carry those ids.
_REQUIRED_IDS = ("task_bundle_id", "episode_id")


def base(report_id: str, ids: dict[str, str]) -> dict[str, Any]:
    """The identity every report carries, refused when it is incomplete."""
    missing = [key for key in _REQUIRED_IDS if not ids.get(key)]
    if missing:
        raise ValueError(f"lowering a report needs {', '.join(missing)}")
    return {"report_id": report_id,
            "task_bundle_id": ids["task_bundle_id"],
            "episode_id": ids["episode_id"]}


def score_for_aggregate(report: Mapping[str, Any]) -> float | None:
    """Return higher-is-better quality, or ``None`` when it was withheld.

    The report remains in the episode record either way.  This function owns
    the narrower question of whether its number may enter ``scores_by_class``.
    ``measured`` is the canonical quality status: a continuous score needs no
    pass/fail threshold before it can be compared. PASS/FAIL remain readable
    only for legacy record compatibility. An unanswered report carrying 0.0
    is rejected rather than silently interpreted as a bad scene.
    """
    status = report.get("status")
    score = report.get("score")
    report_id = report.get("report_id", "<unknown>")
    if status in WITHHELD_STATUSES:
        if score is not None:
            raise ValueError(
                f"{report_id}: {status} reports must carry score=None, not {score!r}"
            )
        return None
    if status not in {MEASURED, PASS, FAIL}:
        raise ValueError(f"{report_id}: unsupported report status {status!r}")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"{report_id}: {status} reports need a numeric score")
    value = float(score)
    _fraction(value, f"{report_id}.score")
    metadata = report.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    direction = metadata.get("score_direction", "higher_is_better")
    if direction == "lower_is_better":
        return 1.0 - value
    if direction != "higher_is_better":
        raise ValueError(
            f"{report_id}: unsupported score direction {direction!r}"
        )
    return value


def artifacts(record: dict[str, Any]) -> dict[str, str]:
    """Paths from the record that a report may point at — strings, per schema."""
    return {key: str(record[key]) for key in ("umap",) if record.get(key)}


def metric_id(record: dict[str, Any], prefix: str) -> str:
    """Which version of a measurement produced this record's numbers."""
    ids = (record.get("provenance") or {}).get("metric_ids") or []
    return next((i for i in ids if i.startswith(prefix)), f"{prefix}_v1")


# ── what a MEASUREMENT looks like, before and after scoring ──────────────
#
# One property's raw value and its normalized result. They live beside the
# report envelope because they answer the same question — what shape does a
# verifier's output take — and a second module for that is a second place to
# look. Validation is here so a metric cannot write a NaN, a score outside
# [0, 1], or a `not_applicable` that quietly carries a measured zero.
ResultStatus = Literal[
    "measured", "pass", "fail", "not_evaluated", "not_applicable", "error"
]
MeasurementStatus = Literal["measured", "not_evaluated", "not_applicable", "error"]

EvidenceT = TypeVar("EvidenceT")
PolicyT = TypeVar("PolicyT")
RawT = TypeVar("RawT")
ResultRawT = TypeVar("ResultRawT")


def _fraction(value: float | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")


def _json_value(value: Any, path: str = "result") -> Any:
    """Return a JSON-native value and reject non-finite numbers early."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value), path)
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            converted[key] = _json_value(item, f"{path}.{key}")
        return converted
    if isinstance(value, (tuple, list)):
        return [_json_value(item, f"{path}[]") for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


@dataclass(frozen=True)
class RawMeasurement(Generic[RawT]):
    """The output of measurement, before any scoring policy is applied."""

    id: str
    instance_id: str
    metric_version: str
    applicable: bool
    status: MeasurementStatus
    coverage: float | None
    raw: RawT | None
    evidence: tuple[Mapping[str, Any], ...] = ()
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("id", self.id),
            ("instance_id", self.instance_id),
            ("metric_version", self.metric_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.status not in {"measured", "not_evaluated", "not_applicable", "error"}:
            raise ValueError(f"unsupported measurement status {self.status!r}")
        _fraction(self.coverage, "coverage")
        if self.status == "not_applicable":
            if self.applicable or self.coverage is not None or self.raw is not None:
                raise ValueError(
                    "not_applicable measurements need applicable=false, no coverage and no raw"
                )
        elif not self.applicable:
            raise ValueError(f"{self.status} measurements must be applicable")
        if self.status == "measured" and (self.raw is None or self.coverage != 1.0):
            raise ValueError("measured results need raw evidence and complete coverage")
        if self.status == "not_evaluated" and not self.failure_reason:
            raise ValueError("not_evaluated measurements need a failure reason")
        _json_value(self.raw, "raw")
        if self.status == "error" and not self.failure_reason:
            raise ValueError("error measurements need a failure reason")
        _json_value(self.evidence, "evidence")


@dataclass(frozen=True)
class MetricResult(Generic[ResultRawT]):
    """A normalized, versioned, one-property verifier result."""

    id: str
    instance_id: str
    metric_version: str
    dimension: str
    applicability_policy: str
    required_evidence: tuple[str, ...]
    applicable: bool
    status: ResultStatus
    coverage: float | None
    raw: ResultRawT | None
    score: float | None
    normalization_policy: str | None
    normalization_parameters: Mapping[str, Any]
    calibration_status: str
    contributes_to_aggregate: bool
    evidence: tuple[Mapping[str, Any], ...] = ()
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("id", self.id),
            ("instance_id", self.instance_id),
            ("metric_version", self.metric_version),
            ("dimension", self.dimension),
            ("applicability_policy", self.applicability_policy),
            ("calibration_status", self.calibration_status),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if (
            not isinstance(self.required_evidence, tuple)
            or not self.required_evidence
            or any(not isinstance(item, str) or not item for item in self.required_evidence)
        ):
            raise ValueError("required_evidence must be a non-empty tuple of strings")
        if not isinstance(self.contributes_to_aggregate, bool):
            raise ValueError("contributes_to_aggregate must be boolean")
        if self.status not in {
            "measured", "pass", "fail", "not_evaluated", "not_applicable", "error"
        }:
            raise ValueError(f"unsupported metric status {self.status!r}")
        _fraction(self.coverage, "coverage")
        _fraction(self.score, "score")
        if self.status == "not_applicable":
            if (
                self.applicable
                or self.coverage is not None
                or self.score is not None
                or self.raw is not None
            ):
                raise ValueError(
                    "not_applicable results need applicable=false and no coverage, raw or score"
                )
        elif not self.applicable:
            raise ValueError(f"{self.status} results must be applicable")
        if self.status == "error" and self.score is not None:
            raise ValueError("error results cannot carry a score")
        if self.status == "error" and not self.failure_reason:
            raise ValueError("error results need a failure reason")
        if self.status == "not_evaluated" and self.score is not None:
            raise ValueError("not_evaluated results cannot carry a score")
        if self.status == "not_evaluated" and not self.failure_reason:
            raise ValueError("not_evaluated results need a failure reason")
        if self.status in {"measured", "pass", "fail"}:
            if (
                self.coverage != 1.0
                or self.raw is None
                or not self.normalization_policy
            ):
                raise ValueError(
                    "measured/pass/fail results need complete coverage, raw and normalization"
                )
            if self.score is None and (
                self.status == "measured" or self.contributes_to_aggregate
            ):
                raise ValueError(
                    "measured and aggregate-contributing results need a score"
                )
        if self.status == "fail" and not self.failure_reason:
            raise ValueError("fail results need a failure reason")
        if self.status == "pass" and self.failure_reason is not None:
            raise ValueError("pass results cannot carry a failure reason")
        _json_value(self.raw, "raw")
        _json_value(self.normalization_parameters, "normalization_parameters")
        _json_value(self.evidence, "evidence")

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize without tuples, dataclass instances, or non-finite floats."""
        return _json_value(asdict(self))


class Metric(Protocol[EvidenceT, PolicyT, RawT, ResultRawT]):
    """The interface every metric slice implements."""

    id: ClassVar[str]
    metric_version: ClassVar[str]
    dimension: ClassVar[str]
    applicability_policy: ClassVar[str]
    required_evidence: ClassVar[tuple[str, ...]]

    def measure(self, evidence: EvidenceT) -> RawMeasurement[RawT]: ...

    def normalize(
        self, measurement: RawMeasurement[RawT], policy: PolicyT
    ) -> MetricResult[ResultRawT]: ...


__all__ = ["ERROR", "FAIL", "INVALID", "MEASURED", "PASS", "VALID", "UNPUBLISHABLE_EXIT_REASONS", "WITHHELD_STATUSES", "MeasurementStatus",
           "Metric", "MetricResult", "RawMeasurement", "ResultStatus", "artifacts",
           "base", "metric_id", "score_for_aggregate", "unpublishable"]
