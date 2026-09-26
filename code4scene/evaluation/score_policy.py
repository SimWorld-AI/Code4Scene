"""Versioned, explicit aggregation policies for artifact evaluation results.

Verifier reports remain the primary evidence.  A score policy only combines
their already-recorded outputs, which makes the aggregation reproducible
offline without UE, screenshots, or new model calls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import yaml

from code4scene.evaluation import case_outcome, physics_score, primary_score
from code4scene.evaluation.semantic_scoring import SEMANTIC_FAMILY_ORDER
from code4scene.evaluation.scalar_score import without_intervals


SCHEMA_VERSION = "scenebenchmark.score_policy.v1"
RESULT_SCHEMA_VERSION = "scenebenchmark.score_policy.result.v4"

COMPONENT_ORDER = ("semantic", "physics", "holistic_overview")
COMPONENT_REPORT_IDS = {
    "semantic": "semantic_requirements",
    "physics": "physical_safety",
    "holistic_overview": "overview_prompt_alignment",
}

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "policy_id",
        "description",
        "required_task_verifiers",
        "semantic_family_weights",
        "total_weights",
        "missing_family_policy",
        "missing_component_policy",
        "published_score",
    }
)
_BAD_STATUSES = frozenset(
    {
        "blocked",
        "error",
        "invalid",
        "not_evaluated",
        "refused",
        "unavailable",
    }
)


class ScorePolicyError(ValueError):
    """The policy or the reports being aggregated violate its contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weights(
    raw: Any,
    *,
    field: str,
    expected: Sequence[str],
) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ScorePolicyError(f"{field} must be a mapping")
    unknown = sorted(set(raw) - set(expected))
    missing = sorted(set(expected) - set(raw))
    if unknown or missing:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ScorePolicyError(
            f"{field} must contain exactly {tuple(expected)}: "
            + "; ".join(details)
        )
    parsed: dict[str, float] = {}
    for name in expected:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ScorePolicyError(f"{field}.{name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ScorePolicyError(f"{field}.{name} must be finite and non-negative")
        parsed[name] = number
    total = sum(parsed.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ScorePolicyError(f"{field} must sum to 1.0, got {total:.12g}")
    return parsed


@dataclass(frozen=True)
class ScorePolicy:
    policy_id: str
    description: str
    required_task_verifiers: tuple[str, ...]
    semantic_family_weights: Mapping[str, float]
    total_weights: Mapping[str, float]
    missing_family_policy: str
    missing_component_policy: str
    published_score: str
    source_path: Path
    sha256: str

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "source_path": str(self.source_path),
            "sha256": self.sha256,
            "semantic_family_weights": dict(self.semantic_family_weights),
            "total_weights": dict(self.total_weights),
            "missing_family_policy": self.missing_family_policy,
            "missing_component_policy": self.missing_component_policy,
            "published_score": self.published_score,
        }


def load(path: str | Path) -> ScorePolicy:
    """Load and strictly validate one score-policy YAML file."""

    source = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ScorePolicyError(f"cannot read score policy {source}: {error}") from error
    if not isinstance(document, Mapping):
        raise ScorePolicyError(f"score policy {source} must contain a YAML mapping")
    unknown = sorted(set(document) - _ROOT_KEYS)
    if unknown:
        raise ScorePolicyError(
            f"score policy {source} has unknown keys: {', '.join(unknown)}"
        )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ScorePolicyError(
            f"score policy {source}: schema_version must be exactly {SCHEMA_VERSION!r}"
        )
    policy_id = document.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ScorePolicyError("policy_id must be a non-empty string")
    description = document.get("description", "")
    if not isinstance(description, str):
        raise ScorePolicyError("description must be a string")
    raw_required = document.get("required_task_verifiers")
    if (
        not isinstance(raw_required, list)
        or not raw_required
        or any(not isinstance(value, str) or not value.strip() for value in raw_required)
    ):
        raise ScorePolicyError("required_task_verifiers must be a non-empty string list")
    required = tuple(value.strip() for value in raw_required)
    if len(set(required)) != len(required):
        raise ScorePolicyError("required_task_verifiers must not contain duplicates")

    missing_family_policy = str(
        document.get("missing_family_policy", "renormalize_applicable")
    )
    if missing_family_policy != "renormalize_applicable":
        raise ScorePolicyError(
            "missing_family_policy must be exactly 'renormalize_applicable'"
        )
    missing_component_policy = str(
        document.get("missing_component_policy", "zero_without_renormalization")
    )
    if missing_component_policy not in {"retain_weight_interval", "zero_without_renormalization"}:
        raise ScorePolicyError(
            "missing_component_policy must be 'zero_without_renormalization'"
        )
    published_score = str(document.get("published_score", "score"))
    if published_score not in {"lower_bound", "score"}:
        raise ScorePolicyError("published_score must be 'score'")
    # Accept old configuration spelling, but never reactivate interval scoring.
    missing_component_policy = "zero_without_renormalization"
    published_score = "score"

    return ScorePolicy(
        policy_id=policy_id.strip(),
        description=description.strip(),
        required_task_verifiers=required,
        semantic_family_weights=_weights(
            document.get("semantic_family_weights"),
            field="semantic_family_weights",
            expected=SEMANTIC_FAMILY_ORDER,
        ),
        total_weights=_weights(
            document.get("total_weights"),
            field="total_weights",
            expected=COMPONENT_ORDER,
        ),
        missing_family_policy=missing_family_policy,
        missing_component_policy=missing_component_policy,
        published_score=published_score,
        source_path=source,
        sha256=_sha256(source),
    )


def default_for_task(task: Any) -> ScorePolicy | None:
    """Use the canonical T2S policy without a repository-relative config path."""
    declared = {
        value.get("name") for value in (getattr(task, "verifiers", ()) or ())
        if isinstance(value, Mapping)
    }
    required = ("semantic_requirements", "overview_prompt_alignment")
    if (
        getattr(task, "case_type", None) == "image_to_scene"
        or "gt_repair" in declared
        or not set(required).issubset(declared)
    ):
        return None
    return ScorePolicy(
        policy_id="text-to-scene-human-aligned",
        description="Canonical T2S semantic, physics and overview aggregation.",
        required_task_verifiers=required,
        semantic_family_weights={
            "identity_environment": 0.25, "content_quantity": 0.40,
            "spatial_composition": 0.20, "attributes_materials": 0.15,
        },
        total_weights={"semantic": 0.20, "physics": 0.20, "holistic_overview": 0.60},
        missing_family_policy="renormalize_applicable",
        missing_component_policy="zero_without_renormalization",
        published_score="score",
        source_path=Path(__file__).resolve(),
        sha256=_sha256(Path(__file__)),
    )


def validate_task(policy: ScorePolicy, task: Any) -> None:
    """Fail before Candidate access when a policy is used with the wrong task."""

    declared = {
        str(value.get("name"))
        for value in (getattr(task, "verifiers", ()) or ())
        if isinstance(value, Mapping)
    }
    missing = sorted(set(policy.required_task_verifiers) - declared)
    if missing:
        raise ScorePolicyError(
            f"score policy {policy.policy_id!r} cannot score task "
            f"{getattr(task, 'id', '<unknown>')!r}; missing declared verifiers: "
            + ", ".join(missing)
        )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _round4(value: float) -> float:
    return round(float(value), 4)


def _usable(report: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(report, Mapping)
        and str(report.get("status", "")).casefold() not in _BAD_STATUSES
    )


def _semantic_component(
    report: Mapping[str, Any] | None,
    policy: ScorePolicy,
) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report, Mapping) else None
    metrics = metrics if isinstance(metrics, Mapping) else {}
    raw_families = metrics.get("semantic_subscores")
    raw_families = raw_families if isinstance(raw_families, Mapping) else {}
    report_usable = _usable(report)

    applicable: list[tuple[str, float, Mapping[str, Any]]] = []
    family_results: dict[str, dict[str, Any]] = {}
    for family in SEMANTIC_FAMILY_ORDER:
        raw = raw_families.get(family)
        row = without_intervals(raw) if isinstance(raw, Mapping) else {}
        is_applicable = bool(row.get("applicable"))
        configured_weight = float(policy.semantic_family_weights[family])
        if not is_applicable:
            family_results[family] = {
                "applicable": False,
                "configured_weight": configured_weight,
                "effective_weight": 0.0,
                "score": None,
                "known_coverage": None,
            }
            continue
        applicable.append((family, configured_weight, row))

    denominator = sum(weight for _, weight, _ in applicable)
    if denominator <= 0.0:
        return {
            "report_id": COMPONENT_REPORT_IDS["semantic"],
            "status": "unknown",
            "score": 0.0,
            "known_coverage": 0.0,
            "family_weights": dict(policy.semantic_family_weights),
            "missing_family_policy": policy.missing_family_policy,
            "families": family_results,
        }

    score_total = 0.0
    coverage_total = 0.0
    any_unknown = False
    for family, configured_weight, row in applicable:
        effective_weight = configured_weight / denominator
        score = _number(row.get("score")) if report_usable else None
        coverage = _number(row.get("known_coverage")) if report_usable else None
        if score is None or coverage is None:
            score, coverage = 0.0, 0.0
            any_unknown = True
        score_total += effective_weight * score
        coverage_total += effective_weight * coverage
        family_results[family] = {
            "applicable": True,
            "configured_weight": configured_weight,
            "effective_weight": _round4(effective_weight),
            "score": _round4(score),
            "known_coverage": _round4(coverage),
            "active_requirement_count": row.get("active_requirement_count"),
            "clause_count": row.get("clause_count"),
        }
    return {
        "report_id": COMPONENT_REPORT_IDS["semantic"],
        "status": "partial" if any_unknown or coverage_total < 1.0 else "measured",
        "score": _round4(score_total),
        "known_coverage": _round4(coverage_total),
        "family_weights": dict(policy.semantic_family_weights),
        "missing_family_policy": policy.missing_family_policy,
        "families": family_results,
    }


def _scalar_component(
    report: Mapping[str, Any] | None,
    *,
    report_id: str,
) -> dict[str, Any]:
    score = _number(report.get("score")) if _usable(report) else None
    if score is None:
        return {
            "report_id": report_id,
            "status": "unknown",
            "score": 0.0,
            "known_coverage": 0.0,
            "source_status": (
                report.get("status")
                if isinstance(report, Mapping)
                else "missing"
            ),
        }
    return {
        "report_id": report_id,
        "status": "measured",
        "score": _round4(score),
        "known_coverage": 1.0,
        "source_status": report.get("status"),
    }


def _physics_component(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Physical Safety from its two leaves under the paper protocol."""
    measured = primary_score.physics_component(report, weight=0.0)
    coverage = measured["known_coverage"]
    return {
        "report_id": COMPONENT_REPORT_IDS["physics"],
        "status": "unknown" if coverage == 0 else "partial" if coverage < 1 else "measured",
        "score": _round4(measured["score"]),
        "known_coverage": coverage,
        "source_status": report.get("status", "missing") if report else "missing",
        "missing_evidence": measured["missing_evidence"],
        "leaves": measured["leaves"],
        "scoring_tree": measured["scoring_tree"],
        "policy_id": physics_score.POLICY_ID,
    }


def aggregate_reports(
    reports: Sequence[Mapping[str, Any]],
    policy: ScorePolicy,
    *,
    activate_reports: bool = False,
) -> dict[str, Any]:
    """Aggregate one scalar; missing evidence keeps its weight and earns zero."""

    by_id = {
        str(report.get("report_id")): report
        for report in reports
        if isinstance(report, Mapping)
    }
    components = {
        "semantic": _semantic_component(
            by_id.get(COMPONENT_REPORT_IDS["semantic"]), policy
        ),
        "physics": _physics_component(
            by_id.get(COMPONENT_REPORT_IDS["physics"]),
        ),
        "holistic_overview": _scalar_component(
            by_id.get(COMPONENT_REPORT_IDS["holistic_overview"]),
            report_id=COMPONENT_REPORT_IDS["holistic_overview"],
        ),
    }
    quality_score = sum(
        float(policy.total_weights[name]) * float(components[name]["score"])
        for name in COMPONENT_ORDER
    )
    quality_coverage = sum(
        float(policy.total_weights[name]) * float(components[name]["known_coverage"])
        for name in COMPONENT_ORDER
    )
    outcome = case_outcome.from_reports(reports)
    hard_gate_applied = outcome["classification"] in {case_outcome.MODEL_INVALID, case_outcome.INVALID}
    score = 0.0 if hard_gate_applied else quality_score
    coverage = 1.0 if hard_gate_applied else quality_coverage
    for name in COMPONENT_ORDER:
        components[name]["weight"] = float(policy.total_weights[name])

    if activate_reports:
        for name in COMPONENT_ORDER:
            source = by_id.get(COMPONENT_REPORT_IDS[name])
            if not isinstance(source, dict):
                continue
            source["contributes_to_aggregate"] = True
            source["score_role"] = "weighted_overall_component"
            source["aggregate_weight"] = float(policy.total_weights[name])
            source["active_score_policy_id"] = policy.policy_id
            if name == "holistic_overview" and source.get("report_only_reason"):
                source["default_without_score_policy"] = source.pop(
                    "report_only_reason"
                )

        semantic_report = by_id.get(COMPONENT_REPORT_IDS["semantic"])
        if isinstance(semantic_report, dict):
            semantic = components["semantic"]
            metrics = semantic_report.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
                semantic_report["metrics"] = metrics
            metrics.setdefault(
                "score_before_score_policy", semantic_report.get("score")
            )
            metrics.setdefault("score_policy_before", metrics.get("score_policy"))
            metrics.setdefault(
                "semantic_family_weights_before_score_policy",
                metrics.get("semantic_family_weights"),
            )
            semantic_report["score"] = semantic["score"]
            metrics["semantic_score"] = semantic["score"]
            metrics["weighted_semantic_score"] = semantic["score"]
            metrics["weighted_coverage"] = semantic["known_coverage"]
            metrics["known_coverage"] = semantic["known_coverage"]
            metrics["semantic_family_weights"] = dict(
                policy.semantic_family_weights
            )
            raw_subscores = metrics.get("semantic_subscores")
            if isinstance(raw_subscores, dict):
                for family, family_result in semantic["families"].items():
                    raw_family = raw_subscores.get(family)
                    if not isinstance(raw_family, dict):
                        continue
                    raw_family["family_weight"] = family_result[
                        "configured_weight"
                    ]
                    raw_family["effective_family_weight"] = family_result[
                        "effective_weight"
                    ]
            metrics["semantic_subscores_policy"] = semantic["families"]
            metrics["score_policy"] = f"{policy.policy_id}.semantic"

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "policy": policy.provenance(),
        "score": _round4(score),
        "known_coverage": _round4(coverage),
        "case_outcome": outcome,
        "hard_gate": {
            "applied": hard_gate_applied,
            "rule": "candidate_invalid_fixed_zero_v2",
            "score": 0.0 if hard_gate_applied else None,
        },
        "quality_before_gate": {
            "score": _round4(quality_score),
            "known_coverage": _round4(quality_coverage),
        },
        "published_score": policy.published_score,
        "missing_component_policy": policy.missing_component_policy,
        "components": components,
    }


def apply_to_result(
    result: Mapping[str, Any],
    policy: ScorePolicy,
) -> dict[str, Any]:
    """Return a policy-augmented copy of an existing artifact result."""

    rebuilt = without_intervals(result)
    reports = rebuilt.get("reports")
    if not isinstance(reports, list):
        raise ScorePolicyError("artifact result must contain a reports list")
    aggregation = aggregate_reports(reports, policy, activate_reports=True)
    rebuilt["case_outcome"] = aggregation["case_outcome"]
    rebuilt["score_policy_result"] = aggregation
    primary = primary_score.from_reports(
        reports,
        score_policy_result=aggregation,
    )
    rebuilt.update(primary_score.result_fields(primary))
    rebuilt["score_breakdown"] = primary_score.breakdown(reports, primary)
    return rebuilt


__all__ = [
    "default_for_task",
    "COMPONENT_ORDER",
    "RESULT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ScorePolicy",
    "ScorePolicyError",
    "aggregate_reports",
    "apply_to_result",
    "load",
    "validate_task",
]
