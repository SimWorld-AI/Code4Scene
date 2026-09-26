"""``ground_gap`` — how far the placed Actors float above the ground they sit on.

One property, measured once per physics assertion the task declares. Version
``ground-gap-v1`` freezes the choices:

* the population comes from the assertion's ``candidate_all`` selector, not
  from the separately configured measurement-target list;
* only surface detection, a non-negative gap, and the measurement method are
  read — grounded, penetration and support are different properties and
  cannot move this number;
* a gap equal to the configured maximum passes; only a strictly larger gap is
  affected;
* P95 is the nearest-rank percentile over Actor gaps;
* missing surfaces, actors, methods, invalid values and ambiguous identities
  reduce coverage and WITHHOLD the score. A missing surface is not a zero gap.

``measure()`` collects and decides nothing. ``normalize()`` scores and
acquires nothing. Keeping those apart is what lets a threshold change be
re-scored from the stored measurement instead of re-run in an editor.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from .. import contracts
from .. import ue_evidence
from ..contracts import MetricResult, RawMeasurement
from ..case_spec import (validate_candidate_contract,
                                validate_case_contract)
from ..physics_evidence import (measurement_for_actor, measurement_records,
                               validate_measurement_envelope)
from ..selection import actor_identifier, select_candidate_actors
from ..values import as_text, nonnegative_finite
from ..context import Context, error



#: Which class this verifier's number belongs in. A `gt` verifier reads the
#: answer key — a `.label.json` or a canonical scene — and can only run where
#: that key exists; an `open_ended` one answers from the candidate alone. The
#: two are reported side by side and never averaged, which is why every
#: verifier has to say which it is.
CLASS = "open_ended"


@dataclass(frozen=True)
class GroundGapPolicy:
    requirement_id: str
    maximum_ground_gap_cm: float

    def __post_init__(self) -> None:
        value = self.maximum_ground_gap_cm
        if (
            not self.requirement_id
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("ground-gap policy needs an id and a finite non-negative maximum")


@dataclass(frozen=True)
class ActorGroundGapEvidence:
    actor_id: str
    label: str | None
    surface_detected: bool
    gap_cm: float
    measurement_method: str
    measurement_key: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "label": self.label,
            "surface_detected": self.surface_detected,
            "ground_gap_cm": self.gap_cm,
            "measurement_method": self.measurement_method,
            "measurement_key": self.measurement_key,
        }


@dataclass(frozen=True)
class MissingGroundGapEvidence:
    actor_id: str
    label: str | None
    missing_reason: str
    surface_detected: bool | None = None
    gap_cm: float | None = None
    measurement_method: str | None = None
    measurement_key: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "label": self.label,
            "surface_detected": self.surface_detected,
            "ground_gap_cm": self.gap_cm,
            "measurement_method": self.measurement_method,
            "measurement_key": self.measurement_key,
            "missing_reason": self.missing_reason,
        }


@dataclass(frozen=True)
class GroundGapEvidence:
    requirement_id: str
    target_actor_ids: tuple[str, ...]
    actors: tuple[ActorGroundGapEvidence, ...]
    missing: tuple[MissingGroundGapEvidence, ...]
    producer: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GroundGapMeasurement:
    """Policy-free raw values. ``gaps_cm`` is retained for normalization."""

    unit: str
    target_actor_count: int
    observed_actor_count: int
    gaps_cm: tuple[float, ...]
    mean: float | None
    p95: float | None
    max: float | None


@dataclass(frozen=True)
class GroundGapRaw:
    """The stable public raw payload stored in an ``MetricResult``."""

    unit: str
    target_actor_count: int
    observed_actor_count: int
    mean: float | None
    p95: float | None
    max: float | None
    affected_actor_count: int | None
    affected_actor_fraction: float | None


def _nearest_rank_p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


class GroundGapVerifier:
    id: ClassVar[str] = "physics.ground_gap"
    metric_version: ClassVar[str] = "ground-gap-v1"
    dimension: ClassVar[str] = "actor-to-ground surface separation"
    applicability_policy: ClassVar[str] = (
        "one result per physics assertion; the selected candidate_all population "
        "must contain at least one actor"
    )
    required_evidence: ClassVar[tuple[str, ...]] = (
        "candidate actor population",
        "surface_detected",
        "ground_gap_cm",
        "measurement_method",
        "physics measurement provenance",
    )
    normalization_policy: ClassVar[str] = "ground-gap-threshold-fraction-v1"
    calibration_status: ClassVar[str] = "not_human_validated_diagnostic"

    @classmethod
    def instance_id(cls, requirement_id: str) -> str:
        return f"{cls.id}:{requirement_id}"

    def measure(
        self, evidence: GroundGapEvidence
    ) -> RawMeasurement[GroundGapMeasurement]:
        target_count = len(evidence.target_actor_ids)
        instance_id = self.instance_id(evidence.requirement_id)
        if target_count == 0:
            return RawMeasurement(
                id=self.id,
                instance_id=instance_id,
                metric_version=self.metric_version,
                applicable=False,
                status="not_applicable",
                coverage=None,
                raw=None,
                evidence=({"applicability_reason": "selector_matched_no_candidate_actors"},),
            )

        gaps = tuple(actor.gap_cm for actor in evidence.actors)
        observed_count = len(gaps)
        coverage = observed_count / target_count
        raw = GroundGapMeasurement(
            unit="cm",
            target_actor_count=target_count,
            observed_actor_count=observed_count,
            gaps_cm=gaps,
            mean=(sum(gaps) / observed_count if observed_count else None),
            p95=_nearest_rank_p95(gaps),
            max=(max(gaps) if gaps else None),
        )
        output_evidence = ((evidence.producer,) if evidence.producer else ())
        output_evidence += tuple(actor.to_json_dict() for actor in evidence.actors)
        output_evidence += tuple(item.to_json_dict() for item in evidence.missing)
        complete = observed_count == target_count and not evidence.missing
        reason = None
        if not complete:
            details = ", ".join(
                f"{item.actor_id} ({item.missing_reason})" for item in evidence.missing[:8]
            )
            reason = (
                f"ground-gap evidence is incomplete for "
                f"{target_count - observed_count}/{target_count} target actor(s)"
                + (f": {details}" if details else "")
            )
        return RawMeasurement(
            id=self.id,
            instance_id=instance_id,
            metric_version=self.metric_version,
            applicable=True,
            status="measured" if complete else "not_evaluated",
            coverage=coverage,
            raw=raw,
            evidence=output_evidence,
            failure_reason=reason,
        )

    def normalize(
        self,
        measurement: RawMeasurement[GroundGapMeasurement],
        policy: GroundGapPolicy,
    ) -> MetricResult[GroundGapRaw]:
        if measurement.id != self.id or measurement.metric_version != self.metric_version:
            raise ValueError("ground-gap normalizer received another metric's measurement")
        if measurement.instance_id != self.instance_id(policy.requirement_id):
            raise ValueError("ground-gap measurement and policy requirement ids do not match")
        parameters = {"maximum_ground_gap_cm": policy.maximum_ground_gap_cm}
        raw = measurement.raw
        public_raw = (
            GroundGapRaw(
                unit=raw.unit,
                target_actor_count=raw.target_actor_count,
                observed_actor_count=raw.observed_actor_count,
                mean=raw.mean,
                p95=raw.p95,
                max=raw.max,
                affected_actor_count=None,
                affected_actor_fraction=None,
            )
            if raw is not None
            else None
        )
        common = {
            "id": self.id,
            "instance_id": measurement.instance_id,
            "metric_version": self.metric_version,
            "dimension": self.dimension,
            "applicability_policy": self.applicability_policy,
            "required_evidence": self.required_evidence,
            "coverage": measurement.coverage,
            "raw": public_raw,
            "normalization_policy": self.normalization_policy,
            "normalization_parameters": parameters,
            "calibration_status": self.calibration_status,
            "contributes_to_aggregate": False,
            "evidence": measurement.evidence,
        }
        if measurement.status == "not_applicable":
            return MetricResult(
                **common,
                applicable=False,
                status="not_applicable",
                score=None,
                failure_reason=None,
            )
        if measurement.status != "measured" or raw is None:
            return MetricResult(
                **common,
                applicable=True,
                status="not_evaluated",
                score=None,
                failure_reason=measurement.failure_reason,
            )

        affected_count = sum(
            gap > policy.maximum_ground_gap_cm for gap in raw.gaps_cm
        )
        affected_fraction = affected_count / raw.target_actor_count
        public_raw = GroundGapRaw(
            unit=raw.unit,
            target_actor_count=raw.target_actor_count,
            observed_actor_count=raw.observed_actor_count,
            mean=raw.mean,
            p95=raw.p95,
            max=raw.max,
            affected_actor_count=affected_count,
            affected_actor_fraction=affected_fraction,
        )
        failed = affected_count > 0
        return MetricResult(
            **{**common, "raw": public_raw},
            applicable=True,
            status="fail" if failed else "pass",
            score=1.0 - affected_fraction,
            failure_reason=(
                f"{affected_count}/{raw.target_actor_count} target actor(s) exceed "
                f"{policy.maximum_ground_gap_cm:g} cm ground gap"
                if failed
                else None
            ),
        )

    def unavailable(
        self, policy: GroundGapPolicy, reason: str, evidence: Mapping[str, Any]
    ) -> MetricResult[GroundGapRaw]:
        measurement: RawMeasurement[GroundGapMeasurement] = RawMeasurement(
            id=self.id,
            instance_id=self.instance_id(policy.requirement_id),
            metric_version=self.metric_version,
            applicable=True,
            status="not_evaluated",
            coverage=None,
            raw=None,
            evidence=(evidence,),
            failure_reason=reason,
        )
        return self.normalize(measurement, policy)

    def invalid_policy(
        self, requirement_id: str, reason: str, received: Any
    ) -> MetricResult[GroundGapRaw]:
        received_json = received
        if (
            not isinstance(received, (str, int, float, bool, type(None)))
            or isinstance(received, float) and not math.isfinite(received)
        ):
            received_json = repr(received)
        return MetricResult(
            id=self.id,
            instance_id=self.instance_id(requirement_id),
            metric_version=self.metric_version,
            dimension=self.dimension,
            applicability_policy=self.applicability_policy,
            required_evidence=self.required_evidence,
            applicable=True,
            status="not_evaluated",
            coverage=None,
            raw=None,
            score=None,
            normalization_policy=None,
            normalization_parameters={},
            calibration_status=self.calibration_status,
            contributes_to_aggregate=False,
            evidence=({
                "policy_validation_error": reason,
                "received_type": type(received).__name__,
                "received_value": received_json,
            },),
            failure_reason=f"invalid ground-gap normalization policy: {reason}",
        )



def _evidence_for_targets(
    selected: Sequence[Mapping[str, Any]], measurements: Any,
    producer: Mapping[str, Any],
) -> GroundGapEvidence:
    records = measurement_records(measurements)
    labels = [as_text(actor.get("label")) for actor in selected]
    label_counts = Counter(label for label in labels if label is not None)
    observed: list[ActorGroundGapEvidence] = []
    missing: list[MissingGroundGapEvidence] = []
    target_ids: list[str] = []
    used_keys: set[str] = set()
    for index, actor in enumerate(selected):
        actor_id = actor_identifier(actor, index)
        label = as_text(actor.get("label"))
        target_ids.append(actor_id)
        item, key, match_error = measurement_for_actor(
            actor, records, label_counts, used_keys
        )
        if match_error or item is None or key is None:
            missing.append(MissingGroundGapEvidence(
                actor_id=actor_id,
                label=label,
                missing_reason=match_error or "measurement_missing",
                measurement_key=key,
            ))
            continue
        surface = item.get("surface_detected")
        method = as_text(item.get("measurement_method"))
        gap = nonnegative_finite(item.get("ground_gap_cm"))
        reason = None
        if surface is not True:
            reason = "surface_not_detected" if surface is False else "surface_detection_missing"
        elif gap is None:
            reason = "ground_gap_missing_or_invalid"
        elif method is None:
            reason = "measurement_method_missing"
        if reason:
            missing.append(MissingGroundGapEvidence(
                actor_id=actor_id,
                label=label,
                missing_reason=reason,
                surface_detected=surface if isinstance(surface, bool) else None,
                gap_cm=gap,
                measurement_method=method,
                measurement_key=key,
            ))
            continue
        observed.append(ActorGroundGapEvidence(
            actor_id=actor_id,
            label=label,
            surface_detected=True,
            gap_cm=gap,
            measurement_method=method,
            measurement_key=key,
        ))
    return GroundGapEvidence(
        requirement_id="",
        target_actor_ids=tuple(target_ids),
        actors=tuple(observed),
        missing=tuple(missing),
        producer=producer,
    )



def _invalid_contract_result(errors: Sequence[str]) -> MetricResult[GroundGapRaw]:
    verifier = GroundGapVerifier()
    return MetricResult(
        id=verifier.id,
        instance_id=verifier.instance_id("invalid-contract"),
        metric_version=verifier.metric_version,
        dimension=verifier.dimension,
        applicability_policy=verifier.applicability_policy,
        required_evidence=verifier.required_evidence,
        applicable=True,
        status="not_evaluated",
        coverage=None,
        raw=None,
        score=None,
        normalization_policy=None,
        normalization_parameters={},
        calibration_status=verifier.calibration_status,
        contributes_to_aggregate=False,
        evidence=({"contract_validation_errors": list(errors)},),
        failure_reason="invalid ground-gap evidence contract: " + "; ".join(errors),
    )


def _not_configured_result() -> MetricResult[GroundGapRaw]:
    verifier = GroundGapVerifier()
    return MetricResult(
        id=verifier.id,
        instance_id=verifier.instance_id("not-configured"),
        metric_version=verifier.metric_version,
        dimension=verifier.dimension,
        applicability_policy=verifier.applicability_policy,
        required_evidence=verifier.required_evidence,
        applicable=False,
        status="not_applicable",
        coverage=None,
        raw=None,
        score=None,
        normalization_policy=None,
        normalization_parameters={},
        calibration_status=verifier.calibration_status,
        contributes_to_aggregate=False,
        evidence=({"applicability_reason": "physics_assertion_not_configured"},),
    )


def _ground_gap_policy(
    assertion: Mapping[str, Any], requirement_id: str
) -> tuple[GroundGapPolicy | None, str | None]:
    value = assertion.get("maximum_ground_gap_cm")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "maximum_ground_gap_cm must be a JSON number, not a coerced value"
    if not math.isfinite(float(value)) or float(value) < 0:
        return None, "maximum_ground_gap_cm must be finite and non-negative"
    return GroundGapPolicy(requirement_id, float(value)), None


def ground_gap_results(
    candidate: Any,
    measurements: Mapping[str, Any] | None,
    case_spec: Any,
    collection_provenance: Mapping[str, Any] | None = None,
    expected_case_id: str | None = None,
) -> tuple[MetricResult[GroundGapRaw], ...]:
    """One typed result per physics assertion the case declares."""
    contract_errors, assertions = validate_case_contract(case_spec, expected_case_id)
    if contract_errors:
        return (_invalid_contract_result(contract_errors),)
    physics = [
        item for item in assertions if item.get("primitive") == "physics"
    ]
    if not physics:
        return (_not_configured_result(),)

    candidate_errors = validate_candidate_contract(candidate)
    if candidate_errors:
        return (_invalid_contract_result(candidate_errors),)

    candidate_actors = [
        actor for actor in candidate.get("actors") or [] if isinstance(actor, Mapping)
    ]
    producer, envelope_errors = validate_measurement_envelope(
        candidate, measurements, collection_provenance,
    )
    producer = {
        **producer,
        "contract_validation": {"validator": "atomic.population", "valid": True},
    }
    verifier = GroundGapVerifier()
    results: list[MetricResult[GroundGapRaw]] = []
    for assertion in physics:
        requirement_id = as_text(assertion.get("id")) or "unnamed-physics-assertion"
        policy, policy_error = _ground_gap_policy(assertion, requirement_id)
        if policy is None:
            results.append(verifier.invalid_policy(
                requirement_id,
                policy_error or "unknown policy error",
                assertion.get("maximum_ground_gap_cm"),
            ))
            continue
        selector_value = assertion.get("target_selector")
        selector = selector_value if isinstance(selector_value, Mapping) else {}
        scope = as_text(selector.get("scope") or assertion.get("scope")) or "primary_additions"
        if scope != "candidate_all":
            results.append(verifier.unavailable(
                policy,
                f"ground-gap-v1 does not yet resolve actor scope {scope!r}",
                {"unsupported_scope": scope},
            ))
            continue
        selected = select_candidate_actors(candidate_actors, selector)
        if selected and envelope_errors:
            results.append(verifier.unavailable(
                policy,
                "invalid physics measurement envelope: " + "; ".join(envelope_errors),
                {**producer, "evidence_validation_errors": list(envelope_errors)},
            ))
            continue
        evidence = _evidence_for_targets(selected, measurements, producer)
        evidence = GroundGapEvidence(
            requirement_id=requirement_id,
            target_actor_ids=evidence.target_actor_ids,
            actors=evidence.actors,
            missing=evidence.missing,
            producer=evidence.producer,
        )
        results.append(verifier.normalize(verifier.measure(evidence), policy))
    return tuple(results)


def evaluate_ground_gap(
    candidate: Any,
    measurements: Mapping[str, Any] | None,
    case_spec: Any,
    collection_provenance: Mapping[str, Any] | None = None,
    expected_case_id: str | None = None,
) -> list[dict[str, Any]]:
    """The JSON boundary: typed results as the dicts a report carries."""
    return [result.to_json_dict() for result in ground_gap_results(
        candidate, measurements, case_spec, collection_provenance, expected_case_id,
    )]


def verify(context: Context) -> dict[str, Any]:
    """Collect the evidence, measure the one property, report the worst case.

    One report, N metric results — one per physics assertion. The report's
    score is the WORST of them, not their average: a task that asserts two
    populations is only as grounded as the one that floats. Results that
    withheld a score (missing evidence, unresolved scope) do not dilute it
    either — they make the report an error, because a number that silently
    covers half the assertions is worse than no number.
    """
    try:
        evidence = ue_evidence.collect(context)
    except Exception as e:          # noqa: BLE001 — missing evidence is a report
        return error("ground_gap", context, f"{type(e).__name__}: {e}")

    results = evaluate_ground_gap(
        evidence.candidate, evidence.measurements, evidence.case_spec,
        evidence.provenance(), evidence.case_id,
    )
    path = evidence.root / "metric-results.json"
    ue_evidence.write_json(path, results)

    scored = [r["score"] for r in results if r.get("score") is not None]
    # `not_applicable` was excluded here, so a case declaring no physics
    # assertion — and one whose selector matched nothing — fell through to the
    # PASS body carrying score 0.0: a pass with a zero, for work that never
    # ran. `assertions.report` turns the same condition into an error, and the
    # two must agree.
    withheld = [r for r in results if r["status"] not in ("pass", "fail")]
    report = {
        **contracts.base("ground_gap", context.ids),
        "status": contracts.PASS,
        "score": min(scored) if scored else 0.0,
        "metrics": {"metric_results": results,
                    "measured_result_count": len(scored),
                    "result_count": len(results)},
        "evidence": {**evidence.evidence(),
                     "metric_result_count": len(results)},
        "artifacts": {**evidence.artifacts(), "metric_results": str(path)},
        "probes_used": evidence.probes_used(),
    }
    if withheld or not results:
        reasons = [r.get("failure_reason") for r in withheld if r.get("failure_reason")]
        return {**report, "status": contracts.ERROR, "score": None,
                "failure_reason": (
                    "; ".join(str(r) for r in reasons[:4]) if reasons
                    else "the task declares no physics assertion to measure, "
                         "so nothing was checked; a pass here would credit a "
                         "scene for a requirement nobody stated")}
    if any(r["status"] == "fail" for r in results):
        failed = [r for r in results if r["status"] == "fail"]
        return {**report, "status": contracts.FAIL,
                "failure_reason": "; ".join(
                    str(r.get("failure_reason")) for r in failed[:4])}
    return report


__all__ = [
    "ActorGroundGapEvidence",
    "GroundGapEvidence",
    "GroundGapPolicy",
    "GroundGapRaw",
    "GroundGapVerifier",
    "MissingGroundGapEvidence",
    "evaluate_ground_gap",
    "ground_gap_results",
    "verify",
]
