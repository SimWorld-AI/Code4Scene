"""``reachability`` — the share of placed Actors an embodied agent can walk up to.

A scene can be collision-clean, grounded and inside its plate and still be
unusable: a room whose only door is blocked by the wardrobe that was placed in
front of it scores 1.0 on every geometric metric this benchmark had.
Executability is the missing axis, and a navmesh is how the engine answers it.

Version ``reachability-v1`` freezes the choices:

* the population is every placed Actor with collision. An Actor with none is
  not a destination and never was — a BoxReflectionCapture standing in the
  middle of a room projects nothing and would score the room unreachable;
* an Actor counts as reachable when its footprint projects onto the navmesh
  within the probe's projection extent AND a navigation path exists from there
  to the largest path-connected component;
* the score is ``connected / eligible``. One question, one number;
* an unbuilt, empty or unsettled navmesh WITHHOLDS the score rather than
  reporting zero, and so does a scene where most eligible Actors do not project
  onto the navmesh at all, and so does a component pass that ran out of query
  budget — a truncated union-find does not report a slower answer, it reports a
  different one;
* the score and the component structure are certified apart. The probe's fast
  settle was certified against repeated live rebuilds for the approachability
  share (a stated bias of at most 0.01, downward) and NOT for the component
  statistics, which ran 2.75x the live spread under it. So a payload that only
  ever saw the fast settle still scores, and its ``component_count`` and
  ``largest_component_size`` are withheld from the reported raw unless a
  multi-pass spread came with them. Two questions, one settle test, and only
  one of them is answered by it.

``measure()`` reads the probe payload and decides nothing about the threshold.
``normalize()`` scores and opens no editor. A projection-floor change is
re-scored from the stored measurement instead of re-run against a rebuilt
navmesh, which takes minutes per level.

The Embodied Utility Score
--------------------------

``embodied_utility()`` derives one scene-level number from a normalized
result: ``EUS = A x W`` in [0, 1], where

* ``A`` is the approachability share above — this verifier's own score;
* ``W`` is the walk-validation rate, the fraction of sampled oracle routes a
  walker completed without stalling. Code4Scene has no walker yet, so
  ``W`` is an OPTIONAL payload input, read from
  ``walk_validation.completed_route_fraction``. **When it is absent the EUS is
  reported as ``A`` under the label "A-only, walk-validation pending"** — a
  provisional that says so,
  never a silent ``W = 1.0``. A projected navmesh point is a claim that a path
  exists; a completed route is a claim that an agent walked it. Multiplying a
  missing second claim by one asserts it.

Two rules decide zero from absent, and they are not the same rule:

* **the empty-world rule.** Below ``MINIMUM_USEFUL_ACTOR_COUNT`` eligible
  Actors the EUS is ``0.0``, SCORED. Utility requires something to use, and a
  scene with three chairs in it is not a scene an agent can be embodied in
  however cleanly it can walk to all three. This is a definitional stance, not
  a measurement: it holds even when the navmesh never built, because no
  measurement can add actors to a level that has none;
* **the absent line.** Every measurement failure — a degenerate or empty
  navmesh, a settle that never converged, a scene under the projection floor,
  a truncated union-find — WITHHOLDS the EUS. "Nothing there" is a fact about
  the scene; "we could not see" is a fact about the measurement, and a
  leaderboard column that renders both as ``0.0`` has published the second as
  the first.

Combining EUS ACROSS scenes is deliberately not here. The mean-over-measured-
scenes and its mandatory ``measured/submitted`` coverage figure are a decision
about comparing runs, and this repo puts those in the report layer:
``composite.composite_report`` already publishes exactly that shape — an
unweighted macro mean next to a ``score_coverage`` — and ``aggregate.py``
carries the same rule for the geometry rates. ``navigation.reachability`` is
currently unmapped in ``components.COMPONENTS``; when the team re-registers
it, ``reachability.embodied_utility`` is the ``report_path`` to register, and
the aggregation is then free.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from .. import contracts
from .. import ue_evidence
from ..contracts import MetricResult, RawMeasurement
from ..context import Context, error
from ..values import as_text


#: Which class this verifier's number belongs in. Nothing here reads an answer
#: key: the navmesh is a property of the scene the agent handed back.
CLASS = "open_ended"

#: Fewer eligible Actors than this and the scene has nothing to be used, so its
#: embodied utility is zero rather than unmeasured. Five is the smallest
#: population over which the approachability share is a share of anything: at
#: four it moves in steps of 0.25 and at one it is 0.0 or 1.0, so a near-empty
#: level is the cheapest way to buy a perfect score off this metric. The
#: threshold is a stance about what a usable scene is, not a measurement
#: property, which is why it is a constant here and not a policy field — a run
#: that could tune it could tune its way back to rewarding empty rooms.
MINIMUM_USEFUL_ACTOR_COUNT = 5


@dataclass(frozen=True)
class ReachabilityPolicy:
    """The one number normalization needs, and the one it may withhold on."""

    scene_id: str
    #: Below this share of eligible Actors projecting onto the navmesh at all,
    #: the navmesh is not describing this scene and the score is withheld.
    minimum_projection_fraction: float = 0.5

    def __post_init__(self) -> None:
        value = self.minimum_projection_fraction
        if (
            not self.scene_id
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(
                "reachability policy needs a scene id and a projection floor in [0, 1]")


@dataclass(frozen=True)
class ReachabilityMeasurement:
    """Policy-free raw values, everything normalization is allowed to read."""

    eligible_actor_count: int
    projected_actor_count: int
    connected_actor_count: int
    ineligible_actor_count: int
    navmesh_present: bool
    navmesh_settled: bool
    navmesh_sample_count: int
    component_count: int
    largest_component_size: int
    agent_radius_cm: float | None
    #: Which settle produced this navmesh: ``witnessed_drain`` (fast) or
    #: ``conservative``. A payload from before the fast path existed says
    #: nothing, and was the conservative one.
    settle_path: str = "conservative"
    #: Settled settle-and-regroup passes behind the component numbers. Two or
    #: more make a spread; one makes a point estimate.
    settled_component_passes: int = 0
    #: The union-find stopped early on its query budget, so its components are
    #: partly manufactured.
    component_query_budget_truncated: bool = False


@dataclass(frozen=True)
class ReachabilityRaw:
    """The stable public raw payload stored in a ``MetricResult``."""

    eligible_actor_count: int
    projected_actor_count: int
    connected_actor_count: int
    ineligible_actor_count: int
    navmesh_present: bool
    navmesh_settled: bool
    navmesh_sample_count: int
    #: None when the component structure is withheld — see
    #: ``component_structure_withheld_reason``. The score is not withheld with
    #: it: the share was certified under the fast settle and these were not.
    component_count: int | None
    largest_component_size: int | None
    agent_radius_cm: float | None
    settle_path: str = "conservative"
    settled_component_passes: int = 0
    component_query_budget_truncated: bool = False
    component_structure_withheld_reason: str | None = None
    projected_actor_fraction: float | None = None
    unreachable_actor_count: int | None = None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _public_raw(raw: ReachabilityMeasurement, **scored: Any) -> ReachabilityRaw:
    """The reported raw, minus any component structure nothing certifies."""
    reason = None
    if raw.component_query_budget_truncated:
        reason = ("the component union-find truncated on its query budget, so its "
                  "components are partly made of samples it never tested")
    elif raw.settle_path == "witnessed_drain" and raw.settled_component_passes < 2:
        reason = (
            "the navmesh was settled on the witnessed-drain path only. That settle "
            "is certified for the reachable share and not for component structure, "
            "which ran 2.75x the live rebuild spread under it; report these from a "
            "conservative settle, or from component_spread_passes above 1")
    fields = dict(vars(raw))
    if reason is not None:
        fields.update(component_count=None, largest_component_size=None)
    return ReachabilityRaw(**fields, component_structure_withheld_reason=reason,
                           **scored)


def _settled_passes(spread: Any) -> int:
    """How many settle-and-regroup passes the probe actually settled."""
    if not isinstance(spread, list):
        return 0
    return sum(1 for entry in spread
               if isinstance(entry, Mapping) and entry.get("settled") is True)


def _agent_radius(navmesh: Mapping[str, Any]) -> float | None:
    """What the navmesh actor said about itself after the rebuild, not before."""
    parameters = _mapping(navmesh.get("parameters_after_rebuild"))
    for key in ("agent_radius", "nav_data_config.agent_radius"):
        value = parameters.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def producer_summary(payload: Any,
                     collection_provenance: Mapping[str, Any] | None = None,
                     ) -> dict[str, Any]:
    """Where the numbers came from, on every report that carries them."""
    payload = _mapping(payload)
    provenance = _mapping(payload.get("runtime_provenance"))
    diagnostics = _mapping(payload.get("diagnostics"))
    navmesh = _mapping(payload.get("navmesh"))
    components = _mapping(payload.get("components"))
    spread = components.get("spread")
    summary = {
        "evidence_type": "reachability_measurement_producer",
        "present": bool(payload),
        "schema_version": as_text(payload.get("schema_version")),
        "measurement_type": as_text(payload.get("measurement_type")),
        "map_path": as_text(payload.get("map_path")),
        "runtime_provenance": {
            "project_file_path": as_text(provenance.get("project_file_path")),
            "engine_version": as_text(provenance.get("engine_version")),
            "map_path": as_text(provenance.get("map_path")),
        },
        "navmesh_parameters_after_rebuild": dict(
            _mapping(navmesh.get("parameters_after_rebuild"))),
        "navmesh_settle": dict(_mapping(navmesh.get("settle"))),
        # What the component stage cost and how it was capped, next to the
        # spread it was capped over: a truncated union-find is only readable as
        # a defect if the budget it hit is on the same page.
        "component_query_cost": dict(_mapping(components.get("cost"))),
        "component_spread": list(spread) if isinstance(spread, list) else [],
        "diagnostics": {
            "status": as_text(diagnostics.get("status")),
            "level_actor_count": diagnostics.get("level_actor_count"),
            "eligible_actor_count": diagnostics.get("eligible_actor_count"),
            "unresolved_count": len(diagnostics.get("unresolved") or [])
            if isinstance(diagnostics.get("unresolved"), list) else None,
            "measurement_error_count": len(diagnostics.get("measurement_errors") or [])
            if isinstance(diagnostics.get("measurement_errors"), list) else None,
        },
    }
    if collection_provenance:
        summary["collection_provenance"] = dict(collection_provenance)
    return summary


def validate_payload(payload: Any) -> tuple[str, ...]:
    """Is this the probe's payload, and is it sound enough to count?"""
    if not isinstance(payload, Mapping):
        return ("reachability payload must be an object",)
    errors: list[str] = []
    version = payload.get("schema_version")
    if not isinstance(version, str) or not version.startswith("0.1."):
        errors.append("schema_version must be a 0.1.x string")
    if payload.get("measurement_type") != "ue_editor_reachability":
        errors.append("measurement_type must be ue_editor_reachability")
    if not isinstance(payload.get("actors"), Mapping):
        errors.append("actors must be an object")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        errors.append("diagnostics must be an object")
    elif diagnostics.get("status") == "error":
        errors.append(
            "the probe reported an error: "
            f"{as_text(diagnostics.get('error')) or 'unknown error'}")
    # An unreadable W is rejected here rather than dropped by the reader: a
    # dropped one reads as "no walker ran" and scores HIGHER than a bad walk.
    if payload.get("walk_validation") is not None:
        rate = _mapping(payload.get("walk_validation")).get("completed_route_fraction")
        # NaN and the infinities fail the range comparison, so it is the only
        # check needed once the type is known.
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not 0.0 <= float(rate) <= 1.0):
            errors.append(
                "walk_validation.completed_route_fraction must be a number in [0, 1]")
    return tuple(errors)


def walk_validation_rate(payload: Any) -> float | None:
    """W: the share of sampled oracle routes a walker completed, if one ran.

    ``None`` means no walker ran, which is every payload today — never 1.0.
    ``validate_payload`` has already refused a present-but-unreadable rate.
    """
    rate = _mapping(_mapping(payload).get("walk_validation")).get(
        "completed_route_fraction")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        return None
    return float(rate)


def embodied_utility(result: MetricResult[ReachabilityRaw],
                     rate: float | None = None) -> dict[str, Any]:
    """Scene-level ``EUS = A x W``, or zero, or withheld — labelled which.

    ``result`` is this verifier's own normalized result and ``rate`` is the
    optional W. Nothing else is read, which is why this is a derived output of
    ``normalize()`` and not a report-layer function: it combines one verifier's
    evidence with one input that arrived on that verifier's own payload.
    Combining EUS across scenes is the report layer's job — see the module
    docstring.
    """
    # The only ``not_applicable`` this verifier has is "the level holds no
    # collidable actor", so an inapplicable result IS a zero-eligible one.
    eligible = (0 if result.status == "not_applicable"
                else result.raw.eligible_actor_count if result.raw is not None
                else None)
    # First, and ahead of every withholding condition: how many actors a level
    # holds is a fact about the scene that no failed measurement can revise.
    if eligible is not None and eligible < MINIMUM_USEFUL_ACTOR_COUNT:
        return {
            "embodied_utility_score": 0.0,
            "status": "scored",
            "basis": "empty_world",
            "walk_validation_rate": rate,
            "note": (f"{eligible} eligible actor(s), below the "
                     f"{MINIMUM_USEFUL_ACTOR_COUNT}-actor floor: there is nothing "
                     f"here to use, which is a score of zero and not an absence "
                     f"of one"),
        }
    if result.score is None:
        return {
            "embodied_utility_score": None,
            "status": "absent",
            "basis": "withheld",
            "walk_validation_rate": rate,
            "note": ("the approachability share was withheld, so the utility is "
                     "unmeasured rather than zero: "
                     + (result.failure_reason or "no reason recorded")),
        }
    if rate is None:
        return {
            "embodied_utility_score": result.score,
            "status": "scored",
            "basis": "approachability_only",
            "walk_validation_rate": None,
            "note": ("A-only, walk-validation pending: no walker ran, so this is "
                     "the approachability share alone and an upper bound on the "
                     "utility, not the product"),
        }
    return {
        # Rounded like every other combined number in this repo
        # (``composite_report``): a leaderboard cell is not the place to
        # publish 0.30000000000000004.
        "embodied_utility_score": round(result.score * rate, 4),
        "status": "scored",
        "basis": "approachability_x_walk_validation",
        "walk_validation_rate": rate,
        "note": None,
    }


class ReachabilityVerifier:
    id: ClassVar[str] = "navigation.reachability"
    metric_version: ClassVar[str] = "reachability-v1"
    dimension: ClassVar[str] = "share of placed actors an embodied agent can walk up to"
    applicability_policy: ClassVar[str] = (
        "one result per scene; the level must contain at least one placed actor "
        "with collision"
    )
    required_evidence: ClassVar[tuple[str, ...]] = (
        "placed actor population with collision",
        "navmesh build parameters read back after the rebuild",
        "resample settle test, and which settle path produced the navmesh",
        "navmesh footprint projection",
        "path-connected component membership, and what the union-find spent",
    )
    normalization_policy: ClassVar[str] = "reachability-connected-fraction-v1"
    calibration_status: ClassVar[str] = "not_human_validated_diagnostic"

    @classmethod
    def instance_id(cls, scene_id: str) -> str:
        return f"{cls.id}:{scene_id}"

    def measure(self, payload: Any, scene_id: str,
                collection_provenance: Mapping[str, Any] | None = None,
                ) -> RawMeasurement[ReachabilityMeasurement]:
        """Read the probe payload. Collects, counts, and decides no score."""
        instance_id = self.instance_id(scene_id)
        producer = producer_summary(payload, collection_provenance)
        payload_errors = validate_payload(payload)
        if payload_errors:
            return RawMeasurement(
                id=self.id,
                instance_id=instance_id,
                metric_version=self.metric_version,
                applicable=True,
                status="not_evaluated",
                coverage=None,
                raw=None,
                evidence=(producer, {"payload_validation_errors": list(payload_errors)}),
                failure_reason="invalid reachability evidence: " + "; ".join(payload_errors),
            )

        payload = _mapping(payload)
        records = [record for record in _mapping(payload.get("actors")).values()
                   if isinstance(record, Mapping)]
        eligible = [record for record in records if record.get("eligible") is True]
        ineligible = [record for record in records if record.get("eligible") is not True]
        if not eligible:
            return RawMeasurement(
                id=self.id,
                instance_id=instance_id,
                metric_version=self.metric_version,
                applicable=False,
                status="not_applicable",
                coverage=None,
                raw=None,
                evidence=(producer,
                          {"applicability_reason": "level_has_no_collidable_actor"}),
            )

        projected = [record for record in eligible
                     if isinstance(record.get("projected_point_cm"), list)]
        # A projected Actor whose path query never answered is not a
        # disconnected Actor; it is an Actor this pass could not see.
        unresolved = [record for record in projected
                      if not isinstance(record.get("connected"), bool)]
        connected = [record for record in projected if record.get("connected") is True]

        navmesh = _mapping(payload.get("navmesh"))
        components = _mapping(payload.get("components"))
        samples = _mapping(payload.get("samples"))
        settle = _mapping(navmesh.get("settle"))
        cost = _mapping(components.get("cost"))
        raw = ReachabilityMeasurement(
            eligible_actor_count=len(eligible),
            projected_actor_count=len(projected),
            connected_actor_count=len(connected),
            ineligible_actor_count=len(ineligible),
            navmesh_present=bool(as_text(navmesh.get("actor_path"))),
            navmesh_settled=settle.get("settled") is True,
            navmesh_sample_count=int(samples.get("accepted_count") or 0),
            component_count=int(components.get("count") or 0),
            largest_component_size=int(components.get("largest_size") or 0),
            agent_radius_cm=_agent_radius(navmesh),
            settle_path=as_text(settle.get("path")) or "conservative",
            settled_component_passes=_settled_passes(components.get("spread")),
            component_query_budget_truncated=(
                cost.get("query_budget_truncated") is True),
        )
        coverage = (len(projected) - len(unresolved)) / len(projected) if projected else 1.0
        if unresolved:
            return RawMeasurement(
                id=self.id,
                instance_id=instance_id,
                metric_version=self.metric_version,
                applicable=True,
                status="not_evaluated",
                coverage=coverage,
                raw=raw,
                evidence=(producer, {
                    "unresolved_actor_paths": [
                        as_text(record.get("actor_path")) for record in unresolved[:8]],
                }),
                failure_reason=(
                    f"{len(unresolved)}/{len(projected)} projected actor(s) have no "
                    f"path-query answer"),
            )
        return RawMeasurement(
            id=self.id,
            instance_id=instance_id,
            metric_version=self.metric_version,
            applicable=True,
            status="measured",
            coverage=1.0,
            raw=raw,
            evidence=(producer, {
                "unreachable_actor_paths": [
                    as_text(record.get("actor_path")) for record in projected
                    if record.get("connected") is False][:16],
                "unprojected_actor_paths": [
                    as_text(record.get("actor_path")) for record in eligible
                    if not isinstance(record.get("projected_point_cm"), list)][:16],
                "ineligible_reasons": sorted({
                    as_text(record.get("ineligible_reason")) or "unstated"
                    for record in ineligible}),
            }),
        )

    def normalize(
        self,
        measurement: RawMeasurement[ReachabilityMeasurement],
        policy: ReachabilityPolicy,
    ) -> MetricResult[ReachabilityRaw]:
        """Score connected/eligible, or withhold. Acquires no new evidence."""
        if measurement.id != self.id or measurement.metric_version != self.metric_version:
            raise ValueError("reachability normalizer received another metric's measurement")
        if measurement.instance_id != self.instance_id(policy.scene_id):
            raise ValueError("reachability measurement and policy scene ids do not match")
        raw = measurement.raw
        parameters = {"minimum_projection_fraction": policy.minimum_projection_fraction}
        public_raw = _public_raw(raw) if raw is not None else None
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
            return MetricResult(**common, applicable=False, status="not_applicable",
                                score=None, failure_reason=None)
        if measurement.status != "measured" or raw is None:
            return MetricResult(**common, applicable=True, status="not_evaluated",
                                score=None, failure_reason=measurement.failure_reason)

        projected_fraction = raw.projected_actor_count / raw.eligible_actor_count
        withheld = None
        if not raw.navmesh_present or raw.navmesh_sample_count == 0 or (
            raw.largest_component_size == 0
        ):
            withheld = (
                "the navmesh is empty or degenerate "
                f"(present={raw.navmesh_present}, samples={raw.navmesh_sample_count}, "
                f"largest_component={raw.largest_component_size}); an unbuilt navmesh "
                f"is not an unreachable scene")
        elif not raw.navmesh_settled:
            withheld = (
                "the navmesh rebuild never settled: two consecutive samples "
                "disagreed, so the geometry measured is not the geometry built")
        elif raw.component_query_budget_truncated:
            withheld = (
                "the component union-find stopped on its query budget, so the "
                "components it reports are partly made of samples it never tested "
                "and the largest one the actors were measured against may not be "
                "the largest; truncation is a different answer, not a slower one")
        elif projected_fraction < policy.minimum_projection_fraction:
            withheld = (
                f"only {raw.projected_actor_count}/{raw.eligible_actor_count} eligible "
                f"actor(s) project onto the navmesh, below the "
                f"{policy.minimum_projection_fraction:g} floor; the navmesh does not "
                f"cover the floors these actors stand on")
        if withheld:
            # The counts stay on the withheld result: a reader needs to see
            # WHICH of the three conditions fired without opening the payload.
            return MetricResult(
                **common,
                applicable=True,
                status="not_evaluated",
                score=None,
                failure_reason=withheld,
            )

        unreachable = raw.eligible_actor_count - raw.connected_actor_count
        public_raw = _public_raw(
            raw,
            projected_actor_fraction=projected_fraction,
            unreachable_actor_count=unreachable,
        )
        return MetricResult(
            **{**common, "raw": public_raw},
            applicable=True,
            status="fail" if unreachable else "pass",
            score=raw.connected_actor_count / raw.eligible_actor_count,
            failure_reason=(
                f"{unreachable}/{raw.eligible_actor_count} eligible actor(s) cannot be "
                f"walked up to by a {raw.agent_radius_cm:g} cm-radius agent"
                if unreachable and raw.agent_radius_cm is not None else
                f"{unreachable}/{raw.eligible_actor_count} eligible actor(s) cannot be "
                f"walked up to" if unreachable else None),
        )

    def unavailable(self, policy: ReachabilityPolicy, reason: str,
                    evidence: Mapping[str, Any]) -> MetricResult[ReachabilityRaw]:
        measurement: RawMeasurement[ReachabilityMeasurement] = RawMeasurement(
            id=self.id,
            instance_id=self.instance_id(policy.scene_id),
            metric_version=self.metric_version,
            applicable=True,
            status="not_evaluated",
            coverage=None,
            raw=None,
            evidence=(evidence,),
            failure_reason=reason,
        )
        return self.normalize(measurement, policy)


def evaluate_reachability(
    payload: Any,
    scene_id: str,
    minimum_projection_fraction: float = 0.5,
    collection_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The JSON boundary: one typed result as the dict a report carries."""
    verifier = ReachabilityVerifier()
    policy = ReachabilityPolicy(scene_id, minimum_projection_fraction)
    measurement = verifier.measure(payload, scene_id, collection_provenance)
    return verifier.normalize(measurement, policy).to_json_dict()


def _projection_floor(context: Context) -> float:
    value = context.spec.get("minimum_projection_fraction")
    if value is None:
        return 0.5
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("minimum_projection_fraction must be a JSON number")
    return float(value)


def verify(context: Context) -> dict[str, Any]:
    """Rebuild the navmesh, ask it who can be reached, and report one number."""
    try:
        floor = _projection_floor(context)
        root = ue_evidence.artifact_root(context)
        output = root / "candidate.reachability.json"
        options = context.spec.get("reachability_options") or {}
        if not isinstance(options, dict):
            raise ValueError("reachability_options must be an object")
        payload, collected_from, independent = ue_evidence.run_probe(
            context, ue_evidence.REACHABILITY_SCRIPT,
            {"SCENE_REACHABILITY_OUTPUT": str(output),
             "SCENE_REACHABILITY_OPTIONS_JSON": json.dumps(options)},
            output, "_SB_SCENE_REACHABILITY",
        )
    except Exception as e:               # noqa: BLE001 — missing evidence is a report
        return error("reachability", context, f"{type(e).__name__}: {e}")

    scene_id = ue_evidence.case_id(context)
    verifier = ReachabilityVerifier()
    normalized = verifier.normalize(
        verifier.measure(payload, scene_id, {
            "measurements_collected_from": collected_from,
            "measurements_from_independent_editor": independent,
        }),
        ReachabilityPolicy(scene_id, floor),
    )
    result = normalized.to_json_dict()
    # Beside the metric result, never inside it: the EUS is derived from a
    # scored MetricResult plus one optional input, and the typed result carries
    # only what this metric measured.
    utility = embodied_utility(normalized, walk_validation_rate(payload))
    path = root / "metric-results.reachability.json"
    ue_evidence.write_json(path, [result])
    report = {
        **contracts.base("reachability", context.ids),
        "status": contracts.PASS,
        "score": result["score"],
        "metrics": {"metric_results": [result], "embodied_utility": utility},
        "evidence": {"evidence_source": ue_evidence.EVIDENCE_SOURCE,
                     "measurements_collected_from": collected_from,
                     "measurements_from_independent_editor": independent},
        "artifacts": {"candidate_reachability": str(output),
                      "metric_results": str(path)},
        "probes_used": ["ue_reachability"],
    }
    if result["status"] in ("not_evaluated", "not_applicable"):
        # Absent, not zero: a scene whose navmesh could not answer did not fail
        # the question, and a leaderboard column defaulted to zero says it did.
        return {**report, "status": contracts.ERROR, "score": 0.0,
                "failure_reason": result.get("failure_reason")
                or "the level holds no actor an agent could be asked to walk to"}
    if result["status"] == "fail":
        return {**report, "status": contracts.FAIL,
                "failure_reason": result["failure_reason"]}
    return report


__all__ = [
    "MINIMUM_USEFUL_ACTOR_COUNT",
    "ReachabilityMeasurement",
    "ReachabilityPolicy",
    "ReachabilityRaw",
    "ReachabilityVerifier",
    "embodied_utility",
    "evaluate_reachability",
    "producer_summary",
    "validate_payload",
    "verify",
    "walk_validation_rate",
]
