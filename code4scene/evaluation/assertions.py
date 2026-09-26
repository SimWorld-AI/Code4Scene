"""The declared assertions a case carries, and what each one is asked about.

A task states what its scene must be true of in a frozen case specification —
a list of assertions, each naming a primitive (`structure`, `no_overlap`,
`spatial_relation`, …), the Actors it is about, and the thresholds it holds
them to. `case_spec.py` says whether that document is well-formed. This module
says which Actors an assertion is ABOUT, and lowers the resulting checks into
one verifier report.

Both halves are here because both were the same mistake waiting to happen. A
scope resolved slightly differently by two metrics means two metrics
describing two different populations of one scene and reporting numbers a
reader will compare. And a report envelope restated per verifier is how one of
them ends up scoring an assertion it could not evaluate: the rule the whole
scoring layer runs on is that missing evidence WITHHOLDS a number rather than
producing a low one, and it is enforced in exactly one place.

The score every assertion-driven verifier reports is the share of its declared
checks that passed. That is one number about one property — "how much of what
this task declared about spatial relations holds" — not a blend of unrelated
axes; combining across verifiers stays where it belongs, in `aggregate`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import contracts
from . import ue_evidence
from .case_spec import validate_case_contract
from .context import Context, error
from .scene_diff import (
    SceneDiff,
    diff_scenes,
    edited_candidate_actors,
    scene_actors,
)
from .scene_geometry import bounds_coverage
from .selection import select_candidate_actors
from .structure_rules import classify_additions, structure_spec
from .values import as_text

#: Populations an assertion may name, and where each comes from.
#:
#: `candidate_all` is the whole finished scene. The `*_additions` scopes are
#: what the edit ADDED, which needs the pre-edit scene to know — and the three
#: classified ones need the case's `structure` assertion to say which
#: additions were asked for. A scope that cannot be resolved is reported as
#: unresolved; it is never quietly read as the empty set, because every
#: assertion over an empty population passes.
SCOPES = ("candidate_all", "edited_actors", "all_additions", "source_actors",
          "primary_additions", "companion_additions", "unexpected_additions")

DEFAULT_SCOPE = "primary_additions"


class ScopeError(Exception):
    """An assertion names a population that cannot be resolved."""


@dataclass(frozen=True)
class Check:
    """One declared requirement, and whether the scene met it."""

    id: str
    status: str                                  # pass | fail | not_evaluated
    expected: Mapping[str, Any] = field(default_factory=dict)
    observed: Mapping[str, Any] = field(default_factory=dict)
    evidence: Sequence[Mapping[str, Any]] = ()
    failure_reason: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {"id": self.id, "status": self.status,
                "expected": dict(self.expected), "observed": dict(self.observed),
                "evidence": [dict(item) for item in self.evidence][:50],
                "failure_reason": self.failure_reason}


def check(check_id: str, ok: bool, expected: Mapping[str, Any],
          observed: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]] = (),
          failure_reason: str | None = None) -> Check:
    """A pass/fail check. A failure must say why; a pass may not."""
    return Check(id=check_id, status=contracts.PASS if ok else contracts.FAIL,
                 expected=expected, observed=observed, evidence=evidence,
                 failure_reason=None if ok else (failure_reason or "requirement not met"))


def unevaluated(check_id: str, reason: str,
                observed: Mapping[str, Any] | None = None) -> Check:
    """A check that COULD have run and did not: the evidence is missing.

    Withholds the whole report, because a rate over the half of a case that
    happened to be answerable is a different measurement wearing the same name.
    """
    return Check(id=check_id, status="not_evaluated", expected={},
                 observed=observed or {}, evidence=(), failure_reason=reason)


def inapplicable(check_id: str, reason: str,
                 observed: Mapping[str, Any] | None = None) -> Check:
    """A check nothing could ever answer here, and that is not a gap.

    Distinct from `unevaluated`, and the distinction is the whole point: "the
    probe failed on these Actors" is a hole in this run's evidence, while "the
    probe does not produce this measurement at all" is a property of the
    build. The first must withhold the report; the second must not, or a case
    that asks four questions of a probe that answers three can never be scored
    on any of them.
    """
    return Check(id=check_id, status="not_applicable", expected={},
                 observed=observed or {}, evidence=(), failure_reason=reason)


@dataclass(frozen=True)
class Case:
    """One episode's evidence, with the declared assertions resolved against it."""

    evidence: Any                                 # ue_evidence.SceneEvidence
    assertions: tuple[Mapping[str, Any], ...]
    diff: SceneDiff | None
    scope_error: str | None = None

    @property
    def candidate(self) -> list[Mapping[str, Any]]:
        return self.evidence.candidate_actors()

    @property
    def input_actors(self) -> list[Mapping[str, Any]]:
        return self.evidence.input_actors()

    def of(self, primitive: str) -> list[Mapping[str, Any]]:
        """Every declared assertion of one primitive, in declaration order."""
        return [item for item in self.assertions
                if item.get("primitive") == primitive]

    def resolve_scope(self, scope: str) -> list[Mapping[str, Any]]:
        """The Actors a scope names, or a `ScopeError` saying why not."""
        if scope == "candidate_all":
            return self.candidate
        if self.diff is None:
            # Two ways to have no diff, and they are different findings: the
            # task supplied no input_scene, or it did and the comparison
            # raised. Reporting the second as the first sent readers to the
            # task file for a bug that lives in the scene pair.
            if self.scope_error is not None:
                raise ScopeError(
                    f"scope {scope!r} is about what the edit changed, and "
                    f"comparing the candidate with the input_scene failed: "
                    f"{self.scope_error}")
            raise ScopeError(
                f"scope {scope!r} is about what the edit changed, and the task "
                f"supplied no input_scene to compare the candidate with")
        if scope == "all_additions":
            return list(self.diff.added)
        if scope == "edited_actors":
            return edited_candidate_actors(self.diff)
        if scope == "source_actors":
            return [change.after for change in self.diff.matched]
        spec = structure_spec(self.of("structure"))
        if spec is None:
            raise ScopeError(
                f"scope {scope!r} is defined by the case's structure assertion, "
                f"which this case does not declare")
        classified = classify_additions(list(self.diff.added), spec)
        by_scope = {"primary_additions": classified.primary,
                    "companion_additions": classified.companion,
                    "unexpected_additions": classified.unexpected}
        if scope not in by_scope:
            raise ScopeError(f"unsupported Actor scope {scope!r}")
        return by_scope[scope]

    def population(self, assertion: Mapping[str, Any],
                   default_scope: str = DEFAULT_SCOPE) -> list[Mapping[str, Any]]:
        """The Actors one assertion is about: its scope, then its selector."""
        selector_value = assertion.get("target_selector")
        selector = selector_value if isinstance(selector_value, Mapping) else None
        scope = as_text((selector or {}).get("scope")
                        or assertion.get("scope")) or default_scope
        actors = self.resolve_scope(scope)
        return select_candidate_actors(actors, selector) if selector else list(actors)

    def selection(self, selector: Any) -> list[Mapping[str, Any]]:
        """A relation rule's side: `{scope, …selector fields}`."""
        selector = selector if isinstance(selector, Mapping) else {}
        actors = self.resolve_scope(as_text(selector.get("scope")) or "candidate_all")
        return select_candidate_actors(actors, selector)


def load(context: Context) -> Case:
    """Collect the episode's evidence and validate the declared assertions."""
    evidence = ue_evidence.collect(context)
    errors, assertions = validate_case_contract(evidence.case_spec, evidence.case_id)
    if errors:
        raise ue_evidence.EvidenceError(
            "the case specification is not well-formed, so nothing declared in "
            "it can be scored: " + "; ".join(errors))
    diff = None
    scope_error = None
    if evidence.input_scene is not None:
        try:
            diff = diff_scenes(evidence.input_scene, evidence.candidate)
        except Exception as e:                    # noqa: BLE001 — carried
            scope_error = f"{type(e).__name__}: {e}"
    return Case(evidence=evidence, assertions=tuple(assertions), diff=diff,
                scope_error=scope_error)


def assertion_id(assertion: Mapping[str, Any], primitive: str) -> str:
    return as_text(assertion.get("id")) or f"unnamed-{primitive}-assertion"


def report(kind: str, context: Context, case: Case, checks: Sequence[Check],
           *, dimension: str, primitive: str,
           score: Any = None) -> dict[str, Any]:
    """Lower a verifier's checks into its one report and its one number.

    The score defaults to the share of declared checks that passed, which is
    right when each check is one declared requirement. A verifier whose one
    property is already a RATE — what share of the source scene survived
    untouched — passes ``score``, a callable over the checks, so that the
    number it reports is the rate rather than a boolean wearing a fraction's
    clothes. The checks still decide pass/fail either way.

    Three ways out without a score, and all three are refusals rather than
    zeros:

    * the task declares nothing of this primitive — `not_applicable`, reported
      as an error so it cannot be read as a clean bill of health for work that
      never ran;
    * a check could not be evaluated — the report is an error, because a rate
      over the half that could be measured is worse than no rate;
    * the evidence itself was missing — handled by the caller, which never
      reaches here.
    """
    evidence = case.evidence
    results = []
    for item in checks:
        value = item.to_json_dict()
        if item.status in {contracts.PASS, contracts.FAIL}:
            matched = item.status == contracts.PASS
            value.update({
                "status": contracts.MEASURED,
                "score": 1.0 if matched else 0.0,
                "outcome": "match" if matched else "mismatch",
                "rationale": item.failure_reason,
            })
            value.pop("failure_reason", None)
        else:
            value["score"] = None
            value["outcome"] = item.status
        results.append(value)
    base = {**contracts.base(kind, context.ids),
            "metrics": {"checks": results,
                        "check_count": len(results),
                        "not_applicable_check_count": sum(
                            item.status == "not_applicable" for item in checks),
                        "satisfied_check_count": sum(
                            item.status == contracts.PASS for item in checks)},
            "evidence": {**evidence.evidence(), "dimension": dimension,
                         "primitive": primitive,
                         "assertion_count": len(case.of(primitive))},
            "artifacts": evidence.artifacts(),
            "probes_used": evidence.probes_used()}
    if not checks:
        return {**base, "status": "not_applicable", "score": None,
                "failure_reason": (
                    f"the task declares no {primitive} assertion, so there was "
                    f"nothing to measure")}
    withheld = [item for item in checks if item.status == "not_evaluated"]
    if withheld:
        return {**base, "status": "not_evaluated", "score": None,
                "failure_reason": "; ".join(
                    str(item.failure_reason) for item in withheld[:4])}
    # `not_applicable` leaves the denominator: a question this build cannot
    # ask is not a question the scene failed, and counting it either way would
    # make the rate depend on what the probe happens to implement.
    decided = [item for item in checks if item.status != "not_applicable"]
    if not decided:
        return {**base, "status": "not_applicable", "score": None,
                "failure_reason": "; ".join(
                    str(item.failure_reason) for item in checks[:4])
                    or f"nothing this build measures answers any {primitive} check"}
    passing = sum(item.status == contracts.PASS for item in decided)
    value = score(checks) if callable(score) else passing / len(decided)
    body = {**base, "score": round(max(0.0, min(1.0, float(value))), 4)}
    return {**body, "status": contracts.MEASURED}


def run(kind: str, context: Context, primitive: str, dimension: str,
        produce: Any, score: Any = None) -> dict[str, Any]:
    """The whole envelope: collect, resolve, produce checks, lower.

    ``produce(case, assertion)`` returns the checks for one assertion. It may
    raise `ScopeError`, which becomes a withheld check rather than a failure —
    an assertion whose population could not be resolved was not evaluated, and
    a scene is not wrong because the case named a scope this build cannot
    resolve.
    """
    try:
        case = load(context)
    except Exception as e:                        # noqa: BLE001 — reported
        return error(kind, context, f"{type(e).__name__}: {e}")
    checks: list[Check] = []
    for assertion in case.of(primitive):
        identifier = assertion_id(assertion, primitive)
        try:
            checks.extend(produce(case, assertion))
        except ScopeError as e:
            checks.append(unevaluated(f"{primitive}.{identifier}", str(e)))
        except Exception as e:                    # noqa: BLE001 — reported
            checks.append(unevaluated(f"{primitive}.{identifier}",
                                      f"{type(e).__name__}: {e}"))
    return report(kind, context, case, checks, dimension=dimension,
                  primitive=primitive, score=score)


def actors_or_unresolved(case: Case, assertion: Mapping[str, Any],
                         default_scope: str = DEFAULT_SCOPE,
                         *, needs_bounds: bool = False
                         ) -> list[Mapping[str, Any]]:
    """The assertion's population, with two ways of having no population.

    An EMPTY selection is a refusal because every requirement over nothing
    holds. So is a selection the export never measured, when the metric is
    about size: `scene_geometry.extent_cm` falls back to a point, a point
    intersects nothing and overlaps nothing, and a scene of points scores a
    clean 1.0 on every geometric requirement in the layer. The exporter
    writes exactly that shape when its bounds read fails, so this is the live
    form of "a clean bill of health for work that never ran".
    """
    actors = case.population(assertion, default_scope)
    if not actors:
        raise ScopeError(
            "the assertion's Actor selector matched nothing, and every "
            "requirement over an empty population passes trivially")
    if needs_bounds:
        coverage, missing = bounds_coverage(actors)
        if missing:
            raise ScopeError(
                f"the export carries no usable bounds for {len(missing)} of "
                f"{len(actors)} selected Actor(s) "
                f"({', '.join(missing[:6])}), and an Actor of unknown size "
                f"intersects nothing — scoring this population would report a "
                f"clean scene for geometry that was never read "
                f"(coverage {coverage:.2f})")
    return actors


__all__ = ["Case", "Check", "DEFAULT_SCOPE", "SCOPES", "ScopeError",
           "actors_or_unresolved", "assertion_id", "check", "inapplicable",
           "load", "report", "run", "scene_actors", "unevaluated"]
