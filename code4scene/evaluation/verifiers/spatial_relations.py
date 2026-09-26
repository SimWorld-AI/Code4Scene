"""``spatial_relations`` — do the arrangements the task declared actually hold.

One check per declared relation rule — "the bookshelf is against a wall", "the
chairs are near the table", "nothing is on top of the stove" — and the score
is the share that hold. The relation vocabulary and the geometry behind it are
in `evaluation.spatial_relations`, shared with the rubric side so that one
word means one thing across the benchmark.

Each rule names its own two populations, so a rule can relate additions to
scenery the agent never touched. That is usually the interesting question: the
relations that matter are between what was built and what was already there.

Serves component `spatial.absolute_constraints`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import assertions
from .. import spatial_relations as relations
from ..assertions import Case, Check
from ..context import Context
from ..structure_rules import slug



def _checks(case: Case, assertion: Mapping[str, Any]) -> list[Check]:
    rules = assertion.get("relations") or []
    if not rules:
        return [assertions.unevaluated(
            f"spatial.relation.{assertions.assertion_id(assertion, 'spatial_relation')}",
            "the spatial_relation assertion declares no relations")]
    checks: list[Check] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            # A rule that is not an object cannot be evaluated, and dropping
            # it silently would also drop it from the denominator — the share
            # of the SURVIVING rules that hold is a different measurement
            # wearing the declared one's name. `case_spec` closes the same
            # hole for structure rules; relations are not deep-validated
            # there, so it is closed here, as a withheld check.
            checks.append(assertions.unevaluated(
                f"spatial.relation.relations_{index}",
                f"relations[{index}] is not an object and cannot be evaluated "
                f"as a relation rule; a case-file mistake is reported, never "
                f"silently dropped from the score"))
            continue
        check_id = f"spatial.relation.{slug(rule.get('id'))}"
        try:
            result = relations.evaluate(rule, case.selection(rule.get("subject")),
                                        case.selection(rule.get("object")))
        except relations.RelationError as e:
            checks.append(assertions.unevaluated(check_id, str(e)))
            continue
        if result["missing_selection"]:
            # A rule about objects the scene does not contain was not
            # answered. Scoring it as a failed check puts a zero in the
            # denominator and reads as "the layout is wrong", which is a
            # different statement — and the other three spatial verifiers
            # withhold on exactly this condition.
            checks.append(assertions.unevaluated(
                check_id,
                f"the selectors for rule {rule.get('id')!r} did not resolve "
                f"both sides ({result['observed']['subject_count']} subject(s), "
                f"{result['observed']['object_count']} object(s)), so the "
                f"relation was not evaluated",
                result["observed"]))
            continue
        checks.append(assertions.check(
            check_id, result["pass"], result["expected"], result["observed"],
            result["rows"],
            f"the declared {rule.get('relation')} relation does not hold"))
    return checks


def verify(context: Context) -> dict[str, Any]:
    return assertions.run("spatial_relations", context, "spatial_relation",
                          "share of declared spatial relations that hold",
                          _checks)


__all__ = ["verify"]
