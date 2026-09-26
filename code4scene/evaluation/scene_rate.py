"""One rate out of the closing measurement, as one verifier's answer.

Three metrics read the same measurement and differ only in which rate they
take: how much of the scene intersects itself, how much of it floats, how much
of it left the plate. They used to be one verifier that reported the worst of
the three, which is a scoring decision — and a scoring decision belongs where
it can be seen and changed, not inside a metric. Each rate is its own verifier
now; combining them is `aggregate.geometry_score`.

The guards live here because all three need the same ones, and a guard that
each metric restates is a guard two of them eventually get wrong:

* never measured is an ERROR, not a zero — a run that could not be measured is
  not a run that measured badly;
* an ABSENT rate while others are present is an error too. Reading absent as
  zero is how "we did not check" becomes "it was flawless";
* zero actors is an ERROR because every rate over an empty scene is 0.0 and
  would otherwise publish an empty level as perfect;
* a caller may publish the measured defect rate directly as a lower-is-better
  result. No ``max_rate`` turns it into a pass/fail verdict. Generic and
  composite aggregation orient that result to higher-is-better quality only
  at the aggregation boundary; the leaf itself remains the measured rate.
"""

from __future__ import annotations

from typing import Any

from . import contracts


def rate_report(
    kind: str,
    context: Any,
    *,
    rate_key: str,
    dimension: str,
    publish_direct_rate: bool = False,
) -> dict[str, Any]:
    """Publish one measured rate, optionally without complementing it."""
    record = context.record
    metrics = record.get("metrics") or {}
    files = contracts.artifacts(record)
    report = contracts.base(kind, context.ids)
    common = {"artifacts": files, "probes_used": ("measure_scene",)}

    actors = metrics.get("actors")
    if actors is None:
        return {**report, "status": contracts.ERROR, "score": None,
                "failure_reason": (record.get("infra_error")
                                   or "the scene was never measured"),
                "metrics": {}, "evidence": {}, **common}
    # `measure()` returns the actor LIST under this key and the record wants a
    # count. A caller that passed the list through produced a TypeError deep in
    # the arithmetic, which reached the report as the failure reason — a stack
    # type name where a reader expects to be told what was wrong with the run.
    if isinstance(actors, bool) or not isinstance(actors, (int, float)):
        return {**report, "status": contracts.ERROR, "score": None,
                "failure_reason": (
                    f"the record's metrics.actors is a {type(actors).__name__}, "
                    f"not a count; `measure()` returns the actor list under "
                    f"that name and the record wants len() of it"),
                "metrics": {}, "evidence": {}, **common}

    rate = metrics.get(rate_key)
    if rate is None:
        return {**report, "status": contracts.ERROR, "score": None,
                "failure_reason": (f"the measurement reported an actor count but no "
                                   f"{rate_key}, and treating an absent rate as zero "
                                   f"would publish a flawless scene"),
                "metrics": {"actors": actors}, "evidence": {}, **common}

    rate = float(rate)
    bounded_rate = max(0.0, min(1.0, rate))
    score = bounded_rate if publish_direct_rate else 1.0 - bounded_rate
    score_direction = (
        "lower_is_better" if publish_direct_rate else "higher_is_better"
    )
    body = {**report,
            "score": round(score, 4),
            "metrics": {"actors": actors, rate_key: rate},
            "evidence": {"metric_id": contracts.metric_id(record, "measure"),
                         "dimension": dimension,
                         "publication": (
                             "direct_measured_rate_without_pass_fail_threshold"
                             if publish_direct_rate else
                             "continuous_score_without_pass_fail_threshold"
                         )},
            "metadata": {
                "score_direction": score_direction,
                "result_semantics": (
                    "direct_measured_rate" if publish_direct_rate
                    else "normalized_quality"
                ),
            },
            **common}

    if actors == 0:
        return {**body, "status": contracts.ERROR, "score": None,
                "failure_reason": "no actors were placed, so the rate was not measurable"}
    return {**body, "status": contracts.MEASURED}


__all__ = ["rate_report"]
