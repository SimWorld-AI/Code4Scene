"""``spatial_cluster`` — did what should be one arrangement stay one.

A task that asks for a dining set wants the chairs near the table, not four
chairs correctly built and scattered across a level. The measure is the XY
span of the population's centres — the bounding width and depth of where they
were placed — held against the span the case declares.

Centres, not bounds: a long table legitimately spans four metres, and
measuring the extents would make the rule about how big the furniture is
rather than how spread out it was placed.

Z is deliberately not measured. A cluster on two floors is a different
failure, and `spatial_relations` is where a case says so.

Serves component `spatial.absolute_constraints`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import assertions
from ..assertions import Case, Check
from ..context import Context
from ..scene_geometry import location_cm, rounded



def _checks(case: Case, assertion: Mapping[str, Any]) -> list[Check]:
    identifier = assertions.assertion_id(assertion, "compact_cluster")
    maximum = assertion.get("maximum_center_span_cm")
    if not isinstance(maximum, list) or len(maximum) != 2:
        return [assertions.unevaluated(
            f"spatial.compact_cluster.{identifier}",
            "the cluster assertion declares no maximum_center_span_cm [x, y], "
            "so there is no span to hold the arrangement to")]
    actors = assertions.actors_or_unresolved(case, assertion)
    centres = [location_cm(actor) for actor in actors]
    span = [rounded(max(axis) - min(axis))
            for axis in (tuple(c[0] for c in centres), tuple(c[1] for c in centres))]
    ok = span[0] <= float(maximum[0]) and span[1] <= float(maximum[1])
    return [assertions.check(
        f"spatial.compact_cluster.{identifier}", ok,
        {"maximum_center_span_cm": list(maximum)},
        {"center_span_cm": span, "actor_count": len(actors)},
        [{"label": actor.get("label"), "location_cm": location_cm(actor)}
         for actor in actors],
        f"the declared Actors span {span[0]:g} x {span[1]:g} cm, over the "
        f"declared {float(maximum[0]):g} x {float(maximum[1]):g}")]


def verify(context: Context) -> dict[str, Any]:
    return assertions.run("spatial_cluster", context, "compact_cluster",
                          "XY span of the declared Actors' centres", _checks)


__all__ = ["verify"]
