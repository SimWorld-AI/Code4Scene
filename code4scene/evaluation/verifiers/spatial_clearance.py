"""``spatial_clearance`` — did the edit block something that must stay clear.

Two shapes of protected space, because scenes have two:

* **rectangles** — a doorway, a fire lane, the footprint of a machine that has
  to be reachable. An Actor whose footprint overlaps one at all is a
  violation; there is no partial credit for mostly not blocking a door.
* **route corridors** — a polyline with a width, which is how a walkable path
  is actually described. An Actor intrudes when its footprint radius plus the
  corridor's half-width exceeds its distance to the line.

Routes are given in METRES because that is the unit the navigation side uses,
and Actors are in centimetres because that is the unit the editor uses. The
conversion happens in one place, `scene_geometry.placement`, rather than in
each rule — a factor of 100 applied twice or not at all is the kind of bug
that produces a plausible number.

Serves component `spatial.absolute_constraints`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import assertions
from ..assertions import Case, Check
from ..context import Context
from ..scene_geometry import clearance_violations



def _checks(case: Case, assertion: Mapping[str, Any]) -> list[Check]:
    identifier = assertions.assertion_id(assertion, "clearance")
    rectangles = assertion.get("protected_rectangles") or []
    routes = assertion.get("routes") or []
    if not rectangles and not routes:
        return [assertions.unevaluated(
            f"spatial.clearance.{identifier}",
            "the clearance assertion names neither a protected rectangle nor a "
            "route, so it protects nothing")]
    # A route needs at least two points to be a line. One that does not is a
    # protection the case believes it declared and this build cannot apply, so
    # it is refused rather than skipped in silence.
    unusable = [str(route.get("id")) for route in routes
                if len(route.get("points") or []) < 2]
    if unusable:
        return [assertions.unevaluated(
            f"spatial.clearance.{identifier}",
            f"route(s) {', '.join(unusable)} carry fewer than two points, so "
            f"the corridor they declare cannot be measured and the Actors "
            f"crossing it would be reported as clear")]
    actors = assertions.actors_or_unresolved(case, assertion, needs_bounds=True)
    violations = clearance_violations(actors, rectangles, routes)
    intruders = {str(item.get("actor_id") or item.get("actor"))
                 for item in violations}
    return [assertions.check(
        f"spatial.clearance.{identifier}", not violations,
        {"protected_region_hits": 0, "protected_rectangles": len(rectangles),
         "routes": len(routes)},
        # Both, because they are different numbers: one Actor can intrude on
        # two rectangles, and the route loop stops at the first corridor it
        # hits. Reporting only the pair count and calling it Actors was wrong
        # in both directions.
        {"protected_region_hits": len(violations),
         "intruding_actor_count": len(intruders),
         "actor_count": len(actors)},
        violations,
        f"{len(intruders)} of {len(actors)} declared Actor(s) intrude on a "
        f"protected region or route corridor")]


def verify(context: Context) -> dict[str, Any]:
    return assertions.run("spatial_clearance", context, "clearance",
                          "declared Actors intruding on protected space",
                          _checks)


__all__ = ["verify"]
