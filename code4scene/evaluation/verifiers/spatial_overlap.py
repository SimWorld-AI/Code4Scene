"""``spatial_overlap`` — do the Actors this task is about sit inside each other.

Distinct from the legacy scene-wide AABB collision diagnostic, which takes an
intersection rate out of the closing measurement. This formal verifier is
about the population an assertion names — the four chairs, the props added to
one room — and it fails when any two of them intersect, because a task that
asked for four chairs around a table is not satisfied by four chairs in the
same cubic metre.

Both use the same 5 cm touch tolerance and the same all-three-axes rule, from
`scene_geometry`. They have to: one record carrying two different definitions
of "inside another Actor" is a record a reader cannot use.

Serves component `spatial.absolute_constraints`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .. import assertions
from ..assertions import Case, Check
from ..context import Context
from ..scene_geometry import TOUCH_TOLERANCE_CM, aabb_overlaps



def _checks(case: Case, assertion: Mapping[str, Any]) -> list[Check]:
    actors = assertions.actors_or_unresolved(case, assertion, needs_bounds=True)
    tolerance = assertion.get("touch_tolerance_cm")
    if tolerance is None:
        tolerance = TOUCH_TOLERANCE_CM
    elif isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) \
            or not math.isfinite(float(tolerance)) or float(tolerance) < 0:
        # A non-finite threshold makes every `>` comparison false and the
        # scene comes back clean. Refused rather than coerced.
        raise assertions.ScopeError(
            f"touch_tolerance_cm must be a finite non-negative JSON number, "
            f"got {tolerance!r}; an unusable threshold silently clears every "
            f"pair")
    tolerance = float(tolerance)
    overlaps = aabb_overlaps(actors, tolerance)
    return [assertions.check(
        f"spatial.no_overlap.{assertions.assertion_id(assertion, 'no_overlap')}",
        not overlaps,
        {"overlapping_pairs": 0, "method": "world_aabb",
         "touch_tolerance_cm": tolerance},
        {"overlapping_pairs": len(overlaps), "actor_count": len(actors)},
        overlaps,
        f"{len(overlaps)} pair(s) of the declared Actors intersect beyond "
        f"{tolerance:g} cm")]


def verify(context: Context) -> dict[str, Any]:
    return assertions.run("spatial_overlap", context, "no_overlap",
                          "declared Actors intersecting one another", _checks)


__all__ = ["verify"]
