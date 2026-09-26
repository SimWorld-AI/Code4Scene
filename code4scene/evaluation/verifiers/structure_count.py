"""``structure_count`` — did the edit add the number of Actors it was told to.

The rawest question a structured case asks, and deliberately separate from
`structure_concepts`: an edit can add exactly four chairs and also add nine
things nobody asked for, and one number covering both cannot say which
happened. This counts UE Actors, not logical objects — a table whose top and
four legs are five Actors counts as five here — because the contract it checks
is about what was spawned.

Serves component `semantic.structured_requirements`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import assertions
from ..assertions import Case, Check
from ..context import Context
from ..scene_diff import actor_summary



def _checks(case: Case, assertion: Mapping[str, Any]) -> list[Check]:
    expected = assertion.get("expected_raw_added_count")
    if not isinstance(expected, int) or isinstance(expected, bool):
        return [assertions.unevaluated(
            "structure.actor_count",
            "the structure assertion declares no expected_raw_added_count, so "
            "there is no number to hold the edit to")]
    added = case.resolve_scope("all_additions")
    return [assertions.check(
        "structure.actor_count", len(added) == expected,
        {"added_ue_actors": expected}, {"added_ue_actors": len(added)},
        [actor_summary(actor) for actor in added],
        f"the edit added {len(added)} UE Actors where the case contract "
        f"declares {expected}")]


def verify(context: Context) -> dict[str, Any]:
    return assertions.run("structure_count", context, "structure",
                          "added UE Actor count against the case contract",
                          _checks)


__all__ = ["verify"]
