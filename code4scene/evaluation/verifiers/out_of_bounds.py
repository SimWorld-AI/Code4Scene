"""``out_of_bounds`` — how much left the frozen evaluation bounds.

For a task with 3D GT, the boundary is the GT content AABB measured in the
independent scoring editor. A task-provided plate is only the fallback when no
canonical scene exists. Distinct from `bounds_discipline`, which counts what
the harness had to correct: this is what remained outside afterwards.

**A run with neither GT bounds nor a task plate has no OOB question.** It is
``not_applicable``, not an error and not a fabricated rate. The historical
measurement fallback is deliberately ignored because its origin-centred
default was a number nobody chose.
"""

from __future__ import annotations

from typing import Any

from .. import bounds as bounds_eval
from .. import contracts
from ..context import Context
from ..scene_rate import rate_report



def verify(context: Context) -> dict[str, Any]:
    evaluation_bounds = context.record.get("evaluation_bounds")
    if not isinstance(evaluation_bounds, dict):
        legacy_size = context.record.get("size_m")
        if (
            isinstance(legacy_size, (int, float))
            and not isinstance(legacy_size, bool)
            and legacy_size > 0
        ):
            evaluation_bounds = bounds_eval.from_half_extent_m(
                float(legacy_size) / 2.0
            )
    if not isinstance(evaluation_bounds, dict):
        return {**contracts.base("out_of_bounds", context.ids),
                "status": "not_applicable",
                "score": None,
                "failure_reason": (
                    "the task has neither canonical 3D GT nor an explicit "
                    "size_m boundary requirement, so there is no intended "
                    "region to leave"
                ),
                "metrics": {},
                "evidence": {"evaluation_bounds": None},
                "probes_used": ("measure_scene",),
                "artifacts": contracts.artifacts(context.record)}
    report = rate_report(
        "out_of_bounds",
        context,
        rate_key="oob_rate",
        dimension="share of Actors outside the frozen evaluation bounds",
    )
    report.setdefault("evidence", {})["evaluation_bounds"] = evaluation_bounds
    return report


__all__ = ["verify"]
