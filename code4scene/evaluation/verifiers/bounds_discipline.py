"""``bounds_discipline`` — how much the harness had to correct the agent.

The world plate is enforced deterministically after every round: actors fully
outside it are deleted, oversized flat sheets are squashed back. Those
corrections are the measurement — the agent is measured, never silently
corrected — and this verifier is what turns them into a rate.
"""

from __future__ import annotations

from typing import Any

from .. import contracts
from ..context import Context



#: Which class this verifier's number belongs in. A `gt` verifier reads the
#: answer key — a `.label.json` or a canonical scene — and can only run where
#: that key exists; an `open_ended` one answers from the candidate alone. The
#: two are reported side by side and never averaged, which is why every
#: verifier has to say which it is.
def verify(context: Context) -> dict[str, Any]:
    """Plate discipline: corrections per actor, or a refusal to score."""
    record = context.record
    report = contracts.base("bounds_discipline", context.ids)
    edge = record.get("edge_discipline") or {}
    clamped = edge.get("total_clamped") or 0
    deleted = edge.get("total_deleted") or 0
    corrections = clamped + deleted
    passes = edge.get("passes") or []
    latest_pass = passes[-1] if passes and isinstance(passes[-1], dict) else {}
    actors = latest_pass.get("checked")
    if actors is None:
        actors = (record.get("metrics") or {}).get("actors") or 0
    if isinstance(actors, bool) or not isinstance(actors, (int, float)):
        return {**report, "status": contracts.ERROR, "score": None,
                "failure_reason": (
                    f"the record's metrics.actors is a {type(actors).__name__}, "
                    f"not a count; corrections are reported per Actor and "
                    f"there is nothing to divide by"),
                "metrics": {}, "evidence": {}, "artifacts": {},
                "probes_used": ("bounds",)}
    # Corrections mean something only relative to how much was built. With no
    # measured scene there is nothing to normalize against, so the score is
    # withheld rather than invented.
    rate = (corrections / actors) if actors else None
    # No pass ran, so there is nothing to report and certainly nothing to
    # pass. `corrections == 0` meant both "the plate was respected" and "no
    # boundary was ever enforced". The resolved boundary normally comes from
    # canonical GT and only falls back to size_m when a no-GT prompt explicitly
    # constrains the build region. With neither, bounds discipline is simply
    # outside the task rather than an infrastructure failure.
    ran = bool(passes) or corrections > 0
    if not ran:
        return {**report, "status": "not_applicable", "score": None,
                "failure_reason": ("the task has neither canonical 3D GT nor "
                                   "an explicit size_m boundary requirement, "
                                   "so no bounds pass is applicable"),
                "metrics": {},
            "evidence": {
                "metric_id": contracts.metric_id(record, "bounds"),
                "evaluation_bounds": record.get("evaluation_bounds"),
            },
                "artifacts": contracts.artifacts(record),
                "probes_used": ("bounds_pass",)}
    if rate is None:
        return {**report, "status": contracts.ERROR, "score": None,
                "failure_reason": (
                    "bounds corrections were recorded without a measured Actor "
                    "population, so corrections_per_actor is undefined"),
                "metrics": {"clamped": clamped, "deleted": deleted,
                            "corrections_per_actor": None},
                "evidence": {"metric_id": contracts.metric_id(record, "bounds"),
                             "passes": passes,
                             "evaluation_bounds": record.get("evaluation_bounds")},
                "artifacts": contracts.artifacts(record),
                "probes_used": ("bounds_pass",)}
    return {**report, "status": contracts.MEASURED,
            "score": round(max(0.0, 1.0 - rate), 4),
            "metrics": {"clamped": clamped, "deleted": deleted,
                        "corrections_per_actor": round(rate, 4)},
            "evidence": {"metric_id": contracts.metric_id(record, "bounds"),
                         "passes": passes,
                         "evaluation_bounds": record.get("evaluation_bounds")},
            "artifacts": contracts.artifacts(record),
            "probes_used": ("bounds_pass",)}


__all__ = ["verify"]
