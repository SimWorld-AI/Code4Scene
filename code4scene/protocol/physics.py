"""Physical Safety as scored in the paper (Appendix C.3).

``S_physics = 0.5 * (1 - r_unsup) + 0.5 * S_pen``

* ``r_unsup`` is the fraction of evaluated actors with no support (the
  ``floating`` leaf, lower is better);
* ``S_pen`` is the collision-free fraction of eligible actors (the
  ``solid_penetration`` leaf, higher is better).

Both leaves keep their fixed 0.5 weight. A required leaf without a numeric
measurement (not evaluated, error, missing, or explicitly not applicable)
contributes zero (policy ``physics-all-leaves-zero.v2``). The case value is
rounded to four decimals. An invalid or missing candidate scores zero.

Text-to-scene support uses world-space AABBs over the saved scene snapshot:
an actor is supported when its bottom is at most 5 cm above the world ground
plane, or when another actor lies beneath it with a vertical gap in [0, 5] cm
and a footprint that contains its pivot (metric
``t2s-aabb-floating-contact-5cm.v2``). This part is recomputed offline from
the snapshot by :func:`t2s_floating_rate`. Penetration requires confirmed
Unreal collision-body overlap and native minimum-translation depth, so it is
taken from the recorded engine measurement. Image-to-scene support uses
targeted in-engine support checks and is likewise taken from the recorded
measurement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..evaluation import measure_payload
from .constants import PHYSICS_DECIMALS, PHYSICS_LEAF_WEIGHTS, PHYSICS_POLICY, T2S_FLOATING_METRIC

MEASURED = "measured"


def _unit(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def leaf_values(report: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Read the two Physical Safety leaves from a ``physical_safety`` report.

    Returns, per leaf, the recorded status, the raw recorded score and the
    safety value in [0, 1] that enters the physics score (``None`` when the
    leaf contributes zero).
    """

    report = report or {}
    metrics = report.get("metrics") or {}
    by_id = {}
    for leaf in metrics.get("leaf_results") or ():
        if isinstance(leaf, Mapping):
            key = str(leaf.get("leaf_id") or str(leaf.get("report_id", "")).split(".")[-1])
            by_id[key] = leaf
    directions = metrics.get("leaf_score_directions") or {}
    out = {}
    for name in PHYSICS_LEAF_WEIGHTS:
        leaf = by_id.get(name)
        status = str((leaf or {}).get("status") or "missing")
        raw = _unit((leaf or {}).get("score")) if leaf else None
        # The leaf's own metadata wins (as in the verifier layer), then the
        # parent's declared directions, then the canonical default.
        direction = ((leaf or {}).get("metadata") or {}).get("score_direction") or directions.get(
            name, "lower_is_better" if name == "floating" else "higher_is_better"
        )
        safety = None
        if status == MEASURED and raw is not None:
            safety = 1.0 - raw if direction == "lower_is_better" else raw
        out[name] = {"status": status, "raw_score": raw, "direction": direction,
                     "safety": safety}
    return out


def combine(floating_safety: float | None, penetration: float | None) -> float:
    """``round(0.5 * floating_safety + 0.5 * penetration, 4)``; missing leaves are 0.

    The expression order matches the reference implementation so that the
    rounding of exact ties is reproduced bit for bit.
    """

    f = 0.0 if floating_safety is None else float(floating_safety)
    p = 0.0 if penetration is None else float(penetration)
    return round(
        PHYSICS_LEAF_WEIGHTS["floating"] * f + PHYSICS_LEAF_WEIGHTS["solid_penetration"] * p,
        PHYSICS_DECIMALS,
    )


def score_from_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Physics score from recorded leaves (image-to-scene, or T2S as recorded).

    As in the reference policy, a complete legacy report that carries only a
    measured scalar and no leaf evidence keeps that measured value.
    """

    leaves = leaf_values(report)
    report = report or {}
    if not (report.get("metrics") or {}).get("leaf_results"):
        legacy = _unit(report.get("score")) if report.get("status") == MEASURED else None
        if legacy is not None:
            return {"policy": PHYSICS_POLICY, "score": round(legacy, PHYSICS_DECIMALS),
                    "leaves": {name: {**leaf, "safety": legacy} for name, leaf in leaves.items()},
                    "floating_source": "recorded_legacy_scalar"}
    score = combine(leaves["floating"]["safety"], leaves["solid_penetration"]["safety"])
    return {"policy": PHYSICS_POLICY, "score": score, "leaves": leaves,
            "floating_source": "recorded"}


def t2s_floating_rate(scene: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the text-to-scene unsupported-actor rate from a scene snapshot.

    Returns ``{"actor_count", "floating_count", "floating_labels", "rate"}``;
    ``rate`` is rounded to four decimals as in the published tables and is
    ``None`` when the snapshot has no eligible actor.
    """

    rows = measure_payload.rows_from_scene(dict(scene))
    failed = measure_payload.floating_actors(rows)
    rate = round(len(failed) / len(rows), 4) if rows else None
    return {"metric": T2S_FLOATING_METRIC, "actor_count": len(rows),
            "floating_count": len(failed), "floating_labels": failed, "rate": rate}


def t2s_score(
    scene: Mapping[str, Any] | None,
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Text-to-scene physics: floating recomputed from ``scene``, penetration recorded.

    When no snapshot is available, or it has no eligible actor, the recorded
    floating leaf is used (this is how the paper treated empty scenes).
    """

    leaves = leaf_values(report)
    floating = dict(leaves["floating"])
    source = "recorded"
    measurement = None
    if scene is not None:
        measurement = t2s_floating_rate(scene)
        if measurement["rate"] is not None:
            rate = measurement["rate"]
            floating.update(status=MEASURED, raw_score=rate, safety=round(1.0 - rate, 4),
                            metric=T2S_FLOATING_METRIC)
            source = "recomputed_from_snapshot"
    penetration = leaves["solid_penetration"]["safety"]
    if source == "recomputed_from_snapshot":
        # Same floating-point ordering as the reference rescoring.
        score = round(0.5 * (1.0 - floating["raw_score"]) + 0.5 * (penetration or 0.0),
                      PHYSICS_DECIMALS)
    else:
        score = combine(floating["safety"], penetration)
    return {"policy": PHYSICS_POLICY, "score": score,
            "leaves": {"floating": floating, "solid_penetration": leaves["solid_penetration"]},
            "floating_source": source, "floating_measurement": measurement}


__all__ = ["combine", "leaf_values", "score_from_report", "t2s_floating_rate", "t2s_score"]
