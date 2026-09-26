"""Image-to-scene case score (paper protocol, Eq. 6).

``S_I2S = 0.8 * RepairF1 + 0.2 * S_physics``, rounded to six decimals
(policy ``i2s-actor-f1-0.8-physics-0.2-case-macro.v1``).

An invalid candidate (failed Candidate Integrity), a missing submission or a
missing result scores zero, and so do its Precision, Recall and F1; these are
reporting zeros, not measured TP/FP/FN counts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from . import actor_f1, physics
from .constants import I2S_CASE_POLICY, I2S_DECIMALS, I2S_PHYSICS_WEIGHT, I2S_REPAIR_WEIGHT


def _unit(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number in [0, 1], got {value!r}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"expected a number in [0, 1], got {value!r}")
    return value


def case_score(*, valid: bool, repair_f1: float | None, physics_score: float | None) -> dict:
    """``round(0.8 * F1 + 0.2 * physics, 6)``; zero for an invalid candidate."""

    if not valid:
        return {"policy": I2S_CASE_POLICY, "status": "zero_invalid_candidate", "score": 0.0,
                "repair_f1": 0.0, "physics": 0.0}
    f1 = _unit(repair_f1)
    phys = _unit(physics_score)
    score = round(math.fsum((I2S_REPAIR_WEIGHT * f1, I2S_PHYSICS_WEIGHT * phys)), I2S_DECIMALS)
    return {"policy": I2S_CASE_POLICY, "status": "measured", "score": score,
            "repair_f1": f1, "physics": phys}


def score_case(
    *,
    valid: bool,
    input_scene: Mapping[str, Any] | None,
    ground_truth_scene: Mapping[str, Any] | None,
    candidate_scene: Mapping[str, Any] | None,
    physical_safety_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Full image-to-scene case: Actor F1 from snapshots plus recorded physics."""

    if not valid:
        return {**case_score(valid=False, repair_f1=None, physics_score=None),
                "actor_f1": actor_f1.zero_for_invalid_case(), "physics_detail": None}
    if input_scene is None or ground_truth_scene is None or candidate_scene is None:
        raise ValueError("a valid image-to-scene case needs input, ground-truth and "
                         "candidate scene snapshots")
    values, audit = actor_f1.measure(input_scene, ground_truth_scene, candidate_scene)
    phys = physics.score_from_report(physical_safety_report)
    result = case_score(valid=True, repair_f1=values["f1"], physics_score=phys["score"])
    return {**result, "actor_f1": values, "actor_f1_audit": audit, "physics_detail": phys}


__all__ = ["case_score", "score_case"]
