"""Model score from case scores (paper protocol, Appendix C.6, Eq. 7).

* Every scheduled case stays in its setting's denominator. A valid case
  contributes its case score; an authoritative invalid verdict, a missing
  submission or a missing result contributes zero.
* ``S_T2S`` is the mean of the text-to-scene case scores.
* ``S_I2S`` is the mean of ALL image-to-scene case scores, indoor and outdoor
  cases weighted equally: ``(N_in * S_in + N_out * S_out) / (N_in + N_out)``
  (1/3 indoor and 2/3 outdoor on the public set).
* ``S_model = 0.5 * S_T2S + 0.5 * S_I2S``.

If any scheduled case is unresolved (for example an evaluator error), the
affected aggregate is not reported; it is never computed over fewer cases.
Case scores are kept at full precision for aggregation (they were already
rounded at the case level: four decimals for T2S and six for I2S).
Macro Repair F1 averages case F1; it is not a harmonic mean of macro
precision and recall.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .constants import (
    I2S_CASE_POLICY,
    MODEL_POLICY,
    MODEL_WEIGHTS,
    SETTING_INDOOR,
    SETTING_OUTDOOR,
    SETTING_T2S,
    SETTINGS,
)

#: Case statuses that contribute a (possibly zero) score.
RESOLVED = {"measured", "measured_with_missing_zero", "zero_invalid_candidate",
            "zero_missing_submission", "zero_missing_result"}
ZERO_STATUSES = {"zero_invalid_candidate", "zero_missing_submission", "zero_missing_result"}


class AggregationError(ValueError):
    """The case table cannot produce a paper aggregate."""


def _unit(value: Any, what: str) -> float:
    if isinstance(value, str):
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregationError(f"{what} is not a number: {value!r}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise AggregationError(f"{what} is outside [0, 1]: {value!r}")
    return value


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise AggregationError("cannot average an empty case set")
    return math.fsum(values) / len(values)


def pooled(indoor: float, outdoor: float, n_indoor: int, n_outdoor: int) -> float:
    """Equal weight per case across the two image-to-scene domains."""

    return math.fsum((n_indoor * indoor, n_outdoor * outdoor)) / (n_indoor + n_outdoor)


def normalize_setting(value: str) -> str:
    text = str(value).strip().lower().replace("_", "-")
    aliases = {
        "t2s": SETTING_T2S, "text-to-scene": SETTING_T2S,
        "indoor": SETTING_INDOOR, "image-to-scene/indoor": SETTING_INDOOR,
        "image-to-scene-indoor": SETTING_INDOOR, "i2s-indoor": SETTING_INDOOR,
        "outdoor": SETTING_OUTDOOR, "image-to-scene/outdoor": SETTING_OUTDOOR,
        "image-to-scene-outdoor": SETTING_OUTDOOR, "i2s-outdoor": SETTING_OUTDOOR,
    }
    if text not in aliases:
        raise AggregationError(f"unknown setting {value!r}; expected one of {SETTINGS}")
    return aliases[text]


def _complete(
    rows: Sequence[Mapping[str, Any]],
    schedule: Mapping[str, Sequence[str]] | None,
    missing_as_zero: bool,
) -> tuple[dict[tuple[str, str], dict[str, dict[str, Any]]], dict[str, list[str]]]:
    """Index rows by (model, setting) and fill scheduled-but-absent cases."""

    table: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for raw in rows:
        model = str(raw["model"])
        setting = normalize_setting(raw["setting"])
        case = str(raw["case_id"])
        if case in table[model, setting]:
            raise AggregationError(f"duplicate case row: {model} / {setting} / {case}")
        table[model, setting][case] = dict(raw, setting=setting)
    models = sorted({m for m, _ in table})
    if schedule is None:
        schedule = {}
        for setting in SETTINGS:
            cases = sorted({c for (m, s), group in table.items() if s == setting for c in group})
            if cases:
                schedule[setting] = cases
    schedule = {normalize_setting(k): list(v) for k, v in schedule.items()}
    for model in models:
        for setting, cases in schedule.items():
            group = table[model, setting]
            extra = sorted(set(group) - set(cases))
            if extra:
                raise AggregationError(
                    f"{model} / {setting}: cases outside the schedule: {extra[:5]}")
            for case in cases:
                if case not in group:
                    group[case] = {
                        "model": model, "setting": setting, "case_id": case, "score": 0.0,
                        "status": "zero_missing_result" if missing_as_zero else "unresolved",
                        "imputed": True,
                    }
    return table, schedule


def aggregate(
    rows: Iterable[Mapping[str, Any]],
    *,
    schedule: Mapping[str, Sequence[str]] | None = None,
    missing_as_zero: bool = True,
) -> list[dict[str, Any]]:
    """Aggregate case rows into one paper-protocol row per model.

    ``rows`` need ``model``, ``setting``, ``case_id`` and ``score``; optional
    ``status`` (default ``measured``), ``repair_f1``, ``physics``, ``precision``
    and ``recall`` (image-to-scene). ``schedule`` maps a setting to its case
    IDs; by default it is the set of cases observed for that setting.
    ``missing_as_zero`` scores a scheduled case without a row as a missing
    result (zero), as in the paper.
    """

    table, schedule = _complete(list(rows), schedule, missing_as_zero)
    models = sorted({m for m, _ in table})
    out = []
    for model in models:
        record: dict[str, Any] = {"model": model}
        unresolved: list[str] = []
        per_setting: dict[str, dict[str, Any]] = {}
        for setting, cases in schedule.items():
            group = [table[model, setting][c] for c in cases]
            statuses = [str(r.get("status") or "measured") for r in group]
            bad = [r["case_id"] for r, s in zip(group, statuses, strict=True) if s not in RESOLVED]
            unresolved += [f"{setting}:{c}" for c in bad]
            zero = [s in ZERO_STATUSES for s in statuses]
            scores = [0.0 if z else _unit(r["score"], f"{model}/{r['case_id']} score")
                      for r, z in zip(group, zero, strict=True)]
            stats = {"cases": len(cases), "zero_cases": sum(zero),
                     "score": None if bad else _mean(scores)}
            if setting != SETTING_T2S and not bad:
                for key in ("repair_f1", "physics", "precision", "recall"):
                    values = []
                    for r, z in zip(group, zero, strict=True):
                        if z:
                            values.append(0.0)
                        elif r.get(key) not in (None, ""):
                            values.append(_unit(r[key], f"{model}/{r['case_id']} {key}"))
                    stats[key] = _mean(values) if len(values) == len(group) else None
            per_setting[setting] = stats
        t2s = per_setting.get(SETTING_T2S, {})
        indoor = per_setting.get(SETTING_INDOOR, {})
        outdoor = per_setting.get(SETTING_OUTDOOR, {})

        def pooled_key(key: str, indoor=indoor, outdoor=outdoor) -> float | None:
            if indoor.get(key) is None or outdoor.get(key) is None:
                return None
            return pooled(indoor[key], outdoor[key], indoor["cases"], outdoor["cases"])

        i2s = pooled_key("score")
        t2s_score = t2s.get("score")
        model_score = (
            MODEL_WEIGHTS["t2s"] * t2s_score + MODEL_WEIGHTS["i2s"] * i2s
            if t2s_score is not None and i2s is not None else None
        )
        # (t + i) / 2 is bit-identical to 0.5 t + 0.5 i; keep the published form.
        if model_score is not None:
            model_score = (t2s_score + i2s) / 2
        record.update(
            model_score=model_score,
            model_score_100=None if model_score is None else 100 * model_score,
            t2s_score=t2s_score, i2s_score=i2s,
            i2s_score_100=None if i2s is None else 100 * i2s,
            indoor_i2s_score=indoor.get("score"), outdoor_i2s_score=outdoor.get("score"),
            indoor_f1_case_macro=indoor.get("repair_f1"),
            outdoor_f1_case_macro=outdoor.get("repair_f1"),
            i2s_f1_case_macro=pooled_key("repair_f1"),
            i2s_precision_case_macro=pooled_key("precision"),
            i2s_recall_case_macro=pooled_key("recall"),
            i2s_physics_case_macro=pooled_key("physics"),
            indoor_physics_case_macro=indoor.get("physics"),
            outdoor_physics_case_macro=outdoor.get("physics"),
            t2s_cases=t2s.get("cases", 0), indoor_cases=indoor.get("cases", 0),
            outdoor_cases=outdoor.get("cases", 0),
            t2s_zero_cases=t2s.get("zero_cases", 0),
            indoor_zero_cases=indoor.get("zero_cases", 0),
            outdoor_zero_cases=outdoor.get("zero_cases", 0),
            unresolved_cases=unresolved,
            i2s_policy=I2S_CASE_POLICY, model_policy=MODEL_POLICY,
        )
        out.append(record)

    def ranks(field: str) -> dict[str, int | None]:
        scored = [r for r in out if r[field] is not None]
        return {r["model"]: (1 + sum(x[field] > r[field] for x in scored)
                             if r[field] is not None else None) for r in out}

    overall, i2s_rank = ranks("model_score"), ranks("i2s_score")
    for r in out:
        r["model_rank"], r["i2s_rank"] = overall[r["model"]], i2s_rank[r["model"]]
    return sorted(out, key=lambda r: (r["model_score"] is None, -(r["model_score"] or 0),
                                      r["model"]))


__all__ = ["AggregationError", "aggregate", "normalize_setting", "pooled"]
