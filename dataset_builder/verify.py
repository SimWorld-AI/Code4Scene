"""Compare locally built levels with the benchmark's expected fingerprints.

Inputs are the snapshots exported by the dataset builder:
``<dataset>/snapshots/gt/<scene_id>.scene.json`` and
``<dataset>/snapshots/input/<case_id>.scene.json``.

Three checks per level:

1. ``ground_truth``: the local GT against ``scenes/<scene>/gt.fingerprint.json``.
2. ``input``: the local Input against its expected fingerprint (the GT's
   fingerprint plus the case's ``input.fingerprint.json`` delta).
3. ``recipe_consistency``: the recipe replayed offline on the *local* GT must
   predict the *local* Input. This isolates editor-side materialisation
   problems from pack-content differences: if (1) fails but (3) passes, the
   pack content differs from the one the benchmark was built from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import catalog
from . import fingerprint as fp
from . import recipe as rc

EXPLANATIONS = {
    "missing": "expected actors are absent. Stable IDs derive from the demo map's actor names, so the "
               "installed pack's demo map differs (different pack version, or the map was re-saved/edited).",
    "extra": "the level has actors the benchmark level does not (different pack version, or plugins "
             "that add actors on load).",
    "identity": "class, label, mesh assets or tags differ (usually a different pack version).",
    "materials": "material assignments differ (different pack version, or a material dependency that "
                 "failed to load and fell back to a default material).",
    "transform": "placement differs beyond float noise (differences under 0.1 cm, about 0.006 degrees "
                 "and 1e-4 in scale always match); usually a different pack version.",
    "properties": "light/fog/post-process properties differ.",
}


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _reasons(entry: dict[str, Any]) -> list[str]:
    return entry.get("reasons") or ([entry["reason"]] if entry.get("reason") else [])


def _error(exception: Exception) -> dict[str, Any]:
    """A level the comparison could not handle; reported, and the other levels still are."""
    return {"status": "error", "reason": f"{type(exception).__name__}: {exception}"}


def _summarize(result: dict[str, Any]) -> dict[str, Any]:
    status = "match" if result["match"] else "mismatch"
    facets = fp.summarize_facets(result["mismatched"])
    reasons = []
    if result["missing"]:
        reasons.append(f"{len(result['missing'])} missing: " + EXPLANATIONS["missing"])
    if result["extra"]:
        reasons.append(f"{len(result['extra'])} extra: " + EXPLANATIONS["extra"])
    for name, count in facets.items():
        reasons.append(f"{count} actors differ in {name}: " + EXPLANATIONS[name])
    return {
        "status": status,
        "expected_root": result["expected_root"],
        "expected_actor_count": result["expected_actor_count"],
        "local_actor_count": result["local_actor_count"],
        "matched_after_float_probe": result["matched_after_float_probe"],
        "mismatched_facets": facets,
        "reasons": reasons,
        "mismatched": result["mismatched"][:50],
        "missing": result["missing"][:50],
        "extra": result["extra"][:50],
    }


def sublevel_rename(recipe: dict[str, Any]) -> dict[str, str]:
    return {op["from"]: op["to"] for op in recipe.get("level_structure") or []
            if op.get("op") == "retarget_streaming_level"}


def verify(dataset: Path, cases: list[catalog.Case]) -> dict[str, Any]:
    snapshots = dataset / "snapshots"
    report: dict[str, Any] = {"schema_version": "code4scene.verify_report.v1", "scenes": {}, "cases": {}}
    expected_gt: dict[str, dict] = {}
    local_gt: dict[str, dict] = {}
    for case in cases:
        if not case.is_i2s or not case.recipe:
            continue
        scene_id = case.scene_id
        if scene_id not in report["scenes"]:
            expected = fp.load_expected(catalog.load_expected_gt(scene_id))
            expected_gt[scene_id] = expected
            local = _load(snapshots / "gt" / f"{scene_id}.scene.json")
            if local is None:
                report["scenes"][scene_id] = {"status": "not_built"}
            else:
                local_gt[scene_id] = local
                try:
                    report["scenes"][scene_id] = _summarize(fp.compare(local, expected))
                except (ValueError, KeyError, TypeError) as exception:
                    report["scenes"][scene_id] = _error(exception)
        entry: dict[str, Any] = {"scene_id": scene_id}
        local_input = _load(snapshots / "input" / f"{case.case_id}.scene.json")
        if local_input is None:
            entry["input"] = {"status": "not_built"}
        else:
            try:
                expected_input = fp.expand_delta(expected_gt[scene_id], catalog.load_input_delta(case))
                entry["input"] = _summarize(fp.compare(local_input, expected_input))
            except (ValueError, KeyError, TypeError) as exception:
                entry["input"] = _error(exception)
            if scene_id in local_gt:
                # A recipe target missing from the local GT (a different pack build)
                # raises here; that is a finding for this case, not a failed run.
                try:
                    predicted = rc.apply_offline(local_gt[scene_id], case.recipe,
                                                 sublevel_rename=sublevel_rename(case.recipe))
                    consistency = fp.compare(local_input, fp.fingerprint_snapshot(predicted))
                    entry["recipe_consistency"] = {
                        "status": "match" if consistency["match"] else "mismatch",
                        "mismatched_facets": fp.summarize_facets(consistency["mismatched"]),
                        "missing": consistency["missing"][:20],
                        "extra": consistency["extra"][:20],
                    }
                except (ValueError, KeyError, TypeError) as exception:
                    entry["recipe_consistency"] = _error(exception)
        report["cases"][case.case_id] = entry
    report["summary"] = {
        "scenes_matching": sum(1 for v in report["scenes"].values() if v.get("status") == "match"),
        "scenes_total": len(report["scenes"]),
        "inputs_matching": sum(1 for v in report["cases"].values() if v["input"].get("status") == "match"),
        "inputs_total": len(report["cases"]),
        "recipe_consistent": sum(1 for v in report["cases"].values()
                                 if (v.get("recipe_consistency") or {}).get("status") == "match"),
    }
    return report


def render_text(report: dict[str, Any]) -> str:
    lines = ["Code4Scene dataset verification", "=" * 31, ""]
    s = report["summary"]
    lines.append(f"GT levels matching the benchmark:    {s['scenes_matching']}/{s['scenes_total']}")
    lines.append(f"Input levels matching the benchmark: {s['inputs_matching']}/{s['inputs_total']}")
    lines.append(f"Inputs consistent with their recipe: {s['recipe_consistent']}/{s['inputs_total']}")
    lines.append("")
    for scene_id, entry in sorted(report["scenes"].items()):
        lines.append(f"[GT] {scene_id}: {entry['status'].upper()}")
        for reason in _reasons(entry):
            lines.append(f"      - {reason}")
    for case_id, entry in sorted(report["cases"].items()):
        status = entry["input"]["status"].upper()
        consistency = entry.get("recipe_consistency") or {}
        lines.append(f"[Input] {case_id}: {status} (recipe consistency: {consistency.get('status', 'n/a')})")
        for reason in _reasons(entry["input"]):
            lines.append(f"      - {reason}")
        if consistency.get("reason"):
            lines.append(f"      - recipe consistency: {consistency['reason']}")
    lines.append("")
    lines.append("A GT mismatch makes every Input of that scene mismatch as well. Scores obtained on a")
    lines.append("mismatching level are not comparable with the published leaderboard.")
    return "\n".join(lines) + "\n"
