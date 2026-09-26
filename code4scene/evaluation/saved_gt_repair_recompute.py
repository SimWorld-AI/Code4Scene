"""Recompute GT-repair correspondence leaves from frozen structured evidence.

This module never opens Unreal Engine and never calls a model.  It reads the
three scene graphs and correspondence audit referenced by an authoritative
result, reruns only the deterministic repair-target measurement, and rebuilds
its composite ancestors with the production GT-repair report functions.  A
stale local visual error is replaced only when deterministic correspondence
proves that no Candidate target exists and the zero prerequisite gate makes
that visual channel inapplicable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import contracts, primary_score, repair_score
from .assignment import match
from .composite import composite_report
from .context import Context
from .gt_geometry_compare import actor_pair_attribute_metrics
from .repair_target_scope import (
    MATCHING_ALGORITHM_VERSION,
    RepairTargetScope,
    measure_repair_target,
)
from .requirement_graph.repair_target_authoring import derive_repair_targets
from .verifiers import gt_repair


SCHEMA_VERSION = "saved-gt-repair-correspondence-recomputation.v3"
POLICY_ID = "gt-repair-independent-actor-correspondence-v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _leaf(report: Mapping[str, Any], leaf_id: str) -> dict[str, Any]:
    leaves = (report.get("metrics") or {}).get("leaf_results") or []
    matches = [
        value
        for value in leaves
        if isinstance(value, Mapping) and value.get("leaf_id") == leaf_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{report.get('report_id', '<report>')}: expected one "
            f"{leaf_id!r} leaf, found {len(matches)}"
        )
    return copy.deepcopy(dict(matches[0]))


def _replace_top_report(
    reports: Sequence[Mapping[str, Any]],
    report_id: str,
    replacement: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    replaced = 0
    for report in reports:
        if report.get("report_id") == report_id:
            values.append(copy.deepcopy(dict(replacement)))
            replaced += 1
        else:
            values.append(copy.deepcopy(dict(report)))
    if replaced != 1:
        raise ValueError(f"expected one top-level {report_id!r} report, found {replaced}")
    return values


def _gt_root(source_result: Mapping[str, Any]) -> dict[str, Any]:
    roots = [
        report
        for report in source_result.get("reports") or ()
        if isinstance(report, Mapping) and report.get("report_id") == "gt_repair"
    ]
    if len(roots) != 1:
        raise ValueError(f"source result needs one gt_repair report, found {len(roots)}")
    return copy.deepcopy(dict(roots[0]))


def _artifact(
    root: Mapping[str, Any],
    key: str,
) -> Path:
    artifacts = root.get("artifacts") or {}
    value = artifacts.get(key) if isinstance(artifacts, Mapping) else None
    path = Path(str(value or "")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source gt_repair artifact {key!r} is missing: {path}")
    return path


def frozen_structured_evidence(
    source_result: Mapping[str, Any],
) -> dict[str, Path]:
    """Resolve the exact structured artifacts recorded by a source result."""

    root = _gt_root(source_result)
    candidate = _artifact(root, "locality_audit.candidate_scene_graph")
    ground_truth = _artifact(
        root,
        "scene_diff.structured_scene_diff.actor_correspondence.ground_truth_scene_graph",
    )
    correspondence = _artifact(
        root,
        "scene_diff.structured_scene_diff.actor_correspondence.correspondence",
    )
    input_scene = candidate.with_name("input.scene.json")
    if not input_scene.is_file():
        raise FileNotFoundError(
            f"frozen Input scene graph is missing beside Candidate: {input_scene}"
        )
    return {
        "input": input_scene.resolve(),
        "candidate": candidate,
        "ground_truth": ground_truth,
        "correspondence": correspondence,
    }


def _read_object(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object: {path}")
    return value


def measure_frozen_repair_target(
    source_result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Run the production local measurement over frozen source artifacts."""

    root = _gt_root(source_result)
    old_local = _leaf(root, "repair_target_diff")
    evidence_paths = frozen_structured_evidence(source_result)
    input_scene = _read_object(evidence_paths["input"], "Input scene")
    candidate_scene = _read_object(evidence_paths["candidate"], "Candidate scene")
    ground_truth_scene = _read_object(evidence_paths["ground_truth"], "GT scene")
    correspondence = _read_object(evidence_paths["correspondence"], "correspondence audit")
    targets = tuple(derive_repair_targets(input_scene, ground_truth_scene))
    if not targets:
        raise ValueError("frozen GT-minus-Input scene graphs have no repair targets")
    scope = RepairTargetScope(
        input_scene=input_scene,
        canonical_scene=ground_truth_scene,
        targets=targets,
    )
    alignment = old_local.get("evidence") or {}
    alignment = alignment.get("global_alignment") or {}
    comparison = {
        "audit": correspondence,
        "metrics": {
            "alignment_anchor_count": alignment.get("anchor_count"),
            "global_translation_error_cm": alignment.get("translation_error_cm"),
            "global_yaw_error_deg": alignment.get("yaw_error_deg"),
        },
    }
    return (
        measure_repair_target(scope, candidate_scene, comparison),
        evidence_paths,
    )


def rebuild_result(
    source_result: Mapping[str, Any],
    repair_target: Mapping[str, Any],
    *,
    output_case_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace deterministic local leaves and rebuild production ancestors."""

    top_reports = source_result.get("reports")
    if not isinstance(top_reports, list):
        raise ValueError("source result has no report list")
    old_root = _gt_root(source_result)
    old_local = _leaf(old_root, "repair_target_diff")
    old_scene = _leaf(old_root, "scene_diff")
    locality = _leaf(old_root, "locality_audit")
    target_visual = _leaf(old_local, "target_visual_diff")
    ids = {
        "task_bundle_id": str(old_root["task_bundle_id"]),
        "episode_id": str(old_root["episode_id"]),
    }
    context = Context(record={}, task=None, ids=ids, out_dir=output_case_dir)
    local = gt_repair.report_from_repair_target_measurement(
        context,
        repair_target,
        target_visual=target_visual,
    )
    local["leaf_id"] = "repair_target_diff"
    root = composite_report(
        "gt_repair",
        context,
        [local, old_scene, locality],
        evidence=old_root.get("evidence") or {},
    )
    root["evidence"]["saved_evidence_recomputation"] = dict(provenance)
    repair_score.carry_actor_f1(root, old_root)
    root = gt_repair.finalize_report(root, local, old_scene)

    rebuilt = copy.deepcopy(dict(source_result))
    rebuilt["reports"] = _replace_top_report(
        top_reports,
        "gt_repair",
        root,
    )
    error_count = sum(
        report.get("status") == contracts.ERROR
        for report in rebuilt["reports"]
        if isinstance(report, Mapping)
    )
    rebuilt["error_report_count"] = error_count
    rebuilt["batch_status"] = "complete_with_errors" if error_count else "complete"
    rebuilt["saved_evidence_recomputation"] = dict(provenance)
    return primary_score.apply_to_result(rebuilt)


def recompute(source_result_path: Path, output_root: Path) -> Path:
    """Write one provenance-bearing offline override result."""

    started = time.monotonic()
    source_result_path = source_result_path.resolve()
    source_result = _read_object(source_result_path, "source result")
    task_id = str(source_result.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("source result has no task_id")
    old_root = _gt_root(source_result)
    old_local = _leaf(old_root, "repair_target_diff")
    old_visual = _leaf(old_local, "target_visual_diff")
    repair_target, evidence_paths = measure_frozen_repair_target(source_result)
    target_visual_short_circuit_kind = gt_repair._target_visual_short_circuit_kind(
        repair_target
    )
    target_visual_short_circuit = target_visual_short_circuit_kind is not None
    output_case_dir = output_root.resolve() / "results" / task_id
    output_case_dir.mkdir(parents=True, exist_ok=True)
    code_files = [
        Path(gt_repair.repair_success.__file__).resolve(),
        Path(gt_repair.__file__).resolve(),
        Path(__file__).resolve(),
        Path(measure_repair_target.__code__.co_filename).resolve(),
        Path(actor_pair_attribute_metrics.__code__.co_filename).resolve(),
        Path(match.__code__.co_filename).resolve(),
    ]
    upstream = source_result.get("saved_evidence_recomputation")
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_result": str(source_result_path),
        "source_result_sha256": _sha256(source_result_path),
        "source_batch_status": source_result.get("batch_status"),
        "scope": ("repair_target_deterministic_leaves_and_composite_ancestors_only"),
        "frozen_structured_evidence": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in evidence_paths.items()
        },
        "source_target_visual_leaf_sha256": _json_sha256(old_visual),
        "preserved_target_visual_leaf_sha256": (
            None if target_visual_short_circuit else _json_sha256(old_visual)
        ),
        "preserved_scene_diff_leaf_sha256": _json_sha256(_leaf(old_root, "scene_diff")),
        "preserved_locality_audit_leaf_sha256": _json_sha256(
            _leaf(old_root, "locality_audit")
        ),
        "ue_recapture_performed": False,
        "model_calls_performed": 0,
        "global_scene_diff_recomputed": False,
        "target_visual_recomputed": False,
        "target_visual_replaced_by_prerequisite_short_circuit": (
            target_visual_short_circuit
        ),
        "target_visual_short_circuit_kind": target_visual_short_circuit_kind,
        "upstream_saved_evidence_recomputation": (
            copy.deepcopy(upstream) if isinstance(upstream, Mapping) else None
        ),
        "code_sha256": {str(path): _sha256(path) for path in sorted(set(code_files))},
        "old_repair_target_score": old_local.get("score"),
        "matching_algorithm_version": MATCHING_ALGORITHM_VERSION,
        "failure_classification": repair_target.get("failure_classification"),
        "primary_classification": repair_target.get("primary_classification"),
        "exchangeable_group_count": repair_target.get("exchangeable_group_count"),
        "exchangeable_reassigned_actor_count": repair_target.get(
            "exchangeable_reassigned_actor_count"
        ),
    }
    rebuilt = rebuild_result(
        source_result,
        repair_target,
        output_case_dir=output_case_dir,
        provenance=provenance,
    )
    rebuilt_root = _gt_root(rebuilt)
    rebuilt_local = _leaf(rebuilt_root, "repair_target_diff")
    provenance["new_repair_target_score"] = rebuilt_local.get("score")
    provenance["new_gt_repair_score"] = rebuilt_root.get("score")
    rebuilt["saved_evidence_recomputation"] = copy.deepcopy(provenance)
    rebuilt_root["evidence"]["saved_evidence_recomputation"] = copy.deepcopy(provenance)
    rebuilt["reports"] = _replace_top_report(
        rebuilt["reports"],
        "gt_repair",
        rebuilt_root,
    )
    timings = dict(rebuilt.get("timings") or {})
    timings["offline_gt_repair_correspondence_recompute_seconds"] = round(
        time.monotonic() - started,
        3,
    )
    rebuilt["timings"] = timings
    output_path = output_case_dir / "result.json"
    output_path.write_text(
        json.dumps(rebuilt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute deterministic GT-repair correspondence leaves from "
            "frozen structured evidence without opening UE or calling a model."
        )
    )
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    output = recompute(args.source_result, args.out_dir)
    result = _read_object(output, "rebuilt result")
    root = _gt_root(result)
    local = _leaf(root, "repair_target_diff")
    print(
        json.dumps(
            {
                "result": str(output),
                "batch_status": result["batch_status"],
                "exchangeable_group_count": (
                    result["saved_evidence_recomputation"].get("exchangeable_group_count")
                ),
                "exchangeable_reassigned_actor_count": (
                    result["saved_evidence_recomputation"].get(
                        "exchangeable_reassigned_actor_count"
                    )
                ),
                "repair_target_score": local["score"],
                "gt_repair_score": root["score"],
                "matching_algorithm_version": (
                    result["saved_evidence_recomputation"].get("matching_algorithm_version")
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
