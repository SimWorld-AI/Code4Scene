"""Score one saved image-to-scene level with the canonical verifier layer.

This is the "Recommended" route of docs/EVIDENCE_BUNDLE.md made concrete:

1. start a scoring editor on a project that holds the task's packs, the built
   Code4SceneInputs/Code4SceneGT levels and the candidate .umap (copied to the
   /Game path it was saved under), running tools/c4s_editor_bridge.py:

       C4S_BRIDGE_SOCK=/tmp/c4s.sock C4S_START_MAP=/Game/<candidate map> \\
       UnrealEditor-Cmd <Project>.uproject \\
           -ExecutePythonScript=<repo>/tools/c4s_editor_bridge.py \\
           -unattended -nosplash -nop4 -nosound -NullRHI

2. run this script against that socket:

       python tools/score_saved_level.py --task benchmark/public/.../task.yaml \\
           --candidate-map /Game/<candidate map> --bridge /tmp/c4s.sock --out runs/<case>

It writes <out>/result.json (overall_score = the paper case score) and the
scene evidence under <out>/scene_evidence/, ready for `code4scene make-bundle`.
Pass --shutdown to close the scoring editor afterwards.

As in the paper's scorer, the level is scored inside the answer scene's
footprint: the bounds pass (code4scene/evaluation/bounds.py) deletes actors
wholly outside the GT content box and clamps oversized flat sheets to it. The
pass edits a working copy saved next to the candidate (<candidate map>__c4s_scoring);
the saved level itself is never modified. result.json records what it changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from code4scene.core import inventory
from code4scene.core import scene as core_scene
from code4scene.core.bridge import Bridge
from code4scene.evaluation import bounds, primary_score, ue_evidence, verifiers
from code4scene.tasks import task as task_mod

DEPENDENCY_SCRIPT = (
    Path(ue_evidence.__file__).resolve().parents[1] / "ue_scripts" / "export_scene_dependencies.py"
)
WORKING_COPY_SUFFIX = "__c4s_scoring"


def _open(bridge: Bridge, package: str) -> dict:
    """Open a level in the scoring editor; report whether it is the one open now."""
    bridge.exec_python("globals().pop('_SB_LOADED', None)", timeout=60.0)
    return bridge.exec_python_result(core_scene.load_script(package), "_SB_LOADED", timeout=900.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, type=Path, help="the case's task.yaml")
    parser.add_argument("--candidate-map", required=True, help="/Game/... package of the saved level")
    parser.add_argument("--bridge", required=True, help="unix socket of tools/c4s_editor_bridge.py")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--episode-id", default="episode")
    parser.add_argument("--shutdown", action="store_true", help="stop the scoring editor when done")
    args = parser.parse_args(argv)

    task = task_mod.load(args.task)
    environment = str(getattr(task, "scene_environment", "") or "")
    if getattr(task, "case_type", "") != "image_to_scene" or environment not in ("indoor", "outdoor"):
        parser.error("this helper scores image-to-scene tasks only")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bridge = Bridge.unix(args.bridge)
    bridge.ping()

    # Candidate Integrity needs the saved level's dependency manifest.
    deps_path = out / "candidate.dependencies.json"
    prelude = (f"SCENE_DEPENDENCIES_OUTPUT = {str(deps_path)!r}\n"
               f"SCENE_DEPENDENCIES_MAP = {args.candidate_map!r}")
    dependencies = ue_evidence._run_editor_export(
        bridge, DEPENDENCY_SCRIPT, prelude, deps_path, "_C4S_DEPENDENCIES", 900.0)
    # The evaluation boundary comes from the answer scene, never the candidate.
    # Without an openable GT there is no boundary, and gt_repair reports the
    # missing GT itself.
    candidate = args.candidate_map.split(".", 1)[0]
    gt_map = task.ground_truth_map
    boundary, edge, scored_map = None, None, candidate
    if gt_map and _open(bridge, gt_map).get("loaded"):
        boundary = bounds.from_inventory(inventory.read(bridge, timeout=900.0), source_map=gt_map)
        loaded = _open(bridge, candidate)
        if not loaded.get("loaded"):
            raise SystemExit(f"scoring editor did not open {args.candidate_map}: {loaded}")
        edge = bounds.run(bridge, evaluation_bounds=boundary, timeout=900.0)
        root, name = (candidate + WORKING_COPY_SUFFIX).rsplit("/", 1)
        scored_map = core_scene.save(bridge, name, root=root, timeout=900.0)
    else:
        print(f"cannot open the answer scene {gt_map}; no bounds pass")
    # The scene snapshot exporter reads the level that is currently open, and
    # every Input/GT capture restores this level from disk afterwards.
    loaded = _open(bridge, scored_map)
    if not loaded.get("loaded"):
        raise SystemExit(f"scoring editor did not open {scored_map}: {loaded}")

    edge_discipline = ({"total_clamped": edge.get("clamped", 0), "total_deleted": edge.get("deleted", 0),
                        "passes": [edge]} if edge is not None else None)
    record = {"scene_map": scored_map, "official": {"level": scored_map},
              "scene_dependencies": dependencies, "evaluation_bounds": boundary,
              "edge_discipline": edge_discipline}
    ids = {"task_bundle_id": task.id, "episode_id": args.episode_id}
    # `scoring` marks the editor as the independent scoring editor; gt_repair and
    # physical_safety refuse to capture runtime scenes from any other editor.
    scoring = SimpleNamespace(bridge=bridge, measure_path=str(out / "measure.json"))
    reports = verifiers.run(task, record, ids, bridge=None, scoring=scoring, out_dir=str(out))
    result = {"benchmark_track": f"image_to_scene_{environment}",
              "scene_environment": environment, "task_file": str(args.task),
              "task_id": task.id, "candidate_map": candidate, "scored_map": scored_map,
              "evaluation_bounds": boundary, "edge_discipline": edge_discipline,
              "reports": reports}
    result = primary_score.apply_to_result(result)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    for report in reports:
        print(report.get("report_id"), report.get("status"), report.get("score"))
    print("overall_score", result.get("overall_score"))
    if args.shutdown:
        bridge.command("shutdown", {}, timeout=60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
