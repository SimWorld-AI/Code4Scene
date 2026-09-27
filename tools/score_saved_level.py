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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from code4scene.core.bridge import Bridge
from code4scene.evaluation import primary_score, ue_evidence, verifiers
from code4scene.tasks import task as task_mod

DEPENDENCY_SCRIPT = (
    Path(ue_evidence.__file__).resolve().parents[1] / "ue_scripts" / "export_scene_dependencies.py"
)


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
    # The scene snapshot exporter reads the level that is currently open.
    loaded = bridge.exec_python_result(
        "import unreal\n"
        f"unreal.EditorLoadingAndSavingUtils.load_map({args.candidate_map!r})\n"
        "world = unreal.EditorLevelLibrary.get_editor_world()\n"
        "globals()['_C4S_OPEN'] = {'map': str(world.get_path_name()).split('.', 1)[0]}\n",
        "_C4S_OPEN", timeout=900.0)
    if loaded.get("map") != args.candidate_map.split(".", 1)[0]:
        raise SystemExit(f"scoring editor did not open {args.candidate_map}: {loaded}")

    record = {"scene_map": args.candidate_map, "official": {"level": args.candidate_map},
              "scene_dependencies": dependencies}
    ids = {"task_bundle_id": task.id, "episode_id": args.episode_id}
    # `scoring` marks the editor as the independent scoring editor; gt_repair and
    # physical_safety refuse to capture runtime scenes from any other editor.
    scoring = SimpleNamespace(bridge=bridge, measure_path=str(out / "measure.json"))
    reports = verifiers.run(task, record, ids, bridge=None, scoring=scoring, out_dir=str(out))
    result = {"benchmark_track": f"image_to_scene_{environment}",
              "scene_environment": environment, "task_file": str(args.task),
              "task_id": task.id, "reports": reports}
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
