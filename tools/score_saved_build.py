"""Score one saved text-to-scene level with the canonical verifier layer.

The text-to-scene counterpart of tools/score_saved_level.py. Overview Alignment
judges four pictures of the level and Detailed Alignment photographs it too, so
the scoring editor has to render:

1. start a scoring editor with tools/c4s_render_bridge.py (the command line is
   in that file: a GPU, -ExecCmds, no -NullRHI) on a project that holds the
   task's packs and the candidate .umap (copied to the /Game path it was saved
   under). One editor scores one level: once it has taken pictures, the open
   level cannot change;

2. run this script against that socket, with the VLM judge configured
   (docs/SCORING.md, "The VLM judge"):

       python tools/score_saved_build.py --task benchmark/public/text-to-scene/<case>/task.yaml \\
           --candidate-map /Game/<candidate map> --bridge "$C4S_BRIDGE_SOCK" --out runs/<case>

As in the paper's scorer, the task plate is enforced first: the bounds pass
(code4scene/evaluation/bounds.py) deletes actors wholly outside the size_m
plate and clamps oversized flat sheets to it, on a working copy saved next to
the candidate (<candidate map>__c4s_scoring); the saved level itself is never
modified. The copy is photographed for the overview judge and scored. It
writes <out>/result.json (overall_score = the paper case score) with the
pictures and scene evidence beside it, ready for `code4scene make-bundle`, and
exits with an error instead when the pictures could not be taken or Detailed
or Overview Alignment could not be measured. Pass --shutdown to close the
scoring editor afterwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from code4scene.core import scene as core_scene
from code4scene.core.bridge import Bridge
from code4scene.evaluation import bounds, case_outcome, render_capture, score_policy, ue_evidence, verifiers
from code4scene.resources import config_file
from code4scene.tasks import task as task_mod

DEPENDENCY_SCRIPT = (
    Path(ue_evidence.__file__).resolve().parents[1] / "ue_scripts" / "export_scene_dependencies.py"
)
WORKING_COPY_SUFFIX = "__c4s_scoring"
RENDERER_SETTINGS = Path(__file__).resolve().parents[1] / "benchmark" / "renderer_settings.txt"
# The settings that change what the pictures look like; checked in the scoring editor.
RENDERER_CHECKED = ("r.DynamicGlobalIlluminationMethod", "r.ReflectionMethod", "r.Shadow.Virtual.Enable",
                    "r.GenerateMeshDistanceFields", "r.AllowStaticLighting",
                    "r.DefaultFeature.AutoExposure.ExtendDefaultLuminanceRange",
                    "r.DefaultFeature.LocalExposure.HighlightContrastScale",
                    "r.DefaultFeature.LocalExposure.ShadowContrastScale")
POLICY = ("score-policies", "text-to-scene-human-aligned.yaml")
JUDGED = ("semantic_requirements", "overview_prompt_alignment")


def _open(bridge: Bridge, package: str) -> dict:
    """Open a level in the scoring editor; report whether it is the one open now."""
    bridge.exec_python("globals().pop('_SB_LOADED', None)", timeout=60.0)
    return bridge.exec_python_result(core_scene.load_script(package), "_SB_LOADED", timeout=900.0)


def _renderer_mismatches(bridge: Bridge) -> dict:
    """Checked renderer settings of the scoring editor that differ from benchmark/renderer_settings.txt."""
    wanted = {}
    for line in RENDERER_SETTINGS.read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().partition("=")
        if key in RENDERER_CHECKED:
            wanted[key] = 1.0 if value == "True" else 0.0 if value == "False" else float(value)
    observed = bridge.exec_python_result(
        "import unreal\n"
        f"globals()['_C4S_RENDERER'] = {{name: unreal.SystemLibrary.get_console_variable_float_value(name) "
        f"for name in {sorted(wanted)!r}}}\n",
        "_C4S_RENDERER", timeout=60.0)
    return {key: (observed.get(key), value) for key, value in wanted.items()
            if not isinstance(observed.get(key), (int, float)) or abs(observed[key] - value) > 1e-6}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, type=Path, help="the case's task.yaml")
    parser.add_argument("--candidate-map", required=True, help="/Game/... package of the saved level")
    parser.add_argument("--bridge", required=True, help="unix socket of tools/c4s_render_bridge.py")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--episode-id", default="episode")
    parser.add_argument("--shutdown", action="store_true", help="stop the scoring editor when done")
    args = parser.parse_args(argv)

    task = task_mod.load(args.task)
    if getattr(task, "case_type", "") != "prompt_to_scene":
        parser.error("this helper scores text-to-scene tasks only")
    if task.half_extent_m is None:
        parser.error("the task declares no plate (inputs.size_m)")
    candidate = args.candidate_map.split(".", 1)[0]
    if candidate == str(task.init_map).split(".", 1)[0]:
        parser.error(f"{args.candidate_map} is the task's own start level; save the agent's scene "
                     f"under a path of its own and score that")
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
    if dependencies.get("status") != "success":
        # A failed export lists no unresolved packages; recorded as it is, that
        # would read as "everything resolved". Without a manifest Candidate
        # Integrity reports an error and the score is withheld.
        print(f"dependency export failed: {dependencies.get('error') or dependencies.get('status')}")
        dependencies = None
    loaded = _open(bridge, candidate)
    if not loaded.get("loaded"):
        raise SystemExit(f"scoring editor did not open {args.candidate_map}: {loaded}")
    boundary = bounds.from_half_extent_m(task.half_extent_m)
    edge = bounds.run(bridge, evaluation_bounds=boundary, timeout=900.0)
    root, name = (candidate + WORKING_COPY_SUFFIX).rsplit("/", 1)
    scored_map = core_scene.save(bridge, name, root=root, timeout=900.0)
    loaded = _open(bridge, scored_map)
    if not loaded.get("loaded"):
        raise SystemExit(f"scoring editor did not open {scored_map}: {loaded}")

    mismatches = _renderer_mismatches(bridge)
    if mismatches:
        raise SystemExit(f"the scoring project's renderer settings differ from benchmark/renderer_settings.txt "
                         f"(observed, expected): {mismatches}; run `python -m dataset_builder.build --project "
                         f"<Project>.uproject --steps init-project` and restart the editor")
    # From here the editor draws frames between requests and the open level is fixed.
    bridge.command("begin_rendering", {}, timeout=60.0)
    renders, render_errors = render_capture.capture_render_evidence(task, bridge, out)
    if render_errors:
        raise SystemExit(f"could not photograph {scored_map} for the overview judge: {render_errors}")

    edge_discipline = {"total_clamped": edge.get("clamped", 0), "total_deleted": edge.get("deleted", 0),
                       "passes": [edge]}
    record = {"scene_map": scored_map, "official": {"level": scored_map},
              "scene_dependencies": dependencies, "size_m": task.size_m,
              "evaluation_bounds": boundary, "edge_discipline": edge_discipline,
              "render_evidence": {protocol: rendered.as_dict() for protocol, rendered in sorted(renders.items())}}
    ids = {"task_bundle_id": task.id, "episode_id": args.episode_id}
    # `scoring` marks the editor as the independent scoring editor, and Detailed
    # Alignment photographs the level through its bridge.
    scoring = SimpleNamespace(bridge=bridge, measure_path=str(out / "measure.json"))
    reports = verifiers.run(task, record, ids, bridge=None, scoring=scoring, out_dir=str(out),
                            artifacts_dir=str(out), render_evidence=renders)
    result = {"benchmark_track": "text_to_scene", "scene_environment": None, "task_file": str(args.task),
              "task_id": task.id, "candidate_map": candidate, "scored_map": scored_map,
              "evaluation_bounds": boundary, "edge_discipline": edge_discipline,
              "render_evidence": record["render_evidence"], "reports": reports}
    result = score_policy.apply_to_result(result, score_policy.load(config_file(*POLICY)))
    for report in reports:
        print(report.get("report_id"), report.get("status"), report.get("score"))
    if args.shutdown:
        bridge.command("shutdown", {}, timeout=60.0)
    # Missing evidence earns zero in the policy, which would publish a
    # physics-only score as if the scene had been judged. An invalid candidate
    # is different: it scores 0 by the protocol's hard gate.
    gated = (result.get("case_outcome") or {}).get("classification") in {
        case_outcome.MODEL_INVALID, case_outcome.INVALID}
    unmeasured = {str(r.get("report_id")): r.get("failure_reason") or r.get("status")
                  for r in reports if r.get("report_id") in JUDGED and r.get("status") != "measured"}
    if unmeasured and not gated:
        (out / "result.unscored.json").write_text(json.dumps(result, indent=1, default=str))
        raise SystemExit(f"not scored: {unmeasured} (reports in {out / 'result.unscored.json'})")
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    print("overall_score", result.get("overall_score"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
