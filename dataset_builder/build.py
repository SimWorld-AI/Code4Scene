"""Build the Code4Scene public dataset from Fab packs on a stock UE 5.8 editor.

    python -m dataset_builder.build --project /path/to/Code4SceneData.uproject \\
        --editor /path/to/UnrealEditor-Cmd --dataset ./code4scene-dataset --steps all

Steps (run in this order by ``all``):

  init-project  enable the Python Editor Script and Editor Scripting Utilities
                plugins in the .uproject (a backup is written first)
  check         host-side check that every required pack folder and key asset
                is installed under <Project>/Content (no editor needed)
  blank         create the empty start level used by text-to-scene tasks
  gt            build each GT level from its pack demo map (identity tags +
                scene patches) and export a scene snapshot
  inputs        build each Input level from the local GT and its recipe and
                export a scene snapshot
  verify        compare the exported snapshots with the shipped fingerprints
  package       copy the task files into the dataset directory
  render        render the reference views (needs a GPU; not headless-NullRHI)

Editor jobs are launched once per step with all selected cases. Use
``--dry-run`` to write the job files and print the commands without running.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import catalog
from . import verify as verify_mod

UE_DIR = Path(__file__).resolve().parent / "ue"
ALL_STEPS = ("init-project", "check", "blank", "gt", "inputs", "verify", "package", "render")
REQUIRED_PLUGINS = ("PythonScriptPlugin", "EditorScriptingUtilities")


def log(message: str) -> None:
    print(f"[code4scene] {message}", flush=True)


# ---------------------------------------------------------------------------
# Host-side pack check
# ---------------------------------------------------------------------------

def asset_file(content: Path, game_path: str) -> Path | None:
    """Map /Game/A/B/C(.C[_C]) to its .uasset/.umap under Content, if present."""

    package = game_path.split(".", 1)[0]
    relative = package[len("/Game/"):]
    for suffix in (".uasset", ".umap"):
        candidate = content / (relative + suffix)
        if candidate.exists():
            return candidate
    return None


def recipe_assets(recipe: dict) -> list[str]:
    assets = []
    for op in recipe.get("operations") or []:
        if op.get("op") == "set_static_mesh" and op.get("static_mesh"):
            assets.append(op["static_mesh"])
        if op.get("op") == "spawn_actor":
            source = op.get("from") or {}
            for key in ("static_mesh", "class"):
                if str(source.get(key, "")).startswith("/Game/"):
                    assets.append(source[key])
    return assets


def check_packs(project: Path, cases: list[catalog.Case]) -> dict:
    content = project.parent / "Content"
    listing = catalog.pack_listing()
    report = {"content_dir": str(content), "roots": {}, "cases": {}}
    for case in cases:
        if not case.task_text:
            report["cases"][case.case_id] = {"status": "no_task_definition"}
            continue
        roots = [p for p in case.packs if p not in catalog.BUILDER_ROOTS]
        assets: list[str] = []
        if case.is_i2s:
            scene = catalog.load_scene(case.scene_id)
            roots = sorted(set(roots) | set(scene.get("content_roots") or []))
            assets = list(scene.get("key_assets") or []) + recipe_assets(case.recipe)
        else:
            assets = catalog.palette_assets(case)
        missing_roots = [r for r in roots if not (content / r).is_dir()]
        missing_assets = [a for a in assets if a.split("/")[2] not in missing_roots and asset_file(content, a) is None]
        for root in roots:
            entry = report["roots"].setdefault(root, {"installed": (content / root).is_dir(), "cases": []})
            entry["cases"].append(case.case_id)
            if root in listing:
                entry["listing"] = {k: listing[root].get(k) for k in ("title", "seller", "fab_url") if listing[root].get(k)}
        report["cases"][case.case_id] = {
            "status": "ready" if not missing_roots and not missing_assets else "blocked",
            "roots": roots, "missing_roots": missing_roots,
            "checked_assets": len(assets), "missing_assets": missing_assets[:25],
            "missing_asset_count": len(missing_assets),
        }
    report["summary"] = {
        "ready": sum(1 for v in report["cases"].values() if v["status"] == "ready"),
        "total": len(report["cases"]),
        "missing_roots": sorted(r for r, v in report["roots"].items() if not v["installed"]),
    }
    return report


# ---------------------------------------------------------------------------
# Editor jobs
# ---------------------------------------------------------------------------

def editor_command(args, job_path: Path, render: bool) -> list[str]:
    script = str(UE_DIR / "c4s_job.py")
    common = ["-unattended", "-nosplash", "-nop4", "-nosound", "-stdout", "-FullStdOutLogOutput"]
    if args.mode == "commandlet" and not render:
        return [args.editor, str(args.project), "-run=pythonscript", f"-script={script}", *common]
    if render:
        # -ExecutePythonScript closes the editor when the script returns, before
        # a single frame is rendered; -ExecCmds keeps it ticking, and the job
        # quits the editor itself (tick_driven job, see ue/c4s_job.py).
        command = [args.editor, str(args.project), f"-ExecCmds=py {script}", *common,
                   "-RenderOffscreen"]
        return command + list(args.editor_arg or [])
    command = [args.editor, str(args.project), f"-ExecutePythonScript={script}", *common]
    command += ["-NullRHI"]
    return command + list(args.editor_arg or [])


def run_job(args, name: str, tasks: list[dict], render: bool = False) -> dict:
    jobs = args.dataset / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    job_path = jobs / f"{name}.job.json"
    result_path = jobs / f"{name}.result.json"
    job = {"schema_version": "code4scene.ue_job.v1", "name": name, "result_path": str(result_path),
           "tasks": tasks, "quit_when_done": True, "commandlet": args.mode == "commandlet" and not render}
    if render:
        job.update(tick_driven=True, settle_ticks=args.render_settle_ticks,
                   settle_seconds=args.render_settle_seconds)
    job_path.write_text(json.dumps(job, indent=1))
    command = editor_command(args, job_path, render)
    env = dict(os.environ, C4S_JOB=str(job_path), C4S_UE_DIR=str(UE_DIR))
    log(f"{name}: {len(tasks)} task(s)")
    # The job script reads its job file from the environment; print it too so a
    # command copied from --dry-run output can be run by hand.
    log(f"command: C4S_JOB={job_path} C4S_UE_DIR={UE_DIR} "
        + " ".join(f'"{c}"' if " " in c else c for c in command))
    if args.dry_run:
        return {"dry_run": True, "tasks": []}
    if result_path.exists():
        result_path.unlink()
    started = time.time()
    pid_path = jobs / f"{name}.pid"
    with open(jobs / f"{name}.log", "w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        pid_path.write_text(f"{process.pid}\n")
        try:
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            log(f"{name}: editor timed out after {args.timeout} s (see jobs/{name}.log); stopping it")
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
        finally:
            pid_path.unlink(missing_ok=True)
    if not result_path.exists():
        raise SystemExit(f"{name}: the editor produced no result file; see {jobs / (name + '.log')}")
    result = json.loads(result_path.read_text())
    failed = [t for t in result["tasks"] if t["status"] != "ok"]
    log(f"{name}: {len(result['tasks']) - len(failed)}/{len(tasks)} ok in {time.time() - started:.0f} s"
        + ("" if result.get("finished") else " (job did not finish)"))
    for task in failed:
        log(f"  FAILED {task.get('id')}: {task.get('error')}")
    return result


def selected_scenes(cases: list[catalog.Case]) -> list[str]:
    seen = []
    for case in cases:
        if case.is_i2s and case.recipe and case.scene_id not in seen:
            seen.append(case.scene_id)
    return seen


def step_blank(args, cases):
    if not any(not c.is_i2s for c in cases):
        return
    run_job(args, "blank", [{"kind": "blank_stage", "id": "blank-stage", "map": catalog.BLANK_STAGE,
                            "force": args.force}])


def step_gt(args, cases, ready):
    tasks = []
    for scene_id in selected_scenes(cases):
        scene_cases = [c.case_id for c in cases if c.scene_id == scene_id]
        if not any(ready.get(c, True) for c in scene_cases):
            log(f"gt: skipping {scene_id} (packs missing)")
            continue
        tasks.append({"kind": "canonicalize", "id": scene_id, "scene": catalog.load_scene(scene_id),
                      "force": args.force,
                      "export": str(args.dataset / "snapshots" / "gt" / f"{scene_id}.scene.json")})
    if tasks:
        run_job(args, "gt", tasks)


def step_inputs(args, cases, ready):
    tasks = []
    for case in cases:
        if not case.is_i2s or not case.recipe or not ready.get(case.case_id, True):
            continue
        tasks.append({"kind": "materialize", "id": case.case_id, "recipe": case.recipe, "force": args.force,
                      "export": str(args.dataset / "snapshots" / "input" / f"{case.case_id}.scene.json")})
    if tasks:
        run_job(args, "inputs", tasks)


def step_verify(args, cases):
    report = verify_mod.verify(args.dataset, cases)
    reports = args.dataset / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "verify.json").write_text(json.dumps(report, indent=1))
    text = verify_mod.render_text(report)
    (reports / "verify.txt").write_text(text)
    print(text)


def _trim_png(raw: Path) -> None:
    """Drop bytes after the PNG IEND chunk.

    UE 5.8's export_render_target can leave trailing bytes after IEND; ffmpeg's
    image demuxer then reports "Invalid PNG signature" for a second frame.
    """

    data = raw.read_bytes()
    end = data.find(b"IEND")
    if data[:8] == b"\x89PNG\r\n\x1a\n" and end != -1 and len(data) > end + 8:
        raw.write_bytes(data[:end + 8])


def _mean_luminance(path: Path) -> float | None:
    try:
        from PIL import Image, ImageStat  # type: ignore
    except ImportError:
        return None
    with Image.open(path) as image:
        return float(ImageStat.Stat(image.convert("L")).mean[0])


def _publish(raw: Path, publish: dict, target: Path) -> str:
    _trim_png(raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(publish["width"]), int(publish["height"])
    if publish.get("format") == "jpeg" and shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-vf", f"scale={width}:{height}",
                        "-q:v", "3", str(target)], check=True)
        return "ffmpeg"
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        shutil.copyfile(raw, target.with_suffix(raw.suffix))
        return "copied (install Pillow or ffmpeg to resize/encode)"
    image = Image.open(raw).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)
    if publish.get("format") == "jpeg":
        image.save(target, "JPEG", quality=90)
    else:
        image.save(target, "PNG")
    return "pillow"


def step_render(args, cases, ready):
    tasks = []
    for case in cases:
        if not case.is_i2s or not case.cameras or not ready.get(case.case_id, True):
            continue
        raw_dir = args.dataset / case.setting / case.case_id / "references" / "raw"
        tasks.append({"kind": "render", "id": case.case_id, "map": case.recipe["ground_truth_map"],
                      "cameras": case.cameras, "output_dir": str(raw_dir)})
    if not tasks:
        return
    # One editor per GT map: the map is loaded before the editor starts ticking.
    by_map: dict[str, list[dict]] = {}
    for task in tasks:
        by_map.setdefault(task["map"], []).append(task)
    for game_map, group in by_map.items():
        run_job(args, "render-" + game_map.rstrip("/").split("/")[-2], group, render=True)
    if args.dry_run:
        return
    for case in cases:
        if not case.cameras:
            continue
        case_dir = args.dataset / case.setting / case.case_id
        for view in case.cameras["views"]:
            raw = case_dir / "references" / "raw" / f"{view['name']}.png"
            if raw.exists():
                how = _publish(raw, view["publish"], case_dir / view["publish"]["file"])
                log(f"render: {case.case_id}/{view['publish']['file']} ({how})")
                mean = _mean_luminance(raw)
                if mean is not None and mean < 20.0:
                    log(f"  WARNING {case.case_id}/{view['name']}: nearly black frame (mean luminance "
                        f"{mean:.1f}/255); raise --render-settle-ticks and render again")
            else:
                log(f"render: {case.case_id}/{view['name']} missing")


def step_package(args, cases):
    for case in cases:
        if not case.task_text:
            continue
        target = args.dataset / case.setting / case.case_id
        target.mkdir(parents=True, exist_ok=True)
        for item in case.directory.iterdir():
            destination = target / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)
    manifest = {"schema_version": "code4scene.dataset_manifest.v1",
                "cases": [{"id": c.case_id, "setting": c.setting} for c in cases if c.task_text]}
    (args.dataset / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"package: {len(manifest['cases'])} task directories under {args.dataset}")


def step_init_project(args):
    data = json.loads(args.project.read_text())
    plugins = data.setdefault("Plugins", [])
    names = {p.get("Name"): p for p in plugins}
    changed = False
    for name in REQUIRED_PLUGINS:
        if name not in names:
            plugins.append({"Name": name, "Enabled": True})
            changed = True
        elif not names[name].get("Enabled", False):
            names[name]["Enabled"] = True
            changed = True
    if changed:
        backup = args.project.with_suffix(".uproject.bak")
        shutil.copy2(args.project, backup)
        args.project.write_text(json.dumps(data, indent="\t") + "\n")
        log(f"init-project: enabled {', '.join(REQUIRED_PLUGINS)} (backup: {backup.name})")
    else:
        log("init-project: required plugins already enabled")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, required=True, help="path to the .uproject")
    parser.add_argument("--editor", default=os.environ.get("UE_EDITOR", ""),
                        help="UnrealEditor-Cmd executable (or set UE_EDITOR)")
    parser.add_argument("--dataset", type=Path, default=Path("code4scene-dataset"))
    parser.add_argument("--steps", nargs="+", default=["all"])
    parser.add_argument("--cases", nargs="*", help="limit to these case IDs")
    parser.add_argument("--settings", nargs="*", help="t2s, indoor and/or outdoor")
    parser.add_argument("--mode", choices=("editor", "commandlet"), default="editor",
                        help="how non-render jobs start the editor (default: -ExecutePythonScript)")
    parser.add_argument("--editor-arg", action="append", help="extra argument passed to the editor")
    parser.add_argument("--timeout", type=int, default=6 * 3600, help="seconds per editor job")
    parser.add_argument("--force", action="store_true", help="rebuild levels that already exist")
    parser.add_argument("--allow-missing-packs", action="store_true",
                        help="continue with the cases whose packs are installed")
    parser.add_argument("--render-settle-ticks", type=int, default=400,
                        help="editor frames to render before each reference capture")
    parser.add_argument("--render-settle-seconds", type=float, default=20.0,
                        help="minimum wall time before each reference capture")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.project = args.project.resolve()
    args.dataset = args.dataset.resolve()
    steps = list(ALL_STEPS) if "all" in args.steps else args.steps
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        parser.error(f"unknown step(s): {unknown}")
    needs_editor = any(s in steps for s in ("blank", "gt", "inputs", "render"))
    if needs_editor and not args.editor and not args.dry_run:
        parser.error("--editor (or UE_EDITOR) is required for editor steps")
    cases = catalog.load_cases(args.cases, args.settings)
    if not cases:
        parser.error("no cases selected")
    if any(step != "init-project" for step in steps):
        # init-project only edits the .uproject; do not leave an empty dataset dir behind.
        args.dataset.mkdir(parents=True, exist_ok=True)
    ready: dict[str, bool] = {}
    if "init-project" in steps:
        step_init_project(args)
    if "check" in steps or needs_editor:
        report = check_packs(args.project, cases)
        (args.dataset / "reports").mkdir(parents=True, exist_ok=True)
        (args.dataset / "reports" / "packs.json").write_text(json.dumps(report, indent=1))
        ready = {k: v["status"] == "ready" for k, v in report["cases"].items()}
        s = report["summary"]
        log(f"check: {s['ready']}/{s['total']} cases have their packs installed")
        for root in s["missing_roots"]:
            info = report["roots"][root].get("listing") or {}
            hint = ""
            if info.get("title") and info.get("fab_url"):
                hint = f" ({info['title']}: {info['fab_url']})"
            elif info.get("title"):
                hint = f" ({info['title']})"
            log(f"  missing Content/{root}{hint}")
        blocked = [k for k, v in ready.items() if not v]
        if blocked and needs_editor and not args.allow_missing_packs:
            log("some packs are missing; install them or pass --allow-missing-packs")
            return 2
    if "blank" in steps:
        step_blank(args, cases)
    if "gt" in steps:
        step_gt(args, cases, ready)
    if "inputs" in steps:
        step_inputs(args, cases, ready)
    if "verify" in steps:
        step_verify(args, cases)
    if "package" in steps:
        step_package(args, cases)
    if "render" in steps:
        step_render(args, cases, ready)
    return 0


if __name__ == "__main__":
    sys.exit(main())
