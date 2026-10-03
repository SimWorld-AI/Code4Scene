"""Build the Code4Scene public dataset from Fab packs on a stock UE 5.8 editor.

    python -m dataset_builder.build --project /path/to/Code4SceneData.uproject \\
        --editor /path/to/UnrealEditor-Cmd --dataset ./code4scene-dataset --steps all

Steps (run in this order by ``all``):

  init-project  enable the Python Editor Script and Editor Scripting Utilities
                plugins in the .uproject (a backup is written first)
  check         host-side check that every required pack folder and key asset
                is installed under <Project>/Content (no editor needed)
  blank         create the empty start level used by text-to-scene tasks
  gt            generate any scene supplements (e.g. a stand-in texture a pack
                references but does not ship), build each GT level from its
                pack demo map (identity tags + scene patches) and export a
                scene snapshot
  inputs        build each Input level from the local GT and its recipe and
                export a scene snapshot
  verify        compare the exported snapshots with the shipped fingerprints
  package       write the dataset as two trees: agent/ (what an agent may see:
                prompt, case facts, reference views) and scorer/ (task files,
                labels, recipes, cameras, fingerprints; never shown to agents)
  render        render the reference views into agent/ (needs a GPU; not
                headless-NullRHI)

Editor jobs are launched once per step with all selected cases. Use
``--dry-run`` to write the job files and print the commands without running.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import catalog
from . import verify as verify_mod

UE_DIR = Path(__file__).resolve().parent / "ue"
PACKAGE_REDIRECTS = catalog.BENCHMARK / "package_redirects.txt"
RENDERER_SETTINGS = catalog.BENCHMARK / "renderer_settings.txt"
RENDERER_SECTION = "[/Script/Engine.RendererSettings]"
ALL_STEPS = ("init-project", "check", "blank", "gt", "inputs", "verify", "package", "render")
REQUIRED_PLUGINS = ("PythonScriptPlugin", "EditorScriptingUtilities")


def log(message: str) -> None:
    print(f"[code4scene] {message}", flush=True)


def problem(args, message: str) -> None:
    """Log a failure the run must not report as success (main exits non-zero)."""
    log(message)
    if not hasattr(args, "problems"):
        args.problems = []
    args.problems.append(message)


# The dataset directory holds two trees (docs/BUILD_DATASET.md, "Using the built dataset"):
#   agent/<setting>/<case>/   the only files an agent under test may see
#   scorer/<setting>/<case>/  task files, labels, recipes, cameras and fingerprints
def agent_dir(dataset: Path, case) -> Path:
    return dataset / "agent" / case.setting / case.case_id


def scorer_dir(dataset: Path, case) -> Path:
    return dataset / "scorer" / case.setting / case.case_id


def raw_render_dir(dataset: Path, case) -> Path:
    return dataset / "work" / "references_raw" / case.setting / case.case_id


# ---------------------------------------------------------------------------
# What a task file may make the editor do
# ---------------------------------------------------------------------------

#: The only /Game roots the builder saves levels under. With --force it replaces
#: the levels it builds, so a task file may not name a level anywhere else.
OWNED_ROOTS = ("/Game/Code4SceneGT/", "/Game/Code4SceneInputs/", catalog.BLANK_STAGE.rsplit("/", 1)[0] + "/")


def _level_problems(levels) -> list:
    return [f"refusing to save {level!r}: the builder saves levels only under "
            + ", ".join(OWNED_ROOTS)
            for level in levels
            if not str(level or "").startswith(OWNED_ROOTS) or ".." in str(level).split("/")]


def _scene_levels(scene: dict) -> list:
    """The levels that canonicalizing a scene saves."""
    return [scene.get("ground_truth_map")] + [
        op.get("level") for op in scene.get("patches") or () if op.get("op") == "add_streaming_level"]


def _recipe_levels(recipe: dict) -> list:
    """The levels that materializing a recipe saves."""
    return [recipe.get("input_map")] + [
        op.get("to") for op in recipe.get("level_structure") or () if op.get("copy_package")]


def _console_allowed(command) -> bool:
    """Renderer and scalability settings and the view mode; nothing that runs code."""
    text = str(command)
    head = text.split(" ", 1)[0]
    return (head == "viewmode" or head.startswith(("r.", "sg."))) and not any(c in text for c in ";|\r\n")


def _inside(name) -> bool:
    """A relative file name that stays inside the directory it is joined to."""
    text = str(name or "").replace("\\", "/")
    return bool(text) and not text.startswith("/") and ":" not in text and ".." not in text.split("/")


def _camera_problems(cameras: dict) -> list:
    found = [f"console command {command!r} is not a renderer setting or view mode"
             for command in (cameras.get("capture") or {}).get("console") or ()
             if not _console_allowed(command)]
    for view in cameras.get("views") or ():
        for what, name in (("view name", view.get("name")),
                           ("output file", (view.get("publish") or {}).get("file"))):
            if not _inside(name):
                found.append(f"{what} {name!r} would leave the case directory")
    return found


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


def supplement_sources(scene: dict) -> list[str]:
    """Installed assets a scene's supplements are generated from (engine content excluded)."""
    return [op["source"] for op in scene.get("supplements") or ()
            if str(op.get("source", "")).startswith("/Game/")]


def supplement_tasks(scene_ids: list[str]) -> list[dict]:
    """One editor task per scene that has supplements (idempotent in the editor)."""
    tasks = []
    for scene_id in scene_ids:
        ops = catalog.load_scene(scene_id).get("supplements") or []
        if ops:
            tasks.append({"kind": "supplement", "id": f"{scene_id}:supplements", "ops": ops})
    return tasks


def check_packs(project: Path, cases: list[catalog.Case]) -> dict:
    content = project.parent / "Content"
    listing = catalog.pack_listing()
    report = {"content_dir": str(content), "roots": {}, "cases": {}}
    for case in cases:
        if not case.task_text:
            report["cases"][case.case_id] = {"status": "no_task_definition"}
            continue
        roots = [p for p in case.packs if p not in catalog.BUILDER_ROOTS + catalog.GENERATED_ROOTS]
        assets: list[str] = []
        if case.is_i2s:
            scene = catalog.load_scene(case.scene_id)
            roots = sorted(set(roots) | set(scene.get("content_roots") or []))
            assets = (list(scene.get("key_assets") or []) + recipe_assets(case.recipe)
                      + supplement_sources(scene))
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
        return [args.editor, str(args.project), "-run=pythonscript", f"-script={script}", *common,
                *list(args.editor_arg or [])]
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
    shown = [c or "<UnrealEditor-Cmd>" for c in command]
    log(f"command: C4S_JOB={shlex.quote(str(job_path))} C4S_UE_DIR={shlex.quote(str(UE_DIR))} "
        + shlex.join(shown))
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
    result = json.loads(result_path.read_text(encoding="utf-8"))
    failed = [t for t in result["tasks"] if t["status"] != "ok"]
    log(f"{name}: {len(result['tasks']) - len(failed)}/{len(tasks)} ok in {time.time() - started:.0f} s"
        + ("" if result.get("finished") else " (job did not finish)"))
    for task in failed:
        log(f"  FAILED {task.get('id')}: {task.get('error')}")
    missing = max(len(tasks) - len(result["tasks"]), 0)
    if failed or missing or not result.get("finished"):
        problem(args, f"{name}: " + ("" if result.get("finished") else "the job did not finish; ")
                + f"{len(failed)} task(s) failed, {missing} did not run (see jobs/{name}.log)")
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
        scene = catalog.load_scene(scene_id)
        refused = _level_problems(_scene_levels(scene))
        if refused:
            problem(args, f"gt: {scene_id}: " + "; ".join(refused))
            continue
        tasks.append({"kind": "canonicalize", "id": scene_id, "scene": scene,
                      "force": args.force,
                      "export": str(args.dataset / "snapshots" / "gt" / f"{scene_id}.scene.json")})
    if tasks:
        run_job(args, "gt", supplement_tasks([t["id"] for t in tasks]) + tasks)


def step_inputs(args, cases, ready):
    tasks = []
    for case in cases:
        if not case.is_i2s or not case.recipe or not ready.get(case.case_id, True):
            continue
        refused = _level_problems(_recipe_levels(case.recipe))
        if refused:
            problem(args, f"inputs: {case.case_id}: " + "; ".join(refused))
            continue
        tasks.append({"kind": "materialize", "id": case.case_id, "recipe": case.recipe, "force": args.force,
                      "export": str(args.dataset / "snapshots" / "input" / f"{case.case_id}.scene.json")})
    if tasks:
        scenes = selected_scenes([c for c in cases if c.case_id in {t["id"] for t in tasks}])
        run_job(args, "inputs", supplement_tasks(scenes) + tasks)


def step_verify(args, cases):
    if getattr(args, "dry_run", False):
        log(f"verify: would compare the snapshots under {args.dataset / 'snapshots'} and write reports/verify.*")
        return
    report = verify_mod.verify(args.dataset, cases)
    reports = args.dataset / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "verify.json").write_text(json.dumps(report, indent=1))
    text = verify_mod.render_text(report)
    (reports / "verify.txt").write_text(text, encoding="utf-8")
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


def _have_pillow() -> bool:
    try:
        import PIL  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _publish(raw: Path, publish: dict, target: Path) -> str:
    _trim_png(raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(publish["width"]), int(publish["height"])
    jpeg = publish.get("format") == "jpeg"
    if (jpeg or not _have_pillow()) and shutil.which("ffmpeg"):
        quality = ["-q:v", "3"] if jpeg else []
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-vf", f"scale={width}:{height}",
                        *quality, str(target)], check=True)
        return "ffmpeg"
    from PIL import Image  # type: ignore
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
        refused = _camera_problems(case.cameras)
        if refused:
            problem(args, f"render: {case.case_id}: " + "; ".join(refused))
            continue
        raw_dir = raw_render_dir(args.dataset, case)
        tasks.append({"kind": "render", "id": case.case_id, "map": case.recipe["ground_truth_map"],
                      "cameras": case.cameras, "output_dir": str(raw_dir)})
    if not tasks:
        return
    if not args.dry_run and not _have_pillow() and not shutil.which("ffmpeg"):
        raise SystemExit("render: publishing the reference views needs Pillow (pip install pillow) or ffmpeg")
    # One editor per GT map: the map is loaded before the editor starts ticking.
    by_map: dict[str, list[dict]] = {}
    for task in tasks:
        by_map.setdefault(task["map"], []).append(task)
    for game_map, group in by_map.items():
        try:
            run_job(args, "render-" + game_map.rstrip("/").split("/")[-2], group, render=True)
        except SystemExit as failure:
            # One map's editor failing must not cost the other maps their views.
            problem(args, f"render: {failure}")
    if args.dry_run:
        return
    rendered = {task["id"] for task in tasks}
    for case in cases:
        if not case.cameras or case.case_id not in rendered:
            continue
        case_dir = agent_dir(args.dataset, case)
        for view in case.cameras["views"]:
            raw = raw_render_dir(args.dataset, case) / f"{view['name']}.png"
            if raw.exists():
                how = _publish(raw, view["publish"], case_dir / view["publish"]["file"])
                log(f"render: {case.case_id}/{view['publish']['file']} ({how})")
                mean = _mean_luminance(raw)
                if mean is not None and mean < 20.0:
                    log(f"  WARNING {case.case_id}/{view['name']}: nearly black frame (mean luminance "
                        f"{mean:.1f}/255); raise --render-settle-ticks and render again")
            else:
                problem(args, f"render: {case.case_id}/{view['name']} missing")


def _agent_case(case, task: dict) -> dict:
    """Case facts a harness may pass to the agent, besides the prompt and the reference views."""

    inputs = task.get("inputs") or {}
    source = task.get("source") or {}
    facts = {"schema_version": "code4scene.agent_case.v1", "id": case.case_id, "setting": case.setting,
             "init_map": inputs.get("init_map"), "budget": inputs.get("budget"),
             "packs": list((task.get("assets") or {}).get("packs") or [])}
    if "size_m" in inputs:
        facts["size_m"] = inputs["size_m"]
    if source.get("reference_views"):
        facts["reference_views"] = list(source["reference_views"])
    return facts


def step_package(args, cases):
    """Write the task files as two trees.

    ``agent/<setting>/<case>/`` holds what an agent under test may see: the task prompt
    (``prompt.txt``), the case facts (``case.json``) and, after ``render``, the reference
    views. ``scorer/<setting>/<case>/`` holds everything else: ``task.yaml``, the label, the
    edit recipe, the reference cameras, the fingerprints and, for text-to-scene, the
    requirement bundle. Only the agent tree may be exposed to an agent.
    """

    if getattr(args, "dry_run", False):
        log(f"package: would write {sum(1 for c in cases if c.task_text)} cases under "
            f"{args.dataset / 'agent'} and {args.dataset / 'scorer'}")
        return
    try:
        import yaml  # type: ignore
    except ImportError:
        raise SystemExit("package needs PyYAML to read the task prompts: pip install pyyaml") from None
    packaged = []
    for case in cases:
        if not case.task_text:
            continue
        scorer = scorer_dir(args.dataset, case)
        scorer.mkdir(parents=True, exist_ok=True)
        for item in case.directory.iterdir():
            destination = scorer / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)
        task = yaml.safe_load(case.task_text) or {}
        agent = agent_dir(args.dataset, case)
        agent.mkdir(parents=True, exist_ok=True)
        prompt = str((task.get("inputs") or {}).get("prompt") or "")
        (agent / "prompt.txt").write_text(prompt, encoding="utf-8")
        (agent / "case.json").write_text(json.dumps(_agent_case(case, task), indent=1) + "\n",
                                         encoding="utf-8")
        old = args.dataset / case.setting / case.case_id
        if old.is_dir():
            log(f"package: WARNING {old} is from the old layout, which put scorer files next to "
                f"the reference views; delete it (agents may only see {args.dataset / 'agent'})")
        packaged.append(case)
    manifest = {"schema_version": "code4scene.dataset_manifest.v2",
                "layout": {"agent": "agent/<setting>/<case>", "scorer": "scorer/<setting>/<case>"},
                "cases": [{"id": c.case_id, "setting": c.setting} for c in packaged]}
    (args.dataset / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"package: {len(packaged)} cases; agent-visible files under {args.dataset / 'agent'}, "
        f"scorer-only files under {args.dataset / 'scorer'}")


def ensure_package_redirects(project: Path, dry_run: bool = False) -> int:
    """Add the package redirects the content packs need to the project's DefaultEngine.ini."""

    wanted = [line.strip() for line in PACKAGE_REDIRECTS.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith(("#", ";"))]
    ini = project.parent / "Config" / "DefaultEngine.ini"
    raw = ini.read_bytes() if ini.exists() else b""
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    text = raw.decode("utf-16" if encoding == "utf-16" else "utf-8-sig")
    present = {line.strip() for line in text.splitlines()}
    missing = [line for line in wanted if line not in present]
    if not missing:
        return 0
    if dry_run:
        log(f"package redirects: {len(missing)} line(s) would be added to {ini}")
        return len(missing)
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.strip() == "[CoreRedirects]"), None)
    if header is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines += ["[CoreRedirects]", *missing]
    else:
        lines[header + 1:header + 1] = missing
    if ini.exists():
        shutil.copy2(ini, ini.with_name(ini.name + ".bak"))
    ini.parent.mkdir(parents=True, exist_ok=True)
    ini.write_bytes((newline.join(lines) + newline).encode(encoding))
    log(f"package redirects: added {len(missing)} line(s) to {ini}")
    return len(missing)


def ensure_renderer_settings(project: Path, dry_run: bool = False) -> int:
    """Set the benchmark's renderer settings in the project's DefaultEngine.ini.

    Values already set to something else are replaced; everything else in the
    file is kept. Returns how many settings were added or changed.
    """

    wanted = {}
    for line in RENDERER_SETTINGS.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith(("#", ";")):
            key, value = line.strip().split("=", 1)
            wanted[key] = value
    ini = project.parent / "Config" / "DefaultEngine.ini"
    raw = ini.read_bytes() if ini.exists() else b""
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    text = raw.decode("utf-16" if encoding == "utf-16" else "utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.strip() == RENDERER_SECTION), None)
    end = len(lines)
    if header is not None:
        end = next((i for i in range(header + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
    present = {}
    if header is not None:
        for i in range(header + 1, end):
            key = lines[i].split("=", 1)[0].strip()
            if "=" in lines[i] and key in wanted:
                present.setdefault(key, []).append(i)
    changed = [key for key, value in wanted.items()
               if [lines[i].split("=", 1)[1].strip() for i in present.get(key, [])] != [value]]
    if not changed:
        return 0
    if dry_run:
        log(f"renderer settings: {len(changed)} would be set in {ini}")
        return len(changed)
    drop = {i for key in changed for i in present.get(key, [])}
    section = [f"{key}={wanted[key]}" for key in changed]
    if header is None:
        kept = lines
        if kept and kept[-1].strip():
            kept.append("")
        kept += [RENDERER_SECTION, *section]
    else:
        body = [line for i, line in enumerate(lines[header + 1:end], start=header + 1) if i not in drop]
        while body and not body[-1].strip():
            body.pop()
        tail = lines[end:]
        kept = lines[:header + 1] + body + section + ([""] if tail else []) + tail
    backup = ini.with_name(ini.name + ".bak")
    if ini.exists() and not backup.exists():
        shutil.copy2(ini, backup)
    ini.parent.mkdir(parents=True, exist_ok=True)
    ini.write_bytes((newline.join(kept) + newline).encode(encoding))
    log(f"renderer settings: set {len(changed)} in {ini}")
    return len(changed)


def missing_plugins(project: Path) -> list[str]:
    """Required editor plugins the .uproject does not enable (neither is on by default)."""

    try:
        plugins = json.loads(project.read_text(encoding="utf-8")).get("Plugins") or []
    except (OSError, ValueError):
        return list(REQUIRED_PLUGINS)
    enabled = {p.get("Name") for p in plugins if isinstance(p, dict) and p.get("Enabled")}
    return [name for name in REQUIRED_PLUGINS if name not in enabled]


def script_path_problem(script: Path, render: bool) -> str | None:
    """What in the job script's path the editor's command line would mangle, if anything.

    UE takes everything up to the first ".py" of the argument as the script, and
    -ExecCmds (render) also splits at commas and turns apostrophes into quotes.
    """

    text = str(script)
    if text.lower().find(".py") != len(text) - len(".py"):
        return "'.py' before the script name"
    if render and ("," in text or "'" in text):
        return "a comma or an apostrophe"
    return None


def step_init_project(args):
    data = json.loads(args.project.read_text(encoding="utf-8"))
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
    if changed and args.dry_run:
        log(f"init-project: would enable {', '.join(REQUIRED_PLUGINS)} in {args.project.name}")
    elif changed:
        backup = args.project.with_suffix(".uproject.bak")
        shutil.copy2(args.project, backup)
        args.project.write_text(json.dumps(data, indent="\t") + "\n", encoding="utf-8")
        log(f"init-project: enabled {', '.join(REQUIRED_PLUGINS)} (backup: {backup.name})")
    else:
        log("init-project: required plugins already enabled")
    ensure_package_redirects(args.project, args.dry_run)
    ensure_renderer_settings(args.project, args.dry_run)


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
    args.problems = []
    steps = list(ALL_STEPS) if "all" in args.steps else args.steps
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        parser.error(f"unknown step(s): {unknown}")
    needs_editor = any(s in steps for s in ("blank", "gt", "inputs", "render"))
    if needs_editor and not args.editor and not args.dry_run:
        parser.error("--editor (or UE_EDITOR) is required for editor steps")
    if (needs_editor or "init-project" in steps or "check" in steps) and not args.project.is_file():
        parser.error(f"no .uproject at {args.project}")
    settings = {name for pair in catalog.SETTINGS for name in pair}
    unknown = [s for s in args.settings or [] if s not in settings]
    if unknown:
        parser.error(f"unknown setting(s) {unknown}; use t2s, indoor and/or outdoor")
    cases = catalog.load_cases(args.cases, args.settings)
    unknown = sorted(set(args.cases or []) - {c.case_id for c in cases})
    if unknown:
        parser.error(f"unknown case id(s) {unknown}{' in the selected --settings' if args.settings else ''}; "
                     "the case ids are listed in benchmark/public-*-cases.txt")
    if not cases:
        parser.error("no cases selected")
    if needs_editor:
        wrong = script_path_problem(UE_DIR / "c4s_job.py", render="render" in steps)
        if wrong:
            parser.error(f"the editor cannot run {UE_DIR / 'c4s_job.py'}: the path contains {wrong}; "
                         "move the repository to a path without it")
    if any(step != "init-project" for step in steps):
        # init-project only edits the .uproject; do not leave an empty dataset dir behind.
        args.dataset.mkdir(parents=True, exist_ok=True)
    ready: dict[str, bool] = {}
    if "init-project" in steps:
        step_init_project(args)
    if needs_editor:
        absent = missing_plugins(args.project)
        if absent and not args.dry_run:
            log(f"{args.project.name} does not enable {', '.join(absent)}; run --steps init-project "
                f"first or enable them in Edit > Plugins")
            return 2
        if absent:
            log(f"WARNING: {args.project.name} does not enable {', '.join(absent)} (init-project enables them)")
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
        # Only these steps use the packs; the blocked cases are skipped either way.
        if blocked and any(s in steps for s in ("gt", "inputs", "render")) and not args.allow_missing_packs:
            if not args.dry_run:
                log("some packs are missing; install them or pass --allow-missing-packs")
                return 2
            log("some packs are missing; a real run needs them installed or --allow-missing-packs")
    if needs_editor:
        ensure_package_redirects(args.project, args.dry_run)
        ensure_renderer_settings(args.project, args.dry_run)
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
    if args.problems:
        log(f"finished with {len(args.problems)} problem(s):")
        for message in args.problems:
            log(f"  {message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
