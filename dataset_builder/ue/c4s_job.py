"""Entry point executed by the Unreal Editor for one dataset-builder job.

The host driver (``python -m dataset_builder.build``) writes a job file and
starts the editor with either

    UnrealEditor-Cmd <Project>.uproject -run=pythonscript -script="<this file>"

or

    UnrealEditor-Cmd <Project>.uproject -ExecutePythonScript="<this file>"

with the environment variable ``C4S_JOB`` pointing at the job file. Tasks run
in order; the result file is rewritten after every task so an interrupted
job still reports what finished.
"""

import json
import os
import sys
import time
import traceback

HERE = os.environ.get("C4S_UE_DIR") or os.path.dirname(os.path.abspath(globals().get("__file__", "")))
if HERE and HERE not in sys.path:
    sys.path.insert(0, HERE)

import unreal  # noqa: E402

import c4s_levels  # noqa: E402
import c4s_render  # noqa: E402
import c4s_snapshot  # noqa: E402


def _write(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def run_task(task):
    kind = task["kind"]
    if kind == "blank_stage":
        return c4s_levels.create_blank_stage(task["map"], task.get("force", False))
    if kind == "canonicalize":
        result = c4s_levels.canonicalize(task["scene"], task.get("force", False))
        if task.get("export"):
            c4s_levels.load_map(task["scene"]["ground_truth_map"])
            result["export"] = c4s_snapshot.export_snapshot(task["export"])
        return result
    if kind == "materialize":
        result = c4s_levels.materialize(task["recipe"], task.get("force", False))
        if task.get("export"):
            c4s_levels.load_map(task["recipe"]["input_map"])
            result["export"] = c4s_snapshot.export_snapshot(task["export"])
        return result
    if kind == "export":
        c4s_levels.load_map(task["map"])
        return c4s_snapshot.export_snapshot(task["output"])
    if kind == "render":
        c4s_levels.load_map(task["map"])
        cameras = task["cameras"]
        rendered = []
        for view in cameras["views"]:
            output = os.path.join(task["output_dir"], view["name"] + ".png")
            rendered.append(c4s_render.render_view(cameras["capture"], view, output))
        return {"map": task["map"], "views": rendered}
    raise RuntimeError("unknown task kind {}".format(kind))


def run_tick_driven(job, results, result_path):
    """Capture render tasks from a post-tick callback so the editor renders frames.

    Every task of a tick-driven job uses the same map. It is loaded here, in the
    script body: loading a map from inside a Slate tick callback crashes the
    editor. The editor then ticks ``settle_ticks`` times (and at least
    ``settle_seconds``) so shaders compile and textures stream before the first
    capture, and quits when every task has finished.
    """
    tasks = list(job["tasks"])
    maps = sorted({task["map"] for task in tasks})
    if len(maps) != 1:
        raise RuntimeError("a tick-driven job renders exactly one map, got {}".format(maps))
    settle_ticks = int(job.get("settle_ticks", 400))
    settle_seconds = float(job.get("settle_seconds", 20.0))
    load_started = time.time()
    c4s_levels.load_map(maps[0])
    state = {"index": 0, "ticks": 0, "since": time.time(), "load_seconds": time.time() - load_started}

    def finish():
        results["finished"] = True
        _write(result_path, results)
        unreal.unregister_slate_post_tick_callback(state["handle"])
        unreal.SystemLibrary.quit_editor()

    def tick(_delta):
        if state["index"] >= len(tasks):
            finish()
            return
        state["ticks"] += 1
        if state["ticks"] < settle_ticks or time.time() - state["since"] < settle_seconds:
            return
        task = tasks[state["index"]]
        started = time.time()
        entry = {"kind": task["kind"], "id": task.get("id")}
        try:
            cameras = task["cameras"]
            rendered = []
            for view in cameras["views"]:
                output = os.path.join(task["output_dir"], view["name"] + ".png")
                rendered.append(c4s_render.render_view(cameras["capture"], view, output))
            entry["result"] = {"map": task["map"], "views": rendered, "settle_ticks": state["ticks"],
                               "map_load_seconds": round(state["load_seconds"], 1)}
            entry["status"] = "ok"
        except Exception as error:
            entry["status"] = "error"
            entry["error"] = str(error)
            entry["traceback"] = traceback.format_exc()
            unreal.log_error("[code4scene] render {} failed: {}".format(task.get("id"), error))
        entry["seconds"] = round(time.time() - started, 1)
        results["tasks"].append(entry)
        _write(result_path, results)
        state["index"] += 1

    state["handle"] = unreal.register_slate_post_tick_callback(tick)


def main():
    job_path = os.environ.get("C4S_JOB")
    if not job_path:
        raise RuntimeError("C4S_JOB is not set")
    with open(job_path, "r", encoding="utf-8") as handle:
        job = json.load(handle)
    result_path = job["result_path"]
    results = {"schema_version": "code4scene.ue_job_result.v1", "job": job.get("name"),
               "engine_version": str(unreal.SystemLibrary.get_engine_version()), "tasks": []}
    _write(result_path, results)
    if job.get("tick_driven"):
        try:
            run_tick_driven(job, results, result_path)
        except Exception as error:
            # -ExecCmds keeps the editor open: record the failure and quit.
            for task in job["tasks"]:
                results["tasks"].append({"kind": task["kind"], "id": task.get("id"),
                                         "status": "error", "error": str(error),
                                         "traceback": traceback.format_exc()})
            results["finished"] = True
            _write(result_path, results)
            unreal.SystemLibrary.quit_editor()
        return
    for task in job["tasks"]:
        started = time.time()
        entry = {"kind": task["kind"], "id": task.get("id")}
        try:
            entry["result"] = run_task(task)
            entry["status"] = "ok"
        except Exception as error:
            entry["status"] = "error"
            entry["error"] = str(error)
            entry["traceback"] = traceback.format_exc()
            unreal.log_error("[code4scene] {} {} failed: {}".format(task["kind"], task.get("id"), error))
        entry["seconds"] = round(time.time() - started, 1)
        results["tasks"].append(entry)
        _write(result_path, results)
    results["finished"] = True
    _write(result_path, results)
    if job.get("quit_when_done", True) and not job.get("commandlet", False):
        unreal.SystemLibrary.quit_editor()


main()
