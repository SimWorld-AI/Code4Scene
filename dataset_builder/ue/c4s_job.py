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
