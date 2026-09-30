"""Editor-side bridge that can also take pictures, for scoring text-to-scene levels.

The protocol of tools/c4s_editor_bridge.py (code4scene.core.bridge.Bridge over a
unix socket) plus the screenshot commands code4scene.evaluation.render sends,
``take_screenshot`` and ``take_screenshot_batch``. A screenshot is drawn on a
later frame, so this editor has to render: start it on a GPU, without -NullRHI,
and with -ExecCmds (unlike -ExecutePythonScript it keeps the editor running
after the script returns):

    C4S_BRIDGE_SOCK=/tmp/c4s.sock [C4S_START_MAP=/Game/...] \\
    UnrealEditor-Cmd <Project>.uproject "-ExecCmds=py <repo>/tools/c4s_render_bridge.py" \\
        -RenderOffscreen -unattended -nosplash -nop4 -nosound [-graphicsadapter=<gpu index>]

Requests are first served from this script, where levels can still be opened.
``begin_rendering`` (or the first screenshot request) moves serving to a Slate
post-tick callback, so the editor draws frames between requests. From then on
the open level is fixed: opening a level from a Slate tick crashes the editor,
so a script that tries is refused. Pictures are taken as the benchmark's
scoring editors took them: one CameraActor labelled ``_SwShotCam`` with the
requested pose, field of view and optional exposure bias, photographed by
AutomationLibrary.take_high_res_screenshot. ``shutdown`` quits the editor; so
do 30 minutes without a request.
"""
#
# A reply to a screenshot request is sent once its files are on disk, which is
# frames later, so screenshot connections stay open across ticks. Requests are
# answered in arrival order: nothing new is accepted while a capture runs.
import ast
import contextlib
import gc
import io
import json
import math
import os
import socket
import time
import traceback

import unreal

SOCK = os.environ["C4S_BRIDGE_SOCK"]
START_MAP = os.environ.get("C4S_START_MAP", "")
IDLE_LIMIT_S = 1800.0
SHOT_TIMEOUT_S = 25.0  # per picture, as in the benchmark's bridge
SETTLE_TICKS = 10  # frames between the warm-up picture and the first real one
CAMERA_LABEL = "_SwShotCam"
POST_PROCESS_BIAS = "camera-component-post-process-bias"
MANUAL_FIXED_BIAS = "camera-component-manual-fixed-bias"
EXPOSURE_MODES = (POST_PROCESS_BIAS, MANUAL_FIXED_BIAS)
LEVEL_CHANGES = ("load_level", "load_map", "new_level")
NS = {"__name__": "__c4s_bridge__"}
STATE = {"server": None, "handle": None, "job": None, "last": time.time(), "warmed": False}


def _execute(script):
    out = io.StringIO()
    status, err = "ok", None
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(script, "<c4s-bridge>", "exec"), NS)
    except BaseException:
        status, err = "error", traceback.format_exc()
        out.write(err)
    return {"status": status, "error": err,
            "result": {"python_logs": out.getvalue().splitlines()}}


def _changes_level(script):
    """Whether a script opens or creates a level in its own top-level code.

    Function bodies are not searched: the scene exporters define a loader that
    they only call when asked for another level.
    """
    try:
        nodes = list(ast.parse(script).body)
    except SyntaxError:
        return False
    while nodes:
        node = nodes.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in LEVEL_CHANGES:
            return True
        nodes.extend(ast.iter_child_nodes(node))
    return False


def _read_request(conn):
    conn.settimeout(30.0)
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf += chunk
    return json.loads(buf.decode() or "{}")


def _send(conn, resp):
    try:
        conn.sendall(json.dumps(resp, default=str).encode())
    except Exception:
        unreal.log_error("c4s render bridge: " + traceback.format_exc())
    finally:
        try:
            conn.close()
        except Exception:
            pass


# -- pictures ----------------------------------------------------------------

def _shots(params, batch):
    """Validate a request into shot dicts; return (shots, error)."""
    raw = params.get("shots") if batch else [params]
    if batch and (not isinstance(raw, list) or not 1 <= len(raw) <= 16):
        return None, "shots must contain between 1 and 16 screenshot requests"
    shots = []
    for index, value in enumerate(raw):
        where = f"shots[{index}]" if batch else "the request"
        if not isinstance(value, dict):
            return None, f"{where} must be an object"
        filepath = value.get("filepath") or value.get("filename")
        location, rotation = value.get("camera_location"), value.get("camera_rotation")
        if not filepath:
            return None, f"{where} needs a filepath"
        if not (isinstance(location, (list, tuple)) and len(location) == 3
                and isinstance(rotation, (list, tuple)) and len(rotation) == 3):
            return None, f"{where} needs three-value camera_location and camera_rotation"
        bias, mode = value.get("exposure_bias_ev"), value.get("exposure_mode")
        if bias is not None:
            try:
                bias = float(bias)
            except (TypeError, ValueError):
                bias = math.nan
            if not math.isfinite(bias):
                return None, f"{where}: exposure_bias_ev must be finite"
            mode = mode or POST_PROCESS_BIAS
        elif mode is not None:
            return None, f"{where}: exposure_mode requires exposure_bias_ev"
        if mode is not None and mode not in EXPOSURE_MODES:
            return None, f"{where} has unsupported exposure_mode {mode!r}"
        filepath = os.path.abspath(str(filepath))
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError as exc:
            return None, f"cannot prepare {filepath}: {exc}"
        shots.append({"filepath": filepath, "filename": os.path.basename(filepath),
                      "width": int(value.get("width") or 1920), "height": int(value.get("height") or 1080),
                      "camera_location": [float(v) for v in location],
                      "camera_rotation": [float(v) for v in rotation],
                      "field_of_view": float(value.get("field_of_view") or 55.0),
                      "exposure_bias_ev": bias, "exposure_mode": mode})
    if batch and len({(s["exposure_bias_ev"], s["exposure_mode"]) for s in shots}) != 1:
        return None, "all shots in a batch must use one exposure_bias_ev and exposure_mode"
    return shots, None


def _camera():
    """The shared screenshot camera, created once; duplicates are removed."""
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    camera = component = None
    for actor in list(actors.get_all_level_actors()):
        try:
            if actor.get_actor_label() != CAMERA_LABEL:
                continue
            found = actor.get_component_by_class(unreal.CameraComponent)
            if camera is None and found is not None:
                camera, component = actor, found
            else:
                actors.destroy_actor(actor)
        except Exception:
            pass
    if camera is None:
        camera = actors.spawn_actor_from_class(unreal.CameraActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        component = camera.get_component_by_class(unreal.CameraComponent) if camera else None
    if camera is None or component is None:
        raise RuntimeError("could not create the screenshot camera")
    camera.set_actor_label(CAMERA_LABEL)
    return camera, component


def _shoot(camera, component, shot):
    camera.set_actor_location(unreal.Vector(*shot["camera_location"]), False, False)
    # Positional: camera_rotation is in unreal.Rotator's (roll, pitch, yaw) order.
    camera.set_actor_rotation(unreal.Rotator(*shot["camera_rotation"]), False)
    component.set_editor_property("field_of_view", shot["field_of_view"])
    if shot["exposure_bias_ev"] is not None:
        post = component.get_editor_property("post_process_settings")
        if shot["exposure_mode"] == MANUAL_FIXED_BIAS:
            post.set_editor_property("override_auto_exposure_method", True)
            post.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
            post.set_editor_property("override_auto_exposure_apply_physical_camera_exposure", True)
            post.set_editor_property("auto_exposure_apply_physical_camera_exposure", False)
        post.set_editor_property("override_auto_exposure_bias", True)
        post.set_editor_property("auto_exposure_bias", float(shot["exposure_bias_ev"]))
        component.set_editor_property("post_process_settings", post)
        component.set_editor_property("post_process_blend_weight", 1.0)
    unreal.AutomationLibrary.take_high_res_screenshot(shot["width"], shot["height"], shot["filepath"],
                                                     camera=camera)


def _landed(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _start_job(conn, kind, shots):
    # As UE's automation screenshots do: finish pending loads, shader compiles
    # and texture streaming before the first pose.
    try:
        unreal.AutomationLibrary.finish_loading_before_screenshot()
    except Exception:
        unreal.log_warning("c4s render bridge: " + traceback.format_exc())
    # The first picture after rendering begins is taken twice: the level's
    # textures stream in for the camera that first looks at them, so the first
    # render of a view shows unstreamed materials.
    warm = None if STATE["warmed"] else os.path.splitext(shots[0]["filepath"])[0] + ".warmup.png"
    job = {"conn": conn, "kind": kind, "shots": shots, "index": 0, "pending": None, "error": None,
           "warm": warm, "settle": 0,
           "deadline": time.time() + max(SHOT_TIMEOUT_S, SHOT_TIMEOUT_S * (len(shots) + bool(warm)))}
    STATE["job"] = job
    try:
        job["camera"], job["component"] = _camera()
    except Exception as exc:
        job["error"] = str(exc)
        _finish_job()
        return
    _advance_job()


def _advance_job():
    """Photograph the next pose once the previous picture is on disk."""
    job = STATE["job"]
    try:
        if job["warm"]:
            if not job["pending"]:
                _shoot(job["camera"], job["component"], {**job["shots"][0], "filepath": job["warm"]})
                job["pending"] = job["warm"]
                return
            if not _landed(job["pending"]):
                if time.time() > job["deadline"]:
                    _finish_job()
                return
            try:
                os.remove(job["warm"])
            except OSError:
                pass
            job["warm"], job["pending"], job["settle"] = None, None, SETTLE_TICKS
            STATE["warmed"] = True
        if job["settle"] > 0:
            job["settle"] -= 1
            return
        while True:
            if job["pending"]:
                if not _landed(job["pending"]):
                    if time.time() > job["deadline"]:
                        _finish_job()
                    return
                job["index"] += 1
                job["pending"] = None
            if job["index"] >= len(job["shots"]):
                _finish_job()
                return
            shot = job["shots"][job["index"]]
            _shoot(job["camera"], job["component"], shot)
            job["pending"] = shot["filepath"]
            return
    except Exception as exc:
        job["error"] = str(exc)
        _finish_job()


def _finish_job():
    job, STATE["job"] = STATE["job"], None
    shots = job["shots"]
    missing = sorted(s["filepath"] for s in shots if not _landed(s["filepath"]))
    exposure = {}
    if shots[0]["exposure_bias_ev"] is not None:
        exposure = {"camera_exposure_policy": shots[0]["exposure_mode"],
                    "camera_exposure_bias_ev": shots[0]["exposure_bias_ev"]}
    if job["kind"] == "take_screenshot":
        shot = shots[0]
        if missing:
            resp = {"status": "error", "filepath": shot["filepath"], "filename": shot["filename"],
                    "error": job["error"] or "screenshot file not produced at " + shot["filepath"]}
        else:
            resp = {"status": "success", "filepath": shot["filepath"], "filename": shot["filename"], **exposure}
    else:
        resp = {"status": "success" if not missing else "partial",
                "results": [{"status": "error" if s["filepath"] in missing else "success",
                             "filepath": s["filepath"], "filename": s["filename"],
                             "error": "screenshot file not produced" if s["filepath"] in missing else None}
                            for s in shots],
                "missing": missing, "camera_policy": "single-camera-slate-tick-batch", "ue_python_calls": 1,
                **exposure}
        if job["error"]:
            resp["error"] = job["error"]
    _send(job["conn"], resp)


# -- serving -----------------------------------------------------------------

def _screenshot(conn, kind, params):
    shots, error = _shots(params, kind == "take_screenshot_batch")
    if error:
        _send(conn, {"status": "error", "error": error})
        return
    _start_job(conn, kind, shots)


def _stop():
    handle, STATE["handle"] = STATE["handle"], None
    if handle is not None:
        unreal.unregister_slate_post_tick_callback(handle)
    server, STATE["server"] = STATE["server"], None
    if server is not None:
        server.close()
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    # The interpreter is finalized after the world is gone, and a wrapper a
    # script left behind then crashes it on the way out: drop them now.
    STATE["job"] = None
    NS.clear()
    NS["__name__"] = "__c4s_bridge__"
    gc.collect()
    unreal.SystemLibrary.quit_editor()


def _serve_tick(conn):
    """Answer one request while rendering; False when the tick should stop accepting."""
    req = _read_request(conn)
    kind = req.get("type")
    params = req.get("params") or {}
    if kind == "execute_python_script":
        script = params.get("script") or ""
        if _changes_level(script):
            _send(conn, {"status": "error", "error": "the open level cannot change once rendering has begun "
                                                     "(open it before begin_rendering or the first screenshot)"})
        else:
            _send(conn, _execute(script))
    elif kind in ("editor_status", "abandon_job"):
        _send(conn, {"status": "ok", "result": {"busy": False}})
    elif kind == "begin_rendering":
        _send(conn, {"status": "ok"})
    elif kind in ("take_screenshot", "take_screenshot_batch"):
        _screenshot(conn, kind, params)
        return STATE["job"] is None
    elif kind == "shutdown":
        _send(conn, {"status": "ok"})
        _stop()
        return False
    else:
        _send(conn, {"status": "error", "error": f"unsupported command {kind!r}"})
    return True


def _tick(_delta):
    try:
        if STATE["job"] is not None:
            _advance_job()
            if STATE["job"] is not None:
                return
        server = STATE["server"]
        while server is not None:
            try:
                conn, _ = server.accept()
            except (BlockingIOError, InterruptedError, TimeoutError):
                break
            STATE["last"] = time.time()
            try:
                if not _serve_tick(conn):
                    return
            except Exception:
                unreal.log_error("c4s render bridge: " + traceback.format_exc())
                _send(conn, {"status": "error", "error": traceback.format_exc()})
        if STATE["job"] is None and time.time() - STATE["last"] > IDLE_LIMIT_S:
            _stop()
    except Exception:
        unreal.log_error("c4s render bridge: " + traceback.format_exc())


def _begin_rendering():
    STATE["server"].setblocking(False)
    STATE["last"] = time.time()
    STATE["handle"] = unreal.register_slate_post_tick_callback(_tick)


def serve():
    """Serve from the startup script until rendering begins."""
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCK)
    server.listen(8)
    server.settimeout(1.0)
    STATE["server"] = server
    unreal.log("C4S_BRIDGE_READY " + SOCK)
    last = time.time()
    while time.time() - last < IDLE_LIMIT_S:
        try:
            conn, _ = server.accept()
        except TimeoutError:
            continue
        last = time.time()
        try:
            req = _read_request(conn)
            kind = req.get("type")
            params = req.get("params") or {}
            if kind == "execute_python_script":
                _send(conn, _execute(params.get("script") or ""))
            elif kind in ("editor_status", "abandon_job"):
                _send(conn, {"status": "ok", "result": {"busy": False}})
            elif kind == "begin_rendering":
                _begin_rendering()
                _send(conn, {"status": "ok"})
                return
            elif kind in ("take_screenshot", "take_screenshot_batch"):
                _begin_rendering()
                _screenshot(conn, kind, params)
                return
            elif kind == "shutdown":
                _send(conn, {"status": "ok"})
                break
            else:
                _send(conn, {"status": "error", "error": f"unsupported command {kind!r}"})
        except Exception:
            unreal.log_error("c4s render bridge: " + traceback.format_exc())
            _send(conn, {"status": "error", "error": traceback.format_exc()})
    _stop()


if START_MAP:
    unreal.EditorLoadingAndSavingUtils.load_map(START_MAP)
serve()
