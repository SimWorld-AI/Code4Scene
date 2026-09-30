"""Reference editor-side bridge for code4scene.core.bridge.Bridge (unix socket).

Run inside the scoring editor:

    export C4S_BRIDGE_SOCK=$(mktemp -d)/c4s.sock
    [C4S_START_MAP=/Game/...] \
    UnrealEditor-Cmd <Project>.uproject -ExecutePythonScript=<repo>/tools/c4s_editor_bridge.py \
        -unattended -nosplash -nop4 -nosound -NullRHI

The bridge runs any Python it is sent: keep the socket in a directory only you
can open (as above), and start the scoring editor after the agent has exited.

Protocol (see code4scene/core/bridge.py): one JSON request per connection,
{"type": ..., "params": {...}} plus a newline; one JSON object back.
``execute_python_script`` runs params.script in one persistent namespace (so a
payload can store its result in a global that a later call reads) and returns
{"status": "ok"|"error", "result": {"python_logs": [...stdout lines...]}}.
``shutdown`` stops serving; the editor then closes. The bridge stops by itself
after 30 minutes without a request.
"""
#
# UnrealEditor-Cmd closes the editor as soon as the -ExecutePythonScript script
# returns, and UE's embedded Python does not run background threads while the
# game thread is idle, so this serves synchronously from the startup script.
import contextlib
import io
import json
import os
import socket
import stat
import struct
import time
import traceback

import unreal

SOCK = os.environ["C4S_BRIDGE_SOCK"]
START_MAP = os.environ.get("C4S_START_MAP", "")
IDLE_LIMIT_S = 1800.0
MAX_REQUEST_BYTES = 64 << 20  # a script, not a data upload
NS = {"__name__": "__c4s_bridge__"}


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


def _read_request(conn):
    conn.settimeout(30.0)
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf += chunk
        if len(buf) > MAX_REQUEST_BYTES:
            raise ValueError(f"request larger than {MAX_REQUEST_BYTES} bytes")
    return json.loads(buf.decode() or "{}")


def _bind(path):
    """A listening unix socket that only the user running the editor can use."""
    if os.path.lexists(path):
        if not stat.S_ISSOCK(os.lstat(path).st_mode):
            raise RuntimeError(f"C4S_BRIDGE_SOCK {path} exists and is not a socket")
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    previous = os.umask(0o077)
    try:
        server.bind(path)
    finally:
        os.umask(previous)
    return server


def _same_user(conn):
    """The bridge runs any Python it is sent, so it serves only its own user."""
    if not hasattr(socket, "SO_PEERCRED"):
        return True
    creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return struct.unpack("3i", creds)[1] == os.getuid()


def serve_forever():
    server = _bind(SOCK)
    server.listen(8)
    server.settimeout(1.0)
    unreal.log("C4S_BRIDGE_READY " + SOCK)
    last = time.time()
    while time.time() - last < IDLE_LIMIT_S:
        try:
            conn, _ = server.accept()
        except TimeoutError:
            continue
        stop = False
        try:
            if not _same_user(conn):
                resp = {"status": "error", "error": "the bridge serves only the user running the editor"}
            else:
                req = _read_request(conn)
                kind = req.get("type")
                params = req.get("params") or {}
                if kind == "execute_python_script":
                    resp = _execute(params.get("script") or "")
                elif kind in ("editor_status", "abandon_job"):
                    resp = {"status": "ok", "result": {"busy": False}}
                elif kind == "shutdown":
                    resp, stop = {"status": "ok"}, True
                else:
                    resp = {"status": "error", "error": f"unsupported command {kind!r}"}
            conn.sendall(json.dumps(resp, default=str).encode())
        except Exception:
            unreal.log_error("c4s bridge: " + traceback.format_exc())
        finally:
            try:
                conn.close()
            except Exception:
                pass
            last = time.time()
        if stop:
            break
    server.close()
    os.unlink(SOCK)


if START_MAP:
    unreal.EditorLoadingAndSavingUtils.load_map(START_MAP)
serve_forever()
