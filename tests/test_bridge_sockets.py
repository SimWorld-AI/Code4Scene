"""The reference bridges serve only their own user, and the client reads replies cheaply."""

from __future__ import annotations

import json
import os
import socket
import stat
import sys
import tempfile
import threading
import types
from pathlib import Path

import pytest

from code4scene.core import bridge as bridge_mod

REPO = Path(__file__).resolve().parents[1]
BRIDGES = ("c4s_editor_bridge.py", "c4s_render_bridge.py")


def _load(name, monkeypatch, sock):
    """The bridge's definitions, without the startup lines that serve."""
    fake = types.ModuleType("unreal")
    fake.log = fake.log_error = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "unreal", fake)
    monkeypatch.setenv("C4S_BRIDGE_SOCK", str(sock))
    source = (REPO / "tools" / name).read_text(encoding="utf-8")
    module = types.ModuleType(name[:-3])
    exec(compile(source[: source.index("\nif START_MAP:")], name, "exec"), module.__dict__)
    return module


@pytest.fixture
def short_dir():
    # A unix socket path must stay short; pytest's tmp_path can be too long.
    with tempfile.TemporaryDirectory(prefix="c4s", dir="/tmp") as path:
        yield Path(path)


@pytest.mark.parametrize("name", BRIDGES)
def test_the_socket_is_open_to_its_user_only(name, monkeypatch, short_dir):
    sock = short_dir / "s.sock"
    module = _load(name, monkeypatch, sock)
    server = module._bind(str(sock))
    try:
        mode = os.stat(sock).st_mode
        assert stat.S_ISSOCK(mode) and mode & 0o077 == 0
        client, peer = socket.socketpair(socket.AF_UNIX)
        with client, peer:
            assert module._same_user(peer)
    finally:
        server.close()


@pytest.mark.parametrize("name", BRIDGES)
def test_a_path_that_is_not_a_socket_is_never_replaced(name, monkeypatch, short_dir):
    sock = short_dir / "s.sock"
    module = _load(name, monkeypatch, sock)
    sock.write_text("not a socket")
    with pytest.raises(RuntimeError, match="not a socket"):
        module._bind(str(sock))
    assert sock.read_text() == "not a socket"
    sock.unlink()
    stale = socket.socket(socket.AF_UNIX)
    stale.bind(str(sock))
    stale.close()  # left behind by an earlier editor
    module._bind(str(sock)).close()


@pytest.mark.parametrize("name", BRIDGES)
def test_a_request_is_capped(name, monkeypatch, short_dir):
    module = _load(name, monkeypatch, short_dir / "s.sock")
    monkeypatch.setattr(module, "MAX_REQUEST_BYTES", 1000)

    class Conn:
        def settimeout(self, _value):
            pass

        def recv(self, size):
            return b"x" * min(size, 600)

    with pytest.raises(ValueError, match="larger than"):
        module._read_request(Conn())


def test_a_failed_connect_closes_its_socket(monkeypatch, short_dir):
    closed = []
    real = socket.socket

    class Tracking(real):
        def close(self):
            closed.append(True)
            super().close()

    monkeypatch.setattr(bridge_mod.socket, "socket", Tracking)
    with pytest.raises(bridge_mod.BridgeError):
        bridge_mod.Bridge.unix(short_dir / "missing.sock").command("ping", timeout=2.0)
    assert closed


def test_a_large_reply_is_parsed_once(monkeypatch, short_dir):
    sock = short_dir / "s.sock"
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(sock))
    server.listen(1)
    reply = json.dumps({"status": "ok", "result": "x" * 400_000}).encode()

    def serve():
        conn, _ = server.accept()
        with conn:
            conn.recv(65536)
            for start in range(0, len(reply), 50_000):
                conn.sendall(reply[start:start + 50_000])

    thread = threading.Thread(target=serve)
    thread.start()
    parses = []
    real_loads = json.loads
    monkeypatch.setattr(bridge_mod.json, "loads", lambda raw: parses.append(1) or real_loads(raw))
    try:
        answer = bridge_mod.Bridge.unix(sock).command("execute_python_script", timeout=10.0)
    finally:
        thread.join()
        server.close()
    assert answer["result"] == "x" * 400_000 and len(parses) == 1
