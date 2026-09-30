"""Client for the UE python bridge, over TCP or a unix socket.

Protocol: one JSON object per request, newline-terminated, over a fresh
connection per call. The bridge replies with a single JSON object; the byte
stream may arrive fragmented, so the reader accumulates until it parses.

A unix socket is the better default where both ends share a host, for two
reasons that are not about performance:

* **It is the isolation that holds.** An agent with a shell on the same host
  can reach a bridge on ``127.0.0.1``. A socket is a filesystem object: leave
  it out of the agent's mounts and the bridge is unreachable, with no firewall
  rule and no network namespace.
* **It removes a collision surface.** Several environments on one host each
  need their own bridge port, and on a shared host network a taken port is a
  real failure. Paths do not collide.

Every failure raises :class:`BridgeError`, so no scoring pass continues on
empty data. Callers decide explicitly how to degrade.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


#: Prefix the editor prints when handing a stored result back.
RESULT_MARK = "SCENEBENCH_RESULT"
#: How many times to re-read a stored result whose line fell outside the
#: returned log window. Cheap (one short editor call) against a failure
#: mode that discards a whole episode.
RESULT_FETCH_TRIES = 4
RESULT_FETCH_WAIT_S = 3.0
#: Cleanup transport timeout runs only in a daemon thread after the caller's
#: deadline, so it cannot extend the public operation timeout.
ABANDON_NOTIFY_TIMEOUT_S = 2.0

#: Longest usable unix socket path. sockaddr_un.sun_path is 108 bytes on Linux
#: and 104 on macOS/BSD, minus the terminator. Taking the smaller keeps a path
#: that works on a developer's laptop working on the lab hosts.
MAX_SOCKET_PATH = 103


class BridgeError(Exception):
    """Connection, timeout, or protocol failure talking to the UE bridge."""


@dataclass(frozen=True)
class Bridge:
    """One environment's bridge endpoint.

    Either a TCP ``host``/``port`` or a unix socket ``path`` — never both, and
    never neither, because a Bridge that could not say where it connects would
    fail at the first call instead of at construction.
    """

    host: str | None = None
    port: int | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        tcp = self.host is not None and self.port is not None
        if tcp == (self.path is not None):
            raise ValueError(
                "a Bridge needs exactly one endpoint: host and port, or path")
        if self.path is not None and len(str(self.path)) > MAX_SOCKET_PATH:
            # The kernel's sockaddr_un is a fixed-size buffer, so this is a
            # hard limit and not a guideline. An instance directory nested
            # under a long data root reaches it easily, and the raw failure is
            # an OSError from bind() that says nothing about which path or why.
            raise ValueError(
                f"unix socket path is {len(str(self.path))} characters, over the "
                f"{MAX_SOCKET_PATH}-character limit this platform allows: "
                f"{self.path}. Put the socket somewhere shallow (a short "
                f"per-run directory) rather than beside the instance tree.")

    @classmethod
    def unix(cls, path: str | Path) -> Bridge:
        """A bridge reachable only through the filesystem."""
        return cls(path=str(path))

    @property
    def endpoint(self) -> str:
        """How this bridge is addressed, for error messages and records."""
        return self.path if self.path else f"{self.host}:{self.port}"

    def _connect(self, timeout: float) -> socket.socket:
        if self.path:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(self.path)
            return sock
        return socket.create_connection((self.host, self.port), timeout=timeout)

    def command(self, type_: str, params: dict[str, Any] | None = None,
                timeout: float = 120.0) -> dict[str, Any]:
        """Send one command and return the parsed response object."""
        payload = json.dumps({"type": type_, "params": params or {}}) + "\n"
        deadline = time.monotonic() + timeout
        try:
            with self._connect(timeout) as sock:
                sock.sendall(payload.encode())
                buf = b""
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise BridgeError(
                            f"timeout after {timeout:.0f}s waiting for '{type_}' "
                            f"from {self.endpoint}")
                    sock.settimeout(remaining)
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    try:
                        return json.loads(buf)
                    except json.JSONDecodeError:
                        continue  # response not complete yet
        except (TimeoutError, OSError) as e:
            raise BridgeError(f"{self.endpoint} '{type_}' failed: {e}") from e
        # Connection closed; a complete trailing object is still a valid reply.
        if buf:
            try:
                return json.loads(buf)
            except json.JSONDecodeError as e:
                raise BridgeError(
                    f"{self.endpoint} closed mid-response for '{type_}'") from e
        raise BridgeError(f"{self.endpoint} closed without a response for '{type_}'")

    def exec_python(self, script: str, timeout: float = 120.0) -> dict[str, Any]:
        """Run an editor-python script via the bridge."""
        return self.command("execute_python_script", {"script": script}, timeout=timeout)

    def python_logs(self, response: dict[str, Any]) -> str:
        """Join ``result.python_logs`` from an ``exec_python`` response."""
        try:
            return "\n".join(response["result"]["python_logs"])
        except (KeyError, TypeError):
            return ""

    def ping(self, timeout: float = 90.0) -> None:
        """Round-trip a print through the editor python; raise if it fails."""
        marker = "SCENEBENCH_PING_OK"
        response = self.exec_python(f"print({marker!r})", timeout=timeout)
        if marker not in self.python_logs(response):
            raise BridgeError(
                f"{self.endpoint} answered but editor python did not "
                f"echo the ping (response keys: {sorted(response)})")

    def busy(self, timeout: float = 20.0) -> dict[str, Any]:
        """Whether the editor is mid-script, answered WITHOUT asking the editor.

        UE runs python on the game thread and its command queue is serial, so
        anything that asks the editor a question waits behind whatever the
        editor is doing. A health check built that way cannot tell a busy
        editor from a dead one.

        This reads the evidence the job model already writes down instead: a
        capture file that has been opened and not yet renamed means a script is
        running. No editor round trip, so it answers whatever the game thread
        is doing.
        """
        return self.command("editor_status", {}, timeout=timeout)

    def exec_python_result(self, script: str, key: str,
                           timeout: float = 300.0) -> Any:
        """Run a payload that STORES its result, then read the result back.

        The logs a call returns are a slice of the editor's log taken over a
        time window — not that call's own output. A payload that runs for more
        than a fraction of a second loses its earliest or its latest lines,
        whichever falls outside the window. Measured against UE 5.8 on a live
        editor: a 0.5 s payload came back missing its closing line, and a 1 s
        payload came back missing its opening one.

        Every result that travelled only as a printed marker was therefore a
        race, and it lost in the worst direction: the scene save prints its
        marker after ~0.8 s of package writing, so a save that had in fact
        succeeded was recorded as a failure and its .umap — the artifact
        official scoring reloads — was never collected.

        So a slow payload no longer announces its own result. It assigns it to
        a global, and this fetches it with a second call short enough that the
        window cannot drift past the answer.
        """
        deadline = time.monotonic() + timeout
        response = self.exec_python(script, timeout=timeout)
        # Refusals raised by policy guards (for example an ``allow_ai``
        # bridge receiving a frozen-GT path) are complete results, not slow
        # jobs.  Continuing to the global fetch discards the real cause and
        # replaces it with the misleading claim that the payload stored
        # nothing.  Surface the bridge's own explanation before doing any
        # follow-up call so a scorer configuration error is immediately
        # actionable and never retried as if it were a payload race.
        self._raise_response_error(response, f"editor payload for {key!r}")
        # The payload may outlive the server's own wait, in which case it says
        # so and names the job. That handle used to be discarded on this line
        # and the failure raised two lines later: `timeout` was a promise
        # nothing kept, because the fetch below was pinned at 60 s while the
        # server waited 120 s, so no payload could exceed ~190 s however long
        # its caller asked for. A 553-actor save takes longer than that, and
        # was recorded as a failed save whose .umap — the artifact official
        # scoring reloads — was never collected, while the editor wrote it.
        self._await_job(response, deadline)

        fetch = (f"import json as _j\nprint({RESULT_MARK!r} + ' ' + "
                 f"_j.dumps(globals().get({key!r})))")
        # Short, but not shorter than what is left: the fetch queues behind the
        # payload on the editor's serial command queue.
        # The fetch is subject to the very drift it exists to defeat: its own
        # line can fall outside the window it gets back, and then a payload
        # that stored its result perfectly well reads as one that stored
        # nothing. Seen twice in one session — a finished 115-actor scene
        # recorded as `infra_error` with `umap: null`, and a repair input that
        # had loaded (its own SCENE_LOADED ... True came back in these very
        # logs) failing the episode before the agent made a single call.
        # Reading a global is idempotent, so ask again rather than conclude
        # from one missed window.
        line = None
        logs = ""
        for attempt in range(RESULT_FETCH_TRIES):
            if attempt:
                time.sleep(RESULT_FETCH_WAIT_S)
            remaining = max(30.0, min(120.0, deadline - time.monotonic()))
            try:
                logs = self.python_logs(self.exec_python(fetch, timeout=remaining))
            except BridgeError:
                # A fetch that TIMED OUT is the same situation as one whose
                # line fell outside the log window, and it was not retried:
                # the loop below only reruns when a fetch returns without the
                # marker, so a socket timeout escaped as a failure with the
                # payload sitting in a global, already computed.
                #
                # The 120 s cap above is why this happens at all. The fetch
                # queues behind the payload on the editor's serial command
                # queue, and on a loaded host with a large content set the
                # queue is longer than the cap no matter what the caller
                # asked for. Observed against an instance mounting 498 packs.
                if time.monotonic() >= deadline or attempt == RESULT_FETCH_TRIES - 1:
                    raise
                continue
            line = next((line for line in reversed(logs.splitlines())
                         if RESULT_MARK in line), None)
            if line is not None or time.monotonic() >= deadline:
                break
        if line is None:
            raise BridgeError(
                f"the editor did not return a result for {key!r}. The payload "
                f"ran, but nothing was stored — check the editor log for a "
                f"traceback (logs: {logs[-400:] or 'empty'})")
        payload = line.split(RESULT_MARK, 1)[1].strip()
        try:
            result = json.loads(payload)
        except json.JSONDecodeError as e:
            raise BridgeError(f"result for {key!r} was not JSON: {e}") from e
        if result is None:
            raise BridgeError(
                f"the payload for {key!r} completed without storing a result, "
                f"which means it raised before finishing")
        return result

    def _raise_response_error(self, response: Any, context: str) -> None:
        """Raise a bridge-declared failure without replacing its explanation."""
        if not isinstance(response, dict):
            return
        status = str(response.get("status") or "").lower()
        if status not in {"error", "failed", "failure"}:
            return
        detail = response.get("error") or response.get("message")
        if not detail and isinstance(response.get("result"), dict):
            result = response["result"]
            detail = result.get("error") or result.get("message")
        if not detail:
            detail = json.dumps(response, sort_keys=True, default=str)
        raise BridgeError(f"{self.endpoint} {context} failed: {detail}")

    def _notify_abandoned_job(self, job: str) -> None:
        """Best-effort cleanup hint that cannot extend the caller deadline."""

        def notify() -> None:
            try:
                self.command(
                    "abandon_job",
                    {"job_id": job},
                    timeout=ABANDON_NOTIFY_TIMEOUT_S,
                )
            except BridgeError:
                pass

        threading.Thread(
            target=notify,
            name="scenebench-abandon-job",
            daemon=True,
        ).start()

    def _await_job(self, response: Any, deadline: float) -> None:
        """Wait out a payload the server reported as still running.

        Editor python is asynchronous and the editor's command queue is
        serial, so any fixed wait on the far side is a guess; when that guess
        expires the server answers ``{"status": "running", "job_id": ...}``
        rather than inventing a failure. Redeeming the handle here is what
        makes a caller's own ``timeout`` mean something — without it the far
        side's wait was the real ceiling, whatever this side asked for.

        Not finding a job is not an error: a payload that finished inside the
        server's wait has nothing to redeem.
        """
        if not isinstance(response, dict) or response.get("status") != "running":
            return
        job = response.get("job_id")
        if not job:
            return
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # The caller has made the explicit decision to stop waiting.
                # The hint is sent asynchronously: cleanup may continue for a
                # bounded time, but it cannot extend the timeout the caller
                # explicitly selected.
                self._notify_abandoned_job(str(job))
                raise BridgeError(
                    f"job {job} is still running after {self.endpoint} was "
                    f"given its full deadline; the editor has not finished, "
                    f"which is not the same as having failed — collect it "
                    f"with read_job_log({job!r}) if it matters")
            wait = min(remaining, 30.0)
            state = self.command("read_job_log", {"job_id": job, "wait_s": wait},
                                 timeout=wait + 30.0)
            self._raise_response_error(state, f"job {job!r}")
            if not isinstance(state, dict) or state.get("status") != "running":
                return          # done, or gone — either way, stop waiting

    def is_alive(self, timeout: float = 90.0) -> bool:
        """Convenience wrapper over :meth:`ping` for probe-style callers."""
        try:
            self.ping(timeout=timeout)
            return True
        except BridgeError:
            return False
