"""Directories that two uids must agree on.

The harness, the editor and the agent meet on shared volumes (``/measure``,
``/artifacts``): the harness names a path, the editor — a different uid in the
pod topology, 10001 against the operator's own — writes the file. A directory
the harness ``mkdir``s arrives at 775 under the default umask, and the editor
is not in the harness's group, so every hand-off dies with a bare
``PermissionError`` on the editor's side of the mount. Operators used to
``chmod 777`` such directories by hand; this is the code doing it to the
directories it creates itself, so the hand-off stops being a trap.

In ``core`` because both ``evaluation`` (renders, editor exports) and
``infra`` (the episode) create such directories, and neither may import the
other.
"""

from __future__ import annotations

from pathlib import Path


def mkdir_shared(path: Path) -> Path:
    """``mkdir -p``, then open every directory THIS CALL created to 0o777.

    Only the components created here are widened: a mount root or an existing
    parent keeps whatever the operator set on it — widening somebody else's
    directory is not this function's call to make.

    The chmod comes after the mkdir because ``mkdir(mode=...)`` is filtered by
    the umask, which is exactly the mechanism that produced the 775 traps this
    exists to close.
    """
    path = Path(path)
    missing: list[Path] = []
    probe = path
    while not probe.exists():
        missing.append(probe)
        if probe.parent == probe:
            break
        probe = probe.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        try:
            created.chmod(0o777)
        except OSError:
            # Racing the other side: if the editor created it first, the
            # directory is already writable by the uid this widening is for.
            pass
    return path


__all__ = ["mkdir_shared"]
