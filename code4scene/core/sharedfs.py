"""Directories that two users must agree on.

The scorer and the editor meet on shared volumes (``/measure``,
``/artifacts``): the scorer names a path and the editor, which may run as a
different user outside the scorer's group, writes the file. A directory made
under the default umask arrives at 775 and would refuse that write, so the
directories created here are opened to 0o777.

In ``core`` so that any layer that creates such directories can use it.
"""

from __future__ import annotations

from pathlib import Path


def mkdir_shared(path: Path) -> Path:
    """``mkdir -p``, then open every directory THIS CALL created to 0o777.

    Only the components created here are widened: a mount root or an existing
    parent keeps whatever mode it already has — widening somebody else's
    directory is not this function's call to make.

    The chmod comes after the mkdir because ``mkdir(mode=...)`` is filtered by
    the umask, which is what produces the 775 directories this exists to
    avoid.
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
