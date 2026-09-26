"""Locate files shipped inside the installed ``code4scene`` package.

Scoring policies, judge policies and rubrics are package data. They are found
through :mod:`importlib.resources`, never through a repository-relative path,
so a regular (non-editable) install scores exactly like a source checkout.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def package_file(*parts: str) -> Path:
    """Return a filesystem path to a file bundled with the package."""

    traversable = resources.files("code4scene").joinpath(*parts)
    path = Path(str(traversable))
    if not path.exists():
        raise FileNotFoundError(
            "packaged resource is missing from this installation: " + "/".join(parts)
        )
    return path


def config_file(*parts: str) -> Path:
    """A file under ``code4scene/configs``."""

    return package_file("configs", *parts)


def ue_script(name: str) -> Path:
    """An editor-side script under ``code4scene/ue_scripts`` (sent, not imported)."""

    return package_file("ue_scripts", name)


__all__ = ["config_file", "package_file", "ue_script"]
