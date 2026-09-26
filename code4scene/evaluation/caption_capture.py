"""Compatibility imports for the former caption-only planner module.

New code must import :mod:`scene_graph_capture`; the algorithms now serve
three independently captured protocols with distinct policy IDs.
"""

from . import scene_graph_capture as _implementation
from .scene_graph_capture import *  # noqa: F403

__all__ = _implementation.__all__
# Historical callers accessed this implementation constant as a module
# attribute even though it was never part of ``__all__``.
CROP_CM = _implementation.CROP_CM
