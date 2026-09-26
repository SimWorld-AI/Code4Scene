"""Code4Scene-owned frame-provider boundary for merged Stage 2/3 logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from collections.abc import Sequence

from .actor_inventory import ActorInventorySnapshot
from .contracts import CameraPose, SceneBounds


class RgbFrameHealthError(ValueError):
    """A captured RGB frame failed deterministic provider-side validation."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = str(reason)


@dataclass(slots=True)
class CapturedFrame:
    """One RGB observation with the exact requested world-space pose."""

    frame_id: str
    pose: CameraPose
    rgb: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    phase: str = "capture"

    def as_rgb_array(self) -> Any:
        return self.rgb


@runtime_checkable
class FrameProvider(Protocol):
    """The only live-scene interface the merged visual stages consume."""

    def __enter__(self) -> FrameProvider: ...

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...

    def capture(
        self,
        poses: Sequence[CameraPose],
        *,
        phase: str = "capture",
        frame_id_prefix: str = "frame",
    ) -> list[CapturedFrame]: ...

    def get_scene_bounds(
        self, explicit_bounds: SceneBounds | None = None
    ) -> SceneBounds: ...

    def get_actor_inventory(
        self,
        *,
        closed_categories: Sequence[str] = (),
        canonical_identity_closed_categories: Sequence[str] = (),
        scene_bounds: SceneBounds | None = None,
        assume_loaded_world_complete: bool = False,
    ) -> ActorInventorySnapshot: ...

    def close(self) -> None: ...


__all__ = ["CapturedFrame", "FrameProvider", "RgbFrameHealthError"]
