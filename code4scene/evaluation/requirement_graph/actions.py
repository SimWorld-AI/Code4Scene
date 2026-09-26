"""Bounded free-camera actions for static-scene exploration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, ClassVar, TypeAlias
from collections.abc import Mapping

from .contracts import CameraPose, JsonSerializable, SceneBounds, to_jsonable


GRID_SIZE = 5
GRID_CELL_COUNT = GRID_SIZE * GRID_SIZE
YAW_BIN_COUNT = 12
YAW_BIN_DEGREES = 30.0
BOUNDS_EXPANSION_FRACTION = 0.1
SCOUT_INSET_FRACTION = 0.25
MAX_SCOUT_INSET_CM = 500.0
MIN_PITCH_DEGREES = -85.0
MAX_PITCH_DEGREES = 30.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class CameraActionType(str, Enum):
    INSPECT_CELL = "inspect_cell"
    MOVE_RELATIVE = "move_relative"
    ROTATE_CAMERA = "rotate_camera"
    FINISH_EXPLORATION = "finish_exploration"


class HeightMode(str, Enum):
    EYE = "eye"
    MID = "mid"
    AERIAL = "aerial"


class PitchMode(str, Enum):
    UP = "up"
    LEVEL = "level"
    DOWN = "down"
    STEEP_DOWN = "steep_down"


PITCH_DEGREES = {
    PitchMode.UP: 15.0,
    PitchMode.LEVEL: -5.0,
    PitchMode.DOWN: -30.0,
    PitchMode.STEEP_DOWN: -65.0,
}


class MoveDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class MoveScale(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


MOVE_SCALE_MULTIPLIERS = {
    MoveScale.SMALL: 0.5,
    MoveScale.MEDIUM: 1.0,
    MoveScale.LARGE: 2.0,
}

ALLOWED_YAW_DELTAS = frozenset({-60.0, -30.0, 0.0, 30.0, 60.0})
ALLOWED_PITCH_DELTAS = frozenset({-30.0, -15.0, 0.0, 15.0, 30.0})


class CameraAction(JsonSerializable):
    """Base class for actions emitted by the exploration model."""

    ACTION_TYPE: ClassVar[CameraActionType]

    @property
    def action(self) -> str:
        return self.ACTION_TYPE.value

    def to_dict(self) -> dict[str, Any]:
        payload = {item.name: to_jsonable(getattr(self, item.name)) for item in fields(self)}
        return {"action": self.action, **payload}


@dataclass(frozen=True)
class InspectCellAction(CameraAction):
    ACTION_TYPE: ClassVar[CameraActionType] = CameraActionType.INSPECT_CELL

    cell_x: int
    cell_y: int
    height: HeightMode | str
    yaw_bin: int
    pitch_mode: PitchMode | str

    def __post_init__(self) -> None:
        if isinstance(self.cell_x, bool) or not isinstance(self.cell_x, int):
            raise ValueError("cell_x must be an integer")
        if isinstance(self.cell_y, bool) or not isinstance(self.cell_y, int):
            raise ValueError("cell_y must be an integer")
        if not 0 <= self.cell_x < GRID_SIZE or not 0 <= self.cell_y < GRID_SIZE:
            raise ValueError(f"cell coordinates must be in [0, {GRID_SIZE - 1}]")
        if isinstance(self.yaw_bin, bool) or not isinstance(self.yaw_bin, int):
            raise ValueError("yaw_bin must be an integer")
        if not 0 <= self.yaw_bin < YAW_BIN_COUNT:
            raise ValueError(f"yaw_bin must be in [0, {YAW_BIN_COUNT - 1}]")
        object.__setattr__(self, "height", HeightMode(self.height))
        object.__setattr__(self, "pitch_mode", PitchMode(self.pitch_mode))


@dataclass(frozen=True)
class MoveRelativeAction(CameraAction):
    ACTION_TYPE: ClassVar[CameraActionType] = CameraActionType.MOVE_RELATIVE

    direction: MoveDirection | str
    scale: MoveScale | str = MoveScale.MEDIUM

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", MoveDirection(self.direction))
        object.__setattr__(self, "scale", MoveScale(self.scale))


@dataclass(frozen=True)
class RotateCameraAction(CameraAction):
    ACTION_TYPE: ClassVar[CameraActionType] = CameraActionType.ROTATE_CAMERA

    yaw_delta: float = 0.0
    pitch_delta: float = 0.0

    def __post_init__(self) -> None:
        yaw_delta = float(self.yaw_delta)
        pitch_delta = float(self.pitch_delta)
        if yaw_delta not in ALLOWED_YAW_DELTAS:
            raise ValueError(f"yaw_delta must be one of {sorted(ALLOWED_YAW_DELTAS)}")
        if pitch_delta not in ALLOWED_PITCH_DELTAS:
            raise ValueError(f"pitch_delta must be one of {sorted(ALLOWED_PITCH_DELTAS)}")
        object.__setattr__(self, "yaw_delta", yaw_delta)
        object.__setattr__(self, "pitch_delta", pitch_delta)


@dataclass(frozen=True)
class FinishExplorationAction(CameraAction):
    ACTION_TYPE: ClassVar[CameraActionType] = CameraActionType.FINISH_EXPLORATION

    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", str(self.reason).strip())


# Concise alias for controllers which handle termination separately.
FinishAction = FinishExplorationAction
CameraActionValue: TypeAlias = (
    InspectCellAction | MoveRelativeAction | RotateCameraAction | FinishExplorationAction
)


def height_for_mode(
    bounds: SceneBounds,
    mode: HeightMode | str,
    *,
    floor_z: float | None = None,
) -> float:
    """Compute the plan-defined absolute camera height for a scene."""

    mode = HeightMode(mode)
    base_z = bounds.min_z if floor_z is None else float(floor_z)
    if not math.isfinite(base_z) or not bounds.min_z <= base_z <= bounds.max_z:
        raise ValueError("floor_z must be finite and inside scene bounds")
    if mode is HeightMode.EYE:
        return base_z + 220.0
    if mode is HeightMode.MID:
        usable_height = max(1.0, bounds.max_z - base_z)
        return base_z + _clamp(usable_height * 0.5, 500.0, 3000.0)
    aerial_offset = _clamp(bounds.horizontal_diagonal * 0.25, 800.0, 6000.0)
    return bounds.max_z + aerial_offset


def movement_step_cm(bounds: SceneBounds) -> float:
    """Return the scene-relative base translation step in centimetres."""

    return _clamp(max(bounds.width, bounds.depth) * 0.05, 250.0, 1000.0)


def clamp_camera_pose(
    pose: CameraPose,
    bounds: SceneBounds,
    *,
    expansion_fraction: float = BOUNDS_EXPANSION_FRACTION,
) -> CameraPose:
    """Clamp a pose to the camera envelope and valid pitch range.

    Horizontal limits are the requested 10% expansion.  The upper vertical
    limit must include the plan-defined aerial height; clamping it to 10% above
    the content box would collapse ``mid`` and ``aerial`` into the same view for
    most scenes.
    """

    limits = bounds.expanded(expansion_fraction)
    return CameraPose(
        x=_clamp(pose.x, limits.min_x, limits.max_x),
        y=_clamp(pose.y, limits.min_y, limits.max_y),
        z=_clamp(pose.z, limits.min_z, height_for_mode(bounds, HeightMode.AERIAL)),
        pitch=_clamp(pose.pitch, MIN_PITCH_DEGREES, MAX_PITCH_DEGREES),
        yaw=pose.yaw % 360.0,
        roll=pose.roll,
    )


def camera_pose_for_cell(
    action: InspectCellAction,
    bounds: SceneBounds,
    *,
    floor_z: float | None = None,
) -> CameraPose:
    cell_width = bounds.width / GRID_SIZE
    cell_depth = bounds.depth / GRID_SIZE
    pose = CameraPose(
        x=bounds.min_x + (action.cell_x + 0.5) * cell_width,
        y=bounds.min_y + (action.cell_y + 0.5) * cell_depth,
        z=height_for_mode(bounds, action.height, floor_z=floor_z),
        pitch=PITCH_DEGREES[PitchMode(action.pitch_mode)],
        yaw=action.yaw_bin * YAW_BIN_DEGREES,
    )
    return clamp_camera_pose(pose, bounds)


def move_relative_pose(
    action: MoveRelativeAction,
    current_pose: CameraPose,
    bounds: SceneBounds,
) -> CameraPose:
    distance = movement_step_cm(bounds) * MOVE_SCALE_MULTIPLIERS[MoveScale(action.scale)]
    yaw_radians = math.radians(current_pose.yaw)
    forward_x, forward_y = math.cos(yaw_radians), math.sin(yaw_radians)
    right_x, right_y = -forward_y, forward_x
    dx = dy = dz = 0.0
    direction = MoveDirection(action.direction)
    if direction is MoveDirection.FORWARD:
        dx, dy = forward_x * distance, forward_y * distance
    elif direction is MoveDirection.BACKWARD:
        dx, dy = -forward_x * distance, -forward_y * distance
    elif direction is MoveDirection.RIGHT:
        dx, dy = right_x * distance, right_y * distance
    elif direction is MoveDirection.LEFT:
        dx, dy = -right_x * distance, -right_y * distance
    elif direction is MoveDirection.UP:
        dz = distance
    else:
        dz = -distance
    return clamp_camera_pose(
        CameraPose(
            x=current_pose.x + dx,
            y=current_pose.y + dy,
            z=current_pose.z + dz,
            pitch=current_pose.pitch,
            yaw=current_pose.yaw,
            roll=current_pose.roll,
        ),
        bounds,
    )


def rotate_camera_pose(action: RotateCameraAction, current_pose: CameraPose, bounds: SceneBounds) -> CameraPose:
    return clamp_camera_pose(
        CameraPose(
            x=current_pose.x,
            y=current_pose.y,
            z=current_pose.z,
            pitch=current_pose.pitch + action.pitch_delta,
            yaw=current_pose.yaw + action.yaw_delta,
            roll=current_pose.roll,
        ),
        bounds,
    )


def apply_camera_action(
    action: CameraActionValue | Mapping[str, Any],
    current_pose: CameraPose,
    bounds: SceneBounds,
) -> CameraPose:
    """Resolve one model action into the next bounded camera pose.

    A finish action deliberately returns the current pose; the controller owns
    phase-gating and termination policy.
    """

    if isinstance(action, Mapping):
        action = camera_action_from_dict(action)
    if isinstance(action, InspectCellAction):
        return camera_pose_for_cell(action, bounds)
    if isinstance(action, MoveRelativeAction):
        return move_relative_pose(action, current_pose, bounds)
    if isinstance(action, RotateCameraAction):
        return rotate_camera_pose(action, current_pose, bounds)
    if isinstance(action, FinishExplorationAction):
        return current_pose
    raise TypeError(f"unsupported camera action: {type(action).__name__}")


def _decode_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        decoded = json.loads(arguments)
        if not isinstance(decoded, dict):
            raise ValueError("tool-call arguments JSON must contain an object")
        return decoded
    if isinstance(arguments, Mapping):
        return dict(arguments)
    raise ValueError("tool-call arguments must be an object or JSON object string")


def camera_action_from_dict(payload: Mapping[str, Any]) -> CameraActionValue:
    """Parse either a flat action object or a standard model tool call.

    Accepted examples are ``{"action": "rotate_camera", ...}``,
    ``{"name": "rotate_camera", "arguments": {...}}``, and OpenAI's nested
    ``{"function": {"name": ..., "arguments": ...}}`` shape.
    """

    value: Mapping[str, Any] = payload
    if isinstance(payload.get("function"), Mapping):
        value = payload["function"]  # type: ignore[assignment]

    raw_name = value.get("action", value.get("name", value.get("type")))
    if raw_name is None:
        raise ValueError("camera action must include action or name")
    name = str(raw_name).strip().lower()
    aliases = {
        "finish": CameraActionType.FINISH_EXPLORATION.value,
        "stop": CameraActionType.FINISH_EXPLORATION.value,
    }
    name = aliases.get(name, name)

    if "arguments" in value:
        arguments = _decode_arguments(value.get("arguments"))
    else:
        arguments = {
            key: item
            for key, item in value.items()
            if key not in {"action", "name", "type", "function"}
        }

    constructors = {
        CameraActionType.INSPECT_CELL.value: InspectCellAction,
        CameraActionType.MOVE_RELATIVE.value: MoveRelativeAction,
        CameraActionType.ROTATE_CAMERA.value: RotateCameraAction,
        CameraActionType.FINISH_EXPLORATION.value: FinishExplorationAction,
    }
    try:
        constructor = constructors[name]
    except KeyError as error:
        raise ValueError(f"unknown camera action: {name}") from error
    try:
        return constructor(**arguments)
    except TypeError as error:
        raise ValueError(f"invalid arguments for {name}: {error}") from error


def _look_at_yaw(x: float, y: float, target_x: float, target_y: float) -> float:
    return math.degrees(math.atan2(target_y - y, target_x - x)) % 360.0


def generate_scout_poses(
    bounds: SceneBounds,
    *,
    floor_z: float | None = None,
) -> tuple[CameraPose, ...]:
    """Generate four inward-looking interior views and two aerial views.

    A content AABB describes occupied geometry, not navigable free space.  Its
    horizontal faces commonly coincide with exterior walls.  Placing a scout
    exactly on those faces therefore produces a well-exposed but useless
    close-up of a wall.  Step the eye-level scouts a bounded distance into the
    box while retaining the same four complementary viewing directions.
    """

    center_x, center_y, _ = bounds.center_cm
    eye_z = height_for_mode(bounds, HeightMode.EYE, floor_z=floor_z)
    aerial_z = height_for_mode(bounds, HeightMode.AERIAL)
    inset_x = min(bounds.width * SCOUT_INSET_FRACTION, MAX_SCOUT_INSET_CM)
    inset_y = min(bounds.depth * SCOUT_INSET_FRACTION, MAX_SCOUT_INSET_CM)
    raw_poses = (
        CameraPose(
            bounds.min_x + inset_x,
            center_y,
            eye_z,
            PITCH_DEGREES[PitchMode.LEVEL],
            0.0,
        ),
        CameraPose(
            bounds.max_x - inset_x,
            center_y,
            eye_z,
            PITCH_DEGREES[PitchMode.LEVEL],
            180.0,
        ),
        CameraPose(
            center_x,
            bounds.min_y + inset_y,
            eye_z,
            PITCH_DEGREES[PitchMode.LEVEL],
            90.0,
        ),
        CameraPose(
            center_x,
            bounds.max_y - inset_y,
            eye_z,
            PITCH_DEGREES[PitchMode.LEVEL],
            270.0,
        ),
        CameraPose(
            bounds.min_x,
            bounds.min_y,
            aerial_z,
            PITCH_DEGREES[PitchMode.DOWN],
            _look_at_yaw(bounds.min_x, bounds.min_y, center_x, center_y),
        ),
        CameraPose(
            bounds.max_x,
            bounds.max_y,
            aerial_z,
            PITCH_DEGREES[PitchMode.DOWN],
            _look_at_yaw(bounds.max_x, bounds.max_y, center_x, center_y),
        ),
    )
    return tuple(clamp_camera_pose(pose, bounds) for pose in raw_poses)


__all__ = [
    "ALLOWED_PITCH_DELTAS",
    "ALLOWED_YAW_DELTAS",
    "BOUNDS_EXPANSION_FRACTION",
    "MAX_SCOUT_INSET_CM",
    "CameraAction",
    "CameraActionType",
    "CameraActionValue",
    "FinishAction",
    "FinishExplorationAction",
    "GRID_SIZE",
    "HeightMode",
    "InspectCellAction",
    "MoveDirection",
    "MoveRelativeAction",
    "MoveScale",
    "PitchMode",
    "RotateCameraAction",
    "SCOUT_INSET_FRACTION",
    "apply_camera_action",
    "camera_action_from_dict",
    "camera_pose_for_cell",
    "clamp_camera_pose",
    "generate_scout_poses",
    "height_for_mode",
    "movement_step_cm",
]
