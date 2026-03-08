from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel


class Action(str, Enum):
    SCAN_SCENE    = "SCAN_SCENE"
    MOVE_NORTH    = "MOVE_NORTH"
    MOVE_SOUTH    = "MOVE_SOUTH"
    MOVE_EAST     = "MOVE_EAST"
    MOVE_WEST     = "MOVE_WEST"
    ROTATE_LEFT   = "ROTATE_LEFT"
    ROTATE_RIGHT  = "ROTATE_RIGHT"
    WAIT          = "WAIT"
    TOGGLE_LIGHT  = "TOGGLE_LIGHT"
    MOVE_TO_RED   = "MOVE_TO_RED"
    MOVE_TO_BLUE  = "MOVE_TO_BLUE"
    MOVE_TO_GREEN = "MOVE_TO_GREEN"
    MOVE_TO_YELLOW = "MOVE_TO_YELLOW"
    MOVE_TO_PURPLE = "MOVE_TO_PURPLE"
    PICK          = "PICK"
    PLACE_BIN_A   = "PLACE_BIN_A"
    PLACE_BIN_B   = "PLACE_BIN_B"
    CLEAR_BLOCKER = "CLEAR_BLOCKER"


class ObjectInfo(BaseModel):
    name: str
    reachable: bool
    location: Optional[str] = None
    blocking: Optional[str] = None
    in_bin: Optional[str] = None
    is_held: bool = False


class Observation(BaseModel):
    # Task
    instruction: str
    steps_remaining: int

    # Scene
    visible_objects: list[ObjectInfo]
    holding: Optional[str] = None

    # Planning memory
    completed_subgoals: list[str] = []
    known_failures: list[str] = []
    active_constraints: list[str] = []
    action_history: list[str] = []     # last N actions taken

    # Last step
    last_action: Optional[str] = None
    last_result: Optional[str] = None

    # Rich signals (populated when ObsConfig flags are on)
    valid_actions: Optional[list[str]] = None   # actions that make sense right now
    goal_progress: Optional[float] = None       # 0.0–1.0
    goals_remaining: Optional[int] = None
    oracle_hint: Optional[str] = None           # what scripted policy would do
    nav_mode: bool = False
    gripper_cell: Optional[str] = None
    gripper_facing: Optional[str] = None
    discovered_traits: Optional[dict[str, str]] = None
    object_deadlines: Optional[dict[str, int]] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = {}
