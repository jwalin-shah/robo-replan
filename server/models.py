"""
Action and observation schemas for the tabletop planning environment.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class Action(str, Enum):
    SCAN_SCENE = "SCAN_SCENE"
    MOVE_TO_RED = "MOVE_TO_RED"
    MOVE_TO_BLUE = "MOVE_TO_BLUE"
    MOVE_TO_GREEN = "MOVE_TO_GREEN"
    MOVE_TO_YELLOW = "MOVE_TO_YELLOW"
    MOVE_TO_PURPLE = "MOVE_TO_PURPLE"
    PICK = "PICK"
    PLACE_BIN_A = "PLACE_BIN_A"
    PLACE_BIN_B = "PLACE_BIN_B"
    CLEAR_BLOCKER = "CLEAR_BLOCKER"


class ObjectInfo(BaseModel):
    name: str
    reachable: bool
    location: Optional[str] = None  # "left", "center", "right", "unknown"
    blocking: Optional[str] = None  # name of what it is blocking


class Observation(BaseModel):
    # Task info
    instruction: str
    steps_remaining: int

    # Scene
    visible_objects: list[ObjectInfo]
    holding: Optional[str] = None  # name of currently held object

    # Planning context
    completed_subgoals: list[str]
    known_failures: list[str]
    active_constraints: list[str]

    # Last action outcome
    last_action: Optional[str] = None
    last_result: Optional[str] = None  # "SUCCESS", "FAILED_BLOCKED", "FAILED_EMPTY", "FAILED_INVALID"


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict
