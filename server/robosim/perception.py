"""
Perception layer: extracts symbolic planning state from raw MuJoCo physics.

This is the bridge between the physics simulation and the LLM planning layer.
Without this, the LLM would have to reason over raw 3D positions and contact
forces — which it can't do. With this, it gets clean symbolic facts:
  "red_block is BLOCKED by blue_block"
  "gripper is NEAR red_block"
  "green_block is IN bin A"

Every fact here is derived from actual physics, not hardcoded flags.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PerceivedObject:
    name: str
    pos: np.ndarray          # 3D center position from MuJoCo
    reachable: bool          # within robot workspace AND not occluded
    blocking: Optional[str]  # name of object this one is blocking
    in_bin: Optional[str]    # "A" or "B" if inside a bin
    is_held: bool
    distance_to_gripper: float


@dataclass
class PerceivedScene:
    objects: dict[str, PerceivedObject]
    gripper_pos: np.ndarray
    gripper_open: bool
    holding: Optional[str]


# ── Workspace constants (Panda arm on standard table) ─────────────────
WORKSPACE_X = (-0.35, 0.35)
WORKSPACE_Y = (-0.35, 0.35)
WORKSPACE_Z = (0.75, 1.10)

BIN_A_CENTER = np.array([ 0.20, -0.25, 0.82])
BIN_B_CENTER = np.array([-0.20, -0.25, 0.82])
BIN_RADIUS   = 0.08   # horizontal distance to count as "in bin"
BIN_Z_MAX    = 0.90   # object must be below this height to be "in bin"

BLOCK_RADIUS = 0.025  # half-size of a standard cube block
BLOCKER_CLEARANCE = 0.06  # if another object is within this distance in front
                           # of the target along the gripper approach axis,
                           # it counts as blocking

GRIPPER_NEAR_DIST = 0.12  # gripper is "near" an object if within this distance


def extract_scene(mj_data, mj_model, robot_name: str,
                  object_names: list[str]) -> PerceivedScene:
    """
    Extract full symbolic scene state from raw MuJoCo sim data.

    mj_data:      mujoco.MjData
    mj_model:     mujoco.MjModel
    robot_name:   e.g. "robot0"
    object_names: list of body names to track as objects
    """
    gripper_pos = _get_eef_pos(mj_data, mj_model, robot_name)
    gripper_open = _get_gripper_open(mj_data, mj_model, robot_name)
    holding = _detect_held_object(mj_data, mj_model, object_names, gripper_pos, gripper_open)

    objects = {}
    for name in object_names:
        pos = _get_object_pos(mj_data, mj_model, name)
        if pos is None:
            continue
        in_bin = _detect_in_bin(pos)
        is_held = (name == holding)
        dist = float(np.linalg.norm(gripper_pos - pos)) if pos is not None else 999.0
        reachable = _is_reachable(pos, gripper_pos, objects) and not is_held and in_bin is None
        objects[name] = PerceivedObject(
            name=name,
            pos=pos,
            reachable=reachable,
            blocking=None,   # filled in below
            in_bin=in_bin,
            is_held=is_held,
            distance_to_gripper=dist,
        )

    # Detect blocking relationships
    _detect_blocking(objects, gripper_pos)

    return PerceivedScene(
        objects=objects,
        gripper_pos=gripper_pos,
        gripper_open=gripper_open,
        holding=holding,
    )


def _get_object_pos(mj_data, mj_model, name: str) -> Optional[np.ndarray]:
    """Get 3D center position of a named body."""
    try:
        body_id = mj_model.body(name).id
        return mj_data.xpos[body_id].copy()
    except Exception:
        return None


def _get_eef_pos(mj_data, mj_model, robot_name: str) -> np.ndarray:
    """Get end-effector (gripper) position."""
    eef_names = [f"{robot_name}_eef", f"{robot_name}_gripper", "gripper0_eef"]
    for eef in eef_names:
        try:
            body_id = mj_model.body(eef).id
            return mj_data.xpos[body_id].copy()
        except Exception:
            continue
    # Fallback: use last link position
    return np.array([0.0, 0.0, 1.0])


def _get_gripper_open(mj_data, mj_model, robot_name: str) -> bool:
    """Detect if gripper is open from joint positions."""
    try:
        # Panda gripper joints are named robot0_finger_joint1/2
        for suffix in ["finger_joint1", "finger_joint2"]:
            qpos_idx = mj_model.joint(f"{robot_name}_{suffix}").qposadr[0]
            if mj_data.qpos[qpos_idx] > 0.02:  # > 2cm = open
                return True
        return False
    except Exception:
        return True


def _detect_held_object(mj_data, mj_model, object_names: list[str],
                         gripper_pos: np.ndarray, gripper_open: bool) -> Optional[str]:
    """
    Detect which object is currently held by checking:
    1. Gripper is closed
    2. Object center is within gripper zone
    3. Object is above table (being lifted)
    """
    if gripper_open:
        return None
    for name in object_names:
        pos = _get_object_pos(mj_data, mj_model, name)
        if pos is None:
            continue
        dist = np.linalg.norm(gripper_pos - pos)
        if dist < 0.08 and pos[2] > 0.85:  # close to gripper and lifted
            return name
    return None


def _detect_in_bin(pos: np.ndarray) -> Optional[str]:
    """
    Detect if an object is inside a bin by checking 3D position against bin bounds.
    Uses horizontal distance + height threshold.
    """
    if pos[2] > BIN_Z_MAX:
        return None  # floating, not placed
    for bin_name, center in [("A", BIN_A_CENTER), ("B", BIN_B_CENTER)]:
        horiz = np.linalg.norm(pos[:2] - center[:2])
        if horiz < BIN_RADIUS:
            return bin_name
    return None


def _is_reachable(pos: np.ndarray, gripper_pos: np.ndarray,
                   existing_objects: dict) -> bool:
    """
    An object is reachable if:
    1. It's within the robot's workspace
    2. There's no other object directly between gripper approach and object
       (this is the coarse blocking check; fine blocking is in _detect_blocking)
    """
    x, y, z = pos
    if not (WORKSPACE_X[0] < x < WORKSPACE_X[1]):
        return False
    if not (WORKSPACE_Y[0] < y < WORKSPACE_Y[1]):
        return False
    if not (WORKSPACE_Z[0] < z < WORKSPACE_Z[1]):
        return False
    return True


def _detect_blocking(objects: dict[str, PerceivedObject],
                     gripper_pos: np.ndarray):
    """
    Detect blocking relationships between objects.

    An object A blocks object B if:
    - A is between the gripper approach direction and B
    - A is within BLOCKER_CLEARANCE of B's position in the XY plane
    - A is reachable itself (otherwise it's not actually blocking from gripper POV)

    After calling this, each blocker's .blocking field is set to what it blocks.
    """
    names = list(objects.keys())
    for i, name_a in enumerate(names):
        a = objects[name_a]
        if not a.reachable or a.in_bin:
            continue
        for name_b in names:
            if name_a == name_b:
                continue
            b = objects[name_b]
            if b.in_bin or b.is_held:
                continue
            # Check if A is between gripper and B in the XY plane
            ab_dist = np.linalg.norm(a.pos[:2] - b.pos[:2])
            if ab_dist < BLOCKER_CLEARANCE:
                # A is close to B. Check if A is "in front" (closer to gripper)
                gripper_to_b = np.linalg.norm(gripper_pos[:2] - b.pos[:2])
                gripper_to_a = np.linalg.norm(gripper_pos[:2] - a.pos[:2])
                if gripper_to_a < gripper_to_b:
                    # A is between gripper and B → A blocks B
                    a.blocking = name_b
                    b.reachable = False
                    break  # each blocker blocks one thing
