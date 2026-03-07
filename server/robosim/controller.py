"""
Motion primitive controller.

Maps high-level planning actions (PICK, CLEAR_BLOCKER, PLACE_BIN_A) to
sequences of low-level robot motions in MuJoCo/robosuite.

This is the layer between the LLM policy and the physics sim.
The LLM decides WHAT to do. This decides HOW to physically do it.

Without this layer, the LLM would have to output joint angles.
With this layer, "PICK" becomes: approach → descend → grasp → lift.
"""
import numpy as np
from typing import Optional, Callable


# ── Target positions (Panda on standard table) ────────────────────────
BIN_A_POS  = np.array([ 0.20, -0.25, 0.95])  # above bin A for placing
BIN_B_POS  = np.array([-0.20, -0.25, 0.95])
CLEAR_PUSH = np.array([ 0.30,  0.10, 0.85])  # push target for CLEAR_BLOCKER

# Number of simulation substeps per high-level action
STEPS_APPROACH = 50
STEPS_GRASP    = 30
STEPS_LIFT     = 30
STEPS_MOVE     = 80
STEPS_PLACE    = 40


class MotionController:
    """
    Executes motion primitives on a robosuite environment.

    Usage:
        ctrl = MotionController(robosuite_env)
        result = ctrl.execute("PICK", target_object="red_block")
    """

    def __init__(self, env):
        self._env = env  # robosuite environment

    def execute(self, action: str, target_object: Optional[str] = None,
                target_bin: Optional[str] = None) -> str:
        """
        Execute a high-level action.
        Returns: SUCCESS | FAILED_SLIP | FAILED_BLOCKED | FAILED_EMPTY | FAILED_INVALID
        """
        try:
            if action == "SCAN_SCENE":
                return self._scan()
            elif action.startswith("MOVE_TO_"):
                color = action[len("MOVE_TO_"):].lower()
                return self._move_to(color + "_block")
            elif action == "PICK":
                return self._pick(target_object)
            elif action == "PLACE_BIN_A":
                return self._place(BIN_A_POS)
            elif action == "PLACE_BIN_B":
                return self._place(BIN_B_POS)
            elif action == "CLEAR_BLOCKER":
                return self._clear_blocker()
            return "FAILED_INVALID"
        except Exception as e:
            print(f"[Controller] Exception during {action}: {e}")
            return "FAILED_INVALID"

    def _scan(self) -> str:
        """Move camera / head to scan the scene. No physical action needed."""
        self._env.step(self._null_action())
        return "SUCCESS"

    def _move_to(self, object_name: str) -> str:
        """Move gripper above the named object."""
        pos = self._get_object_pos(object_name)
        if pos is None:
            return "FAILED_INVALID"

        target = pos.copy()
        target[2] += 0.12  # hover above

        success = self._move_eef_to(target, n_steps=STEPS_APPROACH)
        return "SUCCESS" if success else "FAILED_BLOCKED"

    def _pick(self, target_object: Optional[str]) -> str:
        """Descend, close gripper, lift."""
        eef_pos = self._get_eef_pos()

        # Descend
        descend_target = eef_pos.copy()
        descend_target[2] -= 0.10
        self._move_eef_to(descend_target, n_steps=STEPS_GRASP)

        # Close gripper
        self._set_gripper(-1.0, n_steps=STEPS_GRASP)

        # Check if something was grasped
        if not self._is_grasping():
            self._set_gripper(1.0, n_steps=10)
            return "FAILED_EMPTY"

        # Lift
        lift_target = self._get_eef_pos().copy()
        lift_target[2] += 0.15
        self._move_eef_to(lift_target, n_steps=STEPS_LIFT)

        # Verify still holding
        if not self._is_grasping():
            return "FAILED_SLIP"

        return "SUCCESS"

    def _place(self, bin_pos: np.ndarray) -> str:
        """Move above bin, descend, open gripper."""
        if not self._is_grasping():
            return "FAILED_EMPTY"

        # Move above bin
        above_bin = bin_pos.copy()
        above_bin[2] += 0.05
        self._move_eef_to(above_bin, n_steps=STEPS_MOVE)

        # Descend slightly
        place_pos = bin_pos.copy()
        self._move_eef_to(place_pos, n_steps=STEPS_PLACE // 2)

        # Open gripper
        self._set_gripper(1.0, n_steps=STEPS_PLACE)
        return "SUCCESS"

    def _clear_blocker(self) -> str:
        """Push the nearest blocking object out of the way."""
        # Find the object closest to current eef that is near another target
        blocker_pos = self._find_nearest_reachable()
        if blocker_pos is None:
            return "FAILED_INVALID"

        # Move to blocker
        approach = blocker_pos.copy()
        approach[2] += 0.08
        self._move_eef_to(approach, n_steps=STEPS_APPROACH)

        # Descend and push sideways
        push_start = blocker_pos.copy()
        self._move_eef_to(push_start, n_steps=20)

        push_end = push_start.copy()
        push_end[0] += 0.18  # push in X direction
        push_end[1] += 0.10
        self._move_eef_to(push_end, n_steps=STEPS_MOVE)

        # Lift back
        lift = self._get_eef_pos().copy()
        lift[2] += 0.15
        self._move_eef_to(lift, n_steps=STEPS_LIFT)

        return "SUCCESS"

    # ── Low-level motion helpers ────────────────────────────────────────

    def _move_eef_to(self, target: np.ndarray, n_steps: int) -> bool:
        """Step the sim to move EEF toward target using position control."""
        for _ in range(n_steps):
            eef = self._get_eef_pos()
            delta = target - eef
            dist = np.linalg.norm(delta)
            if dist < 0.005:
                return True
            action = self._delta_action(delta / max(dist, 0.01) * 0.05)
            obs, _, done, _ = self._env.step(action)
            if done:
                return False
        return True

    def _set_gripper(self, value: float, n_steps: int):
        """Open (1.0) or close (-1.0) gripper."""
        for _ in range(n_steps):
            action = self._null_action()
            action[-1] = value  # last dim is gripper
            self._env.step(action)

    def _null_action(self) -> np.ndarray:
        return np.zeros(self._env.action_spec[0].shape)

    def _delta_action(self, delta_xyz: np.ndarray) -> np.ndarray:
        action = self._null_action()
        action[:3] = delta_xyz
        action[-1] = -1.0  # keep gripper closed if holding
        return action

    def _get_eef_pos(self) -> np.ndarray:
        obs = self._env._get_observations()
        return obs.get("robot0_eef_pos", np.array([0.0, 0.0, 1.0]))

    def _get_object_pos(self, name: str) -> Optional[np.ndarray]:
        obs = self._env._get_observations()
        key = f"{name}_pos"
        return obs.get(key, None)

    def _find_nearest_reachable(self) -> Optional[np.ndarray]:
        obs = self._env._get_observations()
        eef = self._get_eef_pos()
        best_dist = 999.0
        best_pos = None
        for key, val in obs.items():
            if key.endswith("_pos") and "robot" not in key:
                dist = np.linalg.norm(eef[:2] - val[:2])
                if dist < best_dist:
                    best_dist = dist
                    best_pos = val
        return best_pos

    def _is_grasping(self) -> bool:
        """Check if the gripper is currently holding something."""
        try:
            obs = self._env._get_observations()
            # robosuite exposes touch sensor or gripper qpos
            for key in obs:
                if "gripper_qpos" in key:
                    qpos = obs[key]
                    # gripper closed with something: qpos < 0.03 but > 0 (not fully closed)
                    return 0.002 < abs(qpos[0]) < 0.025
            return False
        except Exception:
            return False
