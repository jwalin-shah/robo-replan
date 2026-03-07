"""
SimWrapper — physics backend for RoboReplan.

Two modes:
  use_stub=False  Real MuJoCo + robosuite PickPlace environment.
                  Object positions, blocking, and grasp success come from
                  actual physics. This is what makes training meaningful.

  use_stub=True   Lightweight Python sim for fast local testing.
                  Same interface, no physics dependency.

The wrapper always exposes the same symbolic SimState so the planning
layer above never needs to know which backend is running.
"""
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .randomizer import randomize_scenario, ScenarioConfig


# ── Symbolic state types ───────────────────────────────────────────────

@dataclass
class ObjectState:
    name: str
    pos: np.ndarray
    reachable: bool = True
    blocking: Optional[str] = None
    in_bin: Optional[str] = None
    is_held: bool = False


@dataclass
class SimState:
    objects: dict = field(default_factory=dict)
    gripper_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper_open: bool = True
    holding: Optional[str] = None


# ── Main wrapper ───────────────────────────────────────────────────────

class SimWrapper:
    """
    Wraps either robosuite (real) or a Python stub (fast testing).
    Always produces symbolic SimState for the planning layer.
    """

    def __init__(self, use_stub: bool = True):
        self.use_stub = use_stub
        self._env = None
        self._controller = None
        self._state = SimState()
        self._last_moved_to: Optional[str] = None
        self._current_cfg: Optional[ScenarioConfig] = None

        if not use_stub:
            self._init_robosuite()

    # ── Init ───────────────────────────────────────────────────────────

    def _init_robosuite(self):
        """
        Initialize a real robosuite PickPlace environment.

        We use PickPlace because it already has:
        - Panda arm (most standard research robot)
        - Multiple objects with real physics
        - Multiple bin targets
        - OSC_POSE controller (operational space — moves in Cartesian coords,
          which our high-level controller can use directly)
        """
        try:
            import robosuite as suite
            from robosuite.controllers import load_composite_controller_config

            controller_config = load_composite_controller_config(controller="BASIC")

            self._env = suite.make(
                env_name="PickPlace",
                robots="Panda",
                controller_configs=controller_config,
                has_renderer=False,           # no display on server
                has_offscreen_renderer=True,  # needed for camera obs
                use_camera_obs=True,
                camera_names=["frontview", "agentview"],
                camera_heights=128,
                camera_widths=128,
                reward_shaping=False,         # we compute our own reward
                control_freq=20,
                single_object_mode=0,         # all objects
                object_type=None,             # random objects
            )

            from .controller import MotionController
            self._controller = MotionController(self._env)
            print("[SimWrapper] robosuite PickPlace loaded (Panda arm)")

        except ImportError:
            print("[SimWrapper] robosuite not installed → falling back to stub")
            self.use_stub = True
        except Exception as e:
            print(f"[SimWrapper] robosuite init failed: {e} → falling back to stub")
            self.use_stub = True

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self, scenario: str = "random") -> tuple[SimState, ScenarioConfig]:
        """Reset scene. Returns (SimState, ScenarioConfig)."""
        force_blocked = random.random() < 0.6
        cfg = randomize_scenario(force_blocked=force_blocked)

        if not self.use_stub and self._env is not None:
            self._reset_robosuite(cfg)
        else:
            self._build_state_from_config(cfg)

        return self._state, cfg

    def _reset_robosuite(self, cfg: ScenarioConfig):
        """Reset robosuite and sync symbolic state from physics."""
        obs = self._env.reset()
        self._sync_state_from_obs(obs, cfg)

    def _sync_state_from_obs(self, obs: dict, cfg: ScenarioConfig):
        """
        Extract symbolic state from robosuite observation dict.
        Uses the perception layer to detect blocking from real 3D positions.
        """
        try:
            from .perception import extract_scene
            scene = extract_scene(
                mj_data=self._env.sim.data,
                mj_model=self._env.sim.model,
                robot_name="robot0",
                object_names=list(cfg.objects),
            )
            objects = {}
            for name, p in scene.objects.items():
                objects[name] = ObjectState(
                    name=name,
                    pos=p.pos,
                    reachable=p.reachable,
                    blocking=p.blocking,
                    in_bin=p.in_bin,
                    is_held=p.is_held,
                )
            self._state = SimState(
                objects=objects,
                gripper_pos=scene.gripper_pos,
                gripper_open=scene.gripper_open,
                holding=scene.holding,
            )
        except Exception as e:
            print(f"[SimWrapper] perception sync failed: {e}, using stub fallback")
            self._build_state_from_config(cfg)

        self._current_cfg = cfg

    # ── Stub state builder ─────────────────────────────────────────────

    def get_last_moved_to(self) -> Optional[str]:
        return self._last_moved_to

    def _build_state_from_config(self, cfg: ScenarioConfig):
        """Build stub SimState from randomized scenario config."""
        self._last_moved_to = None
        objects = {}
        for obj_name in cfg.objects:
            x, y = cfg.positions.get(obj_name, (0.0, 0.0))
            is_blocked = obj_name in cfg.blockers.values()
            objects[obj_name] = ObjectState(
                name=obj_name,
                pos=np.array([x, y, 0.82]),
                reachable=not is_blocked,
                blocking=cfg.blockers.get(obj_name),
            )
        self._state = SimState(
            objects=objects,
            gripper_pos=np.array([0.0, 0.25, 1.0]),
            gripper_open=True,
            holding=None,
        )
        self._current_cfg = cfg

    # ── Execute action ─────────────────────────────────────────────────

    def execute(self, action: str) -> str:
        """Execute a high-level action. Returns result string."""
        if not self.use_stub and self._env is not None and self._controller is not None:
            return self._execute_real(action)
        return self._execute_stub(action)

    def _execute_real(self, action: str) -> str:
        """Execute via real robosuite physics + motion controller."""
        result = self._controller.execute(action)
        # Re-sync symbolic state from physics
        obs = self._env._get_observations()
        if self._current_cfg:
            self._sync_state_from_obs(obs, self._current_cfg)
        return result

    def _execute_stub(self, action: str) -> str:
        """Execute in the lightweight Python stub."""
        s = self._state

        if action == "SCAN_SCENE":
            return "SUCCESS"

        elif action.startswith("MOVE_TO_"):
            color = action[len("MOVE_TO_"):].lower()
            name = color + "_block"
            if name not in s.objects:
                return "FAILED_INVALID"
            obj = s.objects[name]
            if not obj.reachable:
                return "FAILED_BLOCKED"
            s.gripper_pos = obj.pos.copy() + np.array([0, 0, 0.05])
            self._last_moved_to = name
            return "SUCCESS"

        elif action == "PICK":
            if s.holding is not None:
                return "FAILED_INVALID"
            candidates = []
            for obj in s.objects.values():
                if obj.reachable and not obj.is_held and obj.in_bin is None:
                    dist = np.linalg.norm(s.gripper_pos[:2] - obj.pos[:2])
                    candidates.append((dist, obj))
            # Prefer the object we last moved to
            candidates.sort(key=lambda x: (
                0 if x[1].name == self._last_moved_to else 1, x[0]
            ))
            for _, obj in candidates:
                dist = np.linalg.norm(s.gripper_pos[:2] - obj.pos[:2])
                if dist < 0.15:
                    obj.is_held = True
                    s.holding = obj.name
                    s.gripper_open = False
                    self._last_moved_to = None
                    return "SUCCESS"
            return "FAILED_EMPTY"

        elif action in ("PLACE_BIN_A", "PLACE_BIN_B"):
            if s.holding is None:
                return "FAILED_EMPTY"
            bin_name = "A" if action == "PLACE_BIN_A" else "B"
            obj = s.objects[s.holding]
            obj.in_bin = bin_name
            obj.is_held = False
            obj.reachable = False
            s.holding = None
            s.gripper_open = True
            return "SUCCESS"

        elif action == "CLEAR_BLOCKER":
            for obj in s.objects.values():
                if obj.blocking is not None and obj.reachable:
                    blocked_name = obj.blocking
                    obj.blocking = None
                    obj.pos = obj.pos + np.array([0.28, 0.1, 0])
                    if blocked_name in s.objects:
                        s.objects[blocked_name].reachable = True
                    return "SUCCESS"
            return "FAILED_INVALID"

        return "FAILED_INVALID"

    def get_state(self) -> SimState:
        return self._state

    def get_camera_obs(self) -> Optional[dict]:
        """
        Return camera observations + vision-extracted symbolic state.

        Stub mode:   returns None (symbolic state comes from sim config directly)
        Real mode:   returns RGB images + runs vision.py to extract object positions

        The planning layer above never needs to know which path ran —
        it always receives the same symbolic SimState either way.
        """
        if self.use_stub:
            return None  # stub: symbolic state already in self._state, no camera needed

        if self._env is not None:
            obs = self._env._get_observations()
            rgb_front = obs.get("frontview_image")
            rgb_agent = obs.get("agentview_image")

            # Run vision pipeline to get symbolic state from images
            if rgb_front is not None and self._current_cfg is not None:
                from .vision import sim_vision
                vision_result = sim_vision(rgb_front)
                # Merge detected positions back into symbolic state
                # (perception layer updates what was set from physics)
                for det in vision_result.detected_objects:
                    name = det["name"]
                    if name in self._state.objects:
                        self._state.objects[name].pos = np.array([det["x"], det["y"], det["z"]])

            return {"frontview": rgb_front, "agentview": rgb_agent}


# ── Re-exports ─────────────────────────────────────────────────────────

__all__ = ["SimWrapper", "SimState", "ObjectState"]
