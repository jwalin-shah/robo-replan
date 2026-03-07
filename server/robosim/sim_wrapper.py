"""
Wraps robosuite to expose a high-level symbolic interface for the planning environment.

This keeps the model from having to deal with raw joint states or image tensors.
It translates high-level actions (e.g. PICK, CLEAR_BLOCKER) into robosuite motions
and reads back symbolic world state.
"""
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .randomizer import randomize_scenario, ScenarioConfig


@dataclass
class ObjectState:
    name: str
    pos: np.ndarray          # (x, y, z)
    reachable: bool = True
    blocking: Optional[str] = None  # name of object being blocked
    in_bin: Optional[str] = None    # "A" or "B"
    is_held: bool = False


@dataclass
class SimState:
    objects: dict[str, ObjectState] = field(default_factory=dict)
    gripper_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper_open: bool = True
    holding: Optional[str] = None


class SimWrapper:
    """
    Wraps robosuite environment and exposes symbolic state + high-level actions.
    Falls back to a lightweight stub when robosuite is not installed, so you
    can test env logic without a full sim install.
    """

    def __init__(self, use_stub: bool = False):
        self.use_stub = use_stub
        self._env = None
        self._state = SimState()
        self._setup()

    def _setup(self):
        if self.use_stub:
            self._init_stub()
            return
        try:
            import robosuite as suite
            from robosuite.controllers import load_composite_controller_config
            controller_config = load_composite_controller_config(controller="BASIC")
            self._env = suite.make(
                "Lift",
                robots="Panda",
                controller_configs=controller_config,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                reward_shaping=False,
                control_freq=20,
            )
            print("[SimWrapper] robosuite loaded")
        except ImportError:
            print("[SimWrapper] robosuite not installed, using stub")
            self.use_stub = True
            self._init_stub()

    def _init_stub(self):
        """Minimal fake world for testing env logic without robosuite."""
        self._state = SimState(
            objects={
                "red_block":   ObjectState("red_block",  np.array([0.1,  0.0, 0.82]), reachable=False, blocking=None),
                "blue_block":  ObjectState("blue_block", np.array([0.0,  0.0, 0.82]), reachable=True,  blocking="red_block"),
                "green_block": ObjectState("green_block",np.array([-0.1, 0.0, 0.82]), reachable=True),
            },
            gripper_pos=np.array([0.0, 0.3, 1.0]),
            gripper_open=True,
            holding=None,
        )

    def reset(self, scenario: str = "random") -> tuple[SimState, ScenarioConfig]:
        """Reset the scene with a randomized or fixed scenario."""
        if not self.use_stub and self._env is not None:
            self._env.reset()

        if scenario == "random":
            cfg = randomize_scenario(force_blocked=random.random() < 0.6)
        elif scenario == "blocked":
            cfg = randomize_scenario(n_objects=3, n_targets=1, n_blockers=1, force_blocked=True)
        elif scenario == "unblocked":
            cfg = randomize_scenario(n_objects=3, n_targets=1, n_blockers=0)
        else:
            cfg = randomize_scenario()

        self._build_state_from_config(cfg)
        return self._state, cfg

    def _build_state_from_config(self, cfg: ScenarioConfig):
        """Build SimState from a ScenarioConfig."""
        objects = {}
        for obj_name in cfg.objects:
            x, y = cfg.positions.get(obj_name, (0.0, 0.0))
            # blocked = something else is blocking this object
            is_blocked = obj_name in cfg.blockers.values()
            objects[obj_name] = ObjectState(
                name=obj_name,
                pos=np.array([x, y, 0.82]),
                reachable=not is_blocked,
                blocking=cfg.blockers.get(obj_name),  # what this object is blocking
            )
        self._state = SimState(
            objects=objects,
            gripper_pos=np.array([0.0, 0.25, 1.0]),
            gripper_open=True,
            holding=None,
        )
        self._current_cfg = cfg

    def get_state(self) -> SimState:
        return self._state

    def execute(self, action: str) -> str:
        """
        Execute a high-level action.
        Returns one of:
          SUCCESS | FAILED_BLOCKED | FAILED_EMPTY | FAILED_INVALID | FAILED_WRONG_TARGET
        """
        s = self._state

        if action == "SCAN_SCENE":
            # Scan reveals hidden info but costs a step
            for obj in s.objects.values():
                obj.reachable = obj.reachable  # in real sim would update from perception
            return "SUCCESS"

        elif action.startswith("MOVE_TO_"):
            # MOVE_TO_RED -> red_block, MOVE_TO_YELLOW -> yellow_block, etc.
            color = action[len("MOVE_TO_"):].lower()
            name = color + "_block"
            if name not in s.objects:
                return "FAILED_INVALID"
            obj = s.objects[name]
            if not obj.reachable:
                return "FAILED_BLOCKED"
            s.gripper_pos = obj.pos.copy() + np.array([0, 0, 0.05])
            return "SUCCESS"

        elif action == "PICK":
            if s.holding is not None:
                return "FAILED_INVALID"
            # find closest reachable object
            for obj in s.objects.values():
                if obj.reachable and not obj.is_held and obj.in_bin is None:
                    dist = np.linalg.norm(s.gripper_pos[:2] - obj.pos[:2])
                    if dist < 0.12:
                        obj.is_held = True
                        s.holding = obj.name
                        s.gripper_open = False
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
            # Remove a blocking object from in front of the target
            for obj in s.objects.values():
                if obj.blocking is not None and obj.reachable:
                    blocked_name = obj.blocking
                    obj.blocking = None
                    obj.pos = obj.pos + np.array([0.3, 0, 0])  # push aside
                    if blocked_name in s.objects:
                        s.objects[blocked_name].reachable = True
                    return "SUCCESS"
            return "FAILED_INVALID"

        return "FAILED_INVALID"
