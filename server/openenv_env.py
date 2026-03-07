"""
RoboReplan — OpenEnv Environment

Implements the OpenEnv Environment interface so this can be deployed
to HF Spaces and consumed by TRL's GRPO trainer via AutoEnv.
"""
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation

from .environment import TabletopPlanningEnv as _InternalEnv
from .robosim.realism import RealismConfig


# ── OpenEnv Action ────────────────────────────────────────────────────

class RoboAction(Action):
    """A single high-level robot action string."""
    action: str


# ── OpenEnv Observation ───────────────────────────────────────────────

class RoboObservation(Observation):
    """
    Structured observation returned at every step.
    Text-friendly so LLMs can read it directly.
    """
    instruction: str
    steps_remaining: int
    visible_objects: list[dict]   # [{name, reachable, blocking, in_bin}]
    holding: Optional[str]
    completed_subgoals: list[str]
    known_failures: list[str]
    active_constraints: list[str]
    last_action: Optional[str]
    last_result: Optional[str]

    # Human-readable prompt for the LLM — pre-built so the trainer
    # doesn't need to do any formatting
    prompt: str


# ── OpenEnv State ─────────────────────────────────────────────────────

class RoboState(RoboObservation):
    total_reward: float = 0.0
    step_count: int = 0


# ── Environment ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a robot planning agent. Complete tabletop manipulation tasks by choosing one action per step.

Actions: SCAN_SCENE | MOVE_TO_RED | MOVE_TO_BLUE | MOVE_TO_GREEN | MOVE_TO_YELLOW | MOVE_TO_PURPLE | PICK | PLACE_BIN_A | PLACE_BIN_B | CLEAR_BLOCKER

Reply with ONLY the action name."""


def _build_prompt(obs) -> str:
    objects = ", ".join(
        f"{o.name}({'reachable' if o.reachable else 'BLOCKED'})"
        for o in obs.visible_objects
    )
    failures = "; ".join(obs.known_failures) or "none"
    subgoals = "; ".join(obs.completed_subgoals) or "none yet"
    constraints = "; ".join(obs.active_constraints) or "none"
    return (
        f"[SYSTEM] {SYSTEM_PROMPT}\n\n"
        f"Instruction: {obs.instruction}\n"
        f"Scene: {objects}\n"
        f"Holding: {obs.holding or 'nothing'}\n"
        f"Completed: {subgoals}\n"
        f"Failures: {failures}\n"
        f"Constraints: {constraints}\n"
        f"Last: {obs.last_action or 'none'} → {obs.last_result or 'n/a'}\n"
        f"Steps left: {obs.steps_remaining}\n\n"
        f"Next action:"
    )


def _to_robo_obs(internal_obs, done: bool, reward: float) -> RoboObservation:
    return RoboObservation(
        done=done,
        reward=reward,
        instruction=internal_obs.instruction,
        steps_remaining=internal_obs.steps_remaining,
        visible_objects=[
            {
                "name": o.name,
                "reachable": o.reachable,
                "blocking": o.blocking,
                "in_bin": None,
            }
            for o in internal_obs.visible_objects
        ],
        holding=internal_obs.holding,
        completed_subgoals=internal_obs.completed_subgoals,
        known_failures=internal_obs.known_failures,
        active_constraints=internal_obs.active_constraints,
        last_action=internal_obs.last_action,
        last_result=internal_obs.last_result,
        prompt=_build_prompt(internal_obs),
    )


class RoboReplanEnv(Environment[RoboAction, RoboObservation, RoboState]):
    """
    RoboReplan: tabletop robot planning environment.

    The agent receives a natural-language instruction and a structured
    observation of the scene, then chooses high-level actions to complete
    multi-step manipulation tasks under partial observability and noise.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, difficulty: str = "easy"):
        super().__init__()
        cfg_map = {
            "easy":   RealismConfig.easy(),
            "medium": RealismConfig.medium(),
            "hard":   RealismConfig.hard(),
        }
        self._env = _InternalEnv(
            use_stub=True,
            realism=cfg_map.get(difficulty, RealismConfig.easy()),
        )
        self._last_obs = None
        self._total_reward = 0.0
        self._step_count = 0

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> RoboObservation:
        import random
        if seed is not None:
            random.seed(seed)
        internal_obs = self._env.reset()
        self._total_reward = 0.0
        self._step_count = 0
        obs = _to_robo_obs(internal_obs, done=False, reward=None)
        self._last_obs = obs
        return obs

    def step(self, action: RoboAction, timeout_s: Optional[float] = None, **kwargs) -> RoboObservation:
        result = self._env.step(action.action)
        self._total_reward += result.reward
        self._step_count += 1
        obs = _to_robo_obs(result.observation, done=result.done, reward=result.reward)
        self._last_obs = obs
        return obs

    @property
    def state(self) -> RoboState:
        if self._last_obs is None:
            raise RuntimeError("Call reset() first.")
        return RoboState(
            **self._last_obs.model_dump(),
            total_reward=self._total_reward,
            step_count=self._step_count,
        )

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="robo-replan",
            description=(
                "Tabletop robot planning benchmark: train LLMs to decompose tasks, "
                "handle blockers, replan after failure, and follow constraints — "
                "with domain randomization and realistic action noise."
            ),
            version="0.1.0",
        )
