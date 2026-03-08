"""
RoboReplan — OpenEnv Environment with full instrumentation.
"""
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation

from .config import EnvConfig, RealismConfig
from .curriculum import CurriculumManager
from .environment import TabletopPlanningEnv as _InternalEnv


# ── OpenEnv types ────────────────────────────────────────────────────

class RoboAction(Action):
    action: str
    reasoning: str = ""   # <think>...</think> content from the model


class RoboObservation(Observation):
    instruction: str
    steps_remaining: int
    visible_objects: list[dict]
    holding: Optional[str] = None
    completed_subgoals: list[str] = []
    known_failures: list[str] = []
    active_constraints: list[str] = []
    action_history: list[str] = []
    last_action: Optional[str] = None
    last_result: Optional[str] = None
    valid_actions: Optional[list[str]] = None
    goal_progress: Optional[float] = None
    goals_remaining: Optional[int] = None
    oracle_hint: Optional[str] = None
    nav_mode: bool = False
    gripper_cell: Optional[str] = None
    gripper_facing: Optional[str] = None
    prompt: str                          # pre-built LLM prompt
    mid_task_changed: bool = False


class RoboState(RoboObservation):
    total_reward: float = 0.0
    step_count: int = 0
    difficulty: str = "easy"


# ── Prompt builder ────────────────────────────────────────────────────

SYSTEM = (
    "You are a robot planning agent on a tabletop. "
    "Complete manipulation tasks by choosing ONE action per step.\n\n"
    "Actions: SCAN_SCENE | MOVE_NORTH | MOVE_SOUTH | MOVE_EAST | MOVE_WEST | ROTATE_LEFT | ROTATE_RIGHT | "
    "MOVE_TO_RED | MOVE_TO_BLUE | MOVE_TO_GREEN | MOVE_TO_YELLOW | MOVE_TO_PURPLE | PICK | PLACE_BIN_A | PLACE_BIN_B | CLEAR_BLOCKER\n\n"
    "Think step by step inside <think>...</think> tags, then output ONLY the action name.\n"
    "Example:\n"
    "<think>Red block is blocked by blue. I must clear blue first, then pick red, then place in bin A.</think>\n"
    "CLEAR_BLOCKER"
)


def _build_prompt(obs) -> str:
    objects = ", ".join(
        f"{o['name']}({'reachable' if o['reachable'] else 'BLOCKED'})"
        for o in obs.visible_objects
    )
    history = " → ".join(obs.action_history[-5:]) or "none"
    failures = "; ".join(obs.known_failures) or "none"
    subgoals = "; ".join(obs.completed_subgoals) or "none yet"
    constraints = "; ".join(obs.active_constraints) or "none"
    valid = ", ".join(obs.valid_actions) if obs.valid_actions else "any"
    progress = f"{obs.goal_progress:.0%}" if obs.goal_progress is not None else "?"
    nav_line = None
    if obs.nav_mode:
        nav_line = f"Navigation: gripper_cell={obs.gripper_cell or '?'} facing={obs.gripper_facing or '?'}"

    lines = [
        f"[SYSTEM] {SYSTEM}",
        "",
        f"Instruction: {obs.instruction}",
        f"Scene: {objects}",
        f"Holding: {obs.holding or 'nothing'}",
        f"Goal progress: {progress}  Goals remaining: {obs.goals_remaining}",
        *( [nav_line] if nav_line else [] ),
        f"Completed: {subgoals}",
        f"Failures: {failures}",
        f"Constraints: {constraints}",
        f"Action history: {history}",
        f"Last step: {obs.last_action or 'none'} → {obs.last_result or 'n/a'}",
        f"Valid actions now: {valid}",
        f"Steps left: {obs.steps_remaining}",
        "",
        "Next action:",
    ]
    return "\n".join(lines)


# ── Environment ───────────────────────────────────────────────────────

class RoboReplanEnv(Environment[RoboAction, RoboObservation, RoboState]):
    """
    RoboReplan with curriculum, logging, mid-task changes, and oracle hints.
    All knobs live in EnvConfig.
    """
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, difficulty: str = "easy"):
        super().__init__()
        base_cfg = {
            "easy":   EnvConfig.easy(),
            "medium": EnvConfig.medium(),
            "hard":   EnvConfig.hard(),
        }.get(difficulty, EnvConfig.easy())

        self._curriculum = CurriculumManager(base_cfg.curriculum)
        self._env = _InternalEnv(config=base_cfg)
        self._last_obs: Optional[RoboObservation] = None
        self._total_reward = 0.0
        self._step_count = 0

    def reset(self, seed=None, episode_id=None, **kwargs) -> RoboObservation:
        import random
        if seed is not None:
            random.seed(seed)

        # Curriculum: update config if level changed
        sr = self._env.logger.metrics.rolling_success_rate()
        new_level = self._curriculum.update(sr)
        if new_level != self._curriculum.current_level:
            self._env.cfg = self._curriculum.current_config()

        internal_obs = self._env.reset()
        self._total_reward = 0.0
        self._step_count = 0
        obs = self._wrap(internal_obs, done=False, reward=None)
        self._last_obs = obs
        return obs

    def step(self, action: RoboAction, timeout_s=None, **kwargs) -> RoboObservation:
        result = self._env.step(action.action, reasoning=action.reasoning or "")
        self._total_reward += result.reward
        self._step_count += 1
        obs = self._wrap(result.observation, done=result.done, reward=result.reward,
                         info=result.info)
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
            difficulty=self._curriculum.current_level,
        )

    @property
    def metrics(self) -> dict:
        m = self._env.logger.metrics.to_dict()
        m["current_difficulty"] = self._curriculum.current_level
        return m

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="robo-replan",
            description=(
                "Tabletop robot planning benchmark with curriculum, domain randomization, "
                "mid-task instruction changes, oracle hints, and full episode logging."
            ),
            version="0.2.0",
        )

    def _wrap(self, obs, done: bool, reward, info: dict = None) -> RoboObservation:
        info = info or {}
        visible = [
            {"name": o.name, "reachable": o.reachable,
             "blocking": o.blocking, "in_bin": None}
            for o in obs.visible_objects
        ]
        robo = RoboObservation(
            done=done,
            reward=reward,
            instruction=obs.instruction,
            steps_remaining=obs.steps_remaining,
            visible_objects=visible,
            holding=obs.holding,
            completed_subgoals=obs.completed_subgoals,
            known_failures=obs.known_failures,
            active_constraints=obs.active_constraints,
            action_history=obs.action_history,
            last_action=obs.last_action,
            last_result=obs.last_result,
            valid_actions=obs.valid_actions,
            goal_progress=obs.goal_progress,
            goals_remaining=obs.goals_remaining,
            oracle_hint=obs.oracle_hint,
            nav_mode=obs.nav_mode,
            gripper_cell=obs.gripper_cell,
            gripper_facing=obs.gripper_facing,
            mid_task_changed=info.get("mid_task_changed", False),
            prompt="",  # fill below
        )
        robo.prompt = _build_prompt(robo)
        return robo
