"""
Episode logger and metrics tracker.

Records every step and episode so you can:
- See exactly what the model chose vs what was optimal
- Analyze failure patterns across episodes
- Export training data for offline analysis
- Feed live stats to the /metrics endpoint and viz
"""
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class StepLog:
    step: int
    action: str
    result: str
    reward: float
    cumulative_reward: float
    valid_actions: list[str]
    oracle_action: Optional[str]      # what scripted policy would do
    chose_oracle: Optional[bool]      # did model match oracle?
    holding: Optional[str]
    n_failures_so_far: int
    n_subgoals_done: int


@dataclass
class EpisodeLog:
    episode_id: int
    instruction: str
    difficulty: str
    n_objects: int
    n_blockers: int
    n_targets: int
    had_mid_task_change: bool

    steps: list[StepLog] = field(default_factory=list)

    # Outcome
    success: bool = False
    total_reward: float = 0.0
    total_steps: int = 0
    failure_types: list[str] = field(default_factory=list)  # unique failure result codes
    repeated_failures: int = 0
    oracle_agreement_rate: float = 0.0  # fraction of steps where model == oracle

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def finish(self, success: bool):
        self.success = success
        self.total_steps = len(self.steps)
        self.total_reward = sum(s.reward for s in self.steps)
        self.end_time = time.time()
        self.failure_types = list({s.result for s in self.steps if not s.result.startswith("SUCCESS")})
        seen = set()
        rf = 0
        for s in self.steps:
            if s.result in seen:
                rf += 1
            seen.add(s.result)
        self.repeated_failures = rf
        oracle_steps = [s for s in self.steps if s.oracle_action is not None]
        if oracle_steps:
            self.oracle_agreement_rate = sum(1 for s in oracle_steps if s.chose_oracle) / len(oracle_steps)

    def to_jsonl(self) -> str:
        d = asdict(self)
        return json.dumps(d)


class MetricsTracker:
    """
    Rolling statistics across episodes.
    Feeds the /metrics endpoint and the curriculum manager.
    """

    def __init__(self, window: int = 20, max_history: int = 200):
        self.window = window
        self._history: deque[EpisodeLog] = deque(maxlen=max_history)
        self._episode_count = 0
        self._current_difficulty = "easy"

    def record(self, ep: EpisodeLog):
        self._history.append(ep)
        self._episode_count += 1

    def rolling_success_rate(self) -> float:
        recent = list(self._history)[-self.window:]
        if not recent:
            return 0.0
        return sum(1 for e in recent if e.success) / len(recent)

    def rolling_avg_reward(self) -> float:
        recent = list(self._history)[-self.window:]
        if not recent:
            return 0.0
        return sum(e.total_reward for e in recent) / len(recent)

    def rolling_avg_steps(self) -> float:
        recent = list(self._history)[-self.window:]
        if not recent:
            return 0.0
        return sum(e.total_steps for e in recent) / len(recent)

    def oracle_agreement_rate(self) -> float:
        recent = list(self._history)[-self.window:]
        if not recent:
            return 0.0
        rates = [e.oracle_agreement_rate for e in recent if e.oracle_agreement_rate > 0]
        return sum(rates) / len(rates) if rates else 0.0

    def failure_breakdown(self) -> dict[str, int]:
        """Count how often each failure type appears in recent episodes."""
        counts: dict[str, int] = {}
        for ep in list(self._history)[-self.window:]:
            for ft in ep.failure_types:
                counts[ft] = counts.get(ft, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def failure_taxonomy(self) -> dict[str, int]:
        tax = {
            "invalid": 0,
            "blocked": 0,
            "empty": 0,
            "slip": 0,
            "other": 0,
        }
        for k, v in self.failure_breakdown().items():
            kk = k.upper()
            if "INVALID" in kk:
                tax["invalid"] += v
            elif "BLOCK" in kk:
                tax["blocked"] += v
            elif "EMPTY" in kk:
                tax["empty"] += v
            elif "SLIP" in kk:
                tax["slip"] += v
            else:
                tax["other"] += v
        return tax

    def reward_curve(self) -> list[float]:
        """Per-episode total reward for plotting."""
        return [e.total_reward for e in self._history]

    def success_curve(self) -> list[int]:
        """Per-episode 0/1 for plotting."""
        return [int(e.success) for e in self._history]

    def to_dict(self) -> dict:
        return {
            "total_episodes": self._episode_count,
            "current_difficulty": self._current_difficulty,
            "rolling_success_rate": round(self.rolling_success_rate(), 3),
            "rolling_avg_reward": round(self.rolling_avg_reward(), 2),
            "rolling_avg_steps": round(self.rolling_avg_steps(), 1),
            "oracle_agreement_rate": round(self.oracle_agreement_rate(), 3),
            "failure_breakdown": self.failure_breakdown(),
            "failure_taxonomy": self.failure_taxonomy(),
            "reward_curve": self.reward_curve()[-50:],    # last 50 for the chart
            "success_curve": self.success_curve()[-50:],
        }


class EpisodeLogger:
    """
    Manages per-episode logging and writes to JSONL.
    """

    def __init__(self, export_path: Optional[str] = None, max_history: int = 200):
        self.metrics = MetricsTracker(max_history=max_history)
        self._current: Optional[EpisodeLog] = None
        self._export_path = export_path
        if export_path:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)

    def begin_episode(self, episode_id: int, instruction: str, difficulty: str,
                      n_objects: int, n_blockers: int, n_targets: int,
                      had_mid_task_change: bool = False):
        self._current = EpisodeLog(
            episode_id=episode_id,
            instruction=instruction,
            difficulty=difficulty,
            n_objects=n_objects,
            n_blockers=n_blockers,
            n_targets=n_targets,
            had_mid_task_change=had_mid_task_change,
        )

    def log_step(self, step: int, action: str, result: str, reward: float,
                 cumulative_reward: float, valid_actions: list[str],
                 oracle_action: Optional[str], holding: Optional[str],
                 n_failures: int, n_subgoals: int):
        if self._current is None:
            return
        self._current.steps.append(StepLog(
            step=step,
            action=action,
            result=result,
            reward=reward,
            cumulative_reward=cumulative_reward,
            valid_actions=valid_actions,
            oracle_action=oracle_action,
            chose_oracle=(action == oracle_action) if oracle_action else None,
            holding=holding,
            n_failures_so_far=n_failures,
            n_subgoals_done=n_subgoals,
        ))

    def end_episode(self, success: bool) -> EpisodeLog:
        if self._current is None:
            raise RuntimeError("No active episode")
        self._current.finish(success)
        ep = self._current
        self._current = None
        self.metrics.record(ep)
        if self._export_path:
            with open(self._export_path, "a") as f:
                f.write(ep.to_jsonl() + "\n")
        return ep
