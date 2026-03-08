"""
EnvConfig — single source of truth for every knob in RoboReplan.

Change one value here and it propagates everywhere: sim, reward,
curriculum, observation, logging.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardWeights:
    # Terminal
    task_complete: float = 10.0
    efficiency_bonus_max: float = 5.0   # scales with steps saved

    # Progress
    correct_placement: float = 2.0
    successful_pick: float = 2.0
    blocker_cleared: float = 2.0
    recovery_after_failure: float = 1.0
    useful_scan: float = 0.5            # scan that reveals new info

    # Penalties
    wrong_bin: float = -3.0
    wrong_pick: float = -1.0
    first_failure: float = -1.0
    repeated_failure: float = -2.5     # same action:result seen before
    constraint_violation: float = -4.0
    step_cost: float = -0.05
    timeout_failure: float = -10.0
    missed_deadline: float = -5.0
    useless_action: float = -0.4
    fragile_pick_penalty: float = -3.0  # picking fragile object without scanning first


@dataclass
class ObsConfig:
    include_valid_actions: bool = True   # which actions make sense right now
    include_goal_progress: float = True  # 0.0–1.0 fraction of goals done
    include_action_history: int = 5      # last N actions in obs (0 = none)
    include_oracle_hint: bool = False    # scripted policy action (teaching signal)
    include_distance_to_goal: bool = True
    include_hidden_traits: bool = True
    include_deadlines: bool = True


@dataclass
class TaskConfig:
    n_objects_min: int = 2
    n_objects_max: int = 5
    n_targets_min: int = 1
    n_targets_max: int = 2
    n_blockers_min: int = 0
    n_blockers_max: int = 2
    max_steps: int = 20
    force_blocked_prob: float = 0.6      # how often to guarantee a blocker

    # Mid-task instruction change — can fire at multiple steps per episode
    mid_task_change_prob: float = 0.0    # prob per candidate step of a change occurring
    mid_task_change_step: int = 8        # kept for backward compat; use mid_task_change_steps
    mid_task_change_steps: list = field(default_factory=lambda: [8])  # all candidate steps
    navigation_mode: bool = False
    lock_wrong_bin_steps: int = 3
    enable_deadlines: bool = False
    deadline_min_step: int = 4
    deadline_max_step: int = 10
    enable_hidden_traits: bool = True
    require_scan_for_traits: bool = True
    enable_distractor_actions: bool = True
    enable_partial_observability_zones: bool = True
    adversarial_sampling_prob: float = 0.0
    scenario_pack: str = "default"


@dataclass
class RealismConfig:
    grasp_fail_prob: float = 0.0
    clear_partial_prob: float = 0.0
    reachability_noise: float = 0.0
    hidden_object_prob: float = 0.0
    object_drift_prob: float = 0.0

    @classmethod
    def easy(cls):
        return cls()

    @classmethod
    def medium(cls):
        return cls(grasp_fail_prob=0.15, clear_partial_prob=0.20,
                   reachability_noise=0.10, hidden_object_prob=0.30)

    @classmethod
    def hard(cls):
        return cls(grasp_fail_prob=0.20, clear_partial_prob=0.25,
                   reachability_noise=0.15, hidden_object_prob=0.40,
                   object_drift_prob=0.02)


@dataclass
class CurriculumConfig:
    enabled: bool = True
    # Advance to next level when rolling success rate crosses these thresholds
    advance_threshold: float = 0.75     # need 75% success to advance
    retreat_threshold: float = 0.35     # fall back if success drops below 35%
    window: int = 20                    # episodes to average over
    levels: list = field(default_factory=lambda: ["easy", "medium", "hard"])


@dataclass
class LogConfig:
    log_every_step: bool = True
    log_episode_summary: bool = True
    max_episode_history: int = 200      # keep last N episodes in memory
    export_path: Optional[str] = "logs/episodes.jsonl"


@dataclass
class EnvConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    realism: RealismConfig = field(default_factory=RealismConfig.easy)
    reward: RewardWeights = field(default_factory=RewardWeights)
    obs: ObsConfig = field(default_factory=ObsConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    log: LogConfig = field(default_factory=LogConfig)

    @classmethod
    def easy(cls):
        return cls(realism=RealismConfig.easy(),
                   task=TaskConfig(n_blockers_max=1, mid_task_change_prob=0.0,
                                   require_scan_for_traits=False))  # no penalty in easy

    @classmethod
    def medium(cls):
        # Enforce scan-before-pick: agent must learn information gathering
        return cls(realism=RealismConfig.medium(),
                   task=TaskConfig(n_blockers_max=2, mid_task_change_prob=0.20,
                                   mid_task_change_steps=[8],
                                   enable_deadlines=True,
                                   require_scan_for_traits=True))

    @classmethod
    def hard(cls):
        # Multiple instruction changes + scan enforcement + navigation
        return cls(realism=RealismConfig.hard(),
                   task=TaskConfig(n_objects_max=5, n_blockers_max=3,
                                   n_targets_max=2, mid_task_change_prob=0.35,
                                   mid_task_change_steps=[6, 12],
                                   navigation_mode=True, enable_deadlines=True,
                                   require_scan_for_traits=True,
                                   adversarial_sampling_prob=0.25))

    @classmethod
    def long_horizon(cls, scenario_pack: str = "default"):
        """
        Extended planning episodes: 3–4 targets, 6–8 objects, chained blockers,
        up to 3 mid-task instruction changes, deadlines, scan-enforced traits.
        Max 50 steps. Tests true multi-step adaptive planning.
        Use scenario_pack="warehouse"|"pharmacy"|"lab" for professional task framing.
        """
        return cls(realism=RealismConfig.hard(),
                   task=TaskConfig(n_objects_min=4, n_objects_max=8,
                                   n_targets_min=3, n_targets_max=4,
                                   n_blockers_min=1, n_blockers_max=4,
                                   max_steps=50,
                                   force_blocked_prob=0.9,
                                   mid_task_change_prob=0.40,
                                   mid_task_change_steps=[8, 18, 30],
                                   navigation_mode=False,
                                   require_scan_for_traits=True,
                                   enable_deadlines=True,
                                   deadline_min_step=8,
                                   deadline_max_step=20,
                                   adversarial_sampling_prob=0.30,
                                   scenario_pack=scenario_pack))
