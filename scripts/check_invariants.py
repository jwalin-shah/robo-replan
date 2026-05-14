#!/usr/bin/env python3
"""
Strict action invariant checks.

Asserts:
1) valid action -> not FAILED_INVALID
2) invalid action -> FAILED_INVALID
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.config import EnvConfig
import server.environment as environment_module
from server.environment import TabletopPlanningEnv
from server.models import Action
from server.robosim.randomizer import ScenarioConfig


ALL = [a.value for a in Action]


def run(level: str, cfg, episodes: int = 40):
    env = TabletopPlanningEnv(config=cfg)
    checks = 0
    for _ in range(episodes):
        obs = env.reset()
        for _ in range(cfg.task.max_steps):
            valid = set(obs.valid_actions or [])
            if not valid:
                valid = {"SCAN_SCENE"}
            bad_choices = [a for a in ALL if a not in valid]
            bad = random.choice(bad_choices) if bad_choices else None

            if bad is not None:
                r_bad = env.step(bad)
                checks += 1
                assert r_bad.info.get("result") == "FAILED_INVALID", (
                    f"[{level}] invalid action not rejected: {bad}, got={r_bad.info.get('result')}"
                )
                obs = r_bad.observation
                if r_bad.done:
                    break

            valid2 = set(obs.valid_actions or [])
            if not valid2:
                valid2 = {"SCAN_SCENE"}
            good = random.choice(list(valid2))
            r_good = env.step(good)
            checks += 1
            assert r_good.info.get("result") != "FAILED_INVALID", (
                f"[{level}] valid action became invalid: {good}"
            )
            obs = r_good.observation
            if r_good.done:
                break
    print(f"[{level}] invariant checks passed: {checks}")


def run_mid_task_change_regression():
    cfg = EnvConfig.easy()
    cfg.task.n_objects_min = 2
    cfg.task.n_objects_max = 2
    cfg.task.n_targets_min = 2
    cfg.task.n_targets_max = 2
    cfg.task.n_blockers_min = 0
    cfg.task.n_blockers_max = 0
    cfg.task.force_blocked_prob = 0.0
    cfg.task.mid_task_change_prob = 1.0
    cfg.task.mid_task_change_steps = [3]
    cfg.task.require_scan_for_traits = False
    cfg.realism.object_drift_prob = 0.0
    cfg.log.log_every_step = False

    scenario = ScenarioConfig(
        objects=["red_block", "blue_block"],
        targets={"red_block": "A", "blue_block": "A"},
        blockers={},
        distractors=[],
        constraint=None,
        instruction="Place the red block in bin A, then the blue block in bin A.",
        positions={"red_block": (0.0, 0.05), "blue_block": (0.15, 0.05)},
        hidden_traits={},
        deadlines={},
    )

    original_randomize = environment_module.randomize_scenario
    original_choice = environment_module.random.choice
    try:
        environment_module.randomize_scenario = lambda **_: scenario
        environment_module.random.choice = lambda seq: seq[0]

        env = TabletopPlanningEnv(config=cfg)
        env.reset()
        env.step("MOVE_TO_RED")
        env.step("PICK")
        env.step("PLACE_BIN_A")

        changed = env.step("SCAN_SCENE")
        assert changed.info["mid_task_changed"] is True
        assert changed.info["goal_progress"] == 0.5, (
            "mid-task change rewrote an already completed placement"
        )
    finally:
        environment_module.randomize_scenario = original_randomize
        environment_module.random.choice = original_choice

    print("[mid_task_change] completed-target regression passed")


def run_action_availability_contract_regression():
    cfg = EnvConfig.easy()
    cfg.task.n_objects_min = 2
    cfg.task.n_objects_max = 2
    cfg.task.n_targets_min = 1
    cfg.task.n_targets_max = 1
    cfg.task.n_blockers_min = 1
    cfg.task.n_blockers_max = 1
    cfg.task.force_blocked_prob = 1.0
    cfg.realism.grasp_fail_prob = 0.0
    cfg.realism.clear_partial_prob = 0.0
    cfg.realism.object_drift_prob = 0.0
    cfg.log.log_every_step = False

    scenario = ScenarioConfig(
        objects=["red_block", "blue_block"],
        targets={"red_block": "A"},
        blockers={"blue_block": "red_block"},
        distractors=["blue_block"],
        constraint=None,
        instruction="Place the red block in bin A.",
        positions={"red_block": (0.0, -0.05), "blue_block": (0.0, 0.05)},
        hidden_traits={},
        deadlines={},
    )

    original_randomize = environment_module.randomize_scenario
    try:
        environment_module.randomize_scenario = lambda **_: scenario

        env = TabletopPlanningEnv(config=cfg)
        obs = env.reset()
        assert list(env._valid_actions_with_reasons().keys()) == env._valid_actions(), (
            "valid action list must be derived from action reasons"
        )
        assert "CLEAR_BLOCKER" in (obs.valid_actions or [])

        moved = env.step("MOVE_TO_BLUE")
        assert moved.info["result"] == "SUCCESS"
        picked = env.step("PICK")
        assert picked.info["result"] == "SUCCESS"
        assert picked.observation.holding == "blue_block"

        reasoned = env._valid_actions_with_reasons()
        listed = env._valid_actions()
        assert list(reasoned.keys()) == listed, (
            f"valid action list drifted from reasons: reasons={reasoned}, list={listed}"
        )
        assert "CLEAR_BLOCKER" not in reasoned, (
            "CLEAR_BLOCKER must not be available while holding an object"
        )
    finally:
        environment_module.randomize_scenario = original_randomize

    print("[action_availability] reasoned valid-action contract passed")


def main():
    random.seed(1337)
    run_mid_task_change_regression()
    run_action_availability_contract_regression()
    run("easy", EnvConfig.easy())
    run("medium", EnvConfig.medium())
    run("hard", EnvConfig.hard())
    print("All invariants passed.")


if __name__ == "__main__":
    main()
