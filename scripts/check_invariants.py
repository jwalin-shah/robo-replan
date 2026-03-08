#!/usr/bin/env python3
"""
Strict action invariant checks.

Asserts:
1) valid action -> not FAILED_INVALID
2) invalid action -> FAILED_INVALID
"""
from __future__ import annotations

import random

from server.config import EnvConfig
from server.environment import TabletopPlanningEnv
from server.models import Action


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


def main():
    random.seed(1337)
    run("easy", EnvConfig.easy())
    run("medium", EnvConfig.medium())
    run("hard", EnvConfig.hard())
    print("All invariants passed.")


if __name__ == "__main__":
    main()
