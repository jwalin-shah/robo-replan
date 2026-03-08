#!/usr/bin/env python3
"""
Fast environment preflight checks.

Run:
  python3 scripts/smoke_env.py
"""
from __future__ import annotations

from collections import Counter

from server.config import EnvConfig
from server.environment import TabletopPlanningEnv
from server.models import Action


ALL_ACTIONS = {a.value for a in Action}


def run_episode(env: TabletopPlanningEnv, max_steps: int = 20):
    obs = env.reset()
    total = 0.0
    for _ in range(max_steps):
        oracle = env._oracle_action() or "SCAN_SCENE"
        if oracle not in ALL_ACTIONS:
            oracle = "SCAN_SCENE"
        r = env.step(oracle)
        total += float(r.reward)
        obs = r.observation
        if r.done:
            break
    return env._all_goals_complete(), total, obs


def assert_valid_actions(obs):
    vals = set(obs.valid_actions or [])
    bad = [a for a in vals if a not in ALL_ACTIONS]
    assert not bad, f"Unknown valid_actions: {bad}"


def check_level(name: str, cfg: EnvConfig, episodes: int = 50):
    env = TabletopPlanningEnv(config=cfg)
    nav_seen = 0
    success = 0
    rewards = []
    top_actions = Counter()

    for _ in range(episodes):
        done, rew, last_obs = run_episode(env, max_steps=cfg.task.max_steps)
        success += int(done)
        rewards.append(rew)
        assert_valid_actions(last_obs)
        if getattr(last_obs, "nav_mode", False):
            nav_seen += 1
        for a in last_obs.action_history[-5:]:
            top_actions[a] += 1

    sr = success / episodes
    avg_r = sum(rewards) / max(1, len(rewards))
    print(
        f"[{name}] success_rate={sr:.2%} avg_reward={avg_r:.3f} "
        f"nav_mode_seen={nav_seen}/{episodes} "
        f"top_actions={top_actions.most_common(5)}"
    )
    return sr


def main():
    print("Running environment smoke tests...")
    sr_easy = check_level("easy", EnvConfig.easy(), episodes=40)
    sr_med = check_level("medium", EnvConfig.medium(), episodes=40)
    sr_hard = check_level("hard", EnvConfig.hard(), episodes=40)

    # Soft floors to catch severe regressions.
    assert sr_easy >= 0.50, f"easy oracle regression: {sr_easy:.2%}"
    assert sr_med >= 0.40, f"medium oracle regression: {sr_med:.2%}"
    assert sr_hard >= 0.20, f"hard oracle regression: {sr_hard:.2%}"
    print("Smoke tests passed.")


if __name__ == "__main__":
    main()

