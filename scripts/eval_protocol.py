#!/usr/bin/env python3
"""
Robust eval protocol: 3 seeds x fixed suite.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from server.config import EnvConfig
from server.environment import TabletopPlanningEnv


SEEDS = [1337, 2024, 4242]
LEVELS = [("easy", EnvConfig.easy), ("medium", EnvConfig.medium), ("hard", EnvConfig.hard)]
OUT = Path("logs/eval_protocol_3seed.json")


def run_level(level_name: str, cfg_fn, seed: int, episodes: int = 30):
    random.seed(seed)
    env = TabletopPlanningEnv(config=cfg_fn())
    succ = 0
    rets = []
    for _ in range(episodes):
        obs = env.reset()
        total = 0.0
        for _ in range(env.cfg.task.max_steps):
            a = env._oracle_action() or "SCAN_SCENE"
            r = env.step(a)
            total += float(r.reward)
            obs = r.observation
            if r.done:
                break
        succ += int(env._all_goals_complete())
        rets.append(total)
    return {
        "level": level_name,
        "seed": seed,
        "episodes": episodes,
        "success_rate": succ / episodes,
        "avg_return": sum(rets) / max(1, len(rets)),
    }


def main():
    rows = []
    for seed in SEEDS:
        for level_name, cfg_fn in LEVELS:
            rows.append(run_level(level_name, cfg_fn, seed))

    grouped = {}
    for level_name, _ in LEVELS:
        vals = [r for r in rows if r["level"] == level_name]
        succ = [r["success_rate"] for r in vals]
        ret = [r["avg_return"] for r in vals]
        grouped[level_name] = {
            "success_mean": sum(succ) / len(succ),
            "return_mean": sum(ret) / len(ret),
        }

    payload = {"seeds": SEEDS, "rows": rows, "summary": grouped}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(grouped, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

