#!/usr/bin/env python3
"""
Deterministic hard benchmark with 20 fixed seeds.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from server.config import EnvConfig
from server.environment import TabletopPlanningEnv


SEEDS = [1000 + i for i in range(20)]
OUT = Path("logs/hard20_benchmark.json")


def run_oracle(seed: int):
    random.seed(seed)
    env = TabletopPlanningEnv(config=EnvConfig.hard())
    obs = env.reset()
    ret = 0.0
    for _ in range(env.cfg.task.max_steps):
        a = env._oracle_action() or "SCAN_SCENE"
        r = env.step(a)
        ret += float(r.reward)
        obs = r.observation
        if r.done:
            break
    return {
        "seed": seed,
        "success": bool(env._all_goals_complete()),
        "return": ret,
        "steps": len(obs.action_history),
    }


def main():
    rows = [run_oracle(s) for s in SEEDS]
    succ = sum(int(r["success"]) for r in rows) / len(rows)
    avg_ret = sum(r["return"] for r in rows) / len(rows)
    avg_steps = sum(r["steps"] for r in rows) / len(rows)
    payload = {
        "seeds": SEEDS,
        "rows": rows,
        "summary": {
            "success_rate": round(succ, 3),
            "avg_return": round(avg_ret, 3),
            "avg_steps": round(avg_steps, 3),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

