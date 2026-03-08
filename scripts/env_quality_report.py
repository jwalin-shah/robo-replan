#!/usr/bin/env python3
"""
Environment quality report for judge-facing confidence.

This script evaluates:
  - Solvability by oracle (per difficulty)
  - Failure-mode coverage (blocked/invalid/empty/wrong-bin)
  - Dynamic behavior (mid-task changes)
  - Navigation complexity signal (hard mode nav actions present)

Run:
  python3 scripts/env_quality_report.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from server.config import EnvConfig
from server.environment import TabletopPlanningEnv
from server.models import Action


OUT = Path("logs/env_quality_report.json")
ALL_ACTIONS = {a.value for a in Action}


@dataclass
class LevelReport:
    level: str
    episodes: int
    success_rate: float
    avg_return: float
    avg_steps: float
    nav_mode_seen: int
    mid_task_changes: int
    failure_counts: dict
    action_counts: dict


def run_oracle_episode(env: TabletopPlanningEnv, max_steps: int):
    obs = env.reset()
    total = 0.0
    step = 0
    failures = Counter()
    actions = Counter()
    mid_change = 0

    for _ in range(max_steps):
        a = env._oracle_action() or "SCAN_SCENE"
        if a not in ALL_ACTIONS:
            a = "SCAN_SCENE"
        r = env.step(a)
        total += float(r.reward)
        step += 1
        actions[a] += 1
        res = r.info.get("result", "")
        if isinstance(res, str) and res.startswith("FAILED"):
            failures[res] += 1
        if r.info.get("mid_task_changed", False):
            mid_change += 1
        obs = r.observation
        if r.done:
            break

    return {
        "done": env._all_goals_complete(),
        "return": total,
        "steps": step,
        "failures": failures,
        "actions": actions,
        "nav_mode": int(bool(getattr(obs, "nav_mode", False))),
        "mid_change": mid_change,
    }


def evaluate_level(level: str, cfg_fn, episodes: int = 100) -> LevelReport:
    cfg = cfg_fn()
    env = TabletopPlanningEnv(config=cfg)
    succ = 0
    returns = []
    steps = []
    nav_seen = 0
    mid_changes = 0
    failure_counts = Counter()
    action_counts = Counter()

    for _ in range(episodes):
        out = run_oracle_episode(env, max_steps=cfg.task.max_steps)
        succ += int(out["done"])
        returns.append(out["return"])
        steps.append(out["steps"])
        nav_seen += out["nav_mode"]
        mid_changes += out["mid_change"]
        failure_counts.update(out["failures"])
        action_counts.update(out["actions"])

    return LevelReport(
        level=level,
        episodes=episodes,
        success_rate=succ / episodes,
        avg_return=sum(returns) / max(1, len(returns)),
        avg_steps=sum(steps) / max(1, len(steps)),
        nav_mode_seen=nav_seen,
        mid_task_changes=mid_changes,
        failure_counts=dict(failure_counts),
        action_counts=dict(action_counts.most_common(12)),
    )


def summarize_judge_story(level_reports: list[LevelReport]) -> dict:
    by_level = {r.level: r for r in level_reports}
    novelty_signals = {
        "dynamic_constraints_present": any(r.mid_task_changes > 0 for r in level_reports),
        "navigation_mode_present": by_level.get("hard", LevelReport("hard", 0, 0, 0, 0, 0, 0, {}, {})).nav_mode_seen > 0,
        "failure_mode_coverage": sorted(
            {k for r in level_reports for k in r.failure_counts.keys()}
        ),
        "action_diversity": {
            r.level: len(r.action_counts) for r in level_reports
        },
    }
    robustness_signals = {
        "oracle_success_rates": {r.level: round(r.success_rate, 3) for r in level_reports},
        "avg_steps": {r.level: round(r.avg_steps, 2) for r in level_reports},
    }
    return {
        "novelty_signals": novelty_signals,
        "robustness_signals": robustness_signals,
    }


def main():
    reports = [
        evaluate_level("easy", EnvConfig.easy, episodes=80),
        evaluate_level("medium", EnvConfig.medium, episodes=80),
        evaluate_level("hard", EnvConfig.hard, episodes=80),
    ]
    story = summarize_judge_story(reports)

    print("=== ENV QUALITY REPORT ===")
    for r in reports:
        print(
            f"[{r.level}] success={r.success_rate:.1%} "
            f"avg_return={r.avg_return:.2f} avg_steps={r.avg_steps:.2f} "
            f"nav_seen={r.nav_mode_seen}/{r.episodes} mid_changes={r.mid_task_changes}"
        )
        print(f"  failure_modes={r.failure_counts}")
        print(f"  top_actions={r.action_counts}")

    print("\n=== JUDGE STORY SIGNALS ===")
    print(json.dumps(story, indent=2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "reports": [asdict(r) for r in reports],
                "story": story,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

