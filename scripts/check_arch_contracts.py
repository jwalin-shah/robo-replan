#!/usr/bin/env python3
"""Validate documented architecture contracts against code wiring."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from server.config import EnvConfig
from server.environment import TabletopPlanningEnv
from server.logger import EpisodeLog, EpisodeLogger, StepLog


ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_episode_evidence_contract() -> None:
    cfg = EnvConfig.easy()
    require(
        cfg.log.export_path == "logs/episodes.local.jsonl",
        "LogConfig.export_path default must remain logs/episodes.local.jsonl",
    )

    env = TabletopPlanningEnv(config=cfg)
    require(
        isinstance(env.logger, EpisodeLogger),
        "TabletopPlanningEnv must use EpisodeLogger for runtime evidence",
    )
    require(
        env.logger._export_path == cfg.log.export_path,
        "TabletopPlanningEnv must pass EnvConfig.log.export_path to EpisodeLogger",
    )

    ep = EpisodeLog(
        episode_id=1,
        instruction="Place the red block in bin A.",
        difficulty="easy",
        n_objects=2,
        n_blockers=1,
        n_targets=1,
        had_mid_task_change=False,
    )
    ep.steps.append(
        StepLog(
            step=1,
            action="SCAN_SCENE",
            result="SUCCESS",
            reward=0.45,
            cumulative_reward=0.45,
            valid_actions=["SCAN_SCENE"],
            oracle_action=None,
            chose_oracle=None,
            holding=None,
            n_failures_so_far=0,
            n_subgoals_done=0,
        )
    )
    ep.finish(success=True)
    serialized = json.loads(ep.to_jsonl())
    require(
        serialized["episode_id"] == 1
        and serialized["success"] is True
        and serialized["steps"][0]["action"] == "SCAN_SCENE",
        "EpisodeLog.to_jsonl must preserve episode and step fields",
    )

    with tempfile.TemporaryDirectory() as tmp:
        export_path = Path(tmp) / "nested" / "episodes.jsonl"
        logger = EpisodeLogger(export_path=str(export_path), max_history=2)
        logger.begin_episode(
            episode_id=7,
            instruction="Place the blue block in bin B.",
            difficulty="medium",
            n_objects=3,
            n_blockers=1,
            n_targets=1,
        )
        logger.log_step(
            step=1,
            action="SCAN_SCENE",
            result="SUCCESS",
            reward=0.45,
            cumulative_reward=0.45,
            valid_actions=["SCAN_SCENE"],
            oracle_action=None,
            holding=None,
            n_failures=0,
            n_subgoals=0,
        )
        logger.end_episode(success=False)
        lines = export_path.read_text().splitlines()
        require(len(lines) == 1, "EpisodeLogger must append one JSONL row per ended episode")
        require(
            json.loads(lines[0])["episode_id"] == 7,
            "EpisodeLogger must write the ended episode to the configured path",
        )

    context = read("CONTEXT.md")
    agents = read("AGENTS.md")
    contracts = read("docs/ARCHITECTURE_CONTRACTS.md")
    required_docs = [
        "server/config.py",
        "server/environment.py",
        "server/logger.py",
        "logs/episodes.jsonl",
        "logs/episodes.local.jsonl",
        "scripts/check.sh",
    ]
    for marker in required_docs:
        require(marker in context or marker in agents, f"{marker} missing from repo guidance")
        require(marker in contracts, f"{marker} missing from architecture contracts doc")


def main() -> None:
    check_episode_evidence_contract()
    print("Architecture contracts passed.")


if __name__ == "__main__":
    main()
