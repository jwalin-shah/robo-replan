#!/usr/bin/env python3
"""Validate committed RoboReplan evidence artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EPISODE_REQUIRED = {
    "episode_id",
    "instruction",
    "difficulty",
    "n_objects",
    "n_blockers",
    "n_targets",
    "steps",
    "success",
    "total_reward",
    "total_steps",
    "failure_types",
}

STEP_REQUIRED = {
    "step",
    "action",
    "result",
    "reward",
    "cumulative_reward",
    "valid_actions",
}


def load_json(path: str) -> object:
    return json.loads((ROOT / path).read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_episode_jsonl() -> None:
    path = ROOT / "logs/episodes.jsonl"
    require(path.exists(), "logs/episodes.jsonl is missing")

    count = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            episode = json.loads(line)
            missing = EPISODE_REQUIRED - set(episode)
            require(not missing, f"{path}:{line_number} missing fields {sorted(missing)}")
            require(isinstance(episode["steps"], list), f"{path}:{line_number} steps must be a list")
            require(isinstance(episode["success"], bool), f"{path}:{line_number} success must be bool")
            require(isinstance(episode["total_steps"], int), f"{path}:{line_number} total_steps must be int")
            for step_index, step in enumerate(episode["steps"], start=1):
                missing_step = STEP_REQUIRED - set(step)
                require(
                    not missing_step,
                    f"{path}:{line_number} step {step_index} missing fields {sorted(missing_step)}",
                )
            count += 1

    require(count > 0, "logs/episodes.jsonl has no episodes")


def validate_summary_artifacts() -> None:
    env_quality = load_json("logs/env_quality_report.json")
    require(isinstance(env_quality, dict), "env quality report must be an object")
    reports = env_quality.get("reports")
    require(isinstance(reports, list) and reports, "env quality report needs reports")
    levels = {report.get("level") for report in reports}
    require({"easy", "medium", "hard"} <= levels, "env quality report needs easy/medium/hard")
    for report in reports:
        for field in ("episodes", "success_rate", "avg_return", "avg_steps"):
            require(field in report, f"env quality report missing {field}")

    for path in ("logs/eval_protocol_3seed.json", "logs/hard20_benchmark.json"):
        data = load_json(path)
        require(isinstance(data, dict), f"{path} must be an object")
        summary = data.get("summary")
        require(isinstance(summary, dict) and summary, f"{path} needs summary")


def validate_runtime_log_contract() -> None:
    from server.config import EnvConfig
    from server.environment import TabletopPlanningEnv

    export_path = EnvConfig().log.export_path
    require(export_path == "logs/episodes.local.jsonl", "default export_path must be local JSONL")

    tracked = subprocess.run(
        ["git", "ls-files", "logs/*.local.jsonl"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(not tracked, "local runtime JSONL files must not be tracked")

    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            cfg = EnvConfig.easy()
            cfg.task.max_steps = 1
            env = TabletopPlanningEnv(config=cfg)
            env.step("SCAN_SCENE")
        finally:
            os.chdir(original_cwd)

        local_runtime_log = Path(tmp) / export_path
        tracked_fixture_log = Path(tmp) / "logs/episodes.jsonl"
        require(local_runtime_log.exists(), "default runtime run must write local JSONL")
        require(
            not tracked_fixture_log.exists(),
            "default runtime run must not write the tracked evidence fixture path",
        )


def main() -> None:
    validate_episode_jsonl()
    validate_summary_artifacts()
    validate_runtime_log_contract()
    print("RoboReplan evidence artifacts validated.")


if __name__ == "__main__":
    main()
