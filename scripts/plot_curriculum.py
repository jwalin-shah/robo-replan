#!/usr/bin/env python3
"""
Plot curriculum/reward traces from logs/episodes.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

LOG = Path("logs/episodes.jsonl")
PNG = Path("logs/curriculum_plot.png")
CSV = Path("logs/curriculum_plot.csv")
REQUIRED_FIELDS = {
    "episode_id": int,
    "success": bool,
    "total_reward": (int, float),
    "total_steps": int,
    "difficulty": str,
}


def _matches_expected_type(value: object, expected_type: type | tuple[type, ...]) -> bool:
    if expected_type is int:
        return type(value) is int
    if expected_type == (int, float):
        return type(value) in (int, float)
    return isinstance(value, expected_type)


def _expected_type_name(expected_type: type | tuple[type, ...]) -> str:
    if expected_type == (int, float):
        return "number"
    return expected_type.__name__


def _parse_episode_row(line: str, line_number: int, path: Path = LOG) -> dict:
    try:
        episode = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_number} invalid JSON: {exc.msg}") from exc

    if not isinstance(episode, dict):
        raise ValueError(f"{path}:{line_number} episode must be a JSON object")

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in episode:
            raise ValueError(f"{path}:{line_number} missing required field {field!r}")
        if not _matches_expected_type(episode[field], expected_type):
            raise ValueError(
                f"{path}:{line_number} field {field!r} must be "
                f"{_expected_type_name(expected_type)}"
            )

    return {
        "episode_id": episode["episode_id"],
        "success": int(episode["success"]),
        "total_reward": float(episode["total_reward"]),
        "total_steps": episode["total_steps"],
        "difficulty": episode["difficulty"],
    }


def main():
    if not LOG.exists():
        raise SystemExit(f"Missing {LOG}")
    rows = []
    for line_number, line in enumerate(LOG.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        rows.append(_parse_episode_row(line, line_number))

    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("w", encoding="utf-8") as f:
        f.write("episode_id,success,total_reward,total_steps,difficulty\n")
        for r in rows:
            f.write(f"{r['episode_id']},{r['success']},{r['total_reward']},{r['total_steps']},{r['difficulty']}\n")
    print(f"Wrote {CSV}")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; CSV exported only.")
        return

    ep = [r["episode_id"] for r in rows]
    rew = [r["total_reward"] for r in rows]
    suc = [r["success"] for r in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(ep, rew, label="episode reward")
    plt.plot(ep, [x * max(1.0, max(rew)) for x in suc], label="success (scaled)", alpha=0.6)
    plt.title("Curriculum / Training Trace")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PNG)
    print(f"Wrote {PNG}")


if __name__ == "__main__":
    main()
