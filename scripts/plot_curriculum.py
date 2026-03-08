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


def main():
    if not LOG.exists():
        raise SystemExit(f"Missing {LOG}")
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append({
            "episode_id": d.get("episode_id"),
            "success": int(bool(d.get("success"))),
            "total_reward": float(d.get("total_reward", 0.0)),
            "total_steps": int(d.get("total_steps", 0)),
            "difficulty": d.get("difficulty", "unknown"),
        })

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

