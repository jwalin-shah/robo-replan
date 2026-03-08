#!/usr/bin/env python3
"""
Plot GRPO training curve from logs/train_metrics_unsloth.jsonl (or train_metrics.jsonl).
Use this to generate the "observable evidence" for hackathon judging (Training Script Showing Improvement).

Usage:
  python scripts/plot_training_curve.py
  python scripts/plot_training_curve.py --metrics logs/train_metrics.jsonl --out logs/curve.png
"""
import argparse
import json
import os

def main():
    p = argparse.ArgumentParser(description="Plot training reward curve from metrics JSONL")
    p.add_argument("--metrics", default="logs/train_metrics_unsloth.jsonl", help="Path to metrics JSONL")
    p.add_argument("--out", default="logs/training_curve.png", help="Output image path")
    args = p.parse_args()

    if not os.path.isfile(args.metrics):
        print(f"No metrics file at {args.metrics}. Run training first (e.g. train/unsloth_train.py).")
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return 1

    calls, means, stds = [], [], []
    with open(args.metrics) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            calls.append(row.get("call", len(calls) + 1))
            means.append(row["batch_mean"])
            stds.append(row.get("batch_std", 0))

    if not calls:
        print("No data in metrics file.")
        return 1

    fig, ax = plt.subplots(figsize=(8, 4))
    lo = [m - s for m, s in zip(means, stds)]
    hi = [m + s for m, s in zip(means, stds)]
    ax.fill_between(calls, lo, hi, alpha=0.2, color="#4a8adc")
    ax.plot(calls, means, color="#4a8adc", linewidth=2, label="batch mean ± std")
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("GRPO reward-function calls")
    ax.set_ylabel("Mean batch reward")
    ax.set_title("RoboReplan · GRPO training reward curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {args.out}")
    return 0

if __name__ == "__main__":
    exit(main())
