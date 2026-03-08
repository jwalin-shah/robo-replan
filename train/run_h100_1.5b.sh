#!/usr/bin/env bash
# Run GRPO training on H100 (or any GPU) with 1.5B model — no Unsloth, so no GRPO crash.
# Use this to get a full run + reward curve for hackathon training evidence.
#
# From repo root:
#   bash train/run_h100_1.5b.sh
#
# Or with custom options:
#   ORACLE_EPISODES=400 FAST_MODE=1 bash train/run_h100_1.5b.sh

set -e
cd "$(dirname "$0")/.."
mkdir -p logs outputs

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
export TRAIN_DIFFICULTY="${TRAIN_DIFFICULTY:-easy}"
export METRICS_JSONL="${METRICS_JSONL:-./logs/train_metrics.jsonl}"
# Optional: SFT warm-start then GRPO (slower but often better)
export ENABLE_SFT_WARMSTART="${ENABLE_SFT_WARMSTART:-0}"
# Optional: faster run (smaller batches, fewer gens)
export FAST_MODE="${FAST_MODE:-0}"

echo "Running GRPO (TRL) with model=$MODEL_NAME difficulty=$TRAIN_DIFFICULTY"
echo "Metrics → $METRICS_JSONL"
python train/run_training.py

echo ""
echo "To plot the curve: python scripts/plot_training_curve.py --metrics $METRICS_JSONL --out logs/training_curve.png"
