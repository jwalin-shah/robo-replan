#!/usr/bin/env bash
# Overnight / quality training: either trigger the Northflank job with the right env,
# or print the env vars and command for Jupyter Terminal.
#
# Option 1 — Set env in Northflank Job UI, then run the job (see TRAIN_ON_NORTHFLANK.md).
# Option 2 — From your machine: trigger job (env must be set on the job in UI first).
# Option 3 — In Jupyter Terminal: nohup with these env vars (see below).

set -e
cd "$(dirname "$0")/.."

OVERNIGHT_ENV="ORACLE_EPISODES=1200 FAST_MODE=0 ENABLE_SFT_WARMSTART=1 MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct"

echo "Overnight / quality training env:"
echo "  $OVERNIGHT_ENV"
echo ""
echo "Northflank Job: set these in Job → Environment, then Runs → Run."
echo ""
echo "Jupyter Terminal (headless, survives browser close):"
echo "  cd /home/jovyan/robo-replan"
echo "  nohup env $OVERNIGHT_ENV python3 train/run_training.py > logs/overnight_train.log 2>&1 &"
echo "  tail -f logs/overnight_train.log   # to watch"
echo ""

# If NORTHFLANK_TOKEN set and user wants to trigger the job (env must already be set on job):
if [[ -n "$NORTHFLANK_TOKEN" ]] && [[ "${1:-}" == "trigger" ]]; then
  PROJECT="${NORTHFLANK_PROJECT_ID:-hackathon}"
  JOB="${NORTHFLANK_TRAIN_JOB_ID:-robo-replan-train}"
  echo "Triggering job run (ensure overnight env is set on job in UI)..."
  curl -s -X POST "https://api.northflank.com/v1/projects/$PROJECT/jobs/$JOB/runs" \
    -H "Authorization: Bearer $NORTHFLANK_TOKEN" \
    -H "Content-Type: application/json" -d '{}'
  echo ""
  echo "Check Runs and Logs in Northflank."
fi
