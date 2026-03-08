#!/usr/bin/env bash
# Trigger a training run on Northflank (H100). The job must already exist and use Dockerfile.train.
# Create the job once in Northflank (see TRAIN_ON_NORTHFLANK.md), then run this script.
#
# Usage: bash scripts/run_northflank_training.sh

set -e
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "Missing .env (need NORTHFLANK_TOKEN)."
  exit 1
fi
export $(grep -v '^#' .env | xargs)
if [[ -z "$NORTHFLANK_TOKEN" ]]; then
  echo "NORTHFLANK_TOKEN not set in .env"
  exit 1
fi

PROJECT_ID="${NORTHFLANK_PROJECT_ID:-hackathon}"
JOB_ID="${NORTHFLANK_TRAIN_JOB_ID:-robo-replan-train}"

echo "Aborting any active runs for $JOB_ID..."
bash "$(dirname "$0")/northflank_abort_running_runs.sh" || true
echo "Waiting 5s for runtimes to close..."
sleep 5

echo "Starting new training job run: $JOB_ID (project: $PROJECT_ID)"
RES=$(curl -s -X POST "https://api.northflank.com/v1/projects/$PROJECT_ID/jobs/$JOB_ID/runs" \
  -H "Authorization: Bearer $NORTHFLANK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}')

if echo "$RES" | grep -q '"id":\|"runId":\|"data":'; then
  echo "Job run started. Check progress in Northflank:"
  echo "  https://app.northflank.com → project $PROJECT_ID → Jobs → $JOB_ID → Runs"
else
  echo "Failed to start run: $RES"
  echo "Make sure the job '$JOB_ID' exists and has at least one successful build (Dockerfile.train)."
  exit 1
fi
