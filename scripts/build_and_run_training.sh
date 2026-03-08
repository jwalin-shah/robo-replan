#!/usr/bin/env bash
# One-shot: push, trigger Northflank build, then (optionally) wait and trigger run.
# Usage:
#   bash scripts/build_and_run_training.sh           # push + build + run when build succeeds (waits up to 15 min)
#   bash scripts/build_and_run_training.sh --no-wait # push + build + run immediately (run may fail if build not done)
#   SKIP_PUSH=1 bash scripts/build_and_run_training.sh  # don't push, just build + run (use current origin/main)

set -e
cd "$(dirname "$0")/.."
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)
[[ -n "$NORTHFLANK_TOKEN" ]] || { echo "NORTHFLANK_TOKEN in .env required"; exit 1; }

PROJECT="${NORTHFLANK_PROJECT_ID:-hackathon}"
JOB="${NORTHFLANK_TRAIN_JOB_ID:-robo-replan-train}"
WAIT_FOR_BUILD=true
for x in "$@"; do [[ "$x" == "--no-wait" ]] && WAIT_FOR_BUILD=false; done

if [[ -z "$SKIP_PUSH" ]]; then
  echo "Pushing to origin main..."
  git push origin main
fi

SHA=$(git rev-parse origin/main)
echo "Triggering build for $JOB @ ${SHA:0:12}..."
BUILD_JSON=$(curl -s -X POST "https://api.northflank.com/v1/projects/$PROJECT/jobs/$JOB/build" \
  -H "Authorization: Bearer $NORTHFLANK_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"sha\":\"$SHA\"}")

BUILD_ID=$(echo "$BUILD_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('id',''))" 2>/dev/null)
if [[ -z "$BUILD_ID" ]]; then
  echo "Build trigger failed: $BUILD_JSON"
  exit 1
fi
echo "Build started: $BUILD_ID"

if $WAIT_FOR_BUILD; then
  echo "Waiting for build to succeed (up to 15 min)..."
  for i in $(seq 1 30); do
    sleep 30
    STATUS=$(curl -s "https://api.northflank.com/v1/projects/$PROJECT/jobs/$JOB/build/$BUILD_ID" \
      -H "Authorization: Bearer $NORTHFLANK_TOKEN" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('status',''))" 2>/dev/null)
    echo "  [$i] build status: $STATUS"
    if [[ "$STATUS" == "SUCCESS" ]]; then
      echo "Build succeeded."
      break
    fi
    if [[ "$STATUS" == "FAILED" ]] || [[ "$STATUS" == "FAILURE" ]]; then
      echo "Build failed. Check Northflank job builds."
      exit 1
    fi
  done
fi

echo "Aborting any active runs (close runtimes before new job)..."
bash "$(dirname "$0")/northflank_abort_running_runs.sh" || true
echo "Waiting 5s for runtimes to close..."
sleep 5

echo "Triggering run..."
RUN_JSON=$(curl -s -X POST "https://api.northflank.com/v1/projects/$PROJECT/jobs/$JOB/runs" \
  -H "Authorization: Bearer $NORTHFLANK_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}')
RUN_ID=$(echo "$RUN_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('data',{}).get('id',''))" 2>/dev/null)
if [[ -n "$RUN_ID" ]]; then
  echo "Run started: $RUN_ID"
  echo "Logs: https://app.northflank.com → $PROJECT → Jobs → $JOB → Runs"
else
  echo "Run failed (build may still be in progress): $RUN_JSON"
  exit 1
fi
