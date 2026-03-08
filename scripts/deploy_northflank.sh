#!/usr/bin/env bash
# Deploy from here: push to GitHub (if needed) and trigger Northflank build.
# Requires: .env with NORTHFLANK_TOKEN, and origin/main up to date.
#
# Usage: bash scripts/deploy_northflank.sh
#        SKIP_PUSH=1 bash scripts/deploy_northflank.sh   # only trigger build, don't push

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
SERVICE_ID="${NORTHFLANK_SERVICE_ID:-robo-replan}"

if [[ -z "$SKIP_PUSH" ]]; then
  echo "Pushing to origin main..."
  git push origin main
fi

SHA=$(git rev-parse origin/main)
echo "Triggering Northflank build for $SERVICE_ID @ $SHA"
RES=$(curl -s -X POST "https://api.northflank.com/v1/projects/$PROJECT_ID/services/$SERVICE_ID/build" \
  -H "Authorization: Bearer $NORTHFLANK_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"sha\":\"$SHA\"}")

if echo "$RES" | grep -q '"id":'; then
  BUILD_ID=$(echo "$RES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('data',{}).get('id',''))")
  echo "Build started: $BUILD_ID"
  echo "Monitor at: https://app.northflank.com (project: $PROJECT_ID, service: $SERVICE_ID)"
else
  echo "Build trigger failed: $RES"
  exit 1
fi
