#!/usr/bin/env bash
# Quick status check for the Northflank training job (build + runs).
# Usage: bash scripts/northflank_train_status.sh

set -e
cd "$(dirname "$0")/.."
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)

PROJECT=hackathon
JOB=robo-replan-train
BUILD_ID="${1:-fearless-fall-7630}"

echo "=== Build $BUILD_ID ==="
curl -s "https://api.northflank.com/v1/projects/$PROJECT/jobs/$JOB/build/$BUILD_ID" \
  -H "Authorization: Bearer ${NORTHFLANK_TOKEN}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'data' in d:
    b=d['data']
    print('  Status:', b.get('status'), '| Concluded:', b.get('concluded'), '| Created:', (b.get('createdAt') or '')[:19])
else:
    print('  ', d.get('error', d))
"

echo ""
echo "=== Recent runs ==="
curl -s "https://api.northflank.com/v1/projects/$PROJECT/jobs/$JOB/runs?limit=3" \
  -H "Authorization: Bearer ${NORTHFLANK_TOKEN}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
runs = (d.get('data') or {}).get('runs') or []
for r in runs[:3]:
    print('  Run', r.get('id'), '|', r.get('status'), '|', (r.get('createdAt') or '')[:19])
if not runs:
    print('  No runs yet (run starts after build succeeds)')
"

echo ""
echo "Live: https://app.northflank.com → project $PROJECT → Jobs → $JOB"
