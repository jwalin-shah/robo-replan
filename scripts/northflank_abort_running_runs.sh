#!/usr/bin/env bash
# Abort all active (running/pending) runs for the Northflank training job.
# Used so we can "close all runtimes" before starting a new job run.
#
# Usage: bash scripts/northflank_abort_running_runs.sh
# Env: NORTHFLANK_TOKEN, NORTHFLANK_PROJECT_ID (default: hackathon), NORTHFLANK_TRAIN_JOB_ID (default: robo-replan-train)

set -e
cd "$(dirname "$0")/.."
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)
[[ -n "$NORTHFLANK_TOKEN" ]] || { echo "NORTHFLANK_TOKEN in .env required"; exit 1; }

PROJECT="${NORTHFLANK_PROJECT_ID:-hackathon}"
JOB="${NORTHFLANK_TRAIN_JOB_ID:-robo-replan-train}"

RUNS_JSON=$(curl -s "https://api.northflank.com/v1/projects/$PROJECT/jobs/$JOB/runs?limit=20" \
  -H "Authorization: Bearer $NORTHFLANK_TOKEN")

# Abort any run that is not in a terminal state (success, failed, cancelled, aborted)
export NF_PROJECT="$PROJECT" NF_JOB="$JOB"
echo "$RUNS_JSON" | python3 -c "
import json, sys, urllib.request, os

d = json.load(sys.stdin)
runs = (d.get('data') or {}).get('runs') or []
token = os.environ.get('NORTHFLANK_TOKEN')
project = os.environ.get('NF_PROJECT', 'hackathon')
job = os.environ.get('NF_JOB', 'robo-replan-train')
terminal = {'success', 'failed', 'cancelled', 'aborted', 'completed'}

for r in runs:
    rid = r.get('id') or r.get('runId')
    status = (r.get('status') or '').lower()
    if not rid:
        continue
    if status in terminal:
        continue
    url = f'https://api.northflank.com/v1/projects/{project}/jobs/{job}/runs/{rid}'
    req = urllib.request.Request(url, method='DELETE', headers={'Authorization': f'Bearer {token}'})
    try:
        urllib.request.urlopen(req)
        print(f'Aborted run {rid} (status was {status})')
    except Exception as e:
        print(f'Failed to abort {rid}: {e}', file=sys.stderr)
        sys.exit(1)
"
echo "Active runs aborted (if any)."
