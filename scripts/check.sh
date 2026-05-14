#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/logs/episodes.jsonl"
TMP_DIR="$(mktemp -d)"
HAD_LOG=0

cleanup() {
  if [[ "$HAD_LOG" == "1" ]]; then
    cp "$TMP_DIR/episodes.jsonl" "$LOG"
  else
    rm -f "$LOG"
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if [[ -f "$LOG" ]]; then
  HAD_LOG=1
  cp "$LOG" "$TMP_DIR/episodes.jsonl"
fi

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/logs"
python3 scripts/smoke_env.py
python3 scripts/check_invariants.py
python3 scripts/check_arch_contracts.py
python3 scripts/validate_evidence_artifacts.py
python3 -m compileall server scripts train
git diff --check
