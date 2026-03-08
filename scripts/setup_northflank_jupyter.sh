#!/usr/bin/env bash
# One-time setup: clone robo-replan and install deps on the Northflank Jupyter PyTorch service
# so you can open Jupyter in the browser and run train/colab_train.ipynb (Run All).
#
# Requires: Northflank CLI (npm i -g @northflank/cli), NORTHFLANK_TOKEN in .env
# Usage: bash scripts/setup_northflank_jupyter.sh

set -e
cd "$(dirname "$0")/.."
[[ -f .env ]] && export $(grep -v '^#' .env | xargs)
[[ -n "$NORTHFLANK_TOKEN" ]] || { echo "NORTHFLANK_TOKEN in .env required"; exit 1; }

PROJECT="${NORTHFLANK_PROJECT_ID:-robo-replan}"
SERVICE="${NORTHFLANK_JUPYTER_SERVICE_ID:-jupyter-pytorch}"

echo "Logging in to Northflank..."
northflank login -n robo-replan -t "$NORTHFLANK_TOKEN" 2>/dev/null || true

echo "Cloning repo and installing deps on Jupyter service (this may take a few min)..."
northflank exec service --projectId "$PROJECT" --serviceId "$SERVICE" --cmd 'bash -c "
  set -e
  cd /home/jovyan
  if [ -d robo-replan ]; then cd robo-replan && git fetch origin && git reset --hard origin/main; else git clone https://github.com/jwalin-shah/robo-replan.git && cd robo-replan; fi
  pip install -q \"trl==0.14.0\" \"vllm>=0.10.2,<0.13\" transformers torch datasets openenv-core==0.2.1 pydantic numpy accelerate
  echo DONE
"'

echo ""
echo "Setup done. Next:"
echo "  1. Open Jupyter: use the service URL (Networking) or run: northflank forward service --projectId $PROJECT --serviceId $SERVICE"
echo "  2. In Jupyter: open robo-replan/train/colab_train.ipynb and Run All"
echo ""
