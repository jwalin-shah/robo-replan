---
title: RoboReplan
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# RoboReplan — Tabletop Robot Planning Environment

An OpenEnv 0.2.1 environment for training LLMs to plan, replan, and recover in
physically grounded robotic manipulation tasks.

## What it is

The agent receives a natural-language instruction and a structured observation
of a tabletop scene, then chooses high-level actions to complete multi-step
manipulation tasks. The challenge is not low-level control — it's planning:

- Decomposing the instruction into subtasks
- Handling blocked objects (clear the blocker first)
- Replanning after failed actions (grasp slips, partial clears)
- Following constraints (fragile first, heavy last)
- Tracking state across multiple steps

## Why it matters

Current LLMs fail on long-horizon robotic tasks not because they can't move,
but because they can't maintain and revise a plan when things go wrong.
RoboReplan benchmarks exactly that failure mode.

## Environment details

- **Actions**: SCAN_SCENE | MOVE_TO_\* | PICK | PLACE_BIN_A | PLACE_BIN_B | CLEAR_BLOCKER
- **Observation**: structured text — instruction, scene state, failures, subgoals, constraints
- **Reward**: +10 task complete, +2 correct placement/pick/clear, −3 wrong bin, −2 repeated failure
- **Domain randomization**: object count (2–5), positions, targets, blockers, constraints all randomized
- **Realism levels**: easy / medium (grasp noise + partial observability) / hard (dynamic world)

## Hackathon problem statement

**Statement 3.1 — World Modeling**: Professional Tasks

> Agents must maintain consistent internal state, update beliefs based on outcomes,
> and orchestrate multi-step workflows in a dynamic, partially observable world.

## API

```python
from openenv import AutoEnv

env = AutoEnv.from_env("YOUR_HF_USERNAME/robo-replan")
obs = env.reset()
result = env.step({"action": "CLEAR_BLOCKER"})
```

### Endpoints

- `GET  /health` — liveness check
- `GET  /schema` — action/observation schema
- `POST /reset`  — start new episode
- `POST /step`   — take one action, get observation + reward
- `GET  /viz`    — interactive browser visualization (no server needed)

## Training

See `train/run_training.py` for the GRPO-only training script used in the Northflank run.

```bash
python train/run_training.py
```

## Hackathon Compliance

- Open source public repository: this repo.
- OpenEnv requirement: environment uses `openenv-core==0.2.1`.
- HF Space deployment: `openenv-community/robo-replan`.
- Minimal training script: HF TRL script in `train/run_training.py` (GRPO-only).
- Problem statement alignment: Statement 3.1 (World Modeling, Professional Tasks).
- Demo requirement: submit a one-minute YouTube video with environment behavior and training results.

### Evidence To Include In Submission

- Baseline metrics from untrained policy (success rate and average reward).
- Post-training metrics from trained policy on unseen episodes.
- One failure trajectory before training and one successful replanning trajectory after training.
- Space links:
  - `/health`
  - `/schema`
  - `/viz`
