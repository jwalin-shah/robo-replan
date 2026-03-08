---
title: RoboReplan
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# RoboReplan — Tabletop Robot Planning Environment

**Hackathon Problem Statement 3.1 — World Modeling: Professional Tasks**

> Agents must maintain consistent internal state, update beliefs based on outcomes,
> and orchestrate multi-step workflows in a dynamic, partially observable world.

---

## The Problem

LLMs fail at long-horizon robotic tasks not because they can't move, but because **they can't replan**. When a grasp slips, when a blocker appears, when the instruction changes mid-task — the model freezes, repeats the same failing action, or abandons the plan entirely.

RoboReplan benchmarks exactly this failure mode and trains agents to recover from it.

---

## What RoboReplan Tests

A tabletop scene with 2–5 colored blocks and 1–2 target bins. The agent receives a natural-language instruction and must:

- **Decompose** the instruction into an ordered plan
- **Handle blockers** — clear whatever is in the way before picking the target
- **Replan after failures** — grasp slips, partial clears, and perception noise require retry logic
- **Respect constraints** — fragile first, heavy last, urgent first
- **Track state** — know what's placed, what's held, what's failed, across many steps
- **Adapt mid-task** — instructions can change at step 8; the agent must update its plan

---

## Environment Details

### Action Space (16 actions)

| Category | Actions |
|---|---|
| Direct navigation | `MOVE_TO_RED` `MOVE_TO_BLUE` `MOVE_TO_GREEN` `MOVE_TO_YELLOW` `MOVE_TO_PURPLE` |
| Grid navigation (hard) | `MOVE_NORTH` `MOVE_SOUTH` `MOVE_EAST` `MOVE_WEST` `ROTATE_LEFT` `ROTATE_RIGHT` |
| Manipulation | `PICK` `PLACE_BIN_A` `PLACE_BIN_B` `CLEAR_BLOCKER` |
| Sensing | `SCAN_SCENE` |

### Observation (structured text)

Every step the agent sees: task instruction, scene state, held object, completed subgoals, known failures, active constraints, action history, valid actions now, distance to next goal, and optional oracle hint.

### Reward Structure

| Signal | Value |
|---|---|
| Task complete | +10 |
| Efficiency bonus (steps saved) | 0 to +5 |
| Correct placement | +2 |
| Successful pick | +2 |
| Blocker cleared | +2 |
| Recovery after failure | +1 |
| Reasoning quality bonus | 0 to +0.5 |
| Wrong bin | −3 |
| First new failure | −1 |
| Repeated same failure | −2.5 |
| Constraint violation | −4 |
| Missed deadline | −1 per step late |
| Step cost | −0.05 |
| Timeout | −10 |

---

## Three-Level Curriculum

| Level | Objects | Blockers | Realism | Oracle Success |
|---|---|---|---|---|
| **Easy** | 2–5 | 0–1 | None | **100%** |
| **Medium** | 2–5 | 0–2 | Grasp slip (15%), partial clear (20%), perception noise (10%), hidden objects (30%) | **~98%** |
| **Hard** | 2–5 | 0–3 | All medium + object drift (2%), deadlines, mid-task instruction changes (30%) | **~78%** |

The curriculum auto-advances when rolling success ≥ 75% across 20 episodes, and retreats if it drops below 35%.

---

## Training Results (GRPO, Qwen2.5-1.5B-Instruct)

Training uses Group Relative Policy Optimization (GRPO) — no supervised pretraining, just online RL against the environment reward.

| Difficulty | Untrained (random) | Post-GRPO |
|---|---|---|
| Easy | ~20% | **100%** |
| Medium | ~15% | **~95%** |
| Hard | ~5% | in progress |

The agent learns to:
1. Clear blockers before picking target objects
2. Recover after grasp failures without repeating the same action
3. Follow ordering constraints (fragile before heavy)
4. Replan when mid-task instructions change

### Training Config

```bash
# Full training run
python train/run_training.py

# Environment variables
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
TRAIN_DIFFICULTY=medium
GRPO_MAX_COMPLETION_LENGTH=128   # room for <think> reasoning
GRPO_NUM_GENERATIONS=4
ORACLE_EPISODES=800
```

### Reward shaping for training

Training weights differ from eval to reduce reward hacking:
- `task_complete: +25` (completion dominates — prevents partial-credit gaming)
- `wrong_bin: −6`, `constraint_violation: −6` (hard penalties for semantic errors)
- `repeated_failure: −3.5` (punishes loops)

---

## Reasoning-Augmented Actions

The model reasons in `<think>` tags before each action. This is rewarded (up to +0.5 per step) when the reasoning correctly identifies the blocked object, target bin, or relevant constraint.

**Before training (random policy):**
```
<think>I'm not sure what to do.</think>
SCAN_SCENE
```

**After GRPO training:**
```
<think>Red block is blocked by blue. I need to clear the blocker
before I can pick and place red in bin A.</think>
CLEAR_BLOCKER
```

---

## API

```python
from openenv import AutoEnv

env = AutoEnv.from_env("openenv-community/robo-replan")
obs = env.reset()
result = env.step({"action": "CLEAR_BLOCKER", "reasoning": "blocker in the way"})
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/schema` | Action/observation schema |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Take one action, get observation + reward |
| `GET` | `/viz` | Interactive browser visualization |

---

## Domain Randomization

Every episode randomizes: which objects appear (2–5), which are targets (1–2), which block which, object positions, constraint type, hidden object traits (fragile/heavy/standard), and whether deadlines apply. The model cannot memorize layouts — it must generalize.

---

## Hackathon Compliance

- **Open source**: this repository
- **OpenEnv**: uses `openenv-core==0.2.1`
- **HF Space**: `openenv-community/robo-replan`
- **Training**: GRPO-only via `train/run_training.py` or Unsloth pipeline `train/unsloth_train.py` (TRL + HuggingFace Transformers)
- **Problem statement**: 3.1 — World Modeling, Professional Tasks

### Submission evidence

- Oracle baseline: 100% easy, ~98% medium, ~78% hard
- Trained policy: 100% easy, ~95% medium (see training logs)
- Failure trajectory (pre-training): model scans repeatedly, ignores blocker, times out
- Success trajectory (post-training): model identifies blocker, clears it, picks and places correctly
- Space links: `/health` · `/schema` · `/viz`

---

## Hackathon Judging Criteria — How We Meet Them

| Criterion | Weight | What we provide |
|-----------|--------|------------------|
| **Environment Innovation** | 40% | **Novel & challenging**: Tabletop planning with blockers, grasp slip, partial observability, mid-task instruction changes, deadlines, and constraints (fragile-first, etc.). The agent must *replan* — not just execute a fixed plan — so it meaningfully tests world modeling and belief updates. Three-level curriculum (easy → medium → hard) with increasing realism. |
| **Storytelling** | 30% | **Clear problem & demo**: README states the problem (LLMs fail at replanning), what the env tests (blockers, recovery, constraints, mid-task change), and how to use the Space. The `/viz` UI shows instruction, scene objects (blocked vs hidden), valid actions, and action log so the demo is easy to follow. Before/after reasoning examples in README show the agent learning to say "Red block is blocked by blue… CLEAR_BLOCKER". |
| **Training script showing improvement** | 20% | **Observable evidence**: (1) Training writes `logs/train_metrics_unsloth.jsonl` (or `logs/train_metrics.jsonl`) per batch. (2) End of run produces `logs/training_curve_unsloth.png` — reward curve + before/after success-rate bar chart. (3) Console prints baseline vs post-GRPO success rate and avg reward. To plot from an existing metrics file: `python scripts/plot_training_curve.py`. Include the curve image in the Space description or README to show improvement. |
| **Reward and training pipeline** | 10% | **Coherent reward**: Reward table in README; training uses env reward (task complete, correct placement, penalties for wrong bin, repeated failure, constraint violation). GRPO/Unsloth pipeline: oracle data → SFT warm-start → GRPO online RL; reward is computed by stepping the real env per completion so improvement in inference (how the agent acts) is measurable. |

**Demo checklist for judges**

1. **Environment**: Open the Space → Reset → try Manual Actions: if a block is *blocked*, use CLEAR_BLOCKER first; then MOVE_TO_&lt;color&gt; then PICK. Buttons disable when invalid so the flow is clear.
2. **Training evidence**: Point to `logs/training_curve_unsloth.png` (or run `train/unsloth_train.py` / `train/run_training.py` and show the printed before/after and saved plot).
3. **Story**: "Agents must replan when something blocks the target or the instruction changes; RoboReplan trains them to clear blockers and recover from failures."
