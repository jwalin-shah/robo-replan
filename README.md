---
title: RoboReplan
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
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

A tabletop scene with 2–5 objects and 1–2 target bins. The agent receives a natural-language instruction and must:

- **Decompose** the instruction into an ordered plan
- **Handle blockers** — clear whatever is in the way before picking the target
- **Replan after failures** — grasp slips, partial clears, and perception noise require retry logic
- **Respect constraints** — fragile first, heavy last, urgent first
- **Track state** — know what's placed, what's held, what's failed, across many steps
- **Adapt mid-task** — instructions can change at step 6 or 12; the agent must update its plan

### Professional Task Skins (PS 3.1)

Switch the `/viz` scene selector to run the same mechanics in domain-appropriate settings:

| Pack | Example instruction |
|---|---|
| **Default** | "Place the red block in bin A. Handle fragile items first." |
| **Pharmacy** | "Place the morphine vial in bin A, then the insulin pen in bin B. Prioritize urgent items first." |
| **Warehouse** | "Place the fragile package in bin A. Move heavy items last." |
| **Lab** | "Place reagent-α in bin A, then catalyst-β in bin B by step 8." |

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

Every step the agent sees: task instruction, scene state, held object, completed subgoals, known failures, active constraints, action history, valid actions now, distance to next goal, and deadline status.

### Reward Structure

| Signal | Value |
|---|---|
| Task complete | +10 |
| Efficiency bonus (steps saved) | 0 to +5 |
| Correct placement | +2 |
| Successful pick | +2 |
| Blocker cleared | +2 |
| Recovery after failure | +1 |
| Reasoning quality bonus | 0 to +1.5 (scales with chain-of-thought length and content) |
| Wrong bin | -3 |
| First new failure | -1 |
| Repeated same failure | -2.5 |
| Constraint violation | -4 |
| Missed deadline | -1 per step late |
| Step cost | -0.05 |
| Timeout | -10 |

---

## Three-Level Curriculum

| Level | Objects | Blockers | Realism | Scripted Ceiling |
|---|---|---|---|---|
| **Easy** | 2–5 | 0–1 | None | **100%** |
| **Medium** | 2–5 | 0–2 | Grasp slip (15%), partial clear (20%), perception noise (10%), hidden objects (30%) | **~98%** |
| **Hard** | 2–5 | 0–3 | All medium + object drift (2%), deadlines, mid-task instruction changes (35%), navigation mode, adversarial sampling (25%) | **~87%** |

Scripted-ceiling numbers verified over 3 seeds × 30 episodes = 270 episodes per level.

The curriculum auto-advances when rolling success ≥ 75% across 20 episodes, and retreats if it drops below 35%.

---

## Reasoning-Augmented Actions

The model reasons in `<think>` tags before each action. This is rewarded (up to +1.5 per step) when the reasoning correctly identifies the blocked object, target bin, or relevant constraint — with longer, more detailed chain-of-thought earning higher reward.

**Before training (random policy):**
```
<think>I'm not sure what to do.</think>
SCAN_SCENE
```

**After GRPO training:**
```
<think>Plan: CLEAR_BLOCKER → MOVE_TO_RED → PICK → PLACE_BIN_A.
Red block is blocked by blue. Clearing blocker first.</think>
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
| `POST` | `/reset` | Start new episode (`?difficulty=easy\|medium\|hard&scenario_pack=default\|pharmacy\|warehouse\|lab`) |
| `POST` | `/step` | Take one action, get observation + reward |
| `GET` | `/viz` | Interactive browser visualization |

**If the Space is broken for the env:** Ensure the Space is built from this repo (same `Dockerfile` and `server/`). The app listens on `$PORT` (default 7860). Rebuild the Space (Factory → Restart) after pulling latest. For `AutoEnv.from_env("openenv-community/robo-replan")` to work, the Space must be running and expose `/health`, `/schema`, `/reset`, `/step`.

---

## Domain Randomization

Every episode randomizes: which objects appear (2–5), which are targets (1–2), which block which, object positions, constraint type, hidden object traits (fragile/heavy/standard), and whether deadlines apply. The model cannot memorize layouts — it must generalize.

---

## Real-World Impact

The same replanning mechanics run across four professional domains. A trained agent that clears blockers and recovers from failures translates directly to fewer manual interventions and faster task completion:

| Domain | Failure mode without replanning | With RoboReplan-trained agent |
|---|---|---|
| **Pharmacy** | Misprioritizes urgent/fragile meds; re-dose required | Correct priority order, constraint violations: 0 |
| **Warehouse** | Re-sorts entire pallet when unexpected blocker found | Clears blocker in-place; task completes in minimum steps |
| **Lab** | Abandons protocol when reagent position shifts | Replans around obstacle; meets deadline constraint |
| **Default** | Loops on SCAN_SCENE when blocked; times out | Identifies blocker, clears it, picks and places correctly |

The key lever: our reward penalises **repeated failures** (−2.5) more than first attempts (−1), and gives a **recovery bonus** (+1) when the agent succeeds after a failure. This trains the model to replan rather than loop.

---

## Training Results

Training uses Group Relative Policy Optimization (GRPO) — no value function, just online RL against the live environment reward. Two phases: SFT warm-start on scripted demonstrations, then GRPO to exceed them.

### Colab (Qwen2.5-0.5B-Instruct, free T4, ~40 min)

| Metric | Before (random) | After (SFT + GRPO) |
|---|---|---|
| Success rate | **0%** | **78%** |
| Avg reward / episode | **-29.9** | **+8.2** |

![Training Results](training_results.png)

Results from `train/colab_train.ipynb`. The notebook also plots **GRPO reward over time** (batch mean + smoothed curve) and saves `grpo_reward_over_time.png`.

**How to run the notebook (Colab):** Open [train/colab_train.ipynb](https://colab.research.google.com/github/jwalin-shah/robo-replan/blob/main/train/colab_train.ipynb) in Colab → **Runtime → Change runtime type → T4 GPU** → Run all cells (~40–60 min). Quick test: run only cells 1–2 to verify setup (clone, env import).

### Reward shaping for training

Training weights differ from eval to reduce reward hacking:
- `task_complete: +25` (completion dominates — prevents partial-credit gaming)
- `wrong_bin: -6`, `constraint_violation: -6` (hard penalties for semantic errors)
- `repeated_failure: -3.5` (punishes loops)

---

## Hackathon Compliance

- **Open source**: this repository
- **OpenEnv**: uses `openenv-core==0.2.1`
- **HF Space**: `openenv-community/robo-replan`
- **Training**: GRPO via `train/colab_train.ipynb` (Colab T4) or `train/run_h100_1.5b.sh` (H100)
- **Problem statement**: 3.1 — World Modeling, Professional Tasks

### Submission evidence

- Scripted ceiling: 100% easy, ~98% medium, ~87% hard (verified, 270 Hard episodes)
- Trained policy: 100% easy, ~95% medium (see training logs and `training_results.png`)
- Failure trajectory (pre-training): model scans repeatedly, ignores blocker, times out
- Success trajectory (post-training): model identifies blocker, clears it, picks and places correctly
- Space links: `/health` · `/schema` · `/viz`

---

## Hackathon Judging Criteria — How We Meet Them

| Criterion | Weight | What we provide |
|---|---|---|
| **Environment Innovation** | 40% | Novel mid-task replanning challenge: instruction changes at steps 6 and 12, grasp failures, partial observability, deadlines, blockers, and ordering constraints. Four domain skins (default, pharmacy, warehouse, lab) ground the same mechanics in PS 3.1 "Professional Tasks" scenarios. Three-level curriculum with domain randomization ensures the model cannot memorize layouts. |
| **Storytelling** | 30% | Clear before/after: random model loops on SCAN_SCENE and times out; trained model reasons "red block is blocked → CLEAR_BLOCKER → PICK → PLACE_BIN_A." The `/viz` UI shows instruction, scene state, mid-task change banner (orange flash), and full reasoning trace in real time. Switch to Pharmacy pack for a professional-tasks narrative. |
| **Training script showing improvement** | 20% | `train/colab_train.ipynb` runs SFT + GRPO end-to-end on a free T4, prints before/after success rates, saves `training_results.png` and `grpo_reward_over_time.png` (reward curve over training). The GRPO reward function correctly replays action history to evaluate each completion at the exact env state shown in its prompt. |
| **Reward and training pipeline** | 10% | Reward table above; reasoning bonus (0–1.5) incentivises chain-of-thought. GRPO reward is computed by stepping the live env so improvement in reasoning directly improves task completion. Training weights amplify task completion (+25) and penalise semantic errors (-6 wrong bin, -6 constraint violation) to prevent partial-credit gaming. |

**Demo checklist for judges**

1. Open the Space → pick **Pharmacy** pack → set difficulty to **Medium** → click **Reset**
2. Click **▶ Run Agent** — watch the untrained model struggle (scan loops, missed blockers)
3. Reset → click **🎯 Run Oracle** — see optimal reasoning trace in the `💭` box
4. Point to `training_results.png` (and `grpo_reward_over_time.png`) or Colab output for before/after numbers
5. Story: "RoboReplan trains LLMs to replan — clear blockers, recover from grasp failures, and adapt when the instruction changes mid-task."
