# Hackathon Win Strategy

## Quick start: training on H100 with 1.5B

**Yes — use the H100 with the smaller 1.5B model first.** That path uses standard TRL (no Unsloth), so it completes and gives you a reward curve for the 20% training-evidence criterion.

1. **On the H100 machine** (e.g. Northflank), clone the repo and run from repo root:
   ```bash
   cd ~/tabletop-planning-env   # or your clone path
   bash train/run_h100_1.5b.sh
   ```
2. **After the run:** plot the curve (on the same machine or copy `logs/train_metrics.jsonl` locally):
   ```bash
   python scripts/plot_training_curve.py --metrics logs/train_metrics.jsonl --out logs/training_curve.png
   ```
3. **Add** `logs/training_curve.png` and the printed "Before vs After" stats to your Space/README.

Optional: `ENABLE_SFT_WARMSTART=1 bash train/run_h100_1.5b.sh` for SFT then GRPO; `FAST_MODE=1` or `ORACLE_EPISODES=400` for a quicker run.

---

## H100 training status (Northflank)

- **Completed:** Oracle collection (2k episodes), model load (Qwen2.5-7B), baseline eval (20% success), **SFT warm-start**.
- **Failed:** GRPO phase crashed with `TypeError: not all arguments converted during string formatting` inside Unsloth’s patched `UnslothGRPOTrainer` (their cache, not our `unsloth_train.py`). So **no GRPO steps ran**, no `train_metrics_unsloth.jsonl` rows, no final model.

**What to do for training evidence (20%):**

1. **Option A — H100 + 1.5B (recommended):**  
   Use the **H100 with the 1.5B model** via `train/run_training.py` (no Unsloth) so GRPO runs without the Unsloth crash. From repo root:
   ```bash
   bash train/run_h100_1.5b.sh
   ```
   Or manually:
   ```bash
   cd /path/to/tabletop-planning-env
   export MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
   python train/run_training.py
   ```
   Metrics go to `logs/train_metrics.jsonl`. Then:
   ```bash
   python scripts/plot_training_curve.py --metrics logs/train_metrics.jsonl --out logs/training_curve.png
   ```
   Use the printed before/after stats and `logs/training_curve.png` in the Space/README.

   **Faster run (optional):** `FAST_MODE=1 bash train/run_h100_1.5b.sh` or `ORACLE_EPISODES=400 bash train/run_h100_1.5b.sh`.

2. **Option A2 — Colab (no H100):**  
   Use `train/colab_train.ipynb` on Colab T4 with 0.5B (or 1.5B if you have more RAM). Clone the Space repo and run the notebook; it uses standard TRL SFT + GRPO and writes metrics you can plot locally.

3. **Option B — Fix or work around Unsloth GRPO:**  
   Debug the Unsloth trainer (their `UnslothGRPOTrainer` in `unsloth_compiled_cache`) or try a different Unsloth/TRL version so GRPO runs without the formatting error. Then re-run `train/unsloth_train.py` on the H100.

4. **Option C — Use existing evidence:**  
   If you already have a past run that produced `logs/train_metrics*.jsonl`, run `python scripts/plot_training_curve.py --metrics logs/train_metrics.jsonl --out logs/training_curve.png` and add the generated image to the Space description or README.

---

## Is the environment good enough? (40% — Environment Innovation)

**Yes, if you frame it clearly.** The rubric asks: *Is the environment novel, creative, or challenging? Does it meaningfully test the agent’s behavior?*

**What you have that stands out:**

- **Replanning, not just planning** — Blockers, grasp slip, partial clear, mid-task instruction change, deadlines. The agent must *update its plan* when the world or the task changes.
- **World modeling** — Partially observable (hidden traits, scan-before-pick), stateful (history, failures, subgoals), and multi-step.
- **Curriculum** — Easy → medium → hard with increasing realism and difficulty.
- **Rich action space** — 16 actions (direct + nav + manipulation + sensing), constraints (fragile first, etc.).

**One-liner for judges:**  
*“RoboReplan tests whether the agent can **replan** when the plan breaks — blockers, slips, and mid-task changes — not just follow a fixed sequence.”*

**To lock in the 40%:**

1. **Say “replan” and “world modeling” explicitly** in the Space description and README (you already do; double-check the first paragraph).
2. **Show one concrete failure mode in the demo:** e.g. “If the agent doesn’t clear the blocker first, it can never pick the target; that’s the replanning challenge.”
3. **Mention what others don’t:** e.g. “Most benchmarks test plan execution; we test **recovery and replanning** under failures and instruction change.”

---

## How to lock in the other 60%

| Criterion | Weight | Lock-in |
|-----------|--------|--------|
| **Storytelling** | 30% | Use PITCH.md: problem (can’t replan) → demo (viz, one episode) → training result (before/after). Keep the Space description and README aligned with that story. |
| **Training improvement** | 20% | Get **one** full run that prints baseline vs post-GRPO and, if possible, saves a reward curve (run_training or colab; or plot from existing metrics). Put the curve or a clear “before/after” table in the Space or README. |
| **Reward / pipeline** | 10% | README already documents reward. In the demo or doc, add one sentence: “Reward penalizes repeated failures and rewards recovery, so the agent learns to replan instead of loop.” |

---

## Pre-submission checklist

- [ ] **Space/viz:** Reset works; manual play: CLEAR_BLOCKER (if blocked) → MOVE_TO_&lt;color&gt; → PICK. Buttons reflect valid actions.
- [ ] **README:** First paragraph states “replan” and “world modeling”; “Hackathon Judging Criteria” section is visible; training evidence (curve or before/after) is linked or inlined.
- [ ] **Training evidence:** At least one of: (a) reward curve image in Space/README, (b) before/after success rate in README or Space, (c) colab/run_training run that completes and logs metrics.
- [ ] **One-liner:** “We test replanning under blockers, slips, and mid-task changes — not just plan execution.”

If the H100 Unsloth GRPO path is fixed later, you can re-run and replace the training evidence with that curve; the environment and story already support a strong score.
