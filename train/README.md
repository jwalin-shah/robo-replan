# Training Scripts

Two-phase training: SFT warm-start on oracle rollouts → GRPO with live environment reward.

## Which script to use

| Script | Platform | Model | Notes |
|---|---|---|---|
| **`colab_train.ipynb`** | Google Colab (free T4) | Qwen2.5-0.5B-Instruct | Start here. SFT + GRPO, ~40 min. |
| **`run_h100_1.5b.sh`** | H100 / high-RAM machine | Qwen2.5-1.5B-Instruct | Runs `run_training.py` with larger model. |
| `run_training.py` | Any GPU | Configurable via env vars | Called by `run_h100_1.5b.sh`; also usable standalone. |
| `grpo_train.py` | Any GPU | Configurable | Standalone GRPO-only script (no SFT phase). |
| `trl_train.py` | Any GPU | Configurable | SFT-only baseline. |
| `unsloth_train.py` | H100 + Unsloth installed | Qwen2.5-7B | Faster large-model path; requires Unsloth. |

## Quickstart (Colab)

Click the badge in `colab_train.ipynb` or open it directly in Google Colab. Runtime: T4 (free tier) works for 0.5B; use Colab Pro or an H100 for 1.5B.

## Quickstart (H100)

```bash
cd /path/to/tabletop-planning-env
bash train/run_h100_1.5b.sh
```

Optional env vars:
```bash
ORACLE_EPISODES=400     # fewer SFT episodes for a faster run (default 1000)
FAST_MODE=1             # halves GRPO steps
ENABLE_SFT_WARMSTART=1  # explicitly enable SFT phase (default on)
USE_SIMPLE_REWARD=0     # use full reward shaping (default)
```

## What the training does

1. **Oracle data collection** — runs the scripted policy for `ORACLE_EPISODES` episodes to build an SFT dataset of (prompt, optimal_action) pairs.
2. **SFT warm-start** — fine-tunes the base model on oracle data so it learns the action space before RL begins.
3. **GRPO** — samples multiple completions per prompt, evaluates each against the live environment (replaying action history for correct state), and updates the policy toward higher-reward outputs.
4. **Evaluation** — prints before/after success rates and saves `training_results.png`.

## Key implementation detail

The GRPO `reward_fn` replays the episode's action history to reconstruct the exact environment state shown in each prompt before evaluating the model's completion. This ensures reward corresponds to the observation the model actually saw — not the env's initial state.
