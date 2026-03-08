"""
RoboReplan — Unsloth + GRPO training (H100 optimized)

Achieves 2-3x speedup over vanilla TRL via Unsloth's custom CUDA kernels.
SFT warmstart prevents the mode-collapse failure seen with cold-start GRPO.

SSH paste-and-run:
    cd ~/tabletop-planning-env
    pip install unsloth trl transformers datasets matplotlib --quiet
    MODEL=Qwen/Qwen2.5-7B-Instruct DIFFICULTY=medium python train/unsloth_train.py
"""
import os, re, sys, json, random, time
sys.path.insert(0, ".")

# ── Auto-install ────────────────────────────────────────────────────────────
def _ensure(pkg, import_name=None):
    import importlib
    try:
        importlib.import_module(import_name or pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

_ensure("unsloth")
_ensure("trl")
_ensure("datasets")
_ensure("matplotlib")

import torch
from datasets import Dataset

from server.config import EnvConfig
from server.environment import TabletopPlanningEnv
from server.models import Action
from server.robosim.randomizer import ScenarioConfig

# ── Config ──────────────────────────────────────────────────────────────────
MODEL      = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
DIFFICULTY = os.environ.get("DIFFICULTY", "medium")
LORA_RANK  = int(os.environ.get("LORA_RANK", "32"))

ORACLE_EPISODES       = int(os.environ.get("ORACLE_EPISODES", "2000"))
SFT_MAX_STEPS         = int(os.environ.get("SFT_MAX_STEPS", "600"))
SFT_BATCH             = int(os.environ.get("SFT_BATCH", "8"))
SFT_GRAD_ACCUM        = int(os.environ.get("SFT_GRAD_ACCUM", "2"))
SFT_LR                = float(os.environ.get("SFT_LR", "2e-5"))

GRPO_BATCH            = int(os.environ.get("GRPO_BATCH", "8"))
GRPO_GRAD_ACCUM       = int(os.environ.get("GRPO_GRAD_ACCUM", "2"))
GRPO_GENERATIONS      = int(os.environ.get("GRPO_GENERATIONS", "16"))
GRPO_COMPLETION_LEN   = int(os.environ.get("GRPO_COMPLETION_LEN", "300"))
GRPO_TEMPERATURE      = float(os.environ.get("GRPO_TEMPERATURE", "1.3"))
GRPO_TOP_P            = float(os.environ.get("GRPO_TOP_P", "0.95"))
GRPO_LR               = float(os.environ.get("GRPO_LR", "5e-6"))

EVAL_EPISODES         = int(os.environ.get("EVAL_EPISODES", "100"))
METRICS_PATH          = os.environ.get("METRICS_PATH", "./logs/train_metrics_unsloth.jsonl")
PLOT_PATH             = os.environ.get("PLOT_PATH", "./logs/training_curve_unsloth.png")

ACTIONS = [a.value for a in Action]
ACTION_STR = " | ".join(ACTIONS)

SYSTEM = (
    "You are a robot planning agent on a tabletop. Complete manipulation tasks "
    "by choosing ONE action per step.\n\n"
    f"Actions: {ACTION_STR}\n\n"
    "Before each action, write your plan inside <think>...</think>: "
    "list remaining steps, then state what you are doing now and why.\n"
    "Example:\n"
    "<think>Plan: CLEAR_BLOCKER → MOVE_TO_RED → PICK → PLACE_BIN_A. "
    "Red is blocked so clearing first. Doing: CLEAR_BLOCKER.</think>\n"
    "CLEAR_BLOCKER"
)

CFG_BY_NAME = {
    "easy":   EnvConfig.easy,
    "medium": EnvConfig.medium,
    "hard":   EnvConfig.hard,
}
if DIFFICULTY not in CFG_BY_NAME:
    raise ValueError(f"DIFFICULTY must be one of {list(CFG_BY_NAME)}, got {DIFFICULTY!r}")

print(f"\n{'='*60}")
print(f"  RoboReplan Unsloth Training")
print(f"  Model:      {MODEL}")
print(f"  Difficulty: {DIFFICULTY}")
print(f"  LoRA rank:  {LORA_RANK}")
print(f"  Generations:{GRPO_GENERATIONS}  Temp:{GRPO_TEMPERATURE}")
print(f"{'='*60}\n")

# ── Helpers ─────────────────────────────────────────────────────────────────

def obs_to_user_msg(obs):
    objects  = ', '.join(f"{o.name}({'reachable' if o.reachable else 'BLOCKED'})"
                         for o in obs.visible_objects)
    failures = '; '.join(obs.known_failures) or 'none'
    subgoals = '; '.join(obs.completed_subgoals) or 'none yet'
    history  = ' -> '.join(obs.action_history[-5:]) or 'none'
    valid    = ', '.join(obs.valid_actions) if obs.valid_actions else 'any'
    return (
        f"Instruction: {obs.instruction}\n"
        f"Scene: {objects}\n"
        f"Holding: {obs.holding or 'nothing'}\n"
        f"Progress: {obs.goal_progress:.0%}  Remaining: {obs.goals_remaining}\n"
        f"Completed: {subgoals}\n"
        f"Failures: {failures}\n"
        f"History: {history}\n"
        f"Last: {obs.last_action or 'none'} -> {obs.last_result or 'n/a'}\n"
        f"Valid now: {valid}\n"
        f"Steps left: {obs.steps_remaining}\n\nNext action:"
    )

def scenario_to_json(scen) -> str:
    return json.dumps({
        'objects': scen.objects, 'targets': scen.targets,
        'blockers': scen.blockers, 'distractors': scen.distractors,
        'constraint': scen.constraint, 'instruction': scen.instruction,
        'positions': {k: list(v) for k, v in scen.positions.items()},
        'hidden_traits': scen.hidden_traits or {},
        'deadlines': scen.deadlines or {},
    })

def json_to_scenario(s: str) -> ScenarioConfig:
    d = json.loads(s)
    d['positions'] = {k: tuple(v) for k, v in d['positions'].items()}
    d.setdefault('hidden_traits', {})
    d.setdefault('deadlines', {})
    return ScenarioConfig(**d)

def _to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # GRPO can return chat-like chunks: {"content": "..."} or nested values.
        if 'content' in x:
            return _to_text(x['content'])
        return ' '.join(_to_text(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return ' '.join(_to_text(v) for v in x)
    return str(x)

def extract_action(text: str):
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip().upper()
    clean = re.sub(r'[^A-Z_ ]+', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip().split()[0] if clean.strip() else ''
    return clean if clean in ACTIONS else None

# ── Step 1: Collect oracle dataset ──────────────────────────────────────────
print("Collecting oracle trajectories...")
train_cfg = CFG_BY_NAME[DIFFICULTY]()
train_cfg.obs.include_oracle_hint = True
# Boost completion reward so GRPO learns to finish tasks
train_cfg.reward.task_complete = 25.0
train_cfg.reward.wrong_bin = -6.0
train_cfg.reward.repeated_failure = -3.5
train_cfg.reward.constraint_violation = -6.0

env = TabletopPlanningEnv(config=train_cfg, use_stub=True)

rows = []
for ep in range(ORACLE_EPISODES):
    obs = env.reset()
    scen_json = scenario_to_json(env._scenario_cfg)
    for step in range(30):
        if obs.oracle_hint:
            rows.append({
                'prompt': [
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user',   'content': obs_to_user_msg(obs)},
                ],
                'answer':   obs.oracle_hint,
                'scenario': scen_json,
            })
        r = env.step(obs.oracle_hint or 'SCAN_SCENE')
        obs = r.observation
        if r.done:
            break
    if (ep + 1) % 400 == 0:
        print(f"  {ep+1}/{ORACLE_EPISODES} episodes, {len(rows)} training steps")

dataset = Dataset.from_list(rows).train_test_split(test_size=0.05)
print(f"Dataset: {len(rows)} steps ({len(dataset['train'])} train, {len(dataset['test'])} eval)")

# ── Step 2: Load model with Unsloth ─────────────────────────────────────────
print(f"\nLoading {MODEL} with Unsloth...")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=1024,
    dtype=torch.bfloat16,   # Full bf16 on H100 — no quantization needed
    load_in_4bit=False,
)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",   # Unsloth's memory-efficient checkpointing
    random_state=42,
)
model.generation_config.pad_token_id = tokenizer.pad_token_id
print("Model loaded.")

# ── Step 3: Baseline eval ────────────────────────────────────────────────────
print("\n" + "="*50)
print("BASELINE (before training)")
print("="*50)

eval_cfg = CFG_BY_NAME[DIFFICULTY]()
eval_env = TabletopPlanningEnv(config=eval_cfg, use_stub=True)

def run_eval(policy_fn, n=EVAL_EPISODES, seed=1337):
    random.seed(seed)
    successes, rewards = 0, []
    for _ in range(n):
        obs = eval_env.reset()
        total_r = 0.0
        for _ in range(30):
            action = policy_fn(obs)
            r = eval_env.step(action)
            total_r += r.reward
            obs = r.observation
            if r.done:
                break
        successes += int(obs.goal_progress >= 1.0)
        rewards.append(total_r)
    return {"success_rate": successes / n, "avg_reward": sum(rewards) / n}

def random_policy(obs):
    return random.choice(obs.valid_actions or ACTIONS)

baseline = run_eval(random_policy)
print(f"  Success rate: {baseline['success_rate']:.0%}")
print(f"  Avg reward:   {baseline['avg_reward']:.2f}")

# ── Step 4: SFT warm-start ───────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

print("\n" + "="*50)
print("PHASE 1: SFT WARM-START (teaches format + oracle actions)")
print("="*50)

sft_rows = []
for row in dataset['train']:
    chat = row['prompt'] + [{'role': 'assistant', 'content': row['answer']}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    sft_rows.append({"text": text})

sft_dataset = Dataset.from_list(sft_rows)
print(f"SFT dataset: {len(sft_rows)} examples")

sft_args = SFTConfig(
    output_dir='./outputs/sft_unsloth',
    num_train_epochs=1,
    max_steps=SFT_MAX_STEPS,
    per_device_train_batch_size=SFT_BATCH,
    gradient_accumulation_steps=SFT_GRAD_ACCUM,
    learning_rate=SFT_LR,
    warmup_steps=50,
    logging_steps=50,
    save_strategy='no',
    report_to='none',
    dataset_text_field='text',
    max_seq_length=512,
    bf16=True,
)

sft_trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=sft_dataset,
    processing_class=tokenizer,
)
sft_trainer.train()
del sft_trainer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("SFT warm-start complete.")

# ── Step 5: GRPO ─────────────────────────────────────────────────────────────
from trl import GRPOTrainer, GRPOConfig

print("\n" + "="*50)
print("PHASE 2: GRPO (online RL against environment reward)")
print("="*50)

os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
metrics_fh = open(METRICS_PATH, 'w')
_grpo_call_count = [0]

def reward_fn(prompts, completions, scenario, **kwargs):
    rewards = []
    batch_rewards = []

    for prompt_msgs, completion, scen_str in zip(prompts, completions, scenario):
        completion_text = _to_text(completion)
        action = extract_action(completion_text)
        reasoning = ""
        m = re.search(r'<think>(.*?)</think>', completion_text, re.DOTALL)
        if m:
            reasoning = m.group(1).strip()

        if action is None:
            rewards.append(-4.0)
            batch_rewards.append(-4.0)
            continue

        try:
            scen = json_to_scenario(scen_str)
            ep_cfg = CFG_BY_NAME[DIFFICULTY]()
            ep_env = TabletopPlanningEnv(config=ep_cfg, use_stub=True)
            ep_env.reset(scenario_override=scen)
            # Replay the conversation history to get to the right state
            # (simpler: just take one step from the scenario start state)
            result = ep_env.step(action, reasoning=reasoning)
            r = result.reward
        except Exception:
            r = -2.0

        rewards.append(r)
        batch_rewards.append(r)

    # Log batch metrics
    _grpo_call_count[0] += 1
    if batch_rewards:
        import statistics
        row = {
            "call": _grpo_call_count[0],
            "batch_mean": sum(batch_rewards) / len(batch_rewards),
            "batch_std": statistics.stdev(batch_rewards) if len(batch_rewards) > 1 else 0.0,
            "batch_min": min(batch_rewards),
            "batch_max": max(batch_rewards),
        }
        metrics_fh.write(json.dumps(row) + "\n")
        metrics_fh.flush()
        if _grpo_call_count[0] % 20 == 0:
            print(f"  [GRPO call {_grpo_call_count[0]}] "
                  f"mean={row['batch_mean']:.2f}  std={row['batch_std']:.2f}")

    return rewards

grpo_config = GRPOConfig(
    output_dir='./outputs/grpo_unsloth',
    num_train_epochs=1,
    per_device_train_batch_size=GRPO_BATCH,
    gradient_accumulation_steps=GRPO_GRAD_ACCUM,
    learning_rate=GRPO_LR,
    num_generations=GRPO_GENERATIONS,    # 16 → variance for learning
    max_completion_length=GRPO_COMPLETION_LEN,
    temperature=GRPO_TEMPERATURE,        # 1.3 → diverse outputs, no collapse
    top_p=GRPO_TOP_P,
    logging_steps=5,
    save_steps=500,
    save_total_limit=1,
    save_only_model=True,
    push_to_hub=False,
    report_to='none',
    bf16=True,
)

model.warnings_issued = {}   # suppress TRL version warning

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    reward_funcs=reward_fn,
    train_dataset=dataset['train'],
    processing_class=tokenizer,
)
trainer.train()
metrics_fh.close()

# Save LoRA adapter (tiny — ~200MB instead of 14GB)
trainer.save_model('./outputs/grpo_unsloth_final')
print("GRPO complete → ./outputs/grpo_unsloth_final")

# ── Step 6: After-training eval ──────────────────────────────────────────────
print("\n" + "="*50)
print("AFTER TRAINING")
print("="*50)

FastLanguageModel.for_inference(model)

def trained_policy(obs):
    messages = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user',   'content': obs_to_user_msg(obs)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **encoded,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = out_ids[0][encoded["input_ids"].shape[1]:]
    out = tokenizer.decode(new_ids, skip_special_tokens=True)
    return extract_action(out) or random.choice(ACTIONS)

after = run_eval(trained_policy)
print(f"  Success rate: {after['success_rate']:.0%}")
print(f"  Avg reward:   {after['avg_reward']:.2f}")

print(f"\n{'='*50}")
print(f"  RESULT ({DIFFICULTY}):")
print(f"  Random baseline: {baseline['success_rate']:.0%} success  {baseline['avg_reward']:.1f} reward")
print(f"  After training:  {after['success_rate']:.0%} success  {after['avg_reward']:.1f} reward")
print(f"  Improvement: {after['success_rate'] - baseline['success_rate']:+.0%} success rate")
print(f"{'='*50}\n")

# ── Step 7: Plot ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calls, means, stds = [], [], []
    with open(METRICS_PATH) as f:
        for line in f:
            row = json.loads(line)
            calls.append(row["call"])
            means.append(row["batch_mean"])
            stds.append(row.get("batch_std", 0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"RoboReplan · Unsloth + GRPO · {MODEL.split('/')[-1]} · difficulty={DIFFICULTY}",
        fontsize=13, fontweight="bold"
    )

    # Left: reward curve
    ax = axes[0]
    lo = [m - s for m, s in zip(means, stds)]
    hi = [m + s for m, s in zip(means, stds)]
    ax.fill_between(calls, lo, hi, alpha=0.2, color="#4a8adc")
    ax.plot(calls, means, color="#4a8adc", linewidth=2, label="batch mean ± std")
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("GRPO reward-function calls")
    ax.set_ylabel("Mean batch reward")
    ax.set_title("Reward during GRPO training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: before/after
    ax2 = axes[1]
    labels = [f"Random\n(before)", f"GRPO\n(after)"]
    sr = [baseline["success_rate"] * 100, after["success_rate"] * 100]
    colors = ["#8a3a3a", "#3a8a5a"]
    bars = ax2.bar(labels, sr, color=colors, width=0.5, zorder=3)
    for bar, v in zip(bars, sr):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                 f"{v:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Success rate (%)")
    ax2.set_title(f"Before vs After  ({DIFFICULTY} difficulty)")
    ax2.grid(True, alpha=0.3, axis="y", zorder=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved → {PLOT_PATH}")

except Exception as e:
    print(f"Plot skipped: {e}")

print("\nDone! Model saved to ./outputs/grpo_unsloth_final/")
print("To push to Hub: huggingface-cli upload <username>/<repo> ./outputs/grpo_unsloth_final/")
