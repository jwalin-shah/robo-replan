"""
RoboReplan Training — GRPO only (no SFT phase)
Usage: python run_training.py
"""
import sys
sys.path.insert(0, '..')

import re
import json
import random
import os
import time
import subprocess
import importlib


AUTO_INSTALL_DEPS = os.environ.get("AUTO_INSTALL_DEPS", "0").lower() in ("1", "true", "yes")
REQUIRED_MODULES = [
    ("datasets", "datasets"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("trl", "trl"),
]
missing_packages = []
for module_name, package_name in REQUIRED_MODULES:
    try:
        importlib.import_module(module_name)
    except ImportError:
        missing_packages.append(package_name)

if missing_packages:
    pip_cmd = f"{sys.executable} -m pip install " + " ".join(missing_packages)
    if not AUTO_INSTALL_DEPS:
        raise RuntimeError(
            "Missing Python packages: "
            + ", ".join(sorted(set(missing_packages)))
            + ". Install with:\n"
            + pip_cmd
            + "\nOr rerun with AUTO_INSTALL_DEPS=1."
        )
    print(f"Installing missing dependencies: {sorted(set(missing_packages))}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *sorted(set(missing_packages))])

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

try:
    from trl import SFTTrainer, SFTConfig
except Exception:
    SFTTrainer = None
    SFTConfig = None

from server.config import EnvConfig
from server.environment import TabletopPlanningEnv
from server.models import Action
from server.robosim.randomizer import ScenarioConfig

ACTIONS = [a.value for a in Action]
ACTION_LIST_STR = " | ".join(ACTIONS)
MODEL   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_FALLBACKS = [
    MODEL,
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]
STRICT_MODEL = os.environ.get("STRICT_MODEL", "0").lower() in ("1", "true", "yes")
ORACLE_EPISODES = int(os.environ.get("ORACLE_EPISODES", "800"))
BASELINE_EPISODES = int(os.environ.get("BASELINE_EPISODES", "50"))
FINAL_EVAL_EPISODES = int(os.environ.get("FINAL_EVAL_EPISODES", "50"))
MONITOR_EVERY = int(os.environ.get("MONITOR_EVERY", "25"))
INCLUDE_VALID_HINT = os.environ.get("INCLUDE_VALID_HINT", "0").lower() in ("1", "true", "yes")
METRICS_JSONL = os.environ.get("METRICS_JSONL", "./logs/train_metrics.jsonl")
TRAIN_DIFFICULTY = os.environ.get("TRAIN_DIFFICULTY", "easy").strip().lower()
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", "100"))
FIXED_EVAL_SEED = int(os.environ.get("FIXED_EVAL_SEED", "1337"))
GRPO_BATCH_SIZE = int(os.environ.get("GRPO_BATCH_SIZE", "4"))
GRPO_GRAD_ACCUM = int(os.environ.get("GRPO_GRAD_ACCUM", "4"))
GRPO_NUM_GENERATIONS = int(os.environ.get("GRPO_NUM_GENERATIONS", "4"))
GRPO_MAX_COMPLETION_LENGTH = int(os.environ.get("GRPO_MAX_COMPLETION_LENGTH", "12"))
GRPO_TEMPERATURE = float(os.environ.get("GRPO_TEMPERATURE", "1.1"))
GRPO_TOP_P = float(os.environ.get("GRPO_TOP_P", "0.98"))
ENABLE_SFT_WARMSTART = os.environ.get("ENABLE_SFT_WARMSTART", "0").lower() in ("1", "true", "yes")
SFT_EPOCHS = float(os.environ.get("SFT_EPOCHS", "1"))
SFT_MAX_STEPS = int(os.environ.get("SFT_MAX_STEPS", "300"))
SFT_LR = float(os.environ.get("SFT_LR", "2e-5"))
SFT_BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "4"))
SFT_GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "2"))
SCAN_ORACLE_KEEP_PROB = float(os.environ.get("SCAN_ORACLE_KEEP_PROB", "0.35"))
PICK_LOOP_PENALTY = float(os.environ.get("PICK_LOOP_PENALTY", "2.5"))
INVALID_ACTION_BASE_PENALTY = float(os.environ.get("INVALID_ACTION_BASE_PENALTY", "4.0"))
EXCEPTION_PENALTY = float(os.environ.get("EXCEPTION_PENALTY", "10.0"))
MOVE_NO_PROGRESS_PENALTY = float(os.environ.get("MOVE_NO_PROGRESS_PENALTY", "0.9"))
DOMINANCE_THRESHOLD = float(os.environ.get("DOMINANCE_THRESHOLD", "0.6"))
DOMINANCE_BASE_PENALTY = float(os.environ.get("DOMINANCE_BASE_PENALTY", "2.0"))

CFG_BY_NAME = {
    "easy": EnvConfig.easy,
    "medium": EnvConfig.medium,
    "hard": EnvConfig.hard,
}
if TRAIN_DIFFICULTY not in CFG_BY_NAME:
    raise ValueError(f"Unsupported TRAIN_DIFFICULTY={TRAIN_DIFFICULTY}; use one of {list(CFG_BY_NAME)}")
TRAIN_CFG = CFG_BY_NAME[TRAIN_DIFFICULTY]()
# Completion-first weighting to reduce reward hacking.
TRAIN_CFG.reward.task_complete = 25.0
TRAIN_CFG.reward.efficiency_bonus_max = 3.0
TRAIN_CFG.reward.successful_pick = 0.4
TRAIN_CFG.reward.blocker_cleared = 0.6
TRAIN_CFG.reward.correct_placement = 1.0
TRAIN_CFG.reward.recovery_after_failure = 0.3
TRAIN_CFG.reward.useful_scan = 0.05
TRAIN_CFG.reward.wrong_bin = -6.0
TRAIN_CFG.reward.first_failure = -2.0
TRAIN_CFG.reward.repeated_failure = -3.5
TRAIN_CFG.reward.constraint_violation = -6.0
TRAIN_CFG.reward.step_cost = -0.08

SYSTEM = (
    "You are a robot planning agent on a tabletop. Complete manipulation tasks "
    "by choosing ONE action per step.\n\n"
    f"Actions: {ACTION_LIST_STR}\n\n"
    "Output ONLY one action name from the list. No explanation, no tags, no extra text.\n"
    "Example:\n"
    "CLEAR_BLOCKER"
)

# ── Helpers ────────────────────────────────────────────────────────────

def obs_to_user_msg(obs):
    objects = ', '.join(
        f"{o.name}({'reachable' if o.reachable else 'BLOCKED'})"
        for o in obs.visible_objects
    )
    valid    = ', '.join(obs.valid_actions) if obs.valid_actions else 'any'
    failures = '; '.join(obs.known_failures) or 'none'
    subgoals = '; '.join(obs.completed_subgoals) or 'none yet'
    history  = ' -> '.join(obs.action_history[-5:]) or 'none'
    return (
        f"Instruction: {obs.instruction}\n"
        f"Scene: {objects}\n"
        f"Holding: {obs.holding or 'nothing'}\n"
        f"Progress: {obs.goal_progress:.0%}  Remaining: {obs.goals_remaining}\n"
        f"Completed: {subgoals}\n"
        f"Failures: {failures}\n"
        f"History: {history}\n"
        f"Last: {obs.last_action or 'none'} -> {obs.last_result or 'n/a'}\n"
        + (f"Valid now: {valid}\n" if INCLUDE_VALID_HINT else "")
        + f"Steps left: {obs.steps_remaining}\n\nNext action:"
    )

def scenario_to_json(scen) -> str:
    return json.dumps({
        'objects': scen.objects, 'targets': scen.targets,
        'blockers': scen.blockers, 'distractors': scen.distractors,
        'constraint': scen.constraint, 'instruction': scen.instruction,
        'positions': {k: list(v) for k, v in scen.positions.items()},
        'hidden_traits': scen.hidden_traits,
        'deadlines': scen.deadlines,
    })

def json_to_scenario(s: str) -> ScenarioConfig:
    d = json.loads(s)
    d['positions'] = {k: tuple(v) for k, v in d['positions'].items()}
    d.setdefault('hidden_traits', {})
    d.setdefault('deadlines', {})
    return ScenarioConfig(**d)

def extract_action(text: str):
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip().upper()
    normalized = re.sub(r'[^A-Z_ ]+', ' ', clean)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    # Parse strictly from the first token span to avoid accidental matches later in text.
    first_span = normalized.split("  ")[0].split("\n")[0].strip()
    if first_span in ACTIONS:
        return first_span
    first_words = " ".join(first_span.split(" ")[:3]).strip()
    spaced_to_action = {a.replace("_", " "): a for a in ACTIONS}
    if first_words in spaced_to_action:
        return spaced_to_action[first_words]
    if first_span in ("PLACE", "PLACE BIN", "PLACE BIN A", "PLACE A"):
        return "PLACE_BIN_A"
    if first_span in ("PLACE BIN B", "PLACE B"):
        return "PLACE_BIN_B"
    if first_span in ("SCAN", "SCAN SCENE"):
        return "SCAN_SCENE"
    if first_span in ("CLEAR", "CLEAR BLOCKER"):
        return "CLEAR_BLOCKER"

    for a in sorted(ACTIONS, key=len, reverse=True):
        if a in clean:
            return a
    # Accept common variants like "MOVE TO RED" / "PLACE BIN A".
    for spaced, canonical in spaced_to_action.items():
        if spaced in normalized:
            return canonical
    # If model outputs just color/direction words, map them conservatively.
    keyword_map = {
        "RED": "MOVE_TO_RED",
        "BLUE": "MOVE_TO_BLUE",
        "GREEN": "MOVE_TO_GREEN",
        "YELLOW": "MOVE_TO_YELLOW",
        "PURPLE": "MOVE_TO_PURPLE",
        "NORTH": "MOVE_NORTH",
        "SOUTH": "MOVE_SOUTH",
        "EAST": "MOVE_EAST",
        "WEST": "MOVE_WEST",
    }
    for k, v in keyword_map.items():
        if normalized == k:
            return v
    if normalized == "PICK":
        return "PICK"
    if normalized in ("PLACE BIN A", "PLACE A"):
        return "PLACE_BIN_A"
    if normalized in ("PLACE BIN B", "PLACE B"):
        return "PLACE_BIN_B"
    if normalized in ("CLEAR BLOCKER", "CLEAR"):
        return "CLEAR_BLOCKER"
    if normalized in ("SCAN SCENE", "SCAN"):
        return "SCAN_SCENE"
    if normalized in ("ROTATE LEFT", "LEFT ROTATE"):
        return "ROTATE_LEFT"
    if normalized in ("ROTATE RIGHT", "RIGHT ROTATE"):
        return "ROTATE_RIGHT"
    if normalized in ("MOVE NORTH",):
        return "MOVE_NORTH"
    if normalized in ("MOVE SOUTH",):
        return "MOVE_SOUTH"
    if normalized in ("MOVE EAST",):
        return "MOVE_EAST"
    if normalized in ("MOVE WEST",):
        return "MOVE_WEST"
    if normalized in ("WAIT",):
        return "WAIT"
    if normalized in ("TOGGLE LIGHT",):
        return "TOGGLE_LIGHT"
    # Fallback: first token exact action.
    token = normalized.split(" ")[0] if normalized else ""
    if token in ACTIONS:
        return token
    return None

def extract_reasoning(text: str) -> str:
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    return m.group(1).strip() if m else ''

def parse_prompt_context(prompt_messages):
    user_text = ''
    for msg in prompt_messages or []:
        if isinstance(msg, dict) and msg.get('role') == 'user':
            user_text = msg.get('content', '')
            break

    valid_actions = set()
    last_action = None
    last_result = None
    history = []
    progress_pct = None

    valid_match = re.search(r'Valid now:\s*(.*)', user_text)
    if valid_match:
        valid_raw = valid_match.group(1).strip()
        if valid_raw.lower() != 'any':
            valid_actions = {a.strip().upper() for a in valid_raw.split(',') if a.strip()}

    last_match = re.search(r'Last:\s*([A-Z_]+|none)\s*->\s*([A-Z_]+|n/a)', user_text, flags=re.IGNORECASE)
    if last_match:
        la = last_match.group(1).strip().upper()
        lr = last_match.group(2).strip().upper()
        last_action = None if la == 'NONE' else la
        last_result = None if lr == 'N/A' else lr

    history_match = re.search(r'History:\s*(.*)', user_text)
    if history_match:
        hist_raw = history_match.group(1).strip()
        if hist_raw.lower() != 'none':
            history = [h.strip().upper() for h in hist_raw.split('->') if h.strip()]

    progress_match = re.search(r'Progress:\s*(\d+)%', user_text)
    if progress_match:
        progress_pct = int(progress_match.group(1)) / 100.0

    return valid_actions, last_action, last_result, history, progress_pct

def eval_policy(policy_fn, n_episodes=50, seed: int | None = None):
    if seed is not None:
        random.seed(seed)
    env = TabletopPlanningEnv(config=TRAIN_CFG)
    rewards, successes = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        total = 0
        for _ in range(20):
            action = policy_fn(obs)
            r = env.step(action)
            total += r.reward
            obs = r.observation
            if r.done:
                break
        rewards.append(total)
        successes.append(env._all_goals_complete())
    return {
        'success_rate': sum(successes) / n_episodes,
        'avg_reward':   sum(rewards)   / n_episodes,
    }

def eval_policy_suite(policy_fn):
    fixed = eval_policy(policy_fn, n_episodes=EVAL_EPISODES, seed=FIXED_EVAL_SEED)
    general = eval_policy(policy_fn, n_episodes=EVAL_EPISODES, seed=None)
    return fixed, general

# ── Baseline ───────────────────────────────────────────────────────────

print("=" * 50)
print("BEFORE TRAINING (random policy)")
print("=" * 50)
baseline = eval_policy(lambda obs: random.choice(ACTIONS), n_episodes=50)
print(f"  Success rate: {baseline['success_rate']:.0%}")
print(f"  Avg reward:   {baseline['avg_reward']:.2f}")

# ── Build dataset ──────────────────────────────────────────────────────

print("\nBuilding oracle dataset...")
cfg = TRAIN_CFG
cfg.obs.include_oracle_hint = True
env = TabletopPlanningEnv(config=cfg)

rows = []
for ep in range(ORACLE_EPISODES):
    obs = env.reset()
    scenario_json = scenario_to_json(env._scenario_cfg)
    step_num = 0
    for _ in range(20):
        if obs.oracle_hint:
            if obs.oracle_hint == 'SCAN_SCENE' and random.random() > SCAN_ORACLE_KEEP_PROB:
                action = obs.oracle_hint or 'SCAN_SCENE'
                r = env.step(action)
                obs = r.observation
                step_num += 1
                if r.done:
                    break
                continue
            rows.append({
                'prompt':     [
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user',   'content': obs_to_user_msg(obs)},
                ],
                'completion': [{'role': 'assistant', 'content': obs.oracle_hint}],
                'answer':     obs.oracle_hint,
                'scenario':   scenario_json,
                'step':       step_num,
                'history_actions': list(obs.action_history),
            })
        action = obs.oracle_hint or 'SCAN_SCENE'
        r = env.step(action)
        obs = r.observation
        step_num += 1
        if r.done:
            break
    if ep % 100 == 0:
        print(f"  Episode {ep}/{ORACLE_EPISODES}, rows: {len(rows)}")

dataset = Dataset.from_list(rows).train_test_split(test_size=0.1)
print(f"Dataset: {len(rows)} steps ({len(dataset['train'])} train, {len(dataset['test'])} eval)")

# ── Load model ─────────────────────────────────────────────────────────

print("\nLoading model...")
load_error = None
resolved_model = None
for candidate in MODEL_FALLBACKS:
    try:
        print(f"Trying model: {candidate}")
        tokenizer = AutoTokenizer.from_pretrained(candidate)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(candidate, dtype='auto', device_map='auto')
        resolved_model = candidate
        break
    except Exception as exc:
        load_error = exc
        print(f"Model load failed for {candidate}: {exc}")
        if STRICT_MODEL and candidate == MODEL:
            raise RuntimeError(
                f"STRICT_MODEL=1 and requested model failed to load: {MODEL}"
            ) from exc
        continue

if resolved_model is None:
    raise RuntimeError(
        "Unable to load any candidate model. "
        f"Requested={MODEL}, tried={MODEL_FALLBACKS}"
    ) from load_error

print(f"Using model: {resolved_model}")
model.generation_config.pad_token_id = tokenizer.pad_token_id
print(f"Loaded on: {next(model.parameters()).device}")

# ── Optional SFT warm-start ───────────────────────────────────────────

if ENABLE_SFT_WARMSTART:
    if SFTTrainer is None or SFTConfig is None:
        raise RuntimeError("ENABLE_SFT_WARMSTART=1 but SFTTrainer/SFTConfig are unavailable in this TRL install")

    print("\n" + "=" * 50)
    print("PHASE: SFT WARM-START")
    print("=" * 50)
    sft_rows = []
    for row in dataset['train']:
        chat = row['prompt'] + [{'role': 'assistant', 'content': row['answer']}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        sft_rows.append({"text": text})
    sft_dataset = Dataset.from_list(sft_rows)
    print(f"SFT rows: {len(sft_rows)}")

    sft_args = SFTConfig(
        output_dir='./outputs/sft_warmstart',
        num_train_epochs=SFT_EPOCHS,
        max_steps=SFT_MAX_STEPS,
        per_device_train_batch_size=SFT_BATCH_SIZE,
        gradient_accumulation_steps=SFT_GRAD_ACCUM,
        learning_rate=SFT_LR,
        logging_steps=20,
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
    print("SFT warm-start done")

# ── Phase: GRPO ────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("PHASE: GRPO (TRL)")
print("=" * 50)

# GRPO directly from base model

def reward_fn(completions, prompts=None, scenario=None, **kwargs):
    if not hasattr(reward_fn, "_calls"):
        reward_fn._calls = 0
        reward_fn._start = time.time()
        reward_fn._flat_streak = 0
        reward_fn._stats = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "parse_fail": 0,
            "exceptions": 0,
            "action_counts": {},
            "dominance_penalty_scale": 1.0,
            "scan_penalty_scale": 1.0,
            "invalid_pick_scale": 1.0,
            "exception_debug_left": 10,
        }
    reward_fn._calls += 1

    total_seen = max(1, reward_fn._stats["total"])
    top_action = None
    top_share = 0.0
    if reward_fn._stats["action_counts"]:
        top_action, top_count = max(reward_fn._stats["action_counts"].items(), key=lambda kv: kv[1])
        top_share = top_count / total_seen
    reward_fn._stats["dominance_penalty_scale"] = 1.0 + max(0.0, top_share - DOMINANCE_THRESHOLD) * 7.0

    batch_actions = []
    batch_rewards = []
    batch_results = []
    batch_oracles = []
    rewards = []
    step_values = kwargs.get("step")
    history_action_values = kwargs.get("history_actions")

    for i, completion in enumerate(completions):
        text      = completion if isinstance(completion, str) else completion[0].get('content', '')
        action    = extract_action(text)
        reasoning = extract_reasoning(text)

        if action is None:
            rewards.append(-3.0)
            reward_fn._stats["total"] += 1
            reward_fn._stats["parse_fail"] += 1
            continue
        reward_fn._stats["total"] += 1
        reward_fn._stats["action_counts"][action] = reward_fn._stats["action_counts"].get(action, 0) + 1
        try:
            valid_actions, last_action, last_result, history, progress_pct = parse_prompt_context(prompts[i] if prompts else None)
            if valid_actions and action not in valid_actions:
                repeat_tail = 0
                for h in reversed(history):
                    if h == action:
                        repeat_tail += 1
                    else:
                        break
                shaped_reward = -INVALID_ACTION_BASE_PENALTY - (0.6 * min(repeat_tail, 6))
                if action == "PICK":
                    shaped_reward -= PICK_LOOP_PENALTY * reward_fn._stats["invalid_pick_scale"]
                rewards.append(shaped_reward)
                reward_fn._stats["invalid"] += 1
                batch_actions.append(action)
                batch_rewards.append(shaped_reward)
                batch_results.append("FAILED_INVALID")
                batch_oracles.append(None)
                continue
            reward_fn._stats["valid"] += 1

            eval_env = TabletopPlanningEnv(config=TRAIN_CFG)
            scen = json_to_scenario(scenario[i])
            eval_env.sim._build_state_from_config(scen)
            eval_env._scenario_cfg        = scen
            eval_env._instruction         = scen.instruction
            eval_env._required_placements = dict(scen.targets)
            eval_env._active_constraints  = [scen.constraint] if scen.constraint else []

            # Reconstruct rollout state from recorded action history.
            replay_done = False
            replay_actions = []
            if history_action_values is not None and history_action_values[i] is not None:
                replay_actions = list(history_action_values[i])
            elif step_values is not None:
                step_i = int(step_values[i])
                replay_actions = [(eval_env._oracle_action() or "SCAN_SCENE") for _ in range(max(0, step_i))]

            for pre_action in replay_actions:
                pre = eval_env.step(pre_action)
                if pre.done:
                    replay_done = True
                    break

            if replay_done:
                shaped_reward = -2.0
                rewards.append(shaped_reward)
                batch_actions.append(action)
                batch_rewards.append(shaped_reward)
                batch_results.append("REPLAY_DONE")
                batch_oracles.append(None)
                continue

            valid_now = set(eval_env._valid_actions())
            if action not in valid_now:
                repeat_tail = 0
                for h in reversed(history):
                    if h == action:
                        repeat_tail += 1
                    else:
                        break
                shaped_reward = -INVALID_ACTION_BASE_PENALTY - (0.8 * min(repeat_tail, 6))
                if action == "PICK":
                    shaped_reward -= PICK_LOOP_PENALTY * reward_fn._stats["invalid_pick_scale"]
                rewards.append(shaped_reward)
                reward_fn._stats["invalid"] += 1
                batch_actions.append(action)
                batch_rewards.append(shaped_reward)
                batch_results.append("FAILED_INVALID")
                batch_oracles.append(eval_env._oracle_action())
                continue

            pre_progress = eval_env._goal_progress()
            oracle_action = eval_env._oracle_action()
            result = eval_env.step(action, reasoning=reasoning)
            shaped_reward = float(result.reward)
            post_progress = eval_env._goal_progress()
            progress_delta = max(0.0, post_progress - pre_progress)

            # Discourage no-op loops like MOVE_TO_X -> MOVE_TO_X after a successful move.
            if (
                last_action
                and last_result == 'SUCCESS'
                and action == last_action
                and action.startswith('MOVE_TO_')
            ):
                shaped_reward -= 0.75

            # Penalize repeated failed PICK attempts to break local loops.
            if action == 'PICK' and last_action == 'PICK' and last_result and last_result.startswith('FAILED'):
                shaped_reward -= 1.5 * reward_fn._stats["invalid_pick_scale"]

            # Discourage no-op behavior and repeated identical actions.
            if action == 'SCAN_SCENE':
                scan_tail = 0
                for h in reversed(history):
                    if h == 'SCAN_SCENE':
                        scan_tail += 1
                    else:
                        break
                scan_streak = scan_tail + 1
                if scan_streak == 1:
                    scan_penalty = 0.35
                elif scan_streak == 2:
                    scan_penalty = 0.9
                elif scan_streak == 3:
                    scan_penalty = 1.8
                else:
                    scan_penalty = 3.0 + 0.5 * (scan_streak - 4)
                shaped_reward -= scan_penalty * reward_fn._stats["scan_penalty_scale"]
                # Repeated scanning when oracle wants progress should be heavily discouraged.
                if oracle_action and oracle_action != 'SCAN_SCENE':
                    shaped_reward -= 1.5 * reward_fn._stats["scan_penalty_scale"]
                else:
                    shaped_reward += 0.05
            if last_action and action == last_action:
                shaped_reward -= 0.5

            if progress_delta <= 1e-6 and last_action and action == last_action:
                repeat_tail = 0
                for h in reversed(history):
                    if h == action:
                        repeat_tail += 1
                    else:
                        break
                shaped_reward -= (0.4 + 0.2 * min(repeat_tail, 6))

            if progress_delta <= 1e-6 and len(history) >= 3 and len(set(history[-3:])) == 1:
                shaped_reward -= 0.9

            if action == "PICK" and result.info.get("result", "").startswith("FAILED"):
                pick_tail = 0
                for h in reversed(history):
                    if h == "PICK":
                        pick_tail += 1
                    else:
                        break
                shaped_reward -= (1.0 + 0.5 * min(pick_tail, 6)) * reward_fn._stats["invalid_pick_scale"]

            if progress_delta <= 1e-6 and progress_pct is not None and progress_pct < 0.95:
                shaped_reward -= 0.15

            if action.startswith("MOVE_TO_") and progress_delta <= 1e-6:
                shaped_reward -= MOVE_NO_PROGRESS_PENALTY

            # Reward oracle alignment only when oracle action is in model action space.
            if oracle_action in ACTIONS and action == oracle_action:
                shaped_reward += 2.0
            elif oracle_action in ACTIONS and action != oracle_action:
                shaped_reward -= 0.6

            # If the policy is collapsing on one dominant action, punish choosing it
            # when it does not match oracle guidance.
            if (
                top_action
                and action == top_action
                and oracle_action in ACTIONS
                and action != oracle_action
            ):
                shaped_reward -= 1.8 * reward_fn._stats["dominance_penalty_scale"]
            if top_action and action == top_action and top_share > DOMINANCE_THRESHOLD:
                shaped_reward -= DOMINANCE_BASE_PENALTY * reward_fn._stats["dominance_penalty_scale"]

            same_tail = 0
            for h in reversed(history):
                if h == action:
                    same_tail += 1
                else:
                    break
            if same_tail >= 2:
                shaped_reward -= 0.8 + 0.4 * min(same_tail - 2, 6)
            if same_tail >= 5:
                shaped_reward -= 6.0

            # Explicitly reward actual task progress.
            if progress_delta > 0:
                shaped_reward += 4.0 * progress_delta

            rewards.append(shaped_reward)
            batch_actions.append(action)
            batch_rewards.append(shaped_reward)
            batch_results.append(result.info.get("result", "UNKNOWN"))
            batch_oracles.append(oracle_action)
        except Exception as exc:
            shaped_reward = -EXCEPTION_PENALTY
            reward_fn._stats["exceptions"] += 1
            rewards.append(shaped_reward)
            batch_actions.append(action)
            batch_rewards.append(shaped_reward)
            batch_results.append("EXCEPTION")
            batch_oracles.append(None)
            if reward_fn._stats["exception_debug_left"] > 0:
                print(
                    "[reward-exception]",
                    f"idx={i}",
                    f"action={action}",
                    f"err={type(exc).__name__}:{exc}",
                )
                reward_fn._stats["exception_debug_left"] -= 1

    if reward_fn._calls % MONITOR_EVERY == 0:
        stats = reward_fn._stats
        total = max(1, stats["total"])
        valid_rate = 100.0 * stats["valid"] / total
        invalid_rate = 100.0 * stats["invalid"] / total
        parse_fail_rate = 100.0 * stats["parse_fail"] / total
        exception_rate = 100.0 * stats["exceptions"] / total
        top_actions = sorted(stats["action_counts"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        elapsed = time.time() - reward_fn._start
        batch_mean = sum(batch_rewards) / max(1, len(batch_rewards))
        batch_std = 0.0
        if len(batch_rewards) > 1:
            m = batch_mean
            batch_std = (sum((x - m) ** 2 for x in batch_rewards) / len(batch_rewards)) ** 0.5
        if batch_std < 0.05:
            reward_fn._flat_streak += 1
        else:
            reward_fn._flat_streak = 0

        print(
            "[reward-debug]",
            f"call={reward_fn._calls}",
            f"mean={batch_mean:.3f}",
            f"actions={batch_actions[:4]}",
            f"results={batch_results[:4]}",
            f"oracles={batch_oracles[:4]}",
        )
        print(
            "[validity]",
            f"valid={valid_rate:.1f}%",
            f"invalid={invalid_rate:.1f}%",
            f"parse_fail={parse_fail_rate:.1f}%",
            f"exception={exception_rate:.1f}%",
            f"top_actions={top_actions}",
        )
        collapse_flag = "OK"
        if reward_fn._flat_streak >= 4:
            collapse_flag = "COLLAPSE_RISK:low_reward_variance"
        if top_actions and top_actions[0][1] / total > 0.7:
            collapse_flag = "COLLAPSE_RISK:action_dominance"

        if collapse_flag.startswith("COLLAPSE_RISK"):
            reward_fn._stats["scan_penalty_scale"] = min(3.0, reward_fn._stats["scan_penalty_scale"] + 0.15)
            reward_fn._stats["invalid_pick_scale"] = min(3.5, reward_fn._stats["invalid_pick_scale"] + 0.2)
        else:
            reward_fn._stats["scan_penalty_scale"] = max(1.0, reward_fn._stats["scan_penalty_scale"] - 0.05)
            reward_fn._stats["invalid_pick_scale"] = max(1.0, reward_fn._stats["invalid_pick_scale"] - 0.05)
        print(
            "[train-monitor]",
            f"elapsed={elapsed/60:.1f}m",
            f"batch_std={batch_std:.3f}",
            f"flat_streak={reward_fn._flat_streak}",
            f"scan_penalty_scale={reward_fn._stats['scan_penalty_scale']:.2f}",
            f"invalid_pick_scale={reward_fn._stats['invalid_pick_scale']:.2f}",
            f"status={collapse_flag}",
        )
        # Live telemetry for dashboards / quick plotting.
        try:
            os.makedirs(os.path.dirname(METRICS_JSONL), exist_ok=True)
            row = {
                "call": reward_fn._calls,
                "elapsed_sec": round(elapsed, 2),
                "batch_mean": round(batch_mean, 5),
                "batch_std": round(batch_std, 5),
                "valid_rate": round(valid_rate, 3),
                "invalid_rate": round(invalid_rate, 3),
                "parse_fail_rate": round(parse_fail_rate, 3),
                "flat_streak": reward_fn._flat_streak,
                "status": collapse_flag,
                "top_actions": top_actions,
            }
            with open(METRICS_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                "[health]",
                f"mean={row['batch_mean']:+.3f}",
                f"std={row['batch_std']:.3f}",
                f"valid={row['valid_rate']:.1f}%",
                f"parse_fail={row['parse_fail_rate']:.1f}%",
                f"top={top_actions[0][0] if top_actions else 'n/a'}",
                f"status={collapse_flag}",
            )
        except Exception as exc:
            print(f"[health] telemetry_write_failed={exc}")
    return rewards

grpo_config = GRPOConfig(
    output_dir='./outputs/grpo',
    num_train_epochs=1,
    per_device_train_batch_size=GRPO_BATCH_SIZE,
    gradient_accumulation_steps=GRPO_GRAD_ACCUM,
    learning_rate=1e-5,
    num_generations=GRPO_NUM_GENERATIONS,
    max_completion_length=GRPO_MAX_COMPLETION_LENGTH,
    temperature=GRPO_TEMPERATURE,
    top_p=GRPO_TOP_P,
    logging_steps=5,
    save_steps=400,
    save_total_limit=2,
    save_only_model=True,
    push_to_hub=False,
    report_to='none',
    bf16=True,
)

model.warnings_issued = {}  # required by this TRL version

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    reward_funcs=reward_fn,
    train_dataset=dataset['train'],
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model('./outputs/grpo_final')
print("GRPO done -> ./outputs/grpo_final")

# ── Evaluate ───────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("AFTER TRAINING")
print("=" * 50)
eval_model = AutoModelForCausalLM.from_pretrained(
    './outputs/grpo_final', dtype='auto', device_map='auto'
)
eval_model.generation_config.pad_token_id = tokenizer.pad_token_id

def trained_policy(obs):
    messages = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user',   'content': obs_to_user_msg(obs)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt_text, return_tensors='pt').to(eval_model.device)
    with torch.no_grad():
        output_ids = eval_model.generate(
            **encoded,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][encoded["input_ids"].shape[1]:]
    out = tokenizer.decode(new_ids, skip_special_tokens=True)
    return extract_action(out) or random.choice(ACTIONS)

after = eval_policy(trained_policy, n_episodes=FINAL_EVAL_EPISODES)
print(f"  Success rate: {after['success_rate']:.0%}")
print(f"  Avg reward:   {after['avg_reward']:.2f}")
print()
print("IMPROVEMENT:")
print(f"  Success: {baseline['success_rate']:.0%} -> {after['success_rate']:.0%}")
print(f"  Reward:  {baseline['avg_reward']:.1f} -> {after['avg_reward']:.1f}")
