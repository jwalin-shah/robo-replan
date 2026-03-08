"""
RoboReplan Training — GRPO only (no SFT phase)
Usage: python run_training.py
"""
import sys
sys.path.insert(0, '..')

import re
import json
import random
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import GRPOTrainer, GRPOConfig

from server.config import EnvConfig
from server.environment import TabletopPlanningEnv
from server.models import Action
from server.robosim.randomizer import ScenarioConfig

ACTIONS = [a.value for a in Action]
MODEL   = 'Qwen/Qwen2.5-0.5B-Instruct'

SYSTEM = (
    "You are a robot planning agent on a tabletop. Complete manipulation tasks "
    "by choosing ONE action per step.\n\n"
    "Actions: SCAN_SCENE | MOVE_TO_RED | MOVE_TO_BLUE | MOVE_TO_GREEN | MOVE_TO_YELLOW | MOVE_TO_PURPLE | PICK | PLACE_BIN_A | PLACE_BIN_B | CLEAR_BLOCKER\n\n"
    "Think step by step inside <think>...</think> tags, then output ONLY the action name.\n"
    "Example:\n"
    "<think>Red block is blocked by blue. Must clear blue first, then pick red.</think>\n"
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
        f"Valid now: {valid}\n"
        f"Steps left: {obs.steps_remaining}\n\nNext action:"
    )

def scenario_to_json(scen) -> str:
    return json.dumps({
        'objects': scen.objects, 'targets': scen.targets,
        'blockers': scen.blockers, 'distractors': scen.distractors,
        'constraint': scen.constraint, 'instruction': scen.instruction,
        'positions': {k: list(v) for k, v in scen.positions.items()},
    })

def json_to_scenario(s: str) -> ScenarioConfig:
    d = json.loads(s)
    d['positions'] = {k: tuple(v) for k, v in d['positions'].items()}
    return ScenarioConfig(**d)

def extract_action(text: str):
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip().upper()
    for a in sorted(ACTIONS, key=len, reverse=True):
        if a in clean:
            return a
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

    return valid_actions, last_action, last_result

def eval_policy(policy_fn, n_episodes=50):
    env = TabletopPlanningEnv(config=EnvConfig.easy())
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

# ── Baseline ───────────────────────────────────────────────────────────

print("=" * 50)
print("BEFORE TRAINING (random policy)")
print("=" * 50)
baseline = eval_policy(lambda obs: random.choice(ACTIONS), n_episodes=50)
print(f"  Success rate: {baseline['success_rate']:.0%}")
print(f"  Avg reward:   {baseline['avg_reward']:.2f}")

# ── Build dataset ──────────────────────────────────────────────────────

print("\nBuilding oracle dataset...")
cfg = EnvConfig.easy()
cfg.obs.include_oracle_hint = True
env = TabletopPlanningEnv(config=cfg)

rows = []
for ep in range(800):
    obs = env.reset()
    scenario_json = scenario_to_json(env._scenario_cfg)
    step_num = 0
    for _ in range(20):
        if obs.oracle_hint:
            rows.append({
                'prompt':     [
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user',   'content': obs_to_user_msg(obs)},
                ],
                'completion': [{'role': 'assistant', 'content': obs.oracle_hint}],
                'answer':     obs.oracle_hint,
                'scenario':   scenario_json,
                'step':       step_num,
            })
        action = obs.oracle_hint or 'SCAN_SCENE'
        r = env.step(action)
        obs = r.observation
        step_num += 1
        if r.done:
            break
    if ep % 100 == 0:
        print(f"  Episode {ep}/800, rows: {len(rows)}")

dataset = Dataset.from_list(rows).train_test_split(test_size=0.1)
print(f"Dataset: {len(rows)} steps ({len(dataset['train'])} train, {len(dataset['test'])} eval)")

# ── Load model ─────────────────────────────────────────────────────────

print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype='auto', device_map='auto')
model.generation_config.pad_token_id = tokenizer.pad_token_id
print(f"Loaded on: {next(model.parameters()).device}")

# ── Phase: GRPO ────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("PHASE: GRPO (TRL)")
print("=" * 50)

# GRPO directly from base model

def reward_fn(completions, prompts=None, scenario=None, **kwargs):
    if not hasattr(reward_fn, "_calls"):
        reward_fn._calls = 0
    reward_fn._calls += 1

    batch_actions = []
    batch_rewards = []
    batch_results = []
    batch_oracles = []
    rewards = []
    for i, completion in enumerate(completions):
        text      = completion if isinstance(completion, str) else completion[0].get('content', '')
        action    = extract_action(text)
        reasoning = extract_reasoning(text)

        if action is None:
            rewards.append(-2.0)
            continue
        try:
            valid_actions, last_action, last_result = parse_prompt_context(prompts[i] if prompts else None)
            if valid_actions and action not in valid_actions:
                shaped_reward = -3.0
                rewards.append(shaped_reward)
                batch_actions.append(action)
                batch_rewards.append(shaped_reward)
                batch_results.append("FAILED_INVALID")
                batch_oracles.append(None)
                continue

            eval_env = TabletopPlanningEnv(config=EnvConfig.easy())
            scen = json_to_scenario(scenario[i])
            eval_env.sim._build_state_from_config(scen)
            eval_env._scenario_cfg        = scen
            eval_env._instruction         = scen.instruction
            eval_env._required_placements = dict(scen.targets)
            eval_env._active_constraints  = [scen.constraint] if scen.constraint else []
            oracle_action = eval_env._oracle_action()
            result = eval_env.step(action, reasoning=reasoning)
            shaped_reward = float(result.reward)

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
                shaped_reward -= 1.0

            # Reward oracle alignment on this state to avoid collapsing to constant MOVE actions.
            if oracle_action and action == oracle_action:
                shaped_reward += 1.0

            rewards.append(shaped_reward)
            batch_actions.append(action)
            batch_rewards.append(shaped_reward)
            batch_results.append(result.info.get("result", "UNKNOWN"))
            batch_oracles.append(oracle_action)
        except Exception:
            shaped_reward = -1.0
            rewards.append(shaped_reward)
            batch_actions.append(action)
            batch_rewards.append(shaped_reward)
            batch_results.append("EXCEPTION")
            batch_oracles.append(None)

    if reward_fn._calls % 50 == 0:
        print(
            "[reward-debug]",
            f"call={reward_fn._calls}",
            f"mean={sum(batch_rewards) / max(1, len(batch_rewards)):.3f}",
            f"actions={batch_actions[:4]}",
            f"results={batch_results[:4]}",
            f"oracles={batch_oracles[:4]}",
        )
    return rewards

grpo_config = GRPOConfig(
    output_dir='./outputs/grpo',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_generations=8,
    max_completion_length=8,
    temperature=1.0,
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

pipe = pipeline('text-generation', model='./outputs/grpo_final',
                tokenizer=tokenizer, max_new_tokens=16, device_map='auto')

def trained_policy(obs):
    messages = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user',   'content': obs_to_user_msg(obs)},
    ]
    out = pipe(messages, return_full_text=False)[0]['generated_text']
    return extract_action(out) or random.choice(ACTIONS)

after = eval_policy(trained_policy, n_episodes=50)
print(f"  Success rate: {after['success_rate']:.0%}")
print(f"  Avg reward:   {after['avg_reward']:.2f}")
print()
print("IMPROVEMENT:")
print(f"  Success: {baseline['success_rate']:.0%} -> {after['success_rate']:.0%}")
print(f"  Reward:  {baseline['avg_reward']:.1f} -> {after['avg_reward']:.1f}")
