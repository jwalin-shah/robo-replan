"""
GRPO training on the Tabletop Planning Environment.

The LLM is the policy. At each step:
  1. Build a text prompt from the observation
  2. Sample K completions (actions) from the model
  3. Execute each action in the env, get reward
  4. GRPO updates the model toward higher-reward actions

Run on Northflank H100 or Colab:
  pip install trl transformers torch datasets

Then:
  python train/grpo_train.py
"""
import sys, re, json
sys.path.insert(0, ".")

from server.environment import TabletopPlanningEnv
from server.models import Action

VALID_ACTIONS = [a.value for a in Action]

# ─────────────────────────────────────────────────────────────────────
#  Prompt builder
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a robot planning agent on a tabletop.
Your job is to complete manipulation tasks by choosing one action per step.

Available actions:
- SCAN_SCENE        : inspect the table to update your view
- MOVE_TO_RED       : move gripper to red block
- MOVE_TO_BLUE      : move gripper to blue block
- MOVE_TO_GREEN     : move gripper to green block
- PICK              : grasp the object under the gripper
- PLACE_BIN_A       : place held object into bin A
- PLACE_BIN_B       : place held object into bin B
- CLEAR_BLOCKER     : push aside the object blocking your target

Reply with ONLY the action name. No explanation."""


def obs_to_prompt(obs) -> list[dict]:
    objects = ", ".join(
        f"{o.name}({'reachable' if o.reachable else 'BLOCKED'})"
        for o in obs.visible_objects
    )
    failures = "; ".join(obs.known_failures) or "none"
    subgoals = "; ".join(obs.completed_subgoals) or "none yet"
    constraints = "; ".join(obs.active_constraints) or "none"

    user_msg = f"""Instruction: {obs.instruction}
Scene: {objects}
Holding: {obs.holding or 'nothing'}
Completed subgoals: {subgoals}
Known failures: {failures}
Active constraints: {constraints}
Last action: {obs.last_action or 'none'} → {obs.last_result or 'n/a'}
Steps remaining: {obs.steps_remaining}

What is your next action?"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]


# ─────────────────────────────────────────────────────────────────────
#  Reward function (called by GRPO after each completion)
# ─────────────────────────────────────────────────────────────────────

def make_reward_fn(env: TabletopPlanningEnv):
    """
    Returns a reward function compatible with TRL's GRPO trainer.

    TRL calls reward_fn(completions, **kwargs) where completions is a list
    of generated strings. We execute each in the env and return env rewards.
    """
    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for completion in completions:
            # Extract action from model output
            action = extract_action(completion)
            if action is None:
                rewards.append(-2.0)  # unparseable output
                continue
            try:
                result = env.step(action)
                rewards.append(result.reward)
            except Exception:
                rewards.append(-1.0)
        return rewards
    return reward_fn


def extract_action(text: str) -> str | None:
    """Parse the model's output to get a valid action name."""
    text = text.strip().upper().replace("-", "_").replace(" ", "_")
    for action in VALID_ACTIONS:
        if action in text:
            return action
    return None


# ─────────────────────────────────────────────────────────────────────
#  Dataset: each row is one step (obs → best action)
#  We build this from scripted-policy rollouts as a warm start.
# ─────────────────────────────────────────────────────────────────────

def build_dataset(n_episodes: int = 300):
    from datasets import Dataset
    sys.path.insert(0, ".")
    from scripts.scripted_policy import scripted_agent

    env = TabletopPlanningEnv(use_stub=True)
    rows = []
    for _ in range(n_episodes):
        obs = env.reset()
        for _ in range(20):
            messages = obs_to_prompt(obs)
            action = scripted_agent(obs)
            rows.append({"prompt": messages, "answer": action})
            result = env.step(action)
            obs = result.observation
            if result.done:
                break

    print(f"Built dataset: {len(rows)} steps from {n_episodes} expert episodes")
    return Dataset.from_list(rows)


# ─────────────────────────────────────────────────────────────────────
#  GRPO training
# ─────────────────────────────────────────────────────────────────────

def train(hf_space_url: str = None):
    try:
        from trl import GRPOTrainer, GRPOConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Install: pip install trl transformers torch")
        return

    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # swap to 1.5B for better results
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    dataset = build_dataset(n_episodes=300)

    # Use remote HF Space env if URL provided, else local
    if hf_space_url:
        print(f"Using remote env: {hf_space_url}")
        from openenv import AutoEnv
        remote_env = AutoEnv.from_env(hf_space_url)
        # Wrap remote env to match local interface
        class RemoteWrapper:
            def reset(self): return remote_env.reset()
            def step(self, action): return remote_env.step({"action": action})
            def _all_goals_complete(self): return False  # check via done flag
        env = RemoteWrapper()
    else:
        env = TabletopPlanningEnv(use_stub=True)

    config = GRPOConfig(
        output_dir="./outputs/grpo",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_generations=4,       # K rollouts per prompt for GRPO
        max_new_tokens=16,       # action names are short
        logging_steps=10,
        save_steps=100,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=make_reward_fn(env),
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Training with GRPO...")
    trainer.train()
    trainer.save_model("./outputs/grpo_final")
    print("Done → ./outputs/grpo_final")


# ─────────────────────────────────────────────────────────────────────
#  Evaluate a trained model
# ─────────────────────────────────────────────────────────────────────

def eval_model(model_path: str, n_episodes: int = 20):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    print(f"Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    max_new_tokens=16, device_map="auto")

    env = TabletopPlanningEnv(use_stub=True)
    successes, rewards = 0, []

    for i in range(n_episodes):
        obs = env.reset()
        total_r = 0.0
        for _ in range(20):
            messages = obs_to_prompt(obs)
            out = pipe(messages)[0]["generated_text"]
            last = out[-1]["content"] if isinstance(out, list) else out
            action = extract_action(last) or "SCAN_SCENE"
            result = env.step(action)
            total_r += result.reward
            obs = result.observation
            if result.done:
                break
        rewards.append(total_r)
        if env._all_goals_complete():
            successes += 1

    print(f"\nEval over {n_episodes} episodes:")
    print(f"  Success rate: {successes}/{n_episodes} = {successes/n_episodes:.0%}")
    print(f"  Avg reward:   {sum(rewards)/len(rewards):.2f}")
    return {"success_rate": successes/n_episodes, "avg_reward": sum(rewards)/n_episodes}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval", "dataset"], default="train")
    p.add_argument("--model", default="./outputs/grpo_final")
    args = p.parse_args()

    if args.mode == "train":
        train(hf_space_url=getattr(args, 'hf_space', None))
    elif args.mode == "eval":
        eval_model(args.model)
    elif args.mode == "dataset":
        ds = build_dataset()
        ds.save_to_disk("train/dataset")
        print("Saved to train/dataset/")
