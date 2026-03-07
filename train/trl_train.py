"""
Minimal TRL training script for the tabletop planning environment.
Trains a small LLM to choose the next action given a structured observation.

This satisfies the hackathon requirement:
  "minimal training script for your environment using Unsloth or HF TRL in Colab"

Run on Northflank GPU or Google Colab with:
  pip install trl transformers torch datasets
  python train/trl_train.py
"""
import json
import random
import sys
sys.path.insert(0, ".")

from server.environment import TabletopPlanningEnv
from server.models import Observation
from scripts.scripted_policy import scripted_agent


# ------------------------------------------------------------------ #
#  Step 1: Collect expert trajectories from scripted policy           #
# ------------------------------------------------------------------ #

def obs_to_text(obs: Observation) -> str:
    """Convert structured observation to an LLM-readable prompt."""
    objects_desc = ", ".join(
        f"{o.name}({'reachable' if o.reachable else 'blocked'})"
        for o in obs.visible_objects
    )
    failures_desc = "; ".join(obs.known_failures) or "none"
    subgoals_desc = "; ".join(obs.completed_subgoals) or "none"
    constraints_desc = "; ".join(obs.active_constraints) or "none"

    return f"""[TABLETOP PLANNING]
Instruction: {obs.instruction}
Scene: {objects_desc}
Holding: {obs.holding or 'nothing'}
Completed: {subgoals_desc}
Known failures: {failures_desc}
Constraints: {constraints_desc}
Last action: {obs.last_action or 'none'} -> {obs.last_result or 'n/a'}
Steps remaining: {obs.steps_remaining}

What is the best next action? Choose one of: SCAN_SCENE, MOVE_TO_RED, MOVE_TO_BLUE, MOVE_TO_GREEN, PICK, PLACE_BIN_A, PLACE_BIN_B, CLEAR_BLOCKER

Action:"""


def collect_trajectories(n_episodes: int = 200) -> list[dict]:
    env = TabletopPlanningEnv(use_stub=True)
    data = []
    for _ in range(n_episodes):
        obs = env.reset()
        for _ in range(20):
            action = scripted_agent(obs)
            prompt = obs_to_text(obs)
            data.append({"prompt": prompt, "completion": f" {action}"})
            result = env.step(action)
            obs = result.observation
            if result.done:
                break
    print(f"Collected {len(data)} steps from {n_episodes} expert episodes")
    return data


# ------------------------------------------------------------------ #
#  Step 2: Fine-tune with TRL SFT                                     #
# ------------------------------------------------------------------ #

def train(data: list[dict]):
    try:
        from trl import SFTTrainer, SFTConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import Dataset
    except ImportError:
        print("TRL not installed. Run: pip install trl transformers datasets torch")
        return

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # small enough to run on Colab
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = Dataset.from_list([{"text": d["prompt"] + d["completion"]} for d in data])
    split = dataset.train_test_split(test_size=0.1)

    training_args = SFTConfig(
        output_dir="./outputs/tabletop_sft",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        dataset_text_field="text",
        max_seq_length=512,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    print("Training...")
    trainer.train()
    trainer.save_model("./outputs/tabletop_sft_final")
    print("Done. Model saved to ./outputs/tabletop_sft_final")
    return trainer


if __name__ == "__main__":
    data = collect_trajectories(n_episodes=200)
    # Save for inspection
    with open("train/expert_trajectories.jsonl", "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    print("Trajectories saved to train/expert_trajectories.jsonl")
    train(data)
