#!/usr/bin/env python3
"""
Push local GRPO (or SFT) weights to Hugging Face Hub.
Run from repo root, or from train/ with MODEL_DIR pointing to outputs.

  cd /home/jovyan/robo-replan
  huggingface-cli login   # one-time, paste your token
  python3 scripts/push_grpo_to_hub.py
  # Or: REPO_ID=yourusername/robo-replan-grpo MODEL_DIR=./outputs/sft_final python3 scripts/push_grpo_to_hub.py
"""
import os
import sys

REPO_ID = os.environ.get("REPO_ID", "jshah13/robo-replan-grpo")
MODEL_DIR_ENV = os.environ.get("MODEL_DIR", "")

def main():
    model_dir = MODEL_DIR_ENV or (
        "train/outputs/grpo_final" if os.path.isdir("train/outputs/grpo_final") else "outputs/grpo_final"
    )
    if not os.path.isdir(model_dir):
        print(f"Model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(1)
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(f"Pushing to https://huggingface.co/{REPO_ID} ...")
    HfApi().create_repo(REPO_ID, repo_type="model", exist_ok=True)
    model.push_to_hub(REPO_ID)
    tokenizer.push_to_hub(REPO_ID)
    print(f"Done → https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
