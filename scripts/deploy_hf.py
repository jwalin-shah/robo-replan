"""
Deploy RoboReplan to Hugging Face Spaces.

Usage:
  python scripts/deploy_hf.py --username YOUR_HF_USERNAME

You'll be prompted for your HF token if not already logged in.
Get your token at: https://huggingface.co/settings/tokens
"""
import argparse
import os
import subprocess
from pathlib import Path

def deploy(username: str, repo_name: str = "robo-replan", token: str = None):
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    repo_id = f"{username}/{repo_name}"

    print(f"Creating Space: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            token=token,
            exist_ok=True,
        )
        print(f"Space ready: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Files to upload
    root = Path(__file__).parent.parent
    files = [
        "README.md",
        "Dockerfile",
        "pyproject.toml",
        "viz_standalone.html",
        "server/__init__.py",
        "server/app.py",
        "server/environment.py",
        "server/models.py",
        "server/openenv_env.py",
        "server/robosim/__init__.py",
        "server/robosim/sim_wrapper.py",
        "server/robosim/realism.py",
        "server/robosim/randomizer.py",
        "server/robosim/vision.py",
        "server/config.py",
        "server/curriculum.py",
        "server/logger.py",
    ]

    print("Uploading files...")
    for f in files:
        path = root / f
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f,
                repo_id=repo_id,
                repo_type="space",
                token=token,
            )
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ missing: {f}")

    print(f"\nDone! Your space: https://huggingface.co/spaces/{repo_id}")
    print(f"API endpoint:    https://{username}-{repo_name}.hf.space")
    print(f"\nTo use in training:")
    print(f"  from openenv import AutoEnv")
    print(f"  env = AutoEnv.from_env('{repo_id}')")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--username", required=True, help="Your HF username")
    p.add_argument("--repo",     default="robo-replan", help="Space name")
    p.add_argument("--token",    default=os.environ.get("HF_TOKEN"), help="HF token")
    args = p.parse_args()

    if not args.token:
        args.token = input("HF Token (from https://huggingface.co/settings/tokens): ").strip()

    deploy(args.username, args.repo, args.token)
