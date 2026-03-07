"""
Evaluation script: run N episodes with a policy and report metrics.
Use this to show before/after training improvement.

Usage:
  python scripts/eval_before_after.py --policy scripted
  python scripts/eval_before_after.py --policy random
"""
import sys
import random
import argparse
sys.path.insert(0, ".")

from server.environment import TabletopPlanningEnv
from server.models import Observation, Action

ACTIONS = [a.value for a in Action]


def random_agent(obs: Observation) -> str:
    return random.choice(ACTIONS)


def run_eval(policy_fn, n_episodes: int = 20) -> dict:
    env = TabletopPlanningEnv(use_stub=True)
    rewards, successes, steps_list = [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        for step in range(20):
            action = policy_fn(obs)
            result = env.step(action)
            total_reward += result.reward
            obs = result.observation
            if result.done:
                break
        rewards.append(total_reward)
        successes.append(env._all_goals_complete())
        steps_list.append(step + 1)
    return {
        "success_rate": sum(successes) / n_episodes,
        "avg_reward": sum(rewards) / n_episodes,
        "avg_steps": sum(steps_list) / n_episodes,
        "rewards": rewards,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    from scripts.scripted_policy import scripted_agent
    policy = scripted_agent if args.policy == "scripted" else random_agent

    print(f"Evaluating {args.policy} policy over {args.n} episodes...")
    metrics = run_eval(policy, args.n)
    print(f"Success rate: {metrics['success_rate']:.0%}")
    print(f"Avg reward:   {metrics['avg_reward']:.2f}")
    print(f"Avg steps:    {metrics['avg_steps']:.1f}")
