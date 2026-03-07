"""
Scripted baseline policy for the tabletop planning environment.

This acts as:
  1. A sanity check that the env works end-to-end
  2. A demonstration of "good" planning behavior
  3. A source of expert trajectories for imitation learning

Run:
  python scripts/scripted_policy.py
"""
import sys
import json
sys.path.insert(0, ".")

from server.environment import TabletopPlanningEnv
from server.models import Observation


def scripted_agent(obs: Observation) -> str:
    """
    Rule-based planning agent.
    Priority order:
      1. If we just moved to an object successfully -> pick it
      2. If holding something -> place it in the correct bin
      3. If blocked and failed before -> clear blocker
      4. If red reachable -> move to red
      5. If green reachable (multi-block task) -> move to green
      6. If blocker present -> clear it
      7. Default: scan
    """
    holding = obs.holding
    failures = set(obs.known_failures)
    reachable = {o.name: o.reachable for o in obs.visible_objects}
    completed = set(obs.completed_subgoals)

    # If we just moved to an object, pick it now
    if obs.last_action in ("MOVE_TO_RED", "MOVE_TO_BLUE", "MOVE_TO_GREEN") and obs.last_result == "SUCCESS":
        return "PICK"

    # Already holding something -> place it
    if holding == "red_block":
        return "PLACE_BIN_A"
    if holding == "green_block":
        return "PLACE_BIN_B"

    # If we failed to reach red because blocked -> clear blocker
    if "MOVE_TO_RED:FAILED_BLOCKED" in failures or "PICK:FAILED_EMPTY" in failures:
        return "CLEAR_BLOCKER"

    # Red not yet placed
    if "placed_red_block_in_bin_A" not in completed:
        if reachable.get("red_block"):
            return "MOVE_TO_RED"
        return "CLEAR_BLOCKER"

    # Green not yet placed (multi-block task)
    if "placed_green_block_in_bin_B" not in completed:
        if reachable.get("green_block"):
            return "MOVE_TO_GREEN"

    return "SCAN_SCENE"


def run_episode(env: TabletopPlanningEnv, verbose: bool = True) -> dict:
    obs = env.reset()
    total_reward = 0.0
    trajectory = []

    if verbose:
        print(f"\n=== Episode ===")
        print(f"Instruction: {obs.instruction}")
        print(f"Scene: {[o.name + ('(blocked)' if not o.reachable else '') for o in obs.visible_objects]}")

    for step in range(20):
        action = scripted_agent(obs)
        result = env.step(action)

        total_reward += result.reward
        trajectory.append({
            "step": step,
            "action": action,
            "result": result.info["result"],
            "reward": result.reward,
        })

        if verbose:
            status = result.info["result"]
            print(f"  step {step:2d}: {action:<20} -> {status:<20} reward={result.reward:+.2f}  total={total_reward:+.2f}")

        obs = result.observation
        if result.done:
            break

    success = env._all_goals_complete()
    if verbose:
        print(f"Done. Success={success}  Total reward={total_reward:.2f}")
    return {"success": success, "total_reward": total_reward, "steps": len(trajectory)}


if __name__ == "__main__":
    env = TabletopPlanningEnv(use_stub=True)
    results = []
    N = 10
    for i in range(N):
        r = run_episode(env, verbose=(i == 0))  # only verbose on first
        results.append(r)

    successes = sum(r["success"] for r in results)
    avg_reward = sum(r["total_reward"] for r in results) / N
    avg_steps = sum(r["steps"] for r in results) / N
    print(f"\n=== Summary over {N} episodes ===")
    print(f"Success rate: {successes}/{N} = {successes/N:.0%}")
    print(f"Avg reward:   {avg_reward:.2f}")
    print(f"Avg steps:    {avg_steps:.1f}")
