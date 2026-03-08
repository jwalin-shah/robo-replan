"""
Scripted baseline policy for the tabletop planning environment.

This acts as:
  1. A sanity check that the env works end-to-end
  2. A demonstration of "good" planning behavior
  3. A source of expert trajectories for imitation learning

Run:
  python scripts/scripted_policy.py
"""
import re
import sys
sys.path.insert(0, ".")

from server.environment import TabletopPlanningEnv
from server.models import Observation


def _parse_cell(s: str):
    """Parse 'x,y' string into (x, y) int tuple."""
    if not s:
        return None
    try:
        parts = s.split(",")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None


def _nav_step(gx: int, gy: int, tx: int, ty: int) -> str:
    """One greedy step toward target cell."""
    if tx > gx:
        return "MOVE_EAST"
    if tx < gx:
        return "MOVE_WEST"
    if ty > gy:
        return "MOVE_NORTH"
    if ty < gy:
        return "MOVE_SOUTH"
    return "PICK"  # already on target cell


def _parse_required_placements(instruction: str, visible_objects=None) -> dict:
    """
    Extract {object_name: bin_label} from a natural-language instruction.
    Works for both default block names and professional pack names.
    """
    placements = {}

    # 1. Try color-based matching (default pack)
    color_pattern = r'\b(red|blue|green|yellow|purple)\b[^.]*?\bin\s+bin\s*([AB])\b'
    for m in re.finditer(color_pattern, instruction, re.IGNORECASE):
        placements[f"{m.group(1).lower()}_block"] = m.group(2).upper()
    if placements:
        return placements

    # 2. General pattern: "the X in bin A/B" — convert "heavy pallet" → "heavy_pallet"
    general_pattern = r'\bthe\s+([\w][^.]*?)\s+in\s+bin\s+([AB])\b'
    for m in re.finditer(general_pattern, instruction, re.IGNORECASE):
        display = m.group(1).strip()
        bin_label = m.group(2).upper()
        obj_name = display.lower().replace(" ", "_")
        placements[obj_name] = bin_label

    # 3. Cross-check against visible objects for fuzzy name matching
    if visible_objects and not placements:
        for obj in visible_objects:
            for bin_label in ("A", "B"):
                if obj.name.replace("_", " ") in instruction.lower() and f"bin {bin_label}" in instruction:
                    placements[obj.name] = bin_label

    return placements


def scripted_agent(obs: Observation) -> str:
    """
    Rule-based planning agent — nav-mode aware.

    Priority order (both modes):
      1. Holding something → place it in the correct bin
      2. (Nav mode) PICK in valid_actions → we're adjacent, grab it
      3. (Nav mode) CLEAR_BLOCKER in valid_actions → clear adjacent blocker
      4. (Nav mode) Navigate toward next_target_cell
      5. (Direct mode) Just moved to object → PICK
      6. (Direct mode) Known failure → CLEAR_BLOCKER
      7. (Direct mode) MOVE_TO_<COLOR> for first incomplete target
      8. Fallback: SCAN_SCENE
    """
    valid = set(obs.valid_actions or [])
    holding = obs.holding
    failures = set(obs.known_failures)
    completed = set(obs.completed_subgoals)

    # Parse required placements from instruction (pass visible objects for fuzzy matching)
    required = _parse_required_placements(obs.instruction, obs.visible_objects)

    # ── 1. Always: if holding, place in correct bin ──────────────────────
    if holding:
        bin_name = required.get(holding)
        if bin_name:
            return f"PLACE_BIN_{bin_name}"
        # Fallback: prefer whichever placement is mentioned first
        if "PLACE_BIN_A" in valid:
            return "PLACE_BIN_A"
        return "PLACE_BIN_B"

    # ── Navigation mode ──────────────────────────────────────────────────
    if obs.nav_mode:
        # Adjacent to a pickable object → pick it now
        if "PICK" in valid:
            return "PICK"

        # Adjacent to a blocker → clear it now
        if "CLEAR_BLOCKER" in valid:
            return "CLEAR_BLOCKER"

        # Navigate toward next incomplete goal
        gripper = _parse_cell(obs.gripper_cell)
        target = _parse_cell(obs.next_target_cell)
        if gripper and target:
            gx, gy = gripper
            tx, ty = target
            if (gx, gy) != (tx, ty):
                return _nav_step(gx, gy, tx, ty)
            # On target cell — pick or scan to update state
            return "PICK"

        # No navigable target — scan once to reveal hidden objects, then move
        recent = list(obs.action_history or [])[-3:]
        if recent.count("SCAN_SCENE") < 2:
            return "SCAN_SCENE"
        # Already scanned repeatedly with no progress — move toward any reachable obj
        for move in ("MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST"):
            if move in valid:
                return move
        return "SCAN_SCENE"

    # ── Direct (non-nav) mode ────────────────────────────────────────────
    reachable = {o.name: o.reachable for o in obs.visible_objects}

    # Just moved to an object → pick it
    if (obs.last_action and obs.last_action.startswith("MOVE_TO_")
            and obs.last_result == "SUCCESS"):
        return "PICK"

    # Known failure → try to clear blocker
    if (any("FAILED_BLOCKED" in f for f in failures)
            or "PICK:FAILED_EMPTY" in failures):
        if "CLEAR_BLOCKER" in valid:
            return "CLEAR_BLOCKER"

    # Work through targets in order; try each unfinished target
    cleared_attempted = False
    for obj_name, bin_name in required.items():
        goal_key = f"placed_{obj_name}_in_bin_{bin_name}"
        if goal_key in completed:
            continue
        color = obj_name.replace("_block", "").upper()
        move_action = f"MOVE_TO_{color}"
        if reachable.get(obj_name) and move_action in valid:
            return move_action
        # This target is blocked — try to clear once (but don't repeat per target)
        if not cleared_attempted and "CLEAR_BLOCKER" in valid:
            cleared_attempted = True
            return "CLEAR_BLOCKER"
        # Can't act on this target right now — try the next one

    return "SCAN_SCENE"


def run_episode(env: TabletopPlanningEnv, verbose: bool = True) -> dict:
    obs = env.reset()
    total_reward = 0.0
    trajectory = []

    if verbose:
        print(f"\n=== Episode ===")
        print(f"Instruction: {obs.instruction}")
        print(f"Nav mode: {obs.nav_mode}")
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
    from server.config import EnvConfig

    for difficulty in ("easy", "medium", "hard"):
        cfg = {"easy": EnvConfig.easy, "medium": EnvConfig.medium, "hard": EnvConfig.hard}[difficulty]()
        env = TabletopPlanningEnv(config=cfg, use_stub=True)
        results = []
        N = 20
        verbose_first = (difficulty == "easy")
        for i in range(N):
            r = run_episode(env, verbose=(i == 0 and verbose_first))
            results.append(r)

        successes = sum(r["success"] for r in results)
        avg_reward = sum(r["total_reward"] for r in results) / N
        avg_steps = sum(r["steps"] for r in results) / N
        print(f"\n=== {difficulty.upper()} — {N} episodes ===")
        print(f"Success rate: {successes}/{N} = {successes/N:.0%}")
        print(f"Avg reward:   {avg_reward:.2f}")
        print(f"Avg steps:    {avg_steps:.1f}")
