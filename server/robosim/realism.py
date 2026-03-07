"""
Realism layer for the tabletop planning environment.

Each function injects a specific real-world failure mode into the sim.
These are the exact things that break in real robot deployments.

Turn them on/off with RealismConfig to control environment difficulty.
"""
import random
from dataclasses import dataclass


@dataclass
class RealismConfig:
    # 1. Probabilistic action failure — gripper slips, push doesn't fully clear
    grasp_fail_prob: float = 0.15       # PICK fails this % of the time even if reachable
    clear_partial_prob: float = 0.20    # CLEAR_BLOCKER partially clears (needs 2nd attempt)

    # 2. Observation noise — perception isn't perfect
    reachability_noise: float = 0.10   # reachable object reported as blocked this % of time
    hidden_object_prob: float = 0.30   # objects not visible until SCAN_SCENE is called

    # 3. World dynamics — things change without the agent acting
    object_drift_prob: float = 0.05    # per step: a reachable object drifts to blocked
    new_obstacle_prob: float = 0.10    # per episode: a random obstacle appears mid-task

    # 4. Delayed/ambiguous reward — success isn't always obvious immediately
    reward_delay_steps: int = 0        # reward shows up N steps after the action (0 = instant)

    @classmethod
    def easy(cls):
        """Close to current clean stub. Good for initial training."""
        return cls(
            grasp_fail_prob=0.0,
            clear_partial_prob=0.0,
            reachability_noise=0.0,
            hidden_object_prob=0.0,
            object_drift_prob=0.0,
            new_obstacle_prob=0.0,
        )

    @classmethod
    def medium(cls):
        """Adds action noise and partial observability. Matches a well-calibrated real robot."""
        return cls(
            grasp_fail_prob=0.15,
            clear_partial_prob=0.20,
            reachability_noise=0.10,
            hidden_object_prob=0.30,
            object_drift_prob=0.0,
            new_obstacle_prob=0.0,
        )

    @classmethod
    def hard(cls):
        """Dynamic world + all noise. Matches a real uncontrolled tabletop."""
        return cls(
            grasp_fail_prob=0.20,
            clear_partial_prob=0.25,
            reachability_noise=0.15,
            hidden_object_prob=0.40,
            object_drift_prob=0.08,
            new_obstacle_prob=0.20,
        )


def apply_action_noise(action: str, result: str, config: RealismConfig) -> str:
    """
    Given a successful action result, maybe flip it to a failure
    to simulate real-world execution uncertainty.
    """
    if result != "SUCCESS":
        return result  # already failed, don't double-fail

    if action == "PICK" and random.random() < config.grasp_fail_prob:
        return "FAILED_SLIP"   # gripper closed but object slipped out

    if action == "CLEAR_BLOCKER" and random.random() < config.clear_partial_prob:
        return "PARTIAL_CLEAR"  # blocker moved but not fully out of the way

    return result


def apply_observation_noise(objects: dict, config: RealismConfig, scanned: bool) -> dict:
    """
    Apply noise to what the agent can observe.

    hidden_object_prob: objects not yet visible without SCAN_SCENE
    reachability_noise: a reachable object sometimes appears blocked
    """
    noisy = {}
    for name, obj in objects.items():
        noisy_obj = dict(obj.__dict__) if hasattr(obj, '__dict__') else dict(obj)

        # Hidden until scanned
        if not scanned and random.random() < config.hidden_object_prob:
            noisy_obj['reachable'] = False
            noisy_obj['_hidden'] = True  # agent doesn't know it's there at all

        # Reachability noise — sensor thinks it's blocked when it isn't
        elif obj.reachable and random.random() < config.reachability_noise:
            noisy_obj['reachable'] = False
            noisy_obj['_noisy'] = True

        noisy[name] = noisy_obj
    return noisy


def apply_world_dynamics(objects: dict, step: int, config: RealismConfig) -> dict:
    """
    Randomly change world state between steps.
    This simulates: objects sliding, another agent acting, vibration, etc.
    """
    if random.random() < config.object_drift_prob:
        # Pick a reachable object and make it drift to blocked
        reachable = [name for name, obj in objects.items()
                     if obj.reachable and not obj.is_held and obj.in_bin is None]
        if reachable:
            drifted = random.choice(reachable)
            objects[drifted].reachable = False
            # find something to mark as its blocker
            for other_name, other in objects.items():
                if other_name != drifted and other.reachable and other.blocking is None:
                    other.blocking = drifted
                    break

    return objects


# ─────────────────────────────────────────────────────────────────────
#  Why each one maps to a real failure mode
# ─────────────────────────────────────────────────────────────────────
#
#  grasp_fail_prob
#    Real: grasping irregular objects with a parallel gripper fails ~15-25%
#    of the time in practice (varies by object shape, friction, placement).
#    The agent needs to learn: retry after slip, adjust approach angle.
#
#  clear_partial_prob
#    Real: a push operation doesn't always fully clear a blocker — the object
#    might still partially occlude the target. Agent must verify and push again.
#
#  reachability_noise
#    Real: depth cameras have noise, segmentation models misclassify objects.
#    An object within reach might be estimated as too far. Agent should
#    attempt approach anyway if it's uncertain.
#
#  hidden_object_prob
#    Real: most grasping scenarios have partial observability — you can't see
#    everything from one camera angle. SCAN_SCENE maps to "move camera" or
#    "look from a different angle." The agent must learn when to gather info.
#
#  object_drift_prob
#    Real: tables vibrate, gripper bumps adjacent objects, another process
#    moves something. Agent must re-verify state before acting on stale info.
#
#  new_obstacle_prob
#    Real: in warehouse / home settings, new objects appear constantly.
#    The agent's plan from 5 steps ago may be invalid now.
