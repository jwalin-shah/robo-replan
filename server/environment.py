"""
TabletopPlanningEnv — OpenEnv environment for multi-step robotic task planning.

The agent must complete manipulation tasks (place target objects in bins) while
handling blockers, hidden state, and mid-task constraint changes.

This is the planning challenge — not low-level control. The sim handles motion.
"""
import random
from typing import Optional

from .models import Action, ObjectInfo, Observation, StepResult
from .robosim import SimWrapper
from .robosim.realism import RealismConfig, apply_action_noise, apply_world_dynamics


MAX_STEPS = 20


class TabletopPlanningEnv:
    def __init__(self, use_stub: bool = True, realism: RealismConfig = None):
        self.sim = SimWrapper(use_stub=use_stub)
        self.realism = realism or RealismConfig.easy()
        self._scanned = False
        self._reset_internal()

    # ------------------------------------------------------------------ #
    #  OpenEnv interface                                                   #
    # ------------------------------------------------------------------ #

    def reset(self) -> Observation:
        self._reset_internal()
        return self._build_obs(last_action=None, last_result=None)

    def step(self, action: str) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        raw_result = self.sim.execute(action)

        # Apply real-world action noise (grasp slips, partial clears)
        # Note: if the sim already failed, noise doesn't apply
        result = apply_action_noise(action, raw_result, self.realism)

        # FAILED_SLIP: sim already applied the pick, undo it
        if result == "FAILED_SLIP" and raw_result == "SUCCESS" and action == "PICK":
            state = self.sim.get_state()
            if state.holding:
                state.objects[state.holding].is_held = False
                state.holding = None

        # If SCAN_SCENE succeeded, mark as scanned (reveals hidden objects)
        if action == "SCAN_SCENE" and result == "SUCCESS":
            self._scanned = True

        # World dynamics: things drift between steps
        apply_world_dynamics(self.sim.get_state().objects, self._steps, self.realism)

        self._steps += 1
        reward = self._compute_reward(action, result)
        self._update_planning_state(action, result)
        done = self._check_done()
        obs = self._build_obs(last_action=action, last_result=result)

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={"result": result, "step": self._steps},
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _reset_internal(self):
        _, cfg = self.sim.reset(scenario="random")
        self._scanned = False
        self._steps = 0
        self._done = False
        self._completed_subgoals: list[str] = []
        self._known_failures: list[str] = []
        self._active_constraints: list[str] = ([cfg.constraint] if cfg.constraint else [])
        self._instruction = cfg.instruction
        self._required_placements: dict[str, str] = dict(cfg.targets)

    def _compute_reward(self, action: str, result: str) -> float:
        reward = -0.05  # step cost

        if result == "FAILED_BLOCKED" or result == "FAILED_EMPTY" or result == "FAILED_INVALID":
            failure_key = f"{action}:{result}"
            if failure_key in self._known_failures:
                reward -= 2.0  # repeated same failure
            else:
                reward -= 1.0
            return reward

        # Positive events
        if action == "CLEAR_BLOCKER" and result == "SUCCESS":
            reward += 2.0
            if "cleared_blocker" not in self._completed_subgoals:
                self._completed_subgoals.append("cleared_blocker")

        if action == "PICK" and result == "SUCCESS":
            reward += 2.0

        if action in ("PLACE_BIN_A", "PLACE_BIN_B") and result == "SUCCESS":
            bin_name = "A" if action == "PLACE_BIN_A" else "B"
            state = self.sim.get_state()
            correct = any(
                state.objects[name].in_bin == bin_name
                for name, req_bin in self._required_placements.items()
                if req_bin == bin_name and name in state.objects
            )
            reward += 2.0 if correct else -3.0

        # Bonus: first useful corrective action after a failure
        if self._known_failures and result == "SUCCESS" and action != "SCAN_SCENE":
            if "recovery" not in self._completed_subgoals:
                reward += 1.0
                self._completed_subgoals.append("recovery")

        # Terminal bonus if all required placements done
        if self._all_goals_complete():
            reward += 10.0
            self._done = True

        return reward

    def _update_planning_state(self, action: str, result: str):
        if result not in ("SUCCESS",):
            failure_key = f"{action}:{result}"
            if failure_key not in self._known_failures:
                self._known_failures.append(failure_key)

        state = self.sim.get_state()
        for obj_name, bin_name in self._required_placements.items():
            placed_key = f"placed_{obj_name}_in_bin_{bin_name}"
            if placed_key not in self._completed_subgoals:
                obj = state.objects.get(obj_name)
                if obj and obj.in_bin == bin_name:
                    self._completed_subgoals.append(placed_key)

        if self._steps >= MAX_STEPS:
            self._done = True

    def _check_done(self) -> bool:
        return self._done

    def _all_goals_complete(self) -> bool:
        state = self.sim.get_state()
        for obj_name, bin_name in self._required_placements.items():
            obj = state.objects.get(obj_name)
            if not obj or obj.in_bin != bin_name:
                return False
        return True

    def _build_obs(self, last_action: Optional[str], last_result: Optional[str]) -> Observation:
        state = self.sim.get_state()
        visible = []
        for obj in state.objects.values():
            visible.append(ObjectInfo(
                name=obj.name,
                reachable=obj.reachable,
                location="unknown" if not obj.reachable and not obj.blocking else "center",
                blocking=obj.blocking,
            ))
        return Observation(
            instruction=self._instruction,
            steps_remaining=MAX_STEPS - self._steps,
            visible_objects=visible,
            holding=state.holding,
            completed_subgoals=list(self._completed_subgoals),
            known_failures=list(self._known_failures),
            active_constraints=list(self._active_constraints),
            last_action=last_action,
            last_result=last_result,
        )
