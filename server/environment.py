"""
TabletopPlanningEnv — fully instrumented RL training environment.

Every knob lives in EnvConfig. Every step is logged. Curriculum auto-advances.
The observation tells the model everything it needs to plan well.
"""
import random
from typing import Optional

from .config import EnvConfig, RealismConfig
from .logger import EpisodeLogger
from .models import Action, ObjectInfo, Observation, StepResult
from .robosim import SimWrapper
from .robosim.randomizer import randomize_scenario


class TabletopPlanningEnv:
    def __init__(self, config: EnvConfig = None, use_stub: bool = True):
        self.cfg = config or EnvConfig.easy()
        self.sim = SimWrapper(use_stub=use_stub)
        self.logger = EpisodeLogger(
            export_path=self.cfg.log.export_path,
            max_history=self.cfg.log.max_episode_history,
        )
        self._episode_id = 0
        self._cumulative_reward = 0.0
        self._action_history: list[str] = []
        self._mid_task_changed = False
        self._reset_internal()

    def _nav_enabled(self) -> bool:
        return bool(getattr(self.cfg.task, "navigation_mode", False))

    def _gripper_cell(self) -> tuple[int, int]:
        p = self.sim.get_state().gripper_pos
        return int(round(float(p[0]) / 0.1)), int(round(float(p[1]) / 0.1))

    def _object_cell(self, obj_name: str) -> Optional[tuple[int, int]]:
        obj = self.sim.get_state().objects.get(obj_name)
        if obj is None:
            return None
        return int(round(float(obj.pos[0]) / 0.1)), int(round(float(obj.pos[1]) / 0.1))

    def _is_adjacent_to(self, obj_name: str) -> bool:
        oc = self._object_cell(obj_name)
        if oc is None:
            return False
        gx, gy = self._gripper_cell()
        ox, oy = oc
        return abs(gx - ox) + abs(gy - oy) <= 1

    def _nav_step_toward(self, target: tuple[int, int]) -> str:
        gx, gy = self._gripper_cell()
        tx, ty = target
        if tx > gx:
            return "MOVE_EAST"
        if tx < gx:
            return "MOVE_WEST"
        if ty > gy:
            return "MOVE_NORTH"
        if ty < gy:
            return "MOVE_SOUTH"
        return "SCAN_SCENE"

    # ── Public interface ────────────────────────────────────────────────

    def reset(self) -> Observation:
        self._reset_internal()
        return self._build_obs(last_action=None, last_result=None)

    def step(self, action: str, reasoning: str = "") -> StepResult:
        """
        action:    the high-level action string
        reasoning: optional <think>...</think> chain-of-thought from the model.
                   Rewarded if it mentions the right objects and constraints.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # Inject mid-task instruction change if configured
        if (not self._mid_task_changed
                and self.cfg.task.mid_task_change_prob > 0
                and self._steps == self.cfg.task.mid_task_change_step
                and random.random() < self.cfg.task.mid_task_change_prob
                and not self._done):
            self._apply_mid_task_change()

        raw_result = self.sim.execute(action)
        result = self._apply_noise(action, raw_result)

        if result == "FAILED_SLIP" and raw_result == "SUCCESS" and action == "PICK":
            state = self.sim.get_state()
            if state.holding:
                state.objects[state.holding].is_held = False
                state.holding = None

        if action == "SCAN_SCENE" and result == "SUCCESS":
            self._scanned = True

        self._apply_world_drift()
        self._action_history.append(action)

        self._steps += 1
        reward = self._compute_reward(action, result)
        reward += self._reasoning_bonus(reasoning, action, result)
        self._cumulative_reward += reward
        self._update_planning_state(action, result)

        # Oracle hint for logging / observation
        oracle = self._oracle_action()

        if self.cfg.log.log_every_step:
            self.logger.log_step(
                step=self._steps,
                action=action,
                result=result,
                reward=reward,
                cumulative_reward=self._cumulative_reward,
                valid_actions=self._valid_actions(),
                oracle_action=oracle if self.cfg.obs.include_oracle_hint else None,
                holding=self.sim.get_state().holding,
                n_failures=len(self._known_failures),
                n_subgoals=len(self._completed_subgoals),
            )

        done = self._check_done()
        if done:
            ep = self.logger.end_episode(success=self._all_goals_complete())
            self.logger.metrics._current_difficulty = self.cfg.log.export_path  # track level

        obs = self._build_obs(last_action=action, last_result=result)
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "result": result,
                "step": self._steps,
                "oracle_action": oracle,
                "valid_actions": self._valid_actions(),
                "goal_progress": self._goal_progress(),
                "mid_task_changed": self._mid_task_changed and self._steps == self.cfg.task.mid_task_change_step + 1,
                "cumulative_reward": self._cumulative_reward,
            },
        )

    @property
    def metrics(self):
        return self.logger.metrics.to_dict()

    # ── Internal reset ──────────────────────────────────────────────────

    def _reset_internal(self):
        tc = self.cfg.task
        force_blocked = random.random() < tc.force_blocked_prob
        scenario_cfg = randomize_scenario(
            n_objects=random.randint(tc.n_objects_min, tc.n_objects_max),
            n_targets=random.randint(tc.n_targets_min, tc.n_targets_max),
            n_blockers=random.randint(tc.n_blockers_min, tc.n_blockers_max),
            force_blocked=force_blocked,
        )

        self.sim._build_state_from_config(scenario_cfg)
        self._scenario_cfg = scenario_cfg

        self._steps = 0
        self._done = False
        self._scanned = False
        self._mid_task_changed = False
        self._cumulative_reward = 0.0
        self._action_history = []
        self._completed_subgoals: list[str] = []
        self._known_failures: list[str] = []
        self._active_constraints: list[str] = ([scenario_cfg.constraint]
                                                if scenario_cfg.constraint else [])
        self._instruction = scenario_cfg.instruction
        self._required_placements: dict[str, str] = dict(scenario_cfg.targets)

        self._episode_id += 1
        self.logger.begin_episode(
            episode_id=self._episode_id,
            instruction=self._instruction,
            difficulty="custom",
            n_objects=len(scenario_cfg.objects),
            n_blockers=len(scenario_cfg.blockers),
            n_targets=len(scenario_cfg.targets),
        )

    # ── Reward ──────────────────────────────────────────────────────────

    def _reasoning_bonus(self, reasoning: str, action: str, result: str) -> float:
        """
        Small bonus for reasoning that mentions relevant objects/constraints.
        Encourages the model to develop coherent internal planning language.
        Not large enough to game — just nudges toward explainable behavior.
        """
        if not reasoning or len(reasoning) < 10:
            return 0.0
        bonus = 0.0
        r = reasoning.lower()
        state = self.sim.get_state()

        # Mentions blocked objects correctly
        for obj in state.objects.values():
            if not obj.reachable and obj.name.replace("_block", "") in r:
                bonus += 0.1

        # Mentions the target object and correct bin
        for obj_name, bin_name in self._required_placements.items():
            color = obj_name.replace("_block", "")
            if color in r and f"bin {bin_name.lower()}" in r:
                bonus += 0.2

        # Mentions relevant constraint
        for c in self._active_constraints:
            if c.replace("_", " ") in r:
                bonus += 0.1

        # Mentions the chosen action or its intent
        action_words = {
            "CLEAR_BLOCKER": ["clear", "move", "push", "unblock"],
            "PICK": ["pick", "grab", "grasp", "lift"],
            "PLACE_BIN_A": ["place", "put", "bin a"],
            "PLACE_BIN_B": ["place", "put", "bin b"],
            "SCAN_SCENE": ["scan", "look", "inspect", "check"],
        }
        for word in action_words.get(action, []):
            if word in r:
                bonus += 0.1
                break

        return min(bonus, 0.5)  # cap at 0.5 so it never dominates task reward

    def _compute_reward(self, action: str, result: str) -> float:
        w = self.cfg.reward
        r = w.step_cost

        if result not in ("SUCCESS", "PARTIAL_CLEAR"):
            failure_key = f"{action}:{result}"
            r += w.repeated_failure if failure_key in self._known_failures else w.first_failure
            return r

        if action == "CLEAR_BLOCKER":
            r += w.blocker_cleared
        if action == "PICK":
            r += w.successful_pick
        if action in ("PLACE_BIN_A", "PLACE_BIN_B"):
            bin_name = "A" if action == "PLACE_BIN_A" else "B"
            state = self.sim.get_state()
            correct = any(
                state.objects[n].in_bin == bin_name
                for n, req in self._required_placements.items()
                if req == bin_name and n in state.objects
            )
            r += w.correct_placement if correct else w.wrong_bin
            if not correct and self._active_constraints:
                r += w.constraint_violation  # extra hit for constraint violation
        if action == "SCAN_SCENE" and not self._scanned:
            r += w.useful_scan  # first scan only

        # First recovery after failure
        if self._known_failures and result == "SUCCESS" and action != "SCAN_SCENE":
            if "recovery" not in self._completed_subgoals:
                r += w.recovery_after_failure

        # Terminal
        if self._all_goals_complete():
            r += w.task_complete
            steps_saved = self.cfg.task.max_steps - self._steps
            r += w.efficiency_bonus_max * (steps_saved / self.cfg.task.max_steps)
            self._done = True

        return r

    # ── Planning state ──────────────────────────────────────────────────

    def _update_planning_state(self, action: str, result: str):
        if result not in ("SUCCESS", "PARTIAL_CLEAR"):
            key = f"{action}:{result}"
            if key not in self._known_failures:
                self._known_failures.append(key)
        else:
            if action == "CLEAR_BLOCKER" and "cleared_blocker" not in self._completed_subgoals:
                self._completed_subgoals.append("cleared_blocker")
            if (self._known_failures and result == "SUCCESS"
                    and "recovery" not in self._completed_subgoals):
                self._completed_subgoals.append("recovery")

        state = self.sim.get_state()
        for obj_name, bin_name in self._required_placements.items():
            key = f"placed_{obj_name}_in_bin_{bin_name}"
            if key not in self._completed_subgoals:
                obj = state.objects.get(obj_name)
                if obj and obj.in_bin == bin_name:
                    self._completed_subgoals.append(key)

        if self._steps >= self.cfg.task.max_steps:
            self._done = True

    def _check_done(self) -> bool:
        return self._done

    def _all_goals_complete(self) -> bool:
        state = self.sim.get_state()
        for name, bin_name in self._required_placements.items():
            obj = state.objects.get(name)
            if not obj or obj.in_bin != bin_name:
                return False
        return True

    # ── Noise / dynamics ────────────────────────────────────────────────

    def _apply_noise(self, action: str, result: str) -> str:
        if result != "SUCCESS":
            return result
        rc = self.cfg.realism
        if action == "PICK" and random.random() < rc.grasp_fail_prob:
            return "FAILED_SLIP"
        if action == "CLEAR_BLOCKER" and random.random() < rc.clear_partial_prob:
            return "PARTIAL_CLEAR"
        return result

    def _apply_world_drift(self):
        if random.random() < self.cfg.realism.object_drift_prob:
            state = self.sim.get_state()
            reachable = [o for o in state.objects.values()
                         if o.reachable and not o.is_held and o.in_bin is None]
            if reachable:
                obj = random.choice(reachable)
                obj.reachable = False

    # ── Mid-task instruction change ─────────────────────────────────────

    def _apply_mid_task_change(self):
        """Swap one target's bin. Agent must replan."""
        from .robosim.randomizer import BINS
        targets = list(self._required_placements.items())
        if not targets:
            return
        obj_name, old_bin = random.choice(targets)
        new_bin = [b for b in BINS if b != old_bin][0]
        self._required_placements[obj_name] = new_bin
        self._mid_task_changed = True
        # Rebuild instruction to reflect change
        from .robosim.randomizer import OBJECT_COLORS
        color = OBJECT_COLORS.get(obj_name, obj_name.replace("_block", ""))
        change_note = f" [UPDATE: place the {color} block in bin {new_bin} instead.]"
        self._instruction = self._instruction + change_note
        self._active_constraints.append("bin_change")

    # ── Valid actions ────────────────────────────────────────────────────

    def _valid_actions(self) -> list[str]:
        """Which actions make sense right now given the current state."""
        state = self.sim.get_state()
        valid = ["SCAN_SCENE"]

        if self._nav_enabled():
            valid += ["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST", "ROTATE_LEFT", "ROTATE_RIGHT"]
        else:
            for obj in state.objects.values():
                if obj.reachable and not obj.is_held and obj.in_bin is None:
                    color = obj.name.replace("_block", "").upper()
                    valid.append(f"MOVE_TO_{color}")

        if state.holding:
            valid += ["PLACE_BIN_A", "PLACE_BIN_B"]
        else:
            has_pick = False
            for obj in state.objects.values():
                if not (obj.reachable and not obj.is_held and obj.in_bin is None):
                    continue
                if self._nav_enabled():
                    if self._is_adjacent_to(obj.name):
                        has_pick = True
                        break
                else:
                    has_pick = True
                    break
            if has_pick:
                valid.append("PICK")

        for obj in state.objects.values():
            if not (obj.blocking and obj.reachable):
                continue
            if self._nav_enabled() and not self._is_adjacent_to(obj.name):
                continue
            valid.append("CLEAR_BLOCKER")
            break

        return valid

    # ── Goal progress ────────────────────────────────────────────────────

    def _goal_progress(self) -> float:
        if not self._required_placements:
            return 1.0
        state = self.sim.get_state()
        done = sum(1 for name, bin_ in self._required_placements.items()
                   if state.objects.get(name) and state.objects[name].in_bin == bin_)
        return done / len(self._required_placements)

    # ── Oracle hint ──────────────────────────────────────────────────────

    def _oracle_action(self) -> Optional[str]:
        """Scripted optimal action for current state (teaching signal)."""
        state = self.sim.get_state()
        failures = set(self._known_failures)
        completed = set(self._completed_subgoals)
        last_action = self._action_history[-1] if self._action_history else None

        def can_clear_now() -> bool:
            for obj in state.objects.values():
                if not (obj.blocking and obj.reachable):
                    continue
                if self._nav_enabled() and not self._is_adjacent_to(obj.name):
                    continue
                return True
            return False

        # Just moved to something → pick it
        if last_action and last_action.startswith("MOVE_TO"):
            return "PICK"

        # Holding → place correctly
        if state.holding:
            target_bin = self._required_placements.get(state.holding)
            if target_bin:
                return f"PLACE_BIN_{target_bin}"
            return "PLACE_BIN_A"

        # Failed to reach a target → clear
        if any(f.startswith("MOVE_TO") and "FAILED_BLOCKED" in f for f in failures) and can_clear_now():
            return "CLEAR_BLOCKER"
        if "PICK:FAILED_EMPTY" in failures and can_clear_now():
            return "CLEAR_BLOCKER"

        # Work through required placements in order
        for obj_name, bin_name in self._required_placements.items():
            key = f"placed_{obj_name}_in_bin_{bin_name}"
            if key in completed:
                continue
            obj = state.objects.get(obj_name)
            if not obj or obj.in_bin:
                continue
            if obj.reachable:
                if self._nav_enabled():
                    if self._is_adjacent_to(obj_name):
                        return "PICK"
                    target = self._object_cell(obj_name)
                    if target is not None:
                        return self._nav_step_toward(target)
                color = obj_name.replace("_block", "").upper()
                return f"MOVE_TO_{color}"
            return "CLEAR_BLOCKER"

        return "SCAN_SCENE"

    # ── Observation ──────────────────────────────────────────────────────

    def _build_obs(self, last_action: Optional[str], last_result: Optional[str]) -> Observation:
        state = self.sim.get_state()
        oc = self.cfg.obs

        visible = []
        for obj in state.objects.values():
            # Apply observation noise
            reachable = obj.reachable
            if (not self._scanned and
                    random.random() < self.cfg.realism.hidden_object_prob):
                reachable = False
            elif (obj.reachable and
                  random.random() < self.cfg.realism.reachability_noise):
                reachable = False

            visible.append(ObjectInfo(
                name=obj.name,
                reachable=reachable,
                location="unknown" if not reachable else "table",
                blocking=obj.blocking,
            ))

        # Recent action history
        history = (self._action_history[-oc.include_action_history:]
                   if oc.include_action_history > 0 else [])

        extra = {}
        if oc.include_valid_actions:
            extra["valid_actions"] = self._valid_actions()
        if oc.include_goal_progress:
            extra["goal_progress"] = round(self._goal_progress(), 2)
        if oc.include_oracle_hint:
            extra["oracle_hint"] = self._oracle_action()
        if oc.include_distance_to_goal:
            remaining = sum(1 for n, b in self._required_placements.items()
                            if not (state.objects.get(n) and state.objects[n].in_bin == b))
            extra["goals_remaining"] = remaining

        return Observation(
            instruction=self._instruction,
            steps_remaining=self.cfg.task.max_steps - self._steps,
            visible_objects=visible,
            holding=state.holding,
            completed_subgoals=list(self._completed_subgoals),
            known_failures=list(self._known_failures),
            active_constraints=list(self._active_constraints),
            last_action=last_action,
            last_result=last_result,
            action_history=history,
            nav_mode=self._nav_enabled(),
            gripper_cell=f"{self._gripper_cell()[0]},{self._gripper_cell()[1]}",
            gripper_facing=self.sim.get_facing(),
            **extra,
        )
