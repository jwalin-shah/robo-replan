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
        self._last_action: Optional[str] = None
        self._last_result: Optional[str] = None
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
        return abs(gx - ox) + abs(gy - oy) <= 2

    def _is_facing_object(self, obj_name: str) -> bool:
        oc = self._object_cell(obj_name)
        if oc is None:
            return False
        gx, gy = self._gripper_cell()
        ox, oy = oc
        dx, dy = (ox - gx), (oy - gy)
        facing = self.sim.get_facing()
        forward = {
            "N": (0, 1),
            "S": (0, -1),
            "E": (1, 0),
            "W": (-1, 0),
        }.get(facing, (0, 1))
        return (dx, dy) == forward

    def _can_pick_object(self, obj_name: str) -> bool:
        obj = self.sim.get_state().objects.get(obj_name)
        if obj is None or not obj.reachable or obj.is_held or obj.in_bin is not None:
            return False
        if self._nav_enabled():
            return self._is_adjacent_to(obj_name)
        gp = self.sim.get_state().gripper_pos
        dx = float(gp[0]) - float(obj.pos[0])
        dy = float(gp[1]) - float(obj.pos[1])
        return (dx * dx + dy * dy) ** 0.5 < 0.15

    def _next_goal_cell(self) -> Optional[tuple[int, int]]:
        state = self.sim.get_state()
        for obj_name, bin_name in self._required_placements.items():
            obj = state.objects.get(obj_name)
            if not obj or obj.in_bin == bin_name:
                continue
            if obj.reachable:
                return self._object_cell(obj_name)
            for blocker in state.objects.values():
                if blocker.blocking == obj_name and blocker.reachable and blocker.in_bin is None:
                    return self._object_cell(blocker.name)
        return None

    def _distance_to_next_goal(self) -> Optional[int]:
        cell = self._next_goal_cell()
        if cell is None:
            return None
        gx, gy = self._gripper_cell()
        tx, ty = cell
        return abs(gx - tx) + abs(gy - ty)

    def _valid_actions_with_reasons(self) -> dict[str, str]:
        state = self.sim.get_state()
        reasons = {"SCAN_SCENE": "refresh scene understanding"}
        if self._nav_enabled():
            reasons.update({
                "MOVE_NORTH": "move gripper one cell north",
                "MOVE_SOUTH": "move gripper one cell south",
                "MOVE_EAST": "move gripper one cell east",
                "MOVE_WEST": "move gripper one cell west",
                "ROTATE_LEFT": "rotate gripper orientation left",
                "ROTATE_RIGHT": "rotate gripper orientation right",
            })
        else:
            for obj in state.objects.values():
                if obj.reachable and not obj.is_held and obj.in_bin is None:
                    color = obj.name.replace("_block", "").upper()
                    reasons[f"MOVE_TO_{color}"] = f"navigate directly to {obj.name}"

        if state.holding:
            reasons["PLACE_BIN_A"] = "place held object in bin A"
            reasons["PLACE_BIN_B"] = "place held object in bin B"
        else:
            for obj in state.objects.values():
                if not self._can_pick_object(obj.name):
                    continue
                reasons["PICK"] = f"pick reachable object ({obj.name})"
                break

        for obj in state.objects.values():
            if not (obj.blocking and obj.reachable):
                continue
            if self._nav_enabled() and not self._is_adjacent_to(obj.name):
                continue
            reasons["CLEAR_BLOCKER"] = f"clear blocker ({obj.name})"
            break
        return reasons

    def _deadline_status(self) -> dict[str, int]:
        status = {}
        deadlines = getattr(self._scenario_cfg, "deadlines", {}) or {}
        for obj_name, deadline_step in deadlines.items():
            obj = self.sim.get_state().objects.get(obj_name)
            target_bin = self._required_placements.get(obj_name)
            done = bool(obj and target_bin and obj.in_bin == target_bin)
            if done:
                continue
            status[obj_name] = int(deadline_step - self._steps)
        return status

    def _observability_map(self) -> list[str]:
        gx, gy = self._gripper_cell()
        lines = []
        for y in range(3, -4, -1):
            row = []
            for x in range(-3, 4):
                if (x, y) == (gx, gy):
                    row.append("G")
                else:
                    row.append(".")
            lines.append("".join(row))
        return lines

    def _nav_step_toward(self, target: tuple[int, int]) -> str:
        """Navigate one step toward target cell (navigates all the way onto the cell)."""
        gx, gy = self._gripper_cell()
        tx, ty = target
        dx, dy = tx - gx, ty - gy
        # Already at target cell — nothing to do
        if dx == 0 and dy == 0:
            return "SCAN_SCENE"
        # Move along the longer axis first
        if abs(dx) >= abs(dy):
            return "MOVE_EAST" if dx > 0 else "MOVE_WEST"
        return "MOVE_NORTH" if dy > 0 else "MOVE_SOUTH"

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

        # Inject mid-task instruction changes — can fire at multiple steps
        change_steps = getattr(self.cfg.task, 'mid_task_change_steps', [self.cfg.task.mid_task_change_step])
        if (self.cfg.task.mid_task_change_prob > 0
                and self._steps in change_steps
                and self._steps not in self._changes_applied
                and random.random() < self.cfg.task.mid_task_change_prob
                and not self._done):
            self._apply_mid_task_change()
            self._changes_applied.add(self._steps)

        pre_holding = self.sim.get_state().holding
        # Snapshot reachability BEFORE execution so reasoning bonus can check the
        # pre-action state (e.g. "blue is blocking red" is true before CLEAR_BLOCKER fires).
        pre_state_snapshot = {
            name: {"reachable": obj.reachable, "blocking": obj.blocking}
            for name, obj in self.sim.get_state().objects.items()
        }
        valid_now = self._valid_actions()
        invalid_reason = None
        if action not in valid_now:
            raw_result = "FAILED_INVALID"
            reasons = self._valid_actions_with_reasons()
            if reasons:
                invalid_reason = "invalid_now; choose one of: " + ", ".join(sorted(reasons.keys()))
        else:
            raw_result = self.sim.execute(action)
        result = self._apply_noise(action, raw_result)

        if result == "FAILED_SLIP" and raw_result == "SUCCESS" and action == "PICK":
            state = self.sim.get_state()
            if state.holding:
                state.objects[state.holding].is_held = False
                state.holding = None

        # SCAN reveals hidden traits of all currently reachable objects
        if action == "SCAN_SCENE" and result == "SUCCESS":
            self._scanned = True
            hidden = getattr(self._scenario_cfg, 'hidden_traits', {}) or {}
            state = self.sim.get_state()
            for obj_name, trait in hidden.items():
                obj = state.objects.get(obj_name)
                if obj and (obj.reachable or obj.in_bin is not None or obj.is_held):
                    self._revealed_traits[obj_name] = trait

        # FAILED_FRAGILE: picking an unscanned fragile object damages it
        if (result == "SUCCESS" and action == "PICK"
                and getattr(self.cfg.task, 'require_scan_for_traits', False)):
            state = self.sim.get_state()
            picked = state.holding
            hidden = getattr(self._scenario_cfg, 'hidden_traits', {}) or {}
            if picked and hidden.get(picked) == "fragile" and picked not in self._revealed_traits:
                # Object is fragile but agent never scanned — it breaks
                state.objects[picked].is_held = False
                state.holding = None
                result = "FAILED_FRAGILE"

        self._apply_world_drift()
        self._action_history.append(action)
        self._last_action = action
        self._last_result = result

        self._steps += 1
        reward = self._compute_reward(action, result, pre_holding=pre_holding,
                                      pre_state_snapshot=pre_state_snapshot)
        reward += self._reasoning_bonus(reasoning, action, result,
                                        pre_state_snapshot=pre_state_snapshot)
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
            self.logger.end_episode(success=self._all_goals_complete())

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
                "action_preconditions": self._valid_actions_with_reasons(),
                "distance_to_next_goal": self._distance_to_next_goal(),
                "deadline_status": self._deadline_status(),
                "invalid_reason": invalid_reason,
                "goal_progress": self._goal_progress(),
                "mid_task_changed": (self._steps - 1) in self._changes_applied,
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
            scenario_pack=getattr(tc, "scenario_pack", "default"),
        )

        self.sim._build_state_from_config(scenario_cfg)
        self._scenario_cfg = scenario_cfg

        self._steps = 0
        self._done = False
        self._scanned = False
        self._mid_task_changed = False
        self._changes_applied: set[int] = set()   # which change-steps have fired
        self._cumulative_reward = 0.0
        self._action_history = []
        self._last_action = None
        self._last_result = None
        self._completed_subgoals: list[str] = []
        self._known_failures: list[str] = []
        self._active_constraints: list[str] = ([scenario_cfg.constraint]
                                                if scenario_cfg.constraint else [])
        self._instruction = scenario_cfg.instruction
        self._required_placements: dict[str, str] = dict(scenario_cfg.targets)
        # Per-object trait reveal: populated by SCAN_SCENE, enforced in PICK
        self._revealed_traits: dict[str, str] = {}

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

    def _reasoning_bonus(self, reasoning: str, action: str, result: str,
                         pre_state_snapshot: Optional[dict] = None) -> float:
        """
        Bonus for reasoning that mentions relevant objects, constraints, and plans.

        Uses pre-action state snapshot so CLEAR_BLOCKER reasoning ("X is blocking Y")
        is rewarded correctly even though the blocker is already gone post-execution.

        The cap scales with reasoning length — longer, more detailed chain-of-thought
        can earn proportionally more reward (up to a hard ceiling of 1.5).
        """
        if not reasoning or len(reasoning) < 10:
            return 0.0
        bonus = 0.0
        r = reasoning.lower()

        # Use pre-action state for blocked-object checks so CLEAR_BLOCKER reasoning
        # ("blue is blocking red") is rewarded even though the blocker is now cleared.
        blocked_before = set()
        if pre_state_snapshot:
            for name, snap in pre_state_snapshot.items():
                if not snap["reachable"]:
                    blocked_before.add(name.replace("_block", "").lower())
        else:
            for obj in self.sim.get_state().objects.values():
                if not obj.reachable:
                    blocked_before.add(obj.name.replace("_block", "").lower())

        # Mentions blocked objects correctly
        for color in blocked_before:
            if color in r:
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

        # Bonus for explicit multi-step plan in reasoning ("plan:" or "→" sequence)
        if "plan:" in r or (" → " in reasoning):
            bonus += 0.15

        # Token-length scaling: longer reasoning unlocks a higher reward cap.
        # Every 50 chars of reasoning raises the cap by 0.1, up to max 1.5.
        # This rewards richer chain-of-thought without rewarding padding.
        length_scale = min(1.5, 0.5 + 0.1 * (len(reasoning) // 50))
        return min(bonus, length_scale)

    def _compute_reward(self, action: str, result: str, pre_holding: Optional[str] = None,
                        pre_state_snapshot: Optional[dict] = None) -> float:
        w = self.cfg.reward
        r = w.step_cost

        if self._nav_enabled():
            if action in ("MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST"):
                r -= 0.03
            if action in ("ROTATE_LEFT", "ROTATE_RIGHT"):
                r -= 0.02

        if result not in ("SUCCESS", "PARTIAL_CLEAR"):
            failure_key = f"{action}:{result}"
            if result == "FAILED_FRAGILE":
                # Larger specific penalty — agent should have scanned first
                r += w.fragile_pick_penalty
                r += w.repeated_failure if failure_key in self._known_failures else w.first_failure
            else:
                r += w.repeated_failure if failure_key in self._known_failures else w.first_failure
            return r

        if action == "CLEAR_BLOCKER":
            r += w.blocker_cleared
        if action == "PICK":
            held = self.sim.get_state().holding
            # Reward only picks that move a required-yet-unfinished target.
            if held and held in self._required_placements:
                target_bin = self._required_placements[held]
                obj = self.sim.get_state().objects.get(held)
                already_done = bool(obj and obj.in_bin == target_bin)
                if not already_done:
                    r += w.successful_pick
                else:
                    r += w.wrong_pick
            else:
                r += w.wrong_pick
        if action in ("PLACE_BIN_A", "PLACE_BIN_B"):
            bin_name = "A" if action == "PLACE_BIN_A" else "B"
            placed_obj = pre_holding
            correct = bool(placed_obj and self._required_placements.get(placed_obj) == bin_name)
            r += w.correct_placement if correct else w.wrong_bin
            if not correct and self._active_constraints:
                r += w.constraint_violation  # extra hit for constraint violation
        if action == "SCAN_SCENE":
            if not self._scanned:
                r += w.useful_scan  # first scan only
            # Penalize avoidable scans — but NOT if scanning is currently needed
            # to reveal a required hidden trait (fragile/heavy) before picking.
            scan_is_needed = False
            if getattr(self.cfg.task, 'require_scan_for_traits', False):
                hidden = getattr(self._scenario_cfg, 'hidden_traits', {}) or {}
                state = self.sim.get_state()
                for obj_name in self._required_placements:
                    obj = state.objects.get(obj_name)
                    if (obj and obj.reachable and obj.in_bin is None
                            and obj_name in hidden
                            and obj_name not in self._revealed_traits):
                        scan_is_needed = True
                        break
            if not scan_is_needed:
                valid_now = self._valid_actions()
                if any(a != "SCAN_SCENE" for a in valid_now):
                    r += w.useless_action
            # Penalize scan loops with increasing severity regardless.
            streak = 0
            for a in reversed(self._action_history):
                if a == "SCAN_SCENE":
                    streak += 1
                else:
                    break
            if streak > 0:
                r -= min(1.5, 0.25 * streak)

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
        elif self._steps >= self.cfg.task.max_steps:
            # Timeout: explicit penalty so the model learns completing > timing out.
            r += w.timeout_failure

        # Deadline pressure: penalize each overdue unfinished target.
        for obj_name, remaining in self._deadline_status().items():
            if remaining < 0:
                r += (w.missed_deadline * 0.2)

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
        state = self.sim.get_state()
        targets = [
            (obj_name, bin_name)
            for obj_name, bin_name in self._required_placements.items()
            if not (
                state.objects.get(obj_name)
                and state.objects[obj_name].in_bin == bin_name
            )
        ] or list(self._required_placements.items())
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
                if self._can_pick_object(obj.name):
                    has_pick = True
                    break
            if has_pick:
                valid.append("PICK")

        if not state.holding:  # can't clear a blocker while holding something
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
        last_action = self._last_action
        last_result = self._last_result

        def can_clear_now() -> bool:
            for obj in state.objects.values():
                if not (obj.blocking and obj.reachable):
                    continue
                if self._nav_enabled() and not self._is_adjacent_to(obj.name):
                    continue
                return True
            return False

        def blocker_for_target(target_name: str) -> Optional[str]:
            for obj in state.objects.values():
                if obj.blocking == target_name and obj.reachable and obj.in_bin is None:
                    return obj.name
            return None

        # If scan is required and next pick target is fragile+unscanned → scan first
        if getattr(self.cfg.task, 'require_scan_for_traits', False):
            hidden = getattr(self._scenario_cfg, 'hidden_traits', {}) or {}
            for obj_name in self._required_placements:
                obj = state.objects.get(obj_name)
                if (obj and obj.reachable and obj.in_bin is None
                        and hidden.get(obj_name) == "fragile"
                        and obj_name not in self._revealed_traits):
                    return "SCAN_SCENE"

        # Just moved to something → pick it
        if last_action and last_action.startswith("MOVE_TO") and last_result == "SUCCESS":
            return "PICK"

        # Holding → place correctly
        if state.holding:
            target_bin = self._required_placements.get(state.holding)
            if target_bin:
                return f"PLACE_BIN_{target_bin}"
            return "PLACE_BIN_A"

        # Failed to reach a target → clear or re-navigate
        if any(f.startswith("MOVE_TO") and "FAILED_BLOCKED" in f for f in failures) and can_clear_now():
            return "CLEAR_BLOCKER"
        # PICK:FAILED_EMPTY means gripper is not adjacent to anything pickable.
        # In nav mode, re-navigate to the next target instead of looping on CLEAR_BLOCKER.
        if "PICK:FAILED_EMPTY" in failures:
            if self._nav_enabled():
                # Fall through to the placement-order loop below which will nav correctly.
                pass
            elif can_clear_now():
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
                    obj_cell = self._object_cell(obj_name)
                    gripper_cell = self._gripper_cell()
                    # Navigate all the way to the object's cell so PICK grabs
                    # the right object (not a closer distractor).
                    if obj_cell is not None and gripper_cell == obj_cell:
                        return "PICK"
                    if obj_cell is not None:
                        return self._nav_step_toward(obj_cell)
                color = obj_name.replace("_block", "").upper()
                return f"MOVE_TO_{color}"
            blocker = blocker_for_target(obj_name)
            if blocker is not None:
                if self._nav_enabled():
                    if self._is_adjacent_to(blocker):
                        return "CLEAR_BLOCKER"
                    blocker_cell = self._object_cell(blocker)
                    if blocker_cell is not None:
                        return self._nav_step_toward(blocker_cell)
                return "CLEAR_BLOCKER"
            if can_clear_now():
                return "CLEAR_BLOCKER"
            return "SCAN_SCENE"

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
                in_bin=obj.in_bin,
                is_held=obj.is_held,
            ))

        # Recent action history
        history = (self._action_history[-oc.include_action_history:]
                   if oc.include_action_history > 0 else [])

        extra = {}
        if oc.include_valid_actions:
            extra["valid_actions"] = self._valid_actions()
            extra["action_preconditions"] = self._valid_actions_with_reasons()
        if oc.include_goal_progress:
            extra["goal_progress"] = round(self._goal_progress(), 2)
        if oc.include_oracle_hint:
            extra["oracle_hint"] = self._oracle_action()
        if oc.include_distance_to_goal:
            remaining = sum(1 for n, b in self._required_placements.items()
                            if not (state.objects.get(n) and state.objects[n].in_bin == b))
            extra["goals_remaining"] = remaining
            extra["distance_to_next_goal"] = self._distance_to_next_goal()
        goal_cell = self._next_goal_cell()
        if goal_cell is not None:
            extra["next_target_cell"] = f"{goal_cell[0]},{goal_cell[1]}"
        extra["deadline_status"] = self._deadline_status()
        extra["object_deadlines"] = getattr(self._scenario_cfg, "deadlines", {}) or {}
        extra["observability_map"] = self._observability_map()
        # Show what traits have been revealed so far (empty until agent scans)
        extra["discovered_traits"] = dict(self._revealed_traits)

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
