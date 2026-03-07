"""
Domain randomization for the tabletop planning environment.

Randomizes everything that can vary in a real tabletop scene:
  - number of objects
  - which object is the target
  - which bin is the target
  - how many blockers, and what they block
  - object positions (within reachable workspace)
  - task instruction (generated from the sampled scene)
  - distractor objects (present but irrelevant to the task)
  - constraint type (fragile first, heavy last, etc.)

The model must generalize across all of this — not memorize one layout.
"""
import random
import string
from dataclasses import dataclass, field
from typing import Optional


OBJECT_NAMES   = ["red_block", "blue_block", "green_block", "yellow_block", "purple_block"]
OBJECT_COLORS  = {"red_block": "red", "blue_block": "blue", "green_block": "green",
                  "yellow_block": "yellow", "purple_block": "purple"}
BINS           = ["A", "B"]
CONSTRAINTS    = ["fragile_first", "heavy_last", "urgent_first", None, None, None]  # None = no constraint


@dataclass
class ScenarioConfig:
    # Objects actually present in the scene
    objects: list[str] = field(default_factory=list)

    # Which objects are targets (must be placed in a bin)
    targets: dict[str, str] = field(default_factory=dict)  # obj_name -> bin

    # Blocking relationships: blocker -> blocked
    blockers: dict[str, str] = field(default_factory=dict)

    # Distractors: present but not part of the task
    distractors: list[str] = field(default_factory=list)

    # Active constraint
    constraint: Optional[str] = None

    # Generated instruction string
    instruction: str = ""

    # Object positions on the table (x, y) — workspace is roughly ±0.25
    positions: dict[str, tuple] = field(default_factory=dict)


def randomize_scenario(
    n_objects: Optional[int] = None,
    n_targets: Optional[int] = None,
    n_blockers: Optional[int] = None,
    force_blocked: bool = False,
) -> ScenarioConfig:
    """
    Generate a fully randomized scenario.

    n_objects:  total objects on table (default: random 2-5)
    n_targets:  how many must be placed in bins (default: random 1-2)
    n_blockers: how many blocking relationships (default: random 0-2)
    force_blocked: always have at least one blocker (good for training recovery)
    """
    # Sample object count
    total = n_objects or random.randint(2, 5)
    total = min(total, len(OBJECT_NAMES))

    # Pick which objects appear
    present = random.sample(OBJECT_NAMES, total)

    # Pick targets (subset of present objects)
    max_targets = min(n_targets or random.randint(1, 2), len(present))
    targets_list = random.sample(present, max_targets)
    target_bins = {obj: random.choice(BINS) for obj in targets_list}

    # Distractors = present but not targets
    distractors = [o for o in present if o not in target_bins]

    # Build blocking relationships
    n_block = n_blockers if n_blockers is not None else random.randint(0, min(2, len(distractors)))
    if force_blocked:
        n_block = max(1, n_block)

    blockers = {}
    # A blocker must be a non-target (distractor) blocking a target
    available_blockers = list(distractors)
    available_targets  = list(targets_list)
    random.shuffle(available_blockers)
    random.shuffle(available_targets)
    for i in range(min(n_block, len(available_blockers), len(available_targets))):
        blockers[available_blockers[i]] = available_targets[i]

    # Positions: place targets first, then put blockers in front of them
    positions = {}
    x_slots = [-0.15, 0.0, 0.15, -0.08, 0.08]
    random.shuffle(x_slots)
    slot_idx = 0
    for obj in present:
        if obj in blockers.values():
            # target that gets blocked — place it further back
            positions[obj] = (x_slots[slot_idx % len(x_slots)], -0.05)
        else:
            positions[obj] = (x_slots[slot_idx % len(x_slots)], 0.05)
        slot_idx += 1

    # Blocker slightly in front of what it blocks
    for blocker, blocked in blockers.items():
        tx, ty = positions[blocked]
        positions[blocker] = (tx + random.uniform(-0.03, 0.03), ty + 0.08)

    # Constraint
    constraint = random.choice(CONSTRAINTS)

    # Generate instruction
    instruction = _build_instruction(target_bins, constraint)

    return ScenarioConfig(
        objects=present,
        targets=target_bins,
        blockers=blockers,
        distractors=distractors,
        constraint=constraint,
        instruction=instruction,
        positions=positions,
    )


def _build_instruction(target_bins: dict[str, str], constraint: Optional[str]) -> str:
    parts = []
    for obj, bin_ in target_bins.items():
        color = OBJECT_COLORS.get(obj, obj.replace("_block", ""))
        parts.append(f"the {color} block in bin {bin_}")

    if len(parts) == 1:
        base = f"Place {parts[0]}."
    else:
        base = "Place " + ", then ".join(parts) + "."

    if constraint == "fragile_first":
        base += " Handle fragile items first."
    elif constraint == "heavy_last":
        base += " Move heavy items last."
    elif constraint == "urgent_first":
        base += " Prioritize urgent items first."

    return base
