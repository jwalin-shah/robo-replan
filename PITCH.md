# RoboReplan -- 3-Minute Pitch Script

## Minute 1: The Problem (0:00 - 1:00)

**Opening hook:**

"LLMs can plan, but they can't replan."

When you ask GPT-4 to sort blocks into bins, it generates a perfect sequence.
But what happens when the grasp slips? When another object is blocking the target?
When the instruction changes halfway through?

The model freezes. It repeats the same failing action in a loop. Or it abandons
the plan entirely.

This is the fundamental gap between language understanding and real-world execution.
Current benchmarks test whether a model can follow a plan -- we test whether it can
*recover* when the plan breaks.

---

## Minute 2: RoboReplan -- The Environment (1:00 - 2:00)

**[SHOW THE VIZ -- https://openenv-community-robo-replan.hf.space/viz]**

RoboReplan is a tabletop manipulation environment built on OpenEnv 0.2.1.

The agent gets a natural-language instruction like "Place red in bin A, blue in bin B.
Handle fragile items first."

**Walk through one episode (hit "Run Agent" or use Oracle mode):**

- The agent scans the scene and sees red is blocked by green
- It clears the blocker, then moves to red and picks it
- Grasp slip! The pick fails. The agent sees the failure in its observation
- It reasons in `<think>` tags: "Pick failed, I need to retry"
- It successfully picks and places red in bin A
- Then mid-task, the instruction changes: "Now place blue in bin B first"
- The agent replans and adapts

**Key features that make this environment novel:**
- 16-action space with navigation, manipulation, and sensing
- Three-level curriculum (easy/medium/hard) with auto-advance
- Domain randomization: blockers, grasp slips, perception noise, mid-task changes
- Reasoning-augmented actions: the model thinks in `<think>` tags before acting
- 4 scenario packs: default, warehouse, pharmacy, lab

---

## Minute 3: Training Results -- The Proof (2:00 - 3:00)

**[SHOW TRAINING RESULTS CHART]**

We trained Qwen2.5-0.5B using two phases:
1. SFT warm-start on oracle rollouts (10 min)
2. GRPO with real environment reward (30 min)

**Before training (random policy):**
- ~20% success rate on easy tasks
- Negative average reward
- The model scans repeatedly, ignores blockers, times out

**After SFT + GRPO:**
- ~95-100% success rate on easy
- Positive reward
- The model clears blockers, recovers from failures, follows constraints

The key insight: our reward function penalizes *repeated* failures (-2.5) more
than first failures (-1), and gives a recovery bonus (+1) when the agent succeeds
after a failure. This is what teaches the model to replan instead of loop.

All of this runs on a free Colab T4 in under an hour.

**Built on OpenEnv 0.2.1. Deployed on HF Spaces. Trained with GRPO via TRL.**

---

## Q&A Preparation

**"How is this different from existing planning benchmarks?"**
> We don't just test plan execution -- we inject failures (grasp slips, blockers,
> mid-task changes) and test whether the agent can recover. The key metric is
> replanning ability, not planning ability.

**"Why stub mode instead of real physics?"**
> Stub mode runs 1000x faster, which is critical for RL training. The action space,
> failure modes, and observation structure are identical. Physics adds latency
> without changing what the model learns.

**"What model did you train?"**
> Qwen2.5-0.5B-Instruct in Colab on a free T4 -- showing that GRPO can teach
> replanning even to very small models. The Unsloth script also supports
> Qwen2.5-7B on an H100 for larger-scale training.

**"How does the reward work?"**
> Composite signal: +10 for task completion, +2 for correct placements, +1 for
> recovery after failure, -2.5 for repeated same failure, -4 for constraint
> violations. Training weights amplify completion (+25) to prevent partial-credit
> gaming.

**"What problem statement does this address?"**
> 3.1 -- World Modeling: Professional Tasks. The agent must maintain consistent
> internal state, update beliefs based on outcomes, and orchestrate multi-step
> workflows in a dynamic, partially observable world.

---

## Key Links

- GitHub: https://github.com/jwalin-shah/robo-replan
- HF Space: https://huggingface.co/spaces/openenv-community/robo-replan
- Live Viz: https://openenv-community-robo-replan.hf.space/viz
- API Health: https://openenv-community-robo-replan.hf.space/health
- Colab Notebook: train/colab_train.ipynb
