# 60-Second YouTube Demo Script

Record your screen (QuickTime: Cmd+Shift+5) while narrating. Target runtime: 58–60 seconds.

---

## 0:00 – 0:10 · Hook

**[Screen: `openenv-community-robo-replan.hf.space/viz` already loaded]**

> "LLMs can follow instructions. But what happens when something goes wrong mid-task?
> This is RoboReplan — an environment built to teach models to recover and replan."

---

## 0:10 – 0:28 · Failure Mode

**[Scene: 💊 Pharmacy, difficulty: Medium, click Reset]**

> "Pharmacy task: sort medication into the right bins, fragile items first.
> Watch a random agent."

**[Click ▶ Run Agent — let it run 5–6 steps, showing scan loops or wrong placements]**

> "It loops on SCAN_SCENE, ignores the blocker, never makes progress. Zero percent success."

**[Click ■ Stop]**

---

## 0:28 – 0:45 · Trained Agent

**[Click Reset, then 🎯 Run Oracle]**

> "Now the trained policy — watch the reasoning box.
> It identifies the blocker, clears it, picks the fragile item first to satisfy the constraint,
> places it correctly."

**[Let oracle run to completion or mid-task banner fires — point it out if it does]**

> "If the instruction changes mid-task, it replans immediately."

---

## 0:45 – 0:57 · Training Results

**[Switch to `training_results.png`]**

> "We trained Qwen 2.5 — SFT warm-start then GRPO — on a free Colab T4.
> Zero percent success before training. Seventy-eight percent after.
> Reward goes from minus thirty to plus eight."

---

## 0:57 – 1:00 · Close

**[Back to /viz briefly]**

> "RoboReplan. OpenEnv 0.2.1. Links below."

---

## Recording tips

- Open `openenv-community-robo-replan.hf.space/viz?difficulty=medium&pack=pharmacy` — pre-loads the right state
- Zoom into the reasoning box during oracle run (Cmd+= in Chrome) so the `<think>` text is legible on video
- If the orange mid-task banner fires, slow down and point to it — it's the visual proof of replanning
- Record at 1280×720 minimum, 30fps
- Don't add music — narration needs to be clean and audible for judges
