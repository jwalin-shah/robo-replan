# 60-Second YouTube Demo Script

Record your screen (QuickTime: Cmd+Shift+5) while narrating. Target runtime: 58–60 seconds.

---

## 0:00 – 0:10 · Hook

**[Screen: HF Space URL bar loading — `openenv-community-robo-replan.hf.space/viz`]**

> "This is RoboReplan — an environment that trains LLMs to replan on the fly.
> Most models can execute a fixed plan. Almost none can recover when something goes wrong."

---

## 0:10 – 0:30 · Live Demo

**[Select "💊 Pharmacy" from the Scene dropdown, set difficulty to Medium, click Reset]**

> "Here's the pharmacy task: sort medication into the right bins — fragile items first.
> Watch what an untrained model does."

**[Click ▶ Run Agent — let it run 6–8 steps showing scan loops / wrong actions]**

> "It scans repeatedly, ignores the blocker, and times out. Classic failure mode."

**[Click ■ Stop, then Reset. Click 🎯 Run Oracle — runs 1 full episode]**

> "Now the scripted policy: it reads the scene, clears the blocker,
> picks the right item — you can see the reasoning trace live."

---

## 0:30 – 0:50 · Training Results

**[Switch screen to Colab output or `training_results.png`]**

> "We trained Qwen 2.5 with SFT plus GRPO on a free Colab T4.
> Before training: 15% success on Medium.
> After: over 95%. The model learned to clear blockers, recover from grasp slips,
> and replan when the instruction changes mid-task."

---

## 0:50 – 1:00 · Close

**[Return to /viz, show difficulty buttons and pack selector briefly]**

> "RoboReplan — OpenEnv 0.2.1, deployed on HF Spaces, trained with GRPO.
> Problem Statement 3.1: World Modeling for Professional Tasks.
> Links in the description."

---

## Recording tips

- Use `?difficulty=medium&pack=pharmacy` in the URL to auto-load the right state
- Zoom in on the reasoning box when showing the oracle trace (Cmd+= in Chrome)
- Show the mid-task change banner (orange flash) if it fires during the oracle run — it makes the replanning point visually
- Record at 1280×720 minimum; export at 30fps
- Add captions for the before/after numbers (DaVinci Resolve free, or iMovie text overlay)
