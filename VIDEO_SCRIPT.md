# 1-Minute YouTube Demo Script

Record your screen (QuickTime or OBS) while narrating:

## 0:00 - 0:15 -- Hook + Problem

"RoboReplan tests whether LLMs can replan after failures in robotic manipulation.
Current models can plan, but when a grasp slips or an instruction changes mid-task,
they freeze or loop. We built an environment that trains this exact skill."

## 0:15 - 0:35 -- Live Demo

**[Screen: open https://openenv-community-robo-replan.hf.space/viz]**

"Here's the environment running live on HF Spaces. The agent gets a natural
language instruction and must place colored blocks in target bins."

**[Click Reset, then click "Run Agent" to run Oracle mode]**

"Watch -- the agent clears a blocker, picks the target, and when things go wrong,
it replans. You can see the reasoning in real-time in the think tags."

## 0:35 - 0:50 -- Training Results

**[Show Colab output or training_results.png]**

"We trained Qwen 2.5 0.5B using SFT plus GRPO in under an hour on a free Colab T4.
Before training, 20% success. After training, over 95% success on easy tasks,
with the model learning to clear blockers and recover from failures."

## 0:50 - 1:00 -- Closing

"RoboReplan -- built on OpenEnv 0.2.1, deployed on HF Spaces, trained with GRPO.
Problem Statement 3.1: World Modeling for Professional Tasks."
