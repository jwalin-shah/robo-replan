# Run training on the H100 (Northflank)

The **API service** (robo-replan) runs the env server; it does **not** run training. To use the **H100** for training, use a **Northflank Job** that builds from **Dockerfile.train** and runs on GPU.

---

## Why are there so many build events?

- **robo-replan (service)** has **CI enabled**: every push to `main` triggers a **service** build (the API image). Those “builds.start” events are often from the service, not the training job.
- **robo-replan-train (job)** was created with **runOnSourceChange: "never"**, so it does **not** auto-build on push. Builds for the job only start when you click **Build** in the job’s Builds tab (or trigger via API).
- So: multiple build events = either the **service** rebuilding on pushes, or **manual** job builds. To reduce noise, you can disable CI on the **service** in Build options if you don’t need the API to rebuild on every push.

---

## Build every time vs just pull / when to build

- **You do not need to build before every train run.**  
  A **run** uses the **last successful build** of the job. So:
  - **Build** = “bake a new image from the repo” (do this when you change training code or env).
  - **Run** = “start a training run using the current image” (no build, starts in a few seconds if the image is ready).

- **Recommended flow:**
  1. **Build once** (or when you change `train/`, `Dockerfile.train`, or deps).
  2. **Run** as many times as you want for training/retraining — each run reuses that image.
  3. When you push new code and want it in training, trigger a **new build** for the job, wait for it to succeed, then **Run** again.

- **“Just pull”** would mean using a pre-built image from a registry instead of building from Git. Northflank can do that, but then you’d have to build and push the image yourself elsewhere. Building from Git on Northflank (when you need a new image) is usually simpler.

So: **build only when code/config changes; run whenever you want to train or retrain.**

---

## Making sure the job “closes” and doesn’t stay running

- **Job runs are one-off.** When you click Run, Northflank starts a container that runs `python3 train/run_training.py`. When the script exits (training done or error), the container exits and the **run** is marked completed/failed. Nothing keeps running; no need to “close” the job.
- The **Building** state is the **build** (compiling the image), not a long-lived service. When the build finishes, it’s done; the next **Run** will use that image.
- So: the run closes itself when training ends; the build closes when the image is built. No extra cleanup needed.

---

## Why do I still have “Running” resources?

**Services vs jobs:**

- **robo-replan** (and **Jupyter PyTorch** if you have it) are **services** — long-lived, always-on deployments. Northflank keeps them **Running** so the API (or Jupyter) is available. They don’t stop by themselves.
- **robo-replan-train** is a **job** — each **Run** starts a container, runs training, then the container exits. The job itself isn’t “running” in the background; only while a run is active.

So the “running” stuff you see is the **services** (API, Jupyter). To stop them when you’re not using them:

- In Northflank: open the **service** (e.g. robo-replan) → **Pause** (or scale to 0 if the UI allows). That stops the containers and stops billing for that service until you resume.
- Resume when you need the API or Jupyter again.

Training doesn’t leave anything running; only the services you’ve deployed do.

## One-time: create the training job

1. In Northflank, open project **hackathon** → **Jobs** → **Create** → **Manual job**.
2. **Name:** e.g. `robo-replan-train`.
3. **Build:** From Git repository  
   - Repository: `jwalin-shah/robo-replan` (same as the API service)  
   - Branch: `main`  
   - **Dockerfile path:** `/Dockerfile.train`  
   - Build context: `/`
4. **Run:**  
   - **Compute plan:** pick a **GPU** plan (e.g. **nf-gpu-hack-16-64** or whatever H100 plan your team has).  
   - Command override: leave default (the image `CMD` runs training).
5. Save. The job will **build** from `Dockerfile.train` when you run it (or when you trigger a build).

## Run training (trigger a job run)

- **In the UI:** Open the job **robo-replan-train** → **Runs** → **Run job** (or “Run” on the latest build).
- **From here (CLI):** After the job exists, get its ID and trigger a run:
  ```bash
  # List jobs to get jobId
  northflank get job list --projectId hackathon

  # Start a run (replace JOB_ID with the job id, e.g. robo-replan-train)
  curl -s -X POST "https://api.northflank.com/v1/projects/hackathon/jobs/robo-replan-train/runs" \
    -H "Authorization: Bearer $(grep NORTHFLANK_TOKEN .env | cut -d= -f2)" \
    -H "Content-Type: application/json" -d '{}'
  ```

Make sure the repo has **Dockerfile.train** and that **train/** and **scripts/** are not in `.dockerignore` so the build can copy them.

## What runs on the H100

- **Dockerfile.train** uses a CUDA base image, installs PyTorch (CUDA 12.1), `trl`, `datasets`, `transformers`, and copies `server/`, `train/`, `scripts/`.
- The container **CMD** runs `python3 train/run_training.py` (with `MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct` by default).
- So when you run the job, Northflank builds this image, schedules it on a GPU node (H100), and training runs there. Logs appear in the job run in Northflank.

## How to track progress and see errors

- **Northflank UI:** Jobs → **robo-replan-train** → **Runs** → click a run → **Logs**. You get live stdout/stderr; when the run ends, status is **Succeeded** or **Failed**.
- **Run status via API:**  
  `GET .../projects/hackathon/jobs/robo-replan-train/runs/{runId}` → `status` (e.g. `RUNNING`, `SUCCEEDED`, `FAILED`), `concluded`, `startedAt`, `concludedAt`.
- **Logs via API:**  
  `GET .../projects/hackathon/jobs/robo-replan-train/logs?runId={runId}` returns log lines (stderr shows import/runtime errors).

If a run **Failed**, open it and read the **Logs** tab (or the API logs); the last stderr lines usually show the exception (e.g. `FSDPModule` import = PyTorch/TRL version mismatch; fixed by pinning PyTorch 2.6+ in Dockerfile.train).

## Optional: shorter run for testing

To test the job quickly, you can override the command when running the job (in Northflank UI: Run → Command override) to something like:

```bash
bash -c "ORACLE_EPISODES=50 FAST_MODE=1 python3 train/run_training.py"
```
