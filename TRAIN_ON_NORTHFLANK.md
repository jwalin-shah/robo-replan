# Run training on the H100 (Northflank)

The **API service** (robo-replan) runs the env server; it does **not** run training. To use the **H100** for training, use a **Northflank Job** that builds from **Dockerfile.train** and runs on GPU.

**We do not use Unsloth** for this job — the image runs `train/run_training.py` with standard TRL (no Unsloth). That avoids the Unsloth GRPO crashes.

---

## Start from scratch (Northflank)

If you want a clean slate or the job keeps failing:

1. **Create the job (or use existing)**  
   Northflank → **Jobs** → **Create** → **Manual job** (or delete the old one and create new).  
   - **Name:** `robo-replan-train`  
   - **Build:** Git repo `jwalin-shah/robo-replan`, branch `main`, **Dockerfile path** `Dockerfile.train` (path: `/Dockerfile.train`), context `/`  
   - **Run:** **Compute plan** = a **GPU** plan (e.g. nf-gpu-hack-16-64). Leave command default.

2. **Build once**  
   In the job → **Builds** → **Start build** (from latest commit). Wait until the build is **green** (SUCCESS). The image includes torch, trl 0.14, vllm, and the training code — no Unsloth.

3. **Run**  
   **Runs** → **Run** (or from repo: `bash scripts/run_northflank_training.sh`).  
   Training runs in the container; watch **Logs** for that run. When it finishes (or fails), the run ends — no SSH.

If the **run** fails, open that run’s **Logs** and scroll to the bottom; the last error line tells you what broke. See **When it keeps failing** below.

---

## Nuke everything and set it up again

You can wipe the **training job**, the **Jupyter PyTorch** service, and (if you want) the **robo-replan** API service, then recreate only what you need.

### 1. Delete the training job

- Northflank → project **hackathon** → **Jobs**.
- Open the job **robo-replan-train** (or whatever you named it).
- Go to **Settings** (or the job’s ⋮ menu) and choose **Delete job** (or **Remove** / **Delete**).
- Confirm. That removes the job and its runs/builds; nothing of that job is left.

### 2. Delete the Jupyter PyTorch service (and optionally robo-replan)

- Northflank → **Services**.
- Open **Jupyter PyTorch** (or whatever your Jupyter service is called) → **Settings** → **Delete service** (or ⋮ menu → Delete). Confirm. That removes the service and its deployments.
- If you also want to remove the API: open **robo-replan** → **Settings** → **Delete service** → confirm. (You can recreate it later from Git; see below.)

### 3. Recreate only what you need

**Training job (H100)**

- **Jobs** → **Create** → **Manual job**.
- **Name:** `robo-replan-train`.
- **Build:** Git repo `jwalin-shah/robo-replan`, branch `main`, **Dockerfile path** `Dockerfile.train`, **Build context** `/`.
- **Run:** **Compute plan** = GPU (e.g. nf-gpu-hack-16-64). Leave command default.
- Save → **Builds** → **Start build** → wait for SUCCESS → **Runs** → **Run**.

**robo-replan API service**

- **Services** → **Create** → **Combined service**.
- **Source:** Git repo `jwalin-shah/robo-replan`, branch `main`.
- **Build:** Dockerfile path `/Dockerfile`, context `/`.
- **Run:** Deploy as service (default; CMD runs the API on port 7860).
- Save. Enable **CI** in build options if you want it to rebuild on push. See `DEPLOY_NORTHFLANK.md` for details.

**Jupyter PyTorch (notebook)**

- **Services** → **Create** → choose **From template** or **Deploy image** (depending on your Northflank project).
- If your project has a **Jupyter / PyTorch** template, use that. Otherwise: create a **Deploy** service with image `quay.io/jupyter/pytorch-notebook:cuda12-2025-08-18`, set port (e.g. 8888), pick a GPU compute plan if you want to run training there, and save. You get a URL (or use port-forward) to open Jupyter in the browser.

**Jupyter service (project robo-replan):** If your Jupyter PyTorch service is in project **robo-replan** with service ID **jupyter-pytorch**:
- **Public URL:** Check **Networking** for the service URL (e.g. `app--jupyter-pytorch--*.code.run`).
- **Port-forward (local Jupyter):** `northflank forward service --projectId robo-replan --serviceId jupyter-pytorch` then open the URL it prints (e.g. localhost:8888 with token).
- **SSH into container:** `northflank ssh service --projectId robo-replan --serviceId jupyter-pytorch`
- **One-off command:** `northflank exec service --projectId robo-replan --serviceId jupyter-pytorch --cmd "ls -la"`
- **Upload a file:** `northflank upload service file --projectId robo-replan --serviceId jupyter-pytorch --localPath train/colab_train.ipynb --remotePath /home/jovyan/colab_train.ipynb`
- **Tail logs:** `northflank get service logs --tail --projectId robo-replan --serviceId jupyter-pytorch`

**Build and run the job**

- In the new job → **Builds** → **Start build** (or “Build” from latest commit).
- Wait until the build status is **SUCCESS** (green). First build can take 10–20 min (pip installs, etc.).

**4. Run**

- **Runs** → **Run**.
- Open the new run → **Logs** to watch training. No SSH; the container runs until the script exits.

From your machine you can also trigger build then run with:

```bash
bash scripts/build_and_run_training.sh
```

(Requires `NORTHFLANK_TOKEN` in `.env`; the script pushes to `main`, starts a build, waits for it to succeed, then starts a run.)

---

## Run overnight / get good results

To let training run all night and get better results, run **headless** (so closing the browser doesn’t matter) and use **quality** settings (more data, SFT warmstart, no fast mode).

### Option A: Northflank Job (recommended — fully headless)

1. **Job** → **robo-replan-train** → **Environment** (or **Run** → environment).
2. Add (or override) these **environment variables** for the run:
   - `ORACLE_EPISODES=1200` — more oracle data (default 400 is for quick runs).
   - `FAST_MODE=0` — use quality defaults (more GRPO generations, longer rollouts).
   - `ENABLE_SFT_WARMSTART=1` — run SFT then GRPO (better than GRPO only).
   - `MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct` — or keep 0.5B for faster steps.
3. **Runs** → **Run**. The container runs `python3 train/run_training.py` and reads these env vars. You can close the browser; the run continues. Check **Logs** in the morning.
4. **Checkpoints:** Training writes to `./outputs/` in the container. To keep them you need a **Volume** mounted at `/app/outputs` (or copy artifacts out via Northflank if your plan supports it). Otherwise at least **Logs** show progress and final metrics.

### Option B: Jupyter Terminal (headless inside Jupyter)

If you prefer to run from the same repo/image as the notebook but don’t want to keep the notebook tab open:

1. In Northflank Jupyter, open **Terminal** (from Launcher or File → New → Terminal).
2. Run the training script in the background with overnight env vars:
   ```bash
   cd /home/jovyan/robo-replan
   nohup env ORACLE_EPISODES=1200 FAST_MODE=0 ENABLE_SFT_WARMSTART=1 \
     python3 train/run_training.py > logs/overnight_train.log 2>&1 &
   ```
3. Close the browser if you want; the process keeps running in the container. To check progress later: re-open Jupyter, Terminal, then `tail -f /home/jovyan/robo-replan/logs/overnight_train.log`. Outputs and checkpoints go to `./outputs/` in the repo directory.

### Suggested “good results” settings (summary)

| Env var | Quick run | Overnight / quality |
|--------|-----------|----------------------|
| `ORACLE_EPISODES` | 400 | 1200–2000 |
| `FAST_MODE` | 1 | 0 |
| `ENABLE_SFT_WARMSTART` | 0 | 1 |
| `MODEL_NAME` | 0.5B or 1.5B | 1.5B for better quality |

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

- **From here (CLI):** Scripts abort any active runs first, then start a new one: `run_northflank_training.sh`, `build_and_run_training.sh`; or only abort: `northflank_abort_running_runs.sh`.
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

## When it keeps failing

1. **Get the actual error**  
   Northflank → **Jobs** → **robo-replan-train** → **Runs** → open the latest **Failed** run → **Logs**. Scroll to the bottom; the last stderr lines are usually the traceback (e.g. `ModuleNotFoundError: ...`, `RuntimeError: ...`). Copy that or note the first line that says what’s wrong.

2. **If you changed code or Dockerfile**  
   A new run still uses the **last successful build**. So after editing `Dockerfile.train`, `train/`, or `server/`, you must **trigger a new build**, wait for it to succeed, then **Run** again.  
   - From repo: `bash scripts/build_and_run_training.sh` (push + build + run, waits for build).  
   - Or in UI: **Builds** → **Start build** (or build from latest commit), then **Runs** → **Run** after the build is green.

3. **Common fixes**  
   - **No module named 'vllm'** → Image was built before vllm was added. Trigger a **new build** (step 2), then run again.  
   - **FSDPModule / torch.distributed** → PyTorch/TRL mismatch; Dockerfile.train pins `torch<2.6` and `trl==0.14.0` to avoid this.  
   - **NVIDIA Driver was not detected** → If the run then fails or doesn’t use GPU, confirm the job’s compute plan is a **GPU** plan (e.g. nf-gpu-hack-16-64) and that the run was scheduled on a GPU node.  
   - **Build failed** (e.g. pip install vllm fails) → Check the **Build** logs for that build; we may need to adjust the vllm version or install order in Dockerfile.train.

4. **Quick retry**  
   If nothing in code or Dockerfile changed and the failure looks transient, just start another run: `bash scripts/run_northflank_training.sh` or **Runs** → **Run**.

## Making it all work faster

**Build faster**
- Northflank caches image layers; the Dockerfile is ordered so pip install is cached and only code (server/train/scripts) changes trigger a short rebuild.
- Rebuild only when you change code or deps; otherwise just click **Run** (reuses last image).

**Training run faster**
- The image defaults to **FAST_MODE=1**, **ORACLE_EPISODES=400**, and smaller baseline/final evals so a single run finishes in tens of minutes instead of hours.
- For an even quicker test, override the job command to e.g. `ORACLE_EPISODES=100 python3 train/run_training.py` (in Northflank: Run → Command override).
- Use **0.5B** model for faster steps: set env `MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct` on the job.

**One command from your machine**
- From repo root, run:
  ```bash
  bash scripts/build_and_run_training.sh
  ```
  This pushes to `main`, triggers a build, waits for it to succeed (up to 15 min), then triggers a run. So you don’t have to open Northflank or click Build then Run.
- To only trigger build and run without waiting: `bash scripts/build_and_run_training.sh --no-wait` (run may fail with “no successful build” if the build isn’t done yet).

## Optional: shorter run for testing

To test the job quickly, you can override the command when running the job (in Northflank UI: Run → Command override) to something like:

```bash
bash -c "ORACLE_EPISODES=50 FAST_MODE=1 python3 train/run_training.py"
```
