# Run training on the H100 (Northflank)

The **API service** (robo-replan) runs the env server; it does **not** run training. To use the **H100** for training, use a **Northflank Job** that builds from **Dockerfile.train** and runs on GPU.

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

## Optional: shorter run for testing

To test the job quickly, you can override the command when running the job (in Northflank UI: Run → Command override) to something like:

```bash
bash -c "ORACLE_EPISODES=50 FAST_MODE=1 python3 train/run_training.py"
```
