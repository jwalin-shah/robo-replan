# Deploy to Northflank (Git repo + Dockerfile)

Deploy this repo to Northflank so it builds from the root **Dockerfile** and runs the OpenEnv API (or use the same VM for training).

**Token:** Stored in `.env` as `NORTHFLANK_TOKEN` (not committed). Use it for the Northflank API and CLI.

**How you “connect” (no raw SSH):** Northflank doesn’t give you a normal SSH host. You get a shell into the running container in one of two ways:

1. **In the UI:** Open your service → **Containers** → click **Shell access** on a container. A browser shell opens inside the container (SSH-like).
2. **CLI (SSH or Exec):** Install the Northflank CLI, log in with the token, then SSH or exec into the service:
   ```bash
   npm i -g @northflank/cli
   northflank login -n hackathon -t "$(grep NORTHFLANK_TOKEN .env | cut -d= -f2)"
   # SSH into the running container:
   northflank ssh service --projectId hackathon --serviceId robo-replan
   # Or run a one-off command:
   northflank exec service --projectId hackathon --serviceId robo-replan --cmd "ls -la"
   ```
   Project ID: **hackathon**, Service ID: **robo-replan**.

   **Note:** SSH/exec only work when the service has a **running deployment**. If you see "No running instance found", wait for the build to finish and a deployment to be running, then try again.

So “connected” here means: service is running and you use Northflank’s UI or CLI to open a shell. The `hackathon-vm-generic-team` host in your SSH config (34.63.31.254) is a different VM (e.g. GCP); Northflank access goes through their API, not that IP.

**Rebuild on push:** In Northflank → your service → Build options → enable **CI**. Then every push to the linked branch (e.g. `main`) triggers a new build and deploy.

**Deploy from here (trigger build now):** From the repo root, run:
```bash
bash scripts/deploy_northflank.sh
```
This pushes `main` to GitHub (if needed) and triggers a Northflank build for the latest commit. Requires `NORTHFLANK_TOKEN` in `.env`. To only trigger a build without pushing: `SKIP_PUSH=1 bash scripts/deploy_northflank.sh`.

## One-time setup

1. **Link GitHub to Northflank**  
   In Northflank: [Link your Git account](https://northflank.com/docs/v1/application/getting-started/link-your-git-account) (GitHub). Authorize the org/user that owns the repo.

2. **Create a Combined Service**  
   - [Create a new service](https://app.northflank.com/s/project/create/service) → choose **Combined service** (build + run from one place).  
   - **Source:** Git repository → select your repo (e.g. `jwalin-shah/robo-replan`).  
   - **Branch:** `main` (or the branch you want to deploy).  
   - **Build type:** Dockerfile.  
   - **Dockerfile path:** `/Dockerfile` (repo root).  
   - **Build context:** `/` (root).  
   - **Run:** Deploy as a service (default). The Dockerfile `CMD` runs uvicorn on port 7860; Northflank will map it.

3. **Deploy**  
   Save the service. Northflank will build the image from the Dockerfile and deploy. Enable **CI** in build options if you want it to rebuild on every push to `main`.

## What gets deployed

- The **Dockerfile** at repo root builds the image (Python, FastAPI, OpenEnv, server).  
- **Build context** is the repo root, so `COPY server/`, `COPY viz_standalone.html`, etc. use the files from the linked branch.  
- Pushing to `main` (with CI enabled) triggers a new build and deploy.

## Optional: run training on Northflank

The same Git repo contains `train/run_h100_1.5b.sh` and `train/run_training.py`. To run training on Northflank:

- Use a **Job** that runs a custom command (e.g. clone repo + `bash train/run_h100_1.5b.sh`) with a GPU-enabled image, or  
- Use a **deployed service** that only runs the API; run training elsewhere (e.g. Colab or a GPU job) and point it at the deployed API if needed.

The root Dockerfile is for the **API server** (no GPU). For a training job you’d typically use a different image (e.g. CUDA base) or run the training script inside a job that has the repo and deps.

## Summary

| Step | Action |
|------|--------|
| 1 | Link GitHub in Northflank |
| 2 | New Combined service → Git repo + branch `main` |
| 3 | Build: Dockerfile, path `/Dockerfile`, context `/` |
| 4 | Deploy; enable CI to rebuild on push |

You’re just deploying the Git repo; Northflank builds from the Dockerfile and runs the container.
