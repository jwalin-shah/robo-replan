"""
RoboReplan server — OpenEnv HTTP protocol + metrics endpoint.
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from openenv.core.env_server import create_fastapi_app

from .openenv_env import RoboReplanEnv, RoboAction, RoboObservation, RoboState

difficulty = os.environ.get("DIFFICULTY", "easy")

# Shared env instance (metrics persist across requests)
_env_instance = RoboReplanEnv(difficulty=difficulty)

app = create_fastapi_app(
    env=lambda: _env_instance,
    action_cls=RoboAction,
    observation_cls=RoboObservation,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_VIZ_HTML = (Path(__file__).parent.parent / "viz_standalone.html").read_text()


@app.get("/")
def root():
    return RedirectResponse(url="/viz")


@app.get("/viz", response_class=HTMLResponse)
def viz():
    return _VIZ_HTML


@app.get("/metrics")
def metrics():
    """Live training metrics: success rate, reward curve, failure breakdown, oracle agreement."""
    return _env_instance.metrics


# ── Demo endpoints — judges can interact live ──────────────────────────

_demo_env = None

@app.post("/demo/reset")
def demo_reset(difficulty: str = "easy"):
    """Start a fresh demo episode."""
    global _demo_env
    _demo_env = RoboReplanEnv(difficulty=difficulty)
    obs = _demo_env.reset()
    return {"observation": obs.model_dump(), "done": False, "reward": 0.0}


@app.post("/demo/step")
def demo_step(action: str):
    """Take one step in the demo episode."""
    global _demo_env
    if _demo_env is None:
        _demo_env = RoboReplanEnv(difficulty="easy")
        _demo_env.reset()
    result = _demo_env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/demo/oracle")
def demo_oracle():
    """Step using the oracle policy — shows optimal behavior for demo."""
    global _demo_env
    if _demo_env is None:
        _demo_env = RoboReplanEnv(difficulty="easy")
        _demo_env.reset()
    oracle = _demo_env._env._oracle_action() or "SCAN_SCENE"
    result = _demo_env.step(oracle)
    return {
        "action_taken": oracle,
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }
