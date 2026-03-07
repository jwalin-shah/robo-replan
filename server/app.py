"""
RoboReplan server — OpenEnv HTTP protocol + metrics endpoint.
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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


@app.get("/viz", response_class=HTMLResponse)
def viz():
    return _VIZ_HTML


@app.get("/metrics")
def metrics():
    """Live training metrics: success rate, reward curve, failure breakdown, oracle agreement."""
    return _env_instance.metrics
