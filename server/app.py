"""
RoboReplan server — exposes the environment via OpenEnv's HTTP protocol.

OpenEnv's create_fastapi_app wraps our Environment subclass and
auto-generates all the /reset, /step, /schema, /health endpoints
in the format TRL's GRPO trainer expects.

Also serves the standalone viz at GET /viz.
"""
import os
from pathlib import Path

from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_fastapi_app

from .openenv_env import RoboReplanEnv, RoboAction, RoboObservation, RoboState

difficulty = os.environ.get("DIFFICULTY", "easy")

app = create_fastapi_app(
    env=lambda: RoboReplanEnv(difficulty=difficulty),
    action_cls=RoboAction,
    observation_cls=RoboObservation,
)

_VIZ_HTML = (Path(__file__).parent.parent / "viz_standalone.html").read_text()


@app.get("/viz", response_class=HTMLResponse)
def viz():
    return _VIZ_HTML
