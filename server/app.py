"""
RoboReplan server — OpenEnv HTTP protocol + metrics endpoint.
"""
import os
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from openenv.core.env_server import create_fastapi_app

from .openenv_env import RoboReplanEnv, RoboAction, RoboObservation, RoboState
from .models import Action as EnvAction

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
_policy_pipe = None
_POLICY_MODEL = os.environ.get("DEMO_POLICY_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
_VALID_ACTIONS = [a.value for a in EnvAction]


def _extract_action(text: str) -> str:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().upper()
    normalized = re.sub(r"[^A-Z_ ]+", " ", clean)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for action in sorted(_VALID_ACTIONS, key=len, reverse=True):
        if action in clean:
            return action
    spaced_map = {a.replace("_", " "): a for a in _VALID_ACTIONS}
    for spaced, action in spaced_map.items():
        if spaced in normalized:
            return action
    return "SCAN_SCENE"


def _parse_valid_actions_from_prompt(prompt: str) -> list[str]:
    m = re.search(r"Valid now:\s*(.*)", prompt, flags=re.IGNORECASE)
    if not m:
        return []
    raw = m.group(1).strip()
    if raw.lower() == "any":
        return []
    items = [x.strip().upper() for x in raw.split(",") if x.strip()]
    return [a for a in items if a in _VALID_ACTIONS]


def _fallback_action(valid: list[str]) -> str:
    if not valid:
        return "SCAN_SCENE"
    priority = [
        "PLACE_BIN_A", "PLACE_BIN_B",
        "PICK",
        "CLEAR_BLOCKER",
        "MOVE_TO_RED", "MOVE_TO_BLUE", "MOVE_TO_GREEN", "MOVE_TO_YELLOW", "MOVE_TO_PURPLE",
        "MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST",
        "ROTATE_LEFT", "ROTATE_RIGHT",
        "SCAN_SCENE",
    ]
    for p in priority:
        if p in valid:
            return p
    return valid[0]


def _get_policy_pipe():
    global _policy_pipe
    if _policy_pipe is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(_POLICY_MODEL)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(_POLICY_MODEL, dtype="auto", device_map="auto")
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        # Avoid max_new_tokens/max_length warning spam in server logs.
        model.generation_config.max_length = None
        _policy_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    return _policy_pipe


class PolicyActionRequest(BaseModel):
    prompt: str

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
    result = _demo_env.step(RoboAction(action=action))
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
    result = _demo_env.step(RoboAction(action=oracle))
    return {
        "action_taken": oracle,
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.post("/demo/policy_action")
def demo_policy_action(req: PolicyActionRequest):
    """
    Returns one model-predicted action for a given prompt.
    The visualization can use this to drive the environment step-by-step.
    """
    try:
        pipe = _get_policy_pipe()
        out = pipe(
            req.prompt,
            return_full_text=False,
            max_new_tokens=12,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
        )[0]["generated_text"]
        action = _extract_action(out)
        valid = _parse_valid_actions_from_prompt(req.prompt)
        if valid and action not in valid:
            action = _fallback_action(valid)
        # Avoid no-op scan loops when other valid actions exist.
        if valid and action == "SCAN_SCENE" and any(v != "SCAN_SCENE" for v in valid):
            action = _fallback_action([v for v in valid if v != "SCAN_SCENE"])
        return {"action": action, "raw_output": out}
    except Exception as exc:
        # Fail soft so the UI can still run with manual/scripted controls.
        return {"action": "SCAN_SCENE", "error": str(exc)}
