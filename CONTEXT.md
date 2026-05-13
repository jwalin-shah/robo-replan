# RoboReplan Context

RoboReplan tests whether an agent can replan in a partially observable tabletop
task: clear blockers, recover from failures, respect ordering constraints, and
adapt to mid-task instruction changes.

The runtime logger appends episode evidence through `LogConfig.export_path`,
which defaults to `logs/episodes.jsonl`. Other files in `logs/` are generated
reports or benchmark snapshots. These artifacts are useful for claims and demos,
but they should be reconciled before implementation work treats them as ground
truth.

Current log policy:

- `server/config.py` is the source of truth for logging knobs.
- `server/logger.py` owns the JSONL episode schema.
- `scripts/*report*.py`, `scripts/*benchmark*.py`, and eval scripts produce
  generated evidence artifacts.
- Operations and validation changes should avoid committing incidental log
  rewrites. If a log needs to change, state the command, seed or episode count,
  and expected evidence claim.
