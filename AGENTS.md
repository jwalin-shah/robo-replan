# RoboReplan Agent Notes

Scope changes tightly. This repo is a small OpenEnv tabletop replanning
benchmark with runtime code under `server/`, training code under `train/`, and
operational helpers under `scripts/`.

## Generated Logs

- Treat `logs/*.json`, `logs/*.jsonl`, and generated plots as evidence
  artifacts, not source code.
- Do not edit or refresh committed logs unless the task explicitly asks for an
  evidence update.
- Before implementing behavior that relies on generated logs, reconcile which
  log artifact is authoritative and record the expected producer command in the
  PR or issue.
- Prefer `scripts/check.sh` for local validation; it preserves
  `logs/episodes.jsonl` after checks that exercise the environment.

## Validation

Run the smallest command that proves the change. For docs/scripts-only changes,
use:

```bash
bash scripts/check.sh
```
