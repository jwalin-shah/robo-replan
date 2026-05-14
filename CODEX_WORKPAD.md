# Portfolio Readiness Reconciliation - 2026-05-12

Scope: part of the workspace-wide portfolio readiness goal. This repo is
included; `physics` and `physical-grounding-lab` are excluded from the wider
goal.

## Live Status

- Branch: `main`
- Dirty surface: `logs/episodes.jsonl`
- Classification: generated/runtime episode evidence. The file is tracked, so
  future runs append to version-controlled data and keep the checkout dirty.

## Validation

```bash
uv run python -m compileall server train
```

Result: passed. The command created local validation byproducts
`openenv_robo_replan.egg-info/` and `uv.lock`; both were removed after the
smoke check.

## Disposition

- Do not discard `logs/episodes.jsonl` without a human decision; it may be
  experiment evidence.
- Presentation-readiness cleanup should decide whether `logs/episodes.jsonl`
  remains a committed fixture, moves to a smaller curated sample, or is replaced
  by ignored runtime output plus a documented reproduction command.
- First follow-up work order: "Separate generated RoboReplan episode logs from
  curated portfolio evidence."

## Evidence Artifact Contract - 2026-05-13

Scope:

- Preserve the existing dirty `logs/episodes.jsonl` append for human review;
  do not commit generated episode additions.
- Move the default runtime log sink to ignored `logs/episodes.local.jsonl`.
- Add a validation command for the committed evidence artifacts.
- Document which `logs/` files are curated fixtures and how runtime logs should
  be promoted.

Validation:

```bash
python3 scripts/validate_evidence_artifacts.py
python3 -m compileall server scripts train
git diff --check
```

Result: passed after adding repo-root import handling to
`scripts/validate_evidence_artifacts.py`.

Gemini secondary review:

- Attempted with Gemini CLI on 2026-05-13 for the scoped evidence-artifact
  contract change.
- Blocked by `MODEL_CAPACITY_EXHAUSTED` / HTTP 429 for
  `gemini-3-flash-preview`; no Gemini findings were returned.
