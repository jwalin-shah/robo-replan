# Evidence Artifacts

RoboReplan keeps a small set of committed evidence files for portfolio review
and hackathon judging. Treat these files as curated fixtures, not as the live
runtime log sink.

## Curated Files

| File | Purpose |
|---|---|
| `logs/episodes.jsonl` | Representative episode trajectories used by plotting and portfolio review. |
| `logs/env_quality_report.json` | Environment quality summary by difficulty level. |
| `logs/eval_protocol_3seed.json` | Three-seed scripted policy evaluation protocol. |
| `logs/hard20_benchmark.json` | Focused hard-level benchmark summary. |

## Runtime Output

New environment runs append to `logs/episodes.local.jsonl` by default through
`LogConfig.export_path`. The local JSONL path is gitignored. If a run should
become portfolio evidence, promote a small reviewed sample into
`logs/episodes.jsonl` and rerun the evidence validator.

## Validation

The authoritative local validation gate is:

```bash
bash scripts/check.sh
```

Expected result:

- Environment smoke tests pass across easy, medium, and hard levels.
- Action invariant checks reject invalid actions and accept valid actions.
- Architecture contracts match the runtime logging path.
- All tracked JSON and JSONL evidence files parse.
- Required summary fields and episode fields are present.
- `LogConfig` defaults to the ignored local runtime JSONL path.
- No ignored local runtime JSONL file is tracked.
- `server/`, `scripts/`, and `train/` compile.
- `git diff --check` passes.
