# RoboReplan Architecture Contracts

These contracts make the repo easier for future agents to navigate. They name
the files that own each behavior and the validation that catches drift.

## Episode Evidence Logging

Runtime episode evidence flows through one path:

1. `server/config.py` owns `LogConfig.export_path`.
2. `server/environment.py` passes `EnvConfig.log.export_path` into `EpisodeLogger`.
3. `server/logger.py` owns the JSONL episode schema and append-only write.
4. `logs/episodes.jsonl` is committed evidence, not source code.
5. `logs/episodes.local.jsonl` is the ignored default runtime sink.
6. `scripts/check.sh` must preserve `logs/episodes.jsonl` while smoke tests run.

Validation: `scripts/check_arch_contracts.py` checks this route against the code
and documentation. Run the full local gate with `bash scripts/check.sh`.
