# Project Analysis Log: UR5e Deploy Telemetry

- Date: 2026-05-24
- Owner: Codex
- Scope: `deploy/`, `src/openpi/policies/umi_ur5e_policy.py`, UR5e real-robot deployment tooling

## Context
- User request: research how to measure model inference latency, real-time action chunk counts, and visualize UR5e trajectory plus gripper opening state during true UR5e deployment experiments.
- Initial assumptions: the target path is the `4070-20_04` branch deployment flow under `deploy/inference_real.py`; changes should be committed after meaningful milestones.

## Timeline

### 16:52 Step 1 - Start Analysis Log
- Action: Created this analysis log before deep code exploration.
- Evidence (files/commands/output): `mkdir -p docs/codex-analysis-logs`; log file under `docs/codex-analysis-logs/`.
- Observation: User explicitly requested research plus careful git management, so the work should leave a durable engineering record and commits.
- Adjustment: Plan to implement small, testable telemetry utilities instead of only giving prose.
- Rationale: The deployment loop already logs some transient values, but experiment analysis needs persistent structured data.
- Next: Inspect deployment loop, environment observation/action recording, and existing tests.

### 16:53 Step 2 - Read Deployment Loop And Recording Path
- Action: Inspected `deploy/inference_real.py`, `deploy/umi/real_world/bimanual_umi_env.py`, replay buffer references, and existing tests.
- Evidence (files/commands/output): `nl -ba deploy/inference_real.py`; `nl -ba deploy/umi/real_world/bimanual_umi_env.py`; `rg "replay_buffer|start_episode|exec_actions|get_obs" deploy`; `rg --files | rg 'test'`.
- Observation: `deploy/inference_real.py` already computes transient `infer_ms`, `len(actions)`, and `len(scheduled_actions)` in the loop, but only logs them. `BimanualUmiEnv` can persist action, EEF pose, joint, and gripper width into `replay_buffer.zarr`, but only after `start_episode()` is called; current real deployment entrypoint never calls it.
- Adjustment: Add explicit structured telemetry and a `--record-episode` switch instead of relying on console logs.
- Rationale: Console logs are hard to post-process and do not provide a stable input for trajectory/gripper visualization.
- Next: Write failing tests for JSONL telemetry records and trajectory plotting data preparation.

### 16:56 Step 3 - RED Tests For Telemetry And Plotting
- Action: Added tests for JSONL inference telemetry and trajectory/gripper data preparation, then ran them before implementation.
- Evidence (files/commands/output): `.venv/bin/python -m pytest deploy/telemetry_test.py deploy/plot_umi_trajectory_test.py` exited with collection errors: `No module named 'deploy.telemetry'` and `No module named 'deploy.plot_umi_trajectory'`.
- Observation: The tests correctly describe missing modules rather than passing against existing behavior.
- Adjustment: Use a minimal local `.venv` with only `pytest numpy` for focused tests instead of full `uv sync`.
- Rationale: Full sync started downloading large training/GPU dependencies; the focused tests only need pytest and NumPy.
- Next: Implement the tested modules and wire telemetry into `deploy/inference_real.py`.

### 17:00 Step 4 - Implement Telemetry And Plotting
- Action: Implemented `deploy/telemetry.py`, `deploy/plot_umi_trajectory.py`, wired telemetry and `--record-episode` into `deploy/inference_real.py`, formatted touched Python files, and committed the first implementation milestone.
- Evidence (files/commands/output): `.venv/bin/python -m pytest deploy/telemetry_test.py deploy/plot_umi_trajectory_test.py` reported `6 passed`; formatter ran `ruff format`; commit `44a10ca Add UR5e deploy telemetry and trajectory plotting`.
- Observation: Each inference can now emit JSONL records with latency, returned chunk size, scheduled chunk size, and executed chunk size; replay buffer recording can be enabled from the real deployment CLI; trajectory plotting is available as an offline utility.
- Adjustment: Add README instructions for experiment operators before finalizing.
- Rationale: The code paths are only useful if the exact run and post-processing commands are documented next to the deployment instructions.
- Next: Update `deploy/README.md`, rerun focused tests, and commit documentation.

### 17:05 Step 5 - Documentation And Verification
- Action: Updated `deploy/README.md` with telemetry fields, run commands, replay buffer recording, plotting command, and quick JSONL summary snippet. Fixed lint issues in the new plotting module.
- Evidence (files/commands/output): Commit `e4dd28f Document UR5e telemetry workflow`; `.venv/bin/python -m pytest deploy/telemetry_test.py deploy/plot_umi_trajectory_test.py` reported `6 passed`; `python -m py_compile deploy/inference_real.py deploy/telemetry.py deploy/plot_umi_trajectory.py deploy/telemetry_test.py deploy/plot_umi_trajectory_test.py` exited 0; `ruff check deploy/telemetry.py deploy/plot_umi_trajectory.py deploy/telemetry_test.py deploy/plot_umi_trajectory_test.py` reported all checks passed.
- Observation: Focused tests and new module lint are clean. A broader `ruff check` including `deploy/inference_real.py` still reports existing path-hack/lazy-import issues (`E402`, `PLC0415`) plus a nested-context suggestion; these are not introduced by the telemetry logic, except that the new telemetry import participates in the same existing path-hack pattern.
- Adjustment: Do not rewrite the deployment script import strategy in this task.
- Rationale: The local imports delay heavy hardware/CV imports and the sys.path block is already used to support direct script execution.
- Next: Commit final cleanup and report usage.

## Final Summary
- What was confirmed: UR5e deployment already computed inference latency and chunk counts in console logs; the implementation now persists them per inference in JSONL. Replay buffer trajectory/gripper data is available when `--record-episode` enables `BimanualUmiEnv.start_episode()`.
- What changed from initial plan: Added code rather than only a research note, because the deployment entrypoint lacked structured metrics and did not expose replay buffer recording.
- Open risks or unknowns: `inference_latency_ms` is client-observed policy call latency, including websocket/serialization/server work; pure model GPU compute time would need server-side instrumentation in `scripts/serve_policy.py` or the policy object. `--record-episode` is intended for actual execution; no-execute/observe-only runs do not produce meaningful executed-action trajectories.
- Recommended next actions: Run a short `--no-execute` telemetry collection to validate policy latency and chunk sizes, then run a guarded real execution with `--record-episode` and plot the resulting `replay_buffer.zarr`.
