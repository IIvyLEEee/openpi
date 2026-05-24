# Project Analysis Log: OpenPI UR5e Async Implementation

- Date: 2026-05-24
- Owner: Codex
- Scope: `deploy/inference_real.py`, `deploy/telemetry.py`, `deploy/README.md`

## Context
- User request: implement the VLASH-inspired async inference direction for OpenPI UR5e real deployment and write usage documentation for the previous and new changes.
- Initial assumptions: Keep the first implementation conservative: one background request at a time, explicit overlap steps, optional future-state conditioning, telemetry for measuring hidden latency, and no change to model server protocol.

## Timeline

### 18:24 Step 1 - Implementation Setup
- Action: Checked git status and reviewed the current deployment, telemetry, and README files.
- Evidence (files/commands/output): `git status --short --branch`; `sed -n '1,240p' deploy/inference_real.py`; `sed -n '1,260p' deploy/telemetry.py`; `sed -n '1,260p' deploy/README.md`.
- Observation: The deploy loop blocks on `policy.infer()` before scheduling actions. Existing telemetry already records raw per-call latency and chunk counts, but lacks async boundary wait/hidden latency fields.
- Adjustment: Add a small deploy-local async worker and targeted tests before touching production code.
- Rationale: The websocket client is synchronous, so hiding latency requires real background execution with one outstanding request and explicit timing metrics.
- Next: Write failing tests for async worker behavior and telemetry async fields.

### 18:30 Step 2 - Failing Tests
- Action: Added tests for the async worker, telemetry async metrics, and async chunk-boundary timestamp scheduling.
- Evidence (files/commands/output): `deploy/async_inference_test.py`; `deploy/telemetry_test.py`; `deploy/inference_real_test.py`; `.venv/bin/python -m pytest deploy/async_inference_test.py deploy/telemetry_test.py deploy/inference_real_test.py`.
- Observation: Tests fail because `deploy.async_inference` is missing and `_future_action_schedule` cannot yet accept a `start_timestamp`. Direct `uv run --group dev` attempted to download large GPU dependencies, so it was stopped.
- Adjustment: Keep verification on the existing `.venv` and avoid unrelated runtime imports during unit tests.
- Rationale: Unit tests should exercise deploy helpers without requiring the full OpenPI GPU/server environment.
- Next: Implement the deploy-local async worker, async metrics plumbing, and lazy policy-client import.

### 18:38 Step 3 - Async Runtime Implementation
- Action: Added `deploy/async_inference.py`, async metric support in telemetry, async chunk-boundary timestamp scheduling, and `--async-inference` / `--inference-overlap-steps` / `--async-future-state` runtime arguments.
- Evidence (files/commands/output): `.venv/bin/python -m pytest deploy/async_inference_test.py deploy/telemetry_test.py deploy/inference_real_test.py` passed with 8 tests.
- Observation: The first chunk remains synchronous; later chunks launch one background policy request while current scheduled actions are executing. Next-chunk timestamps start at the current chunk boundary, avoiding overlap with already scheduled actions.
- Adjustment: Moved `openpi_client`, `tyro`, and `yaml` imports to lazy import sites so deploy helper tests do not require the full runtime environment.
- Rationale: The async behavior can now be tested independently from the real robot and policy server dependencies.
- Next: Write user-facing documentation for sync deployment, async deployment, telemetry interpretation, and trajectory plotting.

### 18:48 Step 4 - Usage Documentation
- Action: Added a Chinese usage guide and linked it from `deploy/README.md`.
- Evidence (files/commands/output): `docs/ur5e-real-deployment-usage.md`; `deploy/README.md`.
- Observation: The guide now covers server startup, observe-only mode, sync and async dry/no-execute flows, real execution, telemetry interpretation, per-chunk delay checks, and trajectory/gripper plotting.
- Adjustment: Keep detailed usage in `docs/ur5e-real-deployment-usage.md` and leave `deploy/README.md` as the concise entry point.
- Rationale: The full workflow is easier to maintain as a standalone Chinese runbook while preserving the existing deploy README.
- Next: Run final formatting/checks, complete this log, and commit documentation separately from code.

## Final Summary
- What was confirmed:
  - Async UR5e inference can be integrated without changing the websocket server protocol by using one background worker request at a time.
  - `--inference-overlap-steps` controls the overlap window, with the effective wall-clock window equal to `steps / frequency`.
  - Telemetry can distinguish raw policy latency from visible chunk-boundary wait.
- What changed from initial plan:
  - The implementation keeps imports for heavy runtime dependencies lazy so deploy helper tests run in the lightweight `.venv`.
  - Documentation was split into a detailed Chinese guide plus a short README pointer.
- Open risks or unknowns:
  - True hardware smoothness still needs UR5e validation; unit tests cover scheduling and telemetry helpers, not robot execution.
  - Large overlap values may require VLASH-style delay-offset fine-tuning to preserve task success.
  - `uv run --group dev` attempted to fetch large GPU dependencies, so final verification uses the existing `.venv` and scoped files.
- Recommended next actions:
  - Run `--no-execute --async-inference --inference-overlap-steps=2` against the real policy server and inspect telemetry.
  - Compare sync vs async `async_chunk_boundary_wait_ms` and trajectory smoothness before increasing overlap.
  - If async behavior regresses, try `--no-async-future-state` and then consider delay-offset fine-tuning.
