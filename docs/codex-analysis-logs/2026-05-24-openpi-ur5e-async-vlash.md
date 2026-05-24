# Project Analysis Log: OpenPI UR5e Async VLASH Design

- Date: 2026-05-24
- Owner: Codex
- Scope: `deploy/inference_real.py`, `deploy/telemetry.py`, `/home/wyj24/project/vlash`

## Context
- User request: read arXiv 2512.01031 and local `~/project/vlash`; reason about how to add async inference to OpenPI UR5e real deployment, whether overlap steps are configurable, and whether this can truly reduce per-chunk delay.
- Initial assumptions: The current OpenPI UR5e loop is synchronous at the policy-call level and already has telemetry fields for latency and scheduled chunk counts from the previous task.

## Timeline

### 17:58 Step 1 - Paper And Local Repo Orientation
- Action: Opened arXiv 2512.01031 and scanned the local `vlash` repository.
- Evidence (files/commands/output): arXiv title/abstract and HTML sections; `rg -n -i "async|delay|overlap|future|chunk" /home/wyj24/project/vlash`; `nl -ba /home/wyj24/project/vlash/vlash/run.py`.
- Observation: VLASH frames async as overlapping execution of the current action chunk with inference of the next chunk. The paper's key stabilizer is future-state awareness: roll robot state forward through actions that execute during inference, then condition the next prediction on that execution-time state.
- Adjustment: Treat OpenPI integration as two layers: real concurrency in the UR5e deployment loop, and optional VLASH-style future-state conditioning that likely requires matching training offsets.
- Rationale: Background inference alone can hide latency, but future-state-aware inputs determine whether the resulting chunks remain stable.
- Next: Compare VLASH overlap semantics with current `deploy/inference_real.py` scheduling.

### 18:01 Step 2 - VLASH Runtime Semantics
- Action: Read the local VLASH runtime and configuration code.
- Evidence (files/commands/output): `/home/wyj24/project/vlash/vlash/run.py:58-246`; `/home/wyj24/project/vlash/vlash/configs/run_config.py:70-133`; `/home/wyj24/project/vlash/examples/inference/async.yaml:41-46`.
- Observation: `inference_overlap_steps` is a first-class runtime parameter. VLASH launches the next chunk when `chunk_index == n_action_steps - overlap_steps`, so the overlap window is `overlap_steps / fps` after action quantization. The runtime replaces `observation["observation.state"]` with the final action of the current chunk before predicting the next chunk.
- Adjustment: For OpenPI, expose `async_inference` and `inference_overlap_steps`, but validate the parameter against the actual scheduled execution horizon rather than only the model chunk length.
- Rationale: OpenPI's UR5e script currently schedules a truncated subset of each policy chunk (`steps_per_inference` / `max_scheduled_actions`), so overlap must be tied to the scheduled actions that really execute before the next chunk boundary.
- Next: Identify the blocking policy call and the websocket concurrency constraints in OpenPI.

### 18:01 Step 3 - OpenPI UR5e Blocking Point
- Action: Read the current real deployment loop and websocket policy client.
- Evidence (files/commands/output): `deploy/inference_real.py:268-357`; `packages/openpi-client/src/openpi_client/websocket_client_policy.py:47-54`; `deploy/inference_real.py:58-72`.
- Observation: The UR5e loop is synchronous: it captures an observation, calls `policy.infer(policy_obs)`, normalizes/safety-filters the full chunk, schedules future actions, writes telemetry, and waits until the next batch boundary. `WebsocketClientPolicy.infer()` is also blocking (`send` then `recv`) and should not be used for concurrent sends on the same websocket.
- Adjustment: Implement OpenPI async as a real background worker with a single outstanding request, preferably owning its own websocket client/connection or using a strict request lock.
- Rationale: Copying VLASH's inline manager would not hide OpenPI server latency, because the OpenPI client blocks until the server response is received.
- Next: Check whether future-state-aware training maps cleanly to UMI/UR5e state/action formats.

### 18:01 Step 4 - Training-Side Future State Compatibility
- Action: Read VLASH dataset delay augmentation and OpenPI UMI/UR5e data transforms.
- Evidence (files/commands/output): `/home/wyj24/project/vlash/vlash/datasets/vlash_dataset.py:17-229`; `src/openpi/policies/umi_ur5e_policy.py:10-64`; `src/openpi/training/config.py:359-393`.
- Observation: VLASH trains random offsets by shifting the action chunk and replacing `observation.state` with either recorded future state or previous action as a proxy, requiring matching dimensions for the action-proxy path. OpenPI UMI/UR5e uses a 7D state and 7D action after transforms (TCP position, axis-angle rotation, gripper width), so a boundary action can serve as a practical future-state proxy in deployment.
- Adjustment: Separate an initial measurement-only runtime async experiment from a training-correct VLASH-style experiment.
- Rationale: Runtime async can prove wall-clock stall reduction without retraining, but stable high-overlap behavior likely needs fine-tuning with delay offsets at least as large as the deployment overlap.
- Next: Define the expected latency metric and recommended implementation phases.

## Final Summary
- What was confirmed:
  - VLASH's overlap step count is configurable through `inference_overlap_steps`; the effective wall-clock window is `overlap_steps / control_frequency` after accounting for action quantization.
  - OpenPI's current UR5e loop is synchronous around `policy.infer()`, and the websocket client is a blocking request/response client.
  - A real OpenPI async port needs a background inference worker plus chunk versioning, not only a different loop index.
  - Async can reduce effective per-chunk boundary wait, but it does not reduce raw model/server inference latency.
- What changed from initial plan:
  - The OpenPI design should start with runtime overlap and telemetry before changing training, because the existing script already has per-inference telemetry and scheduled action windows.
  - A faithful VLASH behavior should later add future-state conditioning and delay-offset fine-tuning for UMI/UR5e.
- Open risks or unknowns:
  - Naive async with stale images and no delay-offset fine-tuning may reduce smoothness or success rate at larger overlaps.
  - The first chunk still pays full synchronous latency, and any inference time longer than the overlap window remains visible as boundary stall.
  - The OpenPI server/client path must avoid concurrent requests on one websocket unless request multiplexing is explicitly added.
- Recommended next actions:
  - Add `--async-inference` and `--inference-overlap-steps` to `deploy/inference_real.py`.
  - Implement a single-request background worker, store `policy_call_ms`, `overlap_window_ms`, `chunk_boundary_wait_ms`, `hidden_inference_ms`, and chunk version IDs in telemetry.
  - Use `scheduled_actions[-1]` or the action nearest the next chunk boundary as `observation/state` when launching the next chunk; keep this optional for A/B tests.
  - Run sync vs async UR5e dry/no-execute tests first, then real hardware with small overlap, then fine-tune with delay augmentation if behavior regresses.
