# Project Analysis Log: LoRA Merge Policy

- Date: 2026-05-25
- Owner: Codex
- Scope: `scripts/serve_policy.py`, `src/openpi/policies/policy_config.py`, `src/openpi/models/lora.py`

## Context
- User request: add logic that can automatically choose whether to merge LoRA weights at inference time, with a command-line switch.
- Initial assumptions: Implement merge at policy loading time for JAX checkpoints, keep deployment clients unchanged, expose `auto/on/off` from `serve_policy.py`.

## Timeline

### 13:55 Step 1 - Current LoRA Load Path
- Action: Read the policy server, policy creation, model load, LoRA modules, and UMI LoRA configs.
- Evidence (files/commands/output): `scripts/serve_policy.py`; `src/openpi/policies/policy_config.py`; `src/openpi/models/lora.py`; `src/openpi/models/gemma.py`; `src/openpi/training/config.py`.
- Observation: LoRA variants create model structures with LoRA params. Current inference loads checkpoint params directly into that LoRA model; LoRA is added dynamically in forward and is not physically folded into base weights.
- Adjustment: Add merge support in `policy_config.create_trained_policy`, before constructing the final `Policy`, so all websocket clients benefit without deploy-side changes.
- Rationale: The decision depends on model config and checkpoint params, both available at server-side policy creation.
- Next: Write failing tests for parameter folding and CLI propagation.

### 14:02 Step 2 - Failing Tests
- Action: Added tests for folding LoRA einsum and feed-forward parameters into non-LoRA modules, and for stripping `_lora` model variants during merge.
- Evidence (files/commands/output): `src/openpi/models/lora_test.py`; `.venv/bin/python -m pytest src/openpi/models/lora_test.py`.
- Observation: The test command currently stops in `src/openpi/conftest.py` because `.venv` lacks `pynvml`; the new imports also target a not-yet-created `openpi.models.lora_merge` module.
- Adjustment: Continue with implementation, then rerun using an environment with the required lightweight test dependency or report the dependency blocker.
- Rationale: The intended RED path is still valid: LoRA merge APIs do not exist yet, and test execution is blocked before collection by an environment dependency.
- Next: Implement a generic JAX/Numpy param-tree LoRA fold utility.

### 14:18 Step 3 - Merge Utility And Policy Loader Integration
- Action: Implemented `openpi.models.lora_merge`, added merge mode plumbing to `policy_config.create_trained_policy`, and exposed `--lora-merge` in `scripts/serve_policy.py`.
- Evidence (files/commands/output): `src/openpi/models/lora_merge.py`; `src/openpi/models/lora_merge_test.py`; `.venv/bin/python -m pytest src/openpi/models/lora_merge_test.py`.
- Observation: The new standalone merge tests pass in the lightweight `.venv` with 4 tests. The merge utility avoids JAX/Flax top-level imports so it can be unit-tested without the full OpenPI runtime stack; it lazily uses `jax.numpy` only if actual JAX arrays are passed.
- Adjustment: Moved new tests out of `lora_test.py` because that existing test file requires Flax/JAX, which are absent from the lightweight `.venv`.
- Rationale: The merge math and mode behavior can be validated with NumPy arrays, while production inference still handles JAX arrays at runtime.
- Next: Document the command-line modes and run formatting/lint/compile checks.

### 14:29 Step 4 - CLI And Documentation
- Action: Added `--lora-merge=auto|on|off` to `scripts/serve_policy.py` and documented the server-side switch in UR5e deployment docs.
- Evidence (files/commands/output): `scripts/serve_policy.py`; `deploy/README.md`; `docs/ur5e-real-deployment-usage.md`; `ruff check src/openpi/models/lora_merge.py src/openpi/models/lora_merge_test.py src/openpi/policies/policy_config.py scripts/serve_policy.py`.
- Observation: `auto` is now the default. `on` forces merge and raises if unsupported or absent; `off` preserves existing runtime LoRA behavior.
- Adjustment: Added a file-level import-order ruff exception in `serve_policy.py` to preserve the existing `datasets` import-order segfault workaround.
- Rationale: Reordering that import would risk regressing the Ubuntu 20.04 workaround already encoded in the script.
- Next: Run final verification and commit.

## Final Summary
- What was confirmed:
  - Current LoRA inference previously used dynamic LoRA branches rather than folded base weights.
  - New merge logic can fold default OpenPI Gemma LoRA pairs into their base weights and strip `_lora` model variants.
  - `serve_policy.py` exposes `--lora-merge=auto|on|off` and passes the setting into policy creation.
- What changed from initial plan:
  - Merge tests are isolated in `lora_merge_test.py` and avoid JAX/Flax top-level dependencies so they run in the lightweight `.venv`.
  - The utility lazily imports `jax.numpy` only for actual JAX arrays.
- Open risks or unknowns:
  - Merge is implemented for the current OpenPI default LoRA axis layout; custom LoRA axes are not encoded in checkpoint params and should be validated separately before using `--lora-merge=on`.
  - PyTorch safetensors checkpoints intentionally reject forced merge for now.
- Recommended next actions:
  - Use `--lora-merge=auto` for normal UR5e policy serving.
  - Use `--lora-merge=off` for A/B checks against previous behavior.
  - Use `--lora-merge=on` only when a run should fail fast if LoRA cannot be folded.
