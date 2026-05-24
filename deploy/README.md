# UMI UR5e WSG50 Deployment

This directory contains a self-contained real-robot listener for a single UMI UR5e setup with one UVC/GoPro camera and one WSG50 gripper.

Start the OpenPI policy server:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi0_umi_ur5e_pick_place_lora \
  --policy.dir=checkpoints/openpi-ur5e-lora/pi0_pick_place/29999
```

For the conveyor checkpoint, use the matching conveyor config:

```bash
uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi0_umi_ur5e_conveyor_lora \
  --policy.dir=checkpoints/openpi-ur5e-lora/pi0_conveyor/29999
```

Run a server-only smoke test:

```bash
uv run --group umi deploy/inference_real.py \
  --policy-server-host=localhost \
  --policy-server-port=8000 \
  --dry-run
```

Observe hardware without policy inference:

```bash
uv run --group umi deploy/inference_real.py \
  --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
  --observe-only \
  --max-duration=10
```

By default this opens an OpenCV window showing `raw | policy`: the raw camera frame and the frame sent to OpenPI. The listener applies the UMI gripper mask before sending the image to OpenPI. Press `Esc` in the window to stop. Use `--no-show-camera` to disable the window.

Run policy inference without executing actions:

```bash
uv run --group umi deploy/inference_real.py \
  --policy-server-host=<server-ip> \
  --policy-server-port=8000 \
  --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
  --prompt="<task instruction>" \
  --no-execute \
  --steps-per-inference=6 \
  --max-duration=10
```

Run real execution:

```bash
LOG=deploy/logs/inference_real_$(date +%Y%m%d_%H%M%S).log
echo "logging to $LOG"
uv run --group umi deploy/inference_real.py \
  --policy-server-host=localhost \
  --policy-server-port=8000 \
  --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
  --steps-per-inference=8 \
  --prompt="Pick up the red block and put it in the green box." 2>&1 | tee "$LOG"
```
