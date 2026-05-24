# Solve `tx_tag_robot` for OpenPI UMI UR5e Deployment

## Goal

We need a 4x4 homogeneous transform:

```text
tx_tag_robot
```

meaning:

```text
pose_in_tag_frame = tx_tag_robot @ pose_in_robot_base_frame
```

This matrix is needed by `/home/liyixuan23/openpi/deploy/configs/umi_ur5e_wsg50.yaml` as `robots[0].unified_tx`.

The OpenPI real listener currently reads UR RTDE poses in robot base frame. The training dataset was generated from hand-held UMI GoPro SLAM and stores TCP poses in the table/tag frame:

```python
tx_tag_cam = tx_tag_slam @ tx_slam_cam
tx_tag_tcp = tx_tag_cam @ tx_cam_tcp
pose_tag_tcp = mat_to_pose(tx_tag_tcp)
```

So the live robot pose must be transformed into the same tag/table frame before being sent to the policy.

## Coordinate Meaning

Use this convention:

```text
T_A_B = pose of frame B expressed in frame A
```

The training data directly contains:

```text
tx_tag_tcp = T_tag_tcp
```

This is the TCP pose in the tag/table frame. It is not a fixed calibration matrix; it changes at every timestep.

The live robot gives:

```text
tx_robot_tcp = T_robot_tcp
```

This is the TCP pose in the UR robot base frame. It also changes at every timestep.

The fixed calibration we need for deployment is:

```text
tx_tag_robot = T_tag_robot
```

The relationship is:

```text
tx_tag_tcp = tx_tag_robot @ tx_robot_tcp
```

During policy inference:

```text
live observation:  tx_tag_tcp_live    = tx_tag_robot @ tx_robot_tcp_live
policy action:     tx_robot_tcp_goal  = inverse(tx_tag_robot) @ tx_tag_tcp_goal
```

`pi0` itself does not know whether the state/action is TCP pose, joint pose, robot frame, or tag frame. It learns the representation provided by the data transform. In this UMI setup, the model was trained on tag-frame TCP poses, so real inference must present observations in tag frame and convert predicted tag-frame TCP targets back to robot frame for execution.

## Why Training Does Not Need `tx_robot_tcp`

The hand-held UMI training collection has no robot base frame, so it does not and cannot use `tx_robot_tcp`.

The training pipeline only needs to reconstruct the hand-held gripper TCP pose in the tag/table frame:

```text
tx_tag_tcp = tx_tag_slam @ tx_slam_cam @ tx_cam_tcp
```

That is enough for training because both state and action are stored in the same tag/table coordinate system.

`tx_robot_tcp` appears only at real-robot deployment time, because the UR controller reports and accepts TCP poses in robot base coordinates. Therefore deployment needs `tx_tag_robot` to bridge between:

```text
model/world frame: tag/table frame
controller frame:  UR robot base frame
```

Original UMI real-robot replay follows the same idea: it stores replay poses as `tx_tag_tcp`, uses a fixed `tx_tag_robot`, then computes `tx_robot_tcp = inverse(tx_tag_robot) @ tx_tag_tcp` before sending poses to the robot. It does not infer `tx_robot_tcp` from hand-held data.

## Important Constraint

The original hand-held dataset alone cannot uniquely determine `tx_tag_robot`, because it contains tag-frame hand-held TCP poses but no UR robot-base poses.

To solve `tx_tag_robot`, we need pose correspondences:

```text
tx_tag_tcp_i    from original dataset / tag-frame target pose
tx_robot_tcp_i  from live UR robot at the matching physical pose
```

Then:

```text
tx_tag_robot_i = tx_tag_tcp_i @ inverse(tx_robot_tcp_i)
```

With one very accurate 6D correspondence this may be usable. With 3+ correspondences, compute all candidates and average/validate them.

## Inputs To Locate On Data Machine

Find the session or zarr used for training, probably created by:

```bash
cd /data/archive_liyixuan23/umi_for_train
python run_slam_pipeline.py <session_dir>
python scripts_slam_pipeline/07_generate_replay_buffer.py <session_dir> -o <mixdataset.zarr.zip>
```

Useful files:

- `<session_dir>/dataset_plan.pkl`
- `<session_dir>/demos/mapping/tx_slam_tag.json`
- `<mixdataset.zarr.zip>`
- The OpenPI converted LeRobot dataset, if available
- Any notes/logs containing live UR poses at manually aligned positions

## Step 1: Extract Training Initial Poses

Run this on the data machine. It extracts `demo_start_pose`, first frame pose, and basic state statistics from the zarr.

```bash
cd /data/archive_liyixuan23/umi_for_train

python - <<'PY'
import json
import pathlib
import pickle
import zipfile
import tempfile

import numpy as np
import zarr

zarr_zip = pathlib.Path("<PATH_TO_MIXDATASET_ZARR_ZIP>")
session_dir = pathlib.Path("<PATH_TO_SESSION_DIR>")  # directory containing dataset_plan.pkl

def arr(x):
    return np.asarray(x, dtype=np.float64)

print("zarr_zip:", zarr_zip)
print("session_dir:", session_dir)

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    with zipfile.ZipFile(zarr_zip) as archive:
        archive.extractall(td / "mixdataset.zarr")

    root = zarr.open_group(str(td / "mixdataset.zarr"), mode="r")
    ends = arr(root["meta/episode_ends"][:]).astype(int)
    starts = np.r_[0, ends[:-1]]

    pos = arr(root["data/robot0_eef_pos"][:])
    rot = arr(root["data/robot0_eef_rot_axis_angle"][:])
    grip = arr(root["data/robot0_gripper_width"][:]).reshape(-1, 1)
    state = np.concatenate([pos, rot, grip], axis=1)

    print("num_episodes", len(ends))
    print("num_frames", len(state))
    print("state mean", np.round(state.mean(axis=0), 6).tolist())
    print("state std ", np.round(state.std(axis=0), 6).tolist())
    print("state q01 ", np.round(np.quantile(state, 0.01, axis=0), 6).tolist())
    print("state q99 ", np.round(np.quantile(state, 0.99, axis=0), 6).tolist())

    print("\nFirst-frame poses for first 20 episodes:")
    for ep, s in enumerate(starts[:20]):
        print(ep, np.round(state[s, :6], 6).tolist(), "gripper", round(float(state[s, 6]), 6))

plan_path = session_dir / "dataset_plan.pkl"
if plan_path.exists():
    plan = pickle.load(plan_path.open("rb"))
    print("\nDataset plan episodes:", len(plan))
    print("First 20 demo_start_pose entries:")
    for ep, item in enumerate(plan[:20]):
        g = item["grippers"][0]
        print(ep, "demo_start_pose", np.round(g["demo_start_pose"], 6).tolist())
        print(ep, "first_tcp_pose ", np.round(g["tcp_pose"][0], 6).tolist())
else:
    print("No dataset_plan.pkl found at", plan_path)
PY
```

Replace:

```text
<PATH_TO_MIXDATASET_ZARR_ZIP>
<PATH_TO_SESSION_DIR>
```

with the actual paths.

Save the output to a text file, for example:

```bash
python extract_training_poses.py | tee /tmp/training_tag_poses.txt
```

## Step 2: Choose Matching Tag-Frame Poses

Pick one or more `tx_tag_tcp` poses from the extracted data that correspond to physical poses we can reproduce with the UR robot.

Best choices:

- start pose of a demo, if the robot was manually placed at the same initial pose during the failed inference logs;
- a pose directly above the red block;
- a pose directly above the target box;
- any pose that can be accurately reproduced with the UR TCP.

For each chosen correspondence, record:

```text
tag_pose_i = [x, y, z, rx, ry, rz] from dataset
robot_pose_i = [x, y, z, rx, ry, rz] from UR getActualTCPPose at the matching physical pose
```

If only one robot pose is available from the current logs, use it as a first diagnostic, but prefer collecting 3+ correspondences later.

## Step 3: Solve Candidate `tx_tag_robot`

Create `/tmp/solve_tx_tag_robot.py`:

```python
import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_mat(pose):
    pose = np.asarray(pose, dtype=np.float64)
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    mat[:3, 3] = pose[:3]
    return mat


def mat_to_pose(mat):
    mat = np.asarray(mat, dtype=np.float64)
    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = mat[:3, 3]
    pose[3:] = Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    return pose


# Fill these with matched pairs.
# tag_pose is from training zarr / dataset_plan, in tag frame.
# robot_pose is from live UR getActualTCPPose, in robot base frame.
pairs = [
    {
        "tag_pose":   [0, 0, 0, 0, 0, 0],
        "robot_pose": [0, 0, 0, 0, 0, 0],
    },
]

candidates = []
for pair in pairs:
    tx_tag_tcp = pose_to_mat(pair["tag_pose"])
    tx_robot_tcp = pose_to_mat(pair["robot_pose"])
    tx_tag_robot = tx_tag_tcp @ np.linalg.inv(tx_robot_tcp)
    candidates.append(tx_tag_robot)

print("candidate tx_tag_robot matrices:")
for i, tx in enumerate(candidates):
    print("candidate", i)
    print(np.array2string(tx, precision=8, suppress_small=False))

if len(candidates) == 1:
    tx = candidates[0]
else:
    # Simple first-pass average: average translation and rotation vectors around candidate 0.
    # Good correspondences should already be very close. If not, do not trust the average.
    translations = np.stack([tx[:3, 3] for tx in candidates])
    rotations = Rotation.from_matrix(np.stack([tx[:3, :3] for tx in candidates]))
    mean_t = translations.mean(axis=0)
    mean_r = rotations.mean().as_matrix()
    tx = np.eye(4)
    tx[:3, :3] = mean_r
    tx[:3, 3] = mean_t

print("\nselected tx_tag_robot:")
print(np.array2string(tx, precision=8, suppress_small=False))

print("\nyaml:")
for row in tx:
    print("      - [" + ", ".join(f"{v:.10g}" for v in row) + "]")

print("\nvalidation per pair:")
for i, pair in enumerate(pairs):
    pred_tag_tcp = tx @ pose_to_mat(pair["robot_pose"])
    pred_pose = mat_to_pose(pred_tag_tcp)
    err = pred_pose - np.asarray(pair["tag_pose"], dtype=np.float64)
    print(i, "pred_tag_pose", np.round(pred_pose, 6).tolist())
    print(i, "target_tag_pose", np.round(pair["tag_pose"], 6).tolist())
    print(i, "pose_error", np.round(err, 6).tolist())
```

Run:

```bash
python /tmp/solve_tx_tag_robot.py
```

## Step 4: Validate Against Training Distribution

Use the selected matrix to transform the failed live log poses from robot frame into tag frame. The transformed values should fall close to training state statistics.

Use these live poses from the failed logs as initial checks:

```text
live_003817_first = [0.0499, -0.4259, 0.1485, -0.1196, -2.7485, 1.1235]
live_003055_first = [0.0351, -0.5575, 0.0897, 0.1531, 2.8719, -1.1247]
```

After applying `tx_tag_robot`, the pose should be comparable to the training `q01/q99` ranges.

Known pick-place training state ranges from the checkpoint on the robot machine:

```text
q01 = [-0.286279, -0.415979, -0.021693, -2.814927, -0.930501, -0.465659, 0.027434]
q99 = [ 0.194232,  0.150012,  0.321300, -2.187532,  1.047045,  0.126189, 0.085716]
```

If the transformed live pose is still far outside these ranges, especially rotation, then the correspondence or transform direction is wrong.

Also test the inverse:

```python
np.linalg.inv(tx_tag_robot)
```

If inverse gives better distribution match, the direction was reversed.

## Step 5: Output Needed For Robot Machine

Return the following:

1. The selected `tx_tag_robot` 4x4 matrix in YAML list format.
2. The pose correspondences used:

```text
tag_pose_i
robot_pose_i
```

3. The validation errors for each correspondence.
4. The transformed failed-log poses and whether they fall in the training `q01/q99` range.
5. The original paths used:

```text
session_dir
mixdataset.zarr.zip
dataset_plan.pkl
tx_slam_tag.json
```

## How It Will Be Used In OpenPI

The final YAML should be inserted here:

```yaml
robots:
  - robot_type: ur5e
    robot_ip: 192.168.1.2
    robot_obs_latency: 0.0001
    robot_action_latency: 0.1
    tcp_offset: 0.235
    height_threshold: -0.024
    sphere_radius: 0.1
    sphere_center: [0, -0.06, -0.185]
    unified_tx:
      - [...]
      - [...]
      - [...]
      - [...]
```

OpenPI's controller will then:

- convert robot observations from robot frame to tag frame before sending state to the policy;
- convert policy actions from tag frame back to robot frame before sending waypoints to UR.

That matches the frame convention used by the hand-held UMI training data.
