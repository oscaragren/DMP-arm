# DMP-Master-Thesis

Master thesis by Oscar Ågren: **personalization using Dynamic Movement Primitives (DMP) and embedded implementation evaluation**.

The project captures human left-arm motion (via RGB/depth or video), derives joint angles, and fits/rolls out DMPs for trajectory representation and reproduction.

---

## Project structure

```
DMP-Master-Thesis/
├── capture/                 # Data capture and pose estimation
│   ├── 2d/
│   │   ├── simple_capture.py   # Record RGB video from OAK-D → data/raw
│   │   ├── pose_estimation.py  # MediaPipe Pose on video → keypoints_3d.npy
│   │   └── clean_keypoints.py  # Interpolate gaps, filter, resample keypoints
│   └── 3d_pose.py              # Live 3D pose (OAK-D RGB + depth) → left_arm_seq_camera.npy
├── dmp/                    # Dynamic Movement Primitives
│   ├── dmp.py              # Fit DMP from demos, rollout_simple (Euler), rollout_rk4
│   └── integration.py      # RK4 ODE integration
├── kinematics/             # Arm geometry and angles
│   ├── left_arm_angles.py  # Elbow flexion, shoulder 3-DOF (elevation, azimuth, internal rotation)
│   └── vector_math.py      # Vector utilities
├── mapping/
│   └── sequence_to_angles.py  # 3D keypoint sequence (T,4,3) → elbow + shoulder angles
├── vis/                    # Visualization
│   ├── plot_trajectory.py     # 3D trajectory from left_arm_seq_camera.npy
│   └── plot_left_arm_angles.py # Joint angles over time (from sequence_to_angles)
├── sim/                    # Simulation (Open3D arm replay)
│   ├── arm_fk.py            # FK from 4 DoF angles to link transforms
│   ├── load_arm_meshes.py    # Load arm STLs with URDF scale/orientation
│   └── visualize_replay.py   # Replay angles.npz in 3D (Open3D)
├── tests/
│   └── test_dmp.py         # DMP fit + rollout reproduction tests
├── data/                   # Main data (raw + processed)
│   ├── raw/                # subject_XX / motion / trial_XXX / video.mp4, meta.json
│   └── processed/          # keypoints, sequences, angles, meta per trial
├── test_data/              # Same layout for testing/development
├── public/                 # External models (e.g. human-pose-estimation-3d-0001)
└── requirements.txt
```

---

## Implementation overview

### Capture

- **`capture/2d/simple_capture.py`** — Records RGB video from OAK-D to `data/raw/<subject>/<motion>/<trial>/` (video.mp4, meta.json). Uses a short countdown then fixed recording duration.
- **`capture/2d/pose_estimation.py`** — Runs MediaPipe Pose Landmarker on recorded video; outputs `keypoints_3d.npy` (T, K, 3) and confidence for left arm (shoulder, elbow, wrist) to `data/processed/...`.
- **`capture/2d/clean_keypoints.py`** — Cleans keypoints: interpolates short low-confidence gaps, optional low-pass filter, resampling to fixed dt. Writes resampled keypoints and meta.
- **`capture/3d_pose.py`** — Live pipeline: OAK-D RGB + depth, MediaPipe 2D pose, depth back-projection to 3D. Records left arm + right shoulder (4 keypoints) in camera frame to `left_arm_seq_camera.npy` and `left_arm_t.npy` under a trial directory (e.g. `test_data/processed/...`).

### Kinematics and mapping

- **`kinematics/left_arm_angles.py`** — From 3D keypoint sequence `(T, 4, 3)` [left_shoulder, left_elbow, left_wrist, right_shoulder]:
  - **Elbow:** flexion angle (degrees).
  - **Shoulder:** 3-DOF in a trunk frame (elevation, azimuth, internal rotation), degrees.
- **`mapping/sequence_to_angles.py`** — Loads `left_arm_seq_camera.npy` from a trial dir, calls kinematics, saves `angles.npz` (elbow_deg, shoulder_deg) and updates `meta.json`. Can be run per trial via CLI (`--path` or `--subject`/`--motion`/`--trial`).

### DMP

- **`dmp/dmp.py`** — Fit a single DMP from a list of joint-angle demos (each shape `(T, n_joints)`). Uses a canonical phase, RBF basis, and transformation system; supports different trajectory lengths via shared resampling (same tau, dt). Exposes:
  - **`fit()`** — Returns a `DMPModel` (weights, centers, widths, alpha/beta, tau, n_joints).
  - **`rollout_simple()`** — Euler integration from initial `q0` to goal `g`.
  - **`rollout_rk4()`** — RK4 integration for the same model and boundary conditions.
- **`dmp/integration.py`** — Generic RK4 ODE integrator used by rollout.

DMPs are defined in joint space (e.g. 5 DoF: elbow + shoulder 3-DOF + optional); demos are expected to be resampled to the same length before fitting.

### Visualization

- **`vis/plot_trajectory.py`** — Loads `left_arm_seq_camera.npy` and `left_arm_t.npy` from a trial dir; plots 3D trajectories and arm stick at sample frames; can save figure or show.
- **`vis/plot_left_arm_angles.py`** — Loads the same sequence, runs `mapping.sequence_to_angles`, plots elbow flexion and shoulder angles over time.

### Data layout (per trial)

- **Raw:** `data/raw/<subject>/<motion>/<trial>/` — `video.mp4`, `meta.json` (subject, motion, trial, n_frames, fps, etc.).
- **Processed:** `data/processed/...` or `test_data/processed/...` — `keypoints_3d.npy`, optional resampled/cleaned keypoints, `left_arm_seq_camera.npy`, `left_arm_t.npy`, `angles.npz` (after mapping), `meta.json`.

---

## Requirements

- Python 3 (tested with 3.10)
- See `requirements.txt`:

  ```
  numpy
  depthai>=3.3.0
  opencv-python>=4.5.0
  mediapipe>=0.10.0
  scipy>=1.9.0
  matplotlib>=3.5.0
  ```

- **Hardware (for live 3D):** OAK-D (or compatible) for RGB + depth. For 2D-only capture, OAK-D RGB or any camera usable with OpenCV is sufficient.

---

## Usage (examples)

- **Record raw video (2D pipeline):**
  ```bash
  python capture/2d/simple_capture.py --subject 1 --motion reach --trial 1
  ```

- **Pose estimation on recorded video:**
  ```bash
  python capture/2d/pose_estimation.py --subject 1 --motion reach --trial 1
  ```

- **Clean and resample keypoints:**
  ```bash
  python capture/2d/clean_keypoints.py --path data/processed/subject_01/reach/trial_001
  ```

- **Live 3D pose recording (OAK-D):**
  ```bash
  python capture/3d_pose.py --subject 1 --motion reach --trial 1
  ```

- **Convert 3D sequence to angles and save in trial dir:**
  ```bash
  python mapping/sequence_to_angles.py --path test_data/processed/subject_01/reach/trial_001
  # or: --subject 1 --motion reach --trial 1
  ```

- **Plot 3D trajectory:**
  ```bash
  python vis/plot_trajectory.py --path test_data/processed/subject_01/reach/trial_001
  ```

- **Plot left-arm angles:**
  ```bash
  python vis/plot_left_arm_angles.py --path test_data/processed/subject_01/reach/trial_001
  ```

- **Plot DMP trajectory (demo vs generated from angles.npz):**
  ```bash
  python vis/plot_dmp_trajectory.py --path test_data/processed/subject_01/reach/trial_001
  ```

- **Replay arm motion in 3D (Open3D):**
  ```bash
  pip install open3d   # if not already installed
  python sim/visualize_replay.py --path test_data/processed/subject_01/reach/trial_001
  ```

- **Run DMP tests:**
  ```bash
  python tests/test_dmp.py
  ```

---

## Current state

- **Done:** 2D capture (video + MediaPipe pose), keypoint cleaning/resampling, live 3D capture (OAK-D + depth), kinematics (elbow + shoulder 3-DOF), mapping from 3D sequence to angles, DMP fit and rollouts (Euler + RK4), trajectory and angle plotting, **simulation visualization (Open3D arm replay from angles.npz)**, unit tests for DMP reproduction.
- **Data:** Raw and processed layouts under `data/` and `test_data/` with subject/motion/trial hierarchy; meta.json and .npy/.npz outputs.
- **Not in repo:** Large assets (e.g. MediaPipe `.task` bundles, OpenVINO model weights) are expected to be added or referenced separately as needed.
