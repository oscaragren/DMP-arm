"""
Simple PyBullet visualization of left-arm trajectories from angles data.

Loads angles.npz (or DMP rollout), maps 4 DOF to arm joints, and plays back in the GUI.
Run from project root: python sim/simulation.py [--trial-dir path]

Currently the arm is not connected.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as pb

# Project root for loading trial data
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def load_angles_trajectory(trial_dir: Path) -> np.ndarray:
    """Load (T, 4) angles from angles.npz. Returns array in radians, finite only."""
    npz_path = trial_dir / "angles.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Expected angles.npz at {npz_path}")
    data = np.load(npz_path)
    if "elbow_rad" in data and "shoulder_rad" in data:
        elbow = np.atleast_1d(data["elbow_rad"])
        shoulder = np.atleast_2d(data["shoulder_rad"])
    else:
        # Backwards compatibility for older files that only store degrees.
        elbow_deg = np.atleast_1d(data["elbow_deg"])
        shoulder_deg = np.atleast_2d(data["shoulder_deg"])
        elbow = np.deg2rad(elbow_deg)
        shoulder = np.deg2rad(shoulder_deg)

    if shoulder.ndim == 1:
        shoulder = shoulder[:, None]
    q = np.column_stack([elbow, shoulder])
    valid = np.all(np.isfinite(q), axis=1)
    q = q[valid]
    if q.shape[0] < 2:
        raise ValueError("Not enough valid samples in angles.npz")
    return q


def get_arm_joint_indices(physics_client: int, body_uid: int, joint_names: list[str]) -> list[int]:
    """Return joint indices for the given names (order preserved)."""
    name_to_index = {}
    num_joints = pb.getNumJoints(body_uid, physicsClientId=physics_client)
    for i in range(num_joints):
        info = pb.getJointInfo(body_uid, i, physicsClientId=physics_client)
        jname = info[1].decode("utf-8") if isinstance(info[1], bytes) else info[1]
        name_to_index[jname] = i
    indices = []
    for name in joint_names:
        if name not in name_to_index:
            raise KeyError(f"Joint {name} not found in URDF")
        indices.append(name_to_index[name])
    return indices


def main():
    parser = argparse.ArgumentParser(description="Visualize left-arm trajectory in PyBullet")
    parser.add_argument(
        "--trial-dir",
        type=Path,
        default=_project_root / "test_data" / "processed" / "subject_01" / "reach" / "trial_002",
        help="Trial directory containing angles.npz",
    )
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Playback time step (s)")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    args = parser.parse_args()

    # 1. Connect and paths (URDF uses "arm/...", so search path = sim so that arm/ = sim/arm/)
    physics_client = pb.connect(pb.GUI)
    sim_dir = _project_root / "sim"
    pb.setAdditionalSearchPath(str(sim_dir))
    urdf_path = sim_dir / "arm" / "left_arm.urdf"
    if not urdf_path.exists():
        pb.disconnect(physics_client)
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # 2. Load arm
    body_uid = pb.loadURDF(
        "arm/left_arm.urdf",
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=True,
        physicsClientId=physics_client,
    )

    # 3. Map our 4 angles to URDF joints; set all 7 arm joints so the chain stays connected
    #    [elbow_flexion, shoulder_elevation, shoulder_azimuth, shoulder_internal_rot]
    #    NOTE: For this URDF, rotx/roty appear swapped visually vs our intended anatomy,
    #    so we map shoulder_elevation -> roty and internal_rot -> rotx.
    #    -> jLeftElbow_roty, jLeftShoulder_rotz, jLeftShoulder_roty, jLeftShoulder_rotx
    controlled_names = [
        "jLeftElbow_roty",
        "jLeftShoulder_rotz",
        "jLeftShoulder_roty",
        "jLeftShoulder_rotx",
    ]
    # Wrist and elbow twist: keep at 0 so arm doesn't drift
    arm_fixed_names = ["jLeftElbow_rotz", "jLeftWrist_rotx", "jLeftWrist_rotz"]
    controlled_indices = get_arm_joint_indices(physics_client, body_uid, controlled_names)
    fixed_arm_indices = get_arm_joint_indices(physics_client, body_uid, arm_fixed_names)

    # 4. Load trajectory in radians
    try:
        q_rad = load_angles_trajectory(args.trial_dir)
    except FileNotFoundError:
        # Fallback: short dummy trajectory so the script still runs
        T = 60
        t = np.linspace(0, 1, T)
        q_rad = np.deg2rad(np.column_stack([
            30 + 20 * np.sin(2 * np.pi * t),   # elbow
            45 + 30 * np.sin(2 * np.pi * t),  # shoulder elevation
            np.zeros(T),                       # azimuth
            np.zeros(T),                       # internal rot
        ]))
        print("angles.npz not found; using dummy trajectory")
    T = q_rad.shape[0]

    # 5. Kinematic playback: no gravity, set joint state directly so the arm stays connected
    pb.setGravity(0, 0, 0, physicsClientId=physics_client)
    num_joints = pb.getNumJoints(body_uid, physicsClientId=physics_client)
    # Disable motors so resetJointState takes effect
    for j in range(num_joints):
        pb.setJointMotorControl2(
            body_uid, j, pb.POSITION_CONTROL, force=0.0, physicsClientId=physics_client
        )

    # 6. Playback loop: set all arm joints each frame (4 from trajectory, 3 fixed at 0)
    dt = args.dt
    while True:
        for t in range(T):
            for i, jidx in enumerate(controlled_indices):
                pb.resetJointState(
                    body_uid, jidx, float(q_rad[t, i]), 0.0, physicsClientId=physics_client
                )
            for jidx in fixed_arm_indices:
                pb.resetJointState(body_uid, jidx, 0.0, 0.0, physicsClientId=physics_client)
            pb.stepSimulation(physicsClientId=physics_client)
            time.sleep(dt)
        if not args.loop:
            break
    print("Playback finished. Close the PyBullet window to exit.")
    input("Press Enter to disconnect and exit...")
    pb.disconnect(physics_client)


if __name__ == "__main__":
    main()
