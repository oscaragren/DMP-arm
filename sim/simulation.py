"""
Play back a DMP‑generated left‑arm trajectory on the standalone arm model in PyBullet.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
import numpy as np

import pybullet as p

# Ensure project root is on sys.path for imports
_sim_dir = Path(__file__).resolve().parent
_project_root = _sim_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sim.joint_limits import clamp_dmp_vector


def _load_dmp_trajectory(trial_dir: Path) -> tuple[np.ndarray, float]:
    q_traj = np.load(trial_dir / "dmp_rollout.npz")["q_gen"]
    dt = np.load(trial_dir / "dmp_rollout.npz")["dt"]
    print(q_traj)
    return np.deg2rad(q_traj), float(dt)

def joint_index(body_uid: int, joint_name: str) -> int:
    """Resolve a joint name to its PyBullet index."""
    num_joints = p.getNumJoints(body_uid)
    for i in range(num_joints):
        info = p.getJointInfo(body_uid, i)
        name = info[1].decode("utf-8") if isinstance(info[1], (bytes, bytearray)) else str(
            info[1]
        )
        if name == joint_name:
            return i
    raise KeyError(f"Joint not found in URDF: {joint_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play back a DMP-generated left-arm trajectory on the standalone arm URDF in PyBullet."
    )
    parser.add_argument("--subject", type=int, default=1, help="Subject number")
    parser.add_argument("--motion", type=str, default="lift", help="Motion name (e.g. reach, lift)")
    parser.add_argument("--trial", type=int, default=6, help="Trial number")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_project_root / "data" / "raw",
        help="Root directory for processed data (subject/motion/trial underneath).",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    args = parser.parse_args()

    # 1) Get the trial directory
    trial_dir = (
        args.data_dir
        / f"subject_{args.subject:02d}"
        / args.motion
        / f"trial_{args.trial:03d}"
    )
    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")
    # 2) Load the DMP trajectory
    q_traj, dt = _load_dmp_trajectory(trial_dir)  # (T, 4), radians
    #q_traj = clamp_dmp_vector(q_traj) # Clamp to joint limits
    print(q_traj)
    T, _n = q_traj.shape
    traj_duration_s = float((T - 1) * dt)
    print(f"Trajectory duration (simulated): {traj_duration_s:.3f} s  (T={T}, dt={dt:.6f})")

    # 2. Connect PyBullet and load standalone arm URDF
    p.connect(p.GUI)
    p.setGravity(0, 0, 0)  # kinematic playback

    p.setAdditionalSearchPath(str(_sim_dir))
    urdf_rel = "arm/left_arm.urdf"
    urdf_path = _sim_dir / urdf_rel
    if not urdf_path.exists():
        p.disconnect()
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Orient the standalone arm so the simulation axes match `convention.md`:
    #   +X = person's right
    #   +Y = up
    #   +Z = forward (out of chest)
    #
    # PyBullet's world uses +Z as up, so we rotate the arm base such that the
    # arm model's +Y aligns with world +Z. With this base orientation, the
    # "arm down" neutral pose corresponds to the upper-arm pointing along -Y
    # in the convention frame.
    base_orn = tuple(float(v) for v in p.getQuaternionFromEuler([math.pi/2.0, 0.0, math.pi/2.0]))
    #base_orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
    robot = p.loadURDF(
        urdf_rel,
        basePosition=[0, 0, 0],
        baseOrientation=base_orn,
        useFixedBase=True,
    )

    print("Loaded arm joints:")
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        print(j, info[1].decode("utf-8"))

    # 3. Map DMP joints to arm joints
    #
    # DMP order (4 DOFs):
    #   0: elbow_flexion
    #   1: shoulder_flexion
    #   2: shoulder_abduction
    #   3: shoulder_internal_rotation
    #
    # Arm URDF joints (left arm, approximate anatomical mapping):
    # NOTE: For this URDF, rotx/roty appear swapped visually vs our intended anatomy,
    # so we map flexion -> roty and internal rotation -> rotx.
    #   jLeftShoulder_rotz  (about Z in arm base frame)  <-- use shoulder_abduction
    #   jLeftShoulder_roty  (about Y)                   <-- use shoulder_flexion
    #   jLeftShoulder_rotx  (about X)                   <-- use shoulder_internal_rotation
    #   jLeftElbow_roty     (about Y)                   <-- use elbow_flexion
    #
    # Additional joints (not driven here, but available):
    #   jLeftElbow_rotz     (forearm rotation)
    #   jLeftWrist_rotx, jLeftWrist_rotz, finger joints ...
    sh_rotz = joint_index(robot, "jLeftShoulder_rotz")
    sh_rotx = joint_index(robot, "jLeftShoulder_rotx")
    sh_roty = joint_index(robot, "jLeftShoulder_roty")
    elbow_roty = joint_index(robot, "jLeftElbow_roty")

    # Disable motors so resetJointState fully controls pose
    num_joints = p.getNumJoints(robot)
    for j in range(num_joints):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, force=0.0)

    print(f"Starting DMP playback on standalone arm from {trial_dir}")
    #abd_offset = math.radians(float(args.abd_offset_deg))
    #abd_sign = float(args.abd_sign)

    try:
        while True:
            for t_idx in range(T):
                # If the GUI window is closed, the physics server disconnects.
                # Exit cleanly instead of raising "Not connected to physics server".
                if not p.isConnected():
                    return
                q_t = q_traj[t_idx]

                elbow = float(q_t[0])
                sh_flex = float(q_t[1])
                sh_abd = -float(q_t[2])
                sh_int = float(q_t[3])

                # Simple mapping as described above.
                #sh_abd_mapped = abd_sign * sh_abd + abd_offset
                p.resetJointState(robot, sh_rotz, sh_abd)
                p.resetJointState(robot, sh_roty, sh_flex)
                p.resetJointState(robot, sh_rotx, sh_int)
                p.resetJointState(robot, elbow_roty, elbow)

                p.stepSimulation()
                time.sleep(dt)

            if not args.loop:
                break
    finally:
        print("Playback finished. Close the PyBullet window to exit.")
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()

