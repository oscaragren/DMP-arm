"""
Play back a DMP‑generated left‑arm trajectory on the standalone arm model in PyBullet.

This is analogous to sim/inmoov_sim.py, but loads the URDF in sim/arm/left_arm.urdf
instead of the full InMoov model.

Usage (from project root):

    python sim/limb_sim.py --trial-dir path/to/trial

where the trial directory contains `angles.npz` produced by your pipeline
(`elbow_deg` + `shoulder_deg`), with the convention:

    [elbow_flexion, shoulder_flexion, shoulder_abduction, shoulder_internal_rotation] (deg)
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

from dmp.dmp import fit, rollout_simple
from kinematics.joint_dynamics import smooth_angles_deg
from vis.plot_dmp_trajectory import load_angles_demo
from sim.joint_limits import clamp_dmp_vector


def load_dmp_trajectory(trial_dir: Path) -> tuple[np.ndarray, float]:
    """
    Fit a DMP on a demo from angles.npz in `trial_dir` and rollout a trajectory.

    Returns:
        q_rad: (T, 4) numpy array, joint angles in radians
        dt:   timestep used for the rollout (seconds, normalized time)
    """
    # angles.npz convention:
    #   0: elbow_flexion
    #   1: shoulder_flexion
    #   2: shoulder_abduction
    #   3: shoulder_internal_rotation
    # load_angles_demo now prefers radians and falls back to degrees.
    q_demo = load_angles_demo(trial_dir)  # (T, 4), radians
    # Smooth in degree domain and convert back to radians for stability.
    q_demo = np.deg2rad(smooth_angles_deg(np.degrees(q_demo)))

    T, n_joints = q_demo.shape
    if n_joints != 4:
        raise ValueError(f"Expected 4-DOF demo, got shape {q_demo.shape}")

    tau = 1.0
    dt = tau / (T - 1)

    model = fit(
        [q_demo],
        tau=tau,
        dt=dt,
        n_basis_functions=15,
        alpha_canonical=4.0,
        alpha_transformation=25.0,
        beta_transformation=6.25,
    )

    q_gen = rollout_simple(model, q_demo[0], q_demo[-1], tau=tau, dt=dt)
    # Clamp to robot joint limits in radians before playback.
    q_gen = clamp_dmp_vector(q_gen)
    return q_gen, dt


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
        description="Play back a DMP‑generated left‑arm trajectory on the standalone arm URDF in PyBullet."
    )
    parser.add_argument(
        "--trial-dir",
        type=Path,
        default=_project_root
        / "test_data"
        / "processed"
        / "subject_01"
        / "lift"
        / "trial_003",
        help="Trial directory containing angles.npz (default: subject_01/lift/trial_003).",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument(
        "--abd-offset-deg",
        type=float,
        default=0.0,
        help="Constant offset applied to shoulder abduction before mapping to shoulder yaw joint (deg).",
    )
    parser.add_argument(
        "--abd-sign",
        type=float,
        default=1.0,
        help="Sign applied to shoulder abduction before offset (use -1 if direction is flipped).",
    )
    args = parser.parse_args()

    trial_dir: Path = args.trial_dir
    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")

    # 1. Load DMP trajectory (elbow + 3 shoulder DOFs)
    q_traj, dt = load_dmp_trajectory(trial_dir)  # (T, 4), radians
    T, n_joints = q_traj.shape

    # 2. Connect PyBullet and load standalone arm URDF
    p.connect(p.GUI)
    p.setGravity(0, 0, 0)  # kinematic playback

    sim_dir = _sim_dir
    p.setAdditionalSearchPath(str(sim_dir))
    urdf_rel = "arm/left_arm.urdf"
    urdf_path = sim_dir / urdf_rel
    if not urdf_path.exists():
        p.disconnect()
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Orient the standalone arm so that, at zero joint angles, the upper arm
    # hangs down along the human body (green Y‑axis pointing downward in the
    # PyBullet world frame).
    base_orn = p.getQuaternionFromEuler([-math.pi / 2.0, 0.0, 0.0])
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
    #   jLeftShoulder_rotz  (about Z in arm base frame)  <-- use shoulder_abduction
    #   jLeftShoulder_rotx  (about X)                   <-- use shoulder_flexion
    #   jLeftShoulder_roty  (about Y)                   <-- use shoulder_internal_rotation
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
    abd_offset = math.radians(float(args.abd_offset_deg))
    abd_sign = float(args.abd_sign)

    try:
        while True:
            for t_idx in range(T):
                q_t = q_traj[t_idx]

                elbow = float(q_t[0])
                sh_flex = float(q_t[1])
                sh_abd = float(q_t[2])
                sh_int = float(q_t[3])

                # Simple mapping as described above.
                sh_abd_mapped = abd_sign * sh_abd + abd_offset
                p.resetJointState(robot, sh_rotz, sh_abd_mapped)
                p.resetJointState(robot, sh_rotx, sh_flex)
                p.resetJointState(robot, sh_roty, sh_int)
                p.resetJointState(robot, elbow_roty, elbow)

                p.stepSimulation()
                time.sleep(dt)

            if not args.loop:
                break
    finally:
        print("Playback finished. Close the PyBullet window to exit.")
        p.disconnect()


if __name__ == "__main__":
    main()

