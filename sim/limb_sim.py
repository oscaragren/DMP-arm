"""
Play back a DMP‑generated left‑arm trajectory on the standalone arm model in PyBullet.

This is analogous to sim/inmoov_sim.py, but loads the URDF in sim/arm/left_arm.urdf
instead of the full InMoov model.

Usage (from project root):

    python sim/limb_sim.py --subject 1 --motion lift --trial 6

or:

    python sim/limb_sim.py --path test_data/processed/subject_01/lift/trial_006

The trial directory should contain either:
- a precomputed rollout `dmp_rollout_{clean|raw}.npz` (preferred), or
- `angles*.npz` produced by your pipeline (fallback; we re-fit a DMP at runtime),
with the convention:

    [elbow_flexion, shoulder_flexion, shoulder_abduction, shoulder_internal_rotation] (deg)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pybullet as p

# Ensure project root is on sys.path for imports
_sim_dir = Path(__file__).resolve().parent
_project_root = _sim_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dmp.trajectory_io import load_dmp_trajectory, resolve_saved_dmp_rollout_path
from sim.joint_limits import clamp_dmp_vector


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
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to trial dir (overrides subject/motion/trial).",
    )
    parser.add_argument("--subject", type=int, default=1, help="Subject number")
    parser.add_argument("--motion", type=str, default="lift", help="Motion name (e.g. reach, lift)")
    parser.add_argument("--trial", type=int, default=6, help="Trial number")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_project_root / "test_data" / "processed",
        help="Root directory for processed data (subject/motion/trial underneath).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["raw", "clean"],
        default="clean",
        help="Which DMP rollout to use (default: clean).",
    )
    parser.add_argument(
        "--n-basis",
        type=int,
        default=None,
        help="If set, load the sweep rollout for this basis count (e.g. 10, 30, 60).",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=None,
        help="For clean sweep rollouts: filter order (required if --source clean and --n-basis is set).",
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

    if args.path is not None:
        trial_dir: Path = args.path
    else:
        trial_dir = (
            args.data_dir
            / f"subject_{args.subject:02d}"
            / args.motion
            / f"trial_{args.trial:03d}"
        )
    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")

    rollout_path = resolve_saved_dmp_rollout_path(
        trial_dir,
        rollout_source=args.source,
        basis_functions=args.n_basis,
        filter_order=args.filter_order,
    )
    if rollout_path is not None:
        print(f"Loading saved DMP rollout: {rollout_path}")
    else:
        print("No saved DMP rollout found; will fit+rollout from angles*.npz at runtime.")

    # 1. Load DMP trajectory (elbow + 3 shoulder DOFs)
    q_traj, dt = load_dmp_trajectory(
        trial_dir,
        rollout_source=args.source,
        basis_functions=args.n_basis,
        filter_order=args.filter_order,
    )  # (T, 4), radians
    q_traj = clamp_dmp_vector(q_traj) # Clamp to joint limits

    T, _n = q_traj.shape

    # 2. Connect PyBullet and load standalone arm URDF
    p.connect(p.GUI)
    p.setGravity(0, 0, 0)  # kinematic playback

    p.setAdditionalSearchPath(str(_sim_dir))
    urdf_rel = "arm/left_arm.urdf"
    urdf_path = _sim_dir / urdf_rel
    if not urdf_path.exists():
        p.disconnect()
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Orient the standalone arm so that, at zero joint angles, the upper arm
    # hangs down along the human body (green Y‑axis pointing downward in the
    # PyBullet world frame).
    base_orn = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
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
                p.resetJointState(robot, sh_roty, sh_flex)
                p.resetJointState(robot, sh_rotx, sh_int)
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

