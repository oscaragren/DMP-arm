"""
Play back a DMP-learned joint trajectory on the Franka Panda arm in PyBullet.

We:
- load a 4-DOF demo (elbow + 3 shoulder DOFs) from angles.npz
- fit a DMP and rollout a new trajectory in joint space (still 4 DOFs)
- map these 4 angles to the first four Panda joints and visualize the motion
"""
from pathlib import Path
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

from dmp.dmp import fit, rollout_simple
from vis.plot_dmp_trajectory import load_angles_demo


def get_project_root() -> Path:
    """Return project root (parent of sim/)."""
    sim_dir = Path(__file__).resolve().parent
    return sim_dir.parent


def load_dmp_trajectory() -> tuple[np.ndarray, float]:
    """Fit a DMP on a demo from angles.npz and rollout a generated trajectory.

    Returns:
        q_rad: (T, 4) numpy array, joint angles in radians
        dt:   timestep used for the rollout (seconds, normalized time)
    """
    project_root = get_project_root()
    # Default trial directory, matching other utilities in this repo
    trial_dir = project_root / "test_data" / "processed" / "subject_01" / "reach" / "trial_004"

    if not trial_dir.exists():
        raise FileNotFoundError(
            f"Trial directory not found: {trial_dir}\n"
            "Adjust the path in sim/test_sim.py or generate test_data/processed first."
        )

    # Demo is in degrees (elbow + 3 shoulder angles), shape (T, 4)
    q_demo_deg = load_angles_demo(trial_dir)
    T, n_joints = q_demo_deg.shape
    if n_joints != 4:
        raise ValueError(f"Expected 4-DOF demo, got shape {q_demo_deg.shape}")

    # Normalized duration tau=1.0, match what vis/plot_dmp_trajectory.py uses
    tau = 1.0
    dt = tau / (T - 1)

    model = fit(
        [q_demo_deg],
        tau=tau,
        dt=dt,
        n_basis_functions=15,
        alpha_canonical=4.0,
        alpha_transformation=25.0,
        beta_transformation=6.25,
    )

    q_gen_deg = rollout_simple(model, q_demo_deg[0], q_demo_deg[-1], tau=tau, dt=dt)
    q_gen_rad = np.deg2rad(q_gen_deg)
    return q_gen_rad, dt


def get_joint_indices_by_name(body_uid: int, names: list[str]) -> list[int]:
    """Utility to resolve PyBullet joint indices from joint names."""
    name_to_index: dict[str, int] = {}
    num_joints = p.getNumJoints(body_uid)
    for j in range(num_joints):
        info = p.getJointInfo(body_uid, j)
        jname = info[1].decode("utf-8") if isinstance(info[1], bytes) else info[1]
        name_to_index[jname] = j

    indices: list[int] = []
    for name in names:
        if name not in name_to_index:
            raise KeyError(f"Joint {name} not found in URDF. Available: {sorted(name_to_index.keys())}")
        indices.append(name_to_index[name])
    return indices


def main() -> None:
    # Ensure project root is on sys.path so imports work if run from elsewhere
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 1. Load DMP trajectory once before starting simulation
    q_traj, dt = load_dmp_trajectory()  # (T, 4), radians
    T, n_joints = q_traj.shape

    # 2. Start PyBullet GUI and basic world
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)  # kinematic playback; no dynamics needed

    plane_id = p.loadURDF("plane.urdf")
    _ = plane_id  # silence unused variable linter

    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # 3. Map our 4-DOF trajectory to the first four Panda joints
    panda_joint_names = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4"]
    controlled_indices = get_joint_indices_by_name(robot, panda_joint_names)
    if len(controlled_indices) != n_joints:
        raise RuntimeError(
            f"Trajectory has {n_joints} joints but mapped to {len(controlled_indices)} Panda joints."
        )

    # Disable default motor forces so resetJointState fully controls pose
    num_joints = p.getNumJoints(robot)
    for j in range(num_joints):
        p.setJointMotorControl2(
            robot,
            j,
            p.POSITION_CONTROL,
            force=0.0,
        )

    # 4. Playback loop: repeatedly run the generated trajectory
    print("Starting DMP playback on Franka Panda. Close the GUI window to exit.")
    try:
        while True:
            for t_idx in range(T):
                q_t = q_traj[t_idx]
                for j, joint_index in enumerate(controlled_indices):
                    p.resetJointState(robot, joint_index, float(q_t[j]))
                p.stepSimulation()
                time.sleep(dt)
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()