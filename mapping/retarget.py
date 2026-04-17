import numpy as np


JOINT_LIMITS_RAD: np.ndarray = np.array(
    [
        [0.0, 1.05],     # 0: Elbow flexion (0 to 60 degrees)
        [0.0, 1.39],     # 1: Shoulder flexion (0 to 80 degrees)
        [0.0, 0.69],     # 2: Shoulder abduction (0 to 40 degrees)
        [-0.69, 0.69],   # 3: Shoulder lateral/medial rotation (-40 to 40 degrees)
    ],
    dtype=float,
)

JOINT_LIMITS_DEG: np.ndarray = np.array(
    [
        [0.0, 60.0],      # 0: Elbow flexion
        [0.0, 80.0],      # 1: Shoulder flexion
        [0.0, 40.0],      # 2: Shoulder abduction
        [-40.0, 40.0],    # 3: Shoulder lateral/medial rotation
    ],
    dtype=float,
)

HUMAN_LIMITS_DEG: np.ndarray = np.array(
    [
        [0.0, 150.0],     # 0: Elbow flexion
        [-60.0, 170.0],   # 1: Shoulder flexion
        [0.0, 180.0],     # 2: Shoulder abduction
        [-90.0, 90.0],    # 3: Shoulder lateral/medial rotation
    ],
    dtype=float,
)


def retarget_global(q_demo: np.ndarray) -> np.ndarray:
    f"""
    Range scaling. Map the human range to the robot range with linear scaling.

    q_demo is the demonstrated trajectory in radians. (from human)
    q_gen is the generated trajectory in radians. (goes onto robot)

    q_robot = q_robot_min + (q_demo - q_demo_min) * (q_robot_max - q_robot_min) / (q_demo_max - q_demo_min)
    Args:
        q_demo: np.ndarray: the demonstrated trajectory
        q_gen: np.ndarray: the generated trajectory

    Returns:
        np.ndarray: the retargeted trajectory
    """
    # Canonical repo order for 4-DoF trajectories:
    #   [elbow_flexion, shoulder_flexion, shoulder_abduction, shoulder_lat/med_rotation]

    q_human_min = HUMAN_LIMITS_DEG[:, 0]
    q_human_max = HUMAN_LIMITS_DEG[:, 1]
    q_robot_min = JOINT_LIMITS_DEG[:, 0]
    q_robot_max = JOINT_LIMITS_DEG[:, 1]

    # Calibration / offset alignment


    # Linear scaling
    q_robot = q_robot_min + (q_demo - q_human_min) * (q_robot_max - q_robot_min) / (q_human_max - q_human_min)

    # Clamp to robot limits for safety
    q_robot = np.clip(q_robot, q_robot_min, q_robot_max)

    # Optional: smooth the trajectory (temporal)
    #q_robot = savgol_filter(q_robot, window_length=11, polyorder=3)

    return q_robot

def retarget(q_demo: np.ndarray) -> np.ndarray:
    """
    Retarget the trajectory.
    Uses trajectory-specific min and max values for the human and robot ranges.

    Args:
        q_demo: np.ndarray: the demonstrated trajectory

    Returns:
        np.ndarray: the retargeted trajectory
    """
    q_human_min = q_demo.min(axis=0)
    q_human_max = q_demo.max(axis=0)
    q_robot_min = JOINT_LIMITS_DEG[:, 0]
    q_robot_max = JOINT_LIMITS_DEG[:, 1]

    q_robot = q_robot_min + (q_demo - q_human_min) * (q_robot_max - q_robot_min) / (q_human_max - q_human_min)
    q_robot = np.clip(q_robot, q_robot_min, q_robot_max)

    # Optional: smooth the trajectory (temporal)
    #q_robot = savgol_filter(q_robot, window_length=11, polyorder=3)

    return q_robot

def retarget_threshold(q_demo: np.ndarray) -> np.ndarray:
    """
    Retarget the trajectory if the trajectory for one joint is out of bounds.
    Args:
        q_demo: np.ndarray: the demonstrated trajectory

    Returns:
        np.ndarray: the retargeted trajectory
    """

    q_robot_min = JOINT_LIMITS_DEG[:, 0]
    q_robot_max = JOINT_LIMITS_DEG[:, 1]
    q_traj_min = q_demo.min(axis=0)
    q_traj_max = q_demo.max(axis=0)
    q_human_min = HUMAN_LIMITS_DEG[:, 0]
    q_human_max = HUMAN_LIMITS_DEG[:, 1]

    q_robot = q_demo.copy()
    for j in range(q_demo.shape[1]):
        if q_traj_max[j] - q_traj_min[j] > q_robot_max[j] - q_robot_min[j]:
            print(f"Retargeting globally with human limits for joint {j}")
            # Retarget globally with human limits
            q_robot[:, j] = q_robot_min[j] + (q_demo[:, j] - q_human_min[j]) * (q_robot_max[j] - q_robot_min[j]) / (q_human_max[j] - q_human_min[j])
        else:
            # Retarget locally with trajectory limits
            q_robot[:, j] = q_robot_min[j] + (q_demo[:, j] - q_traj_min[j]) * (q_robot_max[j] - q_robot_min[j]) / (q_traj_max[j] - q_traj_min[j])
        q_robot[:, j] = np.clip(q_robot[:, j], q_robot_min[j], q_robot_max[j])
    return q_demo