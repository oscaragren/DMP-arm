"""
End-to-end pipeline for the second experiment with timing analysis.

1. Get frames
2. Estimate pose
3. Preprocess
4. DMP fitting
5. DMP rollout
6. Robot command generation
"""
from time import perf_counter_ns

import numpy as np
from pathlib import Path
from dmp.dmp import DMPModel



"""
Here I will have a sequence of frame that I read in
"""

def load_trajectory_demo(trial_dir: Path) -> np.ndarray:
    """Load the trajectory demo from the trial directory."""
    q_demo = np.load(trial_dir / "angles.npz")
    return q_demo

def preprocess_and_map_to_joint_space(q_demo: np.ndarray) -> np.ndarray:
    """Preprocess and map to joint space (angles)."""
    angles = np.zeros((q_demo.shape[0], 4))
    return angles

def fit_dmp(angles: np.ndarray) -> DMPModel:
    """Fit the DMP model."""
    dmp_model = fit(angles)
    return dmp_model

def rollout_dmp(dmp_model: DMPModel, angles: np.ndarray) -> np.ndarray:
    """Rollout the DMP model."""
    q_gen = rollout_dmp(dmp_model, angles[0], angles[-1])
    return q_gen

def generate_robot_commands(q_gen: np.ndarray) -> np.ndarray:
    """Generate the robot commands."""
    pass

def send_robot_commands(robot_commands: np.ndarray) -> bool:
    """Send the robot commands."""
    pass

t0 = perf_counter_ns()

# 1. Load the trajectory demo
t_load_0 = perf_counter_ns()
q_demo = load_trajectory_demo(Path("Some Path here")) # This is a sequence of x, y, z coordinates for each keypoint
t_load_1 = perf_counter_ns()

# 2 Preprocess and map to joint space (angles)
t_pre_0 = perf_counter_ns()
angles = preprocess_and_map_to_joint_space(q_demo)
t_pre_1 = perf_counter_ns()

# 3. DMP fitting
t_dmp_0 = perf_counter_ns()
dmp_model = fit_dmp(angles) 
t_dmp_1 = perf_counter_ns()

# 4. DMP rollout
t_rollout_0 = perf_counter_ns()
q_gen = rollout_dmp(dmp_model, angles[0], angles[-1])
t_rollout_1 = perf_counter_ns()

# 5. Robot command generation (angles to messsages)
t_gen_0 = perf_counter_ns()
robot_commands = generate_robot_commands(q_gen) # Sequence of robot commands
t_gen_1 = perf_counter_ns()

# 6. Send robot commands
#sent = send_robot_commands(robot_commands) # Send the robot commands over CAN