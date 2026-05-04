"""
End-to-end pipeline for the second experiment with timing analysis.

0. Load DMP model and trajectory in angles from file
1. DMP rollout
2. Robot command generation
3. CAN transmission

DEFINITION OF CONSTRAINTS:

Control frequency: 50 Hz
"""
from time import perf_counter_ns

import numpy as np
from dmp.dmp import DMPModel
from pathlib import Path

def load_dmp_model_and_trajectory(path: Path) -> tuple[DMPModel, np.ndarray]:
    """Load the DMP model from the file."""
    dmp_model = DMPModel.load(path)
    angles = np.load(path / "angles.npz")
    return dmp_model, angles

def rollout_dmp(dmp_model: DMPModel, angles: np.ndarray) -> np.ndarray:
    """Rollout the DMP model."""
    return rollout_dmp(dmp_model, angles[0], angles[-1])

def generate_robot_commands(q_gen: np.ndarray) -> np.ndarray:
    """Generate the robot commands."""
    return generate_robot_commands(q_gen) 
    
def send_robot_commands(robot_commands: np.ndarray) -> bool:
    """Send the robot commands."""
    return send_robot_commands(robot_commands)

# 0. Load DMP model from file
t_load_0 = perf_counter_ns()
dmp_model, angles = load_dmp_model_and_trajectory(Path("Some Path here"))  
t_load_1 = perf_counter_ns()

# 1. DMP rollout
t_rollout_0 = perf_counter_ns()
q_gen = rollout_dmp(dmp_model, angles[0], angles[-1])
t_rollout_1 = perf_counter_ns()

# 2. Robot command generation
t_gen_0 = perf_counter_ns()
robot_commands = generate_robot_commands(q_gen)
t_gen_1 = perf_counter_ns()

# 3. CAN transmission
t_send_0 = perf_counter_ns()
sent = send_robot_commands(robot_commands)
t_send_1 = perf_counter_ns()

