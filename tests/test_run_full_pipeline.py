

import os
import sys
import json
from pathlib import Path

import numpy as np

# When you run `python3 tests/test_run_full_pipeline.py`, Python sets sys.path[0]
# to the `tests/` folder. Add the repository root so imports like `vis.*` work.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#from kinematics.left_arm_angles import get_angles
from vis.plotting import plot_3d_trajectory, plot_angles_single, plot_dmp_single
from kinematics.simple_kinematics import get_angles
from dmp.dmp import fit, rollout_simple
from kinematics.clean_angles import _lowpass_angles
from mapping.retarget import retarget

def _load_raw_seq_t(trial_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    return np.load(trial_dir / "left_arm_seq_camera.npy"), np.load(trial_dir / "left_arm_t.npy")

def _load_meta(trial_dir: Path) -> dict:
    return json.load(open(trial_dir / "meta.json"))

def interpolate_nan(angles: np.ndarray) -> np.ndarray:
    """
    Interpolate NaN values over time for each joint column.

    Expected input shape: (T, D) where each column is one joint/DoF.
    """
    a = np.asarray(angles, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"Expected angles shape (T, D), got {a.shape}")

    T, D = a.shape
    x = np.arange(T, dtype=np.float64)
    out = a.copy()

    for j in range(D):
        col = a[:, j]
        valid = np.isfinite(col)
        n_valid = int(np.sum(valid))

        if n_valid == 0:
            # No information to interpolate from; keep NaNs.
            continue
        if n_valid == 1:
            # Only one sample; fill the whole column with that constant.
            out[:, j] = col[valid][0]
            continue

        out[:, j] = np.interp(x, x[valid], col[valid])

    return out

def main():
    
    # Load the data
    trial_dir = Path("test_data/processed/subject_03/random/trial_006")
    seq, t = _load_raw_seq_t(trial_dir)
    meta = _load_meta(trial_dir)

    # Plot the 3D trajectory
    plot_3d_trajectory(seq, t, meta, trial_dir / "keypoints_raw_trajectory.png")

    # Convert to angles
    angles = get_angles(seq) # angles: [elbow flexion, shoulder flexion, shoulder abduction, shoulder lateral medial rotation]
    angles = interpolate_nan(angles)
    clean_angles = _lowpass_angles(angles, fps=25.0, cutoff_hz=5.0, order=2)
    #print(f"Clean angles: {clean_angles}")
    #print(f"Elbow flexion: {angles[:, 0]}")
    #print(f"Shoulder flexion: {angles[:, 1]}")
    #print(f"Shoulder abduction: {angles[:, 2]}")
    #print(f"Shoulder lateral medial rotation: {angles[:, 3]}")

    # Preserve degree values before converting to radians (useful for plotting + exports).
    angles_deg = angles
    clean_angles_deg = clean_angles

    # Retarget the angles (still in degrees) to the robot range.
    angles_deg_retarget = retarget(angles_deg)
    clean_angles_deg_retarget = retarget(clean_angles_deg)

    # Convert to radians (DMP expects radians)
    angles = np.deg2rad(angles_deg)
    clean_angles = np.deg2rad(clean_angles_deg)
    angles_retarget = np.deg2rad(angles_deg_retarget)
    clean_angles_retarget = np.deg2rad(clean_angles_deg_retarget)

    # Export angles for simulation/analysis
    np.savez(
        trial_dir / "angles_raw.npz",
        angles_rad=angles,
        angles_deg=angles_deg,
        elbow_rad=angles[:, 0],
        shoulder_rad=angles[:, 1:4],
        elbow_deg=angles_deg[:, 0],
        shoulder_deg=angles_deg[:, 1:4],
        t=t,
    )
    np.savez(
        trial_dir / "angles_clean.npz",
        angles_rad=clean_angles,
        angles_deg=clean_angles_deg,
        elbow_rad=clean_angles[:, 0],
        shoulder_rad=clean_angles[:, 1:4],
        elbow_deg=clean_angles_deg[:, 0],
        shoulder_deg=clean_angles_deg[:, 1:4],
        t=t,
    )

    # Plot the angles
    plot_angles_single(
        elbow_rad=angles[:, 0], # elbow flexion
        shoulder_rad=angles[:, 1:4], # shoulder flexion, shoulder abduction, shoulder lateral medial rotation
        t=t,
        meta=meta,
        title_suffix="raw",
        units="deg",
        out_path=trial_dir / "angles.png"
    )
    plot_angles_single(
        elbow_rad=clean_angles[:, 0], # elbow flexion
        shoulder_rad=clean_angles[:, 1:4], # shoulder flexion, shoulder abduction, shoulder lateral medial rotation
        t=t,
        meta=meta,
        title_suffix="clean",
        units="deg",
        out_path=trial_dir / "clean_angles.png"
    )
    
    # Fit DMP (no retargeting)
    dt = 1.0 / (angles.shape[0] - 1)
    model_raw = fit([angles], tau=1.0, dt=dt, n_basis_functions=100, alpha_canonical=4.0, alpha_transformation=25.0, beta_transformation=6.25)
    q_gen_raw = rollout_simple(model_raw, angles[0], angles[-1], tau=1.0, dt=dt)
    
    dt_clean = 1.0 / (clean_angles.shape[0] - 1)
    model_clean = fit([clean_angles], tau=1.0, dt=dt_clean, n_basis_functions=100, alpha_canonical=4.0, alpha_transformation=25.0, beta_transformation=6.25)
    q_gen_clean = rollout_simple(model_clean, clean_angles[0], clean_angles[-1], tau=1.0, dt=dt_clean)

    # Fit DMP (with retargeting)
    dt_retarget = 1.0 / (angles_retarget.shape[0] - 1)
    model_raw_retarget = fit(
        [angles_retarget],
        tau=1.0,
        dt=dt_retarget,
        n_basis_functions=100,
        alpha_canonical=4.0,
        alpha_transformation=25.0,
        beta_transformation=6.25,
    )
    q_gen_raw_retarget = rollout_simple(
        model_raw_retarget,
        angles_retarget[0],
        angles_retarget[-1],
        tau=1.0,
        dt=dt_retarget,
    )

    dt_clean_retarget = 1.0 / (clean_angles_retarget.shape[0] - 1)
    model_clean_retarget = fit(
        [clean_angles_retarget],
        tau=1.0,
        dt=dt_clean_retarget,
        n_basis_functions=100,
        alpha_canonical=4.0,
        alpha_transformation=25.0,
        beta_transformation=6.25,
    )
    q_gen_clean_retarget = rollout_simple(
        model_clean_retarget,
        clean_angles_retarget[0],
        clean_angles_retarget[-1],
        tau=1.0,
        dt=dt_clean_retarget,
    )

    # Export rollouts for simulation/analysis
    np.savez(
        trial_dir / "dmp_rollout_raw.npz",
        q_demo=angles,
        q_gen=np.rad2deg(q_gen_raw),
        q_gen_rad=q_gen_raw,
        t=t,
        dt=dt,
        q0=angles[0],
        qT=angles[-1],
    )
    np.savez(
        trial_dir / "dmp_rollout_clean.npz",
        q_demo=clean_angles,
        q_gen=np.rad2deg(q_gen_clean),
        q_gen_rad=q_gen_clean,
        t=t,
        dt=dt_clean,
        q0=clean_angles[0],
        qT=clean_angles[-1],
    )

    # Plot the generated DMP trajectory
    plot_dmp_single(
        q_demo=angles,
        q_gen=np.rad2deg(q_gen_raw),
        meta=meta,
        title_suffix="raw",
        out_path=trial_dir / "dmp_trajectory_raw.png"
    )

    plot_dmp_single(
        q_demo=clean_angles,
        q_gen=np.rad2deg(q_gen_clean),
        meta=meta,
        title_suffix="clean",
        out_path=trial_dir / "dmp_trajectory_clean.png"
    )

    plot_dmp_single(
        q_demo=angles_retarget,
        q_gen=np.rad2deg(q_gen_raw_retarget),
        meta=meta,
        title_suffix="raw_retarget",
        out_path=trial_dir / "dmp_trajectory_raw_retarget.png",
    )

    plot_dmp_single(
        q_demo=clean_angles_retarget,
        q_gen=np.rad2deg(q_gen_clean_retarget),
        meta=meta,
        title_suffix="clean_retarget",
        out_path=trial_dir / "dmp_trajectory_clean_retarget.png",
    )


if __name__ == "__main__":
    main()