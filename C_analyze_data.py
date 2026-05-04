import argparse
import os
from pathlib import Path
import numpy as np
import json
   
from kinematics.simple_kinematics import get_angles
from kinematics.clean_angles import _lowpass_angles
from mapping.retarget import retarget as _retarget_angles
from mapping.retarget import retarget_global as _retarget_angles_global
from mapping.retarget import retarget_threshold as _retarget_angles_threshold
from dmp.dmp import fit as _fit_dmp
from dmp.dmp import rollout_simple as _rollout_dmp
from dmp.dmp import rollout_simple_with_coupling as _rollout_dmp_with_coupling
from dmp.dmp import DMPModel
from vis.plotting import plot_angles_single, plot_dmp_single
from vis.plotting import plot_3d_trajectory as _plot_3d_trajectory
from mapping.retarget import JOINT_LIMITS_DEG

def _load_meta(trial_dir: Path) -> dict:
    with open(os.path.join(trial_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    return meta

def _load_raw_seq_t(trial_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    seq = np.load(os.path.join(trial_dir, "left_arm_seq_camera.npy"))
    t = np.load(os.path.join(trial_dir, "left_arm_t.npy"))
    return seq, t

def _interpolate_nan(angles: np.ndarray) -> np.ndarray:
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

def _clip_angles(angles: np.ndarray) -> np.ndarray:
    """
    Clip the angles to the robot range.
    """
    return np.clip(angles, JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])

def _save_angles(angles: np.ndarray, trial_dir: Path, t: np.ndarray, dt: float) -> None:
    np.savez(
        trial_dir / "angles.npz",
        angles=angles,
        t=t,
        dt=dt
    )

def _save_dmp_model(dmp_model: DMPModel, trial_dir: Path) -> None:
    np.savez(
        trial_dir / "dmp_model.npz",
        weights=dmp_model.weights,
        centers=dmp_model.centers,
        widths=dmp_model.widths,
        alpha_canonical=dmp_model.alpha_canonical,
        alpha_transformation=dmp_model.alpha_transformation,
        beta_transformation=dmp_model.beta_transformation,
        tau=dmp_model.tau
    )

def _save_dmp_rollout(q_gen: np.ndarray, trial_dir: Path, t: np.ndarray, dt: float) -> None:
    np.savez(
        trial_dir / "dmp_rollout.npz",
        q_gen=q_gen,
        t=t,
        dt=dt,
        q0=q_gen[0],
        qT=q_gen[-1],
    )

def _plot_angles(angles: np.ndarray, trial_dir: Path, t: np.ndarray, meta: dict) -> None:
    plot_angles_single(
        elbow_rad=np.deg2rad(angles[:, 0]), # elbow flexion
        shoulder_rad=np.deg2rad(angles[:, 1:4]), # shoulder flexion, shoulder abduction, shoulder lateral medial rotation
        t=t,
        meta=meta,
        title_suffix="angles",
        units="deg",
        out_path=trial_dir / "angles.png"
    )

def _plot_dmp(angles: np.ndarray, q_gen: np.ndarray, trial_dir: Path, meta: dict, out_path_name: str) -> None:
    plot_dmp_single(
        q_demo=angles,
        q_gen=q_gen,
        meta=meta,
        title_suffix="dmp",
        out_path=trial_dir / f"{out_path_name}.png"
    )

def main():

    # 1) Parse session setups
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True, help="Subject ID")
    ap.add_argument("--motion", type=str, required=True, help="Motion type")
    ap.add_argument("--trial", type=int, required=True, help="Trial number")
    args = ap.parse_args()


    # 2) Get the trial directory for input and directory for output
    session_dir = os.path.join(os.path.dirname(__file__), "data", "test", f"subject_{args.subject:02d}", args.motion)
    trial_dir = Path(session_dir) / f"trial_{(args.trial):03d}"
    
    output_dir = Path("data", "test", f"subject_{args.subject:02d}", args.motion, f"trial_{(args.trial):03d}")
    os.makedirs(output_dir, exist_ok=True)

    # 3) Load the data
    seq, t = _load_raw_seq_t(trial_dir)
    meta = _load_meta(trial_dir)
    # 4) Convert the sequence to angles and interpolate NaN values
    angles = _interpolate_nan(get_angles(seq))
    # Maybe add low_pass filter here
    angles = _lowpass_angles(angles, fps=25.0, cutoff_hz=5.0, order=2)


    # 5) Retarget the angles to the robot range
    #angles = _retarget_angles(angles) # Trajectory-specific limits
    #angles = _retarget_angles_global(angles)
    #angles = _retarget_angles_threshold(angles)

    
    # 6) Fit the DMP
    dt = 1.0 / (angles.shape[0] - 1)
    dmp_model = _fit_dmp([angles], tau=1.0, dt=dt, n_basis_functions=100, alpha_canonical=4.0, alpha_transformation=25.0, beta_transformation=6.25)
    q_gen = _rollout_dmp(dmp_model, angles[0], angles[-1], tau=1.0, dt=dt)
    q_gen_2 = _rollout_dmp_with_coupling(dmp_model, angles[0], angles[-1], tau=1.0, dt=dt)
    q_gen = _clip_angles(q_gen)
    q_gen_2 = _clip_angles(q_gen_2)

    # 7) Save the angles and DMP trajectory
    _save_angles(angles, output_dir, t, dt)
    _save_dmp_model(dmp_model, output_dir)
    _save_dmp_rollout(q_gen, output_dir, t, dt)
    _save_dmp_rollout(q_gen_2, output_dir, t, dt)
    
    # 8) Plot the angles and DMP trajectory
    _plot_3d_trajectory(seq, t, meta, output_dir / "sequence.png")
    _plot_angles(angles, output_dir, t, meta)
    _plot_dmp(angles, q_gen, output_dir, meta, "dmp")
    _plot_dmp(angles, q_gen_2, output_dir, meta, "dmp_with_coupling")
if __name__ == "__main__":
    main()