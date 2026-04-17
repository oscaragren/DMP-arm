

import os
import sys
import json
import subprocess
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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

def plot_retarget_options_overlay(
    *,
    q_gen_demo_retarget: np.ndarray,
    q_gen_rollout_retarget: np.ndarray,
    meta: dict,
    out_path: Path,
    title_suffix: str,
) -> None:
    """
    Compare the two retarget strategies on the same axes (robot-space angles).

    Inputs are expected in radians, canonical order:
      [elbow_flex, shoulder_flex, shoulder_abd, shoulder_lat/med_rot]
    """
    joint_names = [
        "Elbow flexion",
        "Shoulder flexion",
        "Shoulder abduction",
        "Shoulder internal rotation",
    ]

    t_gen1 = np.linspace(0.0, 1.0, q_gen_demo_retarget.shape[0])
    t_gen2 = np.linspace(0.0, 1.0, q_gen_rollout_retarget.shape[0])

    gen_demo_deg = np.rad2deg(q_gen_demo_retarget)
    gen_rollout_deg = np.rad2deg(q_gen_rollout_retarget)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()
    for j in range(4):
        ax = axes[j]
        ax.plot(t_gen1, gen_demo_deg[:, j], color="#8e44ad", linewidth=1.2, linestyle="-", label="DMP (retarget demo)")
        ax.plot(t_gen2, gen_rollout_deg[:, j], color="#2980b9", linewidth=1.2, linestyle="--", label="DMP (retarget rollout)")
        ax.set_title(joint_names[j])
        ax.set_ylabel("Angle (deg)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    axes[-1].set_xlabel("Time (normalized)")
    subject = meta.get("subject", "?")
    motion = meta.get("motion", "?")
    trial = meta.get("trial", "?")
    fig.suptitle(
        f"Retarget strategy comparison — subject {subject}, {motion}, trial {trial} ({title_suffix})",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)

def main():
    
    # Load the data
    trial_dir = Path("data/raw/subject_10/test/trial_001")
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

    # Retarget the DEMO trajectories into robot range (degrees)
    # Canonical order: [elbow flexion, shoulder flexion, shoulder abduction, shoulder lat/med rotation]
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

    # Plot the retargeted demo angles (robot-space ranges)
    plot_angles_single(
        elbow_rad=angles_retarget[:, 0],
        shoulder_rad=angles_retarget[:, 1:4],
        t=t,
        meta=meta,
        title_suffix="raw_retarget_demo",
        units="deg",
        out_path=trial_dir / "angles_retarget_demo.png",
    )
    plot_angles_single(
        elbow_rad=clean_angles_retarget[:, 0],
        shoulder_rad=clean_angles_retarget[:, 1:4],
        t=t,
        meta=meta,
        title_suffix="clean_retarget_demo",
        units="deg",
        out_path=trial_dir / "clean_angles_retarget_demo.png",
    )
    
    # Fit DMP (no retargeting)
    dt = 1.0 / (angles.shape[0] - 1)
    model_raw = fit([angles], tau=1.0, dt=dt, n_basis_functions=100, alpha_canonical=4.0, alpha_transformation=25.0, beta_transformation=6.25)
    q_gen_raw = rollout_simple(model_raw, angles[0], angles[-1], tau=1.0, dt=dt)
    
    dt_clean = 1.0 / (clean_angles.shape[0] - 1)
    model_clean = fit([clean_angles], tau=1.0, dt=dt_clean, n_basis_functions=100, alpha_canonical=4.0, alpha_transformation=25.0, beta_transformation=6.25)
    q_gen_clean = rollout_simple(model_clean, clean_angles[0], clean_angles[-1], tau=1.0, dt=dt_clean)

    # Fit DMP (retarget DEMO first): learn directly in robot joint ranges
    dt_demo_retarget = 1.0 / (angles_retarget.shape[0] - 1)
    model_raw_demo_retarget = fit(
        [angles_retarget],
        tau=1.0,
        dt=dt_demo_retarget,
        n_basis_functions=100,
        alpha_canonical=4.0,
        alpha_transformation=25.0,
        beta_transformation=6.25,
    )
    q_gen_raw_demo_retarget = rollout_simple(
        model_raw_demo_retarget,
        angles_retarget[0],
        angles_retarget[-1],
        tau=1.0,
        dt=dt_demo_retarget,
    )

    dt_clean_demo_retarget = 1.0 / (clean_angles_retarget.shape[0] - 1)
    model_clean_demo_retarget = fit(
        [clean_angles_retarget],
        tau=1.0,
        dt=dt_clean_demo_retarget,
        n_basis_functions=100,
        alpha_canonical=4.0,
        alpha_transformation=25.0,
        beta_transformation=6.25,
    )
    q_gen_clean_demo_retarget = rollout_simple(
        model_clean_demo_retarget,
        clean_angles_retarget[0],
        clean_angles_retarget[-1],
        tau=1.0,
        dt=dt_clean_demo_retarget,
    )

    # Retarget AFTER DMP: take generated joint trajectories and map them to the robot range.
    # retarget() operates in degrees, so convert q_gen to degrees first.
    q_gen_raw_deg = np.rad2deg(q_gen_raw)
    q_gen_clean_deg = np.rad2deg(q_gen_clean)
    q_gen_raw_retarget = np.deg2rad(retarget(q_gen_raw_deg))
    q_gen_clean_retarget = np.deg2rad(retarget(q_gen_clean_deg))

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

    # Export rollouts where DEMO was retargeted before fitting
    np.savez(
        trial_dir / "dmp_rollout_raw_demo_retarget.npz",
        q_demo=angles_retarget,
        q_gen=np.rad2deg(q_gen_raw_demo_retarget),
        q_gen_rad=q_gen_raw_demo_retarget,
        t=t,
        dt=dt_demo_retarget,
        q0=angles_retarget[0],
        qT=angles_retarget[-1],
    )
    np.savez(
        trial_dir / "dmp_rollout_clean_demo_retarget.npz",
        q_demo=clean_angles_retarget,
        q_gen=np.rad2deg(q_gen_clean_demo_retarget),
        q_gen_rad=q_gen_clean_demo_retarget,
        t=t,
        dt=dt_clean_demo_retarget,
        q0=clean_angles_retarget[0],
        qT=clean_angles_retarget[-1],
    )

    np.savez(
        trial_dir / "dmp_rollout_raw_retarget.npz",
        q_demo=angles,
        q_gen=np.rad2deg(q_gen_raw_retarget),
        q_gen_rad=q_gen_raw_retarget,
        t=t,
        dt=dt,
        q0=angles[0],
        qT=angles[-1],
    )
    np.savez(
        trial_dir / "dmp_rollout_clean_retarget.npz",
        q_demo=clean_angles,
        q_gen=np.rad2deg(q_gen_clean_retarget),
        q_gen_rad=q_gen_clean_retarget,
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

    # Retarget DEMO (fit/rollout in retarget space)
    plot_dmp_single(
        q_demo=angles_retarget,
        q_gen=np.rad2deg(q_gen_raw_demo_retarget),
        meta=meta,
        title_suffix="raw_demo_retarget",
        out_path=trial_dir / "dmp_trajectory_raw_demo_retarget.png",
    )
    plot_dmp_single(
        q_demo=clean_angles_retarget,
        q_gen=np.rad2deg(q_gen_clean_demo_retarget),
        meta=meta,
        title_suffix="clean_demo_retarget",
        out_path=trial_dir / "dmp_trajectory_clean_demo_retarget.png",
    )

    plot_dmp_single(
        q_demo=angles,
        q_gen=np.rad2deg(q_gen_raw_retarget),
        meta=meta,
        title_suffix="raw_retarget_rollout",
        out_path=trial_dir / "dmp_trajectory_raw_retarget.png",
    )

    plot_dmp_single(
        q_demo=clean_angles,
        q_gen=np.rad2deg(q_gen_clean_retarget),
        meta=meta,
        title_suffix="clean_retarget_rollout",
        out_path=trial_dir / "dmp_trajectory_clean_retarget.png",
    )

    # Overlay comparison: retarget DEMO vs retarget ROLLOUT on the same plot (robot space)
    plot_retarget_options_overlay(
        q_gen_demo_retarget=q_gen_raw_demo_retarget,
        q_gen_rollout_retarget=q_gen_raw_retarget,
        meta=meta,
        out_path=trial_dir / "dmp_compare_retarget_options_raw.png",
        title_suffix="raw",
    )
    plot_retarget_options_overlay(
        q_gen_demo_retarget=q_gen_clean_demo_retarget,
        q_gen_rollout_retarget=q_gen_clean_retarget,
        meta=meta,
        out_path=trial_dir / "dmp_compare_retarget_options_clean.png",
        title_suffix="clean",
    )

    # Start simulation playback using the cleaned demo-retarget version.
    #
    # sim/limb_sim.py discovers rollouts by filename (e.g. dmp_rollout_clean.npz),
    # so we write the desired rollout under that name (backing up the original once).
    clean_rollout_path = trial_dir / "dmp_rollout_clean.npz"
    clean_rollout_backup = trial_dir / "dmp_rollout_clean_original.npz"
    clean_rollout_demo_retarget = trial_dir / "dmp_rollout_clean_demo_retarget.npz"

    if clean_rollout_path.exists() and not clean_rollout_backup.exists():
        clean_rollout_path.replace(clean_rollout_backup)

    data = np.load(clean_rollout_demo_retarget)
    np.savez(
        clean_rollout_path,
        q_demo=data["q_demo"],
        q_gen=data["q_gen"],
        q_gen_rad=data["q_gen_rad"],
        t=data["t"],
        dt=data["dt"],
        q0=data["q0"],
        qT=data["qT"],
    )

    if importlib.util.find_spec("pybullet") is None:
        print("PyBullet is not installed; skipping sim/limb_sim.py playback.")
        print("Install it with: pip install pybullet")
    else:
        subprocess.run(
            [sys.executable, "sim/limb_sim_table.py", "--path", str(trial_dir), "--source", "clean", "--loop"],
            check=False,
        )


if __name__ == "__main__":
    main()