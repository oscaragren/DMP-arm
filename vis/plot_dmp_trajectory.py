"""
Plot DMP-learned trajectory vs demo from angles.npz.

Loads angles.npz from a trial directory, fits a DMP, runs rollout, and plots
demo vs generated joint angles (elbow + 3 shoulder DOFs) over normalized time.
"""
import argparse
import json
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
import numpy as np

from dmp.dmp import fit, rollout_simple


def load_angles_demo(trial_dir: Path) -> np.ndarray:
    """Load elbow + shoulder angles from angles.npz; return (T, 4) demo, finite only."""
    npz_path = trial_dir / "angles.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Expected angles.npz at {npz_path}. "
            "Run mapping/sequence_to_angles.py or vis/plot_left_arm_angles.py first."
        )
    data = np.load(npz_path)
    elbow_deg = data["elbow_deg"]
    shoulder_deg = data["shoulder_deg"]
    if shoulder_deg.ndim == 1:
        shoulder_deg = shoulder_deg[:, None]
    q_demo = np.column_stack([elbow_deg, shoulder_deg])
    valid = np.all(np.isfinite(q_demo), axis=1)
    q_demo = q_demo[valid]
    if q_demo.shape[0] < 10:
        raise ValueError(f"Not enough valid samples after cleaning: {q_demo.shape}")
    return q_demo


def plot_dmp_trajectory(
    trial_dir: Path,
    out_path: Path | None = None,
    n_basis: int = 15,
):
    """Fit DMP from trial angles.npz, rollout, and plot demo vs generated."""
    q_demo = load_angles_demo(trial_dir)
    T, n_joints = q_demo.shape
    tau = 1.0
    dt = tau / (T - 1)

    model = fit(
        [q_demo],
        tau=tau,
        dt=dt,
        n_basis_functions=n_basis,
        alpha_canonical=4.0,
        alpha_transformation=25.0,
        beta_transformation=6.25,
    )
    q_gen = rollout_simple(model, q_demo[0], q_demo[-1], tau=tau, dt=dt)

    t_demo = np.linspace(0, tau, q_demo.shape[0])
    t_gen = np.linspace(0, tau, q_gen.shape[0])

    joint_names = ["Elbow flexion", "Shoulder elevation", "Shoulder azimuth", "Shoulder internal rot."]
    colors_demo = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    colors_gen = ["#2980b9", "#c0392b", "#27ae60", "#8e44ad"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    for j in range(n_joints):
        ax = axes[j]
        ax.plot(t_demo, q_demo[:, j], color=colors_demo[j], linewidth=1.5, label="Demo")
        ax.plot(t_gen, q_gen[:, j], color=colors_gen[j], linestyle="--", linewidth=1.2, label="DMP")
        ax.set_ylabel("Angle (deg)")
        ax.set_title(joint_names[j])
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (normalized)")
    meta_path = trial_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    subject = meta.get("subject", "?")
    motion = meta.get("motion", "?")
    trial = meta.get("trial", "?")
    fig.suptitle(f"DMP trajectory — subject {subject}, {motion}, trial {trial}", fontsize=11)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=120)
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot DMP-learned trajectory vs demo from angles.npz."
    )
    parser.add_argument("--path", type=Path, default=None, help="Path to trial dir")
    parser.add_argument("--subject", type=int, default=1, help="Subject number")
    parser.add_argument("--motion", type=str, default="reach", help="Motion name")
    parser.add_argument("--trial", type=int, default=1, help="Trial number")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("test_data/processed"),
        help="Root directory (subject/motion/trial underneath)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path (default: <trial_dir>/dmp_trajectory.png)",
    )
    parser.add_argument("--n-basis", type=int, default=15, help="Number of RBF basis functions")
    args = parser.parse_args()

    if args.path is not None:
        trial_dir = Path(args.path)
    else:
        trial_dir = args.data_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"

    if not trial_dir.exists():
        sys.exit(f"Trial directory not found: {trial_dir}")

    out_path = args.out if args.out is not None else trial_dir / "dmp_trajectory.png"
    plot_dmp_trajectory(trial_dir, out_path=out_path, n_basis=args.n_basis)


if __name__ == "__main__":
    main()
