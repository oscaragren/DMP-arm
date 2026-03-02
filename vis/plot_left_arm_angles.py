"""
Plot left arm joint angles over time from capture/3d_pose.py output.

Loads left_arm_seq_camera.npy (T, 4, 3) and left_arm_t.npy from a trial directory,
converts sequence to angles via mapping.sequence_to_angles, and plots them vs time.
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path for mapping import
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
import numpy as np

from mapping.sequence_to_angles import sequence_to_angles


def load_trial(trial_dir: Path):
    """Load sequence and time from a trial directory. Returns seq (T, 4, 3), t (T,), meta dict."""
    seq_path = trial_dir / "left_arm_seq_camera.npy"
    t_path = trial_dir / "left_arm_t.npy"
    meta_path = trial_dir / "meta.json"
    if not seq_path.exists():
        raise FileNotFoundError(f"Not found: {seq_path}")
    seq = np.load(seq_path)
    if seq.ndim != 3 or seq.shape[1] != 4 or seq.shape[2] != 3:
        raise ValueError(
            f"Expected seq shape (T, 4, 3) [left_shoulder, left_elbow, left_wrist, right_shoulder], got {seq.shape}"
        )
    t = np.load(t_path) if t_path.exists() else np.arange(seq.shape[0], dtype=np.float64)
    meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return seq, t, meta


def plot_angles_over_time(
    seq: np.ndarray,
    t: np.ndarray,
    meta: dict,
    out_path: Path | None = None,
):
    """Plot elbow flexion and shoulder angles (elevation, azimuth, internal rotation) vs time."""
    elbow_deg, shoulder_deg = sequence_to_angles(seq)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Elbow flexion
    ax = axes[0]
    valid = np.isfinite(elbow_deg)
    if np.any(valid):
        ax.plot(t[valid], elbow_deg[valid], color="#3498db", linewidth=1.5, label="Elbow flexion")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Elbow flexion (upper arm–forearm angle)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 200)

    # Shoulder: elevation, azimuth, internal rotation
    ax = axes[1]
    labels = ["Elevation", "Azimuth", "Internal rotation"]
    colors = ["#e74c3c", "#2ecc71", "#9b59b6"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        vals = shoulder_deg[:, i]
        valid = np.isfinite(vals)
        if np.any(valid):
            ax.plot(t[valid], vals[valid], color=color, linewidth=1.5, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Shoulder 3-DOF (elevation, azimuth, internal rotation)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    subject = meta.get("subject", "?")
    motion = meta.get("motion", "?")
    trial = meta.get("trial", "?")
    fig.suptitle(f"Left arm angles — subject {subject}, {motion}, trial {trial}", fontsize=11)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=120)
        print(f"Saved {out_path}")
        npz_path = out_path.parent / "angles.npz"
        np.savez(npz_path, elbow_deg=elbow_deg, shoulder_deg=shoulder_deg)
        print(f"Saved {npz_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot left arm joint angles over time from 3d_pose.py trial output."
    )
    parser.add_argument("--path", type=Path, default=None, help="Path to trial dir (overrides subject/motion/trial)")
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
        help="Output path for figure (default: <trial_dir>/angles.png)",
    )
    args = parser.parse_args()

    if args.path is not None:
        trial_dir = Path(args.path)
    else:
        trial_dir = args.data_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"

    seq, t, meta = load_trial(trial_dir)
    print(f"Loaded {seq.shape[0]} frames from {trial_dir}")

    if args.out is None:
        args.out = trial_dir / "angles.png"
    plot_angles_over_time(seq, t, meta, out_path=args.out)


if __name__ == "__main__":
    main()
