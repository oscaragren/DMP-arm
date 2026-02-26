"""
Visualize a 3D trajectory from capture/3d_pose.py (test_data/processed/...).

Loads left_arm_seq_camera.npy (T, 3, 3) and left_arm_t.npy, plots shoulder/elbow/wrist
in camera frame and optionally an animated arm.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

KEYPOINT_NAMES = ["left_shoulder", "left_elbow", "left_wrist"]


def load_trajectory(trial_dir: Path):
    """Load trajectory from a trial directory. Returns seq (T, 3, 3), t (T,), meta dict."""
    seq_path = trial_dir / "left_arm_seq_camera.npy"
    t_path = trial_dir / "left_arm_t.npy"
    meta_path = trial_dir / "meta.json"
    if not seq_path.exists():
        raise FileNotFoundError(f"Not found: {seq_path}")
    seq = np.load(seq_path)
    t = np.load(t_path) if t_path.exists() else np.arange(seq.shape[0], dtype=np.float64)
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return seq, t, meta


def plot_3d_trajectory(seq: np.ndarray, t: np.ndarray, meta: dict, out_path: Path = None):
    """Plot 3D trajectories of shoulder, elbow, wrist and arm stick at start/mid/end."""
    T, K, _ = seq.shape
    names = meta.get("keypoint_names", KEYPOINT_NAMES)[:K]

    fig = plt.figure(figsize=(12, 5))

    # Left: 3D trajectory lines + arm at sample frames
    ax1 = fig.add_subplot(121, projection="3d")
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    # 3D plot: (x, z, y) so camera y is on the vertical axis
    for k in range(K):
        valid = ~np.isnan(seq[:, k, 0])
        if np.any(valid):
            ax1.plot(
                seq[valid, k, 0],  # x
                seq[valid, k, 2],  # z
                seq[valid, k, 1],  # y -> vertical
                color=colors[k % len(colors)],
                label=names[k] if k < len(names) else f"kp{k}",
                alpha=0.8,
                linewidth=1.5,
            )
    # Arm stick at a few time indices
    for i, ti in enumerate([0, T // 2, T - 1]):
        if T == 0:
            break
        alpha = 0.3 + 0.35 * (i / max(1, 2))
        for j in range(K - 1):
            p0 = seq[ti, j]
            p1 = seq[ti, j + 1]
            if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                ax1.plot(
                    [p0[0], p1[0]],
                    [p0[2], p1[2]],
                    [p0[1], p1[1]],
                    "k-",
                    alpha=alpha,
                    linewidth=2,
                )
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax1.set_zlabel("y (m)")
    ax1.legend()
    ax1.set_title("3D trajectory (camera frame)")
    set_axes_equal(ax1)

    # Right: x, y, z vs time per keypoint
    ax2 = fig.add_subplot(122)
    for k in range(K):
        valid = ~np.isnan(seq[:, k, 0])
        name = names[k] if k < len(names) else f"kp{k}"
        if np.any(valid):
            ax2.plot(t[valid], seq[valid, k, 0], color=colors[k % len(colors)], alpha=0.8, linestyle="-", label=f"{name} x")
            ax2.plot(t[valid], seq[valid, k, 1], color=colors[k % len(colors)], alpha=0.8, linestyle="--", label=f"{name} y")
            ax2.plot(t[valid], seq[valid, k, 2], color=colors[k % len(colors)], alpha=0.8, linestyle=":", label=f"{name} z")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("position (m)")
    ax2.set_title("Position vs time")
    ax2.legend(loc="upper right", fontsize=7, ncol=1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close()


def set_axes_equal(ax: Axes3D):
    """Set 3D axes to equal aspect."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    radius = (np.abs(limits[:, 1] - limits[:, 0])).max() / 2
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D pose trajectory from 3d_pose.py output.")
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
        help="Output path for figure (default: <trial_dir>/trajectory.png)",
    )
    args = parser.parse_args()

    if args.path is not None:
        trial_dir = Path(args.path)
    else:
        trial_dir = args.data_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"
    if args.out is None:
        args.out = trial_dir / "trajectory.png"
    seq, t, meta = load_trajectory(trial_dir)
    print(f"Loaded {seq.shape[0]} frames from {trial_dir}")
    plot_3d_trajectory(seq, t, meta, out_path=args.out)


if __name__ == "__main__":
    main()
