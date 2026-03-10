"""
Run the full left-arm pipeline for a single trial:

- Clean keypoint sequence (interpolate NaNs, low-pass filter, optional resample)
- Map 3D keypoint sequence to joint angles (saves angles.npz)
- Fit and rollout DMPs
- Plot raw angles over time
- Plot DMP demo vs rollout

Usage (from project root):

    python run_full_pipeline.py --path path/to/trial

or, using subject/motion/trial indexing:

    python run_full_pipeline.py --subject 1 --motion reach --trial 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

# Ensure project root is on sys.path
_here = Path(__file__).resolve()
_project_root = _here.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from capture.clean_keypoints import run_clean_left_arm_sequence
from mapping.sequence_to_angles import save_angles_for_trial
from vis.plot_left_arm_angles import load_trial, plot_angles_over_time
from vis.plot_dmp_trajectory import plot_dmp_trajectory


def run_full_pipeline(
    trial_dir: Path,
    clean: bool = False,
    clean_max_gap_frames: int = 5,
    clean_cutoff_hz: float = 5.0,
    clean_filter_order: int = 2,
    clean_target_dt: float | None = 0.04,
) -> None:
    """Run mapping, DMP rollout, and plotting for a single trial directory."""
    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")

    print(f"Running full pipeline for trial: {trial_dir}")

    # 0) Clean left_arm_seq_camera.npy (interpolate NaNs, low-pass, optional resample)
    if clean:
        print("Step 0/4: cleaning keypoint sequence (interpolate, low-pass, resample)...")
        run_clean_left_arm_sequence(
            trial_dir,
            max_gap_frames=clean_max_gap_frames,
            cutoff_hz=clean_cutoff_hz,
            filter_order=clean_filter_order,
            target_dt=clean_target_dt,
        )
    else:
        print("Step 0/4: skipping keypoint cleaning.")

    # 1) Map sequence to angles and save angles.npz (rad + deg)
    print("Step 1/4: mapping sequence to joint angles (saving angles.npz)...")
    save_angles_for_trial(trial_dir)

    # 2) Plot raw angles over time
    print("Step 2/4: plotting raw joint angles over time...")
    seq, t, meta = load_trial(trial_dir)
    angles_fig_path = trial_dir / "angles.png"
    plot_angles_over_time(seq, t, meta, out_path=angles_fig_path)

    # 3) Fit DMP, rollout, and plot demo vs rollout
    print("Step 3/4: fitting DMP, rolling out, and plotting trajectory...")
    dmp_fig_path = trial_dir / "dmp_trajectory.png"
    plot_dmp_trajectory(trial_dir, out_path=dmp_fig_path)

    print("Full pipeline finished.")
    print(f"  - Angles figure:         {angles_fig_path}")
    print(f"  - DMP trajectory figure: {dmp_fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full left-arm pipeline: clean keypoints, map to angles, rollout DMPs, and plot results."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to trial dir (overrides subject/motion/trial)",
    )
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
        "--clean",
        action="store_true",
        help="Clean keypoint sequence (interpolate NaNs, low-pass filter, optional resample)",
    )
    parser.add_argument(
        "--clean-max-gap-frames",
        type=int,
        default=5,
        help="Max NaN gap length to interpolate (default: 5)",
    )
    parser.add_argument(
        "--clean-cutoff-hz",
        type=float,
        default=5.0,
        help="Low-pass filter cutoff in Hz (default: 5.0)",
    )
    parser.add_argument(
        "--clean-filter-order",
        type=int,
        default=2,
        help="Butterworth filter order (default: 2)",
    )
    parser.add_argument(
        "--clean-target-dt",
        type=float,
        default=0.04,
        help="Resample interval in seconds, or 0 to disable resample (default: 0.04)",
    )
    args = parser.parse_args()

    if args.path is not None:
        trial_dir = Path(args.path)
    else:
        trial_dir = args.data_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"

    clean_dt = args.clean_target_dt if args.clean_target_dt > 0 else None
    run_full_pipeline(
        trial_dir,
        clean=args.clean,
        clean_max_gap_frames=args.clean_max_gap_frames,
        clean_cutoff_hz=args.clean_cutoff_hz,
        clean_filter_order=args.clean_filter_order,
        clean_target_dt=clean_dt,
    )


if __name__ == "__main__":
    main()

