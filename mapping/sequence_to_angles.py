"""
Convert left-arm 3D keypoint sequences (from camera video) to joint angles.

Input: seq (T, 4, 3) with [left_shoulder, left_elbow, left_wrist, right_shoulder]
in camera-frame meters (e.g. from capture/3d_pose.py).

Output: elbow flexion (T,) and shoulder 3-DOF angles (T, 3) in degrees,
suitable for plotting and downstream use.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Project root for kinematics import when run as script
if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from kinematics.left_arm_angles import elbow_flexion_deg, shoulder_angles_3dof


def sequence_to_angles(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a left-arm 3D sequence to elbow and shoulder angles.

    seq: (T, 4, 3) with [left_shoulder, left_elbow, left_wrist, right_shoulder]
         in camera-frame meters.

    Returns:
        elbow_deg: (T,) elbow flexion in degrees; NaN where invalid.
        shoulder_deg: (T, 3) shoulder angles in degrees [elevation, azimuth, internal_rotation]; NaN where invalid.
    """
    if seq.ndim != 3 or seq.shape[1] != 4 or seq.shape[2] != 3:
        raise ValueError(
            f"Expected seq shape (T, 4, 3) [left_shoulder, left_elbow, left_wrist, right_shoulder], got {seq.shape}"
        )
    elbow_deg = elbow_flexion_deg(seq)
    shoulder_deg = shoulder_angles_3dof(seq)
    return elbow_deg, shoulder_deg


def save_angles_for_trial(trial_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load left_arm_seq_camera.npy from a trial directory, compute angles, save to angles.npz.

    trial_dir: path to trial (e.g. test_data/processed/subject_01/reach/trial_001).

    Returns:
        elbow_deg (T,), shoulder_deg (T, 3).
    """
    seq_path = trial_dir / "left_arm_seq_camera.npy"
    if not seq_path.exists():
        raise FileNotFoundError(f"Not found: {seq_path}")
    seq = np.load(seq_path)
    elbow_deg, shoulder_deg = sequence_to_angles(seq)
    out_path = trial_dir / "angles.npz"
    np.savez(out_path, elbow_deg=elbow_deg, shoulder_deg=shoulder_deg)
    meta_path = trial_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    meta["angles_source"] = "mapping/sequence_to_angles.py"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {out_path} (elbow: {elbow_deg.shape}, shoulder: {shoulder_deg.shape})")
    return elbow_deg, shoulder_deg


def main():
    parser = argparse.ArgumentParser(
        description="Convert left-arm 3D sequence to angles and optionally save to trial dir."
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
    args = parser.parse_args()

    if args.path is not None:
        trial_dir = Path(args.path)
    else:
        trial_dir = args.data_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"

    save_angles_for_trial(trial_dir)


if __name__ == "__main__":
    main()
