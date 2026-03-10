"""
Convert left-arm 3D keypoint sequences (from camera video) to joint angles.

Input: seq (T, 4, 3) with [left_shoulder, left_elbow, left_wrist, right_shoulder]
in camera-frame meters (e.g. from capture/3d_pose.py).

Output: elbow flexion (T,) and shoulder 3-DOF angles (T, 3) in **radians**
for downstream use. Convenience helpers exist to also obtain degrees when
plotting, but the canonical representation is radians.
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

from kinematics.left_arm_angles import (
    elbow_flexion_deg,
    shoulder_flex_abd_rot_3dof,
    elbow_flexion_rad,
    shoulder_flex_abd_rot_3dof_rad,
)


def sequence_to_angles_rad(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a left-arm 3D sequence to elbow and shoulder angles in **radians**.

    seq: (T, 4, 3) with [left_shoulder, left_elbow, left_wrist, right_shoulder]
         in camera-frame meters.

    Returns:
        elbow_rad: (T,) elbow flexion in radians; NaN where invalid.
        shoulder_rad: (T, 3) shoulder angles in radians
            [shoulder_flexion, shoulder_abduction, shoulder_internal_rotation]; NaN where invalid.
    """
    if seq.ndim != 3 or seq.shape[1] != 4 or seq.shape[2] != 3:
        raise ValueError(
            f"Expected seq shape (T, 4, 3) [left_shoulder, left_elbow, left_wrist, right_shoulder], got {seq.shape}"
        )
    elbow_rad = elbow_flexion_rad(seq)
    shoulder_rad = shoulder_flex_abd_rot_3dof_rad(seq)
    return elbow_rad, shoulder_rad


def sequence_to_angles(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Backwards-compatible helper that returns angles in **degrees**.

    Prefer `sequence_to_angles_rad` for new code.
    """
    elbow_rad, shoulder_rad = sequence_to_angles_rad(seq)
    return np.degrees(elbow_rad), np.degrees(shoulder_rad)


def save_angles_for_trial(trial_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load left_arm_seq_camera.npy (or left_arm_seq_camera_cleaned.npy if present) from a trial
    directory, compute angles, save to angles.npz.

    trial_dir: path to trial (e.g. test_data/processed/subject_01/reach/trial_001).

    Returns:
        elbow_rad (T,), shoulder_rad (T, 3).
    """
    from capture.clean_keypoints import LEFT_ARM_SEQ_CLEANED

    seq_path = trial_dir / LEFT_ARM_SEQ_CLEANED
    if not seq_path.exists():
        seq_path = trial_dir / "left_arm_seq_camera.npy"
    if not seq_path.exists():
        raise FileNotFoundError(f"Not found: {seq_path} or left_arm_seq_camera.npy")
    seq = np.load(seq_path)
    elbow_rad, shoulder_rad = sequence_to_angles_rad(seq)
    elbow_deg = np.degrees(elbow_rad)
    shoulder_deg = np.degrees(shoulder_rad)
    out_path = trial_dir / "angles.npz"
    # Save both radians (canonical) and degrees (for backwards compatibility / plotting).
    np.savez(
        out_path,
        elbow_rad=elbow_rad,
        shoulder_rad=shoulder_rad,
        elbow_deg=elbow_deg,
        shoulder_deg=shoulder_deg,
    )
    meta_path = trial_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    meta["angles_source"] = "mapping/sequence_to_angles.py"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {out_path} (elbow: {elbow_rad.shape}, shoulder: {shoulder_rad.shape})")
    return elbow_rad, shoulder_rad


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
