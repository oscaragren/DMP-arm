"""
Clean and optionally resample left-arm joint angles in joint space.

This module is complementary to keypoint cleaning:
- keypoint cleaning: filter/resample in x,y,z before inverse kinematics
- angle cleaning (this file): map to angles first, then filter/resample angles
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

from mapping.sequence_to_angles import sequence_to_angles_rad, sequence_to_angles
from vis.trial_naming import trial_prefix


LEFT_ARM_ANGLES_CLEANED = "left_arm_angles_cleaned.npz"


def _estimate_fps(t: np.ndarray) -> float:
    if t.ndim != 1 or t.size < 2:
        return 25.0
    dt_med = np.median(np.diff(t))
    if dt_med <= 1e-6:
        return 25.0
    return float(1.0 / dt_med)


def _lowpass_angles(q: np.ndarray, fps: float, cutoff_hz: float, order: int) -> np.ndarray:
    """Low-pass filter each joint independently. q shape: (T, n_joints)."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"Expected q with shape (T, n_joints), got {q.shape}")
    if q.shape[0] < 3:
        return q.copy()

    nyq = fps * 0.5
    wn = min(float(cutoff_hz) / nyq, 0.99)
    b, a = butter(int(order), wn, btype="low")

    out = np.zeros_like(q)
    for j in range(q.shape[1]):
        x = q[:, j]
        if np.any(~np.isfinite(x)):
            out[:, j] = x
            continue
        out[:, j] = filtfilt(b, a, x)
    return out


def _resample_angles(t: np.ndarray, q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample angles to uniform dt. Handles NaNs by interpolating valid samples."""
    if t.ndim != 1 or t.shape[0] != q.shape[0]:
        raise ValueError(f"Expected t shape ({q.shape[0]},), got {t.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    t_new = np.arange(t[0], t[-1] + 1e-9, dt)
    if t_new[-1] < t[-1] - 1e-6:
        t_new = np.r_[t_new, t[-1]]

    q_new = np.zeros((t_new.shape[0], q.shape[1]), dtype=np.float64)
    for j in range(q.shape[1]):
        x = q[:, j]
        if np.any(~np.isfinite(x)):
            valid = np.isfinite(x)
            if np.sum(valid) < 2:
                q_new[:, j] = np.nan
                continue
            f = interp1d(t[valid], x[valid], kind="linear", fill_value="extrapolate")
            q_new[:, j] = f(t_new)
            continue
        f = interp1d(t, x, kind="linear", fill_value="extrapolate")
        q_new[:, j] = f(t_new)
    return t_new, q_new


def clean_angles_trajectory(
    elbow_rad: np.ndarray,
    shoulder_rad: np.ndarray,
    t: np.ndarray,
    cutoff_hz: float = 5.0,
    filter_order: int = 2,
    target_dt: float | None = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clean a left-arm angle trajectory in joint space.

    Returns:
        elbow_clean_rad: (T_new,)
        shoulder_clean_rad: (T_new, 3)
        t_clean: (T_new,)
    """
    elbow = np.asarray(elbow_rad, dtype=np.float64)
    shoulder = np.asarray(shoulder_rad, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    if elbow.ndim != 1:
        raise ValueError(f"Expected elbow_rad shape (T,), got {elbow.shape}")
    if shoulder.ndim != 2 or shoulder.shape[1] != 3:
        raise ValueError(f"Expected shoulder_rad shape (T, 3), got {shoulder.shape}")
    if t.ndim != 1 or t.shape[0] != elbow.shape[0] or shoulder.shape[0] != elbow.shape[0]:
        raise ValueError("Expected matching lengths for elbow_rad, shoulder_rad, and t")

    q = np.column_stack([elbow, shoulder])  # (T, 4)
    fps = _estimate_fps(t)
    q = _lowpass_angles(q, fps=fps, cutoff_hz=cutoff_hz, order=filter_order)

    if target_dt is not None and target_dt > 0:
        t, q = _resample_angles(t, q, dt=target_dt)

    return q[:, 0], q[:, 1:], t


def run_clean_left_arm_angles(
    trial_dir: Path,
    cutoff_hz: float = 5.0,
    filter_order: int = 2,
    target_dt: float | None = 0.04,
    output_name: str = LEFT_ARM_ANGLES_CLEANED,
) -> Path:
    """
    Load left_arm_seq_camera(.npy/.t.npy), map to angles, clean in joint space, save NPZ.
    """
    seq_path = trial_dir / "left_arm_seq_camera.npy"
    t_path = trial_dir / "left_arm_t.npy"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing {seq_path}")

    seq = np.load(seq_path)
    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] < 4:
        raise ValueError(f"Expected left_arm_seq shape (T, N>=4, 3), got {seq.shape}")

    t = np.load(t_path) if t_path.exists() else np.arange(seq.shape[0], dtype=np.float64)
    if t.ndim != 1 or t.shape[0] != seq.shape[0]:
        t = np.arange(seq.shape[0], dtype=np.float64)

    elbow_rad, shoulder_rad = np.deg2rad(sequence_to_angles(seq))
    elbow_clean_rad, shoulder_clean_rad, t_clean = clean_angles_trajectory(
        elbow_rad,
        shoulder_rad,
        t,
        cutoff_hz=cutoff_hz,
        filter_order=filter_order,
        target_dt=target_dt,
    )

    out_path = trial_dir / output_name
    np.savez(
        out_path,
        elbow_rad=elbow_clean_rad,
        shoulder_rad=shoulder_clean_rad,
        elbow_deg=np.degrees(elbow_clean_rad),
        shoulder_deg=np.degrees(shoulder_clean_rad),
        t=t_clean,
        cutoff_hz=float(cutoff_hz),
        filter_order=int(filter_order),
        target_dt=None if target_dt is None else float(target_dt),
        source="kinematics/clean_angles.py",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean left-arm joint angles in joint space.")
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
    parser.add_argument("--cutoff-hz", type=float, default=5.0, help="Low-pass filter cutoff (Hz)")
    parser.add_argument("--filter-order", type=int, default=2, help="Butterworth filter order")
    parser.add_argument(
        "--target-dt",
        type=float,
        default=0.04,
        help="Resample interval in seconds, or 0 to disable resample",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=LEFT_ARM_ANGLES_CLEANED,
        help=f"Output NPZ filename (default: {LEFT_ARM_ANGLES_CLEANED})",
    )
    args = parser.parse_args()

    if args.path is not None:
        trial_dir = Path(args.path)
    else:
        trial_dir = args.data_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"

    prefix = trial_prefix(trial_dir)
    clean_dt = args.target_dt if args.target_dt > 0 else None
    output_name = args.output_name
    if output_name == LEFT_ARM_ANGLES_CLEANED:
        output_name = f"{prefix}{output_name}"

    out = run_clean_left_arm_angles(
        trial_dir=trial_dir,
        cutoff_hz=args.cutoff_hz,
        filter_order=args.filter_order,
        target_dt=clean_dt,
        output_name=output_name,
    )
    print(f"Saved cleaned angles: {out}")


if __name__ == "__main__":
    main()
