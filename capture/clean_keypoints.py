"""
Clean & resample keypoints (offline).

- Interpolate short gaps (confidence drops)
- Low-pass filter / smoothing
- Resample to fixed dt (e.g. 100 Hz or 60 Hz)

Deliverable: keypoints_3d_resampled.npy, confidence_resampled.npy, dt (in meta).
Sanity plots: keypoint trajectories (x,y,z over time), confidence over time.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt


def _get_fps_from_video(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else 30.0


def _interpolate_gaps(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    t: np.ndarray,
    confidence_threshold: float = 0.5,
    max_gap_frames: int = 5,
) -> np.ndarray:
    """Fill short low-confidence gaps with linear interpolation. keypoints (T, K, 3), confidence (T, K)."""
    T, K, _ = keypoints.shape
    out = keypoints.copy()
    for k in range(K):
        low = confidence[:, k] < confidence_threshold
        if not np.any(low):
            continue
        # Find contiguous gap segments
        gap_start = np.where(np.diff(np.r_[0, low.astype(int), 0]) == 1)[0]
        gap_end = np.where(np.diff(np.r_[0, low.astype(int), 0]) == -1)[0]
        for start, end in zip(gap_start, gap_end):
            n_gap = end - start
            if n_gap > max_gap_frames:
                continue
            i_before = max(0, start - 1)
            i_after = min(T - 1, end) if end < T else start - 1
            for d in range(3):
                vals = out[:, k, d]
                if i_before >= i_after:
                    out[start:end, k, d] = vals[i_before]
                else:
                    out[start:end, k, d] = np.interp(
                        t[start:end],
                        [t[i_before], t[i_after]],
                        [vals[i_before], vals[i_after]],
                    )
    return out


def _lowpass_filter(
    keypoints: np.ndarray,
    fps: float,
    cutoff_hz: float = 5.0,
    order: int = 2,
) -> np.ndarray:
    """Butterworth low-pass filtfilt per channel. keypoints (T, K, 3)."""
    T, K, _ = keypoints.shape
    nyq = fps * 0.5
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = butter(order, wn, btype="low")
    out = np.zeros_like(keypoints)
    for k in range(K):
        for d in range(3):
            x = keypoints[:, k, d]
            if np.any(np.isnan(x)):
                out[:, k, d] = x
                continue
            out[:, k, d] = filtfilt(b, a, x)
    return out


def _interpolate_nan_gaps_seq(
    seq: np.ndarray,
    t: np.ndarray,
    max_gap_frames: int = 5,
) -> np.ndarray:
    """Fill short NaN gaps in (T, 4, 3) left-arm sequence with linear interpolation."""
    T, K, D = seq.shape
    out = seq.copy().astype(np.float64)
    for k in range(K):
        for d in range(D):
            x = out[:, k, d]
            valid = np.isfinite(x)
            if not np.any(~valid):
                continue
            if np.sum(valid) < 2:
                continue
            t_valid = t[valid]
            x_valid = x[valid]
            f = interp1d(t_valid, x_valid, kind="linear", fill_value="extrapolate")
            # Only fill gaps of length <= max_gap_frames to avoid smoothing over long dropouts
            nan_mask = ~valid
            gap_starts = np.where(np.r_[False, nan_mask[1:] & ~nan_mask[:-1]])[0]
            gap_ends = np.where(np.r_[nan_mask[:-1] & ~nan_mask[1:], False])[0] + 1
            if nan_mask[-1]:
                gap_ends = np.r_[gap_ends, T]
            if nan_mask[0]:
                gap_starts = np.r_[0, gap_starts]
            for gs, ge in zip(gap_starts, gap_ends):
                if ge - gs > max_gap_frames:
                    continue
                out[gs:ge, k, d] = f(t[gs:ge])
    return out


def _resample_seq(
    t: np.ndarray,
    seq: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample (T, 4, 3) sequence to uniform dt. Returns t_new, seq_new. Handles NaN by interpolating from valid points."""
    t_new = np.arange(t[0], t[-1] + 1e-9, dt)
    if t_new[-1] < t[-1] - 1e-6:
        t_new = np.r_[t_new, t[-1]]
    T_new = len(t_new)
    K, D = seq.shape[1], seq.shape[2]
    seq_new = np.zeros((T_new, K, D), dtype=np.float32)
    for k in range(K):
        for d in range(D):
            x = seq[:, k, d]
            if np.any(np.isnan(x)):
                valid = np.isfinite(x)
                if np.sum(valid) < 2:
                    seq_new[:, k, d] = np.nan
                    continue
                f = interp1d(t[valid], x[valid], kind="linear", fill_value="extrapolate")
                seq_new[:, k, d] = f(t_new)
            else:
                f = interp1d(t, x, kind="linear", fill_value="extrapolate")
                seq_new[:, k, d] = f(t_new)
    return t_new, seq_new


# Filenames for cleaned left-arm sequence (do not overwrite originals)
LEFT_ARM_SEQ_CLEANED = "left_arm_seq_camera_cleaned.npy"
LEFT_ARM_T_CLEANED = "left_arm_t_cleaned.npy"


def run_clean_left_arm_sequence(
    trial_dir: Path,
    max_gap_frames: int = 5,
    cutoff_hz: float = 5.0,
    filter_order: int = 2,
    target_dt: float | None = 0.04,
) -> None:
    """
    Clean left_arm_seq_camera.npy in trial_dir: interpolate NaN gaps, low-pass filter, optional resample.
    Writes to left_arm_seq_camera_cleaned.npy and left_arm_t_cleaned.npy (originals unchanged).
    Expects (T, 4, 3) and (T,) arrays.
    """
    seq_path = trial_dir / "left_arm_seq_camera.npy"
    t_path = trial_dir / "left_arm_t.npy"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing {seq_path}")
    seq = np.load(seq_path)
    if seq.ndim != 3 or seq.shape[1] != 4 or seq.shape[2] != 3:
        raise ValueError(f"Expected left_arm_seq shape (T, 4, 3), got {seq.shape}")
    t = np.load(t_path) if t_path.exists() else np.arange(seq.shape[0], dtype=np.float64)
    if len(t) != seq.shape[0]:
        t = np.arange(seq.shape[0], dtype=np.float64)

    # Estimate fps from timestamps
    if len(t) > 1:
        dt_med = np.median(np.diff(t))
        fps = float(1.0 / dt_med) if dt_med > 1e-6 else 25.0
    else:
        fps = 25.0

    # 1) Interpolate short NaN gaps
    seq = _interpolate_nan_gaps_seq(seq, t, max_gap_frames=max_gap_frames)

    # 2) Low-pass filter (skip channels that are still all NaN)
    seq = _lowpass_filter(seq, fps, cutoff_hz=cutoff_hz, order=filter_order)

    # 3) Optional resample to uniform dt
    if target_dt is not None and target_dt > 0:
        t, seq = _resample_seq(t, seq, target_dt)

    out_seq_path = trial_dir / LEFT_ARM_SEQ_CLEANED
    out_t_path = trial_dir / LEFT_ARM_T_CLEANED
    np.save(out_seq_path, seq)
    np.save(out_t_path, t.astype(np.float64))
    print(f"Cleaned and saved {out_seq_path.name} (and {out_t_path.name}), {seq.shape[0]} frames.")


def _resample(
    t: np.ndarray,
    keypoints: np.ndarray,
    confidence: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample to uniform dt. Returns t_new, keypoints_new (T_new, K, 3), confidence_new (T_new, K)."""
    t_new = np.arange(0, t[-1], dt)
    if t_new[-1] < t[-1] - 1e-6:
        t_new = np.r_[t_new, t[-1]]
    T_new = len(t_new)
    K = keypoints.shape[1]
    keypoints_new = np.zeros((T_new, K, 3), dtype=np.float32)
    confidence_new = np.zeros((T_new, K), dtype=np.float32)
    for k in range(K):
        for d in range(3):
            f = interp1d(t, keypoints[:, k, d], kind="linear", fill_value="extrapolate")
            keypoints_new[:, k, d] = f(t_new)
        f = interp1d(t, confidence[:, k], kind="linear", fill_value="extrapolate")
        confidence_new[:, k] = f(t_new)
    return t_new, keypoints_new, confidence_new


def _plot_trajectories(
    t: np.ndarray,
    keypoints: np.ndarray,
    keypoint_names: list,
    out_path: Path,
) -> None:
    """Plot x, y, z over time for each keypoint."""
    T, K, _ = keypoints.shape
    fig, axes = plt.subplots(K, 1, figsize=(10, 2.5 * K), sharex=True)
    if K == 1:
        axes = [axes]
    for k in range(K):
        ax = axes[k]
        ax.plot(t, keypoints[:, k, 0], label="x", alpha=0.8)
        ax.plot(t, keypoints[:, k, 1], label="y", alpha=0.8)
        ax.plot(t, keypoints[:, k, 2], label="z", alpha=0.8)
        ax.set_ylabel("m")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(keypoint_names[k] if k < len(keypoint_names) else f"keypoint_{k}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Keypoint trajectories (resampled)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_confidence(t: np.ndarray, confidence: np.ndarray, keypoint_names: list, out_path: Path) -> None:
    """Plot confidence over time for each keypoint."""
    K = confidence.shape[1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    for k in range(K):
        label = keypoint_names[k] if k < len(keypoint_names) else f"kp_{k}"
        ax.plot(t, confidence[:, k], label=label, alpha=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("confidence")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Confidence over time (resampled)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def run_clean_resample(
    processed_dir: Path,
    confidence_threshold: float = 0.5,
    max_gap_frames: int = 5,
    cutoff_hz: float = 5.0,
    filter_order: int = 2,
    target_dt: float = 0.01,
) -> None:
    """
    Load keypoints from processed_dir, clean, filter, resample, save and plot.
    Expects keypoints_3d.npy, confidence.npy, meta.json and source_video in meta.
    """
    keypoints_path = processed_dir / "keypoints_3d.npy"
    confidence_path = processed_dir / "confidence.npy"
    meta_path = processed_dir / "meta.json"
    if not keypoints_path.exists() or not confidence_path.exists():
        raise FileNotFoundError(f"Missing keypoints_3d.npy or confidence.npy in {processed_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {processed_dir}")

    keypoints = np.load(keypoints_path)
    confidence = np.load(confidence_path)
    with open(meta_path) as f:
        meta = json.load(f)

    source_video = meta.get("source_video")
    if source_video:
        video_path = Path(source_video)
        if not video_path.is_absolute():
            video_path = Path.cwd() / video_path
        fps = _get_fps_from_video(video_path)
    else:
        fps = 30.0
    keypoint_names = meta.get("keypoint_names", [f"kp_{i}" for i in range(keypoints.shape[1])])

    T = keypoints.shape[0]
    t = np.arange(T) / fps

    # 1) Interpolate short gaps
    keypoints = _interpolate_gaps(
        keypoints,
        confidence,
        t,
        confidence_threshold=confidence_threshold,
        max_gap_frames=max_gap_frames,
    )

    # 2) Low-pass filter
    keypoints = _lowpass_filter(keypoints, fps, cutoff_hz=cutoff_hz, order=filter_order)

    # 3) Resample to fixed dt
    t_new, keypoints_resampled, confidence_resampled = _resample(t, keypoints, confidence, target_dt)

    # 4) Save deliverables
    np.save(processed_dir / "keypoints_3d_resampled.npy", keypoints_resampled)
    np.save(processed_dir / "confidence_resampled.npy", confidence_resampled)
    meta["dt_resampled"] = float(target_dt)
    meta["fps_resampled"] = float(1.0 / target_dt)
    meta["n_frames_resampled"] = int(len(t_new))
    meta["clean_params"] = {
        "confidence_threshold": confidence_threshold,
        "max_gap_frames": max_gap_frames,
        "cutoff_hz": cutoff_hz,
        "filter_order": filter_order,
    }
    with open(processed_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 5) Sanity plots
    _plot_trajectories(t_new, keypoints_resampled, keypoint_names, processed_dir / "trajectories.png")
    _plot_confidence(t_new, confidence_resampled, keypoint_names, processed_dir / "confidence_over_time.png")

    print(
        f"Clean & resample done: {keypoints_resampled.shape[0]} frames at {1/target_dt:.0f} Hz (dt={target_dt}), "
        f"plots saved in {processed_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Clean and resample keypoints (interpolate gaps, low-pass filter, resample to fixed dt)."
    )
    parser.add_argument("--subject", type=int, required=True, help="Subject number")
    parser.add_argument("--motion", type=str, required=True, help="Motion name (e.g. reach)")
    parser.add_argument("--trial", type=int, required=True, help="Trial number")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("test_data/processed"),
        help="Root of processed data",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Below this confidence, frame is treated as gap for interpolation",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=5,
        help="Interpolate gaps up to this many frames",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=5.0,
        help="Low-pass filter cutoff (Hz)",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=2,
        help="Butterworth filter order",
    )
    parser.add_argument(
        "--target-dt",
        type=float,
        default=0.01,
        help="Target sampling interval in seconds (e.g. 0.01 = 100 Hz)",
    )
    args = parser.parse_args()

    processed_trial_dir = (
        args.processed_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"
    )
    run_clean_resample(
        processed_trial_dir,
        confidence_threshold=args.confidence_threshold,
        max_gap_frames=args.max_gap_frames,
        cutoff_hz=args.cutoff_hz,
        filter_order=args.filter_order,
        target_dt=args.target_dt,
    )


if __name__ == "__main__":
    main()
