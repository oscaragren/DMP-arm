"""
Run a sweep pipeline for a single trial, comparing RAW vs multiple CLEAN settings.

For one trial, this script generates:

Angles (deg) plots:
- 1x RAW angles plot
- 4x CLEAN angles plots (for filter_order in {1, 2, 4, 6})
- 1x overlay grid comparing RAW vs each CLEAN and "all together"

DMP trajectory plots (demo vs rollout, deg, normalized time), for each n_basis in {10, 30, 60}:
- 1x RAW DMP plot
- 4x CLEAN DMP plots (filter_order sweep)
- 1x overlay grid comparing RAW vs each CLEAN and "all together"

All generated PNGs (and intermediate angle NPZs) are placed in a per-trial output
folder and also bundled into a zip file stored in the trial directory.

Usage (from project root):

    python3 run_full_pipeline.py --path path/to/trial

or, using subject/motion/trial indexing:

    python3 run_full_pipeline.py --subject 1 --motion reach --trial 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import sys

# Ensure project root is on sys.path
_here = Path(__file__).resolve()
_project_root = _here.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
import numpy as np

from capture.clean_keypoints import run_clean_left_arm_sequence, LEFT_ARM_SEQ_CLEANED, LEFT_ARM_T_CLEANED
from dmp.dmp import fit, rollout_simple
from kinematics.joint_dynamics import smooth_angles_deg, validate_joint_trajectory_deg
from mapping.sequence_to_angles import sequence_to_angles_rad
from vis.plotting import (
    plot_angles_overlay_grid,
    plot_angles_single,
    plot_dmp_order_basis_grids_per_joint,
    plot_dmp_overlay_grid,
    plot_dmp_single,
)
from vis.trial_naming import trial_prefix


def _load_meta(trial_dir: Path) -> dict:
    meta_path = trial_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_raw_seq_t(trial_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    seq_path = trial_dir / "left_arm_seq_camera.npy"
    t_path = trial_dir / "left_arm_t.npy"
    if not seq_path.exists():
        raise FileNotFoundError(f"Not found: {seq_path}")
    seq = np.load(seq_path)
    if seq.ndim != 3 or seq.shape[1] != 4 or seq.shape[2] != 3:
        raise ValueError(f"Expected seq shape (T, 4, 3), got {seq.shape}")
    t = np.load(t_path) if t_path.exists() else np.arange(seq.shape[0], dtype=np.float64)
    if t.ndim != 1 or len(t) != seq.shape[0]:
        t = np.arange(seq.shape[0], dtype=np.float64)
    return seq, t.astype(np.float64, copy=False)


def _load_clean_seq_t(trial_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    seq_path = trial_dir / LEFT_ARM_SEQ_CLEANED
    t_path = trial_dir / LEFT_ARM_T_CLEANED
    if not seq_path.exists():
        raise FileNotFoundError(f"Not found: {seq_path}")
    seq = np.load(seq_path)
    if seq.ndim != 3 or seq.shape[1] != 4 or seq.shape[2] != 3:
        raise ValueError(f"Expected cleaned seq shape (T, 4, 3), got {seq.shape}")
    t = np.load(t_path) if t_path.exists() else np.arange(seq.shape[0], dtype=np.float64)
    if t.ndim != 1 or len(t) != seq.shape[0]:
        t = np.arange(seq.shape[0], dtype=np.float64)
    return seq, t.astype(np.float64, copy=False)


def _save_angles_npz(out_npz: Path, elbow_rad: np.ndarray, shoulder_rad: np.ndarray) -> None:
    np.savez(
        out_npz,
        elbow_rad=elbow_rad,
        shoulder_rad=shoulder_rad,
        elbow_deg=np.degrees(elbow_rad),
        shoulder_deg=np.degrees(shoulder_rad),
    )


def _fit_rollout_dmp(q_demo: np.ndarray, n_basis: int) -> tuple[np.ndarray, np.ndarray, float, object]:
    """Return (q_demo_smoothed_rad, q_gen_rad, dt, model)."""
    # Smooth demonstrated angles before fitting / finite‑difference derivatives.
    q_demo = np.deg2rad(smooth_angles_deg(np.degrees(q_demo), method="savgol"))
    T = q_demo.shape[0]
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
    #report = validate_joint_trajectory_deg(np.degrees(q_gen), dt, name=f"DMP rollout (deg), n_basis={n_basis}")
    #print(report.reason)
    return q_demo, q_gen, dt, model


def _save_dmp_rollout_npz(
    out_npz: Path,
    q_demo_rad: np.ndarray,
    q_gen_rad: np.ndarray,
    dt: float,
    n_basis: int,
    model: object,
) -> None:
    """Persist demo + rollout and enough DMP metadata to reproduce results."""
    tau = 1.0
    # Save the rollout you already computed plus the model parameters.
    np.savez(
        out_npz,
        q_demo_rad=q_demo_rad,
        q_gen_rad=q_gen_rad,
        q_demo_deg=np.degrees(q_demo_rad),
        q_gen_deg=np.degrees(q_gen_rad),
        dt=float(dt),
        tau=float(tau),
        n_basis=int(n_basis),
        alpha_canonical=float(model.alpha_canonical),
        alpha_transformation=float(model.alpha_transformation),
        beta_transformation=float(model.beta_transformation),
        centers=model.centers,
        widths=model.widths,
        weights=model.weights,
    )


def run_full_pipeline(
    trial_dir: Path,
    clean_cutoff_hz: float = 5.0,
    clean_target_dt: float | None = 0.04,
    filter_orders: list[int] | None = None,
    n_basis_list: list[int] | None = None,
) -> None:
    """Run RAW vs CLEAN sweep pipeline for a single trial directory."""
    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")

    filter_orders = filter_orders or [1, 2, 4, 6]
    n_basis_list = n_basis_list or [10, 30, 60]

    meta = _load_meta(trial_dir)
    prefix = trial_prefix(trial_dir)
    print(f"Running RAW/CLEAN sweep pipeline for trial: {trial_dir}")
    print(f"  filter_orders: {filter_orders}")
    print(f"  n_basis_list:  {n_basis_list}")

    generated_files: list[Path] = []

    def out(name: str) -> Path:
        return trial_dir / f"{prefix}{name}"

    # --- RAW: map to angles + plots ---
    print("RAW: loading sequence...")
    raw_seq, raw_t = _load_raw_seq_t(trial_dir)
    print("RAW: mapping sequence to joint angles...")
    raw_elbow_rad, raw_shoulder_rad = sequence_to_angles_rad(raw_seq)
    raw_angles_npz = out("angles_raw.npz")
    _save_angles_npz(raw_angles_npz, raw_elbow_rad, raw_shoulder_rad)
    generated_files.append(raw_angles_npz)
    raw_angles_fig = out("angles_raw.png")
    plot_angles_single(raw_elbow_rad, raw_shoulder_rad, raw_t, meta, raw_angles_fig, title_suffix="raw")
    generated_files.append(raw_angles_fig)

    raw_triplet = (raw_elbow_rad, raw_shoulder_rad, raw_t)

    # --- CLEAN variants: sweep filter_order ---
    clean_variants: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for order in filter_orders:
        print(f"CLEAN(o{order}): cleaning keypoint sequence...")
        run_clean_left_arm_sequence(
            trial_dir,
            cutoff_hz=clean_cutoff_hz,
            filter_order=order,
            target_dt=clean_target_dt,
        )
        clean_seq, clean_t = _load_clean_seq_t(trial_dir)
        print(f"CLEAN(o{order}): mapping sequence to joint angles...")
        elbow_rad, shoulder_rad = sequence_to_angles_rad(clean_seq)
        clean_variants.append((order, elbow_rad, shoulder_rad, clean_t))

        clean_angles_npz = out(f"angles_clean_o{order}.npz")
        _save_angles_npz(clean_angles_npz, elbow_rad, shoulder_rad)
        generated_files.append(clean_angles_npz)
        clean_angles_fig = out(f"angles_clean_o{order}.png")
        plot_angles_single(elbow_rad, shoulder_rad, clean_t, meta, clean_angles_fig, title_suffix=f"clean o{order}")
        generated_files.append(clean_angles_fig)

    # --- Angles overlay grid ---
    angles_overlay = out("angles_overlay_raw_vs_clean_orders.png")
    plot_angles_overlay_grid(raw_triplet, clean_variants, meta, angles_overlay)
    generated_files.append(angles_overlay)

    # --- DMP sweep ---
    # Build demos (T,4) for raw and each clean.
    q_raw = np.column_stack([raw_elbow_rad, raw_shoulder_rad])
    valid = np.all(np.isfinite(q_raw), axis=1)
    q_raw = q_raw[valid]
    if q_raw.shape[0] < 10:
        raise ValueError(f"Not enough valid raw samples for DMP fit: {q_raw.shape}")

    q_cleans: list[tuple[int, np.ndarray]] = []
    for order, elbow_rad, shoulder_rad, _t in clean_variants:
        q = np.column_stack([elbow_rad, shoulder_rad])
        valid = np.all(np.isfinite(q), axis=1)
        q = q[valid]
        if q.shape[0] < 10:
            raise ValueError(f"Not enough valid clean samples for order={order}: {q.shape}")
        q_cleans.append((order, q))

    raw_by_basis: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    clean_by_basis_order: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    for n_basis in n_basis_list:
        print(f"DMP(n_basis={n_basis}): fitting+rollout for RAW...")
        q_raw_demo, q_raw_gen, _dt, model_raw = _fit_rollout_dmp(q_raw, n_basis=n_basis)
        raw_by_basis[n_basis] = (q_raw_demo, q_raw_gen)
        raw_dmp_fig = out(f"dmp_trajectory_raw_n{n_basis}.png")
        plot_dmp_single(q_raw_demo, q_raw_gen, meta, raw_dmp_fig, title_suffix=f"raw, n_basis={n_basis}")
        generated_files.append(raw_dmp_fig)
        raw_dmp_npz = out(f"dmp_rollout_raw_n{n_basis}.npz")
        _save_dmp_rollout_npz(raw_dmp_npz, q_raw_demo, q_raw_gen, dt=_dt, n_basis=n_basis, model=model_raw)
        generated_files.append(raw_dmp_npz)

        clean_dmp_pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
        for order, q_clean in q_cleans:
            print(f"DMP(n_basis={n_basis}): fitting+rollout for CLEAN(o{order})...")
            q_clean_demo, q_clean_gen, _dt, model_clean = _fit_rollout_dmp(q_clean, n_basis=n_basis)
            clean_dmp_pairs.append((order, q_clean_demo, q_clean_gen))
            clean_by_basis_order[(n_basis, order)] = (q_clean_demo, q_clean_gen)
            clean_dmp_fig = out(f"dmp_trajectory_clean_o{order}_n{n_basis}.png")
            plot_dmp_single(
                q_clean_demo,
                q_clean_gen,
                meta,
                clean_dmp_fig,
                title_suffix=f"clean o{order}, n_basis={n_basis}",
            )
            generated_files.append(clean_dmp_fig)
            clean_dmp_npz = out(f"dmp_rollout_clean_o{order}_n{n_basis}.npz")
            _save_dmp_rollout_npz(clean_dmp_npz, q_clean_demo, q_clean_gen, dt=_dt, n_basis=n_basis, model=model_clean)
            generated_files.append(clean_dmp_npz)

        dmp_overlay_fig = out(f"dmp_overlay_raw_vs_clean_orders_n{n_basis}.png")
        plot_dmp_overlay_grid(
            raw=(q_raw_demo, q_raw_gen),
            cleans=[(order, demo, gen) for (order, demo, gen) in clean_dmp_pairs],
            meta=meta,
            out_path=dmp_overlay_fig,
            n_basis=n_basis,
        )
        generated_files.append(dmp_overlay_fig)

    # --- Per-joint sweep grids: (rows=n_basis, cols=filter_order) ---
    grid_paths = plot_dmp_order_basis_grids_per_joint(
        filter_orders=list(filter_orders),
        n_basis_list=list(n_basis_list),
        raw_by_basis=raw_by_basis,
        clean_by_basis_order=clean_by_basis_order,
        meta=meta,
        out_dir=trial_dir,
        filename_prefix=f"{prefix}",
    )
    generated_files.extend(grid_paths)

    # --- Zip bundle ---
    zip_path = trial_dir / f"{prefix}raw_clean_sweep_outputs.zip"
    print(f"Zipping outputs to {zip_path} ...")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Only zip files generated by this run (avoid bundling the whole trial dir).
        for p in sorted(set(generated_files)):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(trial_dir)))
    print("Sweep pipeline finished.")
    print(f"  - Output folder: {trial_dir}")
    print(f"  - Zip bundle:    {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAW/CLEAN sweep pipeline: angles + DMP overlays across filter orders and basis sizes."
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
        "--clean-cutoff-hz",
        type=float,
        default=5.0,
        help="Low-pass filter cutoff in Hz for cleaning sweep (default: 5.0)",
    )
    parser.add_argument(
        "--clean-target-dt",
        type=float,
        default=0.04,
        help="Resample interval in seconds, or 0 to disable resample (default: 0.04)",
    )
    parser.add_argument(
        "--filter-orders",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6],
        help="Filter orders to sweep for cleaning (default: 1 2 4 6)",
    )
    parser.add_argument(
        "--n-basis",
        type=int,
        nargs="+",
        default=[10, 30, 60],
        help="Basis function counts to sweep for DMP (default: 10 30 60)",
    )
    args = parser.parse_args()

    if args.path is not None:
        trial_dir = Path(args.path)
    else:
        trial_dir = args.data_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"

    clean_dt = args.clean_target_dt if args.clean_target_dt > 0 else None
    run_full_pipeline(
        trial_dir,
        clean_cutoff_hz=args.clean_cutoff_hz,
        clean_target_dt=clean_dt,
        filter_orders=list(args.filter_orders),
        n_basis_list=list(args.n_basis),
    )


if __name__ == "__main__":
    main()

