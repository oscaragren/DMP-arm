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
import time
from datetime import datetime, timezone
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
from kinematics.clean_angles import clean_angles_trajectory
from mapping.sequence_to_angles import sequence_to_angles_rad
from vis.plotting import (
    plot_angles_overlay_grid,
    plot_angles_single,
    plot_3d_trajectory,
    plot_dmp_order_basis_grids_per_joint,
    plot_dmp_overlay_grid,
    plot_dmp_single,
    plot_dmp_forcing_fit_single_joint,
)
from vis.plot_dmp_derivatives_fit import plot_dq_ddq_single_joint
from vis.trial_naming import trial_prefix

# DMP defaults used by this pipeline when calling dmp.fit(...)
DMP_ALPHA_CANONICAL = 4.0
DMP_ALPHA_TRANSFORMATION = 25.0
DMP_BETA_TRANSFORMATION = 6.25
DMP_TAU = 1.0

# Current internal derivative settings used inside dmp.fit(...).
# Included in run reports for traceability.
DMP_FIT_DERIVATIVE_METHOD = "savgol"
DMP_FIT_SAVGOL_WINDOW_LENGTH = 11
DMP_FIT_SAVGOL_POLYORDER = 3


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _array_stats_1d(x: np.ndarray) -> dict:
    y = np.asarray(x, dtype=float).reshape(-1)
    if y.size == 0:
        return {"size": 0}
    return {
        "size": int(y.size),
        "min": float(np.nanmin(y)),
        "max": float(np.nanmax(y)),
        "mean": float(np.nanmean(y)),
        "std": float(np.nanstd(y)),
    }


def _nanminmax(x: np.ndarray) -> tuple[float, float]:
    y = np.asarray(x, dtype=float).reshape(-1)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan"), float("nan")
    return float(np.min(y)), float(np.max(y))


def _trajectory_fit_metrics(q_demo: np.ndarray, q_gen: np.ndarray) -> dict:
    """Compute fit quality metrics for one rollout."""
    err = q_gen - q_demo
    rmse_per_joint = np.sqrt(np.mean(err**2, axis=0))
    mae_per_joint = np.mean(np.abs(err), axis=0)
    max_abs_err_per_joint = np.max(np.abs(err), axis=0)

    # R2 per joint, guarded for near-constant demos.
    denom = np.sum((q_demo - np.mean(q_demo, axis=0, keepdims=True)) ** 2, axis=0)
    sse = np.sum(err**2, axis=0)
    r2_per_joint = 1.0 - (sse / (denom + 1e-12))

    return {
        "n_samples": int(q_demo.shape[0]),
        "n_joints": int(q_demo.shape[1]),
        "rmse_rad_per_joint": [float(v) for v in rmse_per_joint],
        "rmse_deg_per_joint": [float(v) for v in np.degrees(rmse_per_joint)],
        "mae_rad_per_joint": [float(v) for v in mae_per_joint],
        "mae_deg_per_joint": [float(v) for v in np.degrees(mae_per_joint)],
        "max_abs_err_rad_per_joint": [float(v) for v in max_abs_err_per_joint],
        "max_abs_err_deg_per_joint": [float(v) for v in np.degrees(max_abs_err_per_joint)],
        "r2_per_joint": [float(v) for v in r2_per_joint],
        "rmse_rad_global": float(np.sqrt(np.mean(err**2))),
        "rmse_deg_global": float(np.degrees(np.sqrt(np.mean(err**2)))),
        "max_abs_err_rad_global": float(np.max(np.abs(err))),
        "max_abs_err_deg_global": float(np.degrees(np.max(np.abs(err)))),
    }


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
    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] < 4:
        raise ValueError(f"Expected seq shape (T, N>=4, 3), got {seq.shape}")
    t = np.load(t_path) if t_path.exists() else np.arange(seq.shape[0], dtype=np.float64)
    if t.ndim != 1 or len(t) != seq.shape[0]:
        t = np.arange(seq.shape[0], dtype=np.float64)
    t = t.astype(np.float64, copy=False)
    if t.size > 0:
        t = t - t[0]
    return seq, t


def _load_clean_seq_t(trial_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    seq_path = trial_dir / LEFT_ARM_SEQ_CLEANED
    t_path = trial_dir / LEFT_ARM_T_CLEANED
    if not seq_path.exists():
        raise FileNotFoundError(f"Not found: {seq_path}")
    seq = np.load(seq_path)
    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] < 4:
        raise ValueError(f"Expected cleaned seq shape (T, N>=4, 3), got {seq.shape}")
    t = np.load(t_path) if t_path.exists() else np.arange(seq.shape[0], dtype=np.float64)
    if t.ndim != 1 or len(t) != seq.shape[0]:
        t = np.arange(seq.shape[0], dtype=np.float64)
    t = t.astype(np.float64, copy=False)
    if t.size > 0:
        t = t - t[0]
    return seq, t


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
    #q_demo = np.deg2rad(smooth_angles_deg(np.degrees(q_demo), method="savgol"))
    T = q_demo.shape[0]
    tau = DMP_TAU
    dt = tau / (T - 1)
    model = fit(
        [q_demo],
        tau=tau,
        dt=dt,
        n_basis_functions=n_basis,
        alpha_canonical=DMP_ALPHA_CANONICAL,
        alpha_transformation=DMP_ALPHA_TRANSFORMATION,
        beta_transformation=DMP_BETA_TRANSFORMATION,
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
    tau = DMP_TAU
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
    use_keypoint_cleaning: bool = False,
    shoulder_method: str = "vector",
    ik_use_trunk_frame: bool = False,
    plot_units: str = "deg",
) -> None:
    """Run RAW vs CLEAN sweep pipeline for a single trial directory."""
    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial directory not found: {trial_dir}")

    filter_orders = filter_orders or [1, 2, 4, 6]
    n_basis_list = n_basis_list or [10, 30, 60]
    run_t0 = time.time()
    run_started_utc = _iso_utc_now()

    meta = _load_meta(trial_dir)
    prefix = trial_prefix(trial_dir)
    print(f"Running RAW/CLEAN sweep pipeline for trial: {trial_dir}")
    print(f"  filter_orders: {filter_orders}")
    print(f"  n_basis_list:  {n_basis_list}")
    print(f"  clean_mode:    {'keypoints' if use_keypoint_cleaning else 'angles'}")
    print(f"  shoulder_method: {shoulder_method}")
    print(f"  ik_use_trunk_frame: {bool(ik_use_trunk_frame)}")
    print(f"  plot_units:      {plot_units}")

    generated_files: list[Path] = []
    run_report: dict = {
        "run_started_utc": run_started_utc,
        "trial_dir": str(trial_dir.resolve()),
        "trial_prefix": prefix,
        "meta": meta,
        "config": {
            "clean_cutoff_hz": float(clean_cutoff_hz),
            "clean_target_dt": None if clean_target_dt is None else float(clean_target_dt),
            "filter_orders": [int(v) for v in filter_orders],
            "n_basis_list": [int(v) for v in n_basis_list],
            "clean_mode": "keypoints" if use_keypoint_cleaning else "angles",
            "shoulder_method": str(shoulder_method),
            "ik_use_trunk_frame": bool(ik_use_trunk_frame),
            "dmp": {
                "tau": float(DMP_TAU),
                "alpha_canonical": float(DMP_ALPHA_CANONICAL),
                "alpha_transformation": float(DMP_ALPHA_TRANSFORMATION),
                "beta_transformation": float(DMP_BETA_TRANSFORMATION),
                "fit_derivative_method_internal": DMP_FIT_DERIVATIVE_METHOD,
                "fit_savgol_window_length_internal": int(DMP_FIT_SAVGOL_WINDOW_LENGTH),
                "fit_savgol_polyorder_internal": int(DMP_FIT_SAVGOL_POLYORDER),
            },
        },
        "inputs": {},
        "angles_stage": {"raw": {}, "clean_variants": {}},
        "dmp_stage": {"raw_by_basis": {}, "clean_by_basis_order": {}},
        "artifacts": {},
    }

    def out(name: str) -> Path:
        return trial_dir / f"{prefix}{name}"

    # --- RAW: map to angles + plots ---
    print("RAW: loading sequence...")
    raw_seq, raw_t = _load_raw_seq_t(trial_dir)
    run_report["inputs"]["raw_sequence_shape"] = [int(v) for v in raw_seq.shape]
    run_report["inputs"]["raw_t_stats"] = _array_stats_1d(raw_t)
    print("RAW: plotting keypoint trajectories (3D + position vs time)...")
    raw_keypoints_fig = out("keypoints_raw_trajectory.png")
    plot_3d_trajectory(raw_seq, raw_t, meta, raw_keypoints_fig)
    generated_files.append(raw_keypoints_fig)
    print("RAW: mapping sequence to joint angles...")
    raw_elbow_rad, raw_shoulder_rad = sequence_to_angles_rad(
        raw_seq,
        shoulder_method=shoulder_method,
        ik_use_trunk_frame=bool(ik_use_trunk_frame),
    )
    raw_e_min, raw_e_max = _nanminmax(raw_elbow_rad)
    print(
        f"RAW: elbow flexion range: "
        f"{raw_e_min:.4f}–{raw_e_max:.4f} rad "
        f"({np.degrees(raw_e_min):.1f}–{np.degrees(raw_e_max):.1f} deg)"
    )
    run_report["angles_stage"]["raw"] = {
        "elbow_shape": [int(v) for v in raw_elbow_rad.shape],
        "shoulder_shape": [int(v) for v in raw_shoulder_rad.shape],
        "elbow_rad_stats": _array_stats_1d(raw_elbow_rad),
        "shoulder_rad_stats": _array_stats_1d(raw_shoulder_rad),
    }
    raw_angles_npz = out("angles_raw.npz")
    _save_angles_npz(raw_angles_npz, raw_elbow_rad, raw_shoulder_rad)
    generated_files.append(raw_angles_npz)
    raw_angles_fig = out("angles_raw.png")
    plot_angles_single(
        raw_elbow_rad,
        raw_shoulder_rad,
        raw_t,
        meta,
        raw_angles_fig,
        title_suffix="raw",
        units=plot_units,
    )
    generated_files.append(raw_angles_fig)

    raw_triplet = (raw_elbow_rad, raw_shoulder_rad, raw_t)

    # --- CLEAN variants: sweep filter_order ---
    clean_variants: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for order in filter_orders:
        if use_keypoint_cleaning:
            print(f"CLEAN(o{order}): cleaning keypoint sequence...")
            run_clean_left_arm_sequence(
                trial_dir,
                cutoff_hz=clean_cutoff_hz,
                filter_order=order,
                target_dt=clean_target_dt,
            )
            clean_seq, clean_t = _load_clean_seq_t(trial_dir)
            print(f"CLEAN(o{order}): mapping sequence to joint angles...")
            elbow_rad, shoulder_rad = sequence_to_angles_rad(
                clean_seq,
                shoulder_method=shoulder_method,
                ik_use_trunk_frame=bool(ik_use_trunk_frame),
            )
        else:
            print(f"CLEAN(o{order}): cleaning in joint-angle space...")
            elbow_rad, shoulder_rad, clean_t = clean_angles_trajectory(
                raw_elbow_rad,
                raw_shoulder_rad,
                raw_t,
                cutoff_hz=clean_cutoff_hz,
                filter_order=order,
                target_dt=clean_target_dt,
            )
        clean_e_min, clean_e_max = _nanminmax(elbow_rad)
        print(
            f"CLEAN(o{order}): elbow flexion range: "
            f"{clean_e_min:.4f}–{clean_e_max:.4f} rad "
            f"({np.degrees(clean_e_min):.1f}–{np.degrees(clean_e_max):.1f} deg)"
        )
        clean_variants.append((order, elbow_rad, shoulder_rad, clean_t))
        run_report["angles_stage"]["clean_variants"][str(order)] = {
            "elbow_shape": [int(v) for v in elbow_rad.shape],
            "shoulder_shape": [int(v) for v in shoulder_rad.shape],
            "t_stats": _array_stats_1d(clean_t),
            "elbow_rad_stats": _array_stats_1d(elbow_rad),
            "shoulder_rad_stats": _array_stats_1d(shoulder_rad),
        }

        clean_angles_npz = out(f"angles_clean_o{order}.npz")
        _save_angles_npz(clean_angles_npz, elbow_rad, shoulder_rad)
        generated_files.append(clean_angles_npz)
        clean_angles_fig = out(f"angles_clean_o{order}.png")
        plot_angles_single(
            elbow_rad,
            shoulder_rad,
            clean_t,
            meta,
            clean_angles_fig,
            title_suffix=f"clean o{order}",
            units=plot_units,
        )
        generated_files.append(clean_angles_fig)

    # --- Angles overlay grid ---
    angles_overlay = out("angles_overlay_raw_vs_clean_orders.png")
    plot_angles_overlay_grid(raw_triplet, clean_variants, meta, angles_overlay, units=plot_units)
    generated_files.append(angles_overlay)

    # --- DMP sweep ---
    # Build demos (T,4) for raw and each clean.
    q_raw = np.column_stack([raw_elbow_rad, raw_shoulder_rad])
    valid = np.all(np.isfinite(q_raw), axis=1)
    run_report["dmp_stage"]["raw_valid_samples_before"] = int(q_raw.shape[0])
    q_raw = q_raw[valid]
    run_report["dmp_stage"]["raw_valid_samples_after"] = int(q_raw.shape[0])
    if q_raw.shape[0] < 10:
        raise ValueError(f"Not enough valid raw samples for DMP fit: {q_raw.shape}")

    q_cleans: list[tuple[int, np.ndarray]] = []
    for order, elbow_rad, shoulder_rad, _t in clean_variants:
        q = np.column_stack([elbow_rad, shoulder_rad])
        valid = np.all(np.isfinite(q), axis=1)
        n_before = int(q.shape[0])
        q = q[valid]
        run_report["dmp_stage"]["clean_by_basis_order"].setdefault(str(order), {})
        run_report["dmp_stage"]["clean_by_basis_order"][str(order)]["valid_samples_before"] = n_before
        run_report["dmp_stage"]["clean_by_basis_order"][str(order)]["valid_samples_after"] = int(q.shape[0])
        if q.shape[0] < 10:
            raise ValueError(f"Not enough valid clean samples for order={order}: {q.shape}")
        q_cleans.append((order, q))

    raw_by_basis: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    clean_by_basis_order: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    for n_basis in n_basis_list:
        print(f"DMP(n_basis={n_basis}): fitting+rollout for RAW...")
        q_raw_demo, q_raw_gen, _dt, model_raw = _fit_rollout_dmp(q_raw, n_basis=n_basis)
        run_report["dmp_stage"]["raw_by_basis"][str(n_basis)] = {
            "dt": float(_dt),
            "tau": float(model_raw.tau),
            "n_basis": int(n_basis),
            "alpha_canonical": float(model_raw.alpha_canonical),
            "alpha_transformation": float(model_raw.alpha_transformation),
            "beta_transformation": float(model_raw.beta_transformation),
            "metrics": _trajectory_fit_metrics(q_raw_demo, q_raw_gen),
        }
        raw_by_basis[n_basis] = (q_raw_demo, q_raw_gen)
        raw_dmp_fig = out(f"dmp_trajectory_raw_n{n_basis}.png")
        plot_dmp_single(
            np.degrees(q_raw_demo),
            np.degrees(q_raw_gen),
            meta,
            raw_dmp_fig,
            title_suffix=f"raw, n_basis={n_basis}",
        )
        generated_files.append(raw_dmp_fig)
        raw_dmp_npz = out(f"dmp_rollout_raw_n{n_basis}.npz")
        _save_dmp_rollout_npz(raw_dmp_npz, q_raw_demo, q_raw_gen, dt=_dt, n_basis=n_basis, model=model_raw)
        generated_files.append(raw_dmp_npz)

        # Additional diagnostics: forcing fit + derivatives, one file per joint.
        for joint_idx in range(q_raw_demo.shape[1]):
            raw_force_fig = out(f"dmp_forcing_fit_raw_joint{joint_idx}_n{n_basis}.png")
            plot_dmp_forcing_fit_single_joint(
                q_demo=q_raw_demo,
                joint_idx=joint_idx,
                out_path=raw_force_fig,
                n_basis=n_basis,
                tau=DMP_TAU,
                alpha_canonical=DMP_ALPHA_CANONICAL,
                alpha_transformation=DMP_ALPHA_TRANSFORMATION,
                beta_transformation=DMP_BETA_TRANSFORMATION,
                derivative_method=DMP_FIT_DERIVATIVE_METHOD,
                savgol_window_length=DMP_FIT_SAVGOL_WINDOW_LENGTH,
                savgol_polyorder=DMP_FIT_SAVGOL_POLYORDER,
                meta=meta,
                title_suffix=f"raw, n_basis={n_basis}",
            )
            generated_files.append(raw_force_fig)

            raw_deriv_fig = out(f"dmp_derivatives_fit_raw_joint{joint_idx}_n{n_basis}.png")
            plot_dq_ddq_single_joint(
                q_demo=q_raw_demo,
                joint_idx=joint_idx,
                out_path=raw_deriv_fig,
                n_basis=n_basis,
                tau=DMP_TAU,
                alpha_canonical=DMP_ALPHA_CANONICAL,
                alpha_transformation=DMP_ALPHA_TRANSFORMATION,
                beta_transformation=DMP_BETA_TRANSFORMATION,
                derivative_method=DMP_FIT_DERIVATIVE_METHOD,
                savgol_window_length=DMP_FIT_SAVGOL_WINDOW_LENGTH,
                savgol_polyorder=DMP_FIT_SAVGOL_POLYORDER,
                meta=meta,
                title_suffix=f"raw, n_basis={n_basis}",
            )
            generated_files.append(raw_deriv_fig)

        clean_dmp_pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
        for order, q_clean in q_cleans:
            print(f"DMP(n_basis={n_basis}): fitting+rollout for CLEAN(o{order})...")
            q_clean_demo, q_clean_gen, _dt, model_clean = _fit_rollout_dmp(q_clean, n_basis=n_basis)
            run_report["dmp_stage"]["clean_by_basis_order"][str(order)][str(n_basis)] = {
                "dt": float(_dt),
                "tau": float(model_clean.tau),
                "n_basis": int(n_basis),
                "alpha_canonical": float(model_clean.alpha_canonical),
                "alpha_transformation": float(model_clean.alpha_transformation),
                "beta_transformation": float(model_clean.beta_transformation),
                "metrics": _trajectory_fit_metrics(q_clean_demo, q_clean_gen),
            }
            clean_dmp_pairs.append((order, q_clean_demo, q_clean_gen))
            clean_by_basis_order[(n_basis, order)] = (q_clean_demo, q_clean_gen)
            clean_dmp_fig = out(f"dmp_trajectory_clean_o{order}_n{n_basis}.png")
            plot_dmp_single(
                np.degrees(q_clean_demo),
                np.degrees(q_clean_gen),
                meta,
                clean_dmp_fig,
                title_suffix=f"clean o{order}, n_basis={n_basis}",
            )
            generated_files.append(clean_dmp_fig)
            clean_dmp_npz = out(f"dmp_rollout_clean_o{order}_n{n_basis}.npz")
            _save_dmp_rollout_npz(clean_dmp_npz, q_clean_demo, q_clean_gen, dt=_dt, n_basis=n_basis, model=model_clean)
            generated_files.append(clean_dmp_npz)

            for joint_idx in range(q_clean_demo.shape[1]):
                clean_force_fig = out(f"dmp_forcing_fit_clean_o{order}_joint{joint_idx}_n{n_basis}.png")
                plot_dmp_forcing_fit_single_joint(
                    q_demo=q_clean_demo,
                    joint_idx=joint_idx,
                    out_path=clean_force_fig,
                    n_basis=n_basis,
                    tau=DMP_TAU,
                    alpha_canonical=DMP_ALPHA_CANONICAL,
                    alpha_transformation=DMP_ALPHA_TRANSFORMATION,
                    beta_transformation=DMP_BETA_TRANSFORMATION,
                    derivative_method=DMP_FIT_DERIVATIVE_METHOD,
                    savgol_window_length=DMP_FIT_SAVGOL_WINDOW_LENGTH,
                    savgol_polyorder=DMP_FIT_SAVGOL_POLYORDER,
                    meta=meta,
                    title_suffix=f"clean o{order}, n_basis={n_basis}",
                )
                generated_files.append(clean_force_fig)

                clean_deriv_fig = out(f"dmp_derivatives_fit_clean_o{order}_joint{joint_idx}_n{n_basis}.png")
                plot_dq_ddq_single_joint(
                    q_demo=q_clean_demo,
                    joint_idx=joint_idx,
                    out_path=clean_deriv_fig,
                    n_basis=n_basis,
                    tau=DMP_TAU,
                    alpha_canonical=DMP_ALPHA_CANONICAL,
                    alpha_transformation=DMP_ALPHA_TRANSFORMATION,
                    beta_transformation=DMP_BETA_TRANSFORMATION,
                    derivative_method=DMP_FIT_DERIVATIVE_METHOD,
                    savgol_window_length=DMP_FIT_SAVGOL_WINDOW_LENGTH,
                    savgol_polyorder=DMP_FIT_SAVGOL_POLYORDER,
                    meta=meta,
                    title_suffix=f"clean o{order}, n_basis={n_basis}",
                )
                generated_files.append(clean_deriv_fig)

        dmp_overlay_fig = out(f"dmp_overlay_raw_vs_clean_orders_n{n_basis}.png")
        plot_dmp_overlay_grid(
            raw=(q_raw_demo, q_raw_gen),
            cleans=[(order, demo, gen) for (order, demo, gen) in clean_dmp_pairs],
            meta=meta,
            out_path=dmp_overlay_fig,
            n_basis=n_basis,
            units=plot_units,
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
        units=plot_units,
    )
    generated_files.extend(grid_paths)

    # --- Zip bundle ---
    zip_path = trial_dir / f"{prefix}raw_clean_sweep_outputs.zip"
    report_path = trial_dir / f"{prefix}run_report.json"
    run_report["run_finished_utc"] = _iso_utc_now()
    run_report["duration_seconds"] = float(time.time() - run_t0)
    run_report["artifacts"] = {
        "zip_path": str(zip_path),
        "report_path": str(report_path),
        "generated_files": [
            str(p.relative_to(trial_dir)) if p.is_absolute() else str(p)
            for p in sorted(set(generated_files))
        ],
        "generated_file_count": int(len(set(generated_files))),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)
    generated_files.append(report_path)

    print(f"Zipping outputs to {zip_path} ...")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Only zip files generated by this run (avoid bundling the whole trial dir).
        for p in sorted(set(generated_files)):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(trial_dir)))
    print("Sweep pipeline finished.")
    print(f"  - Output folder: {trial_dir}")
    print(f"  - Zip bundle:    {zip_path}")
    print(f"  - Run report:    {report_path}")


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
        default=[10, 30, 60, 100],
        help="Basis function counts to sweep for DMP (default: 10 30 60 100)",
    )
    parser.add_argument(
        "--use-keypoint-cleaning",
        action="store_true",
        help="Use capture/clean_keypoints.py logic (clean in keypoint space before angle mapping).",
    )
    parser.add_argument(
        "--shoulder-method",
        type=str,
        default="vector",
        choices=["vector", "rotmat", "ik"],
        help="Shoulder angle method: 'vector' (legacy), 'rotmat' (uses hand direction if available), or 'ik' (optimization-based IK + constraints).",
    )
    parser.add_argument(
        "--ik-use-trunk-frame",
        action="store_true",
        help="When --shoulder-method ik: transform [shoulder, elbow, wrist] into a per-frame trunk frame (built from left/right shoulders) before solving IK.",
    )
    parser.add_argument(
        "--plot-units",
        type=str,
        default="deg",
        choices=["deg", "rad"],
        help="Units for plots only: 'deg' or 'rad'. (Saved trajectories remain in radians in NPZ.)",
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
        use_keypoint_cleaning=bool(args.use_keypoint_cleaning),
        shoulder_method=str(args.shoulder_method),
        ik_use_trunk_frame=bool(args.ik_use_trunk_frame),
        plot_units=str(args.plot_units),
    )


if __name__ == "__main__":
    main()

