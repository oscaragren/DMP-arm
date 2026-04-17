"""
Central plotting utilities for the project.

This module consolidates all visualization code under `vis/` into a single file,
so other scripts can import a stable set of plotting functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from capture.clean_keypoints import LEFT_ARM_SEQ_CLEANED, LEFT_ARM_T_CLEANED
from dmp.dmp import (
    DMPModel,
    estimate_derivatives,
    fit,
    rollout_simple,
    canonical_phase,
)
from kinematics.joint_dynamics import smooth_angles_deg, validate_joint_trajectory_deg
from kinematics.simple_kinematics import get_angles as get_left_arm_angles_deg
from vis.trial_naming import trial_prefix


# -------------------------
# Generic helpers
# -------------------------


def _load_meta(trial_dir: Path) -> dict:
    meta_path = trial_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _ensure_2d(a: np.ndarray) -> np.ndarray:
    return a[:, None] if a.ndim == 1 else a


def _angles_in_units(x_rad: np.ndarray, units: str) -> np.ndarray:
    u = (units or "deg").strip().lower()
    if u not in {"deg", "rad"}:
        raise ValueError(f"Unknown units '{units}'. Use 'deg' or 'rad'.")
    return np.degrees(x_rad) if u == "deg" else x_rad


def _angle_ylabel(units: str) -> str:
    return "Angle (deg)" if (units or "deg").strip().lower() == "deg" else "Angle (rad)"


# -------------------------
# Angles plots (single + overlays)
# -------------------------


def plot_angles_single(
    elbow_rad: np.ndarray,
    shoulder_rad: np.ndarray,
    t: np.ndarray,
    meta: dict,
    out_path: Path,
    title_suffix: str,
    *,
    units: str = "deg",
) -> None:
    """
    Plot the angles for the left arm.
    Args:
        elbow_rad: np.ndarray: the elbow flexion angles in radians
        shoulder_rad: np.ndarray: the shoulder angles in radians
        t: np.ndarray: the time in seconds
        meta: dict: the metadata
        out_path: Path: the path to save the plot
        title_suffix: str: the suffix for the title
        units: str: the units to use for the plot
    """
    elbow_u = _angles_in_units(elbow_rad, units)
    shoulder_u = _ensure_2d(_angles_in_units(shoulder_rad, units))
    ylab = _angle_ylabel(units)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    valid = np.isfinite(elbow_u)
    if np.any(valid):
        ax.plot(t[valid], elbow_u[valid], color="#3498db", linewidth=1.5, label="Elbow flexion")
    ax.set_ylabel(ylab)
    ax.set_title("Elbow flexion (upper arm–forearm angle)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 200 if (units or "deg").strip().lower() == "deg" else np.deg2rad(200))

    ax = axes[1]
    labels = ["Flexion", "Abduction", "Internal rotation"]
    colors = ["#e74c3c", "#2ecc71", "#9b59b6"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        if i >= shoulder_u.shape[1]:
            continue
        vals = shoulder_u[:, i]
        valid = np.isfinite(vals)
        if np.any(valid):
            ax.plot(t[valid], vals[valid], color=color, linewidth=1.5, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylab)
    ax.set_title("Shoulder 3-DOF (flexion, abduction, internal rotation)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    subject = meta.get("subject", "?")
    motion = meta.get("motion", "?")
    trial = meta.get("trial", "?")
    fig.suptitle(f"Left arm angles — subject {subject}, {motion}, trial {trial} ({title_suffix})", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_angles_overlay_grid(
    raw: tuple[np.ndarray, np.ndarray, np.ndarray],
    cleans: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    meta: dict,
    out_path: Path,
    *,
    units: str = "deg",
) -> None:
    """Rows: raw vs each clean order + 'all together'. Cols: elbow + shoulder."""
    elbow_raw_rad, shoulder_raw_rad, t_raw = raw
    elbow_raw_u = _angles_in_units(elbow_raw_rad, units)
    shoulder_raw_u = _ensure_2d(_angles_in_units(shoulder_raw_rad, units))
    ylab = _angle_ylabel(units)

    rows = len(cleans) + 1
    fig, axes = plt.subplots(rows, 2, figsize=(14, 2.8 * rows), sharex=False)
    if rows == 1:
        axes = np.array([axes])

    # Per-clean comparisons
    for r, (order, elbow_clean_rad, shoulder_clean_rad, t_clean) in enumerate(cleans):
        elbow_clean_u = _angles_in_units(elbow_clean_rad, units)
        shoulder_clean_u = _ensure_2d(_angles_in_units(shoulder_clean_rad, units))

        ax = axes[r, 0]
        valid = np.isfinite(elbow_raw_u)
        if np.any(valid):
            ax.plot(t_raw[valid], elbow_raw_u[valid], color="#3498db", linewidth=1.2, label="raw")
        valid = np.isfinite(elbow_clean_u)
        if np.any(valid):
            ax.plot(
                t_clean[valid],
                elbow_clean_u[valid],
                color="#2980b9",
                linestyle="--",
                linewidth=1.2,
                label=f"clean o{order}",
            )
        ax.set_title(f"Elbow flexion — raw vs clean (filter_order={order})")
        ax.set_xlabel("Time (s) (or index if timestamps missing)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 200 if (units or "deg").strip().lower() == "deg" else np.deg2rad(200))
        ax.legend(loc="upper right", fontsize=8)

        ax = axes[r, 1]
        labels = ["Flex", "Abd", "IR"]
        raw_colors = ["#e74c3c", "#2ecc71", "#9b59b6"]
        clean_colors = ["#c0392b", "#27ae60", "#8e44ad"]
        for i, lab in enumerate(labels):
            if i < shoulder_raw_u.shape[1]:
                vals = shoulder_raw_u[:, i]
                valid = np.isfinite(vals)
                if np.any(valid):
                    ax.plot(t_raw[valid], vals[valid], color=raw_colors[i], linewidth=1.1, label=f"{lab} raw")
            if i < shoulder_clean_u.shape[1]:
                vals = shoulder_clean_u[:, i]
                valid = np.isfinite(vals)
                if np.any(valid):
                    ax.plot(
                        t_clean[valid],
                        vals[valid],
                        color=clean_colors[i],
                        linestyle="--",
                        linewidth=1.1,
                        label=f"{lab} clean",
                    )
        ax.set_title(f"Shoulder 3-DOF — raw vs clean (filter_order={order})")
        ax.set_xlabel("Time (s) (or index if timestamps missing)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", ncols=2, fontsize=7)

    # All together row
    r = rows - 1
    ax = axes[r, 0]
    valid = np.isfinite(elbow_raw_u)
    if np.any(valid):
        ax.plot(t_raw[valid], elbow_raw_u[valid], color="#3498db", linewidth=1.2, label="raw")
    for order, elbow_clean_rad, _, t_clean in cleans:
        elbow_clean_u = _angles_in_units(elbow_clean_rad, units)
        valid = np.isfinite(elbow_clean_u)
        if np.any(valid):
            ax.plot(t_clean[valid], elbow_clean_u[valid], linestyle="--", linewidth=1.0, label=f"clean o{order}")
    ax.set_title("Elbow flexion — raw vs all clean variants")
    ax.set_xlabel("Time (s) (or index if timestamps missing)")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100 if (units or "deg").strip().lower() == "deg" else np.deg2rad(100))
    ax.legend(loc="upper right", ncols=3, fontsize=8)

    ax = axes[r, 1]
    labels = ["Flex", "Abd", "IR"]
    raw_colors = ["#e74c3c", "#2ecc71", "#9b59b6"]
    for i, lab in enumerate(labels):
        if i < shoulder_raw_u.shape[1]:
            vals = shoulder_raw_u[:, i]
            valid = np.isfinite(vals)
            if np.any(valid):
                ax.plot(t_raw[valid], vals[valid], color=raw_colors[i], linewidth=1.2, label=f"{lab} raw")
    for order, _, shoulder_clean_rad, t_clean in cleans:
        shoulder_clean_u = _ensure_2d(_angles_in_units(shoulder_clean_rad, units))
        for i, lab in enumerate(labels):
            if i >= shoulder_clean_u.shape[1]:
                continue
            vals = shoulder_clean_u[:, i]
            valid = np.isfinite(vals)
            if np.any(valid):
                ax.plot(t_clean[valid], vals[valid], linestyle="--", linewidth=1.0, label=f"{lab} clean o{order}")
    ax.set_title("Shoulder 3-DOF — raw vs all clean variants")
    ax.set_xlabel("Time (s) (or index if timestamps missing)")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", ncols=3, fontsize=7)

    subject = meta.get("subject", "?")
    motion = meta.get("motion", "?")
    trial = meta.get("trial", "?")
    fig.suptitle(f"Angles overlay grid — subject {subject}, {motion}, trial {trial}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


# -------------------------
# DMP plots (single + overlay grid)
# -------------------------


def plot_dmp_single(
    q_demo: np.ndarray,
    q_gen: np.ndarray,
    meta: dict,
    out_path: Path,
    title_suffix: str,
) -> None:
    """

    Args:
        q_demo: np.ndarray: the demonstrated trajectory (degrees)
        q_gen: np.ndarray: the generated trajectory (degrees)
        meta: dict: the metadata
        out_path: Path: the path to save the plot
        title_suffix: str: the suffix for the title
    """
    joint_names = [
        "Elbow flexion",
        "Shoulder flexion",
        "Shoulder abduction",
        "Shoulder internal rotation",
    ]
    colors_demo = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    colors_gen = ["#2980b9", "#c0392b", "#27ae60", "#8e44ad"]

    tau = 1.0
    t_demo = np.linspace(0, tau, q_demo.shape[0])
    t_gen = np.linspace(0, tau, q_gen.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()
    ylab = "Angle (deg)"
    for j in range(4):
        ax = axes[j]
        ax.plot(t_demo, q_demo[:, j], color=colors_demo[j], linewidth=1.5, label="Demo")
        ax.plot(t_gen, q_gen[:, j], color=colors_gen[j], linestyle="--", linewidth=1.2, label="DMP")
        ax.set_ylabel(ylab)
        ax.set_title(joint_names[j])
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (normalized)")

    subject = meta.get("subject", "?")
    motion = meta.get("motion", "?")
    trial = meta.get("trial", "?")
    fig.suptitle(f"DMP trajectory — subject {subject}, {motion}, trial {trial} ({title_suffix})", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_dmp_overlay_grid(
    raw: tuple[np.ndarray, np.ndarray],
    cleans: list[tuple[int, np.ndarray, np.ndarray]],
    meta: dict,
    out_path: Path,
    n_basis: int,
    *,
    units: str = "deg",
) -> None:
    """Rows: raw vs each clean + all-together. Cols: 4 joints."""
    q_raw_demo, q_raw_gen = raw
    rows = len(cleans) + 1
    cols = 4

    t_raw_demo = np.linspace(0, 1.0, q_raw_demo.shape[0])
    t_raw_gen = np.linspace(0, 1.0, q_raw_gen.shape[0])

    joint_names = ["Elbow flexion", "Shoulder flexion", "Shoulder abduction", "Shoulder internal rotation"]

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.3 * rows), sharex=True)
    if rows == 1:
        axes = np.array([axes])

    def _plot_pair(ax, t1, y1, t2, y2, label1, label2, color1, color2):
        ax.plot(t1, y1, color=color1, linewidth=1.2, label=label1)
        ax.plot(t2, y2, color=color2, linewidth=1.1, linestyle="--", label=label2)

    ylab = _angle_ylabel(units)
    for r, (order, q_clean_demo, q_clean_gen) in enumerate(cleans):
        t_clean_demo = np.linspace(0, 1.0, q_clean_demo.shape[0])
        t_clean_gen = np.linspace(0, 1.0, q_clean_gen.shape[0])
        for j in range(4):
            ax = axes[r, j]
            _plot_pair(
                ax,
                t_raw_demo,
                _angles_in_units(q_raw_demo[:, j], units),
                t_raw_gen,
                _angles_in_units(q_raw_gen[:, j], units),
                "raw demo",
                "raw dmp",
                "#34495e",
                "#2c3e50",
            )
            _plot_pair(
                ax,
                t_clean_demo,
                _angles_in_units(q_clean_demo[:, j], units),
                t_clean_gen,
                _angles_in_units(q_clean_gen[:, j], units),
                f"clean o{order} demo",
                f"clean o{order} dmp",
                "#8e44ad",
                "#6c3483",
            )
            ax.set_title(f"{joint_names[j]} — raw vs clean (o{order})")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(ylab)
            ax.set_xlabel("Time (normalized)")
            ax.legend(loc="upper right", fontsize=6, framealpha=0.9)

    r = rows - 1
    for j in range(4):
        ax = axes[r, j]
        ax.plot(t_raw_demo, _angles_in_units(q_raw_demo[:, j], units), color="#34495e", linewidth=1.2, label="raw demo")
        ax.plot(t_raw_gen, _angles_in_units(q_raw_gen[:, j], units), color="#2c3e50", linestyle="--", linewidth=1.1, label="raw dmp")
        for order, q_clean_demo, q_clean_gen in cleans:
            t_clean_demo = np.linspace(0, 1.0, q_clean_demo.shape[0])
            t_clean_gen = np.linspace(0, 1.0, q_clean_gen.shape[0])
            ax.plot(t_clean_demo, _angles_in_units(q_clean_demo[:, j], units), linewidth=1.0, label=f"clean o{order} demo")
            ax.plot(t_clean_gen, _angles_in_units(q_clean_gen[:, j], units), linestyle="--", linewidth=0.9, label=f"clean o{order} dmp")
        ax.set_title(f"{joint_names[j]} — raw vs all cleans")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Time (normalized)")
        ax.legend(loc="upper right", fontsize=6, framealpha=0.9)

    subject = meta.get("subject", "?")
    motion = meta.get("motion", "?")
    trial = meta.get("trial", "?")
    fig.suptitle(
        f"DMP overlay grid (n_basis={n_basis}) — subject {subject}, {motion}, trial {trial}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_dmp_order_basis_grids_per_joint(
    *,
    filter_orders: list[int],
    n_basis_list: list[int],
    raw_by_basis: dict[int, tuple[np.ndarray, np.ndarray]],
    clean_by_basis_order: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    meta: dict,
    out_dir: Path,
    filename_prefix: str = "",
    n_time_points: int = 200,
    units: str = "deg",
) -> list[Path]:
    """
    Create 4 figures (one per joint). Each figure is a grid:

    - columns: filter_orders (cleaning filter order)
    - rows:    n_basis_list (DMP basis functions)

    Each cell overlays RAW (demo + DMP) with CLEAN(oX) (demo + DMP) for the same n_basis.

    Returns list of written figure paths (length 4).
    """
    if n_time_points < 2:
        raise ValueError("n_time_points must be >= 2")

    t_common = np.linspace(0.0, 1.0, int(n_time_points))

    def _resample_1d(y: np.ndarray) -> np.ndarray:
        """Resample 1D signal y(t) onto t_common, using linear interpolation in normalized time."""
        if y.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {y.shape}")
        if y.size < 2:
            return np.full_like(t_common, np.nan, dtype=np.float64)
        t_src = np.linspace(0.0, 1.0, y.size)
        return np.interp(t_common, t_src, y).astype(np.float64, copy=False)

    joint_names = [
        "Elbow flexion",
        "Shoulder flexion",
        "Shoulder abduction",
        "Shoulder internal rotation",
    ]

    subject = meta.get("subject", "?")
    motion = meta.get("motion", "?")
    trial = meta.get("trial", "?")

    out_paths: list[Path] = []
    n_rows = len(n_basis_list)
    n_cols = len(filter_orders)
    ylab = _angle_ylabel(units)

    for j, joint_name in enumerate(joint_names):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.8 * n_cols, 2.8 * n_rows),
            sharex=True,
            sharey=True,
        )
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes[:, None]

        for r, n_basis in enumerate(n_basis_list):
            if n_basis not in raw_by_basis:
                continue
            q_raw_demo, q_raw_gen = raw_by_basis[n_basis]
            raw_demo_j = _resample_1d(_angles_in_units(q_raw_demo[:, j], units))
            raw_gen_j = _resample_1d(_angles_in_units(q_raw_gen[:, j], units))

            for c, order in enumerate(filter_orders):
                ax = axes[r, c]
                ax.plot(
                    t_common,
                    raw_demo_j,
                    color="#34495e",
                    linewidth=1.1,
                    label="raw demo",
                )
                ax.plot(
                    t_common,
                    raw_gen_j,
                    color="#2c3e50",
                    linestyle="--",
                    linewidth=1.0,
                    label="raw dmp",
                )

                key = (n_basis, order)
                if key in clean_by_basis_order:
                    q_clean_demo, q_clean_gen = clean_by_basis_order[key]
                    clean_demo_j = _resample_1d(_angles_in_units(q_clean_demo[:, j], units))
                    clean_gen_j = _resample_1d(_angles_in_units(q_clean_gen[:, j], units))
                    ax.plot(
                        t_common,
                        clean_demo_j,
                        color="#8e44ad",
                        linewidth=1.0,
                        label=f"clean o{order} demo",
                    )
                    ax.plot(
                        t_common,
                        clean_gen_j,
                        color="#6c3483",
                        linestyle="--",
                        linewidth=0.9,
                        label=f"clean o{order} dmp",
                    )

                ax.grid(True, alpha=0.25)
                if r == 0:
                    ax.set_title(f"o{order}", fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"n={n_basis}\n{ylab}")
                if r == n_rows - 1:
                    ax.set_xlabel("Time (normalized)")

                # Keep legends from cluttering: show once in top-left subplot.
                if r == 0 and c == 0:
                    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

        fig.suptitle(
            f"DMP sweep grid — {joint_name}\nsubject {subject}, {motion}, trial {trial} (cols=filter_order, rows=n_basis)",
            fontsize=12,
        )
        plt.tight_layout()

        safe_joint = (
            joint_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        out_path = out_dir / f"{filename_prefix}dmp_grid_orders_vs_basis_{safe_joint}.png"
        plt.savefig(out_path, dpi=140)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


# -------------------------
# Trial-level plotting CLIs (kept here to avoid multiple vis plotting files)
# -------------------------


def load_trial_left_arm_sequence(trial_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load left-arm sequence and time. Prefers cleaned arrays when present."""
    seq_path = trial_dir / LEFT_ARM_SEQ_CLEANED
    if not seq_path.exists():
        seq_path = trial_dir / "left_arm_seq_camera.npy"
    t_path = trial_dir / LEFT_ARM_T_CLEANED
    if not t_path.exists():
        t_path = trial_dir / "left_arm_t.npy"

    if not seq_path.exists():
        raise FileNotFoundError(f"Not found: {seq_path}")
    seq = np.load(seq_path)
    if seq.ndim != 3 or seq.shape[2] != 3 or seq.shape[1] < 4:
        raise ValueError(f"Expected seq shape (T, N>=4, 3), got {seq.shape}")
    t = np.load(t_path) if t_path.exists() else np.arange(seq.shape[0], dtype=np.float64)
    return seq, t, _load_meta(trial_dir)


def plot_left_arm_angles_from_trial(trial_dir: Path, out_path: Path | None = None) -> Path:
    """Load trial sequence, compute angles, plot, and save angles.npz alongside."""
    seq, t, meta = load_trial_left_arm_sequence(trial_dir)
    if seq.shape[1] < 6:
        raise ValueError(
            "Expected seq with N>=6 keypoints for trunk frame (LS, LE, LW, RS, LH, RH) "
            f"when using kinematics/simple_kinematics.py, got {seq.shape}. "
            "Re-record/re-process to include hips, or update the kinematics method to support N=4."
        )

    angles_deg = get_left_arm_angles_deg(seq)  # (T,4) deg: [elbow, sh_flex, sh_abd, sh_lat/med_rot]
    elbow_rad = np.deg2rad(angles_deg[:, 0])
    shoulder_rad = np.deg2rad(angles_deg[:, 1:4])
    elbow_deg = np.degrees(elbow_rad)
    shoulder_deg = np.degrees(shoulder_rad)

    if out_path is None:
        out_path = trial_dir / f"{trial_prefix(trial_dir)}angles.png"
    plot_angles_single(elbow_rad, shoulder_rad, t, meta, out_path, title_suffix="auto")

    prefix = ""
    try:
        prefix = trial_prefix(out_path.parent)
    except Exception:
        prefix = ""
    npz_path = out_path.parent / f"{prefix}angles.npz" if prefix else (out_path.parent / "angles.npz")
    np.savez(
        npz_path,
        elbow_rad=elbow_rad,
        shoulder_rad=shoulder_rad,
        elbow_deg=elbow_deg,
        shoulder_deg=shoulder_deg,
    )
    return out_path

def load_angles_demo(trial_dir: Path, source: str = "auto") -> np.ndarray:
    """Load elbow+shoulder angles; return (T,4) radians, finite rows only."""
    if source not in {"auto", "raw", "clean"}:
        raise ValueError(f"Invalid source '{source}', expected one of 'auto', 'raw', 'clean'.")

    if source == "raw":
        prefix = trial_prefix(trial_dir)
        npz_path = trial_dir / f"{prefix}angles_raw.npz"
        if not npz_path.exists():
            npz_path = trial_dir / "angles_raw.npz"
    elif source == "clean":
        prefix = trial_prefix(trial_dir)
        npz_path = trial_dir / f"{prefix}angles_clean.npz"
        if not npz_path.exists():
            npz_path = trial_dir / "angles_clean.npz"
    else:
        prefix = trial_prefix(trial_dir)
        npz_path = trial_dir / f"{prefix}angles.npz"
        if not npz_path.exists():
            npz_path = trial_dir / "angles.npz"
        if not npz_path.exists():
            raw_path = trial_dir / f"{prefix}angles_raw.npz"
            clean_path = trial_dir / f"{prefix}angles_clean.npz"
            if not raw_path.exists():
                raw_path = trial_dir / "angles_raw.npz"
            if not clean_path.exists():
                clean_path = trial_dir / "angles_clean.npz"
            if raw_path.exists():
                npz_path = raw_path
            elif clean_path.exists():
                npz_path = clean_path

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Could not find angles file in {trial_dir}. Expected one of angles*.npz."
        )
    data = np.load(npz_path)
    if "elbow_rad" in data and "shoulder_rad" in data:
        elbow = data["elbow_rad"]
        shoulder = data["shoulder_rad"]
    else:
        elbow = np.deg2rad(data["elbow_deg"])
        shoulder = np.deg2rad(data["shoulder_deg"])

    shoulder = _ensure_2d(shoulder)
    q_demo = np.column_stack([elbow, shoulder])
    valid = np.all(np.isfinite(q_demo), axis=1)
    q_demo = q_demo[valid]
    if q_demo.shape[0] < 10:
        raise ValueError(f"Not enough valid samples after cleaning: {q_demo.shape}")
    return q_demo

def plot_dmp_trajectory(trial_dir: Path, out_path: Path, n_basis: int = 15, angles_source: str = "auto") -> None:
    """Fit DMP from trial angles, rollout, and plot demo vs generated."""
    q_demo = load_angles_demo(trial_dir, source=angles_source)
    #q_demo = np.deg2rad(smooth_angles_deg(np.degrees(q_demo)))

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
    report = validate_joint_trajectory_deg(np.degrees(q_gen), dt, name="DMP rollout (deg)")
    print(report.reason)

    plot_dmp_single(
        np.degrees(q_demo),
        np.degrees(q_gen),
        _load_meta(trial_dir),
        out_path,
        title_suffix=f"{angles_source}, n_basis={n_basis}",
    )


def forcing_target_from_trajectory(
    q: np.ndarray,
    *,
    tau: float,
    dt: float,
    alpha_transformation: float,
    beta_transformation: float,
    diff_g_q0_eps: float = 1e-6,
    derivative_method: str = "savgol",
    savgol_window_length: int = 11,
    savgol_polyorder: int = 3,
) -> np.ndarray:
    """Compute f_target(t) for one joint trajectory."""
    y = np.asarray(q, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"Expected q to be 1D, got shape {y.shape}")
    if y.size < 3:
        raise ValueError("Need at least 3 samples to compute forcing.")

    dq, ddq = estimate_derivatives(
        y,
        dt=dt,
        derivative_method=derivative_method,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )

    q0 = float(y[0])
    g = float(y[-1])
    scale = g - q0
    if abs(scale) < diff_g_q0_eps:
        scale = 1.0

    return (
        tau**2 * ddq
        - alpha_transformation * beta_transformation * (g - y)
        + alpha_transformation * dq
    ) / scale


def forcing_fit_from_phase(model: DMPModel, x: np.ndarray) -> np.ndarray:
    """Compute fitted forcing f(x) from a learned DMP model."""
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    psi = np.exp(-model.widths[None, :] * (x_arr[:, None] - model.centers[None, :]) ** 2)
    psi_norm = psi / (psi.sum(axis=1, keepdims=True) + 1e-10)
    return x_arr[:, None] * (psi_norm @ model.weights.T)


def plot_dmp_forcing_fit_single_joint(
    q_demo: np.ndarray,
    joint_idx: int,
    out_path: Path,
    *,
    n_basis: int = 30,
    tau: float = 1.0,
    alpha_canonical: float = 4.0,
    alpha_transformation: float = 25.0,
    beta_transformation: float = 6.25,
    derivative_method: str = "gradient",
    savgol_window_length: int = 11,
    savgol_polyorder: int = 3,
    meta: dict | None = None,
    title_suffix: str = "",
) -> None:
    """
    Plot DMP forcing-term target vs fitted forcing for one joint.

    q_demo: (T, n_joints) in radians.
    joint_idx: index of the joint to visualize.
    """
    if q_demo.ndim != 2:
        raise ValueError(f"Expected q_demo shape (T, n_joints), got {q_demo.shape}")
    if not (0 <= joint_idx < q_demo.shape[1]):
        raise ValueError(f"joint_idx={joint_idx} out of bounds for q_demo with {q_demo.shape[1]} joints")
    if q_demo.shape[0] < 5:
        raise ValueError(f"Need at least 5 samples, got {q_demo.shape[0]}")

    T = int(q_demo.shape[0])
    dt_eff = tau / (T - 1)
    model = fit(
        [q_demo],
        tau=tau,
        dt=dt_eff,
        n_basis_functions=n_basis,
        alpha_canonical=alpha_canonical,
        alpha_transformation=alpha_transformation,
        beta_transformation=beta_transformation,
    )

    qj = q_demo[:, joint_idx]
    t = np.arange(T, dtype=np.float64) * dt_eff
    x = canonical_phase(t, tau=tau, alpha_canonical=alpha_canonical)

    f_target = forcing_target_from_trajectory(
        qj,
        tau=tau,
        dt=dt_eff,
        alpha_transformation=alpha_transformation,
        beta_transformation=beta_transformation,
        diff_g_q0_eps=1e-6,
        derivative_method=derivative_method,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
    )

    f_fit = forcing_fit_from_phase(model, x)[:, joint_idx]

    rmse = float(np.sqrt(np.mean((f_target - f_fit) ** 2)))
    denom = float(np.sum((f_target - np.mean(f_target)) ** 2))
    r2 = 1.0 - float(np.sum((f_target - f_fit) ** 2)) / (denom + 1e-12)

    joint_names = [
        "Elbow flexion",
        "Shoulder flexion",
        "Shoulder abduction",
        "Shoulder internal rotation",
    ]
    joint_label = joint_names[joint_idx] if joint_idx < len(joint_names) else f"joint {joint_idx}"

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(t, f_target, color="#2c3e50", linewidth=1.5, label="Target f")
    ax.plot(t, f_fit, color="#e67e22", linestyle="--", linewidth=1.5, label="Fitted f")
    ax.set_ylabel("Forcing term")
    ax.set_title(f"Time domain — {joint_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(x, f_target, color="#2c3e50", linewidth=1.4, label="Target f")
    ax.plot(x, f_fit, color="#e67e22", linestyle="--", linewidth=1.4, label="Fitted f")
    ax.set_xlabel("Phase x")
    ax.set_ylabel("Forcing term")
    ax.set_title(f"Phase domain (RMSE={rmse:.4f}, R2={r2:.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    m = meta or {}
    subject = m.get("subject", "?")
    motion = m.get("motion", "?")
    trial = m.get("trial", "?")
    suffix = f" ({title_suffix})" if title_suffix else ""
    fig.suptitle(
        f"DMP forcing fit — subject {subject}, {motion}, trial {trial}, {joint_label}, n_basis={n_basis}{suffix}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_dmp_forcing_fit_from_trial(
    trial_dir: Path,
    *,
    out_path: Path | None = None,
    joint_idx: int = 1,
    n_basis: int = 30,
    angles_source: str = "auto",
    derivative_method: str = "gradient",
    savgol_window_length: int = 11,
    savgol_polyorder: int = 3,
) -> Path:
    """Load trial angles and plot DMP target-vs-fit forcing for one joint."""
    q_demo = load_angles_demo(trial_dir, source=angles_source)
    meta = _load_meta(trial_dir)
    if out_path is None:
        out_path = trial_dir / f"{trial_prefix(trial_dir)}dmp_forcing_fit_joint{joint_idx}_n{n_basis}.png"
    plot_dmp_forcing_fit_single_joint(
        q_demo,
        joint_idx,
        out_path,
        n_basis=n_basis,
        derivative_method=derivative_method,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
        meta=meta,
        title_suffix=angles_source,
    )
    return out_path


KEYPOINT_NAMES = ["left_shoulder", "left_elbow", "left_wrist"]


def load_trajectory(trial_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    seq_path = trial_dir / "left_arm_seq_camera.npy"
    t_path = trial_dir / "left_arm_t.npy"
    if not seq_path.exists():
        raise FileNotFoundError(f"Not found: {seq_path}")
    seq = np.load(seq_path)
    t = np.load(t_path) if t_path.exists() else np.arange(seq.shape[0], dtype=np.float64)
    return seq, t, _load_meta(trial_dir)


def set_axes_equal(ax: Axes3D) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    radius = (np.abs(limits[:, 1] - limits[:, 0])).max() / 2
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


def plot_3d_trajectory(seq: np.ndarray, t: np.ndarray, meta: dict, out_path: Path | None = None) -> None:
    T, K, _ = seq.shape
    names = meta.get("keypoint_names", KEYPOINT_NAMES)[:K]
    name_to_idx = {str(name): idx for idx, name in enumerate(names)}

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for k in range(K):
        valid = ~np.isnan(seq[:, k, 0])
        if np.any(valid):
            ax1.plot(
                seq[valid, k, 0],
                seq[valid, k, 2],
                -seq[valid, k, 1],
                color=colors[k % len(colors)],
                label=names[k] if k < len(names) else f"kp{k}",
                alpha=0.8,
                linewidth=1.5,
            )
    for i, ti in enumerate([0, T // 2, T - 1]):
        if T == 0:
            break
        alpha = 0.3 + 0.35 * (i / max(1, 2))
        # Draw requested skeleton vectors when keypoint names are available.
        requested_edges = [
            ("right_shoulder", "left_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
        ]
        drawn_any_requested = False
        for a_name, b_name in requested_edges:
            if a_name in name_to_idx and b_name in name_to_idx:
                p0 = seq[ti, name_to_idx[a_name]]
                p1 = seq[ti, name_to_idx[b_name]]
                if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                    ax1.quiver(
                        p0[0],
                        p0[2],
                        -p0[1],
                        p1[0] - p0[0],
                        p1[2] - p0[2],
                        -(p1[1] - p0[1]),
                        arrow_length_ratio=0.2,
                        color="k",
                        alpha=alpha,
                        linewidth=2,
                    )
                    drawn_any_requested = True

        # Fallback to previous adjacent-link drawing if requested names are unavailable.
        if drawn_any_requested:
            continue
        for j in range(K - 1):
            p0 = seq[ti, j]
            p1 = seq[ti, j + 1]
            if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                ax1.plot([p0[0], p1[0]], [p0[2], p1[2]], [-p0[1], -p1[1]], "k-", alpha=alpha, linewidth=2)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax1.set_zlabel("up (m)")
    ax1.legend()
    ax1.set_title("3D trajectory (physically up frame)")
    set_axes_equal(ax1)

    ax2 = fig.add_subplot(122)
    for k in range(K):
        valid = ~np.isnan(seq[:, k, 0])
        name = names[k] if k < len(names) else f"kp{k}"
        if np.any(valid):
            ax2.plot(t[valid], seq[valid, k, 0], color=colors[k % len(colors)], alpha=0.8, linestyle="-", label=f"{name} x")
            ax2.plot(t[valid], -seq[valid, k, 1], color=colors[k % len(colors)], alpha=0.8, linestyle="--", label=f"{name} up")
            ax2.plot(t[valid], seq[valid, k, 2], color=colors[k % len(colors)], alpha=0.8, linestyle=":", label=f"{name} z")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("position (m)")
    ax2.set_title("Position vs time")
    ax2.legend(loc="upper right", fontsize=7, ncol=1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
    else:
        plt.show()
    plt.close(fig)

