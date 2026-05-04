from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TrialId:
    subject: str  # e.g. "subject_01"
    motion: str  # e.g. "reach"
    trial: str  # e.g. "trial_003"

    @property
    def rel_dir(self) -> Path:
        return Path(self.subject, self.motion, self.trial)


def _parse_csv_or_all(values: Optional[str], *, kind: str) -> Optional[List[str]]:
    """
    Args:
        values: None => means "all". Otherwise comma-separated string.
        kind: for nicer error messages.
    """
    if values is None:
        return None
    v = values.strip()
    if v.lower() == "all":
        return None
    parts = [p.strip() for p in v.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"--{kind} cannot be empty. Use e.g. 'all' or '1,2,3'.")
    return parts


def _discover_trials(processed_root: Path) -> List[TrialId]:
    """
    Discover trials by finding 'angles.npz' under:
        data/processed/subject_xx/<motion>/trial_yyy/angles.npz
    """
    trials: List[TrialId] = []
    for angles_path in processed_root.glob("subject_*/*/trial_*/angles.npz"):
        try:
            trial_dir = angles_path.parent
            motion_dir = trial_dir.parent
            subject_dir = motion_dir.parent
            trials.append(
                TrialId(
                    subject=subject_dir.name,
                    motion=motion_dir.name,
                    trial=trial_dir.name,
                )
            )
        except Exception:
            # If the path doesn't match expected layout, skip it.
            continue

    # Deterministic ordering
    trials.sort(key=lambda x: (x.subject, x.motion, x.trial))
    return trials


def _filter_trials(
    all_trials: Sequence[TrialId],
    *,
    subjects: Optional[Sequence[str]],
    motions: Optional[Sequence[str]],
    trials: Optional[Sequence[str]],
) -> List[TrialId]:
    def _norm_subject(s: str) -> str:
        s = s.strip()
        if s.startswith("subject_"):
            return s
        if s.isdigit():
            return f"subject_{int(s):02d}"
        return s

    def _norm_trial(t: str) -> str:
        t = t.strip()
        if t.startswith("trial_"):
            return t
        if t.isdigit():
            return f"trial_{int(t):03d}"
        return t

    subjects_n = None if subjects is None else {_norm_subject(s) for s in subjects}
    motions_n = None if motions is None else {m.strip() for m in motions}
    trials_n = None if trials is None else {_norm_trial(t) for t in trials}

    out: List[TrialId] = []
    for tid in all_trials:
        if subjects_n is not None and tid.subject not in subjects_n:
            continue
        if motions_n is not None and tid.motion not in motions_n:
            continue
        if trials_n is not None and tid.trial not in trials_n:
            continue
        out.append(tid)
    return out


def _rmse(a: np.ndarray, b: np.ndarray, *, axis: Optional[int | Tuple[int, ...]] = None) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"RMSE needs same shape, got {a.shape} vs {b.shape}")
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))


def _pearsonr(x: np.ndarray, y: np.ndarray, *, eps: float = 1e-12) -> float:
    """
    Pearson correlation r for 1D signals. Returns NaN if undefined.
    """
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        return float("nan")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    sa = float(np.sqrt(np.mean(a * a)))
    sb = float(np.sqrt(np.mean(b * b)))
    if sa < eps or sb < eps:
        return float("nan")
    return float(np.mean((a / sa) * (b / sb)))


def _inter_joint_corr_matrix(q: np.ndarray) -> np.ndarray:
    """
    Inter-joint correlation matrix across time for multi-DoF trajectories.
    Rows are time, columns are joints/DoFs.
    """
    x = np.asarray(q, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected q shape (T,) or (T,D), got {q.shape}")
    if x.shape[0] < 2:
        return np.full((x.shape[1], x.shape[1]), np.nan, dtype=float)
    # np.corrcoef expects variables in rows when rowvar=True; we want joints as vars => rowvar=False.
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(x, rowvar=False)


def _corr_structure_similarity(c_demo: np.ndarray, c_gen: np.ndarray, *, eps: float = 1e-12) -> Dict[str, float]:
    """
    Compare two correlation matrices by vectorizing upper triangle (excluding diagonal).
    Returns cosine similarity and 1 - MAE similarity.
    """
    a = np.asarray(c_demo, dtype=float)
    b = np.asarray(c_gen, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        return {"cosine_similarity": float("nan"), "one_minus_mae": float("nan"), "mae": float("nan")}
    d = a.shape[0]
    if d < 2:
        return {"cosine_similarity": float("nan"), "one_minus_mae": float("nan"), "mae": float("nan")}

    iu = np.triu_indices(d, k=1)
    va = a[iu]
    vb = b[iu]
    mask = np.isfinite(va) & np.isfinite(vb)
    if np.sum(mask) < 1:
        return {"cosine_similarity": float("nan"), "one_minus_mae": float("nan"), "mae": float("nan")}

    va = va[mask]
    vb = vb[mask]

    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    cos = float((va @ vb) / (denom + eps))
    mae = float(np.mean(np.abs(va - vb)))
    return {
        "cosine_similarity": cos,
        "mae": mae,
        "one_minus_mae": float(1.0 - mae),
    }


def _speed_magnitude(q: np.ndarray, *, dt: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        dq = np.gradient(q, dt)
        return np.abs(dq)
    if q.ndim != 2:
        raise ValueError(f"Expected q shape (T,) or (T,D), got {q.shape}")
    dq = np.gradient(q, dt, axis=0)
    return np.linalg.norm(dq, axis=1)


def _sparc_from_speed(
    speed: np.ndarray,
    *,
    fs: float,
    fc: float = 10.0,
    amp_thresh: float = 0.05,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Spectral Arc Length (SPARC) computed from a 1D speed profile.

    Implementation follows common SPARC usage:
      - Compute magnitude spectrum of speed
      - Normalize by max magnitude
      - Consider frequencies in [0, fc] and above amp threshold
      - Arc length over (f, A(f)) curve, returned as negative value (more smooth -> closer to 0)
    """
    v = np.asarray(speed, dtype=float).reshape(-1)
    n = int(v.size)
    if n < 8 or not np.all(np.isfinite(v)):
        return {"sparc": float("nan")}

    v = v - np.mean(v)
    mag = np.abs(np.fft.rfft(v))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))

    if mag.size == 0:
        return {"sparc": float("nan")}

    mag_norm = mag / (np.max(mag) + eps)

    # Band-limit and threshold
    band = freqs <= float(fc)
    if not np.any(band):
        return {"sparc": float("nan")}

    freqs_b = freqs[band]
    mag_b = mag_norm[band]

    keep = mag_b >= float(amp_thresh)
    if np.sum(keep) < 2:
        # Not enough points to compute an arc length
        return {"sparc": float("nan")}

    freqs_k = freqs_b[keep]
    mag_k = mag_b[keep]

    # Arc length in (f, A) space; use log amplitude to reduce scale effects
    A = np.log(mag_k + eps)
    df = np.diff(freqs_k)
    dA = np.diff(A)
    arc = np.sum(np.sqrt(df**2 + dA**2))
    return {"sparc": -float(arc)}


def _ldlj(
    q: np.ndarray,
    *,
    dt: float,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Log Dimensionless Jerk (LDLJ) for 1D or multi-DoF trajectories.

    Uses:
      DJ = (T^5 / A^2) * ∫ ||d^3 q / dt^3||^2 dt
      LDLJ = log(DJ)

    Where A is the Euclidean distance between start and end in joint space.
    """
    x = np.asarray(q, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected q shape (T,) or (T,D), got {q.shape}")

    Tn = int(x.shape[0])
    if Tn < 8 or not np.all(np.isfinite(x)):
        return {"dimensionless_jerk": float("nan"), "ldlj": float("nan")}

    duration = float((Tn - 1) * dt)
    if duration <= 0:
        return {"dimensionless_jerk": float("nan"), "ldlj": float("nan")}

    A = float(np.linalg.norm(x[-1] - x[0]))
    A = max(A, eps)

    # 3rd derivative (jerk) with repeated gradients
    d1 = np.gradient(x, dt, axis=0)
    d2 = np.gradient(d1, dt, axis=0)
    d3 = np.gradient(d2, dt, axis=0)
    jerk_sq = np.sum(d3**2, axis=1)
    integral = float(np.trapezoid(jerk_sq, dx=dt))

    dj = (duration**5 / (A**2)) * integral
    dj = max(dj, eps)
    return {"dimensionless_jerk": float(dj), "ldlj": float(np.log(dj))}


def _load_trial_arrays(trial_dir: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    angles_path = trial_dir / "angles.npz"
    rollout_path = trial_dir / "dmp_rollout.npz"
    if not angles_path.exists():
        raise FileNotFoundError(f"Missing {angles_path}")
    if not rollout_path.exists():
        raise FileNotFoundError(f"Missing {rollout_path}")

    with np.load(angles_path) as a:
        q_demo = np.asarray(a["angles"], dtype=float)
        dt = float(a["dt"]) if "dt" in a.files else None
    with np.load(rollout_path) as r:
        q_gen = np.asarray(r["q_gen"], dtype=float)
        dt_r = float(r["dt"]) if "dt" in r.files else None

    dt_final = dt if dt is not None else dt_r
    if dt_final is None:
        # Fallback: infer dt from length (tau assumed 1.0 as in analyze_data.py)
        dt_final = 1.0 / max(1, q_demo.shape[0] - 1)

    if q_demo.shape != q_gen.shape:
        # Best-effort: truncate to the shorter length
        n = min(q_demo.shape[0], q_gen.shape[0])
        q_demo = q_demo[:n]
        q_gen = q_gen[:n]

    return q_demo, q_gen, float(dt_final)


def _evaluate_trial(trial_dir: Path) -> Dict[str, Any]:
    q_demo, q_gen, dt = _load_trial_arrays(trial_dir)

    rmse_per_joint = _rmse(q_demo, q_gen, axis=0).tolist()
    rmse_all = float(_rmse(q_demo, q_gen))

    fs = 1.0 / dt if dt > 0 else float("nan")
    speed_demo = _speed_magnitude(q_demo, dt=dt)
    speed_gen = _speed_magnitude(q_gen, dt=dt)

    sparc_demo = _sparc_from_speed(speed_demo, fs=fs)
    sparc_gen = _sparc_from_speed(speed_gen, fs=fs)

    ldlj_demo = _ldlj(q_demo, dt=dt)
    ldlj_gen = _ldlj(q_gen, dt=dt)

    # Inter-joint cross-correlation similarity (0-lag correlation structure).
    corr_demo = _inter_joint_corr_matrix(q_demo)
    corr_gen = _inter_joint_corr_matrix(q_gen)
    corr_sim = _corr_structure_similarity(corr_demo, corr_gen)

    # Angular velocity profile correlation (per DoF): corr(dq_demo_j(t), dq_gen_j(t))
    qd = np.asarray(q_demo, dtype=float)
    qg = np.asarray(q_gen, dtype=float)
    if qd.ndim == 1:
        qd = qd[:, None]
    if qg.ndim == 1:
        qg = qg[:, None]
    dq_demo = np.gradient(qd, dt, axis=0)
    dq_gen = np.gradient(qg, dt, axis=0)

    vel_corr_per_dof = [_pearsonr(dq_demo[:, j], dq_gen[:, j]) for j in range(int(qd.shape[1]))]
    vel_corr_mean = float(np.nanmean(np.asarray(vel_corr_per_dof, dtype=float))) if len(vel_corr_per_dof) else float("nan")

    return {
        "dt": float(dt),
        "n_steps": int(q_demo.shape[0]),
        "n_dofs": int(q_demo.shape[1]) if q_demo.ndim == 2 else 1,
        "rmse": {
            "overall": rmse_all,
            "per_dof": rmse_per_joint,
        },
        "sparc": {
            "demo": sparc_demo,
            "dmp": sparc_gen,
            "delta": {
                "sparc": float(sparc_gen.get("sparc", float("nan")) - sparc_demo.get("sparc", float("nan"))),
            },
        },
        "ldlj": {
            "demo": ldlj_demo,
            "dmp": ldlj_gen,
            "delta": {
                "dimensionless_jerk": float(
                    ldlj_gen.get("dimensionless_jerk", float("nan"))
                    - ldlj_demo.get("dimensionless_jerk", float("nan"))
                ),
                "ldlj": float(ldlj_gen.get("ldlj", float("nan")) - ldlj_demo.get("ldlj", float("nan"))),
            },
        },
        "similarity": {
            "inter_joint_cross_correlation": {
                "cosine_similarity": float(corr_sim.get("cosine_similarity", float("nan"))),
                "mae": float(corr_sim.get("mae", float("nan"))),
                "one_minus_mae": float(corr_sim.get("one_minus_mae", float("nan"))),
            },
            "angular_velocity_profile_correlation": {
                "per_dof_pearson_r": [float(x) for x in vel_corr_per_dof],
                "mean_pearson_r": vel_corr_mean,
            },
        },
    }


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantitatively evaluate DMP rollouts vs demonstrations.")
    ap.add_argument(
        "--processed-root",
        type=str,
        default=str(Path("data", "processed")),
        help="Root folder containing processed trials (default: data/processed).",
    )
    ap.add_argument(
        "--results-root",
        type=str,
        default=str(Path("results")),
        help="Folder to write aggregated results JSON (default: results/).",
    )

    ap.add_argument("--subjects", type=str, default="all", help="Comma-separated subject IDs or names, e.g. '1,2' or 'subject_01'. Use 'all'.")
    ap.add_argument("--motions", type=str, default="all", help="Comma-separated motion names, e.g. 'reach,pour'. Use 'all'.")
    ap.add_argument("--trials", type=str, default="all", help="Comma-separated trial IDs or names, e.g. '1,2' or 'trial_003'. Use 'all'.")

    ap.add_argument(
        "--per-trial-filename",
        type=str,
        default="quant_results.json",
        help="Filename for per-trial results JSON stored inside each trial folder.",
    )
    ap.add_argument(
        "--overall-filename",
        type=str,
        default="quant_results_overall.json",
        help="Filename for the aggregated results JSON stored inside results-root.",
    )
    args = ap.parse_args()

    processed_root = Path(args.processed_root)
    results_root = Path(args.results_root)

    subjects = _parse_csv_or_all(args.subjects, kind="subjects")
    motions = _parse_csv_or_all(args.motions, kind="motions")
    trials = _parse_csv_or_all(args.trials, kind="trials")

    all_trials = _discover_trials(processed_root)
    selected = _filter_trials(all_trials, subjects=subjects, motions=motions, trials=trials)

    overall: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "processed_root": str(processed_root),
        "selection": {
            "subjects": subjects if subjects is not None else "all",
            "motions": motions if motions is not None else "all",
            "trials": trials if trials is not None else "all",
        },
        "n_trials_found": int(len(all_trials)),
        "n_trials_selected": int(len(selected)),
        "trials": [],
        "summary": {},
    }

    per_trial_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for tid in selected:
        trial_dir = processed_root / tid.rel_dir
        try:
            metrics = _evaluate_trial(trial_dir)
            record = {
                "trial_id": asdict(tid),
                "trial_dir": str(trial_dir),
                "metrics": metrics,
            }
            per_trial_records.append(record)

            # Write per-trial JSON inside corresponding processed trial folder
            _json_dump(trial_dir / args.per_trial_filename, record)
        except Exception as e:
            failures.append(
                {
                    "trial_id": json.dumps(asdict(tid)),
                    "trial_dir": str(trial_dir),
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    overall["trials"] = per_trial_records
    overall["failures"] = failures

    # Simple summary aggregates over trials (ignore NaNs)
    def _nanmean(values: List[float]) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(np.nanmean(arr))

    rmse_overall_vals = [float(r["metrics"]["rmse"]["overall"]) for r in per_trial_records]
    sparc_demo_vals = [float(r["metrics"]["sparc"]["demo"]["sparc"]) for r in per_trial_records]
    sparc_dmp_vals = [float(r["metrics"]["sparc"]["dmp"]["sparc"]) for r in per_trial_records]
    ldlj_demo_vals = [float(r["metrics"]["ldlj"]["demo"]["ldlj"]) for r in per_trial_records]
    ldlj_dmp_vals = [float(r["metrics"]["ldlj"]["dmp"]["ldlj"]) for r in per_trial_records]

    overall["summary"] = {
        "rmse_overall_mean": _nanmean(rmse_overall_vals),
        "sparc_demo_mean": _nanmean(sparc_demo_vals),
        "sparc_dmp_mean": _nanmean(sparc_dmp_vals),
        "ldlj_demo_mean": _nanmean(ldlj_demo_vals),
        "ldlj_dmp_mean": _nanmean(ldlj_dmp_vals),
        "n_failures": int(len(failures)),
    }

    # Write overall JSON to results folder
    _json_dump(results_root / args.overall_filename, overall)

    # Print a short, CLI-friendly message
    print(
        json.dumps(
            {
                "selected": len(selected),
                "evaluated": len(per_trial_records),
                "failures": len(failures),
                "overall_json": str((results_root / args.overall_filename).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
