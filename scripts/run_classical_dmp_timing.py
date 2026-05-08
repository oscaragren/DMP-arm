from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_here = Path(__file__).resolve()
_project_root = _here.parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from experiments.classical_dmp_timing_api import (
    ClassicalDMPTimingBudgetsMs,
    ClassicalDMPTimingConfig,
    run_classical_dmp_timing_experiment,
)


def _load_config_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Config must be a JSON object (dict), got {type(obj).__name__}")
    return obj


def _get(cfg: dict[str, Any], key: str, default: Any) -> Any:
    v = cfg.get(key, default)
    return default if v is None else v


def main() -> None:
    ap = argparse.ArgumentParser(description="Run classical DMP real-time timing experiment (API wrapper).")

    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.json",
        help="Path to JSON config file (defaults to scripts/config.json).",
    )

    ap.add_argument("--trial-dir", type=Path, default=None, help="Override config.json trial_dir")
    ap.add_argument("--out-root", type=Path, default=None, help="Override config.json out_root")

    ap.add_argument("--n-iters", type=int, default=None)
    ap.add_argument("--period-ms", type=float, default=None)
    ap.add_argument("--window-size", type=int, default=None)

    # These three are explicit overrides if provided (per requirement)
    ap.add_argument("--phase-mode", type=str, default=None, choices=["time", "human-progress"])
    ap.add_argument("--coupling-mode", type=str, default=None, choices=["none", "pd"])
    ap.add_argument("--comm-mode", type=str, default=None, choices=["none", "sleep", "can"])

    ap.add_argument("--kp", type=float, default=None)
    ap.add_argument("--kd", type=float, default=None)
    ap.add_argument("--autonomy-gain", type=float, default=None)

    ap.add_argument("--comm-sleep-ms", type=float, default=None)

    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--n-basis", type=int, default=None)
    ap.add_argument("--alpha-canonical", type=float, default=None)
    ap.add_argument("--alpha-transformation", type=float, default=None)
    ap.add_argument("--beta-transformation", type=float, default=None)

    ap.add_argument("--pose-input-mode", type=str, default=None, choices=["replay", "realtime"])
    ap.add_argument("--rt-fps", type=float, default=None)
    ap.add_argument("--rt-width", type=int, default=None)
    ap.add_argument("--rt-height", type=int, default=None)
    ap.add_argument("--rt-model", type=str, default=None)
    ap.add_argument("--rt-patch", type=int, default=None)
    ap.add_argument("--rt-min-z", type=float, default=None)
    ap.add_argument("--rt-max-z", type=float, default=None)
    ap.add_argument("--rt-show-window", action="store_true")
    ap.add_argument("--rt-show-depth", action="store_true")
    ap.add_argument("--rt-window-name", type=str, default=None)
    ap.add_argument("--rt-record-video", action="store_true")
    ap.add_argument("--rt-video-path", type=str, default=None)

    ap.add_argument("--deadline-e2e-ms", type=float, default=None)
    ap.add_argument("--budget-pose-ms", type=float, default=None)
    ap.add_argument("--budget-preprocess-ms", type=float, default=None)
    ap.add_argument("--budget-angle-ms", type=float, default=None)
    ap.add_argument("--budget-phase-ms", type=float, default=None)
    ap.add_argument("--budget-dmp-step-ms", type=float, default=None)
    ap.add_argument("--budget-coupling-ms", type=float, default=None)
    ap.add_argument("--budget-comm-ms", type=float, default=None)

    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")
    cfg = _load_config_json(cfg_path)

    trial_dir = Path(args.trial_dir) if args.trial_dir is not None else Path(_get(cfg, "trial_dir", ""))
    out_root = Path(args.out_root) if args.out_root is not None else Path(_get(cfg, "out_root", ""))
    if str(trial_dir) in {"", "."}:
        raise SystemExit("trial_dir must be set in config.json or via --trial-dir")
    if str(out_root) in {"", "."}:
        raise SystemExit("out_root must be set in config.json or via --out-root")

    out_dir = out_root / trial_dir.name
    config = ClassicalDMPTimingConfig(
        n_iters=int(_get(cfg, "n_iters", 2000)) if args.n_iters is None else int(args.n_iters),
        period_ms=float(_get(cfg, "period_ms", 10.0)) if args.period_ms is None else float(args.period_ms),
        window_size=int(_get(cfg, "window_size", 15)) if args.window_size is None else int(args.window_size),
        phase_mode=str(_get(cfg, "phase_mode", "time")) if args.phase_mode is None else str(args.phase_mode),
        coupling_mode=str(_get(cfg, "coupling_mode", "none"))
        if args.coupling_mode is None
        else str(args.coupling_mode),
        kp=float(_get(cfg, "kp", 0.0)) if args.kp is None else float(args.kp),
        kd=float(_get(cfg, "kd", 0.0)) if args.kd is None else float(args.kd),
        autonomy_gain=float(_get(cfg, "autonomy_gain", 1.0))
        if args.autonomy_gain is None
        else float(args.autonomy_gain),
        comm_mode=str(_get(cfg, "comm_mode", "none")) if args.comm_mode is None else str(args.comm_mode),
        comm_sleep_ms=float(_get(cfg, "comm_sleep_ms", 0.0)) if args.comm_sleep_ms is None else float(args.comm_sleep_ms),
        tau=float(_get(cfg, "tau", 1.0)) if args.tau is None else float(args.tau),
        dt=float(_get(cfg, "dt", 0.01)) if args.dt is None else float(args.dt),
        n_basis=int(_get(cfg, "n_basis", 50)) if args.n_basis is None else int(args.n_basis),
        alpha_canonical=float(_get(cfg, "alpha_canonical", 4.0))
        if args.alpha_canonical is None
        else float(args.alpha_canonical),
        alpha_transformation=float(_get(cfg, "alpha_transformation", 25.0))
        if args.alpha_transformation is None
        else float(args.alpha_transformation),
        beta_transformation=float(_get(cfg, "beta_transformation", 6.25))
        if args.beta_transformation is None
        else float(args.beta_transformation),
        pose_input_mode=str(_get(cfg, "pose_input_mode", "replay"))
        if args.pose_input_mode is None
        else str(args.pose_input_mode),
        rt_fps=float(_get(cfg, "rt_fps", 25.0)) if args.rt_fps is None else float(args.rt_fps),
        rt_width=int(_get(cfg, "rt_width", 640)) if args.rt_width is None else int(args.rt_width),
        rt_height=int(_get(cfg, "rt_height", 480)) if args.rt_height is None else int(args.rt_height),
        rt_model_path=str(_get(cfg, "rt_model_path", "capture/pose_landmarker_lite.task"))
        if args.rt_model is None
        else str(args.rt_model),
        rt_patch=int(_get(cfg, "rt_patch", 3)) if args.rt_patch is None else int(args.rt_patch),
        rt_min_z=float(_get(cfg, "rt_min_z", 0.0)) if args.rt_min_z is None else float(args.rt_min_z),
        rt_max_z=float(_get(cfg, "rt_max_z", 3.0)) if args.rt_max_z is None else float(args.rt_max_z),
        rt_show_window=bool(_get(cfg, "rt_show_window", False)) or bool(args.rt_show_window),
        rt_show_depth=bool(_get(cfg, "rt_show_depth", False)) or bool(args.rt_show_depth),
        rt_window_name=str(_get(cfg, "rt_window_name", "RT Pose"))
        if args.rt_window_name is None
        else str(args.rt_window_name),
        rt_record_video=bool(_get(cfg, "rt_record_video", False)) or bool(args.rt_record_video),
        rt_video_path=str(_get(cfg, "rt_video_path", "")) if args.rt_video_path is None else str(args.rt_video_path),
    )
    budgets = ClassicalDMPTimingBudgetsMs(
        pose_ms=float(_get(cfg, "budget_pose_ms", 1.0)) if args.budget_pose_ms is None else float(args.budget_pose_ms),
        preprocess_ms=float(_get(cfg, "budget_preprocess_ms", 2.0))
        if args.budget_preprocess_ms is None
        else float(args.budget_preprocess_ms),
        angle_ms=float(_get(cfg, "budget_angle_ms", 1.0)) if args.budget_angle_ms is None else float(args.budget_angle_ms),
        phase_ms=float(_get(cfg, "budget_phase_ms", 0.5)) if args.budget_phase_ms is None else float(args.budget_phase_ms),
        dmp_step_ms=float(_get(cfg, "budget_dmp_step_ms", 1.0))
        if args.budget_dmp_step_ms is None
        else float(args.budget_dmp_step_ms),
        coupling_ms=float(_get(cfg, "budget_coupling_ms", 0.5))
        if args.budget_coupling_ms is None
        else float(args.budget_coupling_ms),
        comm_ms=float(_get(cfg, "budget_comm_ms", 1.0)) if args.budget_comm_ms is None else float(args.budget_comm_ms),
        e2e_ms=float(_get(cfg, "deadline_e2e_ms", 8.0)) if args.deadline_e2e_ms is None else float(args.deadline_e2e_ms),
    )

    # NOTE: This CLI is hardware-independent. comm_mode='can' is only supported
    # via external importers that can pass a send_can_msg callback to the API.
    if config.comm_mode == "can":
        raise SystemExit(
            "comm_mode='can' requires a send_can_msg callback, which is only available via the API.\n"
            "Use:\n"
            "  from experiments.classical_dmp_timing_api import run_classical_dmp_timing_experiment\n"
            "and pass send_can_msg=... from your hardware repo."
        )

    run_classical_dmp_timing_experiment(
        trial_dir=trial_dir,
        out_dir=out_dir,
        config=config,
        budgets=budgets,
        send_can_msg=None,
    )


if __name__ == "__main__":
    main()

