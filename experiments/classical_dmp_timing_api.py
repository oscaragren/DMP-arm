from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from dmp.dmp import DMPModel, canonical_phase

# Reuse existing repo primitives (data conventions / angle definitions)
from quant_analysis import (
    _clip_angles_deg,
    interpolate_keypoints_cartesian,
    lowpass_keypoints,
    sequence_to_angles_deg,
    fit_dmp_lwr_multi,
)


@dataclass(frozen=True)
class ClassicalDMPTimingConfig:
    # Loop setup
    n_iters: int = 2000
    period_ms: float = 10.0
    window_size: int = 15

    # DMP params (offline fit only)
    tau: float = 1.0
    dt: float = 0.01
    n_basis: int = 100
    alpha_canonical: float = 4.0
    alpha_transformation: float = 25.0
    beta_transformation: float = 6.25

    # Online phase estimation
    phase_mode: str = "path-progress"  # "time" | "human-progress" | "path-progress"

    # Coupling
    coupling_mode: str = "none"  # "none" | "pd"
    kp: float = 0.0
    kd: float = 0.0
    autonomy_gain: float = 1.0

    # Communication
    comm_mode: str = "none"  # "none" | "sleep" | "can"
    comm_sleep_ms: float = 0.0

    # Robust derivatives
    qdot_alpha: float = 0.2  # exp smoothing for finite differences

    # Preprocess
    lowpass_cutoff_hz: float = 20.0
    lowpass_order: int = 2

    # Path-progress
    waypoint_indices: Optional[tuple[int, ...]] = None
    progress_eps: float = 1e-6

    # Outputs
    save_model: bool = True

    # Pose input (online)
    pose_input_mode: str = "realtime"  # "replay" | "realtime"

    # Real-time camera/pose settings (only used if pose_input_mode="realtime")
    rt_fps: float = 25.0
    rt_width: int = 640
    rt_height: int = 480
    rt_model_path: str = "models/pose_landmarker_lite.task"
    rt_patch: int = 3
    rt_min_z: float = 0.0
    rt_max_z: float = 3.0
    rt_show_window: bool = False
    rt_show_depth: bool = False
    rt_window_name: str = "RT Pose"
    rt_record_video: bool = False
    rt_video_path: str = ""  # if empty, writes out_dir/"rt_video.mp4"


POSE_KEYPOINT_IDS = [11, 13, 15, 12, 23, 24]


@dataclass(frozen=True)
class ClassicalDMPTimingBudgetsMs:
    pose_ms: float = 1.0
    preprocess_ms: float = 2.0
    angle_ms: float = 1.0
    phase_ms: float = 0.5
    dmp_step_ms: float = 1.0
    coupling_ms: float = 0.5
    comm_ms: float = 1.0
    e2e_ms: float = 8.0


@dataclass(frozen=True)
class ClassicalDMPTimingResult:
    config: dict[str, Any]
    budgets_ms: dict[str, float]
    paths: dict[str, str]
    offline_fit_ms: float
    summary: dict[str, Any]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _load_seq_and_t(trial_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load recorded keypoints and timestamps from a raw trial directory.

    Expected files:
      - left_arm_seq_camera.npy
      - left_arm_t.npy (optional; if missing, uses index-based timestamps)

    Returns:
      seq: (T, K, 3)
      t:   (T,) seconds, shifted so t[0]=0
    """
    trial_dir = Path(trial_dir)
    if not trial_dir.exists():
        raise FileNotFoundError(f"trial_dir does not exist: {trial_dir}")
    if not trial_dir.is_dir():
        raise NotADirectoryError(f"trial_dir must be a directory: {trial_dir}")

    seq_path = trial_dir / "left_arm_seq_camera.npy"
    t_path = trial_dir / "left_arm_t.npy"
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Could not find keypoint sequence in {trial_dir}. "
            "Expected raw 'left_arm_seq_camera.npy'."
        )

    seq = np.load(seq_path)
    if t_path.exists():
        t = np.asarray(np.load(t_path), dtype=np.float64).reshape(-1)
    else:
        t = np.arange(int(seq.shape[0]), dtype=np.float64)
    if t.shape[0] != seq.shape[0]:
        t = np.arange(int(seq.shape[0]), dtype=np.float64)
    if t.size > 0:
        t = t - t[0]
    return np.asarray(seq, dtype=np.float64), t


def _estimate_fps_from_t(t: np.ndarray, *, default_fps: float = 25.0) -> float:
    t = np.asarray(t, dtype=float).reshape(-1)
    if t.size < 2:
        return float(default_fps)
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 1e-6:
        return float(default_fps)
    return float(1.0 / dt)


def _finite_diff_filtered(
    q_prev: np.ndarray,
    q_curr: np.ndarray,
    *,
    dt: float,
    qdot_prev: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Exponentially-smoothed finite difference velocity estimate.
    alpha in [0,1], higher = more responsive, lower = more smoothing.
    This is used to estimate the velocity of the human joint angles.
    """
    if not (0.0 <= float(alpha) <= 1.0):
        alpha = float(np.clip(alpha, 0.0, 1.0))
    dq = (np.asarray(q_curr, float) - np.asarray(q_prev, float)) / max(float(dt), 1e-9)
    if not np.all(np.isfinite(dq)):
        return np.asarray(qdot_prev, float)
    return (1.0 - float(alpha)) * np.asarray(qdot_prev, float) + float(alpha) * dq

def _path_progress_phase(
    q_human: np.ndarray,
    *,
    waypoints: np.ndarray,
    active_segment: int,
    alpha_canonical: float,
    eps: float = 1e-6,
) -> tuple[float, float, int, float]:
    """
    Estimate progress along a piecewise-linear path.

    waypoints: (M, n_joints)
      Example: [q_start, q_pickup, q_place, q_return]

    active_segment:
      Current segment index. Segment i goes from waypoints[i] to waypoints[i+1].

    Returns:
      global_progress: [0,1]
      x: canonical phase
      active_segment: updated active segment
      local_progress: [0,1]
    """
    q_human = np.asarray(q_human, dtype=float).reshape(-1)
    waypoints = np.asarray(waypoints, dtype=float)

    n_segments = waypoints.shape[0] - 1
    active_segment = int(np.clip(active_segment, 0, n_segments - 1))

    a = waypoints[active_segment]
    b = waypoints[active_segment + 1]
    d = b - a

    denom = float(np.dot(d, d)) + float(eps)

    local_progress = float(np.dot(q_human - a, d) / denom)
    local_progress = float(np.clip(local_progress, 0.0, 1.0))

    # Move to next segment when the current one is nearly completed
    if local_progress >= 0.98 and active_segment < n_segments - 1:
        active_segment += 1

        a = waypoints[active_segment]
        b = waypoints[active_segment + 1]
        d = b - a
        denom = float(np.dot(d, d)) + float(eps)

        local_progress = float(np.dot(q_human - a, d) / denom)
        local_progress = float(np.clip(local_progress, 0.0, 1.0))

    global_progress = (active_segment + local_progress) / float(n_segments)
    global_progress = float(np.clip(global_progress, 0.0, 1.0))

    x = float(math.exp(-float(alpha_canonical) * global_progress))
    x = float(np.clip(x, 0.0, 1.0))

    return global_progress, x, active_segment, local_progress


def _human_progress_phase(
    q_human: np.ndarray,
    *,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    alpha_canonical: float,
) -> tuple[float, float]:
    """
    Project human motion onto the start->goal direction to estimate progress in [0,1],
    then map to canonical phase via x = exp(-alpha_canonical * progress).

    Progress is the fraction of the way from the start to the goal.
    Canonical phase is a value in [0,1] that is used to scale the DMP.

    Returns:
      progress: [0,1]
      x: canonical phase in (0,1]
    """
    q_human = np.asarray(q_human, dtype=float).reshape(-1)
    q_start = np.asarray(q_start, dtype=float).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=float).reshape(-1)
    d = q_goal - q_start
    denom = float(np.dot(d, d))
    if not np.isfinite(denom) or denom < 1e-12:
        progress = 0.0
    else:
        progress = float(np.dot(q_human - q_start, d) / denom)
    progress = float(np.clip(progress, 0.0, 1.0))
    x = float(math.exp(-float(alpha_canonical) * progress))
    x = float(np.clip(x, 0.0, 1.0))
    return progress, x


def _dmp_nominal_ddq(
    model: DMPModel,
    *,
    q: np.ndarray,
    dq: np.ndarray,
    q0: np.ndarray,
    g: np.ndarray,
    x: float,
    tau: float,
) -> np.ndarray:
    """
    Nominal DMP acceleration (vectorized over joints).
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    dq = np.asarray(dq, dtype=float).reshape(-1)
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    g = np.asarray(g, dtype=float).reshape(-1)

    psi = np.exp(-model.widths * (float(x) - model.centers) ** 2)
    psi_norm = psi / (float(np.sum(psi)) + 1e-10)
    f = float(x) * (model.weights @ psi_norm)  # (n_joints,)
    ddq = (
        model.alpha_transformation * model.beta_transformation * (g - q)
        - model.alpha_transformation * dq
        + (g - q0) * f
    ) / (float(tau) ** 2)
    return np.asarray(ddq, dtype=float)


def _stage_stats_ms(x_ms: np.ndarray) -> dict[str, float]:
    x = np.asarray(x_ms, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "max": float(np.max(x)),
        "p95": float(np.percentile(x, 95)),
    }


def _miss_rate(x_ms: np.ndarray, budget_ms: float) -> float:
    x = np.asarray(x_ms, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x > float(budget_ms)))


def run_classical_dmp_timing_experiment(
    trial_dir: Path,
    out_dir: Path,
    config: ClassicalDMPTimingConfig,
    budgets: ClassicalDMPTimingBudgetsMs,
    send_can_msg: Optional[Callable[[np.ndarray], bool]] = None,
) -> dict[str, Any]:
    """
    Hardware-independent, API-based experiment:
      - Offline: fit a classical DMP ONCE from a full recorded trajectory.
      - Online: run a fixed-period loop that simulates real-time pose observation,
        estimates human joint angles, estimates phase, rolls out the pretrained DMP
        (internal DMP state), optionally applies PD coupling, and optionally sends
        desired joint angles via a provided CAN callback.

    Returns a JSON-serializable summary dict (also written to summary.json).
    """
    trial_dir = Path(trial_dir)
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    if config.comm_mode == "can" and send_can_msg is None:
        raise ValueError(
            "comm_mode='can' requires a send_can_msg callback. "
            "Signature: send_can_msg(q_robot_desired: np.ndarray) -> bool"
        )

    # --------------------------
    # OFFLINE PHASE (fit once)
    # --------------------------
    seq_full, t_full = _load_seq_and_t(trial_dir)
    fps = _estimate_fps_from_t(t_full, default_fps=25.0)

    t_fit0 = time.perf_counter_ns()
    seq_i = interpolate_keypoints_cartesian(seq_full)
    seq_f = lowpass_keypoints(
        seq_i, fps=fps, cutoff_hz=float(config.lowpass_cutoff_hz), order=int(config.lowpass_order)
    )
    q_demo = sequence_to_angles_deg(seq_f)  # (T,4) degrees
    q_demo = _clip_angles_deg(q_demo)
    valid = np.all(np.isfinite(q_demo), axis=1)
    q_demo = q_demo[valid]
    if q_demo.shape[0] < max(10, int(config.window_size) + 2):
        raise ValueError(f"Not enough valid angle samples for offline fit: {q_demo.shape}")

    # Normalize dt to tau on the FULL trajectory (fit only once)
    dt_fit = float(config.tau) / float(q_demo.shape[0] - 1)
    model = fit_dmp_lwr_multi(
        [q_demo],
        tau=float(config.tau),
        dt=dt_fit,
        n_basis_functions=int(config.n_basis),
        alpha_canonical=float(config.alpha_canonical),
        alpha_transformation=float(config.alpha_transformation),
        beta_transformation=float(config.beta_transformation),
    )
    offline_fit_ms = (time.perf_counter_ns() - t_fit0) * 1e-6

    if config.save_model:
        np.savez(
            out_dir / "dmp_model_offline.npz",
            weights=model.weights,
            centers=model.centers,
            widths=model.widths,
            alpha_canonical=model.alpha_canonical,
            alpha_transformation=model.alpha_transformation,
            beta_transformation=model.beta_transformation,
            tau=model.tau,
            n_joints=model.n_joints,
            curvature_weights=model.curvature_weights,
            dt_fit=dt_fit,
        )

    q_start = np.asarray(q_demo[0], dtype=float)
    q_goal = np.asarray(q_demo[-1], dtype=float)
    if config.waypoint_indices is None:
        waypoint_indices = (0, q_demo.shape[0] // 3, 2 * q_demo.shape[0] // 3, -1)
    else:
        waypoint_indices = config.waypoint_indices

    waypoint_indices = tuple(
        idx if idx >= 0 else q_demo.shape[0] + idx
        for idx in waypoint_indices
    )

    waypoints = np.asarray(q_demo[list(waypoint_indices)], dtype=float)

    if waypoints.shape[0] < 2:
        raise ValueError("At least two waypoints are required for path-progress phase estimation.")
    # --------------------------
    # ONLINE PHASE (timed loop)
    # --------------------------
    period_ns = int(round(float(config.period_ms) * 1e6))
    dt_loop = float(config.dt)
    if dt_loop <= 0:
        dt_loop = float(config.period_ms) * 1e-3

    # Pre-allocate storage
    n_iters = int(config.n_iters)
    stages = [
        "pose_ms",
        "preprocess_ms",
        "angle_extraction_ms",
        "phase_estimation_ms",
        "dmp_step_ms",
        "coupling_ms",
        "comm_ms",
        "e2e_ms",
        "loop_exec_ms",
        "actual_period_ms",
        "schedule_error_ms",
        "tracking_err_l2",
        "phase_x",
        "progress",
    ]
    rec: dict[str, list[float]] = {k: [] for k in stages}

    # DMP internal state (separate from human state)
    q_dmp: Optional[np.ndarray] = None
    qdot_dmp: Optional[np.ndarray] = None
    q0_dmp: Optional[np.ndarray] = None
    g_dmp = q_goal.copy()

    q_h_prev: Optional[np.ndarray] = None # h for human
    qdot_h_prev: np.ndarray = np.zeros((model.n_joints,), dtype=float) # h for human

    # Circular buffer for keypoint window
    W = int(max(3, config.window_size)) # window size
    window: list[np.ndarray] = []

    next_tick_ns = time.monotonic_ns()
    loop_start_ns = next_tick_ns
    prev_iter_ns: Optional[int] = None

    active_segment = 0

    pose_mode = str(config.pose_input_mode).strip().lower()
    use_realtime = pose_mode == "realtime"
    if pose_mode not in {"replay", "realtime"}:
        raise ValueError("pose_input_mode must be 'replay' or 'realtime'")

    # For real-time, the online filtering should use the camera FPS rather than demo FPS.
    online_fps = float(config.rt_fps) if use_realtime else float(fps)

    # Real-time resources (initialized once, used inside the loop).
    rt_device = None
    rt_pipeline = None
    rt_queue = None
    rt_landmarker = None
    rt_cv2 = None
    rt_mp = None
    rt_writer = None
    rt_fx = rt_fy = rt_cx = rt_cy = None
    rt_last_ts_ms = -1

    def _rt_deproject(u: float, v: float, z_m: float) -> tuple[float, float, float]:
        assert rt_fx is not None and rt_fy is not None and rt_cx is not None and rt_cy is not None
        x = (u - float(rt_cx)) * z_m / float(rt_fx)
        y = (v - float(rt_cy)) * z_m / float(rt_fy)
        return float(x), float(y), float(z_m)

    def _rt_depth_at(depth_mm: np.ndarray, u: float, v: float, *, patch: int) -> Optional[float]:
        h, w = depth_mm.shape[:2]
        u0, v0 = int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))
        r = max(0, int(patch) // 2)
        x1, x2 = max(0, u0 - r), min(w, u0 + r + 1)
        y1, y2 = max(0, v0 - r), min(h, v0 + r + 1)
        roi = depth_mm[y1:y2, x1:x2].astype(np.float32)
        roi = roi[roi > 0]
        if roi.size == 0:
            return None
        return float(np.median(roi)) / 1000.0

    if use_realtime:
        # Heavy imports only when real-time is used.
        import cv2 as _cv2  # type: ignore
        import depthai as dai  # type: ignore
        import mediapipe as mp  # type: ignore
        from datetime import timedelta

        rt_cv2 = _cv2
        rt_mp = mp

        rgb_socket = dai.CameraBoardSocket.CAM_A
        left_socket = dai.CameraBoardSocket.CAM_B
        right_socket = dai.CameraBoardSocket.CAM_C

        rgb_size = (int(config.rt_width), int(config.rt_height))
        stereo_size = (640, 480)

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        model_path = Path(config.rt_model_path)
        if not model_path.exists():
            model_path = (Path(__file__).resolve().parents[1] / config.rt_model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose Landmarker model not found at '{config.rt_model_path}' (also tried '{model_path}')"
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        rt_device = dai.Device()
        rt_pipeline = dai.Pipeline(rt_device)
        rt_landmarker = PoseLandmarker.create_from_options(options)

        cam_rgb = rt_pipeline.create(dai.node.Camera).build(rgb_socket)
        left = rt_pipeline.create(dai.node.Camera).build(left_socket)
        right = rt_pipeline.create(dai.node.Camera).build(right_socket)

        stereo = rt_pipeline.create(dai.node.StereoDepth)
        sync = rt_pipeline.create(dai.node.Sync)
        stereo.setExtendedDisparity(True)
        sync.setSyncThreshold(timedelta(seconds=1 / (2 * max(1.0, float(config.rt_fps)))))

        video_stream = cam_rgb.requestOutput(size=rgb_size, fps=float(config.rt_fps), enableUndistortion=True)
        left.requestOutput(size=stereo_size, fps=float(config.rt_fps)).link(stereo.left)
        right.requestOutput(size=stereo_size, fps=float(config.rt_fps)).link(stereo.right)

        video_stream.link(stereo.inputAlignTo)
        video_stream.link(sync.inputs["rgb"])
        stereo.depth.link(sync.inputs["depth_aligned"])

        rt_queue = sync.out.createOutputQueue()

        calib = rt_device.readCalibration()
        K = calib.getCameraIntrinsics(rgb_socket, rgb_size[0], rgb_size[1])
        rt_fx, rt_fy, rt_cx, rt_cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])

        rt_pipeline.start()

        if bool(config.rt_show_window):
            rt_cv2.namedWindow(str(config.rt_window_name), rt_cv2.WINDOW_NORMAL)
            if bool(config.rt_show_depth):
                rt_cv2.namedWindow(f"{config.rt_window_name} (depth)", rt_cv2.WINDOW_NORMAL)

        if bool(config.rt_record_video):
            p = Path(config.rt_video_path) if str(config.rt_video_path).strip() else (out_dir / "rt_video.mp4")
            fourcc = rt_cv2.VideoWriter_fourcc(*"mp4v")
            rt_writer = rt_cv2.VideoWriter(str(p), fourcc, float(config.rt_fps), rgb_size)

    try:
        for i in range(n_iters):
            iter_sched_ns = next_tick_ns
            now_ns = time.monotonic_ns()
            schedule_error_ms = (now_ns - iter_sched_ns) * 1e-6

            if prev_iter_ns is None:
                actual_period_ms = float("nan")
            else:
                actual_period_ms = (now_ns - prev_iter_ns) * 1e-6
            prev_iter_ns = now_ns

            t_iter0 = time.perf_counter_ns()

            # 1) Online pose input (1 frame)
            t0 = time.perf_counter_ns()
            if not use_realtime:
                idx = int(i % seq_full.shape[0])
                frame = np.asarray(seq_full[idx], dtype=np.float64)
            else:
                assert rt_queue is not None and rt_landmarker is not None and rt_cv2 is not None and rt_mp is not None
                # Block for a synced rgb+depth sample, then run pose estimation and 3D deprojection.
                msg_group = rt_queue.getAll()[-1]
                frame_rgb = msg_group["rgb"]
                frame_depth = msg_group["depth_aligned"]

                frame_bgr = frame_rgb.getCvFrame()
                depth_mm = frame_depth.getFrame()

                t_sec = frame_rgb.getTimestampDevice().total_seconds()
                ts_ms = int(t_sec * 1000)
                if ts_ms <= rt_last_ts_ms:
                    ts_ms = rt_last_ts_ms + 1
                rt_last_ts_ms = ts_ms

                frame_rgb_np = rt_cv2.cvtColor(frame_bgr, rt_cv2.COLOR_BGR2RGB)
                mp_image = rt_mp.Image(image_format=rt_mp.ImageFormat.SRGB, data=frame_rgb_np)
                result = rt_landmarker.detect_for_video(mp_image, ts_ms)

                pose_xyz = np.full((len(POSE_KEYPOINT_IDS), 3), np.nan, dtype=np.float64)
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    pose0 = result.pose_landmarks[0]
                    h, w = frame_bgr.shape[:2]
                    for j, idx in enumerate(POSE_KEYPOINT_IDS):
                        lm = pose0[idx]
                        u, v = float(lm.x) * w, float(lm.y) * h
                        z_m = _rt_depth_at(depth_mm, u, v, patch=int(config.rt_patch))
                        if z_m is None or z_m < float(config.rt_min_z) or z_m > float(config.rt_max_z):
                            continue
                        pose_xyz[j] = _rt_deproject(u, v, float(z_m))

                if rt_writer is not None:
                    rt_writer.write(frame_bgr)

                if bool(config.rt_show_window):
                    rt_cv2.imshow(str(config.rt_window_name), frame_bgr)
                    if bool(config.rt_show_depth):
                        depth_vis = rt_cv2.normalize(depth_mm, None, 0, 255, rt_cv2.NORM_MINMAX, dtype=rt_cv2.CV_8U)
                        depth_vis = rt_cv2.applyColorMap(depth_vis, rt_cv2.COLORMAP_INFERNO)
                        rt_cv2.imshow(f"{config.rt_window_name} (depth)", depth_vis)
                    key = rt_cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                frame = pose_xyz
            pose_ms = (time.perf_counter_ns() - t0) * 1e-6

            # Update window
            window.append(frame)
            if len(window) > W:
                window.pop(0)

            # 2) Preprocess window (interpolate + lowpass in cartesian space)
            t0 = time.perf_counter_ns()
            win = np.stack(window, axis=0)  # (W,K,3)
            win_i = interpolate_keypoints_cartesian(win)
            win_f = lowpass_keypoints(
                win_i, fps=online_fps, cutoff_hz=float(config.lowpass_cutoff_hz), order=int(config.lowpass_order)
            )
            preprocess_ms = (time.perf_counter_ns() - t0) * 1e-6

            # 3) Extract human joint angles (degrees)
            t0 = time.perf_counter_ns()
            q_win = sequence_to_angles_deg(win_f)
            q_win = _clip_angles_deg(q_win)
            q_human = np.asarray(q_win[-1], dtype=float).reshape(-1)
            if not np.all(np.isfinite(q_human)):
                q_human = q_h_prev.copy() if q_h_prev is not None else q_start.copy()
            angle_extraction_ms = (time.perf_counter_ns() - t0) * 1e-6

            # 4) Estimate qdot_human (filtered finite differences)
            if q_h_prev is None or not np.all(np.isfinite(q_h_prev)):
                qdot_human = qdot_h_prev.copy()
            else:
                qdot_human = _finite_diff_filtered(
                    q_prev=q_h_prev,
                    q_curr=q_human,
                    dt=dt_loop,
                    qdot_prev=qdot_h_prev,
                    alpha=float(config.qdot_alpha),
                )
            if not np.all(np.isfinite(qdot_human)):
                qdot_human = qdot_h_prev.copy()

            q_h_prev = q_human.copy()
            qdot_h_prev = qdot_human.copy()

            # Initialize DMP state from first valid human state (do NOT overwrite later)
            if q_dmp is None:
                q_dmp = q_human.copy()
                qdot_dmp = np.zeros_like(q_dmp)
                q0_dmp = q_dmp.copy()

            # 5) Phase estimation
            t0 = time.perf_counter_ns()
            progress = float("nan")
            if config.phase_mode == "time":
                t_s = (time.monotonic_ns() - loop_start_ns) * 1e-9
                x = float(
                    canonical_phase(float(t_s), tau=float(config.tau), alpha_canonical=float(config.alpha_canonical))
                )
            elif config.phase_mode == "human-progress":
                progress, x = _human_progress_phase(
                    q_human, q_start=q_start, q_goal=q_goal, alpha_canonical=float(config.alpha_canonical)
                )
            elif config.phase_mode == "path-progress":
                progress, x, active_segment, local_progress = _path_progress_phase(
                    q_human,
                    waypoints=waypoints,
                    active_segment=active_segment,
                    alpha_canonical=float(config.alpha_canonical),
                    eps=float(config.progress_eps),
                )
            else:
                raise ValueError("phase_mode must be 'time', 'human-progress', or 'path-progress'")
            x = float(np.clip(x, 0.0, 1.0))
            phase_estimation_ms = (time.perf_counter_ns() - t0) * 1e-6

            # 6) DMP nominal step (internal state)
            t0 = time.perf_counter_ns()
            ddq_nom = _dmp_nominal_ddq(
                model,
                q=q_dmp,
                dq=qdot_dmp,
                q0=q0_dmp,
                g=g_dmp,
                x=x,
                tau=float(config.tau),
            )
            dmp_step_ms = (time.perf_counter_ns() - t0) * 1e-6

            # 7) Optional coupling
            t0 = time.perf_counter_ns()
            ddq = ddq_nom
            if config.coupling_mode == "none":
                pass
            elif config.coupling_mode == "pd":
                C = float(config.kp) * (q_human - q_dmp) + float(config.kd) * (qdot_human - qdot_dmp)
                ddq = ddq_nom + float(config.autonomy_gain) * C
            else:
                raise ValueError("coupling_mode must be 'none' or 'pd'")
            coupling_ms = (time.perf_counter_ns() - t0) * 1e-6

            # 8) Integrate DMP state (Euler)
            qdot_dmp = qdot_dmp + ddq * dt_loop
            q_dmp = q_dmp + qdot_dmp * dt_loop
            q_robot_desired = q_dmp.copy()

            # 9) Communication
            t0 = time.perf_counter_ns()
            if config.comm_mode == "none":
                pass
            elif config.comm_mode == "sleep":
                if config.comm_sleep_ms > 0:
                    time.sleep(float(config.comm_sleep_ms) * 1e-3)
            elif config.comm_mode == "can":
                ok = bool(send_can_msg(q_robot_desired))  # type: ignore[misc]
                _ = ok
            else:
                raise ValueError("comm_mode must be 'none', 'sleep', or 'can'")
            comm_ms = (time.perf_counter_ns() - t0) * 1e-6

            # Timings
            e2e_ms = (time.perf_counter_ns() - t_iter0) * 1e-6
            loop_exec_ms = e2e_ms

            # Tracking error (optional diagnostic)
            tracking_err_l2 = float(np.linalg.norm(q_human - q_dmp))

            rec["pose_ms"].append(float(pose_ms))
            rec["preprocess_ms"].append(float(preprocess_ms))
            rec["angle_extraction_ms"].append(float(angle_extraction_ms))
            rec["phase_estimation_ms"].append(float(phase_estimation_ms))
            rec["dmp_step_ms"].append(float(dmp_step_ms))
            rec["coupling_ms"].append(float(coupling_ms))
            rec["comm_ms"].append(float(comm_ms))
            rec["e2e_ms"].append(float(e2e_ms))
            rec["loop_exec_ms"].append(float(loop_exec_ms))
            rec["actual_period_ms"].append(float(actual_period_ms))
            rec["schedule_error_ms"].append(float(schedule_error_ms))
            rec["tracking_err_l2"].append(float(tracking_err_l2))
            rec["phase_x"].append(float(x))
            rec["progress"].append(float(progress))

            # Fixed-period scheduling
            next_tick_ns += period_ns
            sleep_ns = next_tick_ns - time.monotonic_ns()
            if sleep_ns > 0:
                time.sleep(sleep_ns * 1e-9)
    finally:
        # Cleanup real-time resources (best-effort).
        try:
            if rt_writer is not None:
                rt_writer.release()
        except Exception:
            pass
        try:
            if rt_cv2 is not None and bool(config.rt_show_window):
                rt_cv2.destroyWindow(str(config.rt_window_name))
                if bool(config.rt_show_depth):
                    rt_cv2.destroyWindow(f"{config.rt_window_name} (depth)")
        except Exception:
            pass
        try:
            if rt_landmarker is not None:
                rt_landmarker.close()
        except Exception:
            pass
        try:
            if rt_pipeline is not None:
                rt_pipeline.stop()
        except Exception:
            pass
        try:
            if rt_device is not None:
                rt_device.close()
        except Exception:
            pass

    # --------------------------
    # OUTPUTS: timing.csv, summary.json, timing_plot.png
    # --------------------------
    import pandas as pd  # local import to keep API light
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rec)
    timing_csv = out_dir / "timing.csv"
    df.to_csv(timing_csv, index=False)

    budgets_map = {
        "pose_ms": budgets.pose_ms,
        "preprocess_ms": budgets.preprocess_ms,
        "angle_extraction_ms": budgets.angle_ms,
        "phase_estimation_ms": budgets.phase_ms,
        "dmp_step_ms": budgets.dmp_step_ms,
        "coupling_ms": budgets.coupling_ms,
        "comm_ms": budgets.comm_ms,
        "e2e_ms": budgets.e2e_ms,
    }

    summary_stages: dict[str, Any] = {}
    for k, b in budgets_map.items():
        stats = _stage_stats_ms(df[k].to_numpy())
        stats["budget_ms"] = float(b)
        stats["miss_rate"] = _miss_rate(df[k].to_numpy(), float(b))
        summary_stages[k] = stats

    # Additional loop metrics
    schedule_stats = _stage_stats_ms(df["schedule_error_ms"].to_numpy())
    period_stats = _stage_stats_ms(df["actual_period_ms"].to_numpy())

    summary = {
        "offline_fit_ms": float(offline_fit_ms),
        "stages": summary_stages,
        "loop": {
            "target_period_ms": float(config.period_ms),
            "actual_period_ms": period_stats,
            "schedule_error_ms": schedule_stats,
        },
        "tracking": {
            "tracking_err_l2": _stage_stats_ms(df["tracking_err_l2"].to_numpy()),
        },
    }

    summary_json = out_dir / "summary.json"
    _json_dump(
        summary_json,
        {
            "config": asdict(config),
            "budgets_ms": asdict(budgets),
            "paths": {
                "timing_csv": str(timing_csv),
                "summary_json": str(summary_json),
                "timing_plot_png": str(out_dir / "timing_plot.png"),
                "offline_model_npz": str(out_dir / "dmp_model_offline.npz"),
            },
            "summary": summary,
        },
    )

    # Plot: latency vs budget, period vs target, schedule error, tracking error
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    t_idx = np.arange(len(df), dtype=float)

    axs[0].plot(t_idx, df["e2e_ms"], label="e2e_ms")
    axs[0].axhline(float(budgets.e2e_ms), color="r", linestyle="--", label="budget")
    axs[0].set_ylabel("e2e (ms)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t_idx, df["actual_period_ms"], label="actual_period_ms")
    axs[1].axhline(float(config.period_ms), color="k", linestyle="--", label="target")
    axs[1].set_ylabel("period (ms)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t_idx, df["schedule_error_ms"], label="schedule_error_ms")
    axs[2].set_ylabel("sched err (ms)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(t_idx, df["tracking_err_l2"], label="||q_human - q_dmp||")
    axs[3].set_ylabel("tracking err")
    axs[3].set_xlabel("iteration")
    axs[3].legend(loc="upper right")
    axs[3].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = out_dir / "timing_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    result = ClassicalDMPTimingResult(
        config=asdict(config),
        budgets_ms=asdict(budgets),
        paths={
            "timing_csv": str(timing_csv),
            "summary_json": str(summary_json),
            "timing_plot_png": str(plot_path),
            "offline_model_npz": str(out_dir / "dmp_model_offline.npz"),
        },
        offline_fit_ms=float(offline_fit_ms),
        summary=summary,
    )
    return {
        "config": result.config,
        "budgets_ms": result.budgets_ms,
        "paths": result.paths,
        "offline_fit_ms": result.offline_fit_ms,
        "summary": result.summary,
    }

