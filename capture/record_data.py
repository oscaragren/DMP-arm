"""
Trial recording helper used by `collect_data.py`.

Records RGB video from an OAK-D (DepthAI) device, runs MediaPipe Pose Landmarker
online, and stores:
- video.mp4 (with keypoint overlay)
- left_arm_seq_camera.npy (T, K, 3) camera-frame XYZ in meters (NaNs when invalid)
- left_arm_t.npy (T,) DepthAI device timestamps in seconds
- meta.json
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import json
import time

import cv2
import numpy as np


POSE_KEYPOINT_IDS = [11, 13, 15, 12, 23, 24]
POSE_KEYPOINT_NAMES = {
    11: "left_shoulder",
    13: "left_elbow",
    15: "left_wrist",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
}


@dataclass(frozen=True)
class RecordConfig:
    fps: float = 25.0
    width: int = 640#1280
    height: int = 480# 720
    record_duration_sec: float = 8.0
    pre_record_delay_sec: float = 3.0
    n_frames: Optional[int] = None
    model_path: str = "capture/pose_landmarker_heavy.task"
    patch: int = 3 
    min_z: float = 0.0
    max_z: float = 3.0
    show_window: bool = True
    show_depth: bool = False
    window_name: str = "Recording"


def _draw_banner(frame_bgr: np.ndarray, text: str) -> None:
    cv2.putText(
        frame_bgr,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _deproject(u: float, v: float, z_m: float, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    x = (u - cx) * z_m / fx
    y = (v - cy) * z_m / fy
    return float(x), float(y), float(z_m)


def _depth_at(depth_mm: np.ndarray, u: float, v: float, *, patch: int) -> Optional[float]:
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


def record_data(*, args, trial_dir: str | Path, trial: int) -> bool:
    """
    Record one trial into `trial_dir`.

    Returns True on success, False if aborted.
    """
    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    cfg = RecordConfig(
        fps=float(getattr(args, "fps_nominal", 30.0) or 30.0),
        record_duration_sec=float(getattr(args, "record_duration_sec", 8.0) or 8.0),
        pre_record_delay_sec=float(getattr(args, "pre_record_delay_sec", 3.0) or 3.0),
        n_frames=(int(getattr(args, "n_frames")) if getattr(args, "n_frames", None) is not None else None),
    )

    # Optional args (only used if present in Namespace)
    cfg = RecordConfig(
        fps=cfg.fps,
        record_duration_sec=cfg.record_duration_sec,
        pre_record_delay_sec=cfg.pre_record_delay_sec,
        n_frames=cfg.n_frames,
        width=int(getattr(args, "width", cfg.width) or cfg.width),
        height=int(getattr(args, "height", cfg.height) or cfg.height),
        model_path=str(getattr(args, "model", cfg.model_path) or cfg.model_path),
        patch=int(getattr(args, "patch", cfg.patch) or cfg.patch),
        min_z=float(getattr(args, "min_z", cfg.min_z) if hasattr(args, "min_z") else cfg.min_z),
        max_z=float(getattr(args, "max_z", cfg.max_z) if hasattr(args, "max_z") else cfg.max_z),
        show_window=not bool(getattr(args, "no_show", False)),
        show_depth=bool(getattr(args, "show_depth", False)),
        window_name=str(getattr(args, "window_name", cfg.window_name) or cfg.window_name),
    )

    # Heavy imports here so `collect_data.py --help` stays snappy.
    import depthai as dai
    import mediapipe as mp

    rgb_socket = dai.CameraBoardSocket.CAM_A
    left_socket = dai.CameraBoardSocket.CAM_B
    right_socket = dai.CameraBoardSocket.CAM_C

    rgb_size = (int(cfg.width), int(cfg.height))
    stereo_size = (640, 480)
    # MediaPipe Pose Landmarker (Tasks API)
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        # Try relative to repo root if called from elsewhere
        model_path = (Path(__file__).resolve().parents[1] / cfg.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Pose Landmarker model not found at '{cfg.model_path}' (also tried '{model_path}')")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    video_path = trial_dir / "video.mp4"
    npy_seq_path = trial_dir / "left_arm_seq_camera.npy"
    npy_t_path = trial_dir / "left_arm_t.npy"
    meta_path = trial_dir / "meta.json"

    all_frames: list[np.ndarray] = []
    all_t: list[float] = []

    device = dai.Device()
    with dai.Pipeline(device) as pipeline, PoseLandmarker.create_from_options(options) as landmarker:
        cam_rgb = pipeline.create(dai.node.Camera).build(rgb_socket)
        left = pipeline.create(dai.node.Camera).build(left_socket)
        right = pipeline.create(dai.node.Camera).build(right_socket)

        stereo = pipeline.create(dai.node.StereoDepth)
        sync = pipeline.create(dai.node.Sync)
        stereo.setExtendedDisparity(True)
        sync.setSyncThreshold(timedelta(seconds=1 / (2 * max(1.0, cfg.fps))))

        video_stream = cam_rgb.requestOutput(size=rgb_size, fps=float(cfg.fps), enableUndistortion=True)
        left.requestOutput(size=stereo_size, fps=float(cfg.fps)).link(stereo.left)
        right.requestOutput(size=stereo_size, fps=float(cfg.fps)).link(stereo.right)

        video_stream.link(stereo.inputAlignTo)
        video_stream.link(sync.inputs["rgb"])
        stereo.depth.link(sync.inputs["depth_aligned"]) # aligned because of the sync node

        queue = sync.out.createOutputQueue()

        calib = device.readCalibration()
        K = calib.getCameraIntrinsics(rgb_socket, rgb_size[0], rgb_size[1])
        fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])

        pipeline.start()

        if cfg.show_window:
            cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
            if cfg.show_depth:
                cv2.namedWindow(f"{cfg.window_name} (depth)", cv2.WINDOW_NORMAL)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, float(cfg.fps), rgb_size)

        # Operator flow:
        # - show live feed
        # - SPACE starts countdown/warmup (autofocus/exposure)
        # - after countdown, start recording
        waiting_for_space = bool(cfg.show_window)
        warmup_start: Optional[float] = None
        record_start: Optional[float] = None
        last_ts_ms = -1
        aborted = False

        try:
            while pipeline.isRunning():
                msg_group = queue.get()
                frame_rgb = msg_group["rgb"]
                frame_depth = msg_group["depth_aligned"]

                frame_bgr = frame_rgb.getCvFrame()
                depth_mm = frame_depth.getFrame()

                t_sec = frame_rgb.getTimestampDevice().total_seconds()
                ts_ms = int(t_sec * 1000)
                if ts_ms <= last_ts_ms:
                    ts_ms = last_ts_ms + 1
                last_ts_ms = ts_ms

                if waiting_for_space:
                    view = frame_bgr.copy()
                    _draw_banner(view, "Press SPACE to start countdown (q to abort)")
                    cv2.imshow(cfg.window_name, view)
                    if cfg.show_depth:
                        depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                        cv2.imshow(f"{cfg.window_name} (depth)", depth_vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        aborted = True
                        break
                    if key == ord(" "):
                        waiting_for_space = False
                        warmup_start = time.time()
                    continue

                # Headless mode: start countdown immediately.
                if warmup_start is None:
                    warmup_start = time.time()

                # Warmup/countdown period for autofocus/exposure stabilization.
                warmup_elapsed = time.time() - float(warmup_start)
                if warmup_elapsed < float(cfg.pre_record_delay_sec):
                    if cfg.show_window:
                        remaining = max(0.0, float(cfg.pre_record_delay_sec) - warmup_elapsed)
                        view = frame_bgr.copy()
                        _draw_banner(view, f"Warmup... starting in {remaining:0.1f}s (q to abort)")
                        cv2.imshow(cfg.window_name, view)
                        if cfg.show_depth:
                            depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                            cv2.imshow(f"{cfg.window_name} (depth)", depth_vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            aborted = True
                            break
                    continue

                if record_start is None:
                    record_start = time.time()

                # Convert BGR -> RGB for MediaPipe
                frame_rgb_np = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb_np)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                pose_xyz = np.full((len(POSE_KEYPOINT_IDS), 3), np.nan, dtype=np.float32)
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    pose0 = result.pose_landmarks[0]
                    h, w = frame_bgr.shape[:2]
                    for j, idx in enumerate(POSE_KEYPOINT_IDS):
                        lm = pose0[idx]
                        u, v = float(lm.x) * w, float(lm.y) * h
                        z_m = _depth_at(depth_mm, u, v, patch=cfg.patch)
                        if z_m is None or z_m < cfg.min_z or z_m > cfg.max_z:
                            continue
                        pose_xyz[j] = _deproject(u, v, z_m, fx, fy, cx, cy)

                        # overlay
                        cv2.circle(frame_bgr, (int(u), int(v)), 3, (0, 255, 0), -1)
                        cv2.putText(
                            frame_bgr,
                            f"{POSE_KEYPOINT_NAMES[idx]} z={z_m:.2f}m",
                            (int(u) + 5, int(v) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                elapsed = time.time() - float(record_start)
                _draw_banner(frame_bgr, f"Recording...  t={elapsed:0.1f}s   (q to abort)")

                writer.write(frame_bgr)
                all_frames.append(pose_xyz)
                all_t.append(t_sec)

                if cfg.show_window:
                    cv2.imshow(cfg.window_name, frame_bgr)
                    if cfg.show_depth:
                        depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                        cv2.imshow(f"{cfg.window_name} (depth)", depth_vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        aborted = True
                        break

                if cfg.n_frames is not None and len(all_frames) >= int(cfg.n_frames):
                    break
                if elapsed >= float(cfg.record_duration_sec):
                    break
        finally:
            writer.release()
            if cfg.show_window:
                cv2.destroyWindow(cfg.window_name)
                if cfg.show_depth:
                    cv2.destroyWindow(f"{cfg.window_name} (depth)")

    if aborted:
        return False

    seq = np.stack(all_frames, axis=0) if all_frames else np.zeros((0, len(POSE_KEYPOINT_IDS), 3), dtype=np.float32)
    t = np.array(all_t, dtype=np.float64)
    np.save(npy_seq_path, seq)
    np.save(npy_t_path, t)

    meta = {
        "subject": int(getattr(args, "subject", -1)) if getattr(args, "subject", None) is not None else None,
        "motion": getattr(args, "motion", None),
        "trial": trial,
        "trial_dir": str(trial_dir),
        "shape": [int(x) for x in seq.shape],
        "keypoint_names": [POSE_KEYPOINT_NAMES[i] for i in POSE_KEYPOINT_IDS],
        "video": "video.mp4",
        "fps_nominal": float(cfg.fps),
        "record_duration_sec": float(cfg.record_duration_sec),
        "n_frames": int(cfg.n_frames) if cfg.n_frames is not None else None,
        "pose_model": {
            "framework": "MediaPipe Pose Landmarker (Tasks API)",
            "model_asset_path": str(model_path),
            "running_mode": "VIDEO",
            "num_poses": 1,
        },
        "depth_sampling": {
            "patch": int(cfg.patch),
            "min_z": float(cfg.min_z),
            "max_z": float(cfg.max_z),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return True