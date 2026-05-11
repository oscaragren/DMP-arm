"""
DepthAI real-time pose helper used by experiments.

This file started as a standalone demo. It is now an importable API that provides:
- synced RGB + aligned depth frames from an OAK-D
- pose keypoints from a YOLO pose model (e.g. yolo11n-pose)
- 3D camera-frame XYZ in meters (NaNs when invalid)

The experiment code consumes the same keypoint layout as elsewhere in the repo:
`POSE_KEYPOINT_IDS = [11, 13, 15, 12, 23, 24]` (shoulder/elbow/wrist + torso refs).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np

# Heavy deps are imported lazily in `start()` so that importing this module is cheap.


# COCO-17 ids used across this repo for "left arm frame"
POSE_KEYPOINT_IDS = [11, 13, 15, 12, 23, 24]

# The YOLO-pose models typically follow COCO ordering:
# 5=Lshoulder, 7=Lelbow, 9=Lwrist, 6=Rshoulder, 11=Lhip, 12=Rhip
_YOLO_LEFT_ARM_IDS = [5, 7, 9, 6, 11, 12]


@dataclass(frozen=True)
class RealtimeYoloPose3DConfig:
    fps: float = 25.0
    rgb_w: int = 512
    rgb_h: int = 288
    stereo_w: int = 640
    stereo_h: int = 480
    model_path: str = "models/yolo11n-pose-512x288.rvc2.tar.xz"
    depth_patch: int = 5
    min_z_m: float = 0.15
    max_z_m: float = 4.0

    # Optional visualization
    show_window: bool = False
    show_depth: bool = False
    window_name: str = "YOLO Pose 3D"


def _point_to_pixel(x: float, y: float, frame_shape: tuple[int, int, int]) -> tuple[int, int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    u = x * w if 0.0 <= float(x) <= 1.0 else float(x)
    v = y * h if 0.0 <= float(y) <= 1.0 else float(y)
    u_i = int(np.clip(round(u), 0, w - 1))
    v_i = int(np.clip(round(v), 0, h - 1))
    return u_i, v_i


def _depth_at(
    depth_mm: np.ndarray,
    u: int,
    v: int,
    *,
    patch: int,
    min_z_m: float,
    max_z_m: float,
) -> float:
    h, w = depth_mm.shape[:2]
    u = int(np.clip(int(u), 0, w - 1))
    v = int(np.clip(int(v), 0, h - 1))

    r = max(0, int(patch) // 2)
    x1, x2 = max(0, u - r), min(w, u + r + 1)
    y1, y2 = max(0, v - r), min(h, v + r + 1)

    roi = depth_mm[y1:y2, x1:x2].astype(np.float32)
    roi = roi[roi > 0]
    if roi.size == 0:
        return float("nan")

    z_m = float(np.median(roi)) / 1000.0
    if not np.isfinite(z_m) or z_m < float(min_z_m) or z_m > float(max_z_m):
        return float("nan")
    return float(z_m)


def _deproject(u: float, v: float, z_m: float, *, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    x = (float(u) - float(cx)) * float(z_m) / float(fx)
    y = (float(v) - float(cy)) * float(z_m) / float(fy)
    return float(x), float(y), float(z_m)


class RealtimeYoloPose3D:
    """
    Minimal real-time API:
      cap = RealtimeYoloPose3D(cfg); cap.start()
      xyz = cap.read()   # (6,3) float64, NaNs when invalid
      cap.stop()
    """

    def __init__(self, cfg: RealtimeYoloPose3DConfig):
        self.cfg = cfg
        self._device = None
        self._pipeline = None
        self._sync_q = None
        self._det_q = None
        self._cv2 = None
        self._fx = self._fy = self._cx = self._cy = None
        self._last_frame_bgr = None
        self._last_depth_mm = None

    def start(self) -> None:
        import cv2  # type: ignore
        import depthai as dai  # type: ignore
        from depthai_nodes.node import ParsingNeuralNetwork  # type: ignore

        self._cv2 = cv2
        self._device = dai.Device()
        platform = self._device.getPlatform().name
        img_frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p

        rgb_size = (int(self.cfg.rgb_w), int(self.cfg.rgb_h))
        stereo_size = (int(self.cfg.stereo_w), int(self.cfg.stereo_h))
        fps = float(self.cfg.fps)

        pipeline = dai.Pipeline(self._device)

        rgb_socket = dai.CameraBoardSocket.CAM_A
        left_socket = dai.CameraBoardSocket.CAM_B
        right_socket = dai.CameraBoardSocket.CAM_C

        cam = pipeline.create(dai.node.Camera).build(rgb_socket)
        left = pipeline.create(dai.node.Camera).build(left_socket)
        right = pipeline.create(dai.node.Camera).build(right_socket)

        stereo = pipeline.create(dai.node.StereoDepth)
        sync = pipeline.create(dai.node.Sync)

        stereo.setExtendedDisparity(True)
        sync.setSyncThreshold(timedelta(seconds=1 / (2 * max(1.0, fps))))

        rgb_out = cam.requestOutput(rgb_size, type=img_frame_type, fps=fps, enableUndistortion=True)
        left.requestOutput(stereo_size, fps=fps).link(stereo.left)
        right.requestOutput(stereo_size, fps=fps).link(stereo.right)

        # Align depth to RGB
        rgb_out.link(stereo.inputAlignTo)

        model_path = Path(self.cfg.model_path)
        if not model_path.exists():
            model_path = (Path(__file__).resolve().parents[1] / self.cfg.model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO pose model archive not found at '{self.cfg.model_path}' (also tried '{model_path}')")

        nn_archive = dai.NNArchive(str(model_path))
        nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(rgb_out, nn_archive)

        nn_with_parser.passthrough.link(sync.inputs["rgb"])
        stereo.depth.link(sync.inputs["depth_aligned"])

        self._sync_q = sync.out.createOutputQueue(maxSize=4, blocking=False)
        self._det_q = nn_with_parser.out.createOutputQueue(maxSize=4, blocking=False)

        # Intrinsics for deprojection
        calib = self._device.readCalibration()
        K = calib.getCameraIntrinsics(rgb_socket, rgb_size[0], rgb_size[1])
        self._fx, self._fy, self._cx, self._cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])

        self._pipeline = pipeline
        self._pipeline.start()

        if bool(self.cfg.show_window):
            cv2.namedWindow(str(self.cfg.window_name), cv2.WINDOW_NORMAL)
            if bool(self.cfg.show_depth):
                cv2.namedWindow(f"{self.cfg.window_name} (depth)", cv2.WINDOW_NORMAL)

    def read(self) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("RealtimeYoloPose3D is not started. Call start() first.")
        assert self._sync_q is not None and self._det_q is not None

        sync_msg = self._sync_q.tryGet()
        if sync_msg is not None:
            rgb_msg = sync_msg["rgb"]
            depth_msg = sync_msg["depth_aligned"]
            self._last_frame_bgr = rgb_msg.getCvFrame()
            self._last_depth_mm = depth_msg.getFrame()

        det_msg = self._det_q.tryGet()
        if det_msg is None or self._last_frame_bgr is None:
            return np.full((len(POSE_KEYPOINT_IDS), 3), np.nan, dtype=np.float64)

        dets = det_msg.detections
        if len(dets) == 0:
            return np.full((len(POSE_KEYPOINT_IDS), 3), np.nan, dtype=np.float64)

        # Pick most confident person/pose
        det = max(dets, key=lambda d: float(d.confidence))
        print(type(det))

        print("detection fields:", dir(det))
        kpts = det.getKeypoints2f()

        xyz = np.full((len(POSE_KEYPOINT_IDS), 3), np.nan, dtype=np.float64)
        depth_mm = self._last_depth_mm
        assert self._fx is not None and self._fy is not None and self._cx is not None and self._cy is not None

        for j, yolo_idx in enumerate(_YOLO_LEFT_ARM_IDS):
            if yolo_idx >= len(kpts):
                continue
            u, v = _point_to_pixel(float(kpts[yolo_idx].x), float(kpts[yolo_idx].y), self._last_frame_bgr.shape)
            z_m = float("nan") if depth_mm is None else _depth_at(
                depth_mm,
                u,
                v,
                patch=int(self.cfg.depth_patch),
                min_z_m=float(self.cfg.min_z_m),
                max_z_m=float(self.cfg.max_z_m),
            )
            if not np.isfinite(z_m):
                continue
            xyz[j] = _deproject(u, v, z_m, fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy)

        if self._cv2 is not None and bool(self.cfg.show_window) and self._last_frame_bgr is not None:
            frame = self._last_frame_bgr
            for j in range(xyz.shape[0]):
                if not np.all(np.isfinite(xyz[j])):
                    continue
                # Re-project only for display label (we already have u,v above but keep it simple)
            self._cv2.imshow(str(self.cfg.window_name), frame)
            if bool(self.cfg.show_depth) and depth_mm is not None:
                depth_vis = self._cv2.normalize(depth_mm, None, 0, 255, self._cv2.NORM_MINMAX, dtype=self._cv2.CV_8U)
                depth_vis = self._cv2.applyColorMap(depth_vis, self._cv2.COLORMAP_INFERNO)
                self._cv2.imshow(f"{self.cfg.window_name} (depth)", depth_vis)
            _ = self._cv2.waitKey(1) & 0xFF

        return xyz

    def stop(self) -> None:
        try:
            if self._pipeline is not None:
                self._pipeline.stop()
        finally:
            self._pipeline = None
        try:
            if self._device is not None:
                self._device.close()
        finally:
            self._device = None
        try:
            if self._cv2 is not None and bool(self.cfg.show_window):
                self._cv2.destroyWindow(str(self.cfg.window_name))
                if bool(self.cfg.show_depth):
                    self._cv2.destroyWindow(f"{self.cfg.window_name} (depth)")
        except Exception:
            pass


def _demo() -> None:
    cfg = RealtimeYoloPose3DConfig(show_window=True, show_depth=True)
    cap = RealtimeYoloPose3D(cfg)
    cap.start()
    try:
        while True:
            xyz = cap.read()
            if np.any(np.isfinite(xyz)):
                print("left_arm_frame [X,Y,Z] (m):")
                print(xyz)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()


if __name__ == "__main__":
    _demo()