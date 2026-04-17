"""
Test script.

Detect hand open or close using mediapipe and DepthAI depth camera.
"""

from __future__ import annotations

import math
import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

# When you run `python3 tests/test_hand.py`, Python sets sys.path[0]
# to the `tests/` folder. Add the repository root so imports work.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@dataclass(frozen=True)
class HandState:
    label: str  # "open" | "closed" | "unknown"
    extended_fingers: int
    extended: dict[str, bool]
    score: float


def _as_vec3(lm) -> np.ndarray:
    return np.array([float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))], dtype=np.float64)


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle ABC in degrees, robust to degenerate vectors.
    """
    ba = a - b
    bc = c - b
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba < 1e-12 or nbc < 1e-12:
        return 0.0
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = max(-1.0, min(1.0, cosang))
    return float(math.degrees(math.acos(cosang)))


def _finger_extended_by_angle(
    lms: list,
    *,
    mcp: int,
    pip: int,
    tip: int,
    threshold_deg: float,
) -> bool:
    a = _as_vec3(lms[mcp])
    b = _as_vec3(lms[pip])
    c = _as_vec3(lms[tip])
    return _angle_deg(a, b, c) >= float(threshold_deg)


def classify_hand_open_closed(
    hand_landmarks,
    *,
    finger_threshold_deg: float = 160.0,
    thumb_threshold_deg: float = 155.0,
    open_min_extended: int = 4,
) -> HandState:
    """
    Classify a single MediaPipe hand into open/closed using joint angles.

    This is orientation-agnostic compared to y-based heuristics.
    """
    if hand_landmarks is None:
        return HandState(label="unknown", extended_fingers=0, extended={}, score=0.0)

    lms = list(getattr(hand_landmarks, "landmark", hand_landmarks))
    if len(lms) < 21:
        return HandState(label="unknown", extended_fingers=0, extended={}, score=0.0)

    ext = {
        "index": _finger_extended_by_angle(lms, mcp=5, pip=6, tip=8, threshold_deg=finger_threshold_deg),
        "middle": _finger_extended_by_angle(lms, mcp=9, pip=10, tip=12, threshold_deg=finger_threshold_deg),
        "ring": _finger_extended_by_angle(lms, mcp=13, pip=14, tip=16, threshold_deg=finger_threshold_deg),
        "pinky": _finger_extended_by_angle(lms, mcp=17, pip=18, tip=20, threshold_deg=finger_threshold_deg),
        # Thumb: use MCP-IP-TIP angle (2-3-4) as a simple proxy.
        "thumb": _finger_extended_by_angle(lms, mcp=2, pip=3, tip=4, threshold_deg=thumb_threshold_deg),
    }

    extended_fingers = int(sum(bool(v) for v in ext.values()))

    # Score encourages "open" when many fingers are extended.
    score = extended_fingers / 5.0
    label = "open" if extended_fingers >= int(open_min_extended) else "closed"
    return HandState(label=label, extended_fingers=extended_fingers, extended=ext, score=float(score))


def _iter_frames_depthai(*, width: int, height: int, fps: float):
    """
    Yield (BGR frame, timestamp_ms) from an OAK device using DepthAI.
    """
    import depthai as dai

    device = dai.Device()
    with dai.Pipeline(device) as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        q = cam.requestOutput((int(width), int(height)), fps=float(fps), enableUndistortion=True).createOutputQueue()
        pipeline.start()
        last_ts_ms = -1
        while pipeline.isRunning():
            img = q.get()
            t_sec = img.getTimestampDevice().total_seconds()
            ts_ms = int(t_sec * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms
            yield img.getCvFrame(), ts_ms


def run_live_hand_test(
    *,
    seconds: float = 10.0,
    width: int = 4208,
    height: int = 3120,
    fps: float = 30.0,
    max_num_hands: int = 1,
    show_window: bool = True,
    model_path: str = "public/hand_landmarker.task",
) -> None:
    import cv2
    import mediapipe as mp

    # Use the MediaPipe Tasks Hand Landmarker (consistent with capture/pose pipeline).
    # Ref: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    from pathlib import Path

    p = Path(model_path)
    if not p.exists():
        p = (Path(__file__).resolve().parents[1] / model_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Hand Landmarker model not found at '{model_path}' (also tried '{p}')")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(p)),
        running_mode=RunningMode.VIDEO,
        num_hands=int(max_num_hands),
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    #drawer = mp.solutions.drawing_utils

    t0 = time.time()
    window = "OAK-D: Hand open/closed"
    if show_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            for frame_bgr, ts_ms in _iter_frames_depthai(width=width, height=height, fps=fps):
                if seconds is not None and (time.time() - t0) > float(seconds):
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                view = frame_bgr.copy()
                label = "no_hand"
                if result.hand_landmarks and len(result.hand_landmarks) > 0:
                    lm = result.hand_landmarks[0]
                    state = classify_hand_open_closed(lm)
                    label = f"{state.label}  ({state.extended_fingers}/5)"
                    #drawer.draw_landmarks(view, lm, mp.solutions.hands.HAND_CONNECTIONS)

                cv2.putText(view, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(view, "q: quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

                if show_window:
                    cv2.imshow(window, view)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
    finally:
        if show_window:
            cv2.destroyWindow(window)


# -----------------------
# Unit tests (no hardware)
# -----------------------

class _LM:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


def _mk_hand_landmarks(pts: dict[int, tuple[float, float, float]]):
    """
    Create a MediaPipe-like hand landmark object for testing.
    Expects indices 0..20; missing indices default to (0,0,0).
    """
    lms = [_LM(0.0, 0.0, 0.0) for _ in range(21)]
    for idx, (x, y, z) in pts.items():
        lms[int(idx)] = _LM(x, y, z)

    class _Obj:
        landmark: list

        def __init__(self, landmark):
            self.landmark = landmark

    return _Obj(lms)


def test_classify_hand_open_synthetic():
    # Construct mostly straight fingers (MCP-PIP-TIP roughly collinear).
    pts = {
        # Index (5-6-8)
        5: (0.2, 0.5, 0.0),
        6: (0.2, 0.4, 0.0),
        8: (0.2, 0.2, 0.0),
        # Middle (9-10-12)
        9: (0.3, 0.5, 0.0),
        10: (0.3, 0.4, 0.0),
        12: (0.3, 0.2, 0.0),
        # Ring (13-14-16)
        13: (0.4, 0.5, 0.0),
        14: (0.4, 0.4, 0.0),
        16: (0.4, 0.2, 0.0),
        # Pinky (17-18-20)
        17: (0.5, 0.5, 0.0),
        18: (0.5, 0.4, 0.0),
        20: (0.5, 0.22, 0.0),
        # Thumb (2-3-4)
        2: (0.15, 0.55, 0.0),
        3: (0.10, 0.48, 0.0),
        4: (0.05, 0.40, 0.0),
    }
    hand = _mk_hand_landmarks(pts)
    state = classify_hand_open_closed(hand)
    assert state.label == "open"
    assert state.extended_fingers >= 4


def test_classify_hand_closed_synthetic():
    # Bent fingers: PIP joint forms a ~90 degree angle.
    pts = {
        5: (0.2, 0.5, 0.0),
        6: (0.2, 0.4, 0.0),
        8: (0.28, 0.4, 0.0),
        9: (0.3, 0.5, 0.0),
        10: (0.3, 0.4, 0.0),
        12: (0.38, 0.4, 0.0),
        13: (0.4, 0.5, 0.0),
        14: (0.4, 0.4, 0.0),
        16: (0.48, 0.4, 0.0),
        17: (0.5, 0.5, 0.0),
        18: (0.5, 0.4, 0.0),
        20: (0.58, 0.4, 0.0),
        2: (0.15, 0.55, 0.0),
        3: (0.12, 0.50, 0.0),
        4: (0.17, 0.49, 0.0),
    }
    hand = _mk_hand_landmarks(pts)
    state = classify_hand_open_closed(hand)
    assert state.label == "closed"


def test_live_oak_hand_smoke_optional():
    """
    Optional hardware smoke test.

    Enable with:
      RUN_OAK_HAND_TEST=1 pytest -q tests/test_hand.py
    """
    if os.getenv("RUN_OAK_HAND_TEST", "0") != "1":
        # Pytest-friendly skip without importing pytest as a hard dependency.
        return

    # Deps check
    try:
        import depthai  # noqa: F401
        import mediapipe  # noqa: F401
        import cv2  # noqa: F401
    except Exception:
        return

    # Just ensure we can pull a few frames; hand detection is not required.
    frames = 0
    t0 = time.time()
    for _frame, _ts_ms in _iter_frames_depthai(width=4208, height=3120, fps=30.0):
        frames += 1
        if frames >= 10:
            break
        if (time.time() - t0) > 5.0:
            break
    assert frames > 0


def run_synthetic_self_check() -> None:
    test_classify_hand_open_synthetic()
    test_classify_hand_closed_synthetic()


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand open/closed test (MediaPipe Hand Landmarker Tasks API + OAK/DepthAI).")
    ap.add_argument("--seconds", type=float, default=30.0, help="Run duration for live preview.")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--max-hands", type=int, default=1)
    ap.add_argument("--no-show", action="store_true", help="Run without an OpenCV window.")
    ap.add_argument("--synthetic", action="store_true", help="Run synthetic self-check only (no camera/deps).")
    ap.add_argument("--model", type=str, default="public/hand_landmarker.task", help="Path to hand_landmarker.task")
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic_self_check()
        print("Synthetic self-check: OK")
        return

    # If dependencies/hardware aren't present, fall back to synthetic checks.
    try:
        import depthai  # noqa: F401
        import mediapipe  # noqa: F401
        import cv2  # noqa: F401
    except Exception as e:
        run_synthetic_self_check()
        print(f"Live deps unavailable ({e}); synthetic self-check: OK")
        return

    run_live_hand_test(
        seconds=float(args.seconds),
        width=int(args.width),
        height=int(args.height),
        fps=float(args.fps),
        max_num_hands=int(args.max_hands),
        show_window=not bool(args.no_show),
        model_path=str(args.model),
    )


if __name__ == "__main__":
    main()
