"""
Run pose estimation on recorded video streams using MediaPipe Pose Landmarker (Tasks API).
See: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

Extracts 3D arm keypoints (shoulder, elbow, wrist) and confidence scores.
Output: data/processed/<subject>/<motion>/<trial>/keypoints_3d.npy (T, K, 3), confidence.npy (T, K).
"""
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult
from mediapipe.tasks.python.vision.core import image as image_lib

# Pose landmark indices (PoseLandmark enum): shoulder, elbow, wrist for both arms
# LEFT_SHOULDER=11, LEFT_ELBOW=13, LEFT_WRIST=15, RIGHT_SHOULDER=12, RIGHT_ELBOW=14, RIGHT_WRIST=16
ARM_LANDMARK_INDICES = [11, 13, 15] #, 12, 14, 16]
N_KEYPOINTS = len(ARM_LANDMARK_INDICES)

KEYPOINT_NAMES = [
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
]

# Arm connections for drawing: (start_idx, end_idx) in full 33-landmark list
ARM_CONNECTIONS = [(11, 13), (13, 15)]#, (12, 14), (14, 16), (11, 12)]  # left arm, right arm, shoulders


def _draw_pose_overlay(frame: np.ndarray, pose_landmarks, radius: int = 5, thickness: int = 2) -> None:
    """Draw arm keypoints and connections on frame. pose_landmarks: list of 33 NormalizedLandmark (x,y in [0,1])."""
    h, w = frame.shape[:2]
    if pose_landmarks is None or len(pose_landmarks) < 17:
        return
    # Convert normalized to pixel coordinates for our 6 arm indices + connections (need 11,12,13,14,15,16)
    def to_px(idx):
        lm = pose_landmarks[idx]
        x = lm.x if lm.x is not None else 0.0
        y = lm.y if lm.y is not None else 0.0
        return (int(x * w), int(y * h))
    # Draw connections first (so points are on top)
    for i, j in ARM_CONNECTIONS:
        if i < len(pose_landmarks) and j < len(pose_landmarks):
            pt1 = to_px(i)
            pt2 = to_px(j)
            cv2.line(frame, pt1, pt2, (0, 255, 0), thickness)
    # Draw keypoints (arm only)
    for idx in ARM_LANDMARK_INDICES:
        if idx < len(pose_landmarks):
            pt = to_px(idx)
            cv2.circle(frame, pt, radius, (0, 255, 0), -1)
            cv2.circle(frame, pt, radius, (255, 255, 255), 1)


def run_pose_on_video(
    video_path: Path,
    model_path: Path,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    overlay_video_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run MediaPipe Pose Landmarker (Tasks API) on a video file.
    Returns 3D keypoints (world coordinates, meters) and confidence per frame.
    If overlay_video_path is set, writes a video with keypoints drawn on each frame.

    Returns:
        keypoints_3d: (T, K, 3)
        confidence: (T, K)
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Pose Landmarker model not found: {model_path}\n"
            "Download from: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models"
        )

    base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval_ms = int(1000.0 / fps)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = None
    if overlay_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_video_path), fourcc, fps, frame_size)

    keypoints_list = []
    confidence_list = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not rgb.flags["C_CONTIGUOUS"]:
                rgb = np.ascontiguousarray(rgb)
            mp_image = image_lib.Image(image_format=image_lib.ImageFormat.SRGB, data=rgb)
            timestamp_ms = frame_index * frame_interval_ms

            result: PoseLandmarkerResult = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.pose_world_landmarks or not result.pose_landmarks:
                keypoints_list.append(np.full((N_KEYPOINTS, 3), np.nan, dtype=np.float32))
                confidence_list.append(np.zeros(N_KEYPOINTS, dtype=np.float32))
            else:
                wlm = result.pose_world_landmarks[0]
                nlm = result.pose_landmarks[0]
                pts = np.array(
                    [
                        [wlm[i].x, wlm[i].y, wlm[i].z]
                        for i in ARM_LANDMARK_INDICES
                    ],
                    dtype=np.float32,
                )
                conf = np.array(
                    [
                        (nlm[i].visibility if nlm[i].visibility is not None else 0.0)
                        for i in ARM_LANDMARK_INDICES
                    ],
                    dtype=np.float32,
                )
                keypoints_list.append(pts)
                confidence_list.append(conf)

            if writer is not None:
                overlay_frame = frame.copy()
                if result.pose_landmarks:
                    _draw_pose_overlay(overlay_frame, result.pose_landmarks[0])
                writer.write(overlay_frame)

            frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    keypoints_3d = np.stack(keypoints_list, axis=0)
    confidence = np.stack(confidence_list, axis=0)
    return keypoints_3d, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Run pose estimation on recorded video (MediaPipe Pose Landmarker)."
    )
    parser.add_argument("--subject", type=int, required=True, help="Subject number (1, 2, 3, ...)")
    parser.add_argument("--motion", type=str, required=True, help="Motion name (e.g. reach, curved_reach)")
    parser.add_argument("--trial", type=int, required=True, help="Trial number")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root of raw data (subject/motion/trial/video.mp4)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Root of processed output",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("capture/pose_landmarker_lite.task"),
        help="Path to pose_landmarker.task (download from MediaPipe model gallery)",
    )
    parser.add_argument(
        "--min-pose-detection-confidence",
        type=float,
        default=0.5,
        help="Min confidence for pose detection",
    )
    parser.add_argument(
        "--min-pose-presence-confidence",
        type=float,
        default=0.5,
        help="Min confidence for pose presence",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Min confidence for pose tracking",
    )
    parser.add_argument(
        "--no-overlay-video",
        action="store_true",
        help="Skip writing video with keypoints overlay",
    )
    args = parser.parse_args()

    video_path = (
        args.raw_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}" / "video.mp4"
    )
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = (
        args.processed_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = None if args.no_overlay_video else (out_dir / "video_overlay.mp4")
    if overlay_path is not None:
        print(f"Overlay video will be saved to {overlay_path}")
    print(f"Running pose on {video_path} -> {out_dir}")
    keypoints_3d, confidence = run_pose_on_video(
        video_path,
        model_path=args.model,
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        min_pose_presence_confidence=args.min_pose_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        overlay_video_path=overlay_path,
    )
    T, K, _ = keypoints_3d.shape
    print(f"Frames: {T}, Keypoints: {K} ({', '.join(KEYPOINT_NAMES)})")

    np.save(out_dir / "keypoints_3d.npy", keypoints_3d)
    np.save(out_dir / "confidence.npy", confidence)

    meta = {
        "source_video": str(video_path),
        "shape": [int(T), int(K), 3],
        "keypoint_names": KEYPOINT_NAMES,
        "subject": args.subject,
        "motion": args.motion,
        "trial": args.trial,
        "model": str(args.model),
    }
    if overlay_path is not None:
        meta["video_overlay"] = "video_overlay.mp4"
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved keypoints_3d.npy shape ({T}, {K}, 3), confidence.npy shape ({T}, {K})")
    if overlay_path is not None:
        print(f"Saved overlay video: {overlay_path}")


if __name__ == "__main__":
    main()
