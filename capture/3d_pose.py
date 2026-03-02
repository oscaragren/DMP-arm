import depthai as dai
import argparse
import json
import time
import numpy as np
import cv2
from datetime import timedelta
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# MediaPipe Pose landmark indices: left arm + right shoulder (for trunk reference later)
POSE_KEYPOINT_IDS = [11, 13, 15, 12]  # left_shoulder, left_elbow, left_wrist, right_shoulder
POSE_KEYPOINT_NAMES = {
    11: "left_shoulder",
    13: "left_elbow",
    15: "left_wrist",
    12: "right_shoulder",
}

RGB_SOCKET = dai.CameraBoardSocket.CAM_A
LEFT_SOCKET = dai.CameraBoardSocket.CAM_B
RIGHT_SOCKET = dai.CameraBoardSocket.CAM_C

def deproject(u, v, z_m, fx, fy, cx, cy):
    """Pixel (u, v) with depth z (meters) -> camera-frame XYZ (meters)"""
    x = (u - cx) * z_m / fx
    y = (v - cy) * z_m / fy
    return float(x), float(y), float(z_m)

COUNTDOWN_SECONDS = 3  # 3, 2, 1 then Go
RECORD_DURATION = 4.0  # seconds to record after countdown

def _draw_centered_text(frame, text, font_scale=3, thickness=6):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = (w - tw) // 2, (h + th) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

def depth_at(depth_mm, u, v, patch=7):
    h, w  = depth_mm.shape[:2]
    u0, v0 = int(np.clip(u, 0, w-1)), int(np.clip(v, 0, h-1))
    r = patch // 2
    x1, x2 = max(0, u0-r), min(w, u0+r+1)
    y1, y2 = max(0, v0-r), min(h, v0+r+1)
    
    roi = depth_mm[y1:y2, x1:x2].astype(np.float32)
    roi = roi[roi > 0] # Drop invalid

    if roi.size == 0:
        return None
    return float(np.median(roi)) / 1000.0 # mm -> meters

def main():
    parser = argparse.ArgumentParser(description="Live 3D pose from OAK-D (RGB + depth): left arm + right shoulder.")
    parser.add_argument("--subject", type=int, required=True, help="Subject number (1, 2, 3, ...)")
    parser.add_argument("--motion", type=str, required=True, help="Motion name (e.g. reach, curved_reach)")
    parser.add_argument("--trial", type=int, required=True, help="Trial number")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("test_data/processed"),
        help="Root of processed output (subject/motion/trial will be appended)",
    )
    parser.add_argument("--model", type=str, default="capture/pose_landmarker_lite.task")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--patch", type=int, default=7, help="Median patch size for depth sampling")
    parser.add_argument("--json", action="store_true", help="Also save JSON sequence")
    parser.add_argument("--min-z", type=float, default=0.0, help="Minimum depth for valid pose")
    parser.add_argument("--max-z", type=float, default=10.0, help="Maximum depth for valid pose")
    parser.add_argument("--no-show", action="store_true", help="Disable cv2.imshow (run headless)")
    parser.add_argument("--show-depth", action="store_true", help="Show a second window with depth colormap")
    args = parser.parse_args()
    show_window = not args.no_show

    # Output: test_data/processed/subject_01/reach/trial_001/
    outdir = args.processed_dir / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {outdir}")

    # Build pipeline
    rgb_size = (640, 400)

    # Mediapose Pose Landmarker
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode


    base_options = BaseOptions(model_asset_path=args.model)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5, # There can be used as input argument later
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_frames = []  # List of (4, 3) np.ndarray: x,y,z of left_shoulder, left_elbow, left_wrist, right_shoulder
    all_t = [] # device timestamps (seconds)
    all_json = [] # optional

    device = dai.Device()

    with dai.Pipeline(device) as pipeline, PoseLandmarker.create_from_options(options) as landmarker:

        platform = device.getPlatform()
        cam_rgb = pipeline.create(dai.node.Camera).build(RGB_SOCKET)
        left = pipeline.create(dai.node.Camera).build(LEFT_SOCKET)
        right = pipeline.create(dai.node.Camera).build(RIGHT_SOCKET)

        stereo = pipeline.create(dai.node.StereoDepth)
        sync = pipeline.create(dai.node.Sync)

        stereo.setExtendedDisparity(True)
        sync.setSyncThreshold(timedelta(seconds=1/(2*args.fps)))

        # Outputs
        video_stream = cam_rgb.requestOutput(size=rgb_size, fps=args.fps, enableUndistortion=True)
        left.requestOutput(size=rgb_size, fps=args.fps).link(stereo.left)
        right.requestOutput(size=rgb_size, fps=args.fps).link(stereo.right)
        
        video_stream.link(sync.inputs["rgb"])
        video_stream.link(stereo.inputAlignTo)
        stereo.depth.link(sync.inputs["depth_aligned"])

        #mg_align = pipeline.create(dai.node.ImageAlign)
        #stereo.depth.link(img_align.input) #.input is an ImgFrame
        #video_stream.link(img_align.inputAlignTo) # .inputAlignTo is an ImgFrame
        #img_align.outputAligned.link(sync.inputs["depth_aligned"])
        #video_stream.link(sync.inputs["rgb"])

        # Create output queue to get synced frames to the host computer
        queue = sync.out.createOutputQueue()

        #calib = device.readCalibration() # Not sure about this... dai.CalibrationHandler().getCameraIntrinsics()
        calib = device.readCalibration()
        K = calib.getCameraIntrinsics(RGB_SOCKET, rgb_size[0], rgb_size[1])
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2] # K[0][0] maybe

        #q_rgb = video_stream.createOutputQueue()
        #q_depth = stereo.depth.createOutputQueue()

        pipeline.start()

        print(f"Running on platform: {platform.name}")
        if show_window:
            cv2.namedWindow("OAK-D: Left Arm 3D + Pose", cv2.WINDOW_NORMAL)
            if args.show_depth:
                cv2.namedWindow("OAK-D: Depth", cv2.WINDOW_NORMAL)
            # Show "waiting" frame immediately so the window appears (queue.get() can block for a long time)
            wait_img = np.zeros((rgb_size[1], rgb_size[0], 3), dtype=np.uint8)
            wait_img[:] = (40, 40, 40)
            cv2.putText(
                wait_img, "Waiting for first frame...", (50, rgb_size[1] // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2,
            )
            cv2.putText(
                wait_img, "Press 'q' to quit once stream starts.", (50, rgb_size[1] // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
            )
            cv2.imshow("OAK-D: Left Arm 3D + Pose", wait_img)
            cv2.waitKey(1)
        print(f"Countdown {COUNTDOWN_SECONDS}s, then recording {RECORD_DURATION}s. Press 'q' to quit early.")
        last_ts_ms = -1
        phase = "countdown"
        countdown_start = None
        record_start = None
        video_writer = None
        video_path = outdir / "video.mp4"

        while pipeline.isRunning():
            # Use tryGet() so we can keep the window responsive; fall back to get() if tryGet not available
            #print("Waiting for RGB...")
            #rgb = q_rgb.get() # Blocking
            #print("Got RGB frame:", rgb.getTimestampDevice().total_seconds())

            #print("Waiting for depth...")
            #depth = q_depth.get() # Blocking
            #print("Got depth frame:", depth.getTimestampDevice().total_seconds())
            
            msg_group = queue.get()
            frame_rgb = msg_group["rgb"]
            frame_depth = msg_group["depth_aligned"]
            #print(f"Timestamps, message group: rgb: {frame_rgb.getTimestampDevice().total_seconds()}, depth: {frame_depth.getTimestampDevice().total_seconds()}")
            #print(f"Keys: {msg_group.getMessageNames()}")
            #if "rgb" not in msg_group or "depth_aligned" not in msg_group:
            #    print("Here")
            #    continue
            
            #rgb = msg["rgb"]
            #dep = msg["depth_aligned"]

            frame_bgr = frame_rgb.getCvFrame()
            depth_mm = frame_depth.getFrame()
            if countdown_start is None:
                countdown_start = time.time()

            # Use DAI device timstamp as the VIDEO timestamp
            t_sec = frame_rgb.getTimestampDevice().total_seconds()
            ts_ms = int(t_sec * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            now = time.time()

            # Countdown phase: show 3, 2, 1, Go! then switch to recording
            if phase == "countdown":
                elapsed = now - countdown_start
                display_frame = frame_bgr.copy()
                if elapsed < 1:
                    _draw_centered_text(display_frame, "3")
                elif elapsed < 2:
                    _draw_centered_text(display_frame, "2")
                elif elapsed < 3:
                    _draw_centered_text(display_frame, "1")
                elif elapsed < 3.5:
                    _draw_centered_text(display_frame, "Go!")
                else:
                    phase = "recording"
                    record_start = time.time()
                    h, w = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
                    print("Recording...")
                if show_window:
                    cv2.imshow("OAK-D: Left Arm 3D + Pose", display_frame)
                if show_window and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Recording phase: run pose, write to MP4 and buffers for 4 seconds
            if phase == "recording":
                record_elapsed = now - record_start
                if record_elapsed >= RECORD_DURATION:
                    break

            # MediaPipe Image must be SRGB; OpenCV gives BGR -> convert to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # PoseLandmarker VIDEO inference (synchronous). :contentReference[oaicite:4]{index=4}
            result = landmarker.detect_for_video(mp_image, ts_ms)

            pose_xyz = np.full((len(POSE_KEYPOINT_IDS), 3), np.nan, dtype=np.float32)

            frame_json = None
            if args.json:
                frame_json = {"t": t_sec, "joints": {}}
            
            # result.pose_landmarks: List[List[NormalizedLandmark]] (one list per detected pose)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                pose0 = result.pose_landmarks[0] # single-person

                h, w = frame_bgr.shape[:2]

                for j, idx in enumerate(POSE_KEYPOINT_IDS):
                    lm = pose0[idx]
                    u, v = lm.x * w, lm.y * h
                    
                    z_m = depth_at(depth_mm, u, v, patch=args.patch)
            
                    if z_m is None or z_m < args.min_z or z_m > args.max_z:
                        continue

                    x, y, z = deproject(u, v, z_m, fx, fy, cx, cy)
                    pose_xyz[j] = (x, y, z)

                    # Overlay
                    cv2.circle(frame_bgr, (int(u), int(v)), 3, (0, 255, 0), -1)
                    cv2.putText(
                        frame_bgr,
                        f"{POSE_KEYPOINT_NAMES[idx]} z={z:.2f}m",
                        (int(u)+5, int(v)-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                    )

                    if frame_json is not None:
                        frame_json["joints"][POSE_KEYPOINT_NAMES[idx]] = {
                            "x": x, "y": y, "z": z,          # camera-frame meters (from OAK depth)
                            "u": float(u), "v": float(v),     # pixel coords
                            "visibility": float(getattr(lm, "visibility", 1.0)),
                            "presence": float(getattr(lm, "presence", 1.0)),
                        }

            if phase == "recording":
                video_writer.write(frame_bgr)
                all_frames.append(pose_xyz)
                all_t.append(t_sec)
                if frame_json is not None:
                    all_json.append(frame_json)

            if show_window:
                cv2.imshow("OAK-D: Left Arm 3D + Pose", frame_bgr)
                if args.show_depth:
                    # Normalize depth for visibility (mm -> colormap)
                    depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                    cv2.imshow("OAK-D: Depth", depth_vis)
            if show_window and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if show_window:
            cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved video: {video_path}")

    seq = np.stack(all_frames, axis=0) if all_frames else np.zeros((0, len(POSE_KEYPOINT_IDS), 3), dtype=np.float32)
    t = np.array(all_t, dtype=np.float64)

    npy_seq_path = outdir / "left_arm_seq_camera.npy"
    npy_t_path = outdir / "left_arm_t.npy"
    np.save(npy_seq_path, seq)
    np.save(npy_t_path, t)

    meta = {
        "subject": args.subject,
        "motion": args.motion,
        "trial": args.trial,
        "shape": list(seq.shape),
        "keypoint_names": list(POSE_KEYPOINT_NAMES.values()),
        "source": "3d_pose.py (OAK-D live)",
        "record_duration_sec": RECORD_DURATION,
    }
    if (outdir / "video.mp4").exists():
        meta["video"] = "video.mp4"
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved:\n  {npy_seq_path}  shape={seq.shape}\n  {npy_t_path}  shape={t.shape}\n  {outdir / 'meta.json'}")

    if args.json:
        json_path = outdir / "left_arm_sequence.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_json, f, indent=2)
        print(f"  {json_path}  frames={len(all_json)}")



if __name__ == "__main__":
    main()