"""
This is a script for collecting data which is used to evaluate the performance of the DMP model.

1. Parse session setups
2. Initialize the recording device
3. Create the session/trial directories
4. Show a live preview
5. Wait for recording trigger
6. Record the data
7. Mark the trial meta data
8. End and validate the trial

Repeat for multiple trials
"""

import argparse
import os
import numpy as np
import cv2
from pathlib import Path

from capture.live_preview import live_preview
from capture.record_data import record_data


def _validate_arguments(*, args: argparse.Namespace) -> None:
    if args.subject is None:
        raise ValueError("Subject ID is required")
    if args.motion is None:
        raise ValueError("Motion type is required")
    if args.n_trials is None:
        raise ValueError("Number of trials is required")


def _get_stats(*, trial_dir: str) -> None:
    """Get the stats of the trial."""
    # Number of missing values in the left_arm_seq_camera.npy file
    left_arm_seq_camera = np.load(os.path.join(trial_dir, "left_arm_seq_camera.npy"))
    num_missing_values = np.sum(np.isnan(left_arm_seq_camera))
    print(f"Number of missing values: {num_missing_values}")
    print(f"Percentage of missing values: {num_missing_values / left_arm_seq_camera.size * 100}%")

def _replay_trial(*, video_path: Path, window_name: str, replay_speed: float = 0.5) -> bool:
    """
    Replay a recorded trial video until the operator continues.

    Controls:
    - SPACE: continue
    - q: abort (returns False)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return True

    replay_name = f"{window_name} (replay)"
    cv2.namedWindow(replay_name, cv2.WINDOW_NORMAL)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        base_delay_ms = int(round(1000.0 / fps)) if fps and fps > 1e-6 else 33
        speed = float(replay_speed) if float(replay_speed) > 1e-6 else 0.5
        delay_ms = int(round(base_delay_ms / speed))

        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            #_draw_banner(frame, f"Replay {speed:.1f}x (SPACE: continue, q: abort)")
            cv2.imshow(replay_name, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                return False
            if key == ord(" "):
                return True
    finally:
        cap.release()
        cv2.destroyWindow(replay_name)

def main():
    
    # 1) Parse session setups
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True, help="Subject ID")
    ap.add_argument("--motion", type=str, required=True, help="Motion type")
    ap.add_argument("--n_trials", type=int, required=True, help="Number of trials", default=10)
    ap.add_argument("--n_frames", type=int, help="Number of frames", default=240)
    ap.add_argument("--fps_nominal", type=float, help="Nominal FPS", default=25.0)
    ap.add_argument("--record_duration_sec", type=float, help="Record duration in seconds")

    args = ap.parse_args()

    # 1.2) Validate arguments
    _validate_arguments(args=args)

    # 2) Create session directory
    session_dir = os.path.join(os.path.dirname(__file__), "data", "raw", f"subject_{args.subject:02d}", args.motion)
    os.makedirs(session_dir, exist_ok=True)

    # 3) Show a live preview (Could move inside the while loop)
    live_preview(args=args)
    
    trial_number = 0
    while trial_number < args.n_trials:

        # 4) Make a new trial directory
        trial_dir = os.path.join(session_dir, f"trial_{(trial_number+1):03d}")
        os.makedirs(trial_dir, exist_ok=True)

        # 5) Record the data
        ok = record_data(args=args, trial_dir=trial_dir, trial=trial_number+1)
        if not ok:
            print("Recording aborted.")
            break

        # 6) End and validate the trial
        _get_stats(trial_dir=trial_dir)

        # 7) Replay the trial
        input("Press enter to replay the trial...")
        _replay_trial(video_path=os.path.join(trial_dir, "video.mp4"), window_name=f"Trial {trial_number+1}", replay_speed=0.5)

        trial_number += 1
    

if __name__ == "__main__":
    main()
