import argparse
import json
import time
import depthai as dai
import cv2
from pathlib import Path

parser = argparse.ArgumentParser(description="Capture camera frames and save to data/raw.")
parser.add_argument("--subject", type=int, required=True, help="Subject number (1, 2, 3, ...)")
parser.add_argument("--motion", type=str, required=True, help="Motion name (e.g. reach, curved_reach)")
parser.add_argument("--trial", type=int, required=True, help="Trial number")
args = parser.parse_args()

OUTPUT_DIR = Path("data/raw") / f"subject_{args.subject:02d}" / args.motion / f"trial_{args.trial:03d}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_SIZE = (640, 400)
FPS_NOMINAL = 30.0
VIDEO_PATH = OUTPUT_DIR / "video.mp4"
COUNTDOWN_SECONDS = 3  # 3, 2, 1 then Go
RECORD_DURATION = 3.0  # seconds

print(f"Saving to {OUTPUT_DIR}")
print(f"Countdown {COUNTDOWN_SECONDS}s, then recording {RECORD_DURATION}s. Press 'q' to quit early.")

def draw_centered_text(frame, text, font_scale=4, thickness=8):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    video_queue = cam.requestOutput(FRAME_SIZE).createOutputQueue()

    pipeline.start()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_PATH), fourcc, FPS_NOMINAL, FRAME_SIZE)
    frame_index = 0
    timestamps = []
    recording = False
    countdown_start = time.time()
    record_start = None

    try:
        while pipeline.isRunning():
            video_in = video_queue.get()
            assert isinstance(video_in, dai.ImgFrame)
            frame = video_in.getCvFrame().copy()
            now = time.time()

            if not recording:
                elapsed = now - countdown_start
                if elapsed < 1:
                    draw_centered_text(frame, "3")
                elif elapsed < 2:
                    draw_centered_text(frame, "2")
                elif elapsed < 3:
                    draw_centered_text(frame, "1")
                elif elapsed < 3.5:
                    draw_centered_text(frame, "Go!")
                else:
                    recording = True
                    record_start = time.time()
                cv2.imshow("video", frame)
            else:
                record_elapsed = now - record_start
                if record_elapsed >= RECORD_DURATION:
                    break
                writer.write(frame)
                timestamps.append(video_in.getTimestamp().total_seconds())
                frame_index += 1
                cv2.imshow("video", frame)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        writer.release()

    pipeline.stop()
    cv2.destroyAllWindows()

# Save timestamps and meta for this run
if timestamps:
    import numpy as np
    np.save(OUTPUT_DIR / "timestamps.npy", np.array(timestamps))
    meta = {
        "subject": args.subject,
        "motion": args.motion,
        "trial": args.trial,
        "n_frames": frame_index,
        "video": "video.mp4",
        "fps_nominal": FPS_NOMINAL,
        "record_duration_sec": RECORD_DURATION,
    }
    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {frame_index} frames to {OUTPUT_DIR}")
