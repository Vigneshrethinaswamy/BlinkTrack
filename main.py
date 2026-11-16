"""Run BlinkTrack with a webcam or a prerecorded video."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2

from blink_detector import BlinkDetector
from utils.dashboard import draw_dashboard
from utils.ear_calibrator import EARCalibrator
from utils.frame_enhancer import VideoEnhancer
from video_source import VideoSource, parse_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time blink detection with optional low-light enhancement."
    )
    parser.add_argument(
        "--source", default="0",
        help="Camera index (0 by default) or path to a video file.",
    )
    parser.add_argument("--width", type=int, default=640, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height.")
    parser.add_argument(
        "--process-width", type=int, default=480,
        help="Maximum frame width sent to MediaPipe (480 by default for faster CPU processing).",
    )
    parser.add_argument(
        "--enhancement", choices=VideoEnhancer.MODES, default="auto",
        help="auto enhances poor frames, low-light always enhances, off disables it.",
    )
    parser.add_argument(
        "--record", default=None,
        help="Optional output video path. Nothing is saved unless this is supplied.",
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="Do not mirror webcam frames.",
    )
    return parser.parse_args()


def resize_for_processing(frame, max_width: int):
    height, width = frame.shape[:2]
    if max_width > 0 and width > max_width:
        scale = max_width / width
        return cv2.resize(
            frame, (max_width, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def main() -> int:
    args = parse_args()
    source_value = parse_source(args.source)
    try:
        source = VideoSource(source_value, args.width, args.height)
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return 1

    detector = BlinkDetector()
    enhancer = VideoEnhancer(args.enhancement)
    calibrator: EARCalibrator | None = None
    writer = None
    fps = 0.0
    previous_time = time.perf_counter()
    is_camera = source.is_camera
    mirror = is_camera and not args.no_mirror
    display_name = "BlinkTrack | Camera"
    dashboard_name = "BlinkTrack | Dashboard"
    windows_positioned = False

    print(f"Source: {'webcam' if is_camera else source_value}")
    print(f"Input size: {source.frame_size[0]}x{source.frame_size[1]}")
    print(f"Enhancement mode: {enhancer.mode}")
    print("Keys: Q/Esc quit | R reset | C calibrate | E cycle enhancement | S screenshot")

    try:
        while True:
            success, raw_frame = source.read()
            if not success or raw_frame is None:
                if not is_camera:
                    print("Video playback completed.")
                else:
                    print("WARNING: Webcam frame could not be read.")
                break

            raw_frame = resize_for_processing(raw_frame, args.process_width)
            if mirror:
                raw_frame = cv2.flip(raw_frame, 1)

            enhanced_frame, quality = enhancer.process(raw_frame)
            frame, result = detector.process(enhanced_frame)

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else (0.88 * fps + 0.12 * instant_fps)

            if calibrator is not None:
                calibrator.add(result.left_ear, result.right_ear)
                if calibrator.complete:
                    calibration = calibrator.result()
                    detector.set_thresholds(
                        calibration.close_threshold, calibration.open_threshold
                    )
                    print(
                        "Calibration complete: "
                        f"baseline={calibration.baseline_ear:.3f}, "
                        f"close={calibration.close_threshold:.3f}, "
                        f"open={calibration.open_threshold:.3f}"
                    )
                    calibrator = None

            dashboard_frame = draw_dashboard(
                frame, result, fps, quality, enhancer.mode,
                detector.close_threshold, detector.open_threshold,
                calibrator.progress if calibrator is not None else None,
            )

            if args.record:
                if writer is None:
                    output_path = Path(args.record).expanduser()
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_fps = source.fps or (fps if fps > 0 else 20.0)
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create recording: {output_path}")
                    print(f"Recording annotated output to: {output_path}")
                writer.write(frame)

            cv2.imshow(display_name, frame)
            cv2.imshow(dashboard_name, dashboard_frame)
            if not windows_positioned:
                cv2.moveWindow(display_name, 60, 60)
                cv2.moveWindow(dashboard_name, 60 + frame.shape[1] + 24, 60)
                windows_positioned = True
            delay = 1
            if not is_camera and source.fps > 0:
                delay = max(1, int(1000 / source.fps))
            key = cv2.waitKey(delay) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                detector.reset_count()
                print("Session counters reset.")
            elif key == ord("c"):
                calibrator = EARCalibrator(required_samples=45)
                print("Calibration started. Look straight at the camera with both eyes OPEN.")
            elif key == ord("e"):
                modes = list(VideoEnhancer.MODES)
                next_mode = modes[(modes.index(enhancer.mode) + 1) % len(modes)]
                enhancer.set_mode(next_mode)
                print(f"Enhancement mode: {next_mode}")
            elif key == ord("s"):
                capture_dir = Path("captures")
                capture_dir.mkdir(exist_ok=True)
                path = capture_dir / f"blinktrack_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                if cv2.imwrite(str(path), frame):
                    print(f"Dashboard screenshot saved: {path}")

    except KeyboardInterrupt:
        print("\nStopping BlinkTrack...")
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    finally:
        source.release()
        detector.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
