"""Run BlinkTrack using the default webcam."""

from __future__ import annotations

import time

import cv2

from blink_detector import BlinkDetector


def main() -> int:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions and that it is not in use.")
        camera.release()
        return 1

    detector = BlinkDetector()
    previous_time = time.perf_counter()
    fps = 0.0

    print("BlinkTrack is running. Press Q or Esc to quit; press R to reset the count.")
    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("WARNING: Could not read a frame from the webcam.")
                break

            # Mirror the preview so movement feels natural.
            frame = cv2.flip(frame, 1)
            frame, result = detector.process(frame)

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed
                fps = instantaneous_fps if fps == 0 else (0.9 * fps + 0.1 * instantaneous_fps)

            cv2.putText(frame, f"Blinks: {result.blink_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 230, 40), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Eye state: {result.eye_state}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            ear_text = f"EAR: {result.ear:.3f}" if result.ear is not None else "EAR: --"
            cv2.putText(frame, ear_text, (20, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 138),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Q/Esc: Quit  |  R: Reset", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

            cv2.imshow("BlinkTrack - Real-Time Eye Blink Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                detector.reset_count()
    except KeyboardInterrupt:
        print("\nStopping BlinkTrack...")
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
