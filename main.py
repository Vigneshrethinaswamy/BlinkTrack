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
    print("BlinkTrack is running. Press Q or Esc to quit; press R to reset the counts.")
    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("WARNING: Could not read a frame from the webcam.")
                break
            frame = cv2.flip(frame, 1)
            frame, result = detector.process(frame)

            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed
                fps = instantaneous_fps if fps == 0 else (0.9 * fps + 0.1 * instantaneous_fps)

            face_color = (40, 220, 40) if result.face_detected else (0, 0, 255)
            face_text = "FACE: TRACKED" if result.face_detected else "FACE: NOT DETECTED"
            cv2.putText(frame, face_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, face_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"BLINKS: {result.blink_count}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 230, 40), 2, cv2.LINE_AA)
            cv2.putText(frame, f"WINKS  L: {result.left_wink_count}  R: {result.right_wink_count}",
                        (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 120), 2, cv2.LINE_AA)
            left_ear = f"{result.left_ear:.3f}" if result.left_ear is not None else "--"
            right_ear = f"{result.right_ear:.3f}" if result.right_ear is not None else "--"
            cv2.putText(frame, f"LEFT  EAR: {left_ear}  [{result.left_state}]",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 190, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"RIGHT EAR: {right_ear}  [{result.right_state}]",
                        (20, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 188),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Q/Esc: Quit  |  R: Reset counts",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (230, 230, 230), 1, cv2.LINE_AA)

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
