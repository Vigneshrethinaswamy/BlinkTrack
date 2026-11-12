"""MediaPipe Face Mesh landmark extraction and per-eye blink detection."""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import mediapipe as mp
import numpy as np

from utils.blink_counter import BlinkCounter
from utils.eye_aspect_ratio import eye_aspect_ratio

EAR_CLOSE_THRESHOLD = 0.21
EAR_OPEN_THRESHOLD = 0.24
BLINK_SYNC_WINDOW_FRAMES = 5
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


@dataclass
class BlinkResult:
    left_ear: float | None
    right_ear: float | None
    left_state: str
    right_state: str
    blink_count: int
    left_wink_count: int
    right_wink_count: int
    face_detected: bool


class BlinkDetector:
    def __init__(self, close_threshold: float = EAR_CLOSE_THRESHOLD,
                 open_threshold: float = EAR_OPEN_THRESHOLD,
                 sync_window_frames: int = BLINK_SYNC_WINDOW_FRAMES) -> None:
        self.counter = BlinkCounter(close_threshold, open_threshold, sync_window_frames)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    @staticmethod
    def _eye_points(landmarks, indices, width: int, height: int) -> np.ndarray:
        return np.asarray([[landmarks[i].x * width, landmarks[i].y * height]
                           for i in indices], dtype=np.float64)

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, BlinkResult]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        output = self.face_mesh.process(rgb)
        if not output.multi_face_landmarks:
            self.counter.cancel_pending()
            return frame, BlinkResult(
                None, None, "Unknown", "Unknown", self.counter.blink_count,
                self.counter.left_wink_count, self.counter.right_wink_count, False,
            )

        landmarks = output.multi_face_landmarks[0].landmark
        left_points = self._eye_points(landmarks, LEFT_EYE, width, height)
        right_points = self._eye_points(landmarks, RIGHT_EYE, width, height)
        left_ear = eye_aspect_ratio(left_points)
        right_ear = eye_aspect_ratio(right_points)
        left_state, right_state = self.counter.update(left_ear, right_ear)

        # Different colors make the two independent eye tracks easier to read.
        for indices, color in ((LEFT_EYE, (255, 190, 0)), (RIGHT_EYE, (0, 220, 255))):
            contour = np.asarray([[int(landmarks[i].x * width), int(landmarks[i].y * height)]
                                  for i in indices], dtype=np.int32)
            cv2.polylines(frame, [contour], True, color, 2, cv2.LINE_AA)

        return frame, BlinkResult(
            left_ear, right_ear, left_state, right_state, self.counter.blink_count,
            self.counter.left_wink_count, self.counter.right_wink_count, True,
        )

    def reset_count(self) -> None:
        self.counter.reset()

    def close(self) -> None:
        self.face_mesh.close()
