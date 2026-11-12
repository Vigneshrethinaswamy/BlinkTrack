"""MediaPipe Face Mesh landmark extraction and blink event detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from utils.blink_counter import BlinkCounter
from utils.eye_aspect_ratio import eye_aspect_ratio

EAR_THRESHOLD = 0.21
MIN_CLOSED_FRAMES = 2
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


@dataclass
class BlinkResult:
    ear: float | None
    eye_state: str
    blink_count: int
    face_detected: bool


class BlinkDetector:
    def __init__(self, ear_threshold: float = EAR_THRESHOLD,
                 min_closed_frames: int = MIN_CLOSED_FRAMES) -> None:
        self.counter = BlinkCounter(ear_threshold, min_closed_frames)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    @property
    def blink_count(self) -> int:
        return self.counter.count

    @staticmethod
    def _eye_points(landmarks, indices, width: int, height: int) -> np.ndarray:
        return np.asarray([[landmarks[i].x * width, landmarks[i].y * height]
                           for i in indices], dtype=np.float64)

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, BlinkResult]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        output = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not output.multi_face_landmarks:
            self.counter.cancel_pending()
            return frame, BlinkResult(None, "Face not detected", self.blink_count, False)

        landmarks = output.multi_face_landmarks[0].landmark
        left = self._eye_points(landmarks, LEFT_EYE, width, height)
        right = self._eye_points(landmarks, RIGHT_EYE, width, height)
        ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

        for indices in (LEFT_EYE, RIGHT_EYE):
            contour = np.asarray([[int(landmarks[i].x * width), int(landmarks[i].y * height)]
                                  for i in indices], dtype=np.int32)
            cv2.polylines(frame, [contour], True, (0, 220, 255), 1, cv2.LINE_AA)

        state = self.counter.update(ear)
        return frame, BlinkResult(ear, state, self.blink_count, True)

    def reset_count(self) -> None:
        self.counter.reset()

    def close(self) -> None:
        self.face_mesh.close()
