"""MediaPipe Face Mesh landmark extraction and blink event detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from utils.eye_aspect_ratio import eye_aspect_ratio

# Starting values; adjust for camera position and individual eye geometry.
EAR_THRESHOLD = 0.21
MIN_CLOSED_FRAMES = 2

# Six-point eye contours, indexed from MediaPipe Face Mesh landmarks.
# The order matches eye_aspect_ratio(): left corner, upper-left,
# upper-right, right corner, lower-right, lower-left.
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


@dataclass
class BlinkResult:
    """Measurements and state for the most recently processed frame."""

    ear: float | None
    eye_state: str
    blink_count: int
    face_detected: bool


class BlinkDetector:
    """Detect completed blinks from consecutive Face Mesh frames."""

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        min_closed_frames: int = MIN_CLOSED_FRAMES,
    ) -> None:
        if ear_threshold <= 0:
            raise ValueError("ear_threshold must be positive")
        if min_closed_frames < 1:
            raise ValueError("min_closed_frames must be at least 1")

        self.ear_threshold = ear_threshold
        self.min_closed_frames = min_closed_frames
        self.blink_count = 0
        self.closed_frames = 0
        self.was_closed = False
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _eye_points(landmarks, indices, width: int, height: int) -> np.ndarray:
        return np.asarray(
            [[landmarks[i].x * width, landmarks[i].y * height] for i in indices],
            dtype=np.float64,
        )

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, BlinkResult]:
        """Process one BGR frame and return an annotated frame and result."""
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        output = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not output.multi_face_landmarks:
            # Do not count an eye closure if tracking is lost mid-event.
            self.closed_frames = 0
            self.was_closed = False
            return frame, BlinkResult(None, "Face not detected", self.blink_count, False)

        landmarks = output.multi_face_landmarks[0].landmark
        left_points = self._eye_points(landmarks, LEFT_EYE, width, height)
        right_points = self._eye_points(landmarks, RIGHT_EYE, width, height)
        left_ear = eye_aspect_ratio(left_points)
        right_ear = eye_aspect_ratio(right_points)
        ear = (left_ear + right_ear) / 2.0

        # Draw the eye contours to make the landmarks visible in the demo.
        for indices in (LEFT_EYE, RIGHT_EYE):
            contour = np.asarray(
                [[int(landmarks[i].x * width), int(landmarks[i].y * height)] for i in indices],
                dtype=np.int32,
            )
            cv2.polylines(frame, [contour], True, (0, 220, 255), 1, cv2.LINE_AA)

        if ear < self.ear_threshold:
            self.closed_frames += 1
            if self.closed_frames >= self.min_closed_frames:
                self.was_closed = True
            eye_state = "Closed" if self.was_closed else "Closing"
        else:
            if self.was_closed:
                self.blink_count += 1
            self.closed_frames = 0
            self.was_closed = False
            eye_state = "Open"

        return frame, BlinkResult(ear, eye_state, self.blink_count, True)

    def reset_count(self) -> None:
        """Reset the displayed count and any in-progress blink event."""
        self.blink_count = 0
        self.closed_frames = 0
        self.was_closed = False

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.face_mesh.close()
