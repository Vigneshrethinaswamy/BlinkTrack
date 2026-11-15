"""Camera/video-file input with conservative low-latency capture settings."""
from __future__ import annotations

import os
from typing import Any

import cv2


def parse_source(value: str) -> int | str:
    """Convert numeric source strings to camera indices; keep paths as strings."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


class VideoSource:
    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.source = source
        self.is_camera = isinstance(source, int)
        self.capture = self._open_capture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open video source: {source}")

        if self.is_camera:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Supported by some backends; ignored safely by others.
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    @staticmethod
    def _open_capture(source: int | str) -> Any:
        if isinstance(source, int) and os.name == "nt":
            capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if capture.isOpened():
                return capture
            capture.release()
        return cv2.VideoCapture(source)

    @property
    def frame_size(self) -> tuple[int, int]:
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    @property
    def fps(self) -> float:
        value = float(self.capture.get(cv2.CAP_PROP_FPS))
        return value if value > 0 else 0.0

    def read(self):
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()
