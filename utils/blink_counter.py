"""Frame-based blink event state machine, independent of webcam/MediaPipe."""

from __future__ import annotations


class BlinkCounter:
    """Count one event after a sustained eye closure followed by reopening."""

    def __init__(self, ear_threshold: float = 0.21, min_closed_frames: int = 2) -> None:
        if ear_threshold <= 0:
            raise ValueError("ear_threshold must be positive")
        if min_closed_frames < 1:
            raise ValueError("min_closed_frames must be at least 1")
        self.ear_threshold = ear_threshold
        self.min_closed_frames = min_closed_frames
        self.count = 0
        self.closed_frames = 0
        self.was_closed = False

    def update(self, ear: float) -> str:
        """Update state from one EAR value and return Open/Closing/Closed."""
        if ear < self.ear_threshold:
            self.closed_frames += 1
            if self.closed_frames >= self.min_closed_frames:
                self.was_closed = True
            return "Closed" if self.was_closed else "Closing"
        if self.was_closed:
            self.count += 1
        self.closed_frames = 0
        self.was_closed = False
        return "Open"

    def cancel_pending(self) -> None:
        """Discard an incomplete closure when face tracking is lost."""
        self.closed_frames = 0
        self.was_closed = False

    def reset(self) -> None:
        self.count = 0
        self.cancel_pending()
