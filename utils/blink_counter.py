"""Per-eye blink and wink event tracking with temporal pairing.

The state machine is independent of OpenCV/MediaPipe so it can be tested
without a webcam. A blink is counted only when both eye closures begin within
a short synchronization window and both eyes reopen.
"""
from __future__ import annotations


class BlinkCounter:
    """Track bilateral blinks and unilateral winks from separate EAR signals."""

    def __init__(self, close_threshold: float = 0.21, open_threshold: float = 0.24,
                 sync_window_frames: int = 5) -> None:
        if close_threshold <= 0 or open_threshold <= close_threshold:
            raise ValueError("open_threshold must be greater than a positive close_threshold")
        if sync_window_frames < 0:
            raise ValueError("sync_window_frames cannot be negative")
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.sync_window_frames = sync_window_frames
        self.blink_count = 0
        self.left_wink_count = 0
        self.right_wink_count = 0
        self._frame = 0
        self._closed = {"left": False, "right": False}
        self._active = {"left": None, "right": None}
        self._events: list[dict] = []
        self._next_pair_id = 1

    @property
    def left_closed(self) -> bool:
        return self._closed["left"]

    @property
    def right_closed(self) -> bool:
        return self._closed["right"]

    def update(self, left_ear: float, right_ear: float) -> tuple[str, str]:
        """Process one frame of per-eye EAR values and return each eye's state."""
        self._frame += 1
        values = {"left": float(left_ear), "right": float(right_ear)}
        for side, ear in values.items():
            if not self._closed[side] and ear <= self.close_threshold:
                self._closed[side] = True
                event = {"side": side, "start": self._frame, "end": None, "pair": None}
                self._active[side] = event
                self._events.append(event)
                self._try_pair(event)
            elif self._closed[side] and ear >= self.open_threshold:
                self._closed[side] = False
                event = self._active[side]
                if event is not None:
                    event["end"] = self._frame
                self._active[side] = None
        self._resolve_events()
        return ("Closed" if self._closed["left"] else "Open",
                "Closed" if self._closed["right"] else "Open")

    def _try_pair(self, event: dict) -> None:
        opposite = "right" if event["side"] == "left" else "left"
        candidates = [other for other in self._events
                      if other is not event and other["side"] == opposite
                      and other["pair"] is None
                      and abs(other["start"] - event["start"]) <= self.sync_window_frames]
        if candidates:
            other = min(candidates, key=lambda item: abs(item["start"] - event["start"]))
            pair_id = self._next_pair_id
            self._next_pair_id += 1
            event["pair"] = pair_id
            other["pair"] = pair_id

    def _resolve_events(self) -> None:
        handled_pairs: set[int] = set()
        remove: set[int] = set()
        for event in self._events:
            pair_id = event["pair"]
            if pair_id is not None:
                if pair_id in handled_pairs:
                    continue
                handled_pairs.add(pair_id)
                pair = [item for item in self._events if item["pair"] == pair_id]
                if len(pair) == 2 and all(item["end"] is not None for item in pair):
                    self.blink_count += 1
                    remove.update(id(item) for item in pair)
            elif event["end"] is not None and (
                self._frame - event["start"] > self.sync_window_frames
            ):
                if event["side"] == "left":
                    self.left_wink_count += 1
                else:
                    self.right_wink_count += 1
                remove.add(id(event))
        if remove:
            self._events = [event for event in self._events if id(event) not in remove]

    def cancel_pending(self) -> None:
        """Discard unfinished or unclassified events after face tracking is lost."""
        self._closed = {"left": False, "right": False}
        self._active = {"left": None, "right": None}
        self._events.clear()

    def reset(self) -> None:
        self.blink_count = 0
        self.left_wink_count = 0
        self.right_wink_count = 0
        self._frame = 0
        self._next_pair_id = 1
        self.cancel_pending()
