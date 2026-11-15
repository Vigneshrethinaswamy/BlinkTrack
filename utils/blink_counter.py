"""Per-eye blink and wink tracking with temporal pairing and reopen debounce."""
from __future__ import annotations


class BlinkCounter:
    """Track bilateral blinks and unilateral winks from separate EAR signals.

    Eye closure is accepted immediately so a one-frame fast blink can register.
    Reopening is confirmed over multiple consecutive frames to prevent a noisy
    landmark estimate from repeatedly ending/restarting a held wink.
    """

    def __init__(
        self,
        close_threshold: float = 0.21,
        open_threshold: float = 0.24,
        sync_window_frames: int = 5,
        reopen_confirm_frames: int = 2,
    ) -> None:
        if close_threshold <= 0 or open_threshold <= close_threshold:
            raise ValueError(
                "open_threshold must be greater than a positive close_threshold"
            )
        if sync_window_frames < 0:
            raise ValueError("sync_window_frames cannot be negative")
        if reopen_confirm_frames < 1:
            raise ValueError("reopen_confirm_frames must be at least 1")

        self.close_threshold = float(close_threshold)
        self.open_threshold = float(open_threshold)
        self.sync_window_frames = int(sync_window_frames)
        self.reopen_confirm_frames = int(reopen_confirm_frames)
        self.blink_count = 0
        self.left_wink_count = 0
        self.right_wink_count = 0
        self._frame = 0
        self._closed = {"left": False, "right": False}
        self._open_streak = {"left": 0, "right": 0}
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
            if not self._closed[side]:
                if ear <= self.close_threshold:
                    self._closed[side] = True
                    self._open_streak[side] = 0
                    event = {
                        "side": side,
                        "start": self._frame,
                        "end": None,
                        "pair": None,
                    }
                    self._active[side] = event
                    self._events.append(event)
                    self._try_pair(event)
                continue

            # While closed, require consecutive open-threshold readings before
            # re-arming this eye. A single noisy landmark spike is ignored.
            if ear >= self.open_threshold:
                self._open_streak[side] += 1
                if self._open_streak[side] >= self.reopen_confirm_frames:
                    self._closed[side] = False
                    self._open_streak[side] = 0
                    event = self._active[side]
                    if event is not None:
                        event["end"] = self._frame
                    self._active[side] = None
            else:
                self._open_streak[side] = 0

        self._resolve_events()
        return (
            "Closed" if self._closed["left"] else "Open",
            "Closed" if self._closed["right"] else "Open",
        )

    def _try_pair(self, event: dict) -> None:
        opposite = "right" if event["side"] == "left" else "left"
        candidates = [
            other for other in self._events
            if other is not event
            and other["side"] == opposite
            and other["pair"] is None
            and other["end"] is None
            and abs(other["start"] - event["start"]) <= self.sync_window_frames
        ]
        if candidates:
            other = min(
                candidates,
                key=lambda item: abs(item["start"] - event["start"]),
            )
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
            self._events = [
                event for event in self._events if id(event) not in remove
            ]

    def cancel_pending(self) -> None:
        """Discard unfinished or unclassified events after face tracking is lost."""
        self._closed = {"left": False, "right": False}
        self._open_streak = {"left": 0, "right": 0}
        self._active = {"left": None, "right": None}
        self._events.clear()

    def reset(self) -> None:
        self.blink_count = 0
        self.left_wink_count = 0
        self.right_wink_count = 0
        self._frame = 0
        self._next_pair_id = 1
        self.cancel_pending()
