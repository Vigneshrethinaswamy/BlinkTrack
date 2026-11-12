"""Fast unit tests that do not require a webcam."""

import numpy as np
import pytest

from utils.blink_counter import BlinkCounter
from utils.eye_aspect_ratio import eye_aspect_ratio


def test_eye_aspect_ratio_for_open_eye_geometry():
    points = np.asarray([[0, 0], [2, -2], [8, -2], [10, 0], [8, 2], [2, 2]], dtype=float)
    assert eye_aspect_ratio(points) == pytest.approx(0.4)


def test_eye_aspect_ratio_rejects_wrong_number_of_points():
    with pytest.raises(ValueError):
        eye_aspect_ratio([[0, 0], [1, 1]])


def test_eye_aspect_ratio_rejects_zero_horizontal_distance():
    points = np.asarray([[1, 0], [2, 1], [2, -1], [1, 0], [2, -1], [2, 1]])
    with pytest.raises(ValueError):
        eye_aspect_ratio(points)


def test_one_blink_is_counted_after_reopening():
    counter = BlinkCounter(0.2, 2)
    assert counter.update(0.3) == "Open"
    assert counter.update(0.1) == "Closing"
    assert counter.update(0.1) == "Closed"
    assert counter.count == 0
    counter.update(0.1)
    counter.update(0.3)
    assert counter.count == 1


def test_prolonged_closure_counts_only_once():
    counter = BlinkCounter(0.2, 2)
    for _ in range(20):
        counter.update(0.1)
    assert counter.count == 0
    counter.update(0.3)
    assert counter.count == 1


def test_short_eye_movement_does_not_count_as_blink():
    counter = BlinkCounter(0.2, 3)
    counter.update(0.1)
    counter.update(0.3)
    assert counter.count == 0


def test_cancel_pending_closure_when_face_is_lost():
    counter = BlinkCounter(0.2, 2)
    counter.update(0.1)
    counter.update(0.1)
    counter.cancel_pending()
    counter.update(0.3)
    assert counter.count == 0


def test_reset_clears_count_and_pending_state():
    counter = BlinkCounter(0.2, 1)
    counter.update(0.1)
    counter.update(0.3)
    counter.reset()
    assert counter.count == 0
    assert counter.update(0.3) == "Open"
