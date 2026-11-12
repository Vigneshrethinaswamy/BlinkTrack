"""Unit tests for EAR geometry and the blink/wink state machine."""
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


def test_synchronous_fast_blink_counts_once():
    counter = BlinkCounter(sync_window_frames=5)
    counter.update(0.30, 0.31)
    counter.update(0.18, 0.19)
    assert counter.blink_count == 0
    counter.update(0.30, 0.31)
    for _ in range(6):
        counter.update(0.30, 0.31)
    assert counter.blink_count == 1
    assert counter.left_wink_count == 0
    assert counter.right_wink_count == 0


def test_asynchronous_bilateral_blink_within_window_counts_once():
    counter = BlinkCounter(sync_window_frames=5)
    counter.update(0.18, 0.30)
    counter.update(0.30, 0.18)
    counter.update(0.30, 0.30)
    assert counter.blink_count == 1
    assert counter.left_wink_count == 0
    assert counter.right_wink_count == 0


def test_left_wink_is_not_counted_as_bilateral_blink():
    counter = BlinkCounter(sync_window_frames=3)
    counter.update(0.18, 0.30)
    counter.update(0.30, 0.30)
    for _ in range(4):
        counter.update(0.30, 0.30)
    assert counter.blink_count == 0
    assert counter.left_wink_count == 1
    assert counter.right_wink_count == 0


def test_right_wink_is_not_counted_as_bilateral_blink():
    counter = BlinkCounter(sync_window_frames=3)
    counter.update(0.30, 0.18)
    counter.update(0.30, 0.30)
    for _ in range(4):
        counter.update(0.30, 0.30)
    assert counter.blink_count == 0
    assert counter.left_wink_count == 0
    assert counter.right_wink_count == 1


def test_prolonged_bilateral_closure_counts_only_once_after_reopening():
    counter = BlinkCounter()
    for _ in range(30):
        counter.update(0.15, 0.16)
    assert counter.blink_count == 0
    counter.update(0.30, 0.31)
    assert counter.blink_count == 1
    for _ in range(5):
        counter.update(0.30, 0.31)
    assert counter.blink_count == 1


def test_hysteresis_prevents_threshold_chatter():
    counter = BlinkCounter()
    counter.update(0.20, 0.30)
    assert counter.update(0.225, 0.30)[0] == "Closed"
    assert counter.update(0.245, 0.30)[0] == "Open"
    for _ in range(6):
        counter.update(0.30, 0.30)
    assert counter.left_wink_count == 1


def test_face_loss_cancels_unfinished_and_unclassified_events():
    counter = BlinkCounter()
    counter.update(0.18, 0.30)
    counter.update(0.30, 0.30)
    counter.cancel_pending()
    for _ in range(10):
        counter.update(0.30, 0.30)
    assert counter.blink_count == 0
    assert counter.left_wink_count == 0


def test_reset_clears_counts_and_tracking_state():
    counter = BlinkCounter()
    counter.update(0.18, 0.18)
    counter.update(0.30, 0.30)
    assert counter.blink_count == 1
    counter.reset()
    assert counter.blink_count == 0
    assert counter.left_wink_count == 0
    assert counter.right_wink_count == 0
    assert counter.update(0.30, 0.30) == ("Open", "Open")
