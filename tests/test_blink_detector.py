"""Tests for EAR geometry, blink/wink state and video helper modules."""
import numpy as np
import pytest

from utils.blink_counter import BlinkCounter
from utils.ear_calibrator import EARCalibrator
from utils.eye_aspect_ratio import eye_aspect_ratio
from utils.frame_enhancer import VideoEnhancer
from video_source import parse_source


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


def test_synchronous_one_frame_fast_blink_counts_once():
    counter = BlinkCounter()
    counter.update(0.30, 0.31)
    counter.update(0.18, 0.19)  # a single closed frame
    assert counter.blink_count == 0
    counter.update(0.30, 0.31)  # first open frame: confirmation still pending
    assert counter.blink_count == 0
    counter.update(0.30, 0.31)  # second consecutive open frame confirms reopening
    assert counter.blink_count == 1
    for _ in range(5):
        counter.update(0.30, 0.31)
    assert counter.blink_count == 1
    assert counter.left_wink_count == 0
    assert counter.right_wink_count == 0


def test_asynchronous_bilateral_blink_within_window_counts_once():
    counter = BlinkCounter(sync_window_frames=5)
    counter.update(0.18, 0.30)
    counter.update(0.18, 0.18)
    counter.update(0.30, 0.30)
    assert counter.blink_count == 0
    counter.update(0.30, 0.30)
    assert counter.blink_count == 1
    assert counter.left_wink_count == 0
    assert counter.right_wink_count == 0


@pytest.mark.parametrize("side", ["left", "right"])
def test_single_wink_counts_once_after_reopening(side):
    counter = BlinkCounter(sync_window_frames=3)
    values = {"left": 0.30, "right": 0.30}
    values[side] = 0.18
    counter.update(values["left"], values["right"])
    values[side] = 0.30
    counter.update(values["left"], values["right"])
    counter.update(values["left"], values["right"])
    for _ in range(4):
        counter.update(0.30, 0.30)
    assert counter.blink_count == 0
    assert counter.left_wink_count == (1 if side == "left" else 0)
    assert counter.right_wink_count == (1 if side == "right" else 0)


def test_held_one_eye_closure_ignores_single_frame_open_landmark_spikes():
    counter = BlinkCounter()
    for frame in range(120):
        # Simulate a held wink with one noisy frame that briefly looks open
        # every ten frames, as can happen with a low-quality webcam.
        right_ear = 0.27 if frame % 10 == 6 else 0.15
        counter.update(0.30, right_ear)
    assert counter.blink_count == 0
    assert counter.left_wink_count == 0
    assert counter.right_wink_count == 0


def test_held_one_eye_closure_counts_once_when_it_really_reopens():
    counter = BlinkCounter(sync_window_frames=3)
    for _ in range(40):
        counter.update(0.30, 0.15)
    assert counter.left_wink_count == 0
    counter.update(0.30, 0.28)
    assert counter.left_wink_count == 0
    counter.update(0.30, 0.28)
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
    assert counter.update(0.245, 0.30)[0] == "Closed"
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
    counter.update(0.30, 0.30)
    assert counter.blink_count == 1
    counter.reset()
    assert counter.blink_count == 0
    assert counter.left_wink_count == 0
    assert counter.right_wink_count == 0
    assert counter.update(0.30, 0.30) == ("Open", "Open")


def test_sequential_non_overlapping_winks_are_not_paired_as_a_blink():
    counter = BlinkCounter(sync_window_frames=5)
    counter.update(0.18, 0.30)
    counter.update(0.30, 0.30)
    counter.update(0.30, 0.18)
    counter.update(0.30, 0.30)
    counter.update(0.30, 0.30)
    for _ in range(6):
        counter.update(0.30, 0.30)
    assert counter.blink_count == 0
    assert counter.left_wink_count == 1
    assert counter.right_wink_count == 1


def test_ear_calibrator_derives_thresholds_from_open_eye_baseline():
    calibrator = EARCalibrator(required_samples=10)
    for _ in range(10):
        calibrator.add(0.30, 0.32)
    result = calibrator.result()
    assert calibrator.complete
    assert result.baseline_ear == pytest.approx(0.31)
    assert result.close_threshold == pytest.approx(0.2232)
    assert result.open_threshold == pytest.approx(0.2666)


def test_ear_calibrator_ignores_missing_or_closed_eye_samples():
    calibrator = EARCalibrator(required_samples=10)
    for _ in range(5):
        calibrator.add(None, 0.3)
        calibrator.add(0.08, 0.3)
    assert calibrator.progress == 0
    assert not calibrator.complete


def test_enhancer_reports_dark_low_contrast_frame():
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    quality = VideoEnhancer.measure_quality(frame)
    assert quality.brightness == 0
    assert "low light" in quality.issues
    assert "low contrast" in quality.issues


def test_enhancement_off_returns_original_frame():
    frame = np.full((100, 160, 3), 70, dtype=np.uint8)
    enhancer = VideoEnhancer("off")
    output, quality = enhancer.process(frame)
    assert output is frame
    assert quality.brightness == pytest.approx(70)


def test_enhancer_rejects_unknown_mode():
    with pytest.raises(ValueError):
        VideoEnhancer("magic")


def test_video_source_parses_camera_index_and_keeps_video_path():
    assert parse_source("0") == 0
    assert parse_source("2") == 2
    assert parse_source("sample_clip.mp4") == "sample_clip.mp4"



def test_blink_detector_initializes_and_processes_a_frame():
    from blink_detector import BlinkDetector

    detector = BlinkDetector()
    try:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        output, result = detector.process(frame)
        assert output.shape == frame.shape
        assert result.face_detected is False
        assert result.left_ear is None
        assert result.right_ear is None
    finally:
        detector.close()



def test_dashboard_renders_in_separate_image_without_modifying_camera_frame():
    from types import SimpleNamespace
    from utils.dashboard import draw_dashboard

    camera = np.full((360, 480, 3), 127, dtype=np.uint8)
    original = camera.copy()
    result = SimpleNamespace(
        face_detected=True,
        blink_count=3,
        left_wink_count=1,
        right_wink_count=0,
        left_ear=0.28,
        right_ear=0.29,
        left_state="Open",
        right_state="Open",
    )
    quality = SimpleNamespace(
        label="GOOD", issues=[], brightness=120.0,
        contrast=45.0, sharpness=80.0,
    )

    panel = draw_dashboard(camera, result, 30.0, quality, "auto", 0.21, 0.24)

    assert panel.shape == (360, 421, 3)
    assert np.array_equal(camera, original)
    assert not np.array_equal(panel[:, :], camera[:, :panel.shape[1]])
