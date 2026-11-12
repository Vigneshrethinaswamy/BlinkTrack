"""Fast unit tests that do not require a webcam."""

import numpy as np
import pytest

from utils.eye_aspect_ratio import eye_aspect_ratio


def test_eye_aspect_ratio_for_open_eye_geometry():
    points = np.asarray(
        [[0, 0], [2, -2], [8, -2], [10, 0], [8, 2], [2, 2]],
        dtype=float,
    )
    assert eye_aspect_ratio(points) == pytest.approx(0.4)


def test_eye_aspect_ratio_rejects_wrong_number_of_points():
    with pytest.raises(ValueError):
        eye_aspect_ratio([[0, 0], [1, 1]])


def test_eye_aspect_ratio_rejects_zero_horizontal_distance():
    points = np.asarray([[1, 0], [2, 1], [2, -1], [1, 0], [2, -1], [2, 1]])
    with pytest.raises(ValueError):
        eye_aspect_ratio(points)
