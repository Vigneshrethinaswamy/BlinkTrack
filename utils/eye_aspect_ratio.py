"""Geometry helpers for eye landmark measurements."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def eye_aspect_ratio(eye_points: Sequence[Sequence[float]]) -> float:
    """Calculate EAR from six 2D eye landmarks.

    Points must be ordered around the eye as:
    [horizontal-left, upper-left, upper-right,
     horizontal-right, lower-right, lower-left].
    """
    points = np.asarray(eye_points, dtype=np.float64)
    if points.shape != (6, 2):
        raise ValueError("eye_points must have shape (6, 2)")

    horizontal = float(np.linalg.norm(points[0] - points[3]))
    if horizontal <= 1e-9:
        raise ValueError("Horizontal eye landmark distance must be non-zero")

    vertical_1 = float(np.linalg.norm(points[1] - points[5]))
    vertical_2 = float(np.linalg.norm(points[2] - points[4]))
    return (vertical_1 + vertical_2) / (2.0 * horizontal)
