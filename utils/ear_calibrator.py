"""Small open-eye EAR calibration helper."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class CalibrationResult:
    baseline_ear: float
    close_threshold: float
    open_threshold: float
    samples: int


class EARCalibrator:
    """Collect stable open-eye EAR values and derive conservative thresholds."""

    def __init__(self, required_samples: int = 45) -> None:
        if required_samples < 10:
            raise ValueError("required_samples must be at least 10")
        self.required_samples = required_samples
        self.left_samples: list[float] = []
        self.right_samples: list[float] = []

    @property
    def progress(self) -> float:
        return min(1.0, len(self.left_samples) / self.required_samples)

    @property
    def complete(self) -> bool:
        return len(self.left_samples) >= self.required_samples

    def add(self, left_ear: float | None, right_ear: float | None) -> bool:
        """Add a plausible open-eye sample; returns True when collection is complete."""
        if left_ear is None or right_ear is None:
            return self.complete
        left, right = float(left_ear), float(right_ear)
        # Skip obvious closures and invalid landmark measurements.
        if not (0.18 <= left <= 0.8 and 0.18 <= right <= 0.8):
            return self.complete
        self.left_samples.append(left)
        self.right_samples.append(right)
        return self.complete

    def result(self) -> CalibrationResult:
        if not self.complete:
            raise RuntimeError("Calibration does not have enough valid samples")
        left_baseline = median(self.left_samples[: self.required_samples])
        right_baseline = median(self.right_samples[: self.required_samples])
        baseline = (left_baseline + right_baseline) / 2.0

        # Relative thresholds accommodate differences in camera distance and eye shape.
        close_threshold = max(0.12, min(0.30, baseline * 0.72))
        open_threshold = max(close_threshold + 0.02, min(0.40, baseline * 0.86))
        return CalibrationResult(
            baseline, close_threshold, open_threshold, self.required_samples
        )
