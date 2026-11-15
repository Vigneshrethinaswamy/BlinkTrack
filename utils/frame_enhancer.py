"""Lightweight real-time image enhancement and camera-quality diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    brightness: float
    contrast: float
    sharpness: float
    issues: tuple[str, ...]

    @property
    def label(self) -> str:
        if not self.issues:
            return "GOOD"
        return " + ".join(issue.upper() for issue in self.issues)


class VideoEnhancer:
    """Enhance low-light/low-contrast frames without adding heavy dependencies.

    Modes:
      * auto: only enhance frames whose measured brightness/contrast is poor.
      * low-light: always apply gentle local-contrast correction and adaptive gamma.
      * off: return the source frame unchanged.
    """

    MODES = ("auto", "low-light", "off")

    def __init__(self, mode: str = "auto") -> None:
        self.set_mode(mode)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._gamma_lut_cache: dict[int, np.ndarray] = {}

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode

    @staticmethod
    def measure_quality(frame: np.ndarray) -> FrameQuality:
        if frame is None or frame.size == 0:
            raise ValueError("Cannot measure an empty frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        scale = min(1.0, 320.0 / max(width, height))
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        issues: list[str] = []
        if brightness < 65:
            issues.append("low light")
        if contrast < 32:
            issues.append("low contrast")
        if sharpness < 22:
            issues.append("soft image")
        return FrameQuality(brightness, contrast, sharpness, tuple(issues))

    def _gamma_lut(self, gamma: float) -> np.ndarray:
        key = int(round(gamma * 100))
        if key not in self._gamma_lut_cache:
            normalized_gamma = key / 100.0
            table = np.array(
                [((value / 255.0) ** normalized_gamma) * 255
                 for value in range(256)],
                dtype=np.uint8,
            )
            self._gamma_lut_cache[key] = table
        return self._gamma_lut_cache[key]

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, FrameQuality]:
        """Return an enhanced frame plus quality measurements from the raw frame."""
        quality = self.measure_quality(frame)
        if self.mode == "off":
            return frame, quality

        needs_enhancement = (
            self.mode == "low-light"
            or quality.brightness < 92
            or quality.contrast < 40
        )
        if not needs_enhancement:
            return frame, quality

        enhanced = frame.copy()
        if quality.brightness < 105 or quality.contrast < 48:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            lightness, a_channel, b_channel = cv2.split(lab)
            lightness = self._clahe.apply(lightness)
            enhanced = cv2.cvtColor(
                cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR
            )

        if quality.brightness < 100 or self.mode == "low-light":
            gamma = max(0.62, min(0.96, 1.0 - (92.0 - quality.brightness) / 180.0))
            enhanced = cv2.LUT(enhanced, self._gamma_lut(gamma))

        # Very mild unsharp masking can make soft webcam edges easier to locate.
        # It cannot recover detail that the camera never captured.
        if quality.sharpness < 22:
            softened = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
            enhanced = cv2.addWeighted(enhanced, 1.18, softened, -0.18, 0)

        return enhanced, quality
