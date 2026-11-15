"""Compact OpenCV dashboard for live blink diagnostics."""
from __future__ import annotations

import cv2

from utils.frame_enhancer import FrameQuality


def _text(
    image,
    value: str,
    x: int,
    y: int,
    scale: float = 0.48,
    color: tuple[int, int, int] = (225, 232, 240),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def _eye_row(image, label: str, ear: float | None, state: str,
             x: int, y: int, width: int, color, close_threshold: float) -> None:
    value = f"{ear:.3f}" if ear is not None else "--"
    _text(image, f"{label}  {value}", x, y, 0.46)
    state_color = (80, 220, 120) if state == "Open" else (80, 170, 255)
    _text(image, state.upper(), x + width - 58, y, 0.40, state_color, 2)

    bar_y = y + 10
    bar_width = max(90, width - 12)
    cv2.rectangle(image, (x, bar_y), (x + bar_width, bar_y + 7), (55, 65, 78), -1)
    if ear is not None:
        ratio = max(0.0, min(1.0, ear / 0.45))
        fill = int(bar_width * ratio)
        cv2.rectangle(image, (x, bar_y), (x + fill, bar_y + 7), color, -1)
        threshold_x = x + int(bar_width * min(1.0, close_threshold / 0.45))
        cv2.line(image, (threshold_x, bar_y - 3), (threshold_x, bar_y + 10),
                 (245, 245, 245), 1)


def draw_dashboard(
    frame,
    result,
    fps: float,
    quality: FrameQuality,
    enhancement_mode: str,
    close_threshold: float,
    open_threshold: float,
    calibration_progress: float | None = None,
):
    """Draw a readable diagnostics panel while keeping most of the camera visible."""
    height, width = frame.shape[:2]
    scale = max(0.68, min(1.0, height / 480.0))
    y = lambda value: int(round(value * scale))
    panel_width = min(330, width - 24)
    panel_height = min(y(270), height - 24)
    x0, y0 = 12, 12
    x1, y1 = x0 + panel_width, y0 + panel_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 24, 34), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (70, 84, 102), 1)

    left = x0 + 14
    inner_width = panel_width - 28
    _text(frame, "BLINKTRACK", left, y0 + y(25), 0.62 * scale, (245, 248, 252), 2)
    _text(frame, "LIVE EYE TRACKING", left + 145, y0 + y(24), 0.34 * scale, (155, 177, 198))
    cv2.line(frame, (left, y0 + y(36)), (x1 - 12, y0 + y(36)), (65, 78, 94), 1)

    face_color = (75, 220, 130) if result.face_detected else (75, 90, 245)
    face_label = "FACE TRACKED" if result.face_detected else "FACE NOT FOUND"
    _text(frame, face_label, left, y0 + y(60), 0.48 * scale, face_color, 2)
    _text(frame, f"{fps:4.1f} FPS  |  {width}x{height}", left + 155,
          y0 + y(60), 0.36 * scale, (185, 197, 211))

    _text(frame, "BLINKS", left, y0 + y(91), 0.36 * scale, (160, 177, 197))
    _text(frame, str(result.blink_count), left, y0 + y(124), 0.88 * scale, (90, 225, 145), 2)
    _text(frame, f"WINKS   L {result.left_wink_count}   R {result.right_wink_count}",
          left + 108, y0 + y(112), 0.43 * scale, (220, 218, 185))

    _eye_row(frame, "LEFT EAR", result.left_ear, result.left_state,
             left, y0 + y(153), inner_width, (255, 190, 70), close_threshold)
    _eye_row(frame, "RIGHT EAR", result.right_ear, result.right_state,
             left, y0 + y(190), inner_width, (70, 215, 235), close_threshold)

    quality_color = (90, 220, 135) if not quality.issues else (80, 180, 255)
    _text(frame, f"CAMERA: {quality.label}", left, y0 + y(229), 0.39 * scale, quality_color, 1)
    _text(frame, f"Light {quality.brightness:.0f}  Contrast {quality.contrast:.0f}  "
                 f"Sharp {quality.sharpness:.0f}",
          left, y0 + y(247), 0.31 * scale, (175, 190, 205))
    _text(frame, f"ENH: {enhancement_mode.upper()}  |  EAR CLOSE/OPEN: "
                 f"{close_threshold:.3f}/{open_threshold:.3f}",
          left, y0 + y(263), 0.30 * scale, (190, 205, 220))

    if calibration_progress is not None:
        message = f"CALIBRATING OPEN EYES  {int(calibration_progress * 100)}%"
        bar_x, bar_y = left, min(height - 18, y1 + 9)
        _text(frame, message, bar_x, bar_y, 0.42 * scale, (100, 220, 255), 2)
        cv2.rectangle(frame, (bar_x, bar_y + 7),
                      (bar_x + inner_width, bar_y + 13), (55, 65, 78), -1)
        cv2.rectangle(frame, (bar_x, bar_y + 7),
                      (bar_x + int(inner_width * calibration_progress), bar_y + 13),
                      (70, 190, 245), -1)

    footer = "Q quit  |  R reset  |  C calibrate  |  E enhancement  |  S screenshot"
    _text(frame, footer, 12, height - 10, 0.38 * scale, (235, 235, 235))
    return frame
