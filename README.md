# BlinkTrack — Real-Time Eye Blink and Wink Detection

BlinkTrack is a local computer-vision demo using OpenCV, MediaPipe Face Mesh and NumPy. It can process a webcam stream or a prerecorded video, estimate each eye's EAR independently, count bilateral blinks, and classify one-eye closures as winks.

## Features

- Webcam and prerecorded-video input.
- Adaptive frame enhancement for dim or low-contrast webcam images.
- Live camera-quality diagnostics: brightness, contrast and sharpness.
- Open-eye EAR calibration using 45 valid samples.
- Separate left/right EAR, eye state, blink and wink counters.
- Redesigned dashboard with tracking state, counts, EAR bars, quality flags and FPS.
- Optional annotated-video recording and dashboard screenshots.
- Local processing; no frame is uploaded. Recording is off unless explicitly requested.

## Setup (Windows PowerShell)

Run from the repository folder:

    .\setup_and_run.ps1

Or run manually:

    .\.venv\Scripts\Activate.ps1
    python -m pytest -q
    python main.py

## Run a webcam

    python main.py

Default camera settings request 640x480 and process up to 640 pixels wide. If performance is poor, reduce the processing width:

    python main.py --process-width 480

Use a different camera index:

    python main.py --source 1

## Process a video file

    python main.py --source "C:\Videos\blink_test.mp4"

BlinkTrack processes the file frame-by-frame and plays it at the video's reported frame rate. The same enhancement, landmarks, EAR and dashboard pipeline is used. Use a clip with a visible, mostly front-facing face for meaningful blink tests.

## Image enhancement

The default `auto` mode only enhances frames when brightness or contrast measurements indicate that it may help.

    python main.py --enhancement auto
    python main.py --enhancement low-light
    python main.py --enhancement off

Press **E** while running to cycle between these modes.

Enhancement uses CLAHE for local contrast, adaptive gamma brightening for dark images, and very mild sharpening when the frame appears soft. The brightness/contrast/sharpness indicators describe the whole frame, not just the eye region, so treat them as rough diagnostics rather than an autofocus measurement. Enhancement cannot restore eye detail lost to severe blur, focus errors, motion blur or very low resolution. If the camera cannot resolve the eyelids, a better light source or external webcam is still the real fix.

## Calibration

1. Start BlinkTrack and sit in your normal position.
2. Face the camera and keep both eyes comfortably open.
3. Press **C** and continue looking normally for about 1–2 seconds.
4. When 45 valid samples are collected, the app derives new EAR close/open thresholds from your open-eye baseline.

Calibration is optional. Do not blink deliberately while the progress bar is collecting samples. Press **C** again to restart calibration.

## Controls

| Key | Action |
|---|---|
| Q / Esc | Quit |
| R | Reset blink and wink counters |
| C | Calibrate thresholds with eyes open |
| E | Cycle enhancement mode |
| S | Save a dashboard screenshot under `captures/` |

## Optional recording

Nothing is saved by default. To save the annotated output:

    python main.py --record "outputs/blinktrack_annotated.mp4"

For a prerecorded clip, for example:

    python main.py --source "C:\Videos\blink_test.mp4" --record "outputs/test_annotated.mp4"

## Project structure

    BlinkTrack/
    ├── main.py
    ├── blink_detector.py
    ├── video_source.py
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── setup_and_run.ps1
    ├── utils/
    │   ├── blink_counter.py
    │   ├── dashboard.py
    │   ├── ear_calibrator.py
    │   ├── eye_aspect_ratio.py
    │   └── frame_enhancer.py
    └── tests/
        └── test_blink_detector.py

## Limitations

- This is a single-face educational demo, not identity recognition or medical/drowsiness diagnosis.
- EAR and landmark quality depend on face pose, camera focus, lighting, glasses, motion blur and resolution.
- Enhancement can improve exposure and local contrast, but cannot create detail that the camera did not capture.
- Video-file playback is not a substitute for live camera testing; a recorded clip's frame rate and quality affect results.
