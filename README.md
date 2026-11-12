# BlinkTrack — Real-Time Eye Blink and Wink Detection

BlinkTrack is a local computer-vision demo that uses a webcam, OpenCV, and MediaPipe Face Mesh to track each eye independently, count bilateral blinks, and distinguish one-eye winks.

## Features

- Real-time webcam capture and face landmark tracking.
- Separate Eye Aspect Ratio (EAR) measurements for the left and right eyes.
- Bilateral blink counting with a short synchronization window.
- Separate left-wink and right-wink counters; closing one eye does not count as a bilateral blink.
- Hysteresis thresholds to reduce state flicker near the EAR boundary.
- Live face-tracking status, per-eye EAR/state, blink/wink counts, and FPS.
- Q/Esc to quit and R to reset all session counts.
- Local processing only; frames are not uploaded or saved.

## How it works

1. OpenCV captures and mirrors the webcam frame.
2. MediaPipe Face Mesh estimates facial landmarks.
3. BlinkTrack extracts six landmarks around each eye and calculates two independent EAR values.
4. Each eye has its own open/closed state, using a close threshold and a slightly higher reopen threshold.
5. When both eyes begin closing within the synchronization window and both reopen, one bilateral blink is counted.
6. If only one eye closes, the event is classified as a left or right wink after the pairing window expires.

EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

A lower EAR generally indicates a more closed eye. The left and right values are evaluated separately; they are not averaged before classification.

## Requirements

- Windows, macOS, or Linux
- Python 3.11
- A working webcam

## Setup (Windows PowerShell)

In the project folder, run:

    .\setup_and_run.ps1

If PowerShell blocks script execution, run:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\setup_and_run.ps1

The script creates/uses the project virtual environment, installs dependencies, runs unit tests, and launches the webcam demo.

To run manually after setup:

    .\.venv\Scripts\Activate.ps1
    python main.py

Run tests without opening the webcam:

    python -m pytest -q

## Controls and display

- Q or Esc — quit
- R — reset blink and wink counts
- Cyan/yellow eye outlines — separate eye landmark tracks
- FACE: TRACKED — MediaPipe currently sees a face

## Calibration

Initial defaults in blink_detector.py:

- EAR close threshold: 0.21
- EAR reopen threshold: 0.24
- Bilateral synchronization window: 5 frames

The reopen threshold is intentionally higher than the close threshold to prevent rapid state switching when EAR fluctuates near the boundary. The synchronization window allows the two eyes to begin closing a few frames apart.

These are starting values, not universal constants. If an eye is consistently classified incorrectly while open, inspect its displayed EAR and adjust the close/open thresholds together. Use steady front-facing lighting and keep the face reasonably centered. Very low frame rates, glasses, occlusion, and extreme head angles can still reduce reliability.

## Project structure

    BlinkTrack/
    ├── main.py
    ├── blink_detector.py
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── setup_and_run.ps1
    ├── utils/
    │   ├── __init__.py
    │   ├── blink_counter.py
    │   └── eye_aspect_ratio.py
    ├── tests/
    │   └── test_blink_detector.py
    └── README.md

## Limitations

- Single-face, single-webcam demonstration; it does not identify people.
- A wink is an eye closure classified as unilateral, not an intent or emotion label.
- EAR is a geometric estimate and can be affected by glasses, lighting, head pose, and landmark jitter.
- This is an educational computer-vision project, not a medical or drowsiness diagnosis tool.

## Technologies

Python, OpenCV, MediaPipe Face Mesh, NumPy, and pytest.
