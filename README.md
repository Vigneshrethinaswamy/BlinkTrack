# BlinkTrack — Real-Time Eye Blink Detection

BlinkTrack is a small computer vision project that uses a laptop webcam, OpenCV, and MediaPipe Face Mesh to detect and count complete eye blinks in real time.

## Features

- Captures live video from the default webcam.
- Detects facial landmarks using MediaPipe Face Mesh.
- Extracts six landmarks around each eye.
- Calculates the Eye Aspect Ratio (EAR) for both eyes.
- Counts a blink only after the eyes close for a minimum number of frames and reopen.
- Displays the live blink count, EAR, eye state, and FPS.
- Exits safely when you press **Q** or **Esc**.

## How it works

1. OpenCV reads a frame from the webcam.
2. The frame is mirrored and converted from BGR to RGB.
3. MediaPipe Face Mesh locates facial landmarks.
4. BlinkTrack extracts the six landmark points for each eye.
5. It calculates the EAR for each eye and averages the two values.
6. If EAR stays below the configured threshold for enough consecutive frames, the eyes are marked closed.
7. When the eyes reopen, the blink counter increases once.

The Eye Aspect Ratio is:

\[
EAR = \frac{\|p_2-p_6\| + \|p_3-p_5\|}{2\|p_1-p_4\|}
\]

A lower EAR generally indicates a more closed eye. The threshold is configurable because camera angle, face shape, and lighting can affect the measured value.

## Requirements

- Windows, macOS, or Linux
- Python 3.11 recommended
- A working webcam

## Setup (Windows PowerShell)

Clone the repository, open PowerShell in the project folder, and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```bash
python main.py
```

The first run may take a little longer while MediaPipe initializes. Allow camera access if your operating system asks for permission.

### Controls

- **Q** or **Esc** — quit
- **R** — reset the blink counter

## Calibration

The default EAR threshold is `0.21` and the minimum closed duration is `2` consecutive frames. These are starting values, not universal constants.

If blinks are missed, try increasing `EAR_THRESHOLD` slightly (for example, to `0.23`). If normal open eyes are incorrectly treated as closed, decrease it (for example, to `0.19`). Change the constants near the top of `blink_detector.py`, then restart the program.

For a fair demonstration:
- Keep your face visible and reasonably centered.
- Use steady, front-facing lighting.
- Avoid extreme camera angles.
- Look toward the webcam while testing.

## Project structure

```text
BlinkTrack/
├── main.py
├── blink_detector.py
├── requirements.txt
├── utils/
│   ├── __init__.py
│   └── eye_aspect_ratio.py
├── tests/
│   └── test_blink_detector.py
├── .gitignore
└── README.md
```

## Limitations

- This is a simple single-face blink counter, not a medical device.
- It counts completed close-and-reopen events; it does not identify a person's identity.
- Glasses, occlusion, poor lighting, and large head movements can reduce landmark accuracy.
- The counter is session-based and resets when the program is restarted.

## Technologies

- Python
- OpenCV
- MediaPipe Face Mesh
- NumPy

## Academic note

The project demonstrates webcam acquisition, facial landmark detection, geometric feature extraction, threshold-based event detection, and real-time visual feedback.
