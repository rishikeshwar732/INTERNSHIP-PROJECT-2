# 🎥 Face Tracker — Interview Proctor

An AI-powered interview proctoring tool that monitors candidate attention using real-time face and eye tracking via MediaPipe. If a candidate looks away too many times, the interview is automatically cancelled.

---

## ✨ Features

- **Head direction tracking** — detects if the candidate looks LEFT, RIGHT, UP, or DOWN
- **Eye/iris tracking** — tracks iris movement with EMA smoothing for stability
- **No-face detection** — flags when no face is visible in the frame
- **Cooldown-based counters** — prevents rapid duplicate violations from inflating counts
- **Auto-cancel** — displays a cancellation screen and ends the session when limits are exceeded

---

## 📁 Project Structure

```
OpenCV_Task2/
├── opencv/                              # OpenCV resources & assets
├── haarcascade_frontalface_default.xml  # Haar cascade for face detection
├── main.py                              # Main application script
└── requirements.txt                     # Python dependencies
```

---

## 📋 Requirements

- Python 3.11.9
- Mediapipe 0.10.32
- A webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python mediapipe==0.10.32 numpy
```

> ⚠️ This project uses **MediaPipe 0.10.32** on **Python 3.11.9**. Using a different version may cause compatibility issues.

---

## 🚀 Usage

```bash
python main.py
```

- The webcam feed will open in a window titled **"Interview Proctor"**
- Press **`q`** to quit manually at any time

---

## ⚙️ Configuration

All thresholds are defined at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `SMOOTHING` | `0.8` | EMA smoothing factor for iris position |
| `EYE_DEADZONE` | `2` | Pixel deadzone before eye movement is registered |
| `EYE_THRESHOLD` | `4` | Pixel threshold to classify eye direction |
| `MAX_FACE_COUNT` | `5` | Max allowed face-away / no-face violations |
| `MAX_EYE_COUNT` | `8` | Max allowed eye-away violations |
| `FACE_COOLDOWN` | `1.0s` | Minimum seconds between face violation counts |
| `EYE_COOLDOWN` | `0.6s` | Minimum seconds between eye violation counts |

---

## 🚨 Violation Rules

| Event | Effect |
|---|---|
| No face detected | Increments face count immediately |
| Head turns away from center | Increments face count (with cooldown) |
| Eyes look away from center | Increments eye count (with cooldown) |
| `face_count > MAX_FACE_COUNT` | ❌ Interview cancelled |
| `eye_count > MAX_EYE_COUNT` | ❌ Interview cancelled |

When cancelled, a red overlay screen is shown for 2 seconds before the program exits.

---

## 🧠 How It Works

1. **MediaPipe FaceMesh** detects 468 facial landmarks + iris refinement per frame
2. **Head direction** is calculated by comparing the nose tip offset relative to the horizontal face midpoint and vertical face center
3. **Iris direction** is calculated by finding the offset of the iris center from the eye corner average
4. **EMA smoothing** (`new = α × old + (1−α) × raw`) reduces jitter in iris tracking
5. Violations are counted only when direction *changes* from center and a cooldown has elapsed

---

## ⚡ Performance Issues

If you experience lag or slow frame rates, try the following:

- Lower the input resolution in `cv2.resize()` (e.g., from `640x480` to `320x240`) for faster processing
- Reduce `min_detection_confidence` and `min_tracking_confidence` slightly for faster inference
- Ensure no other GPU/CPU-heavy processes are running alongside the application
- On low-end machines, consider disabling `refine_landmarks=True` — this disables iris tracking but significantly improves speed

---

## 🛠️ Built With

- [OpenCV](https://opencv.org/) — Real-time computer vision and webcam handling
- [MediaPipe](https://mediapipe.dev/) — Face mesh and iris landmark detection
- [NumPy](https://numpy.org/) — Numerical operations and coordinate math

---

## 🤝 Support

- **Issues:** Open a ticket on [GitHub Issues](../../issues)
- **Docs:** Refer to the inline comments in `main.py` for implementation details

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
