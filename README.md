[README.md](https://github.com/user-attachments/files/33766752/README.md)
# Neural ML Face Recognition & Attendance System

A standalone desktop application that detects and recognizes faces in real time using a webcam, trains on new people from webcam captures or photo files, and automatically logs attendance to a CSV file.

It is a single Python file with a dark-themed GUI (built with CustomTkinter). All learned identities are saved in one model file, and no face photos are stored on disk.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [How It Works](#how-it-works)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [How to Use](#how-to-use)
6. [Project Files and Folders](#project-files-and-folders)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Limitations](#limitations)
10. [Privacy and Responsible Use](#privacy-and-responsible-use)

---

## What It Does

| Feature | Description |
|---|---|
| **Live recognition** | Opens the webcam and recognizes multiple faces at once, drawing a box and a name badge with a confidence percentage. |
| **Webcam training** | Captures 24 face samples from the camera and teaches the model who the person is. |
| **Photo training** | Select one or more photos (JPG/PNG/BMP/WEBP) from your computer to train a person without using the webcam. |
| **Automatic attendance** | When a known person is recognized, a "Present" entry is saved to a CSV file (once per 5 seconds per person). |
| **Attendance history** | A built-in viewer shows the 50 most recent attendance records. |
| **Incremental learning** | New people are added without erasing people trained earlier. |

---

## How It Works

1. **Face detection**: OpenCV Haar Cascades (frontal, alternate frontal, and profile, including a mirrored profile check) find faces in each frame. The image is first improved with CLAHE contrast enhancement.
2. **Preprocessing**: Each face is cropped, lightly blurred, and resized to 96×96 grayscale pixels.
3. **Feature extraction**: A HOG (Histogram of Oriented Gradients) descriptor turns the face into a numeric vector.
4. **Data augmentation (during training)**: Each captured sample is expanded with a horizontally flipped copy and two tilted copies (−8° and +8°).
5. **Classification**: A scikit-learn pipeline (`StandardScaler` + `SVC` with an RBF kernel and probability output) is trained on all stored people.
6. **Recognition**: The model gives a probability per person. If the best match is below **50%** confidence (or is the background class), the face is labeled **Unknown**.
7. **Saving**: The model, class names, and feature vectors are saved with `joblib` into `neural_biometric_model.pkl`.

> When only one person has been trained, the app adds synthetic random-noise samples as an "Unknown_Background" class so the classifier has two classes to learn from.

---

## Requirements

- Python 3.8 or newer
- A working webcam (for live recognition and webcam training)
- A desktop environment (Windows, macOS, or Linux with Tkinter)

Python libraries:

- `opencv-python`
- `numpy`
- `pillow`
- `scikit-learn`
- `joblib`
- `customtkinter`

---

## Installation

1. Put the script in a folder of your choice (for example `face_app/`).
2. (Recommended) Create a virtual environment:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install scikit-learn joblib opencv-python customtkinter pillow numpy
   ```

---

## How to Use

### 1. Start the app

```bash
python your_script_name.py
```

Replace `your_script_name.py` with the actual file name. On first launch the app creates a `face_ai_system/` folder next to the script.

The sidebar has four sections, and a status indicator at the bottom shows whether a model is trained:

- 🟢 **ML Model Online (.pkl)**: at least one person is trained
- 🟠 **Model Needs Training**: nobody has been trained yet

### 2. Train a person (do this first)

You can train in either of two ways.

**Option A: Webcam Face Training** (👤 sidebar)

1. Open **Webcam Face Training**.
2. Type the person's full name.
3. Click **Capture Face & Train AI Model**.
4. Look at the camera and **slowly turn your head** left and right while the progress bar fills (24 samples).
5. Wait for the green "Face permanently trained" message.

**Option B: Direct Image Training** (📁 sidebar)

1. Open **Direct Image Training**.
2. Type the person's full name.
3. Click **Select Image Files & Train Model** and choose several photos of that person.
4. Wait for the success message.

Tips for better accuracy:

- Use 5–15 clear photos with different angles, expressions, and lighting.
- Each photo should contain the person's face, ideally as the largest face in the picture (the app uses the biggest detected face).
- Avoid heavy shadows, sunglasses, or very small faces.
- Train at least **two different people** for the best classifier behavior.

### 3. Run live recognition

1. Open **Full Screen Live Camera**.
2. Click **▶ Turn Camera On**.
3. Faces appear with a box and badge:
   - 🟢 Green: `✓ NAME (87%)`: recognized and attendance logged
   - 🔴 Red: `? UNKNOWN (32%)`: not recognized
4. Click **⏹ Turn Camera Off** when finished.
5. **Clear Attendance Cooldown** lets someone be logged again immediately, instead of waiting 5 seconds.

The camera is automatically turned off when you switch to another tab.

### 4. View attendance

Open **Attendance History** and click **Refresh Logs**. The latest 50 entries are shown, newest first.

The full log is also in `face_ai_system/attendance_records.csv` and can be opened in Excel or Google Sheets.

| Column | Meaning |
|---|---|
| Timestamp | Date and time of recognition |
| Person Name | Recognized person |
| Status | Always `Present` |
| ML Confidence | Model confidence percentage |
| Model Type | Label of the model used |

---

## Project Files and Folders

```
your_folder/
├── your_script_name.py              # the application
└── face_ai_system/                  # created automatically
    ├── neural_biometric_model.pkl   # trained model + names + feature vectors
    └── attendance_records.csv       # attendance log
```

**Resetting or removing people:** the app has no delete button. To remove everyone and start over, close the app and delete `face_ai_system/neural_biometric_model.pkl`. To clear attendance history, delete `attendance_records.csv` (it is recreated on the next launch).

**Backing up or moving the model:** copy the `.pkl` file to the same `face_ai_system/` folder on another machine running the same library versions.

**Retraining a person:** training with an existing name **replaces** that person's earlier samples with the new ones.

---

## Configuration

Settings are constants near the top of the script:

| Constant | Default | Purpose |
|---|---|---|
| `FACE_SIZE` | `(96, 96)` | Size faces are resized to before feature extraction |
| `MIN_CONFIDENCE_THRESHOLD` | `0.50` | Minimum probability to accept a match |
| `SAMPLES_PER_CAPTURE` | `24` | Number of samples collected in webcam training |

Increase `MIN_CONFIDENCE_THRESHOLD` (e.g. `0.70`) to reduce false matches, or lower it if known people are often shown as Unknown. Note that the live view has an additional check of `conf >= 50.0` in `_camera_stream_loop`, so change that too if you change the threshold.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Required ML/CV library missing` | Run the `pip install` command from the Installation section. |
| `Webcam not detected` | Check that the camera is connected, not in use by another app (Zoom, Teams), and that the OS has granted camera permission. If you have several cameras, change `VideoCapture(0, ...)` to `1` in the script. |
| `No faces could be detected in the chosen photos` | Use clearer, front-facing, well-lit photos with a larger face. |
| Everyone shows as Unknown | Train with more varied samples, make sure lighting is similar to training conditions, or lower the confidence threshold. |
| Wrong person recognized | Train more people and more samples, or raise the confidence threshold. |
| Model fails to load after upgrading libraries | Delete the `.pkl` file and retrain (pickled models can break between scikit-learn versions). |
| Laggy video | Close other apps; detection runs on every frame on the CPU. |
| Window closes on error | The launcher keeps the console open and prints the traceback. Read it, then press Enter. |
| Linux: `No module named tkinter` | Install it: `sudo apt install python3-tk` |

---

## Limitations

- **Not secure authentication.** This uses classic computer vision (Haar cascades + HOG + SVM), which is far less accurate than modern deep-learning face recognition. It can be fooled by photos on a phone or printed pictures (no liveness detection), and similar-looking people may be confused.
- Works best with frontal faces and consistent lighting.
- Only the largest face in each training photo or webcam frame is learned.
- Use for learning projects, demos, and low-stakes attendance, not for security-critical decisions such as access control or payments.
- Attendance only logs "Present" and does not record departures.
- Names are matched exactly, so `Jemish` and `jemish` count as two different people.

---

## Privacy and Responsible Use

- Faces are biometric data. **Only train and recognize people who have given you clear consent**, and follow the privacy laws that apply where you live (such as GDPR or local biometric regulations).
- Although no photos are saved, the `.pkl` file still contains **names and numeric feature vectors derived from faces**, and the attendance CSV contains names with timestamps. Keep both files protected and do not share them casually.
- Tell people when and why recognition is running.

---

## Quick Start Summary

```bash
pip install scikit-learn joblib opencv-python customtkinter pillow numpy
python your_script_name.py
```

1. Train people (Webcam or Image Training)
2. Open Live Camera and click **Turn Camera On**
3. Check **Attendance History** or the CSV file
