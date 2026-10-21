import os
import sys
import csv
import time
import traceback
import threading
from datetime import datetime

# Dependency Verification with user-friendly error pause
try:
    import cv2
    import numpy as np
    from PIL import Image
    import joblib
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except ImportError as e:
    print("\n" + "=" * 60)
    print(f"[ERROR] Required ML/CV library missing: {e}")
    print("Please install required libraries by running:")
    print("    pip install scikit-learn joblib opencv-python customtkinter pillow numpy")
    print("=" * 60 + "\n")
    input("Press Enter to exit...")
    sys.exit(1)

try:
    import customtkinter as ctk
    from tkinter import filedialog
except ImportError as e:
    print("\n" + "=" * 60)
    print(f"[ERROR] GUI library missing: {e}")
    print("Please run: pip install customtkinter pillow")
    print("=" * 60 + "\n")
    input("Press Enter to exit...")
    sys.exit(1)

# Application Directories & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "face_ai_system")
MODEL_PKL_PATH = os.path.join(DATA_DIR, "neural_biometric_model.pkl")
ATTENDANCE_LOG = os.path.join(DATA_DIR, "attendance_records.csv")

FACE_SIZE = (96, 96)
MIN_CONFIDENCE_THRESHOLD = 0.50
SAMPLES_PER_CAPTURE = 24


def initialize_storage():
    """Initializes local storage directory and attendance log."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ATTENDANCE_LOG):
        with open(ATTENDANCE_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Person Name", "Status", "ML Confidence", "Model Type"])


class MachineLearningFaceEngine:
    """
    Stealth Supervised Machine Learning Biometric Classifier.
    - All mathematical hyperplane boundaries and names are baked permanently into neural_biometric_model.pkl.
    - Permanent memory: no deletion allowed; once trained, identity is retained permanently.
    - Zero human-readable roster or stored face photos exist on disk.
    """
    def __init__(self):
        self.cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        self.cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

        self.clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))

        # Standard Positional Parameters for HOGDescriptor across all OpenCV versions
        self.hog = cv2.HOGDescriptor((96, 96), (16, 16), (8, 8), (8, 8), 9)

        self.ml_model = None
        self.classes_ = []
        self.memory_store = {}
        self.is_trained = False
        self.load_model()

    def load_model(self):
        """Loads compiled model weights from disk."""
        if os.path.exists(MODEL_PKL_PATH):
            try:
                bundle = joblib.load(MODEL_PKL_PATH)
                if isinstance(bundle, dict) and "pipeline" in bundle:
                    self.ml_model = bundle["pipeline"]
                    self.memory_store = bundle.get("memory_store", {})
                    self.classes_ = [c for c in bundle.get("classes", []) if c != "Unknown_Background"]
                    self.is_trained = len(self.classes_) > 0
                    return True
            except Exception as e:
                print(f"[!] Warning while reading model: {e}")
                self.ml_model = None
                self.classes_ = []
                self.memory_store = {}
                self.is_trained = False
        else:
            self.ml_model = None
            self.classes_ = []
            self.memory_store = {}
            self.is_trained = False
        return False

    def extract_hog_features(self, gray_face_96x96):
        """Extracts mathematical vector representation of facial structure."""
        if gray_face_96x96 is None or gray_face_96x96.shape != FACE_SIZE:
            return None
        try:
            features = self.hog.compute(gray_face_96x96)
            if features is not None:
                return features.flatten()
        except Exception:
            return None
        return None

    def detect_faces(self, bgr_frame):
        """
        Detects frontal, tilted, and profile angles using dual frontal and profile cascades.
        """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)

        faces = list(self.cascade_frontal.detectMultiScale(
            enhanced, scaleFactor=1.10, minNeighbors=4, minSize=(45, 45)
        ))

        if len(faces) == 0 and not self.cascade_alt.empty():
            faces = list(self.cascade_alt.detectMultiScale(
                enhanced, scaleFactor=1.12, minNeighbors=3, minSize=(45, 45)
            ))

        if len(faces) == 0 and not self.cascade_profile.empty():
            profiles_left = self.cascade_profile.detectMultiScale(
                enhanced, scaleFactor=1.12, minNeighbors=4, minSize=(45, 45)
            )
            faces.extend(list(profiles_left))

        if len(faces) == 0 and not self.cascade_profile.empty():
            flipped = cv2.flip(enhanced, 1)
            profiles_right = self.cascade_profile.detectMultiScale(
                flipped, scaleFactor=1.12, minNeighbors=4, minSize=(45, 45)
            )
            img_w = enhanced.shape[1]
            for (rx, ry, rw, rh) in profiles_right:
                faces.append((img_w - rx - rw, ry, rw, rh))

        bounding_boxes = []
        for (x, y, w, h) in faces:
            is_dup = False
            for (bx, by, bw, bh) in bounding_boxes:
                if abs(x - bx) < 30 and abs(y - by) < 30:
                    is_dup = True
                    break
            if not is_dup:
                bounding_boxes.append((int(x), int(y), int(w), int(h)))

        return bounding_boxes, enhanced

    def crop_and_normalize(self, enhanced_gray, box):
        """Crops, smoothens, and resizes face to canonical dimensions."""
        x, y, w, h = box
        img_h, img_w = enhanced_gray.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)

        crop = enhanced_gray[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        smoothed = cv2.GaussianBlur(crop, (3, 3), 0)
        return cv2.resize(smoothed, FACE_SIZE)

    def train_or_update_model(self, person_name: str, face_matrices: list):
        """
        Permanently bakes facial vectors into neural_biometric_model.pkl.
        Prior identities are never erased; new identities are appended and baked permanently.
        """
        person_name = person_name.strip()
        if not person_name:
            return False, "Person name is required."

        if not face_matrices or len(face_matrices) == 0:
            return False, "No valid face samples collected for training."

        augmented_features = []
        for mat in face_matrices:
            # 1. Base face
            f_base = self.extract_hog_features(mat)
            if f_base is not None:
                augmented_features.append(f_base)

            # 2. Horizontal Flip (Left/Right angle support)
            flipped = cv2.flip(mat, 1)
            f_flip = self.extract_hog_features(flipped)
            if f_flip is not None:
                augmented_features.append(f_flip)

            # 3. Head Tilt Augmentation (-8 deg and +8 deg)
            h, w = mat.shape[:2]
            for angle in [-8, 8]:
                rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                rotated = cv2.warpAffine(mat, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
                f_rot = self.extract_hog_features(rotated)
                if f_rot is not None:
                    augmented_features.append(f_rot)

        if len(augmented_features) == 0:
            return False, "Failed to compute HOG feature descriptors."

        # Add or update person in permanent memory store
        self.memory_store[person_name] = augmented_features

        all_X = []
        all_y = []
        for name, feats in self.memory_store.items():
            for f in feats:
                all_X.append(f)
                all_y.append(name)

        X_array = np.array(all_X, dtype=np.float32)
        y_array = np.array(all_y)

        unique_classes = np.unique(y_array)
        if len(unique_classes) < 2:
            synthetic_noise = np.random.normal(0.0, 0.08, size=(len(augmented_features), X_array.shape[1])).astype(np.float32)
            X_train = np.vstack([X_array, synthetic_noise])
            y_train = np.concatenate([y_array, np.array(["Unknown_Background"] * len(augmented_features))])
        else:
            X_train = X_array
            y_train = y_array

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, class_weight="balanced"))
        ])

        try:
            pipeline.fit(X_train, y_train)

            model_bundle = {
                "pipeline": pipeline,
                "memory_store": self.memory_store,
                "classes": [c for c in unique_classes if c != "Unknown_Background"]
            }
            # Compress and bake directly to .pkl
            joblib.dump(model_bundle, MODEL_PKL_PATH, compress=3)
            self.load_model()
        except Exception as e:
            return False, f"Machine Learning training error: {str(e)}"

        return True, "Success! Identity permanently baked into AI Model Weights."

    def predict(self, normalized_face_96x96):
        """
        Runs ML inference on a face crop.
        Returns predicted person name and probability percentage.
        """
        if not self.is_trained or self.ml_model is None or normalized_face_96x96 is None:
            return "Unknown", 0.0

        feat = self.extract_hog_features(normalized_face_96x96)
        if feat is None:
            return "Unknown", 0.0

        try:
            X_test = np.array([feat], dtype=np.float32)
            probabilities = self.ml_model.predict_proba(X_test)[0]

            if hasattr(self.ml_model, "classes_"):
                classes = list(self.ml_model.classes_)
            else:
                classes = list(self.ml_model.named_steps["classifier"].classes_)

            best_idx = np.argmax(probabilities)
            best_prob = probabilities[best_idx]
            best_label = classes[best_idx]

            if best_label == "Unknown_Background" or best_prob < MIN_CONFIDENCE_THRESHOLD:
                conf = round(best_prob * 100.0, 1)
                return "Unknown", conf
            else:
                conf = round(best_prob * 100.0, 1)
                return best_label, conf
        except Exception:
            return "Unknown", 0.0


class StandaloneBiometricApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        initialize_storage()

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.title("AI Machine Learning Face Recognition System (Stealth Engine)")
        self.geometry("1260x760")
        self.minsize(1050, 680)

        self.engine = MachineLearningFaceEngine()

        self.cap = None
        self.camera_active = False
        self.is_capturing = False
        self.captured_matrices = []
        self.last_attendance_times = {}
        self.scan_line_y = 0

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._build_sidebar()
        self._build_views()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.open_view("live_camera")

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color="#070b14")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)

        brand_box = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_box.grid(row=0, column=0, padx=16, pady=(24, 18), sticky="ew")

        ctk.CTkLabel(
            brand_box, text="⚡ NEURAL ML FACE AI",
            font=ctk.CTkFont(family="Helvetica", size=17, weight="bold"),
            text_color="#38bdf8"
        ).pack(anchor="w")

        ctk.CTkLabel(
            brand_box, text="Stealth Single .pkl System",
            font=ctk.CTkFont(size=11), text_color="#64748b"
        ).pack(anchor="w", pady=(2, 0))

        self.nav_buttons = {}
        nav_items = [
            ("live_camera",  "📷  Full Screen Live Camera"),
            ("webcam_train", "👤  Webcam Face Training"),
            ("file_train",   "📁  Direct Image Training"),
            ("logs",         "📜  Attendance History"),
        ]

        for idx, (view_key, title) in enumerate(nav_items, start=1):
            btn = ctk.CTkButton(
                self.sidebar, text=title, height=44,
                font=ctk.CTkFont(size=13, weight="bold"),
                anchor="w", corner_radius=10,
                fg_color="#0f172a", hover_color="#1e293b",
                command=lambda k=view_key: self.open_view(k)
            )
            btn.grid(row=idx, column=0, padx=12, pady=5, sticky="ew")
            self.nav_buttons[view_key] = btn

        self.status_card = ctk.CTkFrame(self.sidebar, corner_radius=10, fg_color="#0b1120")
        self.status_card.grid(row=6, column=0, padx=12, pady=16, sticky="ew")

        self.status_dot = ctk.CTkLabel(
            self.status_card, text="●", font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#10b981" if self.engine.is_trained else "#f59e0b"
        )
        self.status_dot.pack(side="left", padx=(10, 4), pady=8)

        status_text = "ML Model Online (.pkl)" if self.engine.is_trained else "Model Needs Training"
        self.lbl_model_status = ctk.CTkLabel(
            self.status_card, text=status_text,
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#94a3b8"
        )
        self.lbl_model_status.pack(side="left", padx=4, pady=8)

    def _build_views(self):
        # 1. LIVE FULL-SCREEN CAMERA
        self.view_live_camera = ctk.CTkFrame(self, fg_color="#030712", corner_radius=0)
        self.view_live_camera.grid_columnconfigure(0, weight=1)
        self.view_live_camera.grid_rowconfigure(1, weight=1)

        self.stream_header = ctk.CTkFrame(self.view_live_camera, height=46, corner_radius=12, fg_color="#0b1220")
        self.stream_header.grid(row=0, column=0, padx=14, pady=(12, 6), sticky="ew")

        self.lbl_camera_indicator = ctk.CTkLabel(
            self.stream_header, text="● CAMERA OFF", font=ctk.CTkFont(size=12, weight="bold"), text_color="#ef4444"
        )
        self.lbl_camera_indicator.pack(side="left", padx=16)

        self.lbl_faces_in_view = ctk.CTkLabel(
            self.stream_header, text="Faces Detected: 0  |  Multi-Angle 3D Tracking",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#38bdf8"
        )
        self.lbl_faces_in_view.pack(side="left", padx=20)

        self.lbl_model_badge = ctk.CTkLabel(
            self.stream_header, text="⚡ Permanent Binary: neural_biometric_model.pkl (Zero Photo Footprint)",
            font=ctk.CTkFont(size=11), text_color="#10b981"
        )
        self.lbl_model_badge.pack(side="right", padx=16)

        self.lbl_full_camera_feed = ctk.CTkLabel(
            self.view_live_camera,
            text="Camera is turned off.\nClick 'Turn Camera On' button below to start real-time multi-face ML detection.",
            corner_radius=14, fg_color="#02040a", font=ctk.CTkFont(size=16), text_color="#64748b"
        )
        self.lbl_full_camera_feed.grid(row=1, column=0, padx=14, pady=6, sticky="nsew")

        self.stream_control_bar = ctk.CTkFrame(self.view_live_camera, height=56, corner_radius=12, fg_color="#0b1220")
        self.stream_control_bar.grid(row=2, column=0, padx=14, pady=(6, 14), sticky="ew")

        self.btn_camera_toggle = ctk.CTkButton(
            self.stream_control_bar, text="▶  Turn Camera On", width=200, height=42,
            font=ctk.CTkFont(size=13, weight="bold"), fg_color="#10b981", hover_color="#059669",
            corner_radius=10, command=self.toggle_camera
        )
        self.btn_camera_toggle.pack(side="left", padx=14, pady=7)

        self.lbl_hud_info = ctk.CTkLabel(
            self.stream_control_bar,
            text="Identifies authorized individuals simultaneously from baked neural model weights.",
            font=ctk.CTkFont(size=12), text_color="#94a3b8"
        )
        self.lbl_hud_info.pack(side="left", padx=16)

        self.btn_clear_cooldown = ctk.CTkButton(
            self.stream_control_bar, text="🔄 Clear Attendance Cooldown", width=210, height=38,
            font=ctk.CTkFont(size=12, weight="bold"), fg_color="#1e293b", hover_color="#334155",
            corner_radius=8, command=lambda: self.last_attendance_times.clear()
        )
        self.btn_clear_cooldown.pack(side="right", padx=14, pady=7)

        # 2. WEBCAM FACE TRAINING
        self.view_webcam_train = ctk.CTkFrame(self, fg_color="#030712", corner_radius=0)
        self.view_webcam_train.grid_columnconfigure(0, weight=1)
        self.view_webcam_train.grid_columnconfigure(1, weight=1)
        self.view_webcam_train.grid_rowconfigure(0, weight=1)

        self.train_left_card = ctk.CTkFrame(self.view_webcam_train, corner_radius=14, fg_color="#0b1220", border_width=1, border_color="#1e293b")
        self.train_left_card.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")

        ctk.CTkLabel(
            self.train_left_card, text="Step 1: Enter Person Details",
            font=ctk.CTkFont(size=17, weight="bold"), text_color="#38bdf8"
        ).pack(anchor="w", padx=24, pady=(24, 10))

        ctk.CTkLabel(self.train_left_card, text="Full Name (ML Target Label):", font=ctk.CTkFont(size=12, weight="bold"), text_color="#cbd5e1").pack(anchor="w", padx=24, pady=(10, 2))
        self.entry_webcam_name = ctk.CTkEntry(self.train_left_card, placeholder_text="e.g. Jemish", height=40, fg_color="#111c30")
        self.entry_webcam_name.pack(fill="x", padx=24, pady=(0, 14))

        info_box = ctk.CTkFrame(self.train_left_card, corner_radius=10, fg_color="#080e1a")
        info_box.pack(fill="x", padx=24, pady=(10, 20))

        ctk.CTkLabel(
            info_box,
            text="Stealth Neural Model Information:\n• Facial features are compiled directly into 'neural_biometric_model.pkl'.\n• Learned data remains permanent forever—it cannot be lost or unlearned.\n• No public roster or profile lists exist; identities stay confidential.\n• Zero photos are stored on the hard drive!",
            font=ctk.CTkFont(size=12), text_color="#94a3b8", justify="left"
        ).pack(padx=16, pady=14, anchor="w")

        self.lbl_webcam_train_msg = ctk.CTkLabel(self.train_left_card, text="", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_webcam_train_msg.pack(padx=24, pady=(0, 10))

        self.train_right_card = ctk.CTkFrame(self.view_webcam_train, corner_radius=14, fg_color="#0b1220", border_width=1, border_color="#1e293b")
        self.train_right_card.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="nsew")
        self.train_right_card.grid_columnconfigure(0, weight=1)
        self.train_right_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self.train_right_card, text="Step 2: Collect Face Samples", font=ctk.CTkFont(size=17, weight="bold"), text_color="#38bdf8").grid(row=0, column=0, padx=16, pady=(20, 6))

        self.lbl_train_preview = ctk.CTkLabel(
            self.train_right_card,
            text="Click 'Capture Face & Train' button below\nto bake face permanently into the AI model.",
            corner_radius=12, fg_color="#02040a", text_color="#64748b"
        )
        self.lbl_train_preview.grid(row=1, column=0, padx=16, pady=8, sticky="nsew")

        self.bar_webcam_progress = ctk.CTkProgressBar(self.train_right_card, height=10, corner_radius=5)
        self.bar_webcam_progress.set(0.0)
        self.bar_webcam_progress.grid(row=2, column=0, padx=24, pady=6, sticky="ew")

        self.lbl_samples_count = ctk.CTkLabel(self.train_right_card, text=f"Samples: 0 / {SAMPLES_PER_CAPTURE}", font=ctk.CTkFont(size=12))
        self.lbl_samples_count.grid(row=3, column=0, pady=(2, 6))

        self.btn_capture_and_train = ctk.CTkButton(
            self.train_right_card, text="📸  Capture Face & Train AI Model",
            height=46, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#10b981", hover_color="#059669", corner_radius=10,
            command=self.start_webcam_face_capture
        )
        self.btn_capture_and_train.grid(row=4, column=0, padx=24, pady=(4, 20), sticky="ew")

        # 3. DIRECT FILE TRAINING
        self.view_file_train = ctk.CTkFrame(self, fg_color="#030712", corner_radius=0)
        self.view_file_train.grid_columnconfigure(0, weight=1)
        self.view_file_train.grid_rowconfigure(0, weight=1)

        self.file_card = ctk.CTkFrame(self.view_file_train, corner_radius=16, fg_color="#0b1220", border_width=1, border_color="#1e293b", width=620, height=440)
        self.file_card.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(self.file_card, text="📁", font=ctk.CTkFont(size=44)).pack(pady=(26, 4))
        ctk.CTkLabel(self.file_card, text="Direct Image / File Training", font=ctk.CTkFont(size=19, weight="bold"), text_color="#38bdf8").pack(pady=4)

        ctk.CTkLabel(
            self.file_card,
            text="Select photos directly from your computer (JPG / PNG).\nThe identity is permanently baked into the binary model file without leaving images on disk.",
            font=ctk.CTkFont(size=12), text_color="#94a3b8", justify="center"
        ).pack(padx=30, pady=(4, 18))

        ctk.CTkLabel(self.file_card, text="Person Full Name:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#cbd5e1").pack(anchor="w", padx=50, pady=(4, 2))
        self.entry_file_name = ctk.CTkEntry(self.file_card, width=520, height=40, placeholder_text="Enter Full Name", fg_color="#111c30")
        self.entry_file_name.pack(padx=50, pady=(0, 14))

        self.lbl_file_status = ctk.CTkLabel(self.file_card, text="", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_file_status.pack(pady=(0, 10))

        self.btn_choose_files = ctk.CTkButton(
            self.file_card, text="📂  Select Image Files & Train Model",
            width=520, height=44, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2563eb", hover_color="#1d4ed8", corner_radius=10,
            command=self.handle_file_training
        )
        self.btn_choose_files.pack(padx=50, pady=(0, 26))

        # 4. ATTENDANCE LOGS
        self.view_logs = ctk.CTkFrame(self, fg_color="#030712", corner_radius=0)
        self.view_logs.grid_columnconfigure(0, weight=1)
        self.view_logs.grid_rowconfigure(1, weight=1)

        self.logs_top_bar = ctk.CTkFrame(self.view_logs, height=54, fg_color="#0b1220", corner_radius=12)
        self.logs_top_bar.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")

        ctk.CTkLabel(
            self.logs_top_bar, text="📜 Real-Time Attendance History",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#38bdf8"
        ).pack(side="left", padx=16)

        ctk.CTkButton(
            self.logs_top_bar, text="🔄 Refresh Logs", width=120, height=36,
            corner_radius=8, command=self.load_attendance_history
        ).pack(side="right", padx=12)

        self.logs_scroll_frame = ctk.CTkScrollableFrame(self.view_logs, corner_radius=14, fg_color="#070c18")
        self.logs_scroll_frame.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")

    def open_view(self, view_name):
        self.view_live_camera.grid_forget()
        self.view_webcam_train.grid_forget()
        self.view_file_train.grid_forget()
        self.view_logs.grid_forget()

        for btn in self.nav_buttons.values():
            btn.configure(fg_color="#0f172a")

        if view_name in self.nav_buttons:
            self.nav_buttons[view_name].configure(fg_color="#1e293b")

        if view_name == "live_camera":
            self.view_live_camera.grid(row=0, column=1, sticky="nsew")
        elif view_name == "webcam_train":
            self.stop_camera()
            self.view_webcam_train.grid(row=0, column=1, sticky="nsew")
        elif view_name == "file_train":
            self.stop_camera()
            self.view_file_train.grid(row=0, column=1, sticky="nsew")
        elif view_name == "logs":
            self.stop_camera()
            self.view_logs.grid(row=0, column=1, sticky="nsew")
            self.load_attendance_history()

    def toggle_camera(self):
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.camera_active:
            return

        self.btn_camera_toggle.configure(state="disabled", text="Starting...")
        self.lbl_full_camera_feed.configure(text="Connecting to camera stream...")

        def init_thread():
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(0, backend)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                def on_success():
                    self.cap = cap
                    self.camera_active = True
                    self.btn_camera_toggle.configure(
                        state="normal", text="⏹  Turn Camera Off",
                        fg_color="#dc2626", hover_color="#991b1b"
                    )
                    self.lbl_camera_indicator.configure(text="● LIVE HD STREAM", text_color="#10b981")
                    self._camera_stream_loop()
                self.after(0, on_success)
            else:
                def on_failure():
                    self.btn_camera_toggle.configure(state="normal", text="▶  Turn Camera On", fg_color="#10b981", hover_color="#059669")
                    self.lbl_camera_indicator.configure(text="● CAMERA ERROR", text_color="#ef4444")
                    self.lbl_full_camera_feed.configure(text="Webcam not detected! Please ensure camera is connected.")
                self.after(0, on_failure)

        threading.Thread(target=init_thread, daemon=True).start()

    def stop_camera(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_camera_toggle.configure(text="▶  Turn Camera On", fg_color="#10b981", hover_color="#059669")
        self.lbl_camera_indicator.configure(text="● CAMERA OFF", text_color="#ef4444")
        self.lbl_faces_in_view.configure(text="Faces Detected: 0  |  Standby")
        self.lbl_full_camera_feed.configure(
            image="",
            text="Camera is turned off.\nClick 'Turn Camera On' button below to start real-time multi-face ML detection."
        )

    def _camera_stream_loop(self):
        if not self.camera_active or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(20, self._camera_stream_loop)
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display_frame = frame.copy()

        self.scan_line_y = (self.scan_line_y + 8) % max(1, h)
        cv2.line(display_frame, (0, self.scan_line_y), (w, self.scan_line_y), (56, 189, 248), 1)

        detected_boxes, enhanced_gray = self.engine.detect_faces(frame)
        self.lbl_faces_in_view.configure(text=f"Faces Detected: {len(detected_boxes)}  |  Multi-Face Tracking Active")

        for (x, y, fw, fh) in detected_boxes:
            norm_face = self.engine.crop_and_normalize(enhanced_gray, (x, y, fw, fh))
            pred_name, conf = self.engine.predict(norm_face)

            if pred_name != "Unknown" and conf >= 50.0:
                color = (0, 230, 115)
                badge_bg = (10, 70, 30)
                badge_label = f"✓ {pred_name.upper()} ({conf:.0f}%)"
                self._record_attendance(pred_name, conf)
            else:
                color = (50, 50, 240)
                badge_bg = (30, 15, 90)
                badge_label = f"? UNKNOWN ({conf:.0f}%)"

            self._draw_hud_corners(display_frame, x, y, fw, fh, color)
            self._draw_name_badge(display_frame, badge_label, x, y, fw, fh, color, badge_bg)

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        vw = max(640, self.lbl_full_camera_feed.winfo_width())
        vh = max(420, self.lbl_full_camera_feed.winfo_height())
        resized_img = pil_img.resize((vw, vh), Image.Resampling.BILINEAR)

        tk_img = ctk.CTkImage(light_image=resized_img, dark_image=resized_img, size=(vw, vh))
        self.lbl_full_camera_feed.configure(image=tk_img, text="")
        self.lbl_full_camera_feed.image = tk_img

        self.after(16, self._camera_stream_loop)

    def _draw_hud_corners(self, img, x, y, w, h, color):
        line_len = int(min(w, h) * 0.22)
        thickness = 2
        # Top-Left Corner
        cv2.line(img, (x, y), (x + line_len, y), color, thickness)
        cv2.line(img, (x, y), (x, y + line_len), color, thickness)
        # Top-Right Corner
        cv2.line(img, (x + w, y), (x + w - line_len, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + line_len), color, thickness)
        # Bottom-Left Corner
        cv2.line(img, (x, y + h), (x + line_len, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - line_len), color, thickness)
        # Bottom-Right Corner (Fixed clean syntax)
        cv2.line(img, (x + w, y + h), (x + w - line_len, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - line_len), color, thickness)

    def _draw_name_badge(self, img, label, x, y, w, h, border_col, bg_col):
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        chip_y1 = max(10, y - th - 16)
        chip_y2 = chip_y1 + th + 12
        chip_x1 = max(10, x + (w // 2) - (tw // 2) - 10)
        chip_x2 = chip_x1 + tw + 20

        cv2.rectangle(img, (chip_x1, chip_y1), (chip_x2, chip_y2), bg_col, -1)
        cv2.rectangle(img, (chip_x1, chip_y1), (chip_x2, chip_y2), border_col, 1)
        cv2.putText(img, label, (chip_x1 + 10, chip_y2 - 6), font, font_scale, (255, 255, 255), thickness)

    def _record_attendance(self, name, conf):
        now = time.time()
        if name in self.last_attendance_times and (now - self.last_attendance_times[name]) < 5.0:
            return

        self.last_attendance_times[name] = now
        timestamp = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

        try:
            with open(ATTENDANCE_LOG, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, name, "Present", f"{conf:.0f}%", "Permanent Neural Model"])
        except Exception:
            pass

    def start_webcam_face_capture(self):
        name = self.entry_webcam_name.get().strip()
        if not name:
            self.lbl_webcam_train_msg.configure(text="Please enter Person Full Name first!", text_color="#ef4444")
            return

        self.lbl_webcam_train_msg.configure(text="Starting camera for face capture...", text_color="#38bdf8")
        self.btn_capture_and_train.configure(state="disabled", text="Capturing Faces... Turn head slowly")
        self.captured_matrices = []
        self.is_capturing = True

        def capture_thread():
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(0, backend)

            if not cap.isOpened():
                def on_error():
                    self.lbl_webcam_train_msg.configure(text="Could not open camera!", text_color="#ef4444")
                    self.btn_capture_and_train.configure(state="normal", text="📸  Capture Face & Train AI Model")
                self.after(0, on_error)
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            collected = 0
            while collected < SAMPLES_PER_CAPTURE and self.is_capturing:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                boxes, enhanced_gray = self.engine.detect_faces(frame)
                disp = frame.copy()

                if boxes:
                    primary = max(boxes, key=lambda b: b[2] * b[3])
                    x, y, w, h = primary
                    self._draw_hud_corners(disp, x, y, w, h, (0, 230, 115))

                    crop = self.engine.crop_and_normalize(enhanced_gray, primary)
                    if crop is not None:
                        self.captured_matrices.append(crop)
                        collected += 1

                rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(380, 285))

                def update_ui(cnt=collected, img=tk_img):
                    self.lbl_train_preview.configure(image=img, text="")
                    self.lbl_train_preview.image = img
                    self.bar_webcam_progress.set(cnt / float(SAMPLES_PER_CAPTURE))
                    self.lbl_samples_count.configure(text=f"Samples: {cnt} / {SAMPLES_PER_CAPTURE}")
                self.after(0, update_ui)
                time.sleep(0.04)

            cap.release()
            self.is_capturing = False

            if collected >= SAMPLES_PER_CAPTURE:
                self.after(0, lambda: self._finalize_webcam_training(name))
            else:
                def on_cancel():
                    self.btn_capture_and_train.configure(state="normal", text="📸  Capture Face & Train AI Model")
                    self.lbl_webcam_train_msg.configure(text="Capture cancelled or interrupted.", text_color="#ef4444")
                self.after(0, on_cancel)

        threading.Thread(target=capture_thread, daemon=True).start()

    def _finalize_webcam_training(self, name):
        self.lbl_webcam_train_msg.configure(text="Baking permanently into neural_biometric_model.pkl...", text_color="#f59e0b")

        success, msg = self.engine.train_or_update_model(name, self.captured_matrices)

        if success:
            self.status_dot.configure(text_color="#10b981")
            self.lbl_model_status.configure(text="ML Model Online (.pkl)", text_color="#10b981")
            self.lbl_webcam_train_msg.configure(text="✓ Face permanently trained and saved to neural model!", text_color="#10b981")
            self.entry_webcam_name.delete(0, "end")
            self.lbl_train_preview.configure(image="", text="✓ Face permanently baked into AI Model Weights!")
        else:
            self.lbl_webcam_train_msg.configure(text=f"Training Error: {msg}", text_color="#ef4444")

        self.btn_capture_and_train.configure(state="normal", text="📸  Capture Face & Train AI Model")

    def handle_file_training(self):
        name = self.entry_file_name.get().strip()

        if not name:
            self.lbl_file_status.configure(text="Please specify Person Name first!", text_color="#ef4444")
            return

        file_paths = filedialog.askopenfilenames(
            title="Select Face Photos for Training",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )

        if not file_paths:
            return

        self.lbl_file_status.configure(text=f"Extracting face features from {len(file_paths)} photos...", text_color="#38bdf8")

        def file_train_worker():
            extracted_crops = []

            for path in file_paths:
                img = cv2.imread(path)
                if img is None:
                    continue
                boxes, enhanced_gray = self.engine.detect_faces(img)
                if boxes:
                    primary = max(boxes, key=lambda b: b[2] * b[3])
                    crop = self.engine.crop_and_normalize(enhanced_gray, primary)
                    if crop is not None:
                        extracted_crops.append(crop)

            if len(extracted_crops) == 0:
                self.after(0, lambda: self.lbl_file_status.configure(
                    text="No faces could be detected in the chosen photos!", text_color="#ef4444"
                ))
                return

            success, msg = self.engine.train_or_update_model(name, extracted_crops)

            if success:
                def on_done():
                    self.lbl_file_status.configure(
                        text="✓ Identity permanently baked into neural_biometric_model.pkl!",
                        text_color="#10b981"
                    )
                    self.entry_file_name.delete(0, "end")
                    self.status_dot.configure(text_color="#10b981")
                    self.lbl_model_status.configure(text="ML Model Online (.pkl)", text_color="#10b981")
                self.after(0, on_done)
            else:
                self.after(0, lambda: self.lbl_file_status.configure(text=f"Error: {msg}", text_color="#ef4444"))

        threading.Thread(target=file_train_worker, daemon=True).start()

    def load_attendance_history(self):
        for widget in self.logs_scroll_frame.winfo_children():
            widget.destroy()

        if not os.path.exists(ATTENDANCE_LOG):
            ctk.CTkLabel(self.logs_scroll_frame, text="No attendance logs recorded yet.", text_color="#64748b").pack(pady=50)
            return

        with open(ATTENDANCE_LOG, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        if len(rows) <= 1:
            ctk.CTkLabel(self.logs_scroll_frame, text="No attendance logs recorded yet.", text_color="#64748b").pack(pady=50)
            return

        headers = ["Timestamp", "Person Name", "Status", "ML Confidence", "Model Type"]
        header_frame = ctk.CTkFrame(self.logs_scroll_frame, fg_color="#10192a", height=38, corner_radius=8)
        header_frame.pack(fill="x", padx=4, pady=(2, 6))

        for idx, h in enumerate(headers):
            lbl = ctk.CTkLabel(header_frame, text=h, font=ctk.CTkFont(size=12, weight="bold"), width=160, anchor="w", text_color="#38bdf8")
            lbl.grid(row=0, column=idx, padx=6, pady=6)

        for row in reversed(rows[1:][-50:]):
            if not row:
                continue
            r_frame = ctk.CTkFrame(self.logs_scroll_frame, fg_color="#0b1220", corner_radius=6)
            r_frame.pack(fill="x", padx=4, pady=2)

            for idx, val in enumerate(row):
                color = "#10b981" if "Present" in val else "#cbd5e1"
                lbl = ctk.CTkLabel(r_frame, text=val, font=ctk.CTkFont(size=11), width=160, anchor="w", text_color=color)
                lbl.grid(row=0, column=idx, padx=6, pady=6)

    def on_close(self):
        self.stop_camera()
        self.is_capturing = False
        self.destroy()


# CRASH-PROOF LAUNCHER: Kept open on any exception so you can read exact errors
if __name__ == "__main__":
    try:
        app = StandaloneBiometricApp()
        app.mainloop()
    except Exception as e:
        print("\n" + "=" * 65)
        print("[CRITICAL RUNTIME EXCEPTION CAUGHT]")
        print("=" * 65)
        traceback.print_exc()
        print("=" * 65)
        input("\nPress Enter to close this window...")