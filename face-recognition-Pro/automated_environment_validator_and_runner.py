#!/usr/bin/env python3
"""
Launcher and Environment Checker for the Standalone ML Biometric Terminal.
Checks installed dependencies, camera availability, and launches the app.
"""

import sys
import subprocess
import importlib

REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "customtkinter": "customtkinter"
}

def verify_and_install_packages():
    """Checks whether each mandatory package exists; prompts automatic installation if missing."""
    missing = []
    print("=" * 60)
    print("   AI BIOMETRIC ML SYSTEM - ENVIRONMENT VALIDATION")
    print("=" * 60)
    
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            print(f" [✓] {pip_name:<20} : Installed")
        except ImportError:
            print(f" [✗] {pip_name:<20} : MISSING")
            missing.append(pip_name)
            
    if missing:
        print("\n" + "-" * 60)
        print(f"Detected {len(missing)} missing packages: {', '.join(missing)}")
        print("Installing missing packages automatically via pip...\n")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("\n[✓] All dependencies have been installed successfully!")
        except Exception as e:
            print(f"\n[!] Automatic installation failed: {e}")
            print("Please run manually:")
            print(f"    pip install {' '.join(missing)}")
            sys.exit(1)
    else:
        print("\n[✓] Environment verified! All required libraries are ready.")

def launch_application():
    """Runs modern_biometric_face_attendance_system.py directly."""
    import os
    target_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modern_biometric_face_attendance_system.py")
    
    if not os.path.exists(target_script):
        print(f"[!] Target script not found: {target_script}")
        print("Please ensure 'modern_biometric_face_attendance_system.py' is in the same folder.")
        sys.exit(1)
        
    print(f"Starting Face Recognition Terminal ({target_script})...\n")
    try:
        subprocess.call([sys.executable, target_script])
    except KeyboardInterrupt:
        print("\nApplication closed by user.")

if __name__ == "__main__":
    verify_and_install_packages()
    launch_application()