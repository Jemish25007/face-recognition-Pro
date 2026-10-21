@echo off
title AI Face Biometric System Launcher
color 0b

echo ======================================================================
echo          AI MACHINE LEARNING BIOMETRIC ATTENDANCE TERMINAL
echo ======================================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0c
    echo [ERROR] Python is not installed or not added to your Windows PATH!
    echo Please install Python from https://www.python.org and make sure to check
    echo "Add Python to PATH" during installation.
    echo.
    pause
    exit /b
)

echo [1/3] Python detected successfully.
echo [2/3] Checking and installing required AI dependencies...
echo.

:: 2. Install all required packages automatically
python -m pip install --upgrade pip
python -m pip install opencv-python numpy Pillow scikit-learn joblib customtkinter

if %errorlevel% neq 0 (
    color 0c
    echo.
    echo [ERROR] Package installation failed. Please check your internet connection.
    pause
    exit /b
)

echo.
echo [3/3] Dependencies verified. Starting Biometric Face System...
echo ======================================================================
echo.

:: 3. Launch python file and keep window open if it stops
python modern_biometric_face_attendance_system.py

if %errorlevel% neq 0 (
    color 0c
    echo.
    echo [!] Application exited with an error code. Look above for details.
) else (
    echo.
    echo Application closed successfully.
)

pause