@echo off
echo ===================================================
echo Rehab AI Pipeline - One-Shot Setup & Launch
echo ===================================================

REM 1. Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] First time setup detected. Creating Python Virtual Environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure 'python' is installed and in your PATH.
        pause
        exit /b
    )
    
    echo [INFO] Activating Virtual Environment...
    call .venv\Scripts\activate.bat
    
    echo [INFO] Installing Required Dependencies...
    pip install opencv-python numpy mediapipe==0.10.32
    
    echo [INFO] Setup Complete!
    echo ===================================================
) else (
    echo [INFO] Virtual environment found. Activating...
    call .venv\Scripts\activate.bat
)

REM 2. Launch the Application
echo [INFO] Launching Webcam Tracker...
python webcam_tracker.py
if errorlevel 1 (
    echo [ERROR] The tracker crashed. Please read the error message above.
    pause
)
