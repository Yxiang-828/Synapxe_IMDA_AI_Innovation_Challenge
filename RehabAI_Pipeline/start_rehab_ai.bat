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
)

echo [INFO] Activating Virtual Environment...
call .venv\Scripts\activate.bat

echo [INFO] Checking dependencies...
REM We use python -c to check if the modules are installed. 
REM If they fail to import, errorlevel is 1, so we trigger pip install.
python -c "import cv2, numpy, mediapipe" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Dependencies missing. Installing Required Dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo [INFO] All dependencies are already installed.
)

REM 2. Launch the Application
echo [INFO] Launching Webcam Tracker...
python webcam_tracker.py
if errorlevel 1 (
    echo [ERROR] The tracker crashed. Please read the error message above.
    pause
)
