@echo off
echo ===================================================
echo Rehab AI Pipeline Setup
echo ===================================================

echo [1/3] Creating Python Virtual Environment (.venv)...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment. Ensure 'python' is in your PATH.
    pause
    exit /b
)

echo [2/3] Activating Virtual Environment...
call .venv\Scripts\activate.bat

echo [3/3] Installing Required Dependencies...
REM We use the specific versions that we know work with our Tasks API code
pip install opencv-python numpy
pip install mediapipe==0.10.32

echo.
echo ===================================================
echo Setup Complete!
echo ===================================================
echo To run the tracker, double-click 'run_tracker.bat' or use:
echo .venv\Scripts\python webcam_tracker.py
pause
