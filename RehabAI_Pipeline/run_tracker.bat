@echo off
echo ===================================================
echo Starting Rehab AI Tracker
echo ===================================================

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run setup.bat first!
    pause
    exit /b
)

echo Activating Virtual Environment...
call .venv\Scripts\activate.bat

echo Launching Webcam Tracker...
python webcam_tracker.py
pause
