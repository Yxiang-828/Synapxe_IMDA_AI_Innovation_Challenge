@echo off
setlocal

echo ==========================================
echo  Aiko's Dependency Downloader 💕
echo ==========================================
echo.
echo Checking and installing all required Python dependencies for the backend...
echo.

REM Navigate to the exact script directory
cd /d "%~dp0"

REM Ensure virtual environment exists
if not exist "App Part\backend\.venv" (
    echo [!] Virtual environment not found! Creating one...
    python -m venv "App Part\backend\.venv"
)

REM Activate and Install
echo Activating venv and running pip install...
call "App Part\backend\.venv\Scripts\activate.bat"

python -m pip install --upgrade pip
pip install -r "App Part\backend\requirements.txt"

echo.
echo ==========================================
echo  All dependencies installed successfully!
echo  You are ready to run: .\run_app.bat
echo ==========================================
pause
