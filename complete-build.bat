@echo off
setlocal

echo ==========================================
echo  Building completely MERaLiON Hackathon App
echo ==========================================

REM --- 0. FFmpeg System Check ---
echo.
echo [0/2] Checking System Dependencies (FFmpeg)...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo FFmpeg not found on system! Installing globally via winget...
    winget install -e --id Gyan.FFmpeg
    echo.
    echo PLEASE RESTART YOUR TERMINAL AFTER INSTALLATION TO REFRESH PATH!       
    echo.
) else (
    echo FFmpeg is already installed.
)

REM --- 1. Backend Setup ---
echo.
echo [1/2] Setting up Backend (Python)...
cd /d "%~dp0"
if not exist "App Part\backend\.venv" (
    echo Creating virtual environment for backend...
    python -m venv "App Part\backend\.venv"
)

echo Activating backend virtual environment...
if exist "App Part\backend\.venv\Scripts\activate.bat" (
    call "App Part\backend\.venv\Scripts\activate.bat"
) else (
    echo ERROR: Virtual environment activation script not found!
    echo Please delete "App Part\backend\.venv" and try again.
    pause
    exit /b 1
)

echo Installing backend dependencies...
python -m pip install --upgrade pip
pip install -r "App Part\backend\requirements.txt"
REM Extra tools for document + image understanding
pip install python-telegram-bot httpx uvicorn fastapi easyocr pymupdf4llm
echo Backend setup complete.
deactivate

REM --- 2. Frontend Setup ---
echo.
echo [2/2] Setting up Frontend (Next.js)...

cd /d "%~dp0\App Part\frontend"
if not exist "node_modules" (
    echo Installing frontend dependencies - this may take a while...
    call npm install
) else (
    echo Node modules found. Ensuring dependencies are up to date...
    call npm install
)

echo.
echo ==========================================
echo  Build Complete!
echo  Please review the instructions for ngrok/cloudflared if sharing.
echo  To run the app everywhere, simply execute 'run.bat'
echo ==========================================

pause