@echo off
setlocal

REM Ensure we start from the script's directory
cd /d "%~dp0"

echo ==========================================
echo  Building MERaLiON Health Prototype
echo ==========================================

REM --- 1. Backend Setup ---
echo.
echo [1/2] Setting up Backend (Python)...

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
pip install -r "App Part\backend\requirements.txt"

REM Optional: Install ML dependencies if user wants to run offline models later
REM pip install -r "ML_stash\requirements.txt"

echo Backend setup complete.
REM Only deactivate if we are in a venv we activated (though call activate puts us in subshell if no setlocal? No, call runs in same shell)
REM deactivate simply clears env vars added by activate.

REM --- 2. Frontend Setup ---
echo.
echo [2/2] Setting up Frontend (Next.js)...

cd "App Part\frontend"
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
echo  To run the app, simply execute 'run_app.bat'
echo ==========================================

pause
