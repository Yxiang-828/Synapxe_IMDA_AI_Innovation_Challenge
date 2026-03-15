@echo off
setlocal

echo ==========================================
echo  Starting MERaLiON Health Prototype...
echo ==========================================

REM --- 1. Start Backend ---
echo [1/2] Launching Backend Server...
start "MERaLiON Backend (FastAPI)" cmd /k "cd /d "App Part\backend" && call .venv\Scripts\activate.bat && uvicorn main:app --host 127.0.0.1 --port 8080 --reload"

REM --- 2. Start Frontend ---
echo [2/2] Launching Frontend (Next.js)...
start "MERaLiON Frontend" cmd /k "cd /d "App Part\frontend" && npm run dev"

echo.
echo ==========================================
echo  Services started!
echo  Backend: http://127.0.0.1:8080
echo  Frontend: http://localhost:3000
echo ==========================================
timeout /t 5
start http://localhost:3000
