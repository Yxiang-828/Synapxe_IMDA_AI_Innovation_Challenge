@echo off
setlocal

echo ==========================================
echo  Starting MERaLiON Health Prototype...
echo ==========================================

REM --- 0. Clean up previous instances ---
echo Cleaning up previous instances on ports 3000 and 8080...
taskkill /FI "WINDOWTITLE eq MERaLiON Backend*" /F /T >nul 2>&1
taskkill /FI "WINDOWTITLE eq MERaLiON Frontend*" /F /T >nul 2>&1
taskkill /FI "WINDOWTITLE eq MERaLiON Telegram Bot*" /F /T >nul 2>&1
FOR /F "tokens=5" %%P IN ('netstat -a -n -o ^| findstr :3000') DO taskkill /F /PID %%P >nul 2>&1
FOR /F "tokens=5" %%P IN ('netstat -a -n -o ^| findstr :8080') DO taskkill /F /PID %%P >nul 2>&1

cd /d "%~dp0"

REM --- 1. Start Backend FastAPI ---
echo [1/3] Launching Backend Server...
start "MERaLiON Backend (FastAPI)" cmd /k "cd /d `"App Part\backend`" && call .venv\Scripts\activate.bat && uvicorn main:app --host 127.0.0.1 --port 8080 --reload"

REM --- 2. Start Telegram Bot ---
echo [2/3] Launching Telegram Polling Bot...
start "MERaLiON Telegram Bot" cmd /k "cd /d `"App Part\backend`" && call .venv\Scripts\activate.bat && python bot.py"

REM --- 3. Start Frontend ---
echo [3/3] Launching Frontend (Next.js)...
start "MERaLiON Frontend" cmd /k "cd /d `"App Part\frontend`" && npm run dev"

echo.
echo ==========================================
echo  All Services started!
echo  - Backend API: http://127.0.0.1:8080
echo  - Telegram Bot: Polling active
echo  - Frontend (Mini App Server): http://localhost:3000
echo ==========================================
echo Note: If you need to test the Telegram Mini Apps from your phone, 
echo remember to tunnel port 3000 (using localtunnel, ngrok, or cloudflared)
echo and update BASE_URL in App Part/backend/bot.py to point at your tunnel!
echo ==========================================
pause
