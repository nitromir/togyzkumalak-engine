@echo off
set PORT=8000
echo ============================================================
echo   TOGYZKUMALAK ENGINE - STARTING
echo ============================================================

:: Kill any process on port 8000
echo Checking for existing server on port %PORT%...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :%PORT% ^| findstr LISTENING') do (
    echo Cleaning up port %PORT% (PID %%a)...
    taskkill /F /PID %%a 2>nul
)

:: Activate venv and run
cd gym-togyzkumalak-master\togyzkumalak-engine
echo Starting FastAPI server...
..\venv\Scripts\python.exe run.py

echo Waiting for server to initialize...
timeout /t 3 /nobreak > nul

:: Open browser
start http://localhost:%PORT%
echo Done.

