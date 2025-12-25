@echo off
REM Togyzkumalak Engine - Windows Launcher

echo ============================================================
echo   TOGYZKUMALAK ENGINE
echo   AI-Powered Toguz Kumalak Game
echo ============================================================
echo.

REM Check if virtual environment exists
if exist "..\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ..\venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found.
    echo Creating one...
    python -m venv ..\venv
    call ..\venv\Scripts\activate.bat
    pip install -r requirements.txt
)

REM Install requirements if needed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing requirements...
    pip install -r requirements.txt
)

echo.
echo Starting server...
echo Open http://localhost:8000 in your browser
echo Press Ctrl+C to stop
echo.

python run.py

pause

