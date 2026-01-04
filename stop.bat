@echo off
set PORT=8000
echo ============================================================
echo   TOGYZKUMALAK ENGINE - STOPPING
echo ============================================================

echo Checking for process on port %PORT%...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :%PORT% ^| findstr LISTENING') do (
    echo Killing process PID %%a...
    taskkill /F /PID %%a 2>nul
)

:: Fallback using PowerShell if CMD fails or LISTENING is localized
powershell -Command "Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force; echo \"Killed process PID $($_.OwningProcess) using PowerShell\" }"

echo Port %PORT% is free.
