@echo off
REM Parse all training data for Togyzkumalak AI
REM This script converts human game records to training format

echo ========================================
echo Togyzkumalak Data Parser
echo ========================================
echo.

cd /d "%~dp0\.."

REM Activate virtual environment
if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
) else if exist "..\..\venv\Scripts\activate.bat" (
    call ..\..\venv\Scripts\activate.bat
)

echo Looking for training data files...
echo.

REM Run the parser
python scripts\data_parsers.py ^
    --opening-books "%~dp0\..\..\..\..\Android-APK\assets\internal\open_tree2.txt" ^
                    "%~dp0\..\..\..\..\Android-APK\assets\internal\open_tree3.txt" ^
                    "%~dp0\..\..\..\..\Android-APK\assets\internal\open_tree4.txt" ^
                    "%~dp0\..\..\..\..\Android-APK\assets\internal\open_tree5.txt" ^
                    "%~dp0\..\..\..\..\Android-APK\assets\internal\open_tree6.txt" ^
                    "%~dp0\..\..\..\..\Android-APK\assets\internal\open_tree7.txt" ^
                    "%~dp0\..\..\..\..\Android-APK\assets\internal\open_tree8.txt" ^
                    "%~dp0\..\..\..\..\Android-APK\assets\internal\open_tree9.txt" ^
    --championship "%~dp0\..\..\..\..\games.txt" ^
    --playok "%~dp0\..\..\..\..\all_results_combined.txt" ^
    --output "training_data"

echo.
echo ========================================
echo Parse complete! Check training_data folder
echo ========================================
pause

