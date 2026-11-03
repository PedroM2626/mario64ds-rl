@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment .venv not found. See docs\HUMAN_RECORDING.md.
    pause
    exit /b 1
)
echo Starting the Tick Tock Clock human recorder. Keep this window open.
echo Click the game window, press Enter, then play. F9 stops.
echo.
".venv\Scripts\python.exe" scripts\record_human.py %*
if errorlevel 1 (
    echo.
    echo The recorder stopped with an error. See the message above.
) else (
    echo.
    echo The recorder has stopped. Recordings are under runs\human-demonstrations.
)
pause
