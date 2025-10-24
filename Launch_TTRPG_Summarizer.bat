@echo off
REM TTRPG Session Summarizer Launcher
REM This batch file activates the virtual environment and runs the summarizer

echo ================================================
echo    TTRPG SESSION SUMMARIZER
echo ================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if virtual environment exists in parent directory
if not exist "..\\.venv_py311\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please ensure .venv_py311 exists at E:\Dropbox\Python\.venv_py311
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment from parent directory
echo Activating Python virtual environment...
call "..\\.venv_py311\Scripts\activate.bat"

REM Check if Ollama is running (optional check)
echo.
echo Checking for Ollama...
ollama list >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Ollama may not be running!
    echo Please start Ollama before continuing.
    echo.
    echo Press any key to continue anyway, or close this window to exit...
    pause >nul
)

REM Run the summarizer
echo.
echo Starting TTRPG Summarizer...
echo.
python ttrpg_summarizer_py311.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ================================================
    echo An error occurred!
    echo ================================================
    pause
)
