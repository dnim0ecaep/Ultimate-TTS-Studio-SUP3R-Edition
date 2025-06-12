@echo off
title Ultimate TTS Studio SUP3R Edition - Launcher
color 0b

echo.
echo ========================================
echo  Ultimate TTS Studio SUP3R Edition
echo ========================================
echo.

REM Check if Python environment exists
if not exist "env\Scripts\python.exe" (
    echo [ERROR] Python environment not found!
    echo Please make sure you have installed the requirements first.
    echo.
    echo Expected location: env\Scripts\python.exe
    echo.
    pause
    exit /b 1
)

REM Check if launch.py exists
if not exist "launch.py" (
    echo [ERROR] launch.py not found!
    echo Please make sure you are running this from the correct directory.
    echo.
    pause
    exit /b 1
)

echo [INFO] Checking system environment...
echo [INFO] Activating Python environment...
call env\Scripts\activate.bat

echo [INFO] Gathering system information...
python -c "import sys; print(f'[INFO] Python version: {sys.version.split()[0]}')"

echo [INFO] Starting Ultimate TTS Studio...
echo [INFO] The interface will open shortly
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Start Python application and auto-open browser when ready
start /MIN cmd /c "timeout /t 12 /nobreak >nul & echo [INFO] Opening browser... & start http://127.0.0.1:7860"
python launch.py

REM If we get here, the application has stopped
echo.
echo [INFO] Ultimate TTS Studio has stopped.
pause 