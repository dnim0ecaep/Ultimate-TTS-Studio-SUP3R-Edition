@echo off
setlocal enabledelayedexpansion
echo.
echo ========================================
echo  Ultimate TTS Studio SUP3R Edition
echo     DIRECT INSTALLER
echo ========================================
echo.

REM Check if we're in a conda environment
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda is not activated!
    echo Please run this from an Anaconda/Miniconda prompt.
    pause
    exit /b 1
)

echo [INFO] Starting installation process...
echo.

REM Get current directory
set "APP_DIR=%cd%"
set "ENV_NAME=tts_env"
set "ENV_PATH=%APP_DIR%\%ENV_NAME%"

REM Step 1: Create or update conda environment in local directory
echo [STEP 1/6] Checking conda environment...
if exist "%ENV_PATH%" (
    echo [INFO] Environment already exists at "%ENV_PATH%"
    echo [INFO] Activating existing environment...
) else (
    echo [INFO] Creating new conda environment in "%ENV_PATH%"...
    echo [INFO] This may take a few minutes...
    call conda create --prefix "%ENV_PATH%" python=3.10 -y >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create conda environment!
        pause
        exit /b 1
    )
)
echo [SUCCESS] Conda environment ready!
echo.

REM Step 2: Activate the environment
echo [STEP 2/6] Activating conda environment...
call conda activate "%ENV_PATH%"
if !errorlevel! neq 0 (
    echo [ERROR] Failed to activate conda environment!
    echo [INFO] Trying alternative activation method...
    call activate "%ENV_PATH%"
    if !errorlevel! neq 0 (
        echo [ERROR] Still failed to activate environment!
        pause
        exit /b 1
    )
)
echo [SUCCESS] Environment activated!
echo.

REM Verify we're in the right environment
echo [INFO] Current Python location:
where python
echo.

REM Step 3: Install UV using pip
echo [STEP 3/6] Installing UV package manager...
call python -m pip install --upgrade pip
call python -m pip install uv
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install UV!
    pause
    exit /b 1
)
echo [SUCCESS] UV installed successfully!
echo.

REM Step 4: Install requirements.txt using UV
echo [STEP 4/6] Installing requirements from requirements.txt...
if exist "%APP_DIR%\requirements.txt" (
    call python -m uv pip install -r "%APP_DIR%\requirements.txt"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install requirements!
        pause
        exit /b 1
    )
    echo [SUCCESS] Requirements installed successfully!
) else (
    echo [WARNING] requirements.txt not found, skipping...
)
echo.

REM Step 5: Install pynini using conda
echo [STEP 5/6] Installing pynini from conda-forge...
call conda install -c conda-forge pynini==2.1.6 -y
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install pynini!
    pause
    exit /b 1
)
echo [SUCCESS] pynini installed successfully!
echo.

REM Step 6: Install WeTextProcessing using UV
echo [STEP 6/6] Installing WeTextProcessing using UV...
call python -m uv pip install WeTextProcessing --no-deps
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install WeTextProcessing!
    pause
    exit /b 1
)
echo [SUCCESS] WeTextProcessing installed successfully!
echo.

echo ========================================
echo  Installation completed successfully!
echo ========================================
echo.
echo Environment location: "%ENV_PATH%"
echo To activate this environment in the future, use:
echo   conda activate "%ENV_PATH%"
echo.
echo Press any key to close this window...
pause >nul 