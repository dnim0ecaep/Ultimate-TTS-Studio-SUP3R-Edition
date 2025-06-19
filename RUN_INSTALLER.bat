@echo off
echo.
echo ========================================
echo  Ultimate TTS Studio SUP3R Edition
echo     INSTALLER LAUNCHER
echo ========================================
echo.
echo [INFO] This will launch the installer in Anaconda/Miniconda Prompt
echo.

REM Try to find Anaconda/Miniconda installation
set CONDA_ROOT=
set FOUND_CONDA=0

REM Check common installation paths
if exist "%USERPROFILE%\Anaconda3" (
    set CONDA_ROOT=%USERPROFILE%\Anaconda3
    set FOUND_CONDA=1
    echo [INFO] Found Anaconda at: %USERPROFILE%\Anaconda3
) else if exist "%USERPROFILE%\anaconda3" (
    set CONDA_ROOT=%USERPROFILE%\anaconda3
    set FOUND_CONDA=1
    echo [INFO] Found Anaconda at: %USERPROFILE%\anaconda3
) else if exist "%USERPROFILE%\Miniconda3" (
    set CONDA_ROOT=%USERPROFILE%\Miniconda3
    set FOUND_CONDA=1
    echo [INFO] Found Miniconda at: %USERPROFILE%\Miniconda3
) else if exist "%USERPROFILE%\miniconda3" (
    set CONDA_ROOT=%USERPROFILE%\miniconda3
    set FOUND_CONDA=1
    echo [INFO] Found Miniconda at: %USERPROFILE%\miniconda3
) else if exist "%ProgramData%\Anaconda3" (
    set CONDA_ROOT=%ProgramData%\Anaconda3
    set FOUND_CONDA=1
    echo [INFO] Found Anaconda at: %ProgramData%\Anaconda3
) else if exist "%ProgramData%\Miniconda3" (
    set CONDA_ROOT=%ProgramData%\Miniconda3
    set FOUND_CONDA=1
    echo [INFO] Found Miniconda at: %ProgramData%\Miniconda3
) else if exist "C:\Anaconda3" (
    set CONDA_ROOT=C:\Anaconda3
    set FOUND_CONDA=1
    echo [INFO] Found Anaconda at: C:\Anaconda3
) else if exist "C:\Miniconda3" (
    set CONDA_ROOT=C:\Miniconda3
    set FOUND_CONDA=1
    echo [INFO] Found Miniconda at: C:\Miniconda3
) else if exist "C:\tools\Anaconda3" (
    set CONDA_ROOT=C:\tools\Anaconda3
    set FOUND_CONDA=1
    echo [INFO] Found Anaconda at: C:\tools\Anaconda3
) else if exist "C:\tools\Miniconda3" (
    set CONDA_ROOT=C:\tools\Miniconda3
    set FOUND_CONDA=1
    echo [INFO] Found Miniconda at: C:\tools\Miniconda3
)

if "%FOUND_CONDA%"=="0" (
    echo [ERROR] Could not find Anaconda or Miniconda installation!
    echo.
    echo Please install from:
    echo - Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo - Anaconda: https://www.anaconda.com/products/individual
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] Launching installer in Anaconda/Miniconda prompt...
echo [INFO] A new window will open. Please wait...
echo.

REM Launch the installer in Anaconda prompt
start "TTS Installer" /D "%cd%" "%windir%\System32\cmd.exe" /K ""%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%" && install_direct.bat"

echo.
echo [INFO] The installer is running in the new window.
echo [INFO] This window can be closed.
echo.
pause 