@echo off
title Ultimate TTS Studio SUP3R Edition - Updater
color 0e

echo.
echo ========================================
echo  Ultimate TTS Studio SUP3R Edition
echo               UPDATER
echo ========================================
echo.

REM Check if git is available
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed or not in PATH!
    echo Please install Git from: https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

REM Check if we're in a git repository
if not exist ".git" (
    echo [INFO] Setting up Git repository for updates...
    echo [WARNING] This will replace all files with the latest version from GitHub.
    echo.
    
    REM Initialize git repository
    git init
    git remote add origin https://github.com/SUP3RMASS1VE/Ultimate-TTS-Studio-SUP3R-Edition.git
    
    echo [INFO] Downloading latest files...
    git fetch origin
    
    REM Create main branch and force reset to match origin
    echo [INFO] Setting up main branch...
    git checkout -b main
    git reset --hard origin/main
    
    echo [SUCCESS] Files updated successfully!
) else (
    echo [INFO] Updating files...
    
    REM Handle any local changes by stashing them
    git status --porcelain > temp_status.txt
    set /p git_status=<temp_status.txt
    del temp_status.txt
    
    if not "%git_status%"=="" (
        echo [INFO] Backing up local changes...
        git stash push -m "Auto-backup before update"
    )
    
    REM Fetch latest updates first
    echo [INFO] Fetching latest updates...
    git fetch origin
    
    REM Get current branch
    for /f "tokens=*" %%i in ('git branch --show-current') do set current_branch=%%i
    
    REM If no current branch, switch to main branch
    if "%current_branch%"=="" (
        echo [INFO] Switching to main branch...
        git checkout main
        if errorlevel 1 (
            echo [INFO] Creating main branch...
            git checkout -b main origin/main
            if errorlevel 1 (
                echo [ERROR] Failed to setup main branch!
                pause
                exit /b 1
            )
        )
    )
    
    REM Update to latest version
    echo [INFO] Updating to latest version...
    git reset --hard origin/main
    if errorlevel 1 (
        echo [ERROR] Failed to update files!
        pause
        exit /b 1
    )
    
    echo [SUCCESS] Files updated successfully!
)

REM Check if requirements.txt was updated
git diff HEAD~1 HEAD --name-only 2>nul | findstr "requirements.txt" >nul
if not errorlevel 1 (
    echo.
    echo [INFO] Python dependencies may need updating.
    echo.
    echo Would you like to update Python packages now?
    echo [y] Yes - Update dependencies (recommended)
    echo [n] No - Skip dependency update
    echo.
    set /p update_deps="Enter your choice (y/n): "
    
    if /i "%update_deps%"=="y" (
        if exist "env\Scripts\activate.bat" (
            echo [INFO] Updating dependencies...
            call env\Scripts\activate.bat
            pip install -r requirements.txt
            echo [SUCCESS] Dependencies updated!
        ) else (
            echo [WARNING] Python environment not found. Please update manually.
        )
    )
)

echo.
echo ========================================
echo Update completed! You can now run launch.bat
echo ========================================
echo.
pause 