#!/bin/bash

set -e

APP_DIR="$(pwd)"
ENV_NAME="tts_env"
ENV_PATH="$APP_DIR/$ENV_NAME"

echo
echo "========================================"
echo " Ultimate TTS Studio SUP3R Edition"
echo "               UPDATER"
echo "========================================"
echo

# Check for git
if ! command -v git &>/dev/null; then
    echo "[ERROR] Git is not installed or not in PATH!"
    echo "Please install Git: sudo apt install git"
    exit 1
fi

# Check if in a git repo
if [ ! -d ".git" ]; then
    echo "[INFO] Setting up Git repository for updates..."
    echo "[WARNING] This will replace all files with the latest version from GitHub."
    echo
    git init
    git remote add origin https://github.com/SUP3RMASS1VE/Ultimate-TTS-Studio-SUP3R-Edition.git
    echo "[INFO] Downloading latest files..."
    git fetch origin
    git checkout -b main || git checkout main || git checkout -b main origin/main
    git reset --hard origin/main
    echo "[SUCCESS] Files updated successfully!"
else
    echo "[INFO] Updating files..."

    # Stash any local changes
    if [ -n "$(git status --porcelain)" ]; then
        echo "[INFO] Backing up local changes..."
        git stash push -m "Auto-backup before update"
    fi

    echo "[INFO] Fetching latest updates..."
    git fetch origin
    git checkout main || git checkout -b main origin/main
    git reset --hard origin/main
    echo "[SUCCESS] Files updated successfully!"
fi

# Check if requirements.txt changed
if git diff HEAD~1 HEAD --name-only | grep -q "requirements.txt"; then
    deps_changed=1
else
    deps_changed=0
fi

echo
echo "========================================"
echo " Dependency Update Options"
echo "========================================"
if [ "$deps_changed" -eq 1 ]; then
    echo "[INFO] requirements.txt has been updated!"
    echo "[RECOMMENDED] You should update your dependencies."
else
    echo "[INFO] No changes detected in requirements.txt"
    echo "You can still update dependencies if needed."
fi
echo

read -p "Would you like to update Python packages? [y/n]: " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo
    echo "[INFO] Updating dependencies..."

    if ! command -v conda &>/dev/null; then
        echo "[ERROR] Conda is not available!"
        echo "Please activate the correct Conda environment or install Miniconda."
        exit 1
    fi

    if [ ! -f "$APP_DIR/install_direct.sh" ]; then
        echo "[ERROR] install_direct.sh not found!"
        exit 1
    fi

    bash "$APP_DIR/install_direct.sh"
else
    echo
    echo "[INFO] Skipping dependency update."
fi

echo
echo "========================================"
echo " Update completed!"
echo " To run the app, use: ./launch.sh"
echo "========================================"
echo
