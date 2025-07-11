#!/bin/bash

echo
echo "========================================"
echo "  Ultimate TTS Studio SUP3R Edition"
echo "     DIRECT INSTALLER"
echo "========================================"
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda is not installed or not in PATH."
    echo "Please run this from an Anaconda/Miniconda shell or ensure conda is installed."
    exit 1
fi

echo "[INFO] Starting installation process..."
echo

# Set variables
APP_DIR="$(pwd)"
ENV_NAME="tts_env"
ENV_PATH="${APP_DIR}/${ENV_NAME}"

# Step 1: Create or update conda environment
echo "[STEP 1/6] Checking conda environment..."
if [ -d "$ENV_PATH" ]; then
    echo "[INFO] Environment already exists at '$ENV_PATH'"
    echo "[INFO] Activating existing environment..."
else
    echo "[INFO] Creating new conda environment in '$ENV_PATH'..."
    echo "[INFO] This may take a few minutes..."
    conda create --prefix "$ENV_PATH" python=3.10 -y || { echo "[ERROR] Failed to create conda environment!"; exit 1; }
fi
echo "[SUCCESS] Conda environment ready!"
echo

# Step 2: Activate the environment
echo "[STEP 2/6] Activating conda environment..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH" || { echo "[ERROR] Failed to activate conda environment!"; exit 1; }
echo "[SUCCESS] Environment activated!"
echo

# Step 3: Install UV
echo "[STEP 3/6] Installing UV package manager..."
python -m pip install --upgrade pip
python -m pip install uv || { echo "[ERROR] Failed to install UV!"; exit 1; }
echo "[SUCCESS] UV installed successfully!"
echo

# Step 4: Install requirements.txt
echo "[STEP 4/6] Installing requirements from requirements.txt..."
if [ -f "${APP_DIR}/requirements.txt" ]; then
    python -m uv pip install -r "${APP_DIR}/requirements.txt" || { echo "[ERROR] Failed to install requirements!"; exit 1; }
    echo "[SUCCESS] Requirements installed successfully!"
else
    echo "[WARNING] requirements.txt not found, skipping..."
fi
echo

# Step 5: Install pynini
echo "[STEP 5/6] Installing pynini from conda-forge..."
conda install -c conda-forge pynini==2.1.6 -y || { echo "[ERROR] Failed to install pynini!"; exit 1; }
echo "[SUCCESS] pynini installed successfully!"
echo

# Step 6: Install WeTextProcessing
echo "[STEP 6/6] Installing WeTextProcessing using UV..."
python -m uv pip install WeTextProcessing --no-deps || { echo "[ERROR] Failed to install WeTextProcessing!"; exit 1; }
echo "[SUCCESS] WeTextProcessing installed successfully!"
echo

echo "========================================"
echo "  Installation completed successfully!"
echo "========================================"
echo
echo "Environment location: $ENV_PATH"
echo "To activate this environment in the future, use:"
echo "  conda activate \"$ENV_PATH\""
echo
