#!/bin/bash

echo
echo "========================================"
echo " Ultimate TTS Studio SUP3R Edition"
echo "     INSTALLER LAUNCHER"
echo "========================================"
echo
echo "[INFO] This will launch the installer in a Conda environment"
echo

# Try to find Conda installation
CONDA_ROOT=""
FOUND_CONDA=0

# Common locations for Anaconda/Miniconda on Linux
CANDIDATES=(
  "$HOME/anaconda3"
  "$HOME/miniconda3"
  "/opt/anaconda3"
  "/opt/miniconda3"
  "/usr/local/anaconda3"
  "/usr/local/miniconda3"
)

for path in "${CANDIDATES[@]}"; do
  if [ -d "$path" ]; then
    CONDA_ROOT="$path"
    FOUND_CONDA=1
    echo "[INFO] Found Conda at: $path"
    break
  fi
done

if [ "$FOUND_CONDA" -eq 0 ]; then
  echo "[ERROR] Could not find Anaconda or Miniconda installation!"
  echo
  echo "Please install from:"
  echo "- Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  echo "- Anaconda: https://www.anaconda.com/products/distribution"
  echo
  read -rp "Press Enter to exit..."
  exit 1
fi

echo
echo "[INFO] Launching installer using Conda..."
echo

# Activate Conda and run the installer script
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ROOT"
bash install_direct.sh

echo
echo "[INFO] The installer has completed or is running."
echo "[INFO] You can close this window if done."
read -rp "Press Enter to exit..."
