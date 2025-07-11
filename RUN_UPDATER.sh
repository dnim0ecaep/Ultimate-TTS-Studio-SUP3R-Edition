#!/bin/bash

echo
echo "========================================"
echo " Ultimate TTS Studio SUP3R Edition"
echo "     UPDATER LAUNCHER"
echo "========================================"
echo
echo "[INFO] This will launch the updater in Conda environment."
echo

# Try to find Conda installation
FOUND_CONDA=0
CONDA_ROOT=""

check_conda_path() {
    if [ -d "$1" ]; then
        CONDA_ROOT="$1"
        FOUND_CONDA=1
        echo "[INFO] Found Conda at: $1"
    fi
}

# Check common install paths
check_conda_path "$HOME/anaconda3"
check_conda_path "$HOME/Anaconda3"
check_conda_path "$HOME/miniconda3"
check_conda_path "$HOME/Miniconda3"
check_conda_path "/opt/anaconda3"
check_conda_path "/opt/miniconda3"

if [ "$FOUND_CONDA" -eq 0 ]; then
    echo "[ERROR] Could not find Anaconda or Miniconda installation!"
    echo
    echo "Please install from:"
    echo "- Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo "- Anaconda: https://www.anaconda.com/products/individual"
    echo
    exit 1
fi

echo
echo "[INFO] Launching updater in new terminal..."
echo

# Launch the updater in a new terminal window
gnome-terminal -- bash -c "source \"$CONDA_ROOT/etc/profile.d/conda.sh\" && conda activate base && bash update.sh; exec bash"

echo
echo "[INFO] The updater is running in the new terminal."
echo "[INFO] This window can be closed."
echo
