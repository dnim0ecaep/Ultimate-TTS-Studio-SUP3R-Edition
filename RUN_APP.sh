#!/bin/bash

echo
echo "========================================"
echo " Ultimate TTS Studio SUP3R Edition"
echo "         APP LAUNCHER"
echo "========================================"
echo

# --- add at top ---
if [[ "${CI_DOCKER_RUN:-0}" == "1" ]]; then
  AUTO_BROWSER=0         # disable xdg-open
  WAIT_FOR_KEY=0         # skip read -rp prompts
fi

# ...original script continues...
# replace the final read -rp with:
[[ "${WAIT_FOR_KEY:-1}" == "1" ]] && read -rp "Press Enter to close..."

# Define the local environment path
LOCAL_ENV_PATH="$(pwd)/tts_env"

# Check if local environment exists
if [ ! -d "$LOCAL_ENV_PATH" ]; then
    echo "[ERROR] Local environment not found at: $LOCAL_ENV_PATH"
    echo "Please run RUN_INSTALLER.sh first!"
    read -rp "Press Enter to exit..."
    exit 1
fi

# Try to find Conda installation
CONDA_ROOT=""
FOUND_CONDA=0

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
    break
  fi
done

if [ "$FOUND_CONDA" -eq 0 ]; then
    echo "[ERROR] Could not find Anaconda or Miniconda installation!"
    read -rp "Press Enter to exit..."
    exit 1
fi

echo "[INFO] Found conda at: $CONDA_ROOT"
echo "[INFO] Launching Ultimate TTS Studio..."
echo

# Activate environment
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$LOCAL_ENV_PATH"

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate environment!"
    read -rp "Press Enter to exit..."
    exit 1
fi

echo "[INFO] Starting Ultimate TTS Studio..."
echo "[INFO] The interface will load shortly at http://127.0.0.1:7860"
echo "[INFO] Your browser will open automatically in a few seconds..."
echo "[INFO] Press Ctrl+C to stop the server"
echo

# Launch app in background
python3 launch.py &

# Wait for the server to start
echo "[INFO] Checking server status..."
until ss -ltn | grep -q ':7860'; do
    echo "[INFO] Server is still starting up..."
    sleep 2
done

echo "[INFO] Server is ready! Opening browser..."
xdg-open "http://127.0.0.1:7860" >/dev/null 2>&1 &

echo
echo "[SUCCESS] Ultimate TTS Studio is now running!"
echo "[INFO] Browser should have opened automatically."
echo "[INFO] If not, manually open: http://127.0.0.1:7860"
echo
read -rp "Press Enter to close this window..."
