#!/bin/bash
set -euo pipefail

# CHANGE THESE VARIABLES AS NEEDED

# Adjust to your GPU's PCI bus ID (this will limit the script to run on that specific GPU)
GPU_PCI_BUS="00000000:C2:00.0"
# The project and auxiliary files are generated in the home dir
HOME_DIR="/home/misael"
# This is the name of the project directory (where main.py is located)
PROJECT_NAME="liver-pininos"
# Virtual environment dir
VENV_DIR="${HOME_DIR}/denv"
PROJECT_DIR="${HOME_DIR}/${PROJECT_NAME}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${HOME_DIR}/jobs"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"


# 1. GPU selection
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | grep "$GPU_PCI_BUS" | wc -l)
if [ "$CUDA_VISIBLE_DEVICES" -eq 0 ]; then
    echo "Error: No GPU found with PCI bus ID $GPU_PCI_BUS" >&2
    exit 1
fi
echo "Using GPU with PCI bus ID: $GPU_PCI_BUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

# 2. Paths & Environment
mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# 3. Activate virtual environment
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "Virtual environment not found at ${VENV_DIR}" >&2
    exit 1
fi

# 4. Launch with tmux (falls back to nohup if tmux unavailable)
if command -v tmux &> /dev/null; then
    SESSION="thesis_train_${TIMESTAMP}"
    tmux new-session -d -s "$SESSION" \
        "python -u main.py 2>&1 | tee ${LOG_FILE}"
    echo "Training started in tmux session: ${SESSION}"
    echo "Attach to monitor:   tmux attach -t ${SESSION}"
    echo "Follow logs live:    tail -f ${LOG_FILE}"
    echo "Graceful stop:       tmux send-keys -t ${SESSION} C-c"
else
    echo "tmux not found. Falling back to nohup..."
    nohup python -u main.py > "${LOG_FILE}" 2>&1 &
    echo "Training started in background (PID: $!)"
    echo "Follow logs live:    tail -f ${LOG_FILE}"
    echo "Graceful stop:       kill $!"
fi
