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

TMUX_SESSION_PREFIX="thesis_train"


# 1. GPU selection
GPU_INDEX=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v bus_id="$GPU_PCI_BUS" '$2 == bus_id { print $1; exit }')
if [ -z "$GPU_INDEX" ]; then
    echo "Error: No GPU found with PCI bus ID $GPU_PCI_BUS" >&2
    exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
echo "Using GPU with PCI bus ID: $GPU_PCI_BUS (CUDA index: $GPU_INDEX)"

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

# 4 Cleanup old sessions (Optional but recommended)
echo "Cleaning up old thesis training sessions..."
if command -v tmux >/dev/null 2>&1; then
    tmux list-sessions -F "#{session_name}" 2>/dev/null | grep "^${TMUX_SESSION_PREFIX}_" || true | while read -r session; do
        echo "Killing old session: $session"
        tmux kill-session -t "$session"
    done
fi

# 5. Launch with tmux (falls back to nohup if tmux unavailable)
if command -v tmux &> /dev/null; then
    SESSION="${TMUX_SESSION_PREFIX}_${TIMESTAMP}"
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
