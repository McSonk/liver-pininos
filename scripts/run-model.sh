#!/bin/bash
set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================

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

# ==============================================================================
# ARGUMENTS PARSER
# ==============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Launches the automated liver tumour segmentation training pipeline.

Options:
  -h, --help            Display this help message and exit.
  -v, --verbose         Enable verbose logging for debugging purposes.
  -fr, --fast-run       Enable fast run mode with a smaller subset of the data.
  -r, --resume PATH     Resume training from an existing checkpoint file.
                        The provided file path must exist.

Examples:
  $(basename "$0")
  $(basename "$0") -v -fr
  $(basename "$0") -r /path/to/best_model.pth
EOF
}

VERBOSE_FLAG=""
FAST_RUN_FLAG=""
RESUME_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE_FLAG="--verbose"
            shift
            ;;
        -fr|--fast-run)
            FAST_RUN_FLAG="--fast-run"
            shift
            ;;
        -r|--resume)
            if [[ $# -lt 2 ]]; then
                echo "Error: --resume requires a file path argument." >&2
                exit 1
            fi
            RESUME_PATH="$2"
            if [[ ! -f "$RESUME_PATH" ]]; then
                echo "Error: Resume checkpoint file does not exist: $RESUME_PATH" >&2
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# ==============================================================================
# EXECUTION
# ==============================================================================

# 1. GPU selection
GPU_INDEX=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v bus_id="$GPU_PCI_BUS" '$2 == bus_id { print $1; exit }')
GPU_FOUND=false

if [ -z "$GPU_INDEX" ]; then
    echo "WARNING: No GPU found with PCI bus ID $GPU_PCI_BUS"
    
    # Prompt user if interactive
    if [[ -t 0 ]]; then
        read -r -p "Do you still want to continue without a fixed GPU? [Y/n] " gpu_response
        gpu_response=${gpu_response,,} # Convert to lowercase
        if [[ -n "$gpu_response" && "$gpu_response" != "y" && "$gpu_response" != "yes" ]]; then
            echo "Execution aborted by user."
            exit 1
        fi
    else
        echo "Non-interactive shell detected; continuing without fixed GPU."
    fi
    
    # TWCC Environment: Apply memory fragmentation fix globally for this script execution
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    echo "TWCC environment detected. Applying PyTorch CUDA memory fragmentation fix"
    echo "(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)."
    
    # Restrict to the first V100 to save VRAM (prevents PyTorch from initialising contexts on all GPUs)
    export CUDA_VISIBLE_DEVICES="0"
    echo "Restricting to first GPU (CUDA_VISIBLE_DEVICES=0) to save VRAM on TWCC."
else
    GPU_FOUND=true
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
    echo "Using GPU with PCI bus ID: $GPU_PCI_BUS (CUDA index: $GPU_INDEX)"
fi

# 2. Paths & Environment
mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# 3. Activate virtual environment
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "Error: Virtual environment not found at ${VENV_DIR}" >&2
    exit 1
fi

# 4. Cleanup old sessions
echo "Cleaning up old thesis training sessions..."
if command -v tmux >/dev/null 2>&1; then
    { tmux list-sessions -F "#{session_name}" 2>/dev/null | grep "^${TMUX_SESSION_PREFIX}_" || true; } | while read -r session; do
        echo "Killing old session: $session"
        tmux kill-session -t "$session"
    done
fi

# Build python arguments safely for both tmux (string) and nohup (array)
TMUX_PY_ARGS=""
NOHUP_PY_ARGS=()

if [[ -n "$VERBOSE_FLAG" ]]; then
    TMUX_PY_ARGS+=" --verbose"
    NOHUP_PY_ARGS+=("--verbose")
    echo "Verbose mode enabled."
fi

if [[ -n "$FAST_RUN_FLAG" ]]; then
    TMUX_PY_ARGS+=" --fast-run"
    NOHUP_PY_ARGS+=("--fast-run")
    echo "Fast-run mode enabled."
fi

if [[ -n "$RESUME_PATH" ]]; then
    # Escape quotes for tmux string
    TMUX_PY_ARGS+=" --resume \"${RESUME_PATH}\""
    NOHUP_PY_ARGS+=("--resume" "$RESUME_PATH")
    echo "Resume mode enabled. Checkpoint: $RESUME_PATH"
fi

# Build the GPU export command for tmux (ensures the environment is set correctly inside the tmux session)
GPU_EXPORT_CMD=""
if [ "$GPU_FOUND" = true ]; then
    # A100 Environment: Specific GPU locked via PCI Bus ID
    GPU_EXPORT_CMD="export CUDA_DEVICE_ORDER=\"${CUDA_DEVICE_ORDER}\" CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES}\" && "
else
    # TWCC Environment: Pass the fragmentation fix and GPU restriction into the tmux session
    GPU_EXPORT_CMD="export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF}\" CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES}\" && "
fi

# 5. Launch with tmux (falls back to nohup if tmux unavailable)
if command -v tmux &> /dev/null; then
    SESSION="${TMUX_SESSION_PREFIX}_${TIMESTAMP}"
    
    # Construct the command
    CMD="cd \"${PROJECT_DIR}\" && \
        ${GPU_EXPORT_CMD} \
        . \"${VENV_DIR}/bin/activate\" && \
        python -u main.py ${TMUX_PY_ARGS} 2>&1 | tee \"${LOG_FILE}\""

    # Start the session
    tmux new-session -d -s "$SESSION" "$CMD"

    echo "Training started in tmux session: ${SESSION}"
    echo "Attach to monitor:   tmux attach -t ${SESSION}"
    echo "Follow logs live:    tail -f ${LOG_FILE}"
    echo "Graceful stop:       tmux send-keys -t ${SESSION} C-c"

    # Prompt user to attach to the tmux session automatically (only if interactive)
    if [[ -t 0 ]]; then
        read -r -p "Do you wish to attach to the tmux session now? [Y/n] " attach_response
        attach_response=${attach_response,,} # Convert to lowercase
        if [[ -z "$attach_response" || "$attach_response" == "y" || "$attach_response" == "yes" ]]; then
            tmux attach-session -t "$SESSION"
        else
            echo "Detached mode. Use 'tmux attach -t ${SESSION}' to connect later."
        fi
    else
        echo "Non-interactive shell detected. Use 'tmux attach -t ${SESSION}' to connect."
    fi
else
    echo "tmux not found. Falling back to nohup..."
    nohup python -u main.py "${NOHUP_PY_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
    echo "Training started in background (PID: $!)"
    echo "Follow logs live:    tail -f ${LOG_FILE}"
    echo "Graceful stop:       kill $!"
fi
