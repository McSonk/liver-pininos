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

# This is the name of the project directory (where do_test.py is located)
PROJECT_NAME="liver-pininos"

# Virtual environment dir
VENV_DIR="${HOME_DIR}/denv"
PROJECT_DIR="${HOME_DIR}/${PROJECT_NAME}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${HOME_DIR}/jobs"
LOG_FILE="${LOG_DIR}/test_${TIMESTAMP}.log"

TMUX_SESSION_PREFIX="thesis_test"

# ==============================================================================
# ARGUMENTS PARSER
# ==============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Launches the automated liver tumour segmentation validation/testing pipeline.

Options:
  -h, --help               Display this help message and exit.
  -chk, --checkpoint PATH  Path to the model checkpoint (.pth) to evaluate.
                           The file must exist. The path is converted to absolute.
  -pp, --post-process      Apply post-processing to the predicted segmentation maps.
  
  Any unrecognised arguments are passed directly to do_test.py.

Examples:
  $(basename "$0") --checkpoint /path/to/best_model.pth
  $(basename "$0") -chk ./checkpoints/last_epoch.pth --post-process
EOF
}

CHECKPOINT_PATH=""
ARGS_FOR_PYTHON=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        --checkpoint|-chk)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a file path argument." >&2
                exit 1
            fi
            if [[ ! -f "$2" ]]; then
                echo "Error: Checkpoint file does not exist: $2" >&2
                exit 1
            fi
            
            # Convert to absolute path to avoid working directory issues in tmux/nohup
            if command -v realpath >/dev/null 2>&1; then
                ABS_CHK_PATH=$(realpath "$2")
            else
                ABS_CHK_PATH=$(readlink -f "$2")
            fi
            CHECKPOINT_PATH="$ABS_CHK_PATH"
            
            # Reconstruct the argument for python using the absolute path
            ARGS_FOR_PYTHON+=("$1" "$ABS_CHK_PATH")
            shift 2
            ;;
        *)
            # Pass any other argument (e.g., --post-process / -pp) directly to Python
            ARGS_FOR_PYTHON+=("$1")
            shift
            ;;
    esac
done

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: --checkpoint is required." >&2
    usage
    exit 1
fi

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
else
    GPU_FOUND=true
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
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
echo "Cleaning up old thesis test sessions..."
if command -v tmux >/dev/null 2>&1; then
    { tmux list-sessions -F "#{session_name}" 2>/dev/null | grep "^${TMUX_SESSION_PREFIX}_" || true; } | while read -r session; do
        echo "Killing old session: $session"
        tmux kill-session -t "$session"
    done
fi

# Build python arguments safely for both tmux (string) and nohup (array)
TMUX_PY_ARGS=""
NOHUP_PY_ARGS=()

for arg in "${ARGS_FOR_PYTHON[@]}"; do
    # Escape characters that would still expand inside double quotes when executed via tmux
    escaped_arg="$arg"
    escaped_arg="${escaped_arg//\\/\\\\}"
    escaped_arg="${escaped_arg//\"/\\\"}"
    escaped_arg="${escaped_arg//\$/\\\$}"
    escaped_arg="${escaped_arg//\`/\\\`}"

    TMUX_PY_ARGS+=" \"$escaped_arg\""
    NOHUP_PY_ARGS+=("$arg")
done

echo "Validation arguments: ${NOHUP_PY_ARGS[*]}"

# Build the GPU export command for tmux (only if a specific GPU was selected)
# This prevents accidentally exporting an empty CUDA_VISIBLE_DEVICES which would hide all GPUs
GPU_EXPORT_CMD=""
if [ "$GPU_FOUND" = true ]; then
    # A100 Environment: Specific GPU locked via PCI Bus ID
    GPU_EXPORT_CMD="export CUDA_DEVICE_ORDER=\"${CUDA_DEVICE_ORDER}\" CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES}\" && "
else
    # TWCC Environment: PCI mapping skipped, but V100 GPUs are present.
    # Apply the memory fragmentation fix here to prevent OOM errors.
    GPU_EXPORT_CMD="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "

    echo "TWCC environment detected. Applying PyTorch CUDA memory fragmentation fix"
    echo "(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)."

    # Optional but recommended: Restrict to the first V100. 
    # If left unset, PyTorch will initialise CUDA contexts on both V100s, 
    # which wastes ~1-2 GB of VRAM per unused GPU.
    GPU_EXPORT_CMD+="export CUDA_VISIBLE_DEVICES=0 && "
    echo "Restricting to first GPU (CUDA_VISIBLE_DEVICES=0) to save VRAM on TWCC."
fi

# 5. Launch with tmux (falls back to nohup if tmux unavailable)
if command -v tmux &> /dev/null; then
    SESSION="${TMUX_SESSION_PREFIX}_${TIMESTAMP}"
    
    # Construct the command
    CMD="cd \"${PROJECT_DIR}\" && \
        ${GPU_EXPORT_CMD} \
        . \"${VENV_DIR}/bin/activate\" && \
        python -u do_test.py ${TMUX_PY_ARGS} 2>&1 | tee \"${LOG_FILE}\""

    # Start the session
    tmux new-session -d -s "$SESSION" "$CMD"

    echo "Test evaluation started in tmux session: ${SESSION}"
    echo "Attach to monitor:   tmux attach -t ${SESSION}"
    echo "Follow logs live:    tail -f ${LOG_FILE}"
    echo "Graceful stop:       tmux send-keys -t ${SESSION} C-c"

    # Prompt user to attach to the tmux session automatically (only when interactive)
    if [[ -t 0 ]]; then
        read -r -p "Do you wish to attach to the tmux session now? [Y/n] " attach_response
        attach_response=${attach_response,,} # Convert to lowercase
        if [[ -z "$attach_response" || "$attach_response" == "y" || "$attach_response" == "yes" ]]; then
            tmux attach-session -t "$SESSION"
        else
            echo "Detached mode. Use 'tmux attach -t ${SESSION}' to connect later."
        fi
    else
        echo "Non-interactive shell detected; leaving tmux session detached."
    fi
else
    echo "tmux not found. Falling back to nohup..."
    nohup python -u do_test.py "${NOHUP_PY_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
    echo "Test evaluation started in background (PID: $!)"
    echo "Follow logs live:    tail -f ${LOG_FILE}"
    echo "Graceful stop:       kill $!"
fi
