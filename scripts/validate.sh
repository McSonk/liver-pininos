#!/bin/bash
set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# CHANGE THESE VARIABLES AS NEEDED

# Adjust to your GPU's PCI bus ID (this will limit the script to run on that specific GPU)
GPU_PCI_BUS="00000000:C2:00.0"

# This is the name of the project directory (where do_inference.py is located)
PROJECT_NAME="liver-pininos"

# Virtual environment dir
VENV_DIR="${HOME}/denv"
PROJECT_DIR="${HOME}/${PROJECT_NAME}"

# Default Python executable (can be overridden with --python)
DEFAULT_PYTHON="${VENV_DIR}/bin/python"
PYTHON_BIN="${DEFAULT_PYTHON}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${HOME}/jobs"
LOG_FILE="${LOG_DIR}/inference_${TIMESTAMP}.log"

TMUX_SESSION_PREFIX="thesis_inference"

# ==============================================================================
# ARGUMENTS PARSER
# ==============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Launches the automated liver tumour segmentation inference pipeline on the server.
(Note: Evaluation and metrics computation should be run locally via do_evaluation.py)

Options:
  -h, --help               Display this help message and exit.
  -chk, --checkpoint PATH  Path to the model checkpoint (.pth) to use for inference.
                           The file must exist. The path is converted to absolute.
  -o, --output-dir PATH    Directory to save the raw NIfTI predictions.
                           Defaults to <RUN_DIR>/test_predictions if not provided.
                           The path is converted to absolute.
  -p, --python PATH        Python executable to use for inference.
                           Default: ${DEFAULT_PYTHON}.
                           Example: --python "\${HOME}/mamba-env/bin/python".
  
  Any unrecognised arguments are passed directly to do_inference.py.

Examples:
  $(basename "$0") --checkpoint /path/to/best_model.pth
  $(basename "$0") -chk ./checkpoints/last_epoch.pth -o /path/to/custom/output
  $(basename "$0") --checkpoint /path/to/best_model.pth --python "\${HOME}/mamba-env/bin/python"
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
        --output-dir|-o)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a directory path argument." >&2
                exit 1
            fi
            # Reject option tokens mistakenly passed as the value
            if [[ "$2" == -* ]]; then
                echo "Error: $1 received an option token ('$2') instead of a path." >&2
                echo "       Did you forget to provide the directory path?" >&2
                exit 1
            fi
            
            # Convert to absolute path (allow non-existent paths with -m)
            if command -v realpath >/dev/null 2>&1; then
                ABS_OUT_PATH=$(realpath -m "$2")
            else
                ABS_OUT_PATH=$(readlink -m "$2")
            fi
            
            # Reconstruct the argument for python using the absolute path
            ARGS_FOR_PYTHON+=("$1" "$ABS_OUT_PATH")
            shift 2
            ;;
        --post-process|-pp)
            echo "Warning: --post-process is ignored for server-side inference." >&2
            echo "         Post-processing is applied during local evaluation (do_evaluation.py)." >&2
            shift
            ;;
        -p|--python)
            if [[ $# -lt 2 ]]; then
                echo "Error: --python requires a file path argument." >&2
                exit 1
            fi
            PYTHON_BIN="$2"
            shift 2
            ;;
        *)
            # Pass any other argument directly to Python
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

# Resolve the selected Python executable before changing directories.
PYTHON_BIN="${PYTHON_BIN/#\~/$HOME}"

# If a bare command name was supplied, resolve it from PATH.
if [[ "$PYTHON_BIN" != */* ]]; then
    if resolved_python="$(command -v -- "$PYTHON_BIN")"; then
        PYTHON_BIN="$resolved_python"
    fi
fi

# Make the path absolute without resolving symlinks, because venv/conda
# launchers may rely on the original executable path.
if [[ "$PYTHON_BIN" != /* ]]; then
    PYTHON_BIN="${PWD}/${PYTHON_BIN}"
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

# 3. Validate selected Python environment
if [[ -d "$PYTHON_BIN" ]]; then
    echo "Error: --python expects an executable file, but this is a directory: $PYTHON_BIN" >&2
    echo "Hint: try ${PYTHON_BIN%/}/bin/python" >&2
    exit 1
fi

if [[ ! -f "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    echo "Error: Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

ENV_BIN_DIR="$(dirname -- "$PYTHON_BIN")"
ENV_EXPORT_CMD="export PATH=\"${ENV_BIN_DIR}:${PATH}\" && "
export PATH="${ENV_BIN_DIR}:${PATH}"

echo "Using Python: $PYTHON_BIN"

# 4. Cleanup old sessions
echo "Cleaning up old thesis inference sessions..."
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

echo "Inference arguments: ${NOHUP_PY_ARGS[*]}"

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
        ${ENV_EXPORT_CMD} \
        \"${PYTHON_BIN}\" -u do_inference.py ${TMUX_PY_ARGS} 2>&1 | tee \"${LOG_FILE}\""

    # Start the session
    tmux new-session -d -s "$SESSION" "$CMD"

    echo "Test inference started in tmux session: ${SESSION}"
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
    nohup "$PYTHON_BIN" -u do_inference.py "${NOHUP_PY_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
    echo "Test inference started in background (PID: $!)"
    echo "Follow logs live:    tail -f ${LOG_FILE}"
    echo "Graceful stop:       kill $!"
fi
