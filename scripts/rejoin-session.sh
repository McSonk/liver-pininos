#!/bin/bash
echo "Active training and testing (validation) sessions:"

if ! command -v tmux >/dev/null 2>&1; then
    echo "(none running)"
    echo ""
    echo "tmux is not installed."
    exit 1
fi

# Fetch sessions matching either the training or testing prefixes
raw_sessions=$(tmux list-sessions -F "#{session_name}" 2>/dev/null | grep -E "^(thesis_train_|thesis_test_)" || true)

if [ -n "$raw_sessions" ]; then
    # Sort alphabetically. Since the timestamp format is YYYYMMDD_HHMMSS, 
    # alphabetical sorting guarantees chronological order (oldest first, latest last).
    sessions=$(printf '%s\n' "$raw_sessions" | sort)
    printf '%s\n' "$sessions"
else
    echo "(none running)"
    sessions=""
fi
echo ""

read -rp "Attach to (session name or press Enter for latest): " target
if [ -z "$target" ]; then
    if [ -n "$sessions" ]; then
        # Default to the last line, which is the most recent session
        target=$(printf '%s\n' "$sessions" | tail -n 1)
    fi
fi

if [ -z "$target" ]; then
    echo "No active thesis sessions found."
    exit 1
fi

# Verify the target session actually exists before attempting to attach
if ! tmux has-session -t "$target" 2>/dev/null; then
    echo "Error: Session '$target' does not exist or has already terminated."
    exit 1
fi

echo "Attaching to: $target..."
tmux attach -t "$target"
