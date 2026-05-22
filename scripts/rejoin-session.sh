#!/bin/bash
echo "Active training sessions:"

if ! command -v tmux >/dev/null 2>&1; then
    echo "(none running)"
    echo ""
    echo "tmux is not installed."
    exit 1
fi

sessions=$(tmux list-sessions -F "#{session_name}" 2>/dev/null | grep thesis_train || true)
if [ -n "$sessions" ]; then
    printf '%s\n' "$sessions" | tac
else
    echo "(none running)"
fi
echo ""

read -rp "Attach to (session name or press Enter for latest): " target
if [ -z "$target" ]; then
    target=$(printf '%s\n' "$sessions" | tail -n 1)
fi

if [ -z "$target" ]; then
    echo "No training sessions found."
    exit 1
fi

echo "Attaching to: $target..."
tmux attach -t "$target"
