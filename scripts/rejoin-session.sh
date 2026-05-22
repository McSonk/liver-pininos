#!/bin/bash
echo "Active training sessions:"
tmux list-sessions -F "#{session_name}" 2>/dev/null | grep thesis_train | tac || echo "(none running)"
echo ""

read -rp "Attach to (session name or press Enter for latest): " target
if [ -z "$target" ]; then
    target=$(tmux list-sessions -F "#{session_name}" | grep thesis_train | tail -n 1)
fi

if [ -z "$target" ]; then
    echo "No training sessions found."
    exit 1
fi

echo "Attaching to: $target..."
tmux attach -t "$target"
