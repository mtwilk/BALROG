#!/usr/bin/env bash
# Start WSL (Ubuntu) if it is not running, then run experiments/contingency_table.py
# inside the WSL venv (~/balrog-venv). Any arguments are passed to the Python script.
#
# Usage (from Git Bash, in the repo root or anywhere):
#   bash experiments/run_contingency.sh --dry-run
#   bash experiments/run_contingency.sh --max-parallel 2

set -euo pipefail

# Stop Git Bash from rewriting /unix/style arguments passed to wsl.exe.
export MSYS_NO_PATHCONV=1

DISTRO="Ubuntu"
VENV_PYTHON='$HOME/balrog-venv/bin/python'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_WIN="$(cygpath -w "$SCRIPT_DIR/..")"

if ! wsl.exe -l --running -q 2>/dev/null | tr -d '\0\r' | grep -qx "$DISTRO"; then
    echo "Starting WSL distro '$DISTRO'..."
    wsl.exe -d "$DISTRO" -e true
fi

REPO_WSL="$(wsl.exe -d "$DISTRO" -e wslpath -a "$REPO_WIN" | tr -d '\r')"

wsl.exe -d "$DISTRO" -e bash -c \
    'cd "$1" && shift && exec '"$VENV_PYTHON"' experiments/contingency_table.py "$@"' \
    _ "$REPO_WSL" "$@"
