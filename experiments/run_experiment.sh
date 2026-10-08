#!/usr/bin/env bash
# Run the full experiment: every model on babyai and textworld (every task, 3 episodes each).
#
# The API models (nemotron-3.5-lightning, gemma-4-31b-it) run directly. The local models are
# served one at a time with vLLM, since only one fits in 16 GB VRAM: for each, this starts the
# server, runs its cells, then stops the server. A final --report-only pass rebuilds the combined
# table from all the cells written along the way, without re-running anything.
#
# Output: experiments/results/main/ (contingency.md, contingency.csv, results.json)
set -euo pipefail
export MSYS_NO_PATHCONV=1

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTRO="Ubuntu"
OUT="experiments/results/main"

echo "=== API models: nemotron-3.5-lightning, gemma-4-31b-it"
bash "$DIR/run_contingency.sh" --models nemotron-3.5-lightning,gemma-4-31b-it --output-root "$OUT" || true

SERVE_WSL="$(wsl.exe -d "$DISTRO" -e wslpath -a "$(cygpath -w "$DIR/vllm_serve.sh")" | tr -d '\r')"

# name (as in MODELS in contingency_table.py) and Hugging Face repo id to serve
LOCAL_MODELS=(
    "qwen3-8b-fp8 Qwen/Qwen3-8B-FP8"
    "qwen3-14b-awq Qwen/Qwen3-14B-AWQ"
    "phi-4-awq stelterlab/phi-4-AWQ"
)

for entry in "${LOCAL_MODELS[@]}"; do
    read -r name repo <<< "$entry"
    echo "=== $name"
    if ! wsl.exe -d "$DISTRO" -e bash "$SERVE_WSL" start "$repo"; then
        echo "skipping $name: server did not start"
        wsl.exe -d "$DISTRO" -e bash "$SERVE_WSL" stop
        continue
    fi
    bash "$DIR/run_contingency.sh" --models "$name" --output-root "$OUT" || true
    wsl.exe -d "$DISTRO" -e bash "$SERVE_WSL" stop
done

echo "=== Combined table"
bash "$DIR/run_contingency.sh" --report-only --output-root "$OUT"
