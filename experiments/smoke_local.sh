#!/usr/bin/env bash
# Smoke test for the local models: for each model, start vLLM in WSL, run one babyai episode
# (five steps), then stop vLLM. Only one model is served at a time (16 GB VRAM).
# Output: experiments/results/smoke_local/
set -euo pipefail
export MSYS_NO_PATHCONV=1

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTRO="Ubuntu"

# Path to vllm_serve.sh as seen from WSL.
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
    bash "$DIR/run_contingency.sh" \
        --models "$name" \
        --benchmarks babyai \
        --output-root experiments/results/smoke_local \
        --override eval.num_workers=1 \
        --override eval.num_episodes.babyai=1 \
        --override eval.max_steps_per_episode=5 \
        --override 'tasks.babyai_tasks=["BabyAI-MixedTrainLocal-v0/goto"]' || true
    wsl.exe -d "$DISTRO" -e bash "$SERVE_WSL" stop
done
