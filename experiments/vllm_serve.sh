#!/usr/bin/env bash
# Start or stop the vLLM server for the local models. Runs inside WSL (venv: ~/vllm-venv).
#
#   vllm_serve.sh start <hf-repo-id>   start the server on port 8080 and wait until it answers
#   vllm_serve.sh stop                 stop the server

set -euo pipefail

VLLM="$HOME/vllm-venv/bin/vllm"
LOG="$HOME/vllm_server.log"

case "${1:-}" in
    start)
        MODEL="${2:?usage: vllm_serve.sh start <hf-repo-id>}"
        pkill -f "[v]llm serve" 2>/dev/null || true
        sleep 3

        # Qwen3 thinks by default; the reasoning parser keeps that out of the action text.
        PARSER=()
        case "$MODEL" in *Qwen3*) PARSER=(--reasoning-parser qwen3) ;; esac

        # FlashInfer sampler and DeepGEMM would compile CUDA kernels, which needs nvcc (not installed in WSL).
        # Both are optional, so turn them off. Images are off because the runs are text-only.
        (
            VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 nohup "$VLLM" serve "$MODEL" \
                --port 8080 --max-model-len 12288 --gpu-memory-utilization 0.9 \
                --limit-mm-per-prompt '{"image":0,"video":0}' "${PARSER[@]}" \
                > "$LOG" 2>&1 &
        )

        for _ in $(seq 1 60); do
            if curl -s -m 3 localhost:8080/v1/models >/dev/null 2>&1; then
                echo "vLLM up: $MODEL"
                exit 0
            fi
            if grep -q "Engine core initialization failed" "$LOG" 2>/dev/null; then
                echo "vLLM failed to start for $MODEL; last log lines:" >&2
                tail -n 15 "$LOG" >&2
                exit 1
            fi
            sleep 15
        done
        echo "vLLM did not answer in time; see $LOG" >&2
        exit 1
        ;;
    stop)
        pkill -f "[v]llm serve" 2>/dev/null || true
        echo "vLLM stopped"
        ;;
    *)
        echo "usage: vllm_serve.sh start <hf-repo-id> | stop" >&2
        exit 2
        ;;
esac
