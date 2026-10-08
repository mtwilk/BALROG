#!/usr/bin/env bash
# Smoke test for the API models: one babyai episode, five steps, per model.
# Output: experiments/results/smoke_api/
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$DIR/run_contingency.sh" \
    --models nemotron-3.5-lightning,gemma-4-31b-it \
    --benchmarks babyai \
    --output-root experiments/results/smoke_api \
    --override eval.num_workers=1 \
    --override eval.num_episodes.babyai=1 \
    --override eval.max_steps_per_episode=5 \
    --override 'tasks.babyai_tasks=["BabyAI-MixedTrainLocal-v0/goto"]'
