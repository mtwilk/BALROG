#!/usr/bin/env bash
# Task 2 and Task 3 
# TextWorld experiment for the three harnesses: naive, symbolic and self_refine agents
# coin_collector and treasure_hunter, on the games selected by each seed. One episode per game, two workers.
# Usage (macOS):
# MODEL=gpt-6-luna TAG=luna bash experiments/run_model_suite.sh
# MODEL=gpt-6-luna TAG=luna AGENTS="self_refine" SEEDS=1 bash experiments/run_model_suite.sh
# MODEL=qwen3-14b-awq TAG=qwen14b EXTRA="client.generate_kwargs.max_tokens=4096" bash experiments/run_model_suite.sh
set -euo pipefail

MODEL="${MODEL:?set MODEL to a name from MODELS in contingency_table.py}"
TAG="${TAG:?set TAG, used in the output folder names}"
AGENTS="${AGENTS:-naive symbolic self_refine}"
SEEDS="${SEEDS:-0 1}"
TASKS='["coin_collector","treasure_hunter"]'
EXTRA_ARGS=()
for override in ${EXTRA:-}; do
    EXTRA_ARGS+=(--override "$override")
done

# Fix for macOS fork issues
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Output: experiments/results/<agent>_<TAG>_seed<N>/
for agent in $AGENTS; do
    for seed in $SEEDS; do
        echo "=== $MODEL, $agent, seed $seed"
        python experiments/contingency_table.py \
            --models "$MODEL" --benchmarks textworld \
            --output-root "experiments/results/${agent}_${TAG}_seed$seed" \
            --override agent.type="$agent" \
            --override eval.num_episodes.textworld=1 \
            --override "tasks.textworld_tasks=$TASKS" \
            --override envs.env_kwargs.seed="$seed" \
            --override eval.num_workers=2 \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
            --force
    done
done
