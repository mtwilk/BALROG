# Contingency experiment

Runs a grid of models × benchmarks (babyai, textworld) and writes a table of results.

Models are listed in `MODELS` in `contingency_table.py`. Benchmarks are in `BENCHMARKS`. Episode counts and tasks are in `COMMON_OVERRIDES`: currently 1 episode each of babyai `goto` + `pickup` and textworld `coin_collector`, plus a per-episode step cap (`MAX_STEPS_PER_BENCHMARK`: 64 for babyai, 80 for textworld, matching textworld's own config default) so no single episode can run away. See the rate-limit and step-budget notes below for why.

## Models

- **API:** Nemotron 3.5 Lightning (OpenRouter, free), Gemma 4 31B (Gemini API).
- **Local** (vLLM, one at a time — `experiments/vllm_serve.sh start <repo-id>` downloads weights to the Hugging Face cache on first run):
  - [Qwen3-8B-FP8](https://huggingface.co/Qwen/Qwen3-8B-FP8) — `hf download Qwen/Qwen3-8B-FP8`
  - [Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ) — `hf download Qwen/Qwen3-14B-AWQ`
  - [phi-4-AWQ](https://huggingface.co/stelterlab/phi-4-AWQ) (community quant) — `hf download stelterlab/phi-4-AWQ`

## Setup

- **API models:** `GEMINI_API_KEY` and `OPENROUTER_API_KEY` in `SECRETS` (repo root).
- **Local models:** vLLM in `~/vllm-venv` inside WSL, and the weights downloaded with `hf download` (cache in `~/.cache/huggingface`).
- **Project:** `~/balrog-venv` inside WSL. Everything runs through WSL.

## Run the whole experiment

```
bash experiments/run_experiment.sh
```

Runs the API models directly, then serves each local model with vLLM in turn (start, run, stop), then rebuilds the combined table from whatever is on disk. If one model errors out (quota, a crashed server), the script keeps going with the rest. Output: `experiments/results/main/contingency.md` (and `.csv`, `results.json`).

## Run a subset

`run_contingency.sh` is the lower-level tool `run_experiment.sh` is built on. It starts WSL if needed and runs `contingency_table.py` there.

```
bash experiments/run_contingency.sh --models nemotron-3.5-lightning --benchmarks babyai
```

Useful options:

- `--models a,b` and `--benchmarks x,y` run a subset.
- `--override key=value` changes a setting for every cell.
- `--force` re-runs cells that already have results.
- `--dry-run` prints the commands without running them.
- `--report-only` rebuilds the table from existing cells without running anything (what `run_experiment.sh` uses at the end).

The local models need their vLLM server running first; see `vllm_serve.sh` below. Running them without a server fails.

## Smoke tests

Each runs one babyai episode of five steps per model, to check that the models respond.

- **API models:** `bash experiments/smoke_api.sh` (output in `experiments/results/smoke_api/`).
- **Local models:** `bash experiments/smoke_local.sh` starts vLLM for each model in turn, runs the test, and stops the server (output in `experiments/results/smoke_local/`). It takes a while, since each model has to load.
- **Local server only:** `experiments/vllm_serve.sh start <hf-repo-id>` and `experiments/vllm_serve.sh stop` (run inside WSL).

## Notes

- **Free API quotas** (set in each model's `extra_overrides` in `MODELS`, as a lower `eval.num_workers` to stay under the per-minute limit):
  - OpenRouter (Nemotron 3.5 Lightning): 20 requests/min, 1000/day → `num_workers=5`.
  - Gemini (Gemma 4 31B): 30 requests/min, 14400/day → `num_workers=10`.
  - The per-day limits aren't enforced by the script. If a cell errors out from quota, rerun it later with `--models` and `--force`.
- **Babyai's harder tasks are expensive at these rate limits, not just the free daily cap.** Each step is one more API call, and MiniGrid's step budgets vary a lot by task: `goto` finishes in a handful of steps, but `open` allows up to 128 and `pickup` up to 64. At 20 rpm, one `open` episode alone can take over an hour. Textworld's cap is a flat 80 steps for every game. That's why the default config above uses one task and one episode per benchmark; widen it carefully, and expect run time to scale with the step budgets, not just the episode count.
- The handful of "Missing choices" / "Empty content" errors seen from OpenRouter's free Nemotron endpoint are a small fraction of calls (under 2% in one run) and are not the main cause of slow runs — the rate limit and step budgets are.
- Only one local model fits in VRAM at a time, so run the local models sequentially.
- Qwen3 reasons before answering, so the local output budget is 6144 tokens and the server context is 12288. The Qwen3-14B and phi-4 smoke results were run with older settings (2048 tokens, 8192 context).
- phi-4 (`phi-4-awq`) answers in prose instead of one action per step, so it fails the smoke test.
- `pkill -f eval.py` stops whichever cell is currently running, but the orchestrator just logs it as failed and moves to the next cell rather than halting. To actually stop a run, kill the parent `contingency_table.py` process (inside WSL).
