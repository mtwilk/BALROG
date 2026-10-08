"""Run a model x benchmark grid of BALROG evaluations and build a contingency table.

Each cell runs `eval.py` for one model on one benchmark (a BALROG environment) using Hydra
overrides, then reads the summary.json that eval.py writes. Per-cell outputs go to
<output_root>/cells/<model>/<benchmark>/, and the table is written to
<output_root>/contingency.md, <output_root>/contingency.csv and <output_root>/results.json.

Cells whose model is still a placeholder (model_id starts with PLACEHOLDER) are skipped.

Usage (from the repo root, with the project environment active):
    python experiments/contingency_table.py --dry-run
    python experiments/contingency_table.py --max-parallel 2
    python experiments/contingency_table.py --force          # re-run cells that already have results
    python experiments/contingency_table.py --report-only    # rebuild the table from existing cells only
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Models (rows). client_name must match a client in balrog/client.py, e.g. openai, claude, gemini,
# aws-bedrock, vllm, nvidia, xai, openrouter. base_url is only needed for local or OpenAI-compatible servers.
MODELS = [
    # {
    #     "name": "gemini-3.5-flash-lite", 
    #     "client_name": "gemini", "model_id": 
    #     "gemini-3.5-flash-lite"
    # },
    {
        # OpenRouter free tier: 20 requests/min, 1000 requests/day. num_workers caps concurrent
        # requests well under the per-minute limit; the per-day limit can still be hit mid-run.
        "name": "nemotron-3.5-lightning",
        "client_name": "openrouter",
        "model_id": "nvidia/nemotron-3.5-lightning:free",
        "extra_overrides": ["eval.num_workers=5"],
    },
    {
        # Gemini free tier (this model): 30 requests/min, 14400 requests/day.
        "name": "gemma-4-31b-it",
        "client_name": "gemini",
        "model_id": "gemma-4-31b-it",
        "extra_overrides": ["eval.num_workers=10"],
    },
    # Local models: serve ONE at a time with vLLM on port 8080 (16 GB VRAM). experiments/vllm_serve.sh starts it.
    {
        "name": "qwen3-8b-fp8",
        "client_name": "vllm",
        "model_id": "Qwen/Qwen3-8B-FP8",
        "base_url": "http://localhost:8080/v1",
    },
    {
        "name": "qwen3-14b-awq",
        "client_name": "vllm",
        "model_id": "Qwen/Qwen3-14B-AWQ",
        "base_url": "http://localhost:8080/v1",
    },
    {
        # phi-4 doesn't reason, so it needs far fewer output tokens than Qwen3's 6144 default; keeping
        # that default caused every episode to overflow the 12288-token server context as the verbose
        # prose history grew (prompt + 6144 > 12288), so this overrides LOCAL_OVERRIDES down to 1024.
        "name": "phi-4-awq",
        "client_name": "vllm",
        "model_id": "stelterlab/phi-4-AWQ",
        "base_url": "http://localhost:8080/v1",
        "extra_overrides": ["client.generate_kwargs.max_tokens=1024"],
    },
]

# Benchmarks (columns). Each must be an environment name from `envs.names` in balrog/config/config.yaml.
BENCHMARKS = ["babyai", "textworld"]

# Hydra overrides applied to every cell.
COMMON_OVERRIDES = [
    "agent.type=naive",
    "agent.max_image_history=0",
    "agent.max_text_history=16",
    "eval.num_episodes.babyai=1",
    "eval.num_episodes.textworld=1",
    'tasks.babyai_tasks=["BabyAI-MixedTrainLocal-v0/goto","BabyAI-MixedTrainLocal-v0/pickup"]',
    'tasks.textworld_tasks=["coin_collector"]',
    "eval.num_workers=16",
]

# Per-episode step cap, by benchmark: there's no per-benchmark field in BALROG's own config (unlike
# num_episodes), so this is applied per cell based on which benchmark is running. 64 for babyai is a
# safety cap (some tasks allow up to 128); 80 for textworld matches its own config default exactly.
MAX_STEPS_PER_BENCHMARK = {
    "babyai": 64,
    "textworld": 80,
}

# Extra overrides for local vLLM models. Qwen3 reasons before answering, so the output budget must be large;
# the server (vllm_serve.sh) must have a context of at least prompt + this budget.
LOCAL_OVERRIDES = [
    "client.generate_kwargs.max_tokens=6144",
]


def slugify(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def is_placeholder(model):
    return model["model_id"].startswith("PLACEHOLDER")


def cell_dir(output_root, model, benchmark):
    return output_root / "cells" / slugify(model["name"]) / benchmark


def build_command(model, benchmark, out_rel, extra_overrides):
    # Hydra rejects a key given twice, so later layers replace earlier ones with the same key:
    # COMMON_OVERRIDES, then LOCAL_OVERRIDES (vllm only), then the model's own overrides, then
    # --override from the CLI (handled below, since it's also appended to the command separately).
    defaults = {}
    for o in COMMON_OVERRIDES:
        defaults[o.split("=", 1)[0]] = o
    if benchmark in MAX_STEPS_PER_BENCHMARK:
        defaults["eval.max_steps_per_episode"] = f"eval.max_steps_per_episode={MAX_STEPS_PER_BENCHMARK[benchmark]}"
    if model["client_name"] == "vllm":
        for o in LOCAL_OVERRIDES:
            defaults[o.split("=", 1)[0]] = o
    for o in model.get("extra_overrides", []):
        defaults[o.split("=", 1)[0]] = o
    extra_keys = {o.split("=", 1)[0] for o in extra_overrides}
    common = [o for k, o in defaults.items() if k not in extra_keys]
    overrides = [
        *common,
        f"client.client_name={model['client_name']}",
        f"client.model_id='{model['model_id']}'",
        f"envs.names={benchmark}",
        f"eval.output_dir={out_rel}",
    ]
    if model.get("base_url"):
        overrides.append(f"client.base_url='{model['base_url']}'")
    return [sys.executable, "eval.py", *overrides, *extra_overrides]


def find_summary(out_dir):
    """Return the newest summary.json eval.py wrote under out_dir (eval.py nests it in a run folder)."""
    summaries = sorted(out_dir.glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
    return summaries[-1] if summaries else None


def read_result(summary_path, benchmark):
    with open(summary_path) as f:
        summary = json.load(f)
    env = summary.get("environments", {}).get(benchmark)
    if env is None:
        return None
    return {
        "progression_percentage": env["progression_percentage"],
        "standard_error": env["standard_error"],
        "episodes_played": env["episodes_played"],
    }


def run_cell(model, benchmark, output_root, extra_overrides, force, dry_run, report_only=False):
    out_dir = cell_dir(output_root, model, benchmark)
    # eval.py runs from REPO_ROOT, so the output path is given relative to it.
    out_rel = Path(os.path.relpath(out_dir, REPO_ROOT)).as_posix()
    cmd = build_command(model, benchmark, out_rel, extra_overrides)
    base = {"model": model["name"], "benchmark": benchmark, "output_dir": str(out_dir)}

    if dry_run:
        status = "dry-run (placeholder model)" if is_placeholder(model) else "dry-run"
        return {**base, "status": status, "command": " ".join(cmd)}

    if report_only:
        # Build the table from whatever is already on disk; never runs eval.py.
        existing = find_summary(out_dir) if out_dir.exists() else None
        result = read_result(existing, benchmark) if existing else None
        if result:
            return {**base, "status": "ok", **result}
        return {**base, "status": "not run"}

    if is_placeholder(model):
        return {**base, "status": "skipped (placeholder model)"}

    existing = find_summary(out_dir) if out_dir.exists() else None
    if existing and not force:
        result = read_result(existing, benchmark)
        if result:
            return {**base, "status": "cached", **result}

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stdout.log", "w") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        return {**base, "status": f"error (exit {proc.returncode}, see stdout.log)"}

    summary = find_summary(out_dir)
    result = read_result(summary, benchmark) if summary else None
    if result is None:
        return {**base, "status": "error (no result for benchmark)"}
    return {**base, "status": "ok", **result}


def format_cell(result):
    if result is None:
        return "—"
    if result["status"] in ("ok", "cached"):
        return (
            f"{result['progression_percentage']:.1f} ± {result['standard_error']:.1f} "
            f"(n={result['episodes_played']})"
        )
    return result["status"]


def write_outputs(output_root, results, models, benchmarks):
    output_root.mkdir(parents=True, exist_ok=True)
    by_key = {(r["model"], r["benchmark"]): r for r in results}
    model_names = [m["name"] for m in models]

    with open(output_root / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    fields = ["model", "benchmark", "status", "progression_percentage", "standard_error", "episodes_played"]
    with open(output_root / "contingency.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    lines = [
        "| Model | " + " | ".join(benchmarks) + " |",
        "|" + "---|" * (len(benchmarks) + 1),
    ]
    for name in model_names:
        cells = [format_cell(by_key.get((name, b))) for b in benchmarks]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    with open(output_root / "contingency.md", "w") as f:
        f.write("Progression (%) ± standard error, number of episodes in parentheses.\n\n")
        f.write(table)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="experiments/results/contingency", help="Where to write results")
    parser.add_argument("--max-parallel", type=int, default=1, help="Number of cells to run at the same time")
    parser.add_argument("--force", action="store_true", help="Re-run cells that already have a summary")
    parser.add_argument("--dry-run", action="store_true", help="Print the eval.py commands without running them")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Build the table from existing cell results only; never runs eval.py",
    )
    parser.add_argument(
        "--override", action="append", default=[], help="Extra Hydra override applied to every cell (repeatable)"
    )
    parser.add_argument("--models", help="Comma-separated model names to run (default: all)")
    parser.add_argument("--benchmarks", help="Comma-separated benchmark names to run (default: all)")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    models = [m for m in MODELS if not args.models or m["name"] in args.models.split(",")]
    benchmarks = [b for b in BENCHMARKS if not args.benchmarks or b in args.benchmarks.split(",")]
    if not models or not benchmarks:
        parser.error("--models / --benchmarks matched nothing")

    jobs = [(m, b) for m in models for b in benchmarks]

    def work(job):
        model, benchmark = job
        result = run_cell(
            model, benchmark, output_root, args.override, args.force, args.dry_run, args.report_only
        )
        print(f"[{model['name']} / {benchmark}] {result['status']}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as pool:
        results = list(pool.map(work, jobs))

    if args.dry_run:
        for r in results:
            print(r["command"])
        return

    table = write_outputs(output_root, results, models, benchmarks)
    print()
    print(table)
    print(f"Written to {output_root}")


if __name__ == "__main__":
    main()
