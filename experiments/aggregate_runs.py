"""Join several runs of the contingency experiment into one CSV.

Each run is an output root made by contingency_table.py (cells/<model>/<benchmark>/<run folder>/...).
For every model x benchmark, the newest run folder with a summary.json is used, and its per-episode
results are read, so the pooled numbers are computed over episodes (not averages of averages).

Usage (inside WSL, from the repo root):
    python experiments/aggregate_runs.py run1_dir run2_dir [...] --out experiments/results/joint_contingency.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path


def newest_run_dir(cell):
    runs = sorted(cell.glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
    return runs[-1].parent if runs else None


def episode_progressions(run_dir):
    values = []
    for f in run_dir.rglob("*_run_*.json"):
        with open(f) as fh:
            data = json.load(fh)
        if "progression" in data:
            values.append(data["progression"])
    return values


def mean_se(values):
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return mean, sd / math.sqrt(n)


def pct(x):
    return "" if x is None else round(100 * x, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Output roots, in run order")
    parser.add_argument("--out", default="experiments/results/joint_contingency.csv")
    args = parser.parse_args()

    roots = [Path(r) for r in args.runs]
    cells = sorted({(m.name, b.name) for r in roots for m in (r / "cells").iterdir() for b in m.iterdir()})

    header = ["model", "benchmark"]
    for i in range(len(roots)):
        header += [f"run{i + 1}_progress_pct", f"run{i + 1}_episodes"]
    header += ["pooled_progress_pct", "pooled_se_pct", "pooled_episodes"]

    rows = []
    for model, bench in cells:
        row = [model, bench]
        pooled = []
        for root in roots:
            run_dir = newest_run_dir(root / "cells" / model / bench)
            vals = episode_progressions(run_dir) if run_dir else []
            mean, _ = mean_se(vals)
            row += [pct(mean), len(vals)]
            pooled += vals
        mean, se = mean_se(pooled)
        row += [pct(mean), pct(se), len(pooled)]
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {out}")
    print(",".join(header))
    for r in rows:
        print(",".join(str(x) for x in r))


if __name__ == "__main__":
    main()
