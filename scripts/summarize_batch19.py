#!/usr/bin/env python3
"""Summarize batch 19: best soft per (instance, algo), winners highlighted.

Usage:
    python3 scripts/summarize_batch19.py results/batch_019_validation
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    batch_dir = Path(sys.argv[1])
    csv_path = batch_dir / "summary.csv"
    if not csv_path.exists():
        sys.exit(f"error: {csv_path} not found. Run `make batch19` first.")

    # (instance, algo) -> list of (seed, hard, soft, runtime)
    runs = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["instance"], r["algo"])
            try:
                runs[key].append((
                    int(r["seed"]),
                    int(r["hard"]) if r["hard"].isdigit() else -1,
                    int(r["soft"]) if r["soft"].isdigit() else -1,
                    float(r["runtime_sec"]),
                ))
            except (ValueError, KeyError):
                continue

    # Best per instance: feasible first, then lowest soft
    per_inst_best = defaultdict(lambda: (None, 10**9))
    for (inst, algo), trials in runs.items():
        feasible = [t for t in trials if t[1] == 0]
        if feasible:
            best_s = min(t[2] for t in feasible)
            if best_s < per_inst_best[inst][1]:
                per_inst_best[inst] = (algo, best_s)

    print(f"batch 19 summary — {csv_path}\n")
    instances = sorted({k[0] for k in runs})
    algos = sorted({k[1] for k in runs})

    header = f"{'instance':<24}" + "".join(f"{a:>18}" for a in algos) + f"{'WINNER':>22}"
    print(header)
    print("-" * len(header))

    for inst in instances:
        row = f"{inst:<24}"
        for algo in algos:
            trials = runs.get((inst, algo), [])
            feas = [t for t in trials if t[1] == 0]
            if feas:
                best_s = min(t[2] for t in feas)
                mean_rt = sum(t[3] for t in feas) / len(feas)
                mark = "*" if (per_inst_best[inst][0] == algo) else " "
                cell = f"{mark}{best_s:>6} ({mean_rt:>4.1f}s)"
            elif trials:
                cell = f" infeas({len(trials)})  "
            else:
                cell = f"       -"
            row += f"{cell:>18}"
        w = per_inst_best[inst]
        row += f"{w[0] or '-':>12}→{w[1]:>6}"
        print(row)

    print("\n(* = winner on that instance. Mean wall-clock in parens over feasible seeds.)")


if __name__ == "__main__":
    main()
