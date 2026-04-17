#!/usr/bin/env python3
"""1-D parameter sensitivity sweep — one knob at a time, all other knobs pinned.

Sweeps every algo in :data:`tooling.tuner.search_spaces.SEARCH_SPACES`:
for each algo's knobs, runs 3 values (min / mid / max from the declared range)
across N seeds on a fixed dataset. Writes a long-form CSV compatible with
``utils.plots.tuning.plot_parameter_sensitivity``.

Usage:
    python -m tooling.param_sweep --dataset instances/synthetic_scaling_1000.exam
    python -m tooling.param_sweep --dataset <ds> --seeds 3 --out results/sweep.csv
    python -m tooling.param_sweep --algos tabu,sa --seeds 2      # smoke

Output columns:
    algorithm, param_col, param_value, dataset, seed,
    soft_penalty, runtime, feasible, hard
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tooling.tuned_params import load_params_flat
from tooling.tuner.search_spaces import SEARCH_SPACES


def _sample_values(lo: int | float, hi: int | float, scale: str, n: int = 3) -> list:
    """Pick `n` knob values spanning [lo, hi]. Log scale for ranges, int for counts."""
    if n < 2:
        return [lo]
    if scale == 'log':
        vals = []
        lg_lo, lg_hi = math.log10(max(lo, 1)), math.log10(max(hi, 1))
        for i in range(n):
            t = i / (n - 1)
            vals.append(int(round(10 ** (lg_lo + (lg_hi - lg_lo) * t))))
        # deduplicate while preserving order
        seen, out = set(), []
        for v in vals:
            if v not in seen:
                seen.add(v); out.append(v)
        return out
    # linear int (uniform)
    return sorted({int(round(lo + (hi - lo) * i / (n - 1))) for i in range(n)})


def _cli_flag_name(knob: str) -> str:
    """'tabu_iters' → '--tabu-iters'."""
    return '--' + knob.replace('_', '-')


def _run_once(dataset: str, algo: str, knob: str, value: int,
              seed: int, pinned: dict, out_root: Path) -> dict:
    """Execute main.py with one knob overridden, parse summary.json, return row."""
    run_dir = out_root / f'{algo}_{knob}_{value}_seed{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = ['python', 'main.py',
           '--dataset', dataset, '--algo', algo,
           '--seed', str(seed),
           '--no-batch', '--quiet',
           '--output', str(run_dir)]
    # pin the other knobs on this algo at tuned defaults
    for k, v in pinned.items():
        if k == knob:
            continue
        cmd += [_cli_flag_name(k), str(v)]
    # override the swept knob
    cmd += [_cli_flag_name(knob), str(value)]

    started = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    wall = time.time() - started
    row = {
        'algorithm': algo, 'param_col': knob, 'param_value': value,
        'dataset': Path(dataset).stem, 'seed': seed,
        'soft_penalty': None, 'runtime': None, 'feasible': None, 'hard': None,
        'status': 'ok' if r.returncode == 0 else f'rc={r.returncode}',
        'wall': round(wall, 2),
    }
    summ_file = run_dir / 'summary.json'
    if summ_file.exists():
        try:
            summ = json.loads(summ_file.read_text())
            if summ:
                # there's exactly one entry per single-algo run
                rec = next(iter(summ.values()))
                row.update({
                    'soft_penalty': rec.get('soft'),
                    'runtime':     rec.get('runtime'),
                    'feasible':    rec.get('feasible'),
                    'hard':        rec.get('hard'),
                })
        except (json.JSONDecodeError, OSError) as e:
            row['status'] = f'parse-fail: {e}'
    if r.returncode != 0 and row['status'] == 'ok':
        row['status'] = 'nonzero-exit'
    return row


def build_jobs(algos: list[str], seeds: list[int], dataset: str,
               values_per_knob: int) -> list[tuple]:
    tuned = load_params_flat()
    jobs = []
    for algo in algos:
        space = SEARCH_SPACES.get(algo)
        if not space:
            print(f'[skip] {algo}: no SEARCH_SPACES entry', file=sys.stderr)
            continue
        pinned_all = {k: v for k, v in tuned.items() if k in space}
        for knob, (lo, hi, scale) in space.items():
            values = _sample_values(lo, hi, scale, n=values_per_knob)
            pinned = {k: v for k, v in pinned_all.items() if k != knob}
            for v in values:
                for s in seeds:
                    jobs.append((dataset, algo, knob, v, s, pinned))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True,
                    help='Instance to sweep on (e.g. instances/synthetic_scaling_1000.exam)')
    ap.add_argument('--seeds', type=int, default=3, help='Seeds per cell (default: 3)')
    ap.add_argument('--values', type=int, default=3,
                    help='Knob values per axis (default: 3 = min/mid/max)')
    ap.add_argument('--algos', default='',
                    help='Comma list (default: every algo in SEARCH_SPACES)')
    ap.add_argument('--out', default='results/colab_batch_sweep/sensitivity.csv',
                    help='Output CSV path')
    ap.add_argument('--runs-dir', default='results/colab_batch_sweep/runs',
                    help='Where to drop per-run output folders')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help='Parallel subprocesses (default: cpu_count-1)')
    args = ap.parse_args()

    algos = [a.strip() for a in args.algos.split(',') if a.strip()] \
            if args.algos else sorted(SEARCH_SPACES.keys())
    seeds = list(range(42, 42 + args.seeds))
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    runs_root = Path(args.runs_dir); runs_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(algos, seeds, args.dataset, args.values)
    total = len(jobs)
    print(f'Sweep: {len(algos)} algos, {args.values} values/knob × {args.seeds} seeds '
          f'→ {total} runs on {Path(args.dataset).stem}')
    print(f'Workers: {args.workers}   Output CSV: {out_path}')

    cols = ['algorithm', 'param_col', 'param_value', 'dataset', 'seed',
            'soft_penalty', 'runtime', 'feasible', 'hard', 'status', 'wall']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        started = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_run_once, ds, al, kn, va, se, pi, runs_root): (al, kn, va, se)
                    for (ds, al, kn, va, se, pi) in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                w.writerow(row)
                f.flush()
                done += 1
                elapsed = time.time() - started
                eta = elapsed / done * (total - done)
                print(f'  [{done:>3}/{total}] {row["algorithm"]:<6} '
                      f'{row["param_col"]:<16}={row["param_value"]:<7} '
                      f'seed={row["seed"]}  soft={row["soft_penalty"]} '
                      f'rt={row["runtime"]}  elapsed={elapsed/60:.1f}m  '
                      f'eta={eta/60:.1f}m  {row["status"]}', flush=True)

    print(f'\nDone. Wrote {total} rows to {out_path}')


if __name__ == '__main__':
    main()
