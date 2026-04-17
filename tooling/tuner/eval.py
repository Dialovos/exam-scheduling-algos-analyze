"""Trial execution and multi-seed/multi-dataset evaluation.

Every function here is worker-safe (no shared mutable state) so they can
be called from ``ProcessPoolExecutor`` or ``ThreadPoolExecutor``. The
scoring convention is **lower is better, infeasible always loses** — a
single hard violation inflates the score by 10^9 regardless of soft cost.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from algorithms.cpp_bridge import run_chain as _bridge_run_chain


def run_chain(binary, dataset, chain_steps, seed, work_dir, timeout_per_step=300):
    """Adapter: forwards to cpp_bridge.run_chain.

    The `binary` arg is ignored — the bridge auto-discovers the binary path.
    Kept for backward compatibility with existing call sites in this module.
    """
    return _bridge_run_chain(
        dataset=dataset,
        chain_steps=chain_steps,
        seed=seed,
        work_dir=work_dir,
        timeout_per_step=timeout_per_step,
    )


def run_single_algo(binary, dataset, algo, params, seed, work_dir, timeout=300):
    """Run one algorithm via C++ binary. Returns result dict or None."""
    os.makedirs(work_dir, exist_ok=True)
    cmd = [binary, dataset, '--algo', algo, '--seed', str(seed),
           '--output-dir', work_dir]
    for k, v in params.items():
        cmd.extend(['--' + k.replace('_', '-'), str(int(v))])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
        return data[0] if data else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
        return None


def compute_score(result):
    """Lower is better. Feasibility-first — hard violations dominate soft."""
    if result is None:
        return float('inf')
    hard = result.get('hard_violations', 0)
    soft = result.get('soft_penalty', 0)
    if hard > 0:
        return 1e9 + hard * 1e6 + soft
    return float(soft)


def eval_on_datasets(binary, datasets, algo, params, seed, work_dir,
                     baselines, timeout=300):
    """Run algo on multiple datasets, return geometric mean of normalized scores.

    baselines: {dataset_path: best_known_score}
    Returns: geometric mean of (score / baseline) across datasets.
    """
    log_scores = []
    for ds in datasets:
        ds_name = Path(ds).stem
        ds_dir = os.path.join(work_dir, ds_name)
        result = run_single_algo(binary, ds, algo, params, seed, ds_dir, timeout)
        raw = compute_score(result)
        bl = baselines.get(ds, 1.0)
        if bl <= 0 or bl >= 1e9:
            bl = max(raw, 1.0)
        norm = raw / bl if raw < 1e9 else 1e6
        log_scores.append(math.log(max(norm, 1e-6)))
    if not log_scores:
        return float('inf')
    return math.exp(sum(log_scores) / len(log_scores))


def eval_chain_on_datasets(binary, datasets, chain_steps, seed, work_dir,
                           baselines, timeout_per_step=300):
    """Run chain on multiple datasets, return geometric mean of normalized scores."""
    log_scores = []
    for ds in datasets:
        ds_name = Path(ds).stem
        ds_dir = os.path.join(work_dir, ds_name)
        result = run_chain(binary, ds, chain_steps, seed, ds_dir, timeout_per_step)
        raw = compute_score(result)
        bl = baselines.get(ds, 1.0)
        if bl <= 0 or bl >= 1e9:
            bl = max(raw, 1.0)
        norm = raw / bl if raw < 1e9 else 1e6
        log_scores.append(math.log(max(norm, 1e-6)))
    if not log_scores:
        return float('inf')
    return math.exp(sum(log_scores) / len(log_scores))


def eval_multi_seed(binary, dataset, algo, params, seeds, work_dir, timeout=300):
    """Run algo with multiple fixed seeds, return median score.

    Median is more robust to outliers than mean — a single lucky/unlucky
    seed won't dominate the result. Seeds are evaluated in parallel threads.
    """
    def _run(s):
        sd = os.path.join(work_dir, f's{s}')
        return compute_score(run_single_algo(binary, dataset, algo, params, s, sd, timeout))

    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        scores = sorted(pool.map(_run, seeds))
    return scores[len(scores) // 2]  # median


def eval_multi_seed_datasets(binary, datasets, algo, params, seeds, work_dir,
                             baselines, timeout=300):
    """Run algo on multiple datasets x multiple seeds.

    For each dataset: take median across seeds, then compute geometric mean
    of normalized medians across datasets. All (dataset, seed) pairs run in
    parallel threads.
    """
    jobs = [(ds, s) for ds in datasets for s in seeds]

    def _run(pair):
        ds, s = pair
        sd = os.path.join(work_dir, Path(ds).stem, f's{s}')
        return ds, s, compute_score(run_single_algo(binary, ds, algo, params, s, sd, timeout))

    results_by_ds = defaultdict(list)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for ds, s, score in pool.map(_run, jobs):
            results_by_ds[ds].append(score)

    log_norms = []
    for ds in datasets:
        ds_scores = sorted(results_by_ds[ds])
        raw = ds_scores[len(ds_scores) // 2]  # median
        bl = baselines.get(ds, 1.0)
        if bl <= 0 or bl >= 1e9:
            bl = max(raw, 1.0)
        norm = raw / bl if raw < 1e9 else 1e6
        log_norms.append(math.log(max(norm, 1e-6)))
    if not log_norms:
        return float('inf')
    return math.exp(sum(log_norms) / len(log_norms))


def eval_chain_multi_seed(binary, dataset, chain_steps, seeds, work_dir,
                          timeout_per_step=300):
    """Run chain with multiple fixed seeds in parallel, return median score."""
    def _run(s):
        sd = os.path.join(work_dir, f's{s}')
        return compute_score(run_chain(binary, dataset, chain_steps, s, sd, timeout_per_step))

    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        scores = sorted(pool.map(_run, seeds))
    return scores[len(scores) // 2]


def eval_chain_multi_seed_datasets(binary, datasets, chain_steps, seeds,
                                   work_dir, baselines, timeout_per_step=300):
    """Run chain on multiple datasets x multiple seeds, return geomean of medians.

    All (dataset, seed) pairs run in parallel threads.
    """
    jobs = [(ds, s) for ds in datasets for s in seeds]

    def _run(pair):
        ds, s = pair
        sd = os.path.join(work_dir, Path(ds).stem, f's{s}')
        return ds, s, compute_score(run_chain(binary, ds, chain_steps, s, sd, timeout_per_step))

    results_by_ds = defaultdict(list)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for ds, s, score in pool.map(_run, jobs):
            results_by_ds[ds].append(score)

    log_norms = []
    for ds in datasets:
        ds_scores = sorted(results_by_ds[ds])
        raw = ds_scores[len(ds_scores) // 2]
        bl = baselines.get(ds, 1.0)
        if bl <= 0 or bl >= 1e9:
            bl = max(raw, 1.0)
        norm = raw / bl if raw < 1e9 else 1e6
        log_norms.append(math.log(max(norm, 1e-6)))
    if not log_norms:
        return float('inf')
    return math.exp(sum(log_norms) / len(log_norms))
