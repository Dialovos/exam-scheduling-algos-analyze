#!/usr/bin/env python3
"""
Auto-Tuner for Exam Scheduling Algorithms

Combines parameter optimization + algorithm chaining with tournament-based
natural selection. Evaluates across multiple datasets to avoid overfitting.

Single-dataset mode:
    python -m tooling.auto_tuner instances/exam_comp_set4.exam

Multi-dataset (global) mode:
    python -m tooling.auto_tuner --all-sets
    python -m tooling.auto_tuner --all-sets --synthetic
    python -m tooling.auto_tuner instances/exam_comp_set1.exam instances/exam_comp_set4.exam instances/exam_comp_set7.exam

Resume from checkpoint:
    python -m tooling.auto_tuner --all-sets --resume
"""

import argparse
import glob as globmod
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
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

# ── Search Spaces ────────────────────────────────────────────
# (min, max, scale)  scale: 'log' = log-uniform, 'int' = uniform int

SEARCH_SPACES = {
    'tabu': {
        'tabu_iters':    (500,  20000, 'log'),
        'tabu_tenure':   (5,    50,    'int'),
        'tabu_patience': (50,   2000,  'log'),
    },
    'hho': {
        'hho_pop':   (10,  100,  'int'),
        'hho_iters': (50,  1000, 'log'),
    },
    'sa':    {'sa_iters':    (1000, 50000, 'log')},
    'kempe': {'kempe_iters': (500,  20000, 'log')},
    'alns':  {'alns_iters':  (500,  20000, 'log')},
    'gd':    {'gd_iters':    (1000, 50000, 'log')},
    'abc': {
        'abc_pop':   (10,  100,  'int'),
        'abc_iters': (500,  20000, 'log'),
    },
    'ga': {
        'ga_pop':   (20,  200,  'int'),
        'ga_iters': (100, 5000, 'log'),
    },
    'lahc': {
        'lahc_iters': (1000, 50000, 'log'),
        'lahc_list':  (0,    5000,  'int'),
    },
}

from tooling.tuned_params import load_params as _load_golden, FALLBACK_PARAMS

DEFAULT_PARAMS = _load_golden()

TUNABLE_ALGOS = list(SEARCH_SPACES.keys())


# ── Binary Management ────────────────────────────────────────

def find_or_build_binary():
    root = Path(__file__).parent
    binary = root / 'cpp' / 'exam_solver'
    if binary.is_file() and os.access(binary, os.X_OK):
        return str(binary)
    src = root / 'cpp' / 'main.cpp'
    if not src.is_file():
        raise RuntimeError("Cannot find cpp/main.cpp")
    print("[AutoTuner] Compiling C++ solver...")
    r = subprocess.run(
        ['g++', '-O3', '-std=c++20', '-o', str(binary), str(src)],
        capture_output=True, text=True, timeout=120,
        cwd=str(root / 'cpp'),
    )
    if r.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{r.stderr}")
    print("[AutoTuner] Compiled successfully")
    return str(binary)


# ── Trial Execution (worker-safe, no shared state) ───────────

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


def compute_score(result):
    """Lower is better. Feasibility-first."""
    if result is None:
        return float('inf')
    hard = result.get('hard_violations', 0)
    soft = result.get('soft_penalty', 0)
    if hard > 0:
        return 1e9 + hard * 1e6 + soft
    return float(soft)


# Fixed seed set for fair comparison — every config is tested on the same
# seeds so score differences reflect param quality, not seed luck.
EVAL_SEEDS = [42, 123, 789]


def eval_multi_seed(binary, dataset, algo, params, seeds, work_dir, timeout=300):
    """Run algo with multiple fixed seeds, return median score.

    Median is more robust to outliers than mean — a single lucky/unlucky
    seed won't dominate the result.
    """
    scores = []
    for s in seeds:
        sd = os.path.join(work_dir, f's{s}')
        result = run_single_algo(binary, dataset, algo, params, s, sd, timeout)
        scores.append(compute_score(result))
    scores.sort()
    return scores[len(scores) // 2]  # median


def eval_multi_seed_datasets(binary, datasets, algo, params, seeds, work_dir,
                             baselines, timeout=300):
    """Run algo on multiple datasets x multiple seeds.

    For each dataset: take median across seeds, then compute geometric mean
    of normalized medians across datasets.
    """
    log_norms = []
    for ds in datasets:
        ds_name = Path(ds).stem
        ds_scores = []
        for s in seeds:
            sd = os.path.join(work_dir, ds_name, f's{s}')
            result = run_single_algo(binary, ds, algo, params, s, sd, timeout)
            ds_scores.append(compute_score(result))
        ds_scores.sort()
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
    """Run chain with multiple fixed seeds, return median score."""
    scores = []
    for s in seeds:
        sd = os.path.join(work_dir, f's{s}')
        result = run_chain(binary, dataset, chain_steps, s, sd, timeout_per_step)
        scores.append(compute_score(result))
    scores.sort()
    return scores[len(scores) // 2]


def eval_chain_multi_seed_datasets(binary, datasets, chain_steps, seeds,
                                   work_dir, baselines, timeout_per_step=300):
    """Run chain on multiple datasets x multiple seeds, return geomean of medians."""
    log_norms = []
    for ds in datasets:
        ds_name = Path(ds).stem
        ds_scores = []
        for s in seeds:
            sd = os.path.join(work_dir, ds_name, f's{s}')
            result = run_chain(binary, ds, chain_steps, s, sd, timeout_per_step)
            ds_scores.append(compute_score(result))
        ds_scores.sort()
        raw = ds_scores[len(ds_scores) // 2]
        bl = baselines.get(ds, 1.0)
        if bl <= 0 or bl >= 1e9:
            bl = max(raw, 1.0)
        norm = raw / bl if raw < 1e9 else 1e6
        log_norms.append(math.log(max(norm, 1e-6)))
    if not log_norms:
        return float('inf')
    return math.exp(sum(log_norms) / len(log_norms))


# ── Sampling ─────────────────────────────────────────────────

def _sample_val(lo, hi, scale, rng):
    if scale == 'log':
        return int(round(math.exp(rng.uniform(math.log(max(lo, 1)),
                                               math.log(max(hi, 1))))))
    return rng.randint(lo, hi)


def sample_random(algo, rng):
    return {k: _sample_val(*v, rng) for k, v in SEARCH_SPACES.get(algo, {}).items()}


def perturb(algo, base, rng, intensity=0.3):
    space = SEARCH_SPACES.get(algo, {})
    out = dict(base)
    for name, (lo, hi, scale) in space.items():
        if rng.random() < 0.4:
            continue
        bv = out.get(name, (lo + hi) // 2)
        if scale == 'log':
            log_lo, log_hi = math.log(max(lo, 1)), math.log(max(hi, 1))
            nv = math.exp(math.log(max(bv, 1)) + rng.gauss(0, intensity * (log_hi - log_lo)))
            out[name] = int(round(max(lo, min(hi, nv))))
        else:
            nv = bv + int(round(rng.gauss(0, intensity * (hi - lo))))
            out[name] = max(lo, min(hi, nv))
    return out


def random_chain(top_algos, best_params, rng, length=None):
    if length is None:
        length = rng.randint(2, min(4, len(top_algos)))
    chain = []
    for _ in range(length):
        algo = rng.choice(top_algos)
        params = perturb(algo, best_params.get(algo, DEFAULT_PARAMS.get(algo, {})),
                         rng, intensity=0.2)
        chain.append((algo, params))
    return chain


def mutate_chain(chain, top_algos, best_params, rng):
    chain = list(chain)
    op = rng.choice(['swap', 'perturb', 'add', 'remove'])
    if op == 'swap' and chain:
        i = rng.randrange(len(chain))
        a = rng.choice(top_algos)
        p = perturb(a, best_params.get(a, DEFAULT_PARAMS.get(a, {})), rng)
        chain[i] = (a, p)
    elif op == 'perturb' and chain:
        i = rng.randrange(len(chain))
        a, p = chain[i]
        chain[i] = (a, perturb(a, p, rng))
    elif op == 'add' and len(chain) < 5:
        a = rng.choice(top_algos)
        p = perturb(a, best_params.get(a, DEFAULT_PARAMS.get(a, {})), rng)
        chain.insert(rng.randint(0, len(chain)), (a, p))
    elif op == 'remove' and len(chain) > 2:
        chain.pop(rng.randrange(len(chain)))
    return chain


# ── Checkpoint ───────────────────────────────────────────────

class Checkpoint:
    def __init__(self, path):
        self.path = path

    def save(self, state):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, self.path)

    def load(self):
        if not os.path.isfile(self.path):
            return None
        with open(self.path) as f:
            return json.load(f)


# ── Synthetic Generator ─────────────────────────────────────

def generate_synthetic_dataset(output_dir, num_exams=500, preset='competition',
                               seed=42):
    """Generate a synthetic dataset and return its path."""
    sys.path.insert(0, str(Path(__file__).parent))
    from core.generator import generate_synthetic, write_itc2007_format

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'synthetic_{preset}_{num_exams}.exam')
    if os.path.isfile(path):
        return path

    print(f"[AutoTuner] Generating synthetic ({num_exams} exams, {preset})...")
    prob = generate_synthetic(num_exams=num_exams, preset=preset, seed=seed)
    write_itc2007_format(prob, path)
    print(f"[AutoTuner] Synthetic: {path}")
    return path


# ── AutoTuner ────────────────────────────────────────────────

class AutoTuner:
    def __init__(self, datasets, output_dir='tuning_results',
                 max_workers=6, param_trials=8,
                 chain_pop=8, chain_rounds=3,
                 seed=42, ip_time_limit=120,
                 max_time=None, eval_datasets=3,
                 auto_update=True, force_update=False,
                 resume=False):
        self.datasets = [os.path.abspath(d) for d in datasets]
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.param_trials = param_trials
        self.chain_pop = chain_pop
        self.chain_rounds = chain_rounds
        self.seed = seed
        self.eval_seeds = EVAL_SEEDS
        self.rng = random.Random(seed)
        self.ip_time_limit = ip_time_limit
        self.eval_datasets_n = min(eval_datasets, len(self.datasets))
        # Wall-time budget: default 20 min for multi-dataset, 8 for single
        if max_time is None:
            self.max_time = 1200 if len(self.datasets) > 1 else 480
        else:
            self.max_time = max_time
        self.multi = len(self.datasets) > 1
        self.auto_update = auto_update
        self.force_update = force_update

        # Clean stale output on fresh runs to prevent old .sln artifacts
        # from polluting chain warm-starts or confusing results.
        if not resume and os.path.isdir(output_dir):
            for subdir in ('screen', 'param_tune', 'chains', 'final'):
                stale = os.path.join(output_dir, subdir)
                if os.path.isdir(stale):
                    shutil.rmtree(stale)
            ckpt_file = os.path.join(output_dir, 'checkpoint.json')
            if os.path.isfile(ckpt_file):
                os.remove(ckpt_file)

        os.makedirs(output_dir, exist_ok=True)
        self.ckpt = Checkpoint(os.path.join(output_dir, 'checkpoint.json'))
        self.binary = find_or_build_binary()
        self.start_time = None

        # State
        self.phase = 'init'
        self.total_trials = 0
        self.best_params = {}
        self.best_scores = {}       # algo -> aggregate score
        self.screen_raw = {}        # algo -> {dataset: raw_score}
        self.baselines = {}         # dataset -> best_raw_score across all algos
        self.param_history = []
        self.chain_history = []
        self.best_chain = None
        self.best_chain_score = float('inf')
        self.best_overall_score = float('inf')
        self.best_overall_config = None
        self.core_algos = []

        # Auto-pick representative evaluation subset
        self.eval_subset = self._pick_eval_subset()

    def _pick_eval_subset(self):
        """Pick representative datasets: small, medium, large."""
        if len(self.datasets) <= self.eval_datasets_n:
            return list(self.datasets)
        # Sort by file size as proxy for problem size
        by_size = sorted(self.datasets, key=lambda d: os.path.getsize(d))
        n = self.eval_datasets_n
        # Evenly spaced indices
        indices = [int(i * (len(by_size) - 1) / (n - 1)) for i in range(n)]
        return [by_size[i] for i in sorted(set(indices))]

    def _time_left(self):
        if self.start_time is None:
            return float('inf')
        return self.max_time - (time.time() - self.start_time)

    def _time_up(self):
        return self._time_left() <= 0

    def _ds_label(self, ds):
        return Path(ds).stem

    # ── Scoring ──────────────────────────────────────────────

    def _aggregate_score(self, algo, raw_scores_by_ds):
        """Compute geometric mean of normalized scores across datasets."""
        if not raw_scores_by_ds:
            return float('inf')
        log_norms = []
        for ds, raw in raw_scores_by_ds.items():
            bl = self.baselines.get(ds, raw)
            if bl <= 0 or bl >= 1e9:
                bl = max(raw, 1.0)
            norm = raw / bl if raw < 1e9 else 1e6
            log_norms.append(math.log(max(norm, 1e-6)))
        return math.exp(sum(log_norms) / len(log_norms))

    # ── Checkpoint ───────────────────────────────────────────

    def _save(self):
        self.ckpt.save({
            'datasets': self.datasets, 'phase': self.phase,
            'total_trials': self.total_trials,
            'best_params': self.best_params,
            'best_scores': self.best_scores,
            'screen_raw': self.screen_raw,
            'baselines': self.baselines,
            'param_history': self.param_history,
            'chain_history': self.chain_history,
            'best_chain': self.best_chain,
            'best_chain_score': self.best_chain_score,
            'best_overall_score': self.best_overall_score,
            'best_overall_config': self.best_overall_config,
            'core_algos': self.core_algos,
            'timestamp': datetime.now().isoformat(),
        })

    def _load(self):
        state = self.ckpt.load()
        if not state:
            return False
        saved_ds = state.get('datasets', [state.get('dataset', '')])
        if sorted(saved_ds) != sorted(self.datasets):
            print(f"[AutoTuner] Checkpoint datasets differ, starting fresh")
            return False
        for k in ('phase', 'total_trials', 'best_params', 'best_scores',
                   'screen_raw', 'baselines', 'param_history', 'chain_history',
                   'best_chain', 'best_chain_score', 'best_overall_score',
                   'best_overall_config', 'core_algos'):
            if k in state:
                setattr(self, k, state[k])
        print(f"[AutoTuner] Resumed: phase={self.phase}, trials={self.total_trials}")
        return True

    def _update_best(self, config_type, config, score):
        if score < self.best_overall_score:
            self.best_overall_score = score
            self.best_overall_config = (config_type, config)
            return True
        return False

    def _plateau(self, scores, window=10, eps=0.001):
        if len(scores) < window:
            return False
        recent_best = min(scores[-window:])
        older_best = min(scores[:-window]) if len(scores) > window else scores[0]
        if older_best <= 0 or older_best == float('inf'):
            return False
        return (older_best - recent_best) / abs(older_best) < eps

    # ── Phase 1: Quick Screen ────────────────────────────────

    def _screen(self):
        if self.phase not in ('init', 'screen'):
            return
        self.phase = 'screen'

        remaining = [a for a in TUNABLE_ALGOS if a not in self.screen_raw]
        if not remaining:
            self._compute_baselines()
            self.phase = 'chain_discover'
            self._save()
            return

        ds_labels = [self._ds_label(d) for d in self.datasets]
        print(f"\n{'='*60}")
        print(f"  Phase 1: Quick Screen")
        print(f"  {len(remaining)} algos x {len(self.datasets)} datasets")
        print(f"  Datasets: {', '.join(ds_labels)}")
        print(f"{'='*60}")

        # Screen uses a single seed for speed — just ranking algos, not
        # picking final params. Multi-seed eval happens in Phase 2+.
        futures = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            for algo in remaining:
                params = DEFAULT_PARAMS.get(algo, {})
                for ds in self.datasets:
                    ds_name = self._ds_label(ds)
                    wd = os.path.join(self.output_dir, 'screen', algo, ds_name)
                    futures[pool.submit(run_single_algo, self.binary, ds,
                                        algo, params, self.eval_seeds[0],
                                        wd)] = (algo, ds)

            for fut in as_completed(futures):
                algo, ds = futures[fut]
                result = fut.result()
                score = compute_score(result)
                self.total_trials += 1

                if algo not in self.screen_raw:
                    self.screen_raw[algo] = {}
                self.screen_raw[algo][ds] = score

                if result:
                    h = result.get('hard_violations', 0)
                    s = result.get('soft_penalty', 0)
                    tag = f"soft={s}" if h == 0 else f"INFEASIBLE hard={h}"
                else:
                    tag = "FAILED"
                print(f"  {algo:<12} {self._ds_label(ds):<20} {tag}")

        # Compute baselines and aggregate scores
        self._compute_baselines()

        for algo in TUNABLE_ALGOS:
            if algo in self.screen_raw:
                agg = self._aggregate_score(algo, self.screen_raw[algo])
                self.best_scores[algo] = agg
                self.best_params[algo] = DEFAULT_PARAMS.get(algo, {})

        self.phase = 'chain_discover'
        self._save()

        ranked = sorted(self.best_scores.items(), key=lambda x: x[1])
        score_label = "norm-geomean" if self.multi else "score"
        print(f"\n  Ranking ({score_label}):")
        for i, (a, sc) in enumerate(ranked):
            f_tag = '' if sc < 1e6 else ' (has infeasible)'
            print(f"    {i+1}. {a:<12} {sc:.4f}{f_tag}" if self.multi
                  else f"    {i+1}. {a:<12} {sc:.0f}{f_tag}")

    def _compute_baselines(self):
        """Compute per-dataset baseline = best raw score across all algos."""
        for ds in self.datasets:
            best = float('inf')
            for algo, ds_scores in self.screen_raw.items():
                raw = ds_scores.get(ds, float('inf'))
                if raw < best:
                    best = raw
            self.baselines[ds] = best if best < 1e9 else 1.0

    # ── Phase 3: Deep Parameter Tuning ─────────────────────────

    def _tune_params(self):
        if self.phase != 'param_tune':
            return

        # Use core algos from chain discovery, fallback to top standalone
        if self.core_algos:
            top = self.core_algos
        else:
            ranked = sorted(self.best_scores.items(), key=lambda x: x[1])
            threshold = 1e6 if self.multi else 1e9
            top = [a for a, s in ranked if s < threshold][:4]
            if not top:
                top = [a for a, _ in ranked[:3]]

        # Scale up trials: fewer algos → more depth per algo (same total budget)
        effective_trials = max(self.param_trials,
                               (self.param_trials * 6) // max(len(top), 1))

        eval_ds = self.eval_subset
        eval_labels = [self._ds_label(d) for d in eval_ds]

        print(f"\n{'='*60}")
        print(f"  Phase 3: Deep Parameter Tuning ({effective_trials} trials/algo, {len(self.eval_seeds)} seeds each)")
        print(f"  Tuning: {', '.join(top)}")
        print(f"  Eval on: {', '.join(eval_labels)} ({len(eval_ds)}/{len(self.datasets)} sets)")
        print(f"  Time budget: {self._time_left():.0f}s remaining")
        print(f"{'='*60}")

        for algo in top:
            if self._time_up():
                print(f"\n  Time budget reached, skipping remaining algos")
                break

            space = SEARCH_SPACES.get(algo)
            if not space:
                continue

            existing = sum(1 for a, _, _ in self.param_history if a == algo)
            remaining = effective_trials - existing
            if remaining <= 0:
                print(f"  {algo}: already tuned")
                continue

            print(f"\n  Tuning {algo} ({remaining} trials, {len(self.eval_seeds)} seeds each)...")
            algo_scores = []

            configs = []
            for t in range(remaining):
                if t < 2 or algo not in self.best_params:
                    configs.append(sample_random(algo, self.rng))
                else:
                    configs.append(perturb(algo, self.best_params[algo], self.rng))

            # Each config is evaluated on ALL fixed seeds for fair comparison.
            # Total solver calls = trials * seeds * datasets, but trials are
            # reduced (default 8) so net compute is similar with much better signal.
            for batch_start in range(0, len(configs), self.max_workers):
                if self._time_up():
                    print(f"    Time budget reached")
                    break

                batch = configs[batch_start:batch_start + self.max_workers]
                futures = {}
                with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                    for i, params in enumerate(batch):
                        idx = existing + batch_start + i
                        wd = os.path.join(self.output_dir, 'param_tune', algo, f't{idx}')

                        if self.multi:
                            fut = pool.submit(eval_multi_seed_datasets,
                                              self.binary, eval_ds, algo,
                                              params, self.eval_seeds, wd,
                                              self.baselines)
                        else:
                            fut = pool.submit(eval_multi_seed, self.binary,
                                              self.datasets[0], algo, params,
                                              self.eval_seeds, wd)
                        futures[fut] = (params, idx)

                    for fut in as_completed(futures):
                        params, idx = futures[fut]
                        score = fut.result()

                        self.param_history.append((algo, params, score))
                        self.total_trials += 1
                        algo_scores.append(score)

                        if score < self.best_scores.get(algo, float('inf')):
                            self.best_scores[algo] = score
                            self.best_params[algo] = params
                            self._update_best('single', {'algo': algo, 'params': params}, score)
                            fmt = f"{score:.4f}" if self.multi else f"{score:.0f}"
                            print(f"    t{idx}: NEW BEST {fmt}  {params}")
                        else:
                            fmt = f"{score:.4f}" if self.multi else f"{score:.0f}"
                            print(f"    t{idx}: {fmt}")

                self._save()

                if self._plateau(algo_scores, window=4, eps=0.005):
                    print(f"    Plateau for {algo}, moving on")
                    break

        self.phase = 'chain_rediscover'
        self._save()

        print(f"\n  Tuned results:")
        for a in top:
            sc = self.best_scores.get(a, float('inf'))
            fmt = f"{sc:.4f}" if self.multi else f"{sc:.0f}"
            tag = fmt if sc < 1e6 else "infeasible"
            print(f"    {a:<12} {tag}  {self.best_params.get(a, {})}")

    # ── Phase 2: Chain Discovery (Natural Selection) ──────────

    def _discover_chains(self):
        if self.phase != 'chain_discover':
            return
        if self._time_up():
            print("\n  Time budget reached, skipping chain discovery")
            self.phase = 'extract'
            self._save()
            return

        ranked = sorted(self.best_scores.items(), key=lambda x: x[1])
        threshold = 1e6 if self.multi else 1e9
        top = [a for a, s in ranked if s < threshold]
        if len(top) < 2:
            print("\n  Not enough feasible algos for chaining")
            self.phase = 'extract'
            self._save()
            return

        eval_ds = self.eval_subset

        print(f"\n{'='*60}")
        print(f"  Phase 2: Chain Discovery (Natural Selection)")
        print(f"  Pool: {', '.join(top)}")
        print(f"  Eval on: {', '.join(self._ds_label(d) for d in eval_ds)}")
        print(f"  Population: {self.chain_pop}, Rounds: {self.chain_rounds}")
        print(f"  Time remaining: {self._time_left():.0f}s")
        print(f"{'='*60}")

        population = []

        # Proven combos
        proven = [
            [('sa', self.best_params.get('sa', DEFAULT_PARAMS['sa'])),
             ('gd', self.best_params.get('gd', DEFAULT_PARAMS['gd']))],
            [('sa', self.best_params.get('sa', DEFAULT_PARAMS['sa'])),
             ('lahc', self.best_params.get('lahc', DEFAULT_PARAMS['lahc']))],
            [('tabu', self.best_params.get('tabu', DEFAULT_PARAMS['tabu'])),
             ('sa', self.best_params.get('sa', DEFAULT_PARAMS['sa']))],
            [('kempe', self.best_params.get('kempe', DEFAULT_PARAMS['kempe'])),
             ('sa', self.best_params.get('sa', DEFAULT_PARAMS['sa']))],
            [('sa', self.best_params.get('sa', DEFAULT_PARAMS['sa'])),
             ('kempe', self.best_params.get('kempe', DEFAULT_PARAMS['kempe'])),
             ('gd', self.best_params.get('gd', DEFAULT_PARAMS['gd']))],
            [('tabu', self.best_params.get('tabu', DEFAULT_PARAMS['tabu'])),
             ('sa', self.best_params.get('sa', DEFAULT_PARAMS['sa'])),
             ('gd', self.best_params.get('gd', DEFAULT_PARAMS['gd']))],
        ]
        for combo in proven:
            if all(a in top for a, _ in combo):
                population.append((combo, float('inf')))

        for chain, sc in sorted(self.chain_history, key=lambda x: x[1])[:3]:
            if sc < 1e6:
                population.append((chain, sc))

        while len(population) < self.chain_pop:
            population.append((random_chain(top, self.best_params, self.rng), float('inf')))

        round_bests = []

        for rnd in range(self.chain_rounds):
            if self._time_up():
                print(f"  Time budget reached at round {rnd+1}")
                break

            print(f"\n  Round {rnd+1}/{self.chain_rounds} ({len(population)} chains)")

            to_eval = [(i, c) for i, (c, s) in enumerate(population) if s == float('inf')]
            if to_eval:
                futures = {}
                with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                    for idx, chain in to_eval:
                        wd = os.path.join(self.output_dir, 'chains', f'r{rnd}_c{idx}')

                        if self.multi:
                            fut = pool.submit(eval_chain_multi_seed_datasets,
                                              self.binary, eval_ds, chain,
                                              self.eval_seeds, wd,
                                              self.baselines)
                        else:
                            fut = pool.submit(eval_chain_multi_seed,
                                              self.binary, self.datasets[0],
                                              chain, self.eval_seeds, wd)
                        futures[fut] = idx

                    for fut in as_completed(futures):
                        idx = futures[fut]
                        score = fut.result()
                        chain = population[idx][0]
                        population[idx] = (chain, score)
                        self.total_trials += 1
                        self.chain_history.append((chain, score))

                        desc = ' -> '.join(a for a, _ in chain)
                        fmt = f"{score:.4f}" if self.multi else f"{score:.0f}"
                        if score < self.best_chain_score:
                            self.best_chain_score = score
                            self.best_chain = chain
                            self._update_best('chain', chain, score)
                            print(f"    [{desc}]: NEW BEST {fmt}")
                        else:
                            tag = fmt if score < 1e6 else "infeasible"
                            print(f"    [{desc}]: {tag}")

                self._save()

            population.sort(key=lambda x: x[1])
            rb = population[0][1] if population else float('inf')
            round_bests.append(rb)
            fmt = f"{rb:.4f}" if self.multi else f"{rb:.0f}"
            print(f"  Round {rnd+1} best: {fmt}")

            if self._plateau(round_bests, window=3, eps=0.002):
                print(f"  Chain plateau, stopping early")
                break

            survivors = population[:max(2, len(population) // 2)]
            new_pop = list(survivors)
            while len(new_pop) < self.chain_pop:
                parent = self.rng.choice(survivors)[0]
                new_pop.append((mutate_chain(parent, top, self.best_params, self.rng),
                                float('inf')))
            if rnd % 2 == 1 and len(new_pop) > 2:
                new_pop[-1] = (random_chain(top, self.best_params, self.rng), float('inf'))
            population = new_pop

        self.phase = 'extract'
        self._save()

    # ── Phase 2.5: Extract Core Algos ──────────────────────────

    def _extract_core_algos(self):
        if self.phase != 'extract':
            return

        if not self.chain_history:
            # No chains found — fallback to top standalone algos
            ranked = sorted(self.best_scores.items(), key=lambda x: x[1])
            threshold = 1e6 if self.multi else 1e9
            self.core_algos = [a for a, s in ranked if s < threshold][:4]
            if len(self.core_algos) < 2:
                self.core_algos = [a for a, _ in ranked[:2]]
            print(f"\n  No chain data, falling back to top standalone: "
                  f"{', '.join(self.core_algos)}")
            self.phase = 'param_tune'
            self._save()
            return

        print(f"\n{'='*60}")
        print(f"  Phase 2.5: Extract Core Algos")
        print(f"{'='*60}")

        # Top 30% of chains or at least 5
        sorted_chains = sorted(self.chain_history, key=lambda x: x[1])
        feasible = [(c, s) for c, s in sorted_chains if s < (1e6 if self.multi else 1e9)]
        if not feasible:
            feasible = sorted_chains[:5]
        n_top = max(5, len(feasible) * 3 // 10)
        top_chains = feasible[:n_top]

        # Count algo frequency in top chains
        algo_counts = defaultdict(int)
        for chain, score in top_chains:
            seen = set()
            for algo, _ in chain:
                if algo not in seen:
                    algo_counts[algo] += 1
                    seen.add(algo)

        # Core = algos in ≥40% of top chains, min 2, max 4
        threshold_count = max(1, int(n_top * 0.4))
        core = [a for a, c in sorted(algo_counts.items(), key=lambda x: -x[1])
                if c >= threshold_count]

        if len(core) < 2:
            by_freq = sorted(algo_counts.items(), key=lambda x: -x[1])
            core = [a for a, _ in by_freq[:2]]
        core = core[:4]

        self.core_algos = core

        print(f"  Top {n_top} chains analyzed:")
        for a, c in sorted(algo_counts.items(), key=lambda x: -x[1]):
            pct = c / n_top * 100
            tag = "CORE" if a in core else "cut"
            print(f"    {a:<12} {c}/{n_top} ({pct:.0f}%)  [{tag}]")
        print(f"\n  Core algos: {', '.join(core)}")

        self.phase = 'param_tune'
        self._save()

    # ── Phase 4: Chain Rediscovery ───────────────────────────

    def _rediscover_chains(self):
        if self.phase != 'chain_rediscover':
            return
        if self._time_up():
            print("\n  Time budget reached, skipping chain rediscovery")
            self.phase = 'finalize'
            self._save()
            return

        pool = self.core_algos
        if len(pool) < 2:
            print("\n  Core pool too small for chaining, skipping rediscovery")
            self.phase = 'finalize'
            self._save()
            return

        eval_ds = self.eval_subset
        rounds = max(2, self.chain_rounds - 1)

        print(f"\n{'='*60}")
        print(f"  Phase 4: Chain Rediscovery (tuned params)")
        print(f"  Pool: {', '.join(pool)}")
        print(f"  Eval on: {', '.join(self._ds_label(d) for d in eval_ds)}")
        print(f"  Population: {self.chain_pop}, Rounds: {rounds}")
        print(f"  Time remaining: {self._time_left():.0f}s")
        print(f"{'='*60}")

        population = []

        # Seed with top chains from initial discovery, rebuilt with tuned params
        initial_sorted = sorted(self.chain_history, key=lambda x: x[1])
        for chain, _ in initial_sorted[:5]:
            rebuilt = [(a, self.best_params.get(a, DEFAULT_PARAMS.get(a, {})))
                       for a, _ in chain if a in pool]
            if len(rebuilt) >= 2:
                population.append((rebuilt, float('inf')))

        # Fill with random chains using tuned params
        while len(population) < self.chain_pop:
            population.append((random_chain(pool, self.best_params, self.rng),
                               float('inf')))

        round_bests = []

        for rnd in range(rounds):
            if self._time_up():
                print(f"  Time budget reached at round {rnd+1}")
                break

            print(f"\n  Round {rnd+1}/{rounds} ({len(population)} chains)")

            to_eval = [(i, c) for i, (c, s) in enumerate(population)
                       if s == float('inf')]
            if to_eval:
                futures = {}
                with ProcessPoolExecutor(max_workers=self.max_workers) as pool_exec:
                    for idx, chain in to_eval:
                        wd = os.path.join(self.output_dir, 'chains_r2',
                                          f'r{rnd}_c{idx}')
                        if self.multi:
                            fut = pool_exec.submit(
                                eval_chain_multi_seed_datasets,
                                self.binary, eval_ds, chain,
                                self.eval_seeds, wd, self.baselines)
                        else:
                            fut = pool_exec.submit(
                                eval_chain_multi_seed,
                                self.binary, self.datasets[0],
                                chain, self.eval_seeds, wd)
                        futures[fut] = idx

                    for fut in as_completed(futures):
                        idx = futures[fut]
                        score = fut.result()
                        chain = population[idx][0]
                        population[idx] = (chain, score)
                        self.total_trials += 1
                        self.chain_history.append((chain, score))

                        desc = ' -> '.join(a for a, _ in chain)
                        fmt = f"{score:.4f}" if self.multi else f"{score:.0f}"
                        if score < self.best_chain_score:
                            self.best_chain_score = score
                            self.best_chain = chain
                            self._update_best('chain', chain, score)
                            print(f"    [{desc}]: NEW BEST {fmt}")
                        else:
                            tag = fmt if score < 1e6 else "infeasible"
                            print(f"    [{desc}]: {tag}")

                self._save()

            population.sort(key=lambda x: x[1])
            rb = population[0][1] if population else float('inf')
            round_bests.append(rb)
            fmt = f"{rb:.4f}" if self.multi else f"{rb:.0f}"
            print(f"  Round {rnd+1} best: {fmt}")

            if self._plateau(round_bests, window=2, eps=0.002):
                print(f"  Chain plateau, stopping early")
                break

            survivors = population[:max(2, len(population) // 2)]
            new_pop = list(survivors)
            while len(new_pop) < self.chain_pop:
                parent = self.rng.choice(survivors)[0]
                new_pop.append((mutate_chain(parent, pool, self.best_params,
                                             self.rng), float('inf')))
            population = new_pop

        self.phase = 'finalize'
        self._save()

    # ── Phase 5: Final Validation ────────────────────────────

    def _finalize(self):
        if self.phase != 'finalize':
            return

        print(f"\n{'='*60}")
        print(f"  Phase 5: Final Validation (ALL {len(self.datasets)} datasets, multi-seed)")
        print(f"{'='*60}")

        configs = []
        for algo, sc in sorted(self.best_scores.items(), key=lambda x: x[1])[:3]:
            if sc < (1e6 if self.multi else 1e9):
                configs.append(('single', algo, self.best_params.get(algo, {})))

        if self.best_chain and self.best_chain_score < (1e6 if self.multi else 1e9):
            configs.append(('chain', 'best_chain', self.best_chain))

        if not configs:
            print("  No feasible configurations found!")
            self.phase = 'done'
            self._save()
            return

        test_seeds = [42, 123, 789]
        # Final eval: ALL datasets, not just subset
        final_ds = self.datasets
        results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for ctype, name, cfg in configs:
                for s in test_seeds:
                    wd = os.path.join(self.output_dir, 'final', f'{name}_s{s}')
                    if self.multi:
                        if ctype == 'single':
                            fut = pool.submit(eval_on_datasets, self.binary,
                                              final_ds, name, cfg, s, wd,
                                              self.baselines)
                        else:
                            fut = pool.submit(eval_chain_on_datasets, self.binary,
                                              final_ds, cfg, s, wd, self.baselines)
                    else:
                        if ctype == 'single':
                            fut = pool.submit(run_single_algo, self.binary,
                                              final_ds[0], name, cfg, s, wd)
                        else:
                            fut = pool.submit(run_chain, self.binary,
                                              final_ds[0], cfg, s, wd)
                    futures[fut] = (ctype, name, cfg, s)

            for fut in as_completed(futures):
                ctype, name, cfg, s = futures[fut]
                if self.multi:
                    sc = fut.result()
                else:
                    sc = compute_score(fut.result())
                results.append((ctype, name, cfg, s, sc))
                self.total_trials += 1
                label = name if ctype == 'single' else ' -> '.join(a for a, _ in cfg)
                fmt = f"{sc:.4f}" if self.multi else f"{sc:.0f}"
                print(f"  {label} (seed={s}): {fmt}")

        grouped = defaultdict(list)
        for ct, name, cfg, s, sc in results:
            grouped[(ct, name)].append(sc)

        score_label = "norm-geomean" if self.multi else "avg"
        print(f"\n  Multi-seed {score_label}:")
        best_avg = float('inf')
        for (ct, name), scores in sorted(grouped.items(), key=lambda x: sum(x[1])/len(x[1])):
            avg = sum(scores) / len(scores)
            label = name if ct == 'single' else 'chain'
            fmt_scores = [f'{s:.4f}' if self.multi else f'{s:.0f}' for s in scores]
            fmt_avg = f"{avg:.4f}" if self.multi else f"{avg:.0f}"
            print(f"    {label:<20} {score_label}={fmt_avg}  [{', '.join(fmt_scores)}]")
            if avg < best_avg:
                best_avg = avg

        self.best_overall_score = best_avg
        self.phase = 'done'
        self._save()

    # ── Auto-Update Golden Params ────────────────────────────

    def _maybe_update_params(self, report):
        """Check if tuned params should replace the golden defaults."""
        from tooling.tuned_params import (save_params, check_should_update,
                                  check_plateau, load_metadata)

        if not self.auto_update:
            print("\n  [Params] Auto-update disabled (--no-auto-update)")
            return

        # Build per-dataset scores from best_scores_per_algo isn't quite right —
        # we need the per-dataset raw scores from the best config.
        # Use screen_raw evaluated with best_params as proxy.
        per_ds = {}
        for ds in self.datasets:
            label = self._ds_label(ds)
            # Best raw score across all algos for this dataset
            best_raw = float('inf')
            for algo, ds_scores in self.screen_raw.items():
                raw = ds_scores.get(ds, float('inf'))
                if raw < best_raw:
                    best_raw = raw
            if best_raw < 1e9:
                per_ds[label] = best_raw

        new_score = self.best_overall_score
        new_trials = self.total_trials

        # Plateau check
        is_plateau, plateau_reason = check_plateau()
        if is_plateau and not self.force_update:
            print(f"\n  [Params] {plateau_reason}")
            print(f"  [Params] Skipping update. Use --force-update to override.")
            return

        # Should-update check
        should, reason = check_should_update(
            new_score, per_ds, new_trials)

        if not should and not self.force_update:
            print(f"\n  [Params] Not updating: {reason}")
            return

        if self.force_update and not should:
            print(f"\n  [Params] Force-updating despite: {reason}")

        # Save
        version = save_params(
            params=self.best_params,
            aggregate_score=new_score,
            per_dataset_scores=per_ds,
            trial_count=new_trials,
            source='auto_tuner',
        )
        print(f"\n  [Params] Updated golden params (v{version}): {reason}")
        print(f"  [Params] Saved to tuned_params.json")

        # Also persist the best chain alongside per-algo params
        if self.best_chain and self.best_chain_score < (1e6 if self.multi else 1e9):
            try:
                from tooling.tuned_params import save_best_chain
                save_best_chain(self.best_chain, self.best_chain_score)
                print(f"  [Params] Saved best_chain ({len(self.best_chain)} steps) to tuned_params.json")
            except Exception as e:
                print(f"  [warn] failed to persist best_chain: {e}")

    # ── Main Entry ───────────────────────────────────────────

    def run(self, resume=False):
        if resume:
            self._load()

        self.start_time = time.time()
        ds_labels = [self._ds_label(d) for d in self.datasets]

        print(f"\n{'#'*60}")
        print(f"  Exam Scheduling Auto-Tuner")
        print(f"  Datasets: {', '.join(ds_labels)}")
        if self.multi:
            eval_labels = [self._ds_label(d) for d in self.eval_subset]
            print(f"  Eval subset: {', '.join(eval_labels)}")
        print(f"  Workers: {self.max_workers} | Eval seeds: {self.eval_seeds}")
        print(f"  Time budget: {self.max_time}s ({self.max_time/60:.0f}min)")
        print(f"  Output: {self.output_dir}")
        print(f"{'#'*60}")

        try:
            self._screen()
            self._discover_chains()
            self._extract_core_algos()
            self._tune_params()
            self._rediscover_chains()
            self._finalize()
        except KeyboardInterrupt:
            print(f"\n\n[AutoTuner] Interrupted! Saving checkpoint...")
            self._save()
            print(f"[AutoTuner] Resume with --resume")
            return

        elapsed = time.time() - self.start_time

        print(f"\n{'#'*60}")
        print(f"  RESULTS")
        print(f"{'#'*60}")
        print(f"  Datasets: {len(self.datasets)}")
        print(f"  Total trials: {self.total_trials}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        score_label = "norm-geomean" if self.multi else "score"
        fmt = f"{self.best_overall_score:.4f}" if self.multi else f"{self.best_overall_score:.0f}"
        print(f"  Best {score_label}: {fmt}")

        if self.best_overall_config:
            ct, cfg = self.best_overall_config
            if ct == 'single':
                print(f"  Best: {cfg.get('algo', '?')} params={cfg.get('params', {})}")
            else:
                desc = ' -> '.join(a if isinstance(a, str) else a[0]
                                   for a in (cfg if isinstance(cfg, list) else []))
                print(f"  Best chain: {desc}")

        report = {
            'datasets': [self._ds_label(d) for d in self.datasets],
            'total_trials': self.total_trials,
            'elapsed_seconds': elapsed,
            'scoring': 'normalized_geometric_mean' if self.multi else 'raw_soft_penalty',
            'best_score': self.best_overall_score,
            'best_config': self.best_overall_config,
            'best_params_per_algo': self.best_params,
            'best_scores_per_algo': self.best_scores,
            'baselines': {self._ds_label(d): v for d, v in self.baselines.items()},
            'timestamp': datetime.now().isoformat(),
        }
        rpath = os.path.join(self.output_dir, 'tuning_report.json')
        with open(rpath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report: {rpath}")

        # Auto-update golden params if enabled
        self._maybe_update_params(report)


# ── CLI ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Auto-Tuner for Exam Scheduling Algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tooling.auto_tuner instances/exam_comp_set4.exam            # single dataset
  python -m tooling.auto_tuner --all-sets                               # all ITC 2007 sets
  python -m tooling.auto_tuner --all-sets --synthetic                   # all + synthetic
  python -m tooling.auto_tuner instances/exam_comp_set1.exam instances/exam_comp_set4.exam  # specific sets
  python -m tooling.auto_tuner --all-sets --resume                      # resume from checkpoint
  python -m tooling.auto_tuner --all-sets --max-time 20                 # 20 min budget
        """)

    ap.add_argument('datasets', nargs='*', help='Dataset .exam files')
    ap.add_argument('--all-sets', action='store_true',
                    help='Use all ITC 2007 datasets (instances/exam_comp_set*.exam)')
    ap.add_argument('--synthetic', action='store_true',
                    help='Include a generated synthetic dataset')
    ap.add_argument('--output', default='tuning_results',
                    help='Output directory (default: tuning_results)')
    ap.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    ap.add_argument('--workers', type=int, default=6,
                    help='Max parallel workers (default: 6)')
    ap.add_argument('--param-trials', type=int, default=8,
                    help='Param trials per algo (default: 8)')
    ap.add_argument('--chain-pop', type=int, default=8,
                    help='Chain population (default: 8)')
    ap.add_argument('--chain-rounds', type=int, default=3,
                    help='Chain tournament rounds (default: 3)')
    ap.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    ap.add_argument('--eval-datasets', type=int, default=3,
                    help='Datasets to evaluate per trial in global mode (default: 3)')
    ap.add_argument('--max-time', type=int, default=None,
                    help='Wall-time budget in minutes (default: 30 for multi, 10 for single)')
    ap.add_argument('--ip-time-limit', type=int, default=120,
                    help='IP solver time limit seconds (default: 120)')
    ap.add_argument('--no-auto-update', action='store_true',
                    help='Do not auto-update golden params after tuning')
    ap.add_argument('--force-update', action='store_true',
                    help='Force-update golden params even if checks fail')

    args = ap.parse_args()

    # Resolve datasets
    datasets = list(args.datasets)
    if args.all_sets:
        # auto_tuner.py lives in tooling/, so instances/ is one level up
        root = Path(__file__).resolve().parent.parent / 'instances'
        itc_sets = sorted(root.glob('exam_comp_set*.exam'))
        datasets.extend(str(p) for p in itc_sets)

    if args.synthetic:
        syn_path = generate_synthetic_dataset(
            os.path.join(args.output, '_synthetic'), seed=args.seed)
        datasets.append(syn_path)

    # Deduplicate
    seen = set()
    unique = []
    for d in datasets:
        ad = os.path.abspath(d)
        if ad not in seen:
            seen.add(ad)
            unique.append(ad)
    datasets = unique

    if not datasets:
        print("Error: no datasets specified. Use positional args or --all-sets")
        sys.exit(1)

    for d in datasets:
        if not os.path.isfile(d):
            print(f"Error: dataset not found: {d}")
            sys.exit(1)

    max_time_sec = args.max_time * 60 if args.max_time else None

    AutoTuner(
        datasets=datasets,
        output_dir=args.output,
        max_workers=args.workers,
        param_trials=args.param_trials,
        chain_pop=args.chain_pop,
        chain_rounds=args.chain_rounds,
        seed=args.seed,
        ip_time_limit=args.ip_time_limit,
        max_time=max_time_sec,
        eval_datasets=args.eval_datasets,
        auto_update=not args.no_auto_update,
        force_update=args.force_update,
        resume=args.resume,
    ).run(resume=args.resume)


if __name__ == '__main__':
    main()
