"""The :class:`AutoTuner` orchestrator — phase-by-phase tuning pipeline.

Lives as a single class because the phases share a giant state machine
(``best_params``, ``chain_history``, ``baselines``, checkpoint I/O) that
only makes sense when held together. Splitting the phases into mixins
would scatter that state across files and save nobody any cognitive load.

Pipeline, in order:
    Phase 1  ``_screen``            — single-seed ranking of every algo
    Phase 2  ``_discover_chains``   — natural-selection chain tournament
    Phase 2.5 ``_extract_core_algos`` — distill top chains into a core set
    Phase 3  ``_tune_params``       — derivative-free tuning of core algos
    Phase 4  ``_rescore_top_chains`` — re-score Phase 2 chains w/ tuned params
    Phase 5  ``_finalize``          — multi-seed validation on every dataset
"""
from __future__ import annotations

import math
import os
import random
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tooling.chain_prefix_cache import PrefixCache
from tooling.eval_cache import EvalCache
from tooling.optimizers import optimize_params
from tooling.successive_halving import successive_halving

from tooling.tuner.binary import find_or_build_binary
from tooling.tuner.checkpoint import Checkpoint
from tooling.tuner.eval import (
    compute_score, run_single_algo, run_chain,
    eval_on_datasets, eval_chain_on_datasets,
    eval_multi_seed, eval_multi_seed_datasets,
)
from tooling.tuner.sampling import perturb, random_chain, mutate_chain, vary_chain
from tooling.tuner.search_spaces import (
    CHAIN_EARLY_STOP_RATIO, DEFAULT_PARAMS, EVAL_SEEDS,
    PROVEN_CHAIN_TEMPLATES, SEARCH_SPACES, TUNABLE_ALGOS,
)


def _cap_workers(n_items: int, max_workers: int) -> int:
    """Clamp ThreadPool size so parallel subprocess spawns don't exceed
    the configured max_workers — prevents C++ solver oversubscription."""
    return max(1, min(n_items, max_workers))


def _eta_schedule_for_pop(pop_size: int) -> list[int]:
    """Pop-size-aware promotion ratios for the 3-rung SH tournament.

    Larger populations cull harder at cheap rungs so we don't waste the
    expensive-rung budget evaluating clearly-dominated candidates.
    """
    if pop_size <= 8:
        return [2, 2, 2]
    if pop_size <= 16:
        return [3, 2, 2]
    return [4, 3, 2]


class AutoTuner:
    def __init__(self, datasets, output_dir='tuning_results',
                 max_workers=6, param_trials=8,
                 chain_pop=8, chain_rounds=3,
                 seed=42, ip_time_limit=120,
                 max_time=None, eval_datasets=3,
                 auto_update=True, force_update=False,
                 resume=False,
                 chain_max_len=10, chain_crossover_rate=0.25,
                 chain_allow_duplicates=False,
                 chain_prefix_cache=True,
                 chain_partial_credit=True,
                 chain_early_stop=True,
                 chain_early_stop_ratio=2.5,
                 chain_truncation_penalty=0.025):
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

        # ── Chain-finder v2 knobs ─────────────────────────────
        self.chain_max_len = chain_max_len
        self.chain_crossover_rate = chain_crossover_rate
        self.chain_allow_duplicates = chain_allow_duplicates
        self.chain_partial_credit = chain_partial_credit
        self.chain_early_stop = chain_early_stop
        self.chain_early_stop_ratio = chain_early_stop_ratio
        # truncation_penalty is read from search_spaces by compute_score
        # directly — we only store it for reporting/debugging.
        self.chain_truncation_penalty = chain_truncation_penalty

        # Clean stale output on fresh runs to prevent old .sln artifacts
        # from polluting chain warm-starts or confusing results.
        # Includes 'chains_sh' (actual SH work dir) alongside the legacy
        # 'chains' name we kept for back-compat.
        if not resume and os.path.isdir(output_dir):
            for subdir in ('screen', 'param_tune', 'chains',
                           'chains_sh', 'final'):
                stale = os.path.join(output_dir, subdir)
                if os.path.isdir(stale):
                    shutil.rmtree(stale)
            ckpt_file = os.path.join(output_dir, 'checkpoint.json')
            if os.path.isfile(ckpt_file):
                os.remove(ckpt_file)

        os.makedirs(output_dir, exist_ok=True)
        self.ckpt = Checkpoint(os.path.join(output_dir, 'checkpoint.json'))
        self.binary = find_or_build_binary()
        cache_path = os.path.join(output_dir, 'eval_cache.json')
        self.cache = EvalCache(persist_path=cache_path)
        self.start_time = None

        # ── Prefix .sln cache (Tier A E2) ─────────────────────
        # Disabled automatically if free disk < 5GB to avoid filling a
        # Colab session's scratch disk.
        self.prefix_cache = None
        if chain_prefix_cache:
            if PrefixCache.check_disk(output_dir, min_free_gb=5.0):
                pc_dir = os.path.join(output_dir, 'chains_sh', '_prefix_cache')
                self.prefix_cache = PrefixCache(pc_dir)
            else:
                print(f"[AutoTuner] Free disk < 5GB — prefix cache disabled")

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

    def _threshold_for(self, ds):
        """Per-dataset abort threshold for step-level early stop.

        Computed as ``baseline[ds] * ratio`` where ``ratio`` grows as we
        discover better chains (tighter pruning as the tournament
        progresses). Returns None if no baseline exists yet — falls back
        to no abort.
        """
        bl = self.baselines.get(ds) if hasattr(self, 'baselines') else None
        if not bl or bl <= 0 or bl >= 1e9:
            return None
        multiplier = self.chain_early_stop_ratio
        if self.multi and self.best_chain_score < 1e6:
            # Tighten as we find better chains: use the best normalized
            # score as a multiplier — but clamp to at least the base ratio
            # so we never prune more aggressively than intended.
            multiplier = max(self.chain_early_stop_ratio,
                             self.best_chain_score * self.chain_early_stop_ratio)
        return bl * multiplier

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
        cache_hits = []  # [(algo, ds, result)]
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            for algo in remaining:
                params = DEFAULT_PARAMS.get(algo, {})
                for ds in self.datasets:
                    ds_name = self._ds_label(ds)
                    wd = os.path.join(self.output_dir, 'screen', algo, ds_name)
                    key = EvalCache.key_for('single', algo, ds, self.eval_seeds[0], params)
                    hit = self.cache.get(key)
                    if hit is not None:
                        cache_hits.append((algo, ds, hit))
                        continue
                    fut = pool.submit(run_single_algo, self.binary, ds,
                                      algo, params, self.eval_seeds[0], wd)
                    futures[fut] = (algo, ds, key)

            # Process cache hits synchronously
            for algo, ds, result in cache_hits:
                score = compute_score(result)
                self.screen_raw.setdefault(algo, {})[ds] = score
                h = result.get('hard_violations', 0)
                s = result.get('soft_penalty', 0)
                tag = f"soft={s} (cached)" if h == 0 else f"INFEASIBLE hard={h} (cached)"
                print(f"  {algo:<12} {self._ds_label(ds):<20} {tag}")

            for fut in as_completed(futures):
                algo, ds, key = futures[fut]
                result = fut.result()
                self.cache.put(key, result)
                score = compute_score(result)
                self.total_trials += 1

                self.screen_raw.setdefault(algo, {})[ds] = score

                if result:
                    h = result.get('hard_violations', 0)
                    s = result.get('soft_penalty', 0)
                    tag = f"soft={s}" if h == 0 else f"INFEASIBLE hard={h}"
                else:
                    tag = "FAILED"
                print(f"  {algo:<12} {self._ds_label(ds):<20} {tag}")

        self.cache.save()

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

            print(f"\n  Tuning {algo} ({remaining} evals, derivative-free)...")
            algo_scores = []
            trial_idx = [existing]  # mutable counter for closure
            budget_exceeded = [False]

            def eval_closure(params):
                if self._time_up():
                    budget_exceeded[0] = True
                    return float('inf')
                idx = trial_idx[0]
                trial_idx[0] += 1
                wd = os.path.join(self.output_dir, 'param_tune', algo, f't{idx}')
                if self.multi:
                    score = eval_multi_seed_datasets(
                        self.binary, eval_ds, algo, params,
                        self.eval_seeds, wd, self.baselines)
                else:
                    score = eval_multi_seed(
                        self.binary, self.datasets[0], algo, params,
                        self.eval_seeds, wd)
                self.param_history.append((algo, params, score))
                self.total_trials += 1
                algo_scores.append(score)

                if score < self.best_scores.get(algo, float('inf')):
                    self.best_scores[algo] = score
                    self.best_params[algo] = dict(params)
                    self._update_best('single', {'algo': algo, 'params': dict(params)}, score)
                    fmt = f"{score:.4f}" if self.multi else f"{score:.0f}"
                    print(f"    t{idx}: NEW BEST {fmt}  {params}")
                else:
                    fmt = f"{score:.4f}" if self.multi else f"{score:.0f}"
                    print(f"    t{idx}: {fmt}")

                if trial_idx[0] % 2 == 0:
                    self._save()
                return score

            try:
                optimize_params(algo, eval_closure, n_evals=remaining)
            except Exception as e:
                print(f"    Optimizer error for {algo}: {e}")

            self._save()
            if budget_exceeded[0]:
                print(f"    Time budget reached during {algo}")

        self.phase = 'chain_rediscover'
        self._save()

        print(f"\n  Tuned results:")
        for a in top:
            sc = self.best_scores.get(a, float('inf'))
            fmt = f"{sc:.4f}" if self.multi else f"{sc:.0f}"
            tag = fmt if sc < 1e6 else "infeasible"
            print(f"    {a:<12} {tag}  {self.best_params.get(a, {})}")

    # ── Chain Evaluation Helper (variable fidelity) ──────────

    def _eval_chain_at_fidelity(self, chain, n_seeds, n_datasets, work_dir_tag):
        """Evaluate a chain using n_seeds seeds x n_datasets datasets from eval_subset.

        Uses the first n_seeds of self.eval_seeds and the first n_datasets
        of self.eval_subset. Returns geometric mean of normalized medians
        (multi) or median score (single). Non-cached (ds, seed) pairs run
        in parallel threads.
        """
        seeds = self.eval_seeds[:max(1, n_seeds)]
        ds_list = self.eval_subset[:max(1, n_datasets)]

        # Split into cache hits vs work items
        results_by_ds = defaultdict(list)  # ds -> [score, ...]
        work_items = []  # [(ds, seed)]

        for ds in ds_list:
            for s in seeds:
                key = EvalCache.chain_key(ds, s, chain)
                hit = self.cache.get(key)
                if hit is not None:
                    results_by_ds[ds].append(compute_score(hit))
                else:
                    work_items.append((ds, s))

        if work_items:
            pc_dir = self.prefix_cache.cache_dir if self.prefix_cache else None

            def _run(pair):
                ds, s = pair
                ds_name = self._ds_label(ds)
                wd = os.path.join(self.output_dir, 'chains_sh', work_dir_tag,
                                  ds_name, f's{s}')
                threshold = (self._threshold_for(ds)
                             if self.chain_early_stop else None)
                result = run_chain(
                    self.binary, ds, chain, s, wd,
                    allow_partial=self.chain_partial_credit,
                    abort_threshold_soft=threshold,
                    prefix_cache_dir=pc_dir,
                )
                return ds, s, result

            with ThreadPoolExecutor(
                    max_workers=_cap_workers(len(work_items),
                                              self.max_workers)) as pool:
                for ds, s, result in pool.map(_run, work_items):
                    key = EvalCache.chain_key(ds, s, chain)
                    self.cache.put(key, result)
                    results_by_ds[ds].append(compute_score(result))

        log_norms = []
        for ds in ds_list:
            ds_scores = sorted(results_by_ds[ds])
            raw = ds_scores[len(ds_scores) // 2]  # median
            if not self.multi:
                return raw
            bl = self.baselines.get(ds, raw)
            if bl <= 0 or bl >= 1e9:
                bl = max(raw, 1.0)
            norm = raw / bl if raw < 1e9 else 1e6
            log_norms.append(math.log(max(norm, 1e-6)))
        if not log_norms:
            return float('inf')
        return math.exp(sum(log_norms) / len(log_norms))

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

        # Seed from PROVEN_CHAIN_TEMPLATES (search_spaces is the single source
        # of truth — includes the current champion kempe→alns→kempe→tabu plus
        # a handful of ALNS/VNS/HHO-led combos for 2025 roster coverage).
        seen_seqs = set()
        for template in PROVEN_CHAIN_TEMPLATES:
            if not all(a in top for a in template):
                continue
            if template in seen_seqs:
                continue
            seen_seqs.add(template)
            combo = [(a, dict(self.best_params.get(a, DEFAULT_PARAMS.get(a, {}))))
                     for a in template]
            population.append((combo, float('inf')))

        # Carry best-scoring chains from prior runs (resume case)
        for chain, sc in sorted(self.chain_history, key=lambda x: x[1])[:3]:
            if sc < 1e6:
                population.append((chain, sc))

        while len(population) < self.chain_pop:
            population.append((random_chain(
                top, self.best_params, self.rng,
                allow_duplicates=self.chain_allow_duplicates), float('inf')))

        round_bests = []

        for rnd in range(self.chain_rounds):
            if self._time_up():
                print(f"  Time budget reached at round {rnd+1}")
                break

            print(f"\n  Round {rnd+1}/{self.chain_rounds} ({len(population)} chains)")

            to_eval = [(i, c) for i, (c, s) in enumerate(population) if s == float('inf')]
            if to_eval:
                # Successive-halving rungs: cheap at first, expensive for survivors.
                # Rung 0: 1 seed × 1 dataset  (fastest fidelity)
                # Rung 1: 2 seeds × 2 datasets
                # Rung 2: all seeds × all eval datasets
                n_seeds_full = len(self.eval_seeds)
                n_ds_full = len(eval_ds)
                rungs = [
                    (1, 1),
                    (min(2, n_seeds_full), min(2, n_ds_full)),
                    (n_seeds_full, n_ds_full),
                ]
                dedup = []
                for r in rungs:
                    if not dedup or dedup[-1] != r:
                        dedup.append(r)
                rungs = dedup

                idx_by_id = {id(c): i for i, c in to_eval}
                chain_list = [c for _, c in to_eval]

                def eval_fn(chain, rung_idx, fidelity):
                    n_seeds, n_ds = fidelity
                    tag = f'r{rnd}_rung{rung_idx}_{idx_by_id[id(chain)]}'
                    if self._time_up():
                        return float('inf')
                    return self._eval_chain_at_fidelity(chain, n_seeds, n_ds, tag)

                # Adaptive per-rung promotion. Larger populations can afford
                # aggressive rung-0 pruning (eta=3 or 4) without losing
                # signal; small populations stay at eta=2 to avoid collapsing
                # diversity. See _eta_schedule_for_pop docstring.
                eta_sched = _eta_schedule_for_pop(len(chain_list))
                winner, winner_score, history = successive_halving(
                    chain_list, eval_fn, rungs, eta=2, eta_schedule=eta_sched)

                # Record best score seen for each chain across rungs
                best_by_id = {}
                for chain, rung_idx, score in history:
                    key = id(chain)
                    if key not in best_by_id or score < best_by_id[key]:
                        best_by_id[key] = score

                for idx, chain in to_eval:
                    score = best_by_id.get(id(chain), float('inf'))
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

                self.cache.save()
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
            survivor_chains = [c for c, _ in survivors]
            while len(new_pop) < self.chain_pop:
                parent = self.rng.choice(survivors)[0]
                # vary_chain either mutates (default) or crosses over with
                # another survivor (probability = chain_crossover_rate).
                new_chain = vary_chain(
                    parent, top, self.best_params, self.rng,
                    survivors=survivor_chains,
                    crossover_rate=self.chain_crossover_rate,
                    allow_duplicates=self.chain_allow_duplicates,
                )
                new_pop.append((new_chain, float('inf')))
            if rnd % 2 == 1 and len(new_pop) > 2:
                new_pop[-1] = (random_chain(
                    top, self.best_params, self.rng,
                    allow_duplicates=self.chain_allow_duplicates), float('inf'))
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

    # ── Phase 4: Rescore Top Chains with Tuned Params ────────

    def _rescore_top_chains(self):
        """Take top-5 chains from Phase 2 and re-score them using tuned params.

        The GA-based chain rediscovery (old Phase 4) duplicated work: it
        re-ran the entire tournament with the same chain structures but new
        params. We can get ~95% of that benefit far cheaper by just swapping
        tuned params into the top chains and re-scoring them once each.
        """
        if self.phase != 'chain_rediscover':
            return
        if self._time_up():
            print("\n  Time budget reached, skipping chain rescoring")
            self.phase = 'finalize'
            self._save()
            return

        if not self.chain_history:
            print("\n  No chain history, skipping rescoring")
            self.phase = 'finalize'
            self._save()
            return

        print(f"\n{'='*60}")
        print(f"  Phase 4: Rescore Top Chains (tuned params)")
        print(f"{'='*60}")

        # Collect top 5 unique chain structures (by algo sequence)
        seen_structures = set()
        top_rescored = []
        sorted_chains = sorted(self.chain_history, key=lambda x: x[1])
        for chain, _ in sorted_chains:
            sig = tuple(a for a, _ in chain)
            if sig in seen_structures:
                continue
            seen_structures.add(sig)
            # Rebuild with tuned params
            rebuilt = [(a, dict(self.best_params.get(a, DEFAULT_PARAMS.get(a, {}))))
                       for a, _ in chain]
            top_rescored.append(rebuilt)
            if len(top_rescored) >= 5:
                break

        if not top_rescored:
            print("  No chains to rescore")
            self.phase = 'finalize'
            self._save()
            return

        n_seeds_full = len(self.eval_seeds)
        n_ds_full = len(self.eval_subset)

        # Evaluate all chains in parallel (each _eval_chain_at_fidelity
        # already parallelizes its inner ds×seed loop, so use threads here
        # to overlap the I/O-bound subprocess waits across chains).
        def _rescore(args):
            i, chain = args
            score = self._eval_chain_at_fidelity(
                chain, n_seeds_full, n_ds_full, f'rescore_{i}')
            return i, chain, score

        with ThreadPoolExecutor(
                max_workers=_cap_workers(len(top_rescored),
                                          self.max_workers)) as pool:
            for i, chain, score in pool.map(_rescore, enumerate(top_rescored)):
                self.total_trials += 1
                self.chain_history.append((chain, score))

                desc = ' -> '.join(a for a, _ in chain)
                fmt = f"{score:.4f}" if self.multi else f"{score:.0f}"
                if score < self.best_chain_score:
                    self.best_chain_score = score
                    self.best_chain = chain
                    self._update_best('chain', chain, score)
                    print(f"  [{desc}]: NEW BEST {fmt}")
                else:
                    tag = fmt if score < 1e6 else "infeasible"
                    print(f"  [{desc}]: {tag}")

        self.cache.save()
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

    # ── Sensitivity Grid Export (RQ3) ────────────────────────

    def _rank_params_by_impact(self, algo, tuned_params):
        """Rank an algo's tunable knobs by observed score variance.

        Uses :attr:`param_history` (the per-trial log built during Phase 3)
        when available: for each knob, compute the Pearson correlation
        between knob value and score, then rank by ``|corr|``. Falls back
        to the search-space ordering for algos that weren't tuned in this
        run — good enough because for the 1-knob algos the fallback is
        trivially correct anyway.
        """
        space = SEARCH_SPACES.get(algo, {})
        knobs = [k for k in space.keys() if k in tuned_params]

        rows = [(p, s) for a, p, s in self.param_history
                if a == algo and s < 1e9]
        if len(rows) < 3 or len(knobs) < 2:
            return knobs

        import statistics

        scored = []
        for knob in knobs:
            values = [p.get(knob, tuned_params.get(knob)) for p, _ in rows]
            scores = [s for _, s in rows]
            if len(set(values)) < 2:
                scored.append((knob, 0.0))
                continue
            try:
                vm = statistics.mean(values)
                sm = statistics.mean(scores)
                num = sum((v - vm) * (s - sm) for v, s in zip(values, scores))
                dv = math.sqrt(sum((v - vm) ** 2 for v in values))
                ds = math.sqrt(sum((s - sm) ** 2 for s in scores))
                corr = abs(num / (dv * ds)) if dv and ds else 0.0
            except (ValueError, ZeroDivisionError):
                corr = 0.0
            scored.append((knob, corr))
        scored.sort(key=lambda x: -x[1])
        return [k for k, _ in scored]

    def export_sensitivity_grid(self, algo, tuned_params, out_dir,
                                grid_mults=(0.5, 0.75, 1.0, 1.25, 1.5),
                                n_seeds=3, dataset=None):
        """Sweep the top-2 most-impactful knobs for *algo* on a grid.

        Picks an eval dataset (default: first in :attr:`eval_subset`), runs
        ``len(grid_mults)^2 * n_seeds`` trials, and writes
        ``<out_dir>/sensitivity_<algo>.csv`` in the tidy long form the
        :func:`utils.plots.tuning.plot_tuning_sensitivity` heatmap expects.

        Degrades to a 1-D sweep when *algo* has only one tunable knob —
        param_b columns are left empty so the heatmap can fall back to a
        1-D curve.
        """
        import csv

        os.makedirs(out_dir, exist_ok=True)
        ds = dataset or (self.eval_subset[0] if self.eval_subset else self.datasets[0])
        seeds = list(range(n_seeds))

        ranked = self._rank_params_by_impact(algo, tuned_params)
        if not ranked:
            print(f"[sensitivity] {algo} has no tunable knobs, skipping")
            return None
        pa = ranked[0]
        pb = ranked[1] if len(ranked) > 1 else None

        def _grid_for(p):
            base = tuned_params.get(p)
            if base is None:
                return []
            lo, hi, scale = SEARCH_SPACES[algo][p]
            vals = []
            for m in grid_mults:
                v = int(round(base * m))
                if scale == 'int':
                    v = max(lo, min(hi, v))
                else:
                    v = max(lo, min(hi, max(v, 1)))
                if v not in vals:
                    vals.append(v)
            return vals

        a_vals = _grid_for(pa)
        b_vals = _grid_for(pb) if pb else [None]

        out_path = os.path.join(out_dir, f"sensitivity_{algo}.csv")
        rows_written = 0
        print(f"\n  [sensitivity] {algo}: sweeping {pa}"
              + (f" × {pb}" if pb else "")
              + f" ({len(a_vals)}×{len(b_vals)}×{len(seeds)} trials)")

        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                "algorithm", "param_a", "value_a", "param_b", "value_b",
                "seed", "soft_penalty", "runtime", "feasible",
            ])
            for va in a_vals:
                for vb in b_vals:
                    params = dict(tuned_params)
                    params[pa] = va
                    if pb is not None and vb is not None:
                        params[pb] = vb
                    for s in seeds:
                        wd = os.path.join(
                            out_dir, "trials", algo,
                            f"{pa}{va}" + (f"_{pb}{vb}" if pb else ""),
                            f"s{s}",
                        )
                        r = run_single_algo(self.binary, ds, algo,
                                            params, s, wd)
                        if r is None:
                            w.writerow([algo, pa, va, pb or "", vb if vb is not None else "",
                                        s, "", "", False])
                        else:
                            w.writerow([
                                algo, pa, va, pb or "",
                                vb if vb is not None else "",
                                s, r.get("soft_penalty", ""),
                                r.get("runtime", ""),
                                r.get("hard_violations", 0) == 0,
                            ])
                        rows_written += 1

        print(f"  [sensitivity] {algo}: wrote {rows_written} rows → {out_path}")
        return out_path

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
            self._rescore_top_chains()
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

        import json
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
