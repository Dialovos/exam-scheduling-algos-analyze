"""
12 algorithms via C++ solver (100-200x faster than Python fallback).

Usage:
  python main.py
  python main.py --dataset instances/exam_comp_set4.exam
  python main.py --dataset data.exam --algo tabu
  python main.py --dataset data.exam --algo sa,gd,vns
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models import ProblemInstance, Solution
from core.generator import generate_synthetic, write_itc2007_format
from core.parser import parse_itc2007_exam, write_solution_itc2007, read_solution_itc2007
from algorithms.cpp_bridge import run_cpp_solver
from utils.plotting import plot_soft_constraint_breakdown
from utils.batch_manager import BatchManager
from tooling.tuned_params import load_params_flat, load_metadata

try:
    from algorithms.ip_solver import solve_ip
    HAS_IP = True
except ImportError:
    HAS_IP = False


def _print_comparison(results):
    print(f"\n{'='*72}")
    print("  COMPARISON")
    print(f"{'='*72}")
    print(f"{'Algorithm':<25} {'Runtime':>9} {'Feasible':>9} {'Hard':>7} {'Soft':>9}")
    print(f"{'─'*72}")
    for name, r in results.items():
        ev = r['evaluation']
        feasible = ev.feasible if hasattr(ev, 'feasible') else ev.is_feasible
        hard = ev.hard if hasattr(ev, 'hard') else ev.hard_violations
        soft = ev.soft if hasattr(ev, 'soft') else ev.soft_penalty
        print(f"{name:<25} {r['runtime']:>8.3f}s {'Yes' if feasible else 'No':>9} "
              f"{hard:>7} {soft:>9}")


def _save_summary(results, output_dir):
    """Per-algo runtime + feasibility + totals — one-shot machine-readable summary."""
    summary = {}
    for name, r in results.items():
        ev = r['evaluation']
        summary[name] = {
            'runtime': float(r.get('runtime', 0.0)),
            'feasible': bool(getattr(ev, 'hard_violations', 1) == 0),
            'hard': int(getattr(ev, 'hard_violations', 0)),
            'soft': int(getattr(ev, 'soft_penalty', 0)),
        }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _save_soft_breakdown(results, output_dir):
    breakdown = {}
    for name, r in results.items():
        ev = r['evaluation']
        breakdown[name] = {
            'two_in_a_row': getattr(ev, 'two_in_a_row', 0),
            'two_in_a_day': getattr(ev, 'two_in_a_day', 0),
            'period_spread': getattr(ev, 'period_spread', 0),
            'non_mixed_durations': getattr(ev, 'non_mixed_durations', 0),
            'front_load': getattr(ev, 'front_load', 0),
            'period_penalty': getattr(ev, 'period_penalty', 0),
            'room_penalty': getattr(ev, 'room_penalty', 0),
        }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "soft_breakdown.json"), 'w') as f:
        json.dump(breakdown, f, indent=2)
    plot_soft_constraint_breakdown(breakdown, output_dir)
    return breakdown


def run_demo(size=50, algo=None, verbose=True, output_dir='results', **kwargs):
    print(f"\n{'='*72}")
    print(f"  DEMO: {size}-exam synthetic instance")
    print(f"{'='*72}\n")

    # Extract IP-only kwargs so they don't forward into run_cpp_solver
    ip_time = kwargs.pop('ip_time', 60)
    ip_workers = kwargs.pop('ip_workers', 0)
    ip_warm_solution = kwargs.pop('ip_warm_solution', None)
    kwargs.pop('ip_max_exams', None)  # demo uses a fixed 150-exam cap

    problem = generate_synthetic(
        num_exams=size, student_ratio=7.0, conflict_density=0.15,
        num_rooms=max(3, size // 20), room_capacity=max(50, int(size * 0.4)),
        seed=kwargs.get('seed', 42),
    )
    print(problem.summary())

    os.makedirs("datasets", exist_ok=True)
    exam_path = f"instances/synthetic_{size}.exam"
    write_itc2007_format(problem, exam_path)

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # C++ algorithms
    if algo != 'ip':
        cpp_algo = algo if algo else 'all'
        cpp_results = run_cpp_solver(
            exam_path, problem, algo=cpp_algo, output_dir=output_dir,
            verbose=verbose, **kwargs)
        if cpp_results:
            results.update(cpp_results)

    # IP solver (Python) — demo keeps a tight 150-exam cap so it stays snappy
    if (algo is None or algo == 'ip') and HAS_IP and problem.num_exams() <= 150:
        if verbose:
            print(f"\n{'─'*50}\nInteger Programming (CP-SAT, limit={ip_time}s)...")
        r = solve_ip(problem, time_limit=ip_time, verbose=verbose,
                     warm_start=ip_warm_solution, num_workers=ip_workers)
        results['IP'] = r

    _print_comparison(results)
    bd = _save_soft_breakdown(results, output_dir)
    _save_summary(results, output_dir)
    print(f"\n{'='*72}\n  SOFT CONSTRAINT BREAKDOWN\n{'='*72}")
    for name, b in bd.items():
        total = sum(b.values())
        parts = [f"{k}={v}" for k, v in b.items() if v > 0]
        print(f"{name:<25} total={total:>6}  [{', '.join(parts)}]")


def run_on_dataset(filepath, limit=0, algo=None, verbose=True, output_dir='results',
                   ip_time=120, ip_max_exams=900, ip_warm_solution=None,
                   ip_workers=0, **kwargs):
    limit_str = f" (limit={limit} exams)" if limit > 0 else " (full dataset)"
    algo_str = f" [{algo.upper()}]" if algo else ""
    print(f"\n{'='*72}")
    print(f"  DATASET: {os.path.basename(filepath)}{limit_str}{algo_str}")
    print(f"{'='*72}\n")

    problem = parse_itc2007_exam(filepath, limit=limit)
    print(problem.summary())

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # C++ algorithms
    if algo != 'ip':
        cpp_algo = algo if algo else 'all'
        cpp_results = run_cpp_solver(
            filepath, problem, algo=cpp_algo, limit=limit,
            output_dir=output_dir, verbose=verbose, **kwargs)
        if cpp_results:
            results.update(cpp_results)

    # IP solver (Python)
    if (algo is None or algo == 'ip') and HAS_IP:
        ne = problem.num_exams()
        if ne <= ip_max_exams:
            if verbose:
                hint_note = " + warm-start" if ip_warm_solution is not None else ""
                print(f"\n{'─'*50}\nInteger Programming "
                      f"(CP-SAT, limit={ip_time}s{hint_note})...")
            r = solve_ip(problem, time_limit=ip_time, verbose=verbose,
                         warm_start=ip_warm_solution, num_workers=ip_workers)
            results['IP'] = r
        elif verbose:
            print(f"\n[IP] Skipped (n={ne} > {ip_max_exams})")

    _print_comparison(results)
    _save_soft_breakdown(results, output_dir)
    _save_summary(results, output_dir)

    sln_dir = os.path.join(output_dir, "solutions")
    os.makedirs(sln_dir, exist_ok=True)
    for name, r in results.items():
        safe = name.lower().replace(' ', '_')
        write_solution_itc2007(r['solution'], os.path.join(sln_dir, f"solution_{safe}_{problem.num_exams()}.sln"))
    print(f"\nSolutions saved to {sln_dir}/")

    # Passive regression check against golden params baseline
    _check_regression(filepath, results)


def _check_regression(filepath, results, threshold=0.15):
    """Passive check: warn if any result is much worse than golden baseline."""
    meta = load_metadata()
    if not meta:
        return
    golden_ds = meta.get('per_dataset_scores', {})
    ds_label = os.path.splitext(os.path.basename(filepath))[0]
    expected = golden_ds.get(ds_label)
    if expected is None or expected <= 0:
        return

    for name, r in results.items():
        ev = r['evaluation']
        hard = ev.hard if hasattr(ev, 'hard') else ev.hard_violations
        if hard > 0:
            continue
        soft = ev.soft if hasattr(ev, 'soft') else ev.soft_penalty
        if soft > 0 and expected > 0:
            regression = (soft - expected) / expected
            if regression > threshold:
                print(f"\n  WARNING: {name} scored {soft:.0f} but tuned baseline "
                      f"expects ~{expected:.0f} on {ds_label} "
                      f"(+{regression:.0%} regression)")


def main():
    ap = argparse.ArgumentParser(
        description="Exam Scheduling Benchmark Suite v3 (C++ Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithms:  greedy, tabu, kempe, sa, alns, gd, abc, ga, lahc, woa, cpsat, vns → C++  |  ip → Python

Examples:
  python main.py                                        # Demo 50 exams
  python main.py --dataset data.exam                    # All algos
  python main.py --dataset data.exam --algo tabu        # Tabu only
  python main.py --dataset data.exam --algo sa,gd,vns   # Multiple algos
  python main.py --mode tune --dataset data.exam        # Auto-tune params + chains
  python main.py --mode tune --dataset data.exam --resume  # Resume tuning
        """)

    # Load golden params for argparse defaults
    _gp = load_params_flat()

    ap.add_argument('--mode', choices=['demo', 'plot', 'batches', 'tune'], default='demo')
    ap.add_argument('--algo', choices=['greedy', 'ip', 'tabu', 'kempe', 'sa', 'alns', 'gd', 'abc', 'ga', 'lahc', 'woa', 'hho', 'cpsat', 'vns'])
    ap.add_argument('--size', type=int, default=50)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--output', type=str, default='results')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch', type=str, default=None,
                    help='Batch name (auto-creates). Omit for auto-timestamped batch.')
    ap.add_argument('--load-batch', type=str, default=None,
                    help='Load existing batch by ID/name instead of creating new.')
    ap.add_argument('--no-batch', action='store_true',
                    help='Disable batching, write directly to results/.')
    ap.add_argument('--tabu-iters', type=int, default=_gp.get('tabu_iters', 2000))
    ap.add_argument('--tabu-tenure', type=int, default=_gp.get('tabu_tenure', 20))
    ap.add_argument('--tabu-patience', type=int, default=_gp.get('tabu_patience', 500))
    ap.add_argument('--sa-iters', type=int, default=_gp.get('sa_iters', 5000))
    ap.add_argument('--kempe-iters', type=int, default=_gp.get('kempe_iters', 3000))
    ap.add_argument('--alns-iters', type=int, default=_gp.get('alns_iters', 2000))
    ap.add_argument('--gd-iters', type=int, default=_gp.get('gd_iters', 5000))
    ap.add_argument('--abc-pop', type=int, default=_gp.get('abc_pop', 30))
    ap.add_argument('--abc-iters', type=int, default=_gp.get('abc_iters', 3000))
    ap.add_argument('--ga-pop', type=int, default=_gp.get('ga_pop', 50))
    ap.add_argument('--ga-iters', type=int, default=_gp.get('ga_iters', 500))
    ap.add_argument('--lahc-iters', type=int, default=_gp.get('lahc_iters', 5000))
    ap.add_argument('--lahc-list', type=int, default=_gp.get('lahc_list', 0))
    ap.add_argument('--woa-pop', type=int, default=_gp.get('woa_pop', 25))
    ap.add_argument('--woa-iters', type=int, default=_gp.get('woa_iters', 3000))
    ap.add_argument('--hho-pop', type=int, default=_gp.get('hho_pop', 20))
    ap.add_argument('--hho-iters', type=int, default=_gp.get('hho_iters', 500))
    ap.add_argument('--cpsat-time', type=float, default=_gp.get('cpsat_time', 60.0))
    ap.add_argument('--vns-iters', type=int, default=_gp.get('vns_iters', 5000))
    ap.add_argument('--vns-budget', type=int, default=_gp.get('vns_budget', 0))
    ap.add_argument('--ip-time', type=int, default=120,
                    help='IP (Python CP-SAT) time limit in seconds (default: 120)')
    ap.add_argument('--ip-max-exams', type=int, default=900,
                    help='Skip IP when instance has more exams than this (default: 900)')
    ap.add_argument('--ip-workers', type=int, default=0,
                    help='IP CP-SAT parallel workers; 0 = all cores (default: 0)')
    ap.add_argument('--ip-warmstart', type=str, default=None,
                    help='Path to ITC-format .sln to use as IP warm-start hint')

    # Param management
    ap.add_argument('--show-params', action='store_true',
                    help='Print active param defaults and exit')
    ap.add_argument('--rollback-params', type=int, metavar='VERSION',
                    help='Rollback golden params to a specific version and exit')
    ap.add_argument('--no-auto-update', action='store_true',
                    help='Do not auto-update golden params after tuning')
    ap.add_argument('--force-update', action='store_true',
                    help='Force-update golden params even if checks fail')

    # Auto-tuner options
    ap.add_argument('--tune-workers', type=int, default=6,
                    help='Max parallel workers for tuning (default: 6)')
    ap.add_argument('--param-trials', type=int, default=15,
                    help='Param tuning trials per algo (default: 15)')
    ap.add_argument('--chain-pop', type=int, default=12,
                    help='Chain population size (default: 12)')
    ap.add_argument('--chain-rounds', type=int, default=5,
                    help='Chain tournament rounds (default: 5)')
    ap.add_argument('--resume', action='store_true',
                    help='Resume auto-tuner from checkpoint')

    args = ap.parse_args()

    # --show-params: print active defaults and exit
    if args.show_params:
        from tooling.tuned_params import load_params, list_versions
        params = load_params()
        meta = load_metadata()
        print("Active algorithm defaults:")
        for algo, p in sorted(params.items()):
            print(f"  {algo:<8} {p}")
        if meta:
            print(f"\nSource: tuned_params.json (v{meta.get('version', '?')}, "
                  f"{meta.get('timestamp', '?')})")
            print(f"Aggregate score: {meta.get('aggregate_score', '?')}")
        else:
            print("\nSource: hardcoded fallback (no tuned_params.json)")
        versions = list_versions()
        if versions:
            print(f"\nVersion history ({len(versions)} entries):")
            for v, ts, sc, src in versions[-5:]:
                print(f"  v{v} {ts} score={sc} ({src})")
        return

    # --rollback-params: restore a past version and exit
    if args.rollback_params is not None:
        from tooling.tuned_params import rollback
        v = rollback(args.rollback_params)
        if v is not None:
            print(f"Rolled back to version {v}")
        else:
            print(f"Version {args.rollback_params} not found in log")
        return

    verbose = not args.quiet

    # Optional warm-start solution for IP (parsed once, reused per run)
    ip_warm_solution = None
    if args.ip_warmstart:
        if not args.dataset:
            print("Warning: --ip-warmstart requires --dataset; ignoring.")
        else:
            try:
                problem_for_hint = parse_itc2007_exam(args.dataset, limit=args.limit)
                ip_warm_solution = read_solution_itc2007(
                    args.ip_warmstart, problem_for_hint)
                if verbose:
                    print(f"[IP] Loaded warm-start: {args.ip_warmstart}")
            except Exception as e:
                print(f"Warning: failed to load warm-start {args.ip_warmstart}: {e}")

    kw = dict(tabu_iters=args.tabu_iters, tabu_tenure=args.tabu_tenure,
              tabu_patience=args.tabu_patience,
              sa_iters=args.sa_iters, kempe_iters=args.kempe_iters,
              alns_iters=args.alns_iters, gd_iters=args.gd_iters,
              abc_pop=args.abc_pop, abc_iters=args.abc_iters,
              ga_pop=args.ga_pop, ga_iters=args.ga_iters,
              lahc_iters=args.lahc_iters, lahc_list=args.lahc_list,
              woa_pop=args.woa_pop, woa_iters=args.woa_iters,
              hho_pop=args.hho_pop, hho_iters=args.hho_iters,
              cpsat_time=args.cpsat_time,
              vns_iters=args.vns_iters, vns_budget=args.vns_budget,
              ip_time=args.ip_time, ip_max_exams=args.ip_max_exams,
              ip_workers=args.ip_workers, ip_warm_solution=ip_warm_solution,
              seed=args.seed)

    # Resolve output directory via batch manager
    if args.mode == 'batches':
        bm = BatchManager()
        bm.print_batches()
        return

    if args.no_batch:
        output_dir = args.output
    else:
        bm = BatchManager(args.output)
        if args.load_batch:
            output_dir = bm.load_batch(args.load_batch)
        else:
            output_dir = bm.new_batch(args.batch)

    if args.mode == 'tune':
        from tooling.auto_tuner import AutoTuner
        datasets = []
        if args.dataset:
            datasets.append(args.dataset)
        else:
            # Default to all ITC sets in global mode
            import glob as g
            datasets = sorted(g.glob('instances/exam_comp_set*.exam'))
        if not datasets:
            print("Error: --dataset or instances/exam_comp_set*.exam required for tune mode")
            return
        tune_dir = os.path.join(output_dir, 'tuning')
        AutoTuner(
            datasets=datasets, output_dir=tune_dir,
            max_workers=args.tune_workers,
            param_trials=args.param_trials,
            chain_pop=args.chain_pop,
            chain_rounds=args.chain_rounds,
            seed=args.seed,
            auto_update=not args.no_auto_update,
            force_update=args.force_update,
        ).run(resume=args.resume)
        return

    if args.dataset:
        run_on_dataset(args.dataset, limit=args.limit, algo=args.algo,
                       verbose=verbose, output_dir=output_dir, **kw)
    elif args.mode == 'demo':
        run_demo(size=args.size, algo=args.algo, verbose=verbose,
                 output_dir=output_dir, **kw)
    elif args.mode == 'plot':
        from utils.plotting import generate_all_plots
        generate_all_plots(output_dir)


if __name__ == '__main__':
    main()
