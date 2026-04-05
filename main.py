"""
Exam Scheduling Benchmark Suite

Greedy, Tabu Search, HHO → C++ (100–200x faster)
IP solver → Python (PuLP/CBC)

Usage:
  python main.py
  python main.py --dataset datasets/exam_comp_set4.exam
  python main.py --dataset data.exam --algo tabu
  python main.py --dataset data.exam --algo ip --limit 100
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.models import ProblemInstance, Solution
from data.generator import generate_synthetic, write_itc2007_format
from data.parser import parse_itc2007_exam, write_solution_itc2007
from algorithms.cpp_bridge import run_cpp_solver
from utils.plotting import plot_soft_constraint_breakdown
from utils.batch_manager import BatchManager

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

    problem = generate_synthetic(
        num_exams=size, student_ratio=7.0, conflict_density=0.15,
        num_rooms=max(3, size // 20), room_capacity=max(50, int(size * 0.4)),
        seed=kwargs.get('seed', 42),
    )
    print(problem.summary())

    os.makedirs("datasets", exist_ok=True)
    exam_path = f"datasets/synthetic_{size}.exam"
    write_itc2007_format(problem, exam_path)

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # C++ algorithms (greedy/tabu/hho)
    if algo != 'ip':
        cpp_algo = algo if algo else 'all'
        cpp_results = run_cpp_solver(
            exam_path, problem, algo=cpp_algo, output_dir=output_dir,
            verbose=verbose, **kwargs)
        if cpp_results:
            results.update(cpp_results)

    # IP solver (Python)
    if (algo is None or algo == 'ip') and HAS_IP and problem.num_exams() <= 150:
        if verbose:
            print(f"\n{'─'*50}\nInteger Programming (Python/PuLP, limit=60s)...")
        r = solve_ip(problem, time_limit=60, verbose=verbose)
        results['IP'] = r

    _print_comparison(results)
    bd = _save_soft_breakdown(results, output_dir)
    print(f"\n{'='*72}\n  SOFT CONSTRAINT BREAKDOWN\n{'='*72}")
    for name, b in bd.items():
        total = sum(b.values())
        parts = [f"{k}={v}" for k, v in b.items() if v > 0]
        print(f"{name:<25} total={total:>6}  [{', '.join(parts)}]")


def run_on_dataset(filepath, limit=0, algo=None, verbose=True, output_dir='results', **kwargs):
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
        if ne <= 500:
            if verbose:
                print(f"\n{'─'*50}\nInteger Programming (CP-SAT/HiGHS, limit=120s)...")
            r = solve_ip(problem, time_limit=120, verbose=verbose)
            results['IP'] = r
        elif verbose:
            print(f"\n[IP] Skipped (n={ne} > 500)")

    _print_comparison(results)
    _save_soft_breakdown(results, output_dir)

    sln_dir = os.path.join(output_dir, "solutions")
    os.makedirs(sln_dir, exist_ok=True)
    for name, r in results.items():
        safe = name.lower().replace(' ', '_')
        write_solution_itc2007(r['solution'], os.path.join(sln_dir, f"solution_{safe}_{problem.num_exams()}.sln"))
    print(f"\nSolutions saved to {sln_dir}/")


def main():
    ap = argparse.ArgumentParser(
        description="Exam Scheduling Benchmark Suite v3 (C++ Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithms:  greedy, tabu, hho, kempe, sa, alns, gd → C++  |  ip → Python (OR-Tools)

Examples:
  python main.py                                        # Demo 50 exams
  python main.py --dataset data.exam                    # All algos
  python main.py --dataset data.exam --algo tabu        # Tabu only
  python main.py --dataset data.exam --algo ip --limit 100
  python main.py --mode plot
        """)

    ap.add_argument('--mode', choices=['demo', 'plot', 'batches'], default='demo')
    ap.add_argument('--algo', choices=['greedy', 'ip', 'tabu', 'hho', 'kempe', 'sa', 'alns', 'gd'])
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
    ap.add_argument('--tabu-iters', type=int, default=2000)
    ap.add_argument('--tabu-patience', type=int, default=500)
    ap.add_argument('--hho-pop', type=int, default=50)
    ap.add_argument('--hho-iters', type=int, default=500)
    ap.add_argument('--sa-iters', type=int, default=5000)
    ap.add_argument('--kempe-iters', type=int, default=3000)
    ap.add_argument('--alns-iters', type=int, default=2000)
    ap.add_argument('--gd-iters', type=int, default=5000)

    args = ap.parse_args()
    verbose = not args.quiet
    kw = dict(tabu_iters=args.tabu_iters, tabu_patience=args.tabu_patience,
              hho_pop=args.hho_pop, hho_iters=args.hho_iters,
              sa_iters=args.sa_iters, kempe_iters=args.kempe_iters,
              alns_iters=args.alns_iters, gd_iters=args.gd_iters,
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
