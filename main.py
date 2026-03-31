#!/usr/bin/env python3
"""
Exam Scheduling Benchmark

Usage:
  python main.py --mode demo --size 200
  python main.py --dataset exam_comp_set4.exam
  python main.py --dataset exam_comp_set1.exam --limit 100
  python main.py --dataset data.exam --algo tabu
  python main.py --dataset data.exam --algo hho --limit 200
  python main.py --mode benchmark --sizes 50,100,200
  python main.py --mode params --size 100
  python main.py --mode plot
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.models import ProblemInstance, Solution
from data.generator import generate_synthetic, write_itc2007_format
from data.parser import parse_itc2007_exam, write_solution_itc2007
from data.fast_eval import FastEvaluator
from algorithms.greedy import solve_greedy
from algorithms.tabu_search import solve_tabu
from algorithms.hho import solve_hho
from utils.plotting import generate_all_plots, plot_soft_constraint_breakdown

try:
    from algorithms.ip_solver import solve_ip
    HAS_IP = True
except ImportError:
    HAS_IP = False


ALGO_MAP = {
    'greedy': ('Greedy', lambda p, **kw: solve_greedy(p, **kw)),
    'ip':     ('IP', lambda p, **kw: solve_ip(p, **kw) if HAS_IP else None),
    'tabu':   ('Tabu Search', lambda p, **kw: solve_tabu(p, **kw)),
    'hho':    ('HHO', lambda p, **kw: solve_hho(p, **kw)),
}


def _run_single_algorithm(problem, algo_name, ip_time=300, tabu_iters=200,
                          hho_pop=30, hho_iters=100, verbose=True):
    """Run one specific algorithm and return results dict."""
    results = {}
    key = algo_name.lower()

    if key == 'greedy':
        if verbose:
            print(f"\n{'─'*50}\nGreedy Heuristic...")
        r = solve_greedy(problem, verbose=verbose)
        results['Greedy'] = r

    elif key == 'ip':
        if not HAS_IP:
            print("[IP] PuLP not installed. Run: pip install pulp")
            return results
        if verbose:
            print(f"\n{'─'*50}\nInteger Programming (limit={ip_time}s)...")
        r = solve_ip(problem, time_limit=ip_time, verbose=verbose)
        results['IP'] = r

    elif key == 'tabu':
        if verbose:
            print(f"\n{'─'*50}\nTabu Search ({tabu_iters} iters)...")
        r = solve_tabu(problem, max_iterations=tabu_iters, verbose=verbose)
        results['Tabu Search'] = r

    elif key == 'hho':
        if verbose:
            print(f"\n{'─'*50}\nHarris Hawks Optimization (pop={hho_pop}, {hho_iters} iters)...")
        r = solve_hho(problem, population_size=hho_pop, max_iterations=hho_iters, verbose=verbose)
        results['HHO'] = r

    else:
        print(f"Unknown algorithm: {algo_name}")
        print(f"Choose from: greedy, ip, tabu, hho")

    return results


def _run_all_algorithms(problem, ip_limit=150, ip_time=60, tabu_iters=200,
                        hho_pop=30, hho_iters=100, verbose=True):
    """Run all algorithms on a problem and return results dict."""
    results = {}
    fe = FastEvaluator(problem)

    # Greedy
    if verbose:
        print(f"\n{'─'*50}")
        print("Greedy Heuristic...")
    r = solve_greedy(problem, verbose=verbose)
    results['Greedy'] = r

    # IP
    if HAS_IP and problem.num_exams() <= ip_limit:
        if verbose:
            print(f"\n{'─'*50}")
            print(f"Integer Programming (limit={ip_time}s)...")
        r = solve_ip(problem, time_limit=ip_time, verbose=verbose)
        results['IP'] = r
    elif verbose:
        reason = "PuLP not installed" if not HAS_IP else f"n={problem.num_exams()} > {ip_limit}"
        print(f"\n[IP] Skipped ({reason})")

    # Tabu
    if verbose:
        print(f"\n{'─'*50}")
        print(f"Tabu Search ({tabu_iters} iters)...")
    r = solve_tabu(problem, max_iterations=tabu_iters, verbose=verbose)
    results['Tabu Search'] = r

    # HHO
    if verbose:
        print(f"\n{'─'*50}")
        print(f"Harris Hawks Optimization (pop={hho_pop}, {hho_iters} iters)...")
    r = solve_hho(problem, population_size=hho_pop, max_iterations=hho_iters, verbose=verbose)
    results['HHO'] = r

    return results


def _print_comparison(results):
    """Print side-by-side comparison table."""
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
    """Extract soft constraint breakdown from each algorithm and save."""
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
    path = os.path.join(output_dir, "soft_breakdown.json")
    with open(path, 'w') as f:
        json.dump(breakdown, f, indent=2)

    # Generate the plot
    plot_soft_constraint_breakdown(breakdown, output_dir)
    return breakdown


def run_demo(size=50, algo=None, verbose=True):
    """Demo: run all or one algorithm on a synthetic instance."""
    print(f"\n{'='*72}")
    print(f"  DEMO: {size}-exam synthetic instance")
    print(f"{'='*72}\n")

    problem = generate_synthetic(
        num_exams=size, student_ratio=7.0, conflict_density=0.15,
        num_rooms=max(3, size // 20), room_capacity=max(50, int(size * 0.4)),
        seed=42,
    )
    print(problem.summary())

    os.makedirs("datasets", exist_ok=True)
    write_itc2007_format(problem, f"datasets/synthetic_{size}.exam")

    if algo:
        results = _run_single_algorithm(problem, algo, verbose=verbose)
    else:
        results = _run_all_algorithms(problem, verbose=verbose)
    _print_comparison(results)

    os.makedirs("results", exist_ok=True)
    breakdown = _save_soft_breakdown(results, "results")

    # Print soft breakdown
    print(f"\n{'='*72}")
    print("  SOFT CONSTRAINT BREAKDOWN")
    print(f"{'='*72}")
    for name, bd in breakdown.items():
        total = sum(bd.values())
        components = [f"{k}={v}" for k, v in bd.items() if v > 0]
        print(f"{name:<25} total={total:>6}  [{', '.join(components)}]")


def run_on_dataset(filepath, limit=0, algo=None, verbose=True):
    """Run all or one algorithm on an ITC 2007 .exam file with optional exam limit."""
    limit_str = f" (limit={limit} exams)" if limit > 0 else " (full dataset)"
    algo_str = f" [{algo.upper()}]" if algo else ""
    print(f"\n{'='*72}")
    print(f"  DATASET: {os.path.basename(filepath)}{limit_str}{algo_str}")
    print(f"{'='*72}\n")

    problem = parse_itc2007_exam(filepath, limit=limit)
    print(problem.summary())

    if algo:
        results = _run_single_algorithm(problem, algo, verbose=verbose)
    else:
        results = _run_all_algorithms(problem, verbose=verbose)
    _print_comparison(results)

    os.makedirs("results", exist_ok=True)
    _save_soft_breakdown(results, "results")

    # Save solutions
    for name, r in results.items():
        safe = name.lower().replace(' ', '_')
        sol_path = f"results/solution_{safe}_{problem.num_exams()}.sln"
        write_solution_itc2007(r['solution'], sol_path)
    print(f"\nSolutions saved to results/")


def main():
    parser = argparse.ArgumentParser(
        description="Exam Scheduling Benchmark Suite v2 (Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --mode demo --size 200
  python main.py --dataset data.exam
  python main.py --dataset data.exam --algo tabu
  python main.py --dataset data.exam --algo hho --limit 200
  python main.py --dataset data.exam --algo ip --limit 100
  python main.py --mode benchmark --sizes 50,100,200
  python main.py --mode plot

Algorithms: greedy, ip, tabu, hho
        """
    )

    parser.add_argument('--mode', choices=['demo', 'benchmark', 'params', 'plot'],
                        default='demo')
    parser.add_argument('--algo', type=str, default=None,
                        choices=['greedy', 'ip', 'tabu', 'hho'],
                        help='Run a single algorithm (default: run all)')
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--sizes', type=str, default='50,100,200')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to ITC 2007 .exam file')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of exams loaded from dataset '
                             '(0=all, 50/100/200 for incremental testing)')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()
    verbose = not args.quiet

    if args.dataset:
        run_on_dataset(args.dataset, limit=args.limit, algo=args.algo, verbose=verbose)
        return

    if args.mode == 'demo':
        run_demo(size=args.size, algo=args.algo, verbose=verbose)
    elif args.mode == 'benchmark':
        from utils.benchmark import run_benchmark_suite
        sizes = [int(s.strip()) for s in args.sizes.split(',')]
        results = run_benchmark_suite(sizes=sizes, num_trials=args.trials,
                                       output_dir=args.output, verbose=verbose)
        generate_all_plots(args.output)
    elif args.mode == 'params':
        from utils.benchmark import run_parameter_study
        run_parameter_study(problem_size=args.size, output_dir=args.output, verbose=verbose)
        generate_all_plots(args.output)
    elif args.mode == 'plot':
        generate_all_plots(args.output)


if __name__ == '__main__':
    main()