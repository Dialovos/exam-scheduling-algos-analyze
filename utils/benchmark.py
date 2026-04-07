"""
Q1: How does runtime scale with input size {50..1000}?
Q2: Quality vs time trade-off on a fixed time frame?
Q3: Parameter tuning effects for Tabu and HHO?
Q4: Memory usage scaling, especially for IP?
"""

import json
import csv
import time
import os
import tracemalloc
from typing import Callable
from core.models import ProblemInstance
from core.generator import generate_synthetic
from core.evaluator import evaluate
from algorithms.greedy import solve_greedy
from algorithms.tabu_search import solve_tabu
from algorithms.hho import solve_hho

# Conditional IP import
try:
    from algorithms.ip_solver import solve_ip
    HAS_IP = True
except ImportError:
    HAS_IP = False


def run_single(
    algorithm_fn: Callable,
    problem: ProblemInstance,
    algo_name: str,
    algo_kwargs: dict = None,
    verbose: bool = False,
) -> dict:
    """Run a single algorithm on a single problem and return metrics."""
    if algo_kwargs is None:
        algo_kwargs = {}

    tracemalloc.start()
    try:
        result = algorithm_fn(problem, verbose=verbose, **algo_kwargs)
    except Exception as e:
        tracemalloc.stop()
        return {
            'algorithm': algo_name,
            'error': str(e),
            'runtime': float('inf'),
            'hard_violations': float('inf'),
            'soft_penalty': float('inf'),
            'feasible': False,
            'memory_peak_mb': 0,
        }

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ev = result['evaluation']
    return {
        'algorithm': algo_name,
        'runtime': result['runtime'],
        'hard_violations': ev.hard_violations,
        'soft_penalty': ev.soft_penalty,
        'feasible': ev.is_feasible,
        'memory_peak_mb': result.get('memory_peak_mb', peak_mem / (1024 * 1024)),
        'iterations': result.get('iterations', None),
        'solver_status': result.get('solver_status', None),
        'error': None,
    }


def run_benchmark_suite(
    sizes: list[int] = None,
    num_trials: int = 7,
    ip_time_limit: int = 120,
    tabu_iterations: int = 200,
    hho_population: int = 30,
    hho_iterations: int = 100,
    output_dir: str = "results",
    verbose: bool = True,
) -> list[dict]:
    """
    Args:
        sizes: List of exam counts to test.
        num_trials: Number of trials per stochastic algorithm (greedy is 1).
        ip_time_limit: Time limit for IP solver in seconds.
        tabu_iterations: Max iterations for Tabu Search.
        hho_population: Population size for HHO.
        hho_iterations: Max iterations for HHO.
        output_dir: Directory to save results.
        verbose: Print progress.

    Returns:
        List of result dicts with all metrics.
    """
    if sizes is None:
        sizes = [50, 100, 200]

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for n in sizes:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  BENCHMARK: {n} exams")
            print(f"{'='*60}")

        # Generate problem instance
        problem = generate_synthetic(
            num_exams=n,
            student_ratio=7.0,
            conflict_density=0.15,
            num_rooms=max(3, n // 20),
            room_capacity=max(50, int(n * 0.4)),
            seed=42 + n,
        )

        if verbose:
            print(problem.summary())

        # --- Greedy (deterministic, 1 trial) ---
        if verbose:
            print(f"\n--- Greedy Heuristic (1 trial) ---")
        result = run_single(solve_greedy, problem, "Greedy", verbose=verbose)
        result['num_exams'] = n
        result['trial'] = 0
        all_results.append(result)
        if verbose:
            print(f"  Runtime: {result['runtime']:.4f}s, "
                  f"Feasible: {result['feasible']}, "
                  f"Soft: {result['soft_penalty']}")

        # --- IP Solver ---
        if HAS_IP and n <= 200:  # IP only feasible for smaller instances
            if verbose:
                print(f"\n--- Integer Programming (1 trial, limit={ip_time_limit}s) ---")
            result = run_single(
                solve_ip, problem, "IP",
                algo_kwargs={'time_limit': ip_time_limit},
                verbose=verbose,
            )
            result['num_exams'] = n
            result['trial'] = 0
            all_results.append(result)
            if verbose:
                print(f"  Runtime: {result['runtime']:.2f}s, "
                      f"Feasible: {result['feasible']}, "
                      f"Soft: {result['soft_penalty']}, "
                      f"Status: {result.get('solver_status', 'N/A')}")
        elif n > 200 and verbose:
            print(f"\n--- IP Solver SKIPPED (n={n} too large) ---")

        # --- Tabu Search (multiple trials) ---
        if verbose:
            print(f"\n--- Tabu Search ({num_trials} trials) ---")
        for trial in range(num_trials):
            result = run_single(
                solve_tabu, problem, "Tabu Search",
                algo_kwargs={
                    'max_iterations': tabu_iterations,
                    'tabu_tenure': 15,
                    'patience': 50,
                    'seed': 42 + trial * 1000 + n,
                },
                verbose=False,
            )
            result['num_exams'] = n
            result['trial'] = trial
            all_results.append(result)
            if verbose:
                print(f"  Trial {trial}: runtime={result['runtime']:.2f}s, "
                      f"feasible={result['feasible']}, soft={result['soft_penalty']}")

        # --- HHO (multiple trials) ---
        if verbose:
            print(f"\n--- Harris Hawks Optimization ({num_trials} trials) ---")
        for trial in range(num_trials):
            result = run_single(
                solve_hho, problem, "HHO",
                algo_kwargs={
                    'population_size': hho_population,
                    'max_iterations': hho_iterations,
                    'seed': 42 + trial * 2000 + n,
                },
                verbose=False,
            )
            result['num_exams'] = n
            result['trial'] = trial
            all_results.append(result)
            if verbose:
                print(f"  Trial {trial}: runtime={result['runtime']:.2f}s, "
                      f"feasible={result['feasible']}, soft={result['soft_penalty']}")

    # --- Save results ---
    results_file = os.path.join(output_dir, "benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    csv_file = os.path.join(output_dir, "benchmark_results.csv")
    if all_results:
        keys = all_results[0].keys()
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)

    if verbose:
        print(f"\nResults saved to {results_file} and {csv_file}")

    return all_results


def run_parameter_study(
    problem_size: int = 100,
    output_dir: str = "results",
    verbose: bool = True,
) -> list[dict]:
    """
    Run parameter sensitivity study for Tabu Search and HHO.
    Addresses research question Q3 from the project plan.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    problem = generate_synthetic(
        num_exams=problem_size,
        student_ratio=7.0,
        conflict_density=0.15,
        num_rooms=max(3, problem_size // 20),
        room_capacity=max(50, int(problem_size * 0.4)),
        seed=42 + problem_size,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PARAMETER STUDY: {problem_size} exams")
        print(f"{'='*60}")

    # --- Tabu Search: vary iterations ---
    tabu_iterations_range = [5, 10, 25, 50, 100, 200]
    if verbose:
        print(f"\n--- Tabu Search: varying iterations {tabu_iterations_range} ---")
    for iters in tabu_iterations_range:
        result = run_single(
            solve_tabu, problem, "Tabu Search",
            algo_kwargs={'max_iterations': iters, 'tabu_tenure': 15, 'patience': iters, 'seed': 42},
            verbose=False,
        )
        result['num_exams'] = problem_size
        result['param_name'] = 'tabu_iterations'
        result['param_value'] = iters
        results.append(result)
        if verbose:
            print(f"  iters={iters}: runtime={result['runtime']:.2f}s, "
                  f"soft={result['soft_penalty']}")

    # --- HHO: vary population size ---
    hho_pop_range = [10, 20, 30, 50, 100]
    if verbose:
        print(f"\n--- HHO: varying population {hho_pop_range} ---")
    for pop in hho_pop_range:
        result = run_single(
            solve_hho, problem, "HHO",
            algo_kwargs={'population_size': pop, 'max_iterations': 50, 'seed': 42},
            verbose=False,
        )
        result['num_exams'] = problem_size
        result['param_name'] = 'hho_population'
        result['param_value'] = pop
        results.append(result)
        if verbose:
            print(f"  pop={pop}: runtime={result['runtime']:.2f}s, "
                  f"soft={result['soft_penalty']}")

    # Save
    param_file = os.path.join(output_dir, "parameter_study.json")
    with open(param_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\nParameter study saved to {param_file}")

    return results
