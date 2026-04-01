"""
C++ Solver Bridge — calls the compiled C++ exam_solver binary via subprocess
and returns results in the same format as the Python algorithms.

Falls back to Python implementations if the binary is not found.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from data.models import ProblemInstance, Solution
from data.fast_eval import FastEvaluator, EvalBreakdown


def _find_binary():
    """Locate the compiled C++ exam_solver binary."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cpp', 'exam_solver'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpp', 'exam_solver'),
        os.path.join('.', 'cpp', 'exam_solver'),
        os.path.join('.', 'exam_solver'),
        'exam_solver',
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    return None


def _build_binary():
    """Attempt to compile the C++ solver."""
    src_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cpp', 'main.cpp'),
        os.path.join('.', 'cpp', 'main.cpp'),
    ]
    src = None
    for c in src_candidates:
        if os.path.isfile(c):
            src = os.path.abspath(c)
            break

    if src is None:
        return None

    src_dir = os.path.dirname(src)
    out = os.path.join(src_dir, 'exam_solver')
    print(f"[C++ Bridge] Compiling {src}...")
    try:
        result = subprocess.run(
            ['g++', '-O3', '-std=c++20', '-o', out, src],
            capture_output=True, text=True, timeout=60,
            cwd=src_dir,  # so headers are found
        )
        if result.returncode == 0:
            print(f"[C++ Bridge] Compiled successfully: {out}")
            return out
        else:
            print(f"[C++ Bridge] Compilation failed:\n{result.stderr}")
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[C++ Bridge] Compilation error: {e}")
        return None


def _get_binary():
    """Find or build the C++ binary."""
    binary = _find_binary()
    if binary:
        return binary
    return _build_binary()


def _parse_eval(d: dict) -> EvalBreakdown:
    """Convert a JSON dict to an EvalBreakdown object."""
    eb = EvalBreakdown()
    eb.conflicts = d.get('conflicts', 0)
    eb.room_occupancy = d.get('room_occupancy', 0)
    eb.period_utilisation = d.get('period_utilisation', 0)
    eb.period_related = d.get('period_related', 0)
    eb.room_related = d.get('room_related', 0)
    eb.two_in_a_row = d.get('two_in_a_row', 0)
    eb.two_in_a_day = d.get('two_in_a_day', 0)
    eb.period_spread = d.get('period_spread', 0)
    eb.non_mixed_durations = d.get('non_mixed_durations', 0)
    eb.front_load = d.get('front_load', 0)
    eb.period_penalty = d.get('period_penalty', 0)
    eb.room_penalty = d.get('room_penalty', 0)
    return eb


def _load_solution(problem: ProblemInstance, sln_path: str) -> Solution:
    """Load a .sln file back into a Solution object."""
    sol = Solution(problem)
    if not os.path.isfile(sln_path):
        return sol
    with open(sln_path) as f:
        for eid, line in enumerate(f):
            parts = line.strip().split(',')
            if len(parts) >= 2:
                pid = int(parts[0].strip())
                rid = int(parts[1].strip())
                if pid >= 0 and rid >= 0 and eid < problem.num_exams():
                    sol.assign(eid, pid, rid)
    return sol


def _run_python_fallback(problem, algo='all', tabu_iters=200, tabu_tenure=15,
                         tabu_patience=50, hho_pop=30, hho_iters=100,
                         seed=42, verbose=True):
    """Run Python implementations when C++ binary is unavailable."""
    from algorithms.greedy import solve_greedy
    from algorithms.tabu_search import solve_tabu
    from algorithms.hho import solve_hho

    results = {}

    if algo in ('all', 'greedy'):
        if verbose:
            print(f"\n{'─'*50}\nGreedy (Python)...")
        r = solve_greedy(problem, verbose=verbose, seed=seed)
        r['evaluation'] = _py_to_eval(r['evaluation'])
        results['Greedy'] = r

    if algo in ('all', 'tabu'):
        if verbose:
            print(f"\n{'─'*50}\nTabu Search (Python, iters={tabu_iters})...")
        r = solve_tabu(problem, max_iterations=tabu_iters,
                       tabu_tenure=tabu_tenure, patience=tabu_patience,
                       seed=seed, verbose=verbose)
        r['evaluation'] = _py_to_eval(r['evaluation'])
        results['Tabu Search'] = r

    if algo in ('all', 'hho'):
        if verbose:
            print(f"\n{'─'*50}\nHHO (Python, pop={hho_pop}, iters={hho_iters})...")
        r = solve_hho(problem, population_size=hho_pop,
                      max_iterations=hho_iters, seed=seed, verbose=verbose)
        r['evaluation'] = _py_to_eval(r['evaluation'])
        results['HHO'] = r

    return results


def _py_to_eval(ev):
    """Convert EvalBreakdown to be compatible with the bridge's expected format."""
    # EvalBreakdown already has .feasible, .hard, .soft, and all component attrs
    return ev


def run_cpp_solver(
    exam_filepath: str,
    problem: ProblemInstance,
    algo: str = 'all',
    limit: int = 0,
    tabu_iters: int = 200,
    tabu_tenure: int = 15,
    tabu_patience: int = 50,
    hho_pop: int = 30,
    hho_iters: int = 100,
    seed: int = 42,
    output_dir: str = 'results',
    verbose: bool = True,
) -> dict:
    """
    Run the C++ solver on the given .exam file.

    Returns:
        dict: {algo_name: {'solution': Solution, 'runtime': float,
               'evaluation': EvalBreakdown, 'algorithm': str, 'iterations': int}}
    """
    binary = _get_binary()
    if binary is None:
        print("[C++ Bridge] Binary not available, using Python solvers.")
        return _run_python_fallback(
            problem, algo=algo,
            tabu_iters=tabu_iters, tabu_tenure=tabu_tenure,
            tabu_patience=tabu_patience, hho_pop=hho_pop,
            hho_iters=hho_iters, seed=seed, verbose=verbose)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        binary, exam_filepath,
        '--algo', algo,
        '--limit', str(limit),
        '--tabu-iters', str(tabu_iters),
        '--tabu-tenure', str(tabu_tenure),
        '--tabu-patience', str(tabu_patience),
        '--hho-pop', str(hho_pop),
        '--hho-iters', str(hho_iters),
        '--seed', str(seed),
        '--output-dir', output_dir,
    ]
    if verbose:
        cmd.append('--verbose')

    if verbose:
        print(f"[C++ Bridge] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute timeout
        )
    except subprocess.TimeoutExpired:
        print("[C++ Bridge] Solver timed out after 600s")
        return None

    # Print stderr (verbose logs) to console
    if result.stderr and verbose:
        for line in result.stderr.strip().split('\n'):
            print(f"  {line}")

    if result.returncode != 0:
        print(f"[C++ Bridge] Solver failed (code {result.returncode})")
        if result.stderr:
            print(result.stderr)
        return None

    # Parse JSON from stdout
    try:
        raw_results = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"[C++ Bridge] JSON parse error: {e}")
        print(f"[C++ Bridge] stdout was: {result.stdout[:500]}")
        return None

    # Convert to the same format as Python algorithms
    ne = problem.num_exams()
    algo_results = {}
    for r in raw_results:
        name = r['algorithm']
        ev = _parse_eval(r)

        # Load solution from .sln file
        safe_name = name.lower().replace(' ', '_')
        sln_path = os.path.join(output_dir, f"solution_{safe_name}_{ne}.sln")
        sol = _load_solution(problem, sln_path)

        algo_results[name] = {
            'solution': sol,
            'runtime': r['runtime'],
            'evaluation': ev,
            'algorithm': name,
            'iterations': r.get('iterations', 0),
        }

    return algo_results