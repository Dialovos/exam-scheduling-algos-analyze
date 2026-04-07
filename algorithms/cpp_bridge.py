"""
C++ Solver Bridge — calls the compiled C++ exam_solver binary via subprocess
and returns results in the same format as the Python algorithms.

Falls back to Python implementations if the binary is not found.

Quick usage (for notebooks):
    from algorithms.cpp_bridge import run_solver
    results = run_solver("datasets/exam_comp_set4.exam", algo="all")
    results = run_solver(problem_instance, algo="tabu", tabu_iters=2000)
"""

import json
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

from core.models import ProblemInstance, Solution
from core.fast_eval import FastEvaluator, EvalBreakdown
from tooling.tuned_params import load_params_flat as _load_golden_flat

# Load golden defaults once at import time
_GP = _load_golden_flat()


def run_solver(
    problem_or_path,
    algo: str = 'all',
    tabu_iters: int = _GP.get('tabu_iters', 2000),
    tabu_tenure: int = _GP.get('tabu_tenure', 20),
    tabu_patience: int = _GP.get('tabu_patience', 500),
    hho_pop: int = _GP.get('hho_pop', 30),
    hho_iters: int = _GP.get('hho_iters', 200),
    sa_iters: int = _GP.get('sa_iters', 5000),
    kempe_iters: int = _GP.get('kempe_iters', 3000),
    alns_iters: int = _GP.get('alns_iters', 2000),
    gd_iters: int = _GP.get('gd_iters', 5000),
    abc_pop: int = _GP.get('abc_pop', 30),
    abc_iters: int = _GP.get('abc_iters', 3000),
    ga_pop: int = _GP.get('ga_pop', 50),
    ga_iters: int = _GP.get('ga_iters', 500),
    lahc_iters: int = _GP.get('lahc_iters', 5000),
    lahc_list: int = _GP.get('lahc_list', 0),
    ns_finalists: int = 3,
    seed: int = 42,
    limit: int = 0,
    output_dir: str = 'results',
    verbose: bool = False,
) -> dict:
    """Run C++ solver on a dataset file or ProblemInstance.

    This is the primary interface for notebooks.  It handles:
      - ProblemInstance objects (writes temp .exam file automatically)
      - Filepath strings (passes directly to C++)
      - Automatic C++ binary discovery + compilation
      - Python fallback if C++ unavailable

    Args:
        problem_or_path: either a filepath str or a ProblemInstance
        algo: "greedy", "tabu", "hho", or "all"
        (remaining args): algorithm configuration

    Returns:
        dict: {algo_name: {'solution': Solution, 'runtime': float,
               'evaluation': EvalBreakdown, 'algorithm': str, 'iterations': int}}
    """
    from core.parser import parse_itc2007_exam
    from core.generator import write_itc2007_format

    # Resolve filepath and problem
    if isinstance(problem_or_path, str):
        filepath = problem_or_path
        problem = parse_itc2007_exam(filepath, limit=limit)
    elif isinstance(problem_or_path, ProblemInstance):
        problem = problem_or_path
        if problem.conflict_matrix is None:
            problem.build_derived_data()
        # Write to temp file for C++ binary
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"_tmp_{id(problem)}.exam")
        write_itc2007_format(problem, filepath)
    else:
        raise TypeError(f"Expected str or ProblemInstance, got {type(problem_or_path)}")

    return run_cpp_solver(
        filepath, problem, algo=algo, limit=limit,
        tabu_iters=tabu_iters, tabu_tenure=tabu_tenure,
        tabu_patience=tabu_patience, hho_pop=hho_pop,
        hho_iters=hho_iters, sa_iters=sa_iters,
        kempe_iters=kempe_iters, alns_iters=alns_iters,
        gd_iters=gd_iters, abc_pop=abc_pop,
        abc_iters=abc_iters, ga_pop=ga_pop,
        ga_iters=ga_iters, lahc_iters=lahc_iters,
        lahc_list=lahc_list, ns_finalists=ns_finalists,
        seed=seed, output_dir=output_dir, verbose=verbose,
    )


def _find_binary():
    """Locate the compiled C++ exam_solver binary."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cpp', 'build', 'exam_solver'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpp', 'build', 'exam_solver'),
        os.path.join('.', 'cpp', 'build', 'exam_solver'),
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
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cpp', 'src', 'main.cpp'),
        os.path.join('.', 'cpp', 'src', 'main.cpp'),
    ]
    src = None
    for c in src_candidates:
        if os.path.isfile(c):
            src = os.path.abspath(c)
            break

    if src is None:
        return None

    src_dir = os.path.dirname(src)
    # src_dir is cpp/src; put the binary in cpp/build
    build_dir = os.path.join(os.path.dirname(src_dir), 'build')
    os.makedirs(build_dir, exist_ok=True)
    out = os.path.join(build_dir, 'exam_solver')
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


def _run_python_fallback(problem, algo='all',
                         tabu_iters=_GP.get('tabu_iters', 2000),
                         tabu_tenure=_GP.get('tabu_tenure', 20),
                         tabu_patience=_GP.get('tabu_patience', 500),
                         hho_pop=_GP.get('hho_pop', 30),
                         hho_iters=_GP.get('hho_iters', 200),
                         sa_iters=_GP.get('sa_iters', 5000),
                         kempe_iters=_GP.get('kempe_iters', 3000),
                         alns_iters=_GP.get('alns_iters', 2000),
                         gd_iters=_GP.get('gd_iters', 5000),
                         abc_pop=_GP.get('abc_pop', 30),
                         abc_iters=_GP.get('abc_iters', 3000),
                         ga_pop=_GP.get('ga_pop', 50),
                         ga_iters=_GP.get('ga_iters', 500),
                         lahc_iters=_GP.get('lahc_iters', 5000),
                         lahc_list=_GP.get('lahc_list', 0),
                         ns_finalists=3,
                         seed=42, verbose=True):
    """Run Python implementations when C++ binary is unavailable."""
    from algorithms.greedy import solve_greedy
    from algorithms.tabu_search import solve_tabu
    from algorithms.hho import solve_hho
    from algorithms.kempe_chain import solve_kempe
    from algorithms.simulated_annealing import solve_sa
    from algorithms.alns import solve_alns
    from algorithms.great_deluge import solve_great_deluge
    from algorithms.abc import solve_abc
    from algorithms.ga import solve_ga
    from algorithms.natural_selection import solve_natural_selection

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

    if algo in ('all', 'kempe'):
        if verbose:
            print(f"\n{'─'*50}\nKempe Chain (Python, iters={kempe_iters})...")
        r = solve_kempe(problem, max_iterations=kempe_iters,
                        seed=seed, verbose=verbose)
        r['evaluation'] = _py_to_eval(r['evaluation'])
        results['Kempe Chain'] = r

    if algo in ('all', 'sa'):
        if verbose:
            print(f"\n{'─'*50}\nSimulated Annealing (Python, iters={sa_iters})...")
        r = solve_sa(problem, max_iterations=sa_iters,
                     seed=seed, verbose=verbose)
        r['evaluation'] = _py_to_eval(r['evaluation'])
        results['Simulated Annealing'] = r

    if algo in ('all', 'alns'):
        if verbose:
            print(f"\n{'─'*50}\nALNS (Python, iters={alns_iters})...")
        r = solve_alns(problem, max_iterations=alns_iters,
                       seed=seed, verbose=verbose)
        r['evaluation'] = _py_to_eval(r['evaluation'])
        results['ALNS'] = r

    if algo in ('all', 'gd'):
        if verbose:
            print(f"\n{'─'*50}\nGreat Deluge (Python, iters={gd_iters})...")
        r = solve_great_deluge(problem, max_iterations=gd_iters,
                               seed=seed, verbose=verbose)
        r['evaluation'] = _py_to_eval(r['evaluation'])
        results['Great Deluge'] = r

    if algo in ('all', 'abc'):
        if verbose:
            print(f"\n{'─'*50}\nABC (Python, colony={abc_pop}, iters={abc_iters})...")
        r = solve_abc(problem, colony_size=abc_pop, max_iterations=abc_iters,
                      seed=seed, verbose=verbose)
        r['evaluation'] = _py_to_eval(r['evaluation'])
        results['ABC'] = r

    if algo in ('all', 'ga'):
        if verbose:
            print(f"\n{'─'*50}\nGenetic Algorithm (Python, pop={ga_pop}, gens={ga_iters})...")
        r = solve_ga(problem, pop_size=ga_pop, max_generations=ga_iters,
                     seed=seed, verbose=verbose)
        r['evaluation'] = _py_to_eval(r['evaluation'])
        results['Genetic Algorithm'] = r

    if algo in ('all', 'lahc'):
        if verbose:
            print(f"\n{'─'*50}\nLAHC: no Python implementation, skipping (C++ only)")

    if algo == 'ns':
        if verbose:
            print(f"\n{'─'*50}\nNatural Selection (Python, finalists={ns_finalists})...")
        r = solve_natural_selection(problem, n_finalists=ns_finalists,
                                    tabu_iters=tabu_iters, tabu_patience=tabu_patience,
                                    hho_pop=hho_pop, hho_iters=hho_iters,
                                    sa_iters=sa_iters, kempe_iters=kempe_iters,
                                    alns_iters=alns_iters, gd_iters=gd_iters,
                                    abc_pop=abc_pop, abc_iters=abc_iters,
                                    ga_pop=ga_pop, ga_iters=ga_iters,
                                    seed=seed, verbose=verbose)
        r['evaluation'] = _py_to_eval(r['evaluation'])
        results['Natural Selection'] = r

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
    tabu_iters: int = _GP.get('tabu_iters', 2000),
    tabu_tenure: int = _GP.get('tabu_tenure', 20),
    tabu_patience: int = _GP.get('tabu_patience', 500),
    hho_pop: int = _GP.get('hho_pop', 30),
    hho_iters: int = _GP.get('hho_iters', 200),
    sa_iters: int = _GP.get('sa_iters', 5000),
    kempe_iters: int = _GP.get('kempe_iters', 3000),
    alns_iters: int = _GP.get('alns_iters', 2000),
    gd_iters: int = _GP.get('gd_iters', 5000),
    abc_pop: int = _GP.get('abc_pop', 30),
    abc_iters: int = _GP.get('abc_iters', 3000),
    ga_pop: int = _GP.get('ga_pop', 50),
    ga_iters: int = _GP.get('ga_iters', 500),
    lahc_iters: int = _GP.get('lahc_iters', 5000),
    lahc_list: int = _GP.get('lahc_list', 0),
    ns_finalists: int = 3,
    seed: int = 42,
    output_dir: str = 'results',
    verbose: bool = True,
    init_solution_path: str = '',
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
            hho_iters=hho_iters, sa_iters=sa_iters,
            kempe_iters=kempe_iters, alns_iters=alns_iters,
            gd_iters=gd_iters, abc_pop=abc_pop,
            abc_iters=abc_iters, ga_pop=ga_pop,
            ga_iters=ga_iters, ns_finalists=ns_finalists,
            seed=seed, verbose=verbose)

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
        '--sa-iters', str(sa_iters),
        '--kempe-iters', str(kempe_iters),
        '--alns-iters', str(alns_iters),
        '--gd-iters', str(gd_iters),
        '--abc-pop', str(abc_pop),
        '--abc-iters', str(abc_iters),
        '--ga-pop', str(ga_pop),
        '--ga-iters', str(ga_iters),
        '--lahc-iters', str(lahc_iters),
        '--lahc-list', str(lahc_list),
        '--ns-finalists', str(ns_finalists),
        '--seed', str(seed),
        '--output-dir', output_dir,
    ]
    if init_solution_path:
        cmd.extend(['--init-solution', init_solution_path])
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
        sln_path = os.path.join(output_dir, "solutions", f"solution_{safe_name}_{ne}.sln")
        sol = _load_solution(problem, sln_path)

        algo_results[name] = {
            'solution': sol,
            'runtime': r['runtime'],
            'evaluation': ev,
            'algorithm': name,
            'iterations': r.get('iterations', 0),
        }

    return algo_results