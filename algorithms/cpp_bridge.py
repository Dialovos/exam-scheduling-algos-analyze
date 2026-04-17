"""
Calls the compiled C++ exam_solver binary and returns results in the same
format as the Python algorithms. Falls back to Python if the binary is missing.

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
    woa_pop: int = _GP.get('woa_pop', 25),
    woa_iters: int = _GP.get('woa_iters', 3000),
    hho_pop: int = _GP.get('hho_pop', 20),
    hho_iters: int = _GP.get('hho_iters', 500),
    cpsat_time: float = _GP.get('cpsat_time', 60.0),
    vns_iters: int = _GP.get('vns_iters', 5000),
    vns_budget: int = _GP.get('vns_budget', 0),
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
        algo: single name ("sa"), comma-separated ("sa,gd,tabu"), or "all"
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
        tabu_patience=tabu_patience, sa_iters=sa_iters,
        kempe_iters=kempe_iters, alns_iters=alns_iters,
        gd_iters=gd_iters, abc_pop=abc_pop,
        abc_iters=abc_iters, ga_pop=ga_pop,
        ga_iters=ga_iters, lahc_iters=lahc_iters,
        lahc_list=lahc_list, woa_pop=woa_pop,
        woa_iters=woa_iters, hho_pop=hho_pop, hho_iters=hho_iters,
        cpsat_time=cpsat_time,
        vns_iters=vns_iters, vns_budget=vns_budget,
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
                         sa_iters=_GP.get('sa_iters', 5000),
                         kempe_iters=_GP.get('kempe_iters', 3000),
                         alns_iters=_GP.get('alns_iters', 2000),
                         gd_iters=_GP.get('gd_iters', 5000),
                         abc_pop=_GP.get('abc_pop', 30),
                         abc_iters=_GP.get('abc_iters', 3000),
                         ga_pop=_GP.get('ga_pop', 50),
                         ga_iters=_GP.get('ga_iters', 500),
                         seed=42, verbose=True, **_kwargs):
    """Run Python implementations when C++ binary is unavailable.

    Note: LAHC, WOA, CP-SAT, and GVNS are C++ only.
    """
    from algorithms.greedy import solve_greedy
    from algorithms.tabu_search import solve_tabu
    from algorithms.kempe_chain import solve_kempe
    from algorithms.simulated_annealing import solve_sa
    from algorithms.alns import solve_alns
    from algorithms.great_deluge import solve_great_deluge
    from algorithms.abc import solve_abc
    from algorithms.ga import solve_ga

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

    cpp_only = {'lahc', 'woa', 'hho', 'cpsat', 'vns'}
    if algo in cpp_only or (algo == 'all' and verbose):
        if verbose:
            print(f"\n{'─'*50}\nNote: LAHC, WOA, HHO+, CP-SAT, GVNS are C++ only (skipped in fallback)")

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
    woa_pop: int = _GP.get('woa_pop', 25),
    woa_iters: int = _GP.get('woa_iters', 3000),
    hho_pop: int = _GP.get('hho_pop', 20),
    hho_iters: int = _GP.get('hho_iters', 500),
    cpsat_time: float = _GP.get('cpsat_time', 60.0),
    vns_iters: int = _GP.get('vns_iters', 5000),
    vns_budget: int = _GP.get('vns_budget', 0),
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
            tabu_patience=tabu_patience, sa_iters=sa_iters,
            kempe_iters=kempe_iters, alns_iters=alns_iters,
            gd_iters=gd_iters, abc_pop=abc_pop,
            abc_iters=abc_iters, ga_pop=ga_pop,
            ga_iters=ga_iters,
            seed=seed, verbose=verbose)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        binary, exam_filepath,
        '--algo', algo,
        '--limit', str(limit),
        '--tabu-iters', str(tabu_iters),
        '--tabu-tenure', str(tabu_tenure),
        '--tabu-patience', str(tabu_patience),
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
        '--woa-pop', str(woa_pop),
        '--woa-iters', str(woa_iters),
        '--hho-pop', str(hho_pop),
        '--hho-iters', str(hho_iters),
        '--cpsat-time', str(cpsat_time),
        '--vns-iters', str(vns_iters),
        '--vns-budget', str(vns_budget),
        '--seed', str(seed),
        '--output-dir', output_dir,
    ]
    if init_solution_path:
        cmd.extend(['--init-solution', init_solution_path])
    if verbose:
        cmd.append('--verbose')

    if verbose:
        print(f"[C++ Bridge] Running: {' '.join(cmd)}")

    # Use Popen so we can poll /proc/<pid>/status for peak RSS (VmHWM).
    import threading as _threading
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
    except OSError as e:
        print(f"[C++ Bridge] Failed to start solver: {e}")
        return None

    peak_kb = [0]
    stop_poll = _threading.Event()

    def _poll(pid=proc.pid):
        while not stop_poll.is_set():
            try:
                with open(f'/proc/{pid}/status') as f:
                    for line in f:
                        if line.startswith('VmHWM:'):
                            kb = int(line.split()[1])
                            if kb > peak_kb[0]:
                                peak_kb[0] = kb
                            break
            except (OSError, ValueError):
                return
            time.sleep(0.05)

    poller = _threading.Thread(target=_poll, daemon=True)
    poller.start()

    try:
        stdout_data, stderr_data = proc.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        stop_poll.set()
        print("[C++ Bridge] Solver timed out after 600s")
        return None
    finally:
        stop_poll.set()
        poller.join(timeout=0.2)

    memory_peak_mb = peak_kb[0] / 1024.0

    # Print stderr (verbose logs) to console
    if stderr_data and verbose:
        for line in stderr_data.strip().split('\n'):
            print(f"  {line}")

    if proc.returncode != 0:
        print(f"[C++ Bridge] Solver failed (code {proc.returncode})")
        if stderr_data:
            print(stderr_data)
        return None

    # Parse JSON from stdout
    try:
        raw_results = json.loads(stdout_data)
    except json.JSONDecodeError as e:
        print(f"[C++ Bridge] JSON parse error: {e}")
        print(f"[C++ Bridge] stdout was: {stdout_data[:500]}")
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
            'memory_peak_mb': round(memory_peak_mb, 2),
        }

    return algo_results


# ═══════════════════════════════════════════════════════════════════════════════
# Chain execution (warm-started multi-step runs with memory capture)
# ═══════════════════════════════════════════════════════════════════════════════

def run_chain(dataset, chain_steps, seed=42, work_dir=None, timeout_per_step=300):
    """Run a warm-started chain of algorithms with per-step memory tracking.

    Each step writes its solution file; the next step reads it via --init-solution.
    Memory is captured by polling /proc/<pid>/status for VmHWM (peak RSS) in
    a watchdog thread while each subprocess runs. Linux-only (WSL2 OK).

    Args:
        dataset: Path to an ITC 2007 .exam file.
        chain_steps: List of (algo_name, params_dict) tuples, e.g.
            [("sa", {"sa_iters": 5000}), ("gd", {"gd_iters": 5000})]
        seed: Random seed, passed to every step.
        work_dir: Scratch directory for intermediate .sln files. If None,
            a temp directory is created and removed automatically.
        timeout_per_step: Per-step subprocess timeout in seconds.

    Returns:
        dict with keys:
            soft_penalty     — int, from final step
            hard_violations  — int, from final step
            total_runtime    — float, sum of per-step runtimes (seconds)
            memory_peak_mb   — float, max RSS observed across steps (MB)
            per_step         — list of {algo, runtime, memory_mb}
        Returns None if any step fails (timeout, bad JSON, non-zero exit).
    """
    import glob as _glob
    import shutil as _shutil

    binary = os.path.join(os.path.dirname(__file__), '..', 'cpp', 'build', 'exam_solver')
    binary = os.path.abspath(binary)
    if not os.path.isfile(binary):
        print(f"[run_chain] C++ binary not found at {binary}")
        return None

    cleanup_work_dir = False
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='chain_')
        cleanup_work_dir = True
    else:
        os.makedirs(work_dir, exist_ok=True)

    try:
        import threading as _threading

        sln_path = None
        per_step = []
        final_result = None

        for i, (algo, params) in enumerate(chain_steps):
            step_dir = os.path.join(work_dir, f'step_{i}_{algo}')
            os.makedirs(step_dir, exist_ok=True)

            cmd = [binary, dataset, '--algo', algo, '--seed', str(seed),
                   '--output-dir', step_dir]
            for k, v in params.items():
                cmd.extend(['--' + k.replace('_', '-'), str(int(v))])
            if sln_path and os.path.isfile(sln_path):
                cmd.extend(['--init-solution', sln_path])

            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)
            except OSError:
                return None

            # Watchdog thread polls /proc/<pid>/status for VmHWM (peak RSS).
            # VmHWM is monotone-non-decreasing, so even a coarse poll captures
            # the true peak as long as it samples once before the process exits.
            peak_kb = [0]
            stop_poll = _threading.Event()

            def _poll(pid=proc.pid):
                while not stop_poll.is_set():
                    try:
                        with open(f'/proc/{pid}/status') as f:
                            for line in f:
                                if line.startswith('VmHWM:'):
                                    kb = int(line.split()[1])
                                    if kb > peak_kb[0]:
                                        peak_kb[0] = kb
                                    break
                    except (OSError, ValueError):
                        return
                    time.sleep(0.005)

            poller = _threading.Thread(target=_poll, daemon=True)
            poller.start()

            try:
                stdout_data, _stderr = proc.communicate(timeout=timeout_per_step)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                stop_poll.set()
                return None
            finally:
                stop_poll.set()
                poller.join(timeout=0.1)

            if proc.returncode != 0:
                return None

            try:
                data = json.loads(stdout_data)
                if not data:
                    return None
                step_result = data[0]
            except (json.JSONDecodeError, IndexError):
                return None

            step_mem_mb = peak_kb[0] / 1024.0

            per_step.append({
                'algo': algo,
                'runtime': float(step_result.get('runtime', 0.0)),
                'memory_mb': step_mem_mb,
            })
            final_result = step_result

            sln_files = _glob.glob(os.path.join(step_dir, 'solutions', '*.sln'))
            sln_path = sln_files[0] if sln_files else None

        if final_result is None:
            return None

        total_runtime = sum(s['runtime'] for s in per_step)
        memory_peak_mb = max((s['memory_mb'] for s in per_step), default=0.0)
        chain_label = 'Chain(' + '→'.join(s['algo'] for s in per_step) + ')'

        return {
            'soft_penalty': int(final_result.get('soft_penalty', 0)),
            'hard_violations': int(final_result.get('hard_violations', 0)),
            'total_runtime': total_runtime,
            'memory_peak_mb': memory_peak_mb,
            'per_step': per_step,
            # Keys below let logger.log_run consume this dict directly
            'algorithm': chain_label,
            'runtime': total_runtime,
            'evaluation': _parse_eval(final_result),
            'iterations': sum(int(s.get('iterations', 0) or 0) for s in per_step),
        }
    finally:
        if cleanup_work_dir:
            try:
                _shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass