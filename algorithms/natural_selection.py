"""
Natural Selection — Meta-Algorithm
===================================
Competitive algorithm selection (no ML):
  Phase 1 — Trial:  run each metaheuristic at 20% budget, rank by fitness
  Phase 2 — Finals: top-N finalists run at full budget
  Return:  best solution across all runs (feasibility-first)
"""

import time
from core.models import ProblemInstance
from core.fast_eval import FastEvaluator


def solve_natural_selection(
    problem: ProblemInstance,
    n_finalists: int = 3,
    tabu_iters: int = 2000,
    tabu_patience: int = 500,
    hho_pop: int = 30,
    hho_iters: int = 200,
    sa_iters: int = 5000,
    kempe_iters: int = 3000,
    alns_iters: int = 2000,
    gd_iters: int = 5000,
    abc_pop: int = 30,
    abc_iters: int = 3000,
    ga_pop: int = 50,
    ga_iters: int = 500,
    seed: int = 42,
    verbose: bool = False,
    **kwargs,
) -> dict:
    from algorithms.tabu_search import solve_tabu
    from algorithms.hho import solve_hho
    from algorithms.kempe_chain import solve_kempe
    from algorithms.simulated_annealing import solve_sa
    from algorithms.alns import solve_alns
    from algorithms.great_deluge import solve_great_deluge
    from algorithms.abc import solve_abc
    from algorithms.ga import solve_ga

    t0 = time.time()
    trial_frac = 0.2

    if problem.conflict_matrix is None:
        problem.build_derived_data()

    candidates = []  # (name, id, trial_fitness)
    all_results = []

    if verbose:
        print(f"[NS] Trial phase: testing 8 algorithms at {int(trial_frac*100)}% budget...")

    # ── Trial phase ──

    trials = [
        ("Tabu Search", 0, lambda it: solve_tabu(
            problem, max_iterations=it, patience=tabu_patience, seed=seed, verbose=False)),
        ("HHO", 1, lambda it: solve_hho(
            problem, population_size=hho_pop, max_iterations=it, seed=seed, verbose=False)),
        ("Kempe Chain", 2, lambda it: solve_kempe(
            problem, max_iterations=it, seed=seed, verbose=False)),
        ("Simulated Annealing", 3, lambda it: solve_sa(
            problem, max_iterations=it, seed=seed, verbose=False)),
        ("ALNS", 4, lambda it: solve_alns(
            problem, max_iterations=it, seed=seed, verbose=False)),
        ("Great Deluge", 5, lambda it: solve_great_deluge(
            problem, max_iterations=it, seed=seed, verbose=False)),
        ("ABC", 6, lambda it: solve_abc(
            problem, colony_size=abc_pop, max_iterations=it, seed=seed, verbose=False)),
        ("Genetic Algorithm", 7, lambda it: solve_ga(
            problem, pop_size=ga_pop, max_generations=it, seed=seed, verbose=False)),
    ]

    trial_iters_map = {
        0: max(100, int(tabu_iters * trial_frac)),
        1: max(20, int(hho_iters * trial_frac)),
        2: max(100, int(kempe_iters * trial_frac)),
        3: max(200, int(sa_iters * trial_frac)),
        4: max(100, int(alns_iters * trial_frac)),
        5: max(200, int(gd_iters * trial_frac)),
        6: max(100, int(abc_iters * trial_frac)),
        7: max(50, int(ga_iters * trial_frac)),
    }

    for name, tid, fn in trials:
        r = fn(trial_iters_map[tid])
        ev = r['evaluation']
        fit = ev.fitness
        if verbose:
            print(f"  {name:<20s} fitness={fit:.0f} (hard={ev.hard} soft={ev.soft}) {r['runtime']:.2f}s")
        candidates.append((name, tid, fit))
        all_results.append(r)

    # ── Select finalists ──

    candidates.sort(key=lambda x: x[2])
    nf = max(1, min(n_finalists, len(candidates)))

    if verbose:
        names = [f"{c[0]}({c[2]:.0f})" for c in candidates[:nf]]
        print(f"[NS] Finalists ({nf}): {' '.join(names)}")

    # ── Finals phase ──

    finals_map = {
        0: lambda: solve_tabu(problem, max_iterations=tabu_iters, patience=tabu_patience,
                              seed=seed, verbose=verbose),
        1: lambda: solve_hho(problem, population_size=hho_pop, max_iterations=hho_iters,
                             seed=seed, verbose=verbose),
        2: lambda: solve_kempe(problem, max_iterations=kempe_iters, seed=seed, verbose=verbose),
        3: lambda: solve_sa(problem, max_iterations=sa_iters, seed=seed, verbose=verbose),
        4: lambda: solve_alns(problem, max_iterations=alns_iters, seed=seed, verbose=verbose),
        5: lambda: solve_great_deluge(problem, max_iterations=gd_iters, seed=seed, verbose=verbose),
        6: lambda: solve_abc(problem, colony_size=abc_pop, max_iterations=abc_iters,
                             seed=seed, verbose=verbose),
        7: lambda: solve_ga(problem, pop_size=ga_pop, max_generations=ga_iters,
                            seed=seed, verbose=verbose),
    }

    for name, tid, _ in candidates[:nf]:
        if verbose:
            print(f"[NS] Finals: running {name} at full budget...")
        r = finals_map[tid]()
        all_results.append(r)

    # ── Pick overall best (feasibility-first) ──

    best = None
    for r in all_results:
        ev = r['evaluation']
        if best is None:
            best = r
            continue
        best_ev = best['evaluation']
        bf, cf = best_ev.feasible, ev.feasible
        if (cf and not bf) or (cf == bf and ev.fitness < best_ev.fitness):
            best = r

    winner = best['algorithm']
    total_rt = time.time() - t0

    if verbose:
        ev = best['evaluation']
        print(f"[NS] Winner: {winner}  feasible={ev.feasible} "
              f"hard={ev.hard} soft={ev.soft}  total={total_rt:.1f}s")

    return {
        'solution': best['solution'],
        'runtime': total_rt,
        'evaluation': best['evaluation'],
        'algorithm': 'Natural Selection',
        'iterations': best['iterations'],
    }
