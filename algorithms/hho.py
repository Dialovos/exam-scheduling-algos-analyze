"""
Harris Hawks Optimization (Python fallback)
============================================
Population-based metaheuristic.  Uses the improved greedy for initialization
and FastEvaluator.move_delta for O(k) perturbation scoring.
"""

import time, math, random as _random
from core.models import ProblemInstance, Solution
from core.fast_eval import FastEvaluator
from algorithms.greedy import solve_greedy


def solve_hho(
    problem: ProblemInstance,
    population_size: int = 30,
    max_iterations: int = 200,
    seed: int = 42,
    verbose: bool = False,
    **kw,
) -> dict:
    t0 = time.time()
    rng = _random.Random(seed)

    if problem.conflict_matrix is None:
        problem.build_derived_data()

    n_e = problem.num_exams()
    n_p = problem.num_periods()
    n_r = problem.num_rooms()
    fe = FastEvaluator(problem)

    exam_dur = [e.duration for e in problem.exams]
    period_dur = [p.duration for p in problem.periods]
    valid_p = [[p for p in range(n_p) if exam_dur[e] <= period_dur[p]] for e in range(n_e)]
    valid_r = [[r for r in range(n_r) if problem.exams[e].enrollment <= problem.rooms[r].capacity]
               for e in range(n_e)]

    # ── Initialize population ──
    population = []
    fitness = []

    # Slot 0: deterministic greedy
    g = solve_greedy(problem, verbose=False, seed=42)
    population.append(g['solution'])
    fitness.append(fe.full_eval(g['solution']).fitness)

    # Remaining: randomized greedy
    for i in range(1, population_size):
        g = solve_greedy(problem, verbose=False, seed=seed + i * 100)
        population.append(g['solution'])
        fitness.append(fe.full_eval(g['solution']).fitness)

    # Best = rabbit
    best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
    rabbit = population[best_idx].copy()
    rabbit_fitness = fitness[best_idx]

    if verbose:
        ev = fe.full_eval(rabbit)
        print(f"[HHO] Pop={population_size} Iters={max_iterations} "
              f"Init best: feasible={ev.feasible} hard={ev.hard} soft={ev.soft}")

    # ── HHO loop ──
    for t in range(max_iterations):
        E0 = 2.0 * rng.random() - 1.0
        E = 2.0 * E0 * (1.0 - t / max_iterations)

        for i in range(population_size):
            if abs(E) >= 1.0:
                # Exploration: perturb a random individual
                src = population[rng.randint(0, population_size - 1)]
                intensity = 0.3
            else:
                # Exploitation: perturb the rabbit
                src = rabbit
                intensity = max(0.02, abs(E) * 0.15)

            ns = src.copy()
            nf = fitness[i] if src is population[i] else fe.full_eval(src).fitness
            n_moves = max(1, int(n_e * intensity))
            exams = list(range(n_e))
            rng.shuffle(exams)

            for eid in exams[:n_moves]:
                vp = valid_p[eid]
                if not vp:
                    continue
                pid = rng.choice(vp)
                vr_e = valid_r[eid]
                rid = rng.choice(vr_e) if vr_e else 0
                delta = fe.move_delta(ns, eid, pid, rid)
                # Exploitation: only accept improving
                if abs(E) < 1.0 and delta >= 0:
                    continue
                fe.apply_move(ns, eid, pid, rid)
                nf += delta

            actual = fe.full_eval(ns).fitness
            if actual < fitness[i]:
                population[i] = ns
                fitness[i] = actual

        # Update rabbit
        cur_best = min(range(population_size), key=lambda i: fitness[i])
        if fitness[cur_best] < rabbit_fitness:
            rabbit = population[cur_best].copy()
            rabbit_fitness = fitness[cur_best]
            if verbose and t % 25 == 0:
                ev = fe.full_eval(rabbit)
                print(f"[HHO] Iter {t}: best hard={ev.hard} soft={ev.soft}")
        elif verbose and t % 50 == 0:
            ev = fe.full_eval(rabbit)
            print(f"[HHO] Iter {t}: best hard={ev.hard} soft={ev.soft}")

    ev = fe.full_eval(rabbit)
    runtime = time.time() - t0
    if verbose:
        print(f"[HHO] {max_iterations} iters, {runtime:.2f}s  "
              f"feasible={ev.feasible} hard={ev.hard} soft={ev.soft}")

    return {
        'solution': rabbit,
        'runtime': runtime,
        'evaluation': ev,
        'algorithm': 'HHO',
        'iterations': max_iterations,
    }