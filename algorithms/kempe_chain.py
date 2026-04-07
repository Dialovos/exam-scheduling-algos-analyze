"""
Kempe Chain Local Search
========================
Swaps exams between two periods along conflict chains,
preserving period-conflict feasibility by construction.
Uses full_eval for chain assessment (chains can be large).
"""

import time
import random as _random
from collections import deque
from core.models import ProblemInstance, Solution
from core.fast_eval import FastEvaluator
from algorithms.greedy import solve_greedy


def solve_kempe(
    problem: ProblemInstance,
    max_iterations: int = 3000,
    seed: int = 42,
    verbose: bool = False,
    **kwargs,
) -> dict:
    t0 = time.time()
    rng = _random.Random(seed)

    if problem.conflict_matrix is None:
        problem.build_derived_data()

    n_e = problem.num_exams()
    n_p = problem.num_periods()

    adj = [[] for _ in range(n_e)]
    for (e1, e2) in problem.conflict_matrix:
        adj[e1].append(e2)
        adj[e2].append(e1)

    greedy_result = solve_greedy(problem, verbose=False, seed=seed)
    sol = greedy_result['solution']
    fe = FastEvaluator(problem)

    ev = fe.full_eval(sol)
    current_fitness = ev.fitness
    best_sol = sol.copy()
    best_fitness = current_fitness

    if verbose:
        print(f"[Kempe] Init: feasible={ev.feasible} hard={ev.hard} soft={ev.soft}")

    no_improve = 0
    iters_done = 0

    for it in range(max_iterations):
        iters_done = it + 1

        eid = rng.randint(0, n_e - 1)
        p1 = sol._period_of[eid]
        if p1 < 0:
            continue
        p2 = rng.randint(0, n_p - 1)
        if p2 == p1:
            continue

        # Build Kempe chain via BFS
        chain = set()
        queue = deque([eid])
        chain.add(eid)
        while queue:
            e = queue.popleft()
            ep = sol._period_of[e]
            target = p2 if ep == p1 else p1
            for nb in adj[e]:
                if nb not in chain and sol._period_of[nb] == target:
                    chain.add(nb)
                    queue.append(nb)

        if len(chain) < 1:
            continue

        # Save old assignments
        old_assignments = [(e, sol._period_of[e], sol._room_of[e]) for e in chain]

        # Swap: p1 <-> p2, keep rooms
        for e in chain:
            ep = sol._period_of[e]
            er = sol._room_of[e]
            sol.assign(e, p2 if ep == p1 else p1, er)

        new_fitness = fe.full_eval(sol).fitness

        if new_fitness < current_fitness:
            current_fitness = new_fitness
            if new_fitness < best_fitness:
                best_sol = sol.copy()
                best_fitness = new_fitness
                no_improve = 0
                if verbose and (it < 10 or it % 500 == 0):
                    ev2 = fe.full_eval(sol)
                    print(f"[Kempe] Iter {it}: best hard={ev2.hard} soft={ev2.soft}")
            else:
                no_improve += 1
        else:
            # Rollback
            for e, p, r in old_assignments:
                sol.assign(e, p, r)
            no_improve += 1

        if no_improve > max_iterations // 2:
            break

    ev = fe.full_eval(best_sol)
    runtime = time.time() - t0
    if verbose:
        print(f"[Kempe] {iters_done} iters, {runtime:.2f}s  "
              f"feasible={ev.feasible} hard={ev.hard} soft={ev.soft}")

    return {
        'solution': best_sol,
        'runtime': runtime,
        'evaluation': ev,
        'algorithm': 'Kempe Chain',
        'iterations': iters_done,
    }
