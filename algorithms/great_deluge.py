"""
Level-based acceptance: accepts any move where new fitness <= level.
Level decreases linearly from initial fitness.
Raises level when stuck to escape local optima.
"""

import time
import random as _random
from core.models import ProblemInstance, Solution
from core.fast_eval import FastEvaluator
from algorithms.greedy import solve_greedy


def solve_great_deluge(
    problem: ProblemInstance,
    max_iterations: int = 5000,
    decay_rate: float = 0.0,
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
    n_r = problem.num_rooms()
    fe = FastEvaluator(problem)

    exam_dur = [e.duration for e in problem.exams]
    exam_enr = [e.enrollment for e in problem.exams]
    period_dur = [p.duration for p in problem.periods]
    room_cap = [r.capacity for r in problem.rooms]

    valid_p = [[p for p in range(n_p) if exam_dur[e] <= period_dur[p]] for e in range(n_e)]
    valid_r = [[r for r in range(n_r) if exam_enr[e] <= room_cap[r]] for e in range(n_e)]

    greedy_result = solve_greedy(problem, verbose=False, seed=seed)
    sol = greedy_result['solution']

    ev = fe.full_eval(sol)
    current_fitness = ev.fitness
    best_sol = sol.copy()
    best_fitness = current_fitness

    level = current_fitness
    if decay_rate <= 0:
        decay_rate = current_fitness / max(max_iterations, 1)

    if verbose:
        print(f"[GD] Init: feasible={ev.feasible} hard={ev.hard} soft={ev.soft} level={level:.0f}")

    no_improve = 0
    iters_done = 0

    for it in range(max_iterations):
        iters_done = it + 1

        if it % 50 == 0:
            ev = fe.full_eval(sol)
            current_fitness = ev.fitness

        eid = rng.randint(0, n_e - 1)
        vp = valid_p[eid]
        vr = valid_r[eid]
        if not vp or not vr:
            continue

        new_pid = rng.choice(vp)
        new_rid = rng.choice(vr)
        if new_pid == sol._period_of[eid] and new_rid == sol._room_of[eid]:
            continue

        delta = fe.move_delta(sol, eid, new_pid, new_rid)
        new_fitness = current_fitness + delta

        if new_fitness <= level:
            fe.apply_move(sol, eid, new_pid, new_rid)
            current_fitness = new_fitness

            if current_fitness < best_fitness - 0.5:
                check = fe.full_eval(sol)
                actual = check.fitness
                if actual < best_fitness:
                    best_sol = sol.copy()
                    best_fitness = actual
                    no_improve = 0
                    if verbose and (it < 10 or it % 1000 == 0):
                        print(f"[GD] Iter {it}: best hard={check.hard} soft={check.soft} lvl={level:.0f}")
                current_fitness = actual
            else:
                no_improve += 1
        else:
            no_improve += 1

        level -= decay_rate

        # Raise level if stuck
        if no_improve > 0 and no_improve % 1000 == 0:
            level = current_fitness * 1.02

    ev = fe.full_eval(best_sol)
    runtime = time.time() - t0
    if verbose:
        print(f"[GD] {iters_done} iters, {runtime:.2f}s  "
              f"feasible={ev.feasible} hard={ev.hard} soft={ev.soft}")

    return {
        'solution': best_sol,
        'runtime': runtime,
        'evaluation': ev,
        'algorithm': 'Great Deluge',
        'iterations': iters_done,
    }
