"""
Swarm optimization with employed, onlooker, and scout bee phases.
Uses move_delta for O(k) neighbor evaluation.
"""

import time
import random as _random
from core.models import ProblemInstance, Solution
from core.fast_eval import FastEvaluator
from algorithms.greedy import solve_greedy


def _random_solution(problem, fe, n_e, n_p, n_r, valid_p, valid_r, rng):
    sol = Solution(problem)
    order = list(range(n_e))
    rng.shuffle(order)

    for eid in order:
        blocked = set()
        for nb, _ in problem.conflict_matrix.get(eid, []):
            p = sol._period_of[nb]
            if p >= 0:
                blocked.add(p)

        avail = [p for p in valid_p[eid] if p not in blocked]
        if not avail:
            avail = valid_p[eid] if valid_p[eid] else list(range(n_p))
        rng.shuffle(avail)

        placed = False
        for pid in avail:
            for rid in valid_r[eid]:
                if sol.get_pr_enroll(pid, rid) + problem.exams[eid].enrollment <= problem.rooms[rid].capacity:
                    sol.assign(eid, pid, rid)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            pid = avail[0] if avail else 0
            rid = rng.choice(valid_r[eid]) if valid_r[eid] else 0
            sol.assign(eid, pid, rid)

    return sol


def solve_abc(
    problem: ProblemInstance,
    colony_size: int = 30,
    max_iterations: int = 3000,
    abandon_limit: int = 0,
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

    colony_size = max(colony_size, 5)
    if abandon_limit <= 0:
        abandon_limit = colony_size * 3

    exam_dur = [e.duration for e in problem.exams]
    exam_enr = [e.enrollment for e in problem.exams]
    period_dur = [p.duration for p in problem.periods]
    room_cap = [r.capacity for r in problem.rooms]

    valid_p = [[p for p in range(n_p) if exam_dur[e] <= period_dur[p]] for e in range(n_e)]
    valid_r = [[r for r in range(n_r) if exam_enr[e] <= room_cap[r]] for e in range(n_e)]

    # Initialize food sources
    greedy_result = solve_greedy(problem, verbose=False, seed=seed)
    sources = [greedy_result['solution']]
    fitness = [fe.full_eval(sources[0]).fitness]
    trials = [0]

    for i in range(1, colony_size):
        sol = _random_solution(problem, fe, n_e, n_p, n_r, valid_p, valid_r, rng)
        fitness.append(fe.full_eval(sol).fitness)
        sources.append(sol)
        trials.append(0)

    # Global best
    bi = fitness.index(min(fitness))
    best_sol = sources[bi].copy()
    best_fitness = fitness[bi]

    if verbose:
        ev = fe.full_eval(best_sol)
        print(f"[ABC] Init: colony={colony_size} best hard={ev.hard} soft={ev.soft}")

    iters_done = 0

    for it in range(max_iterations):
        iters_done = it + 1

        # Employed bee phase
        for i in range(colony_size):
            eid = rng.randint(0, n_e - 1)
            vp, vr = valid_p[eid], valid_r[eid]
            if not vp or not vr:
                continue

            k = rng.randint(0, colony_size - 1)
            if k == i:
                k = (k + 1) % colony_size

            if rng.random() < 0.5 and sources[k]._period_of[eid] >= 0:
                new_pid = sources[k]._period_of[eid]
                new_rid = rng.choice(vr)
            else:
                new_pid = rng.choice(vp)
                new_rid = rng.choice(vr)

            delta = fe.move_delta(sources[i], eid, new_pid, new_rid)
            if delta < 0:
                fe.apply_move(sources[i], eid, new_pid, new_rid)
                fitness[i] += delta
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker bee phase
        max_fit = max(fitness)
        probs = [max_fit - f + 1.0 for f in fitness]
        sum_prob = sum(probs)

        for _ in range(colony_size):
            r = rng.random() * sum_prob
            sel, acc = 0, 0.0
            for i in range(colony_size):
                acc += probs[i]
                if r <= acc:
                    sel = i
                    break

            eid = rng.randint(0, n_e - 1)
            vp, vr = valid_p[eid], valid_r[eid]
            if not vp or not vr:
                continue

            k = rng.randint(0, colony_size - 1)
            if k == sel:
                k = (k + 1) % colony_size

            if rng.random() < 0.5 and sources[k]._period_of[eid] >= 0:
                new_pid = sources[k]._period_of[eid]
                new_rid = rng.choice(vr)
            else:
                new_pid = rng.choice(vp)
                new_rid = rng.choice(vr)

            delta = fe.move_delta(sources[sel], eid, new_pid, new_rid)
            if delta < 0:
                fe.apply_move(sources[sel], eid, new_pid, new_rid)
                fitness[sel] += delta
                trials[sel] = 0
            else:
                trials[sel] += 1

        # Scout bee phase
        for i in range(colony_size):
            if trials[i] > abandon_limit:
                sources[i] = _random_solution(problem, fe, n_e, n_p, n_r, valid_p, valid_r, rng)
                fitness[i] = fe.full_eval(sources[i]).fitness
                trials[i] = 0

        # Periodic resync
        if it % 50 == 0:
            for i in range(colony_size):
                fitness[i] = fe.full_eval(sources[i]).fitness

        # Update global best
        for i in range(colony_size):
            if fitness[i] < best_fitness - 0.5:
                check = fe.full_eval(sources[i])
                if check.fitness < best_fitness:
                    best_sol = sources[i].copy()
                    best_fitness = check.fitness
                    if verbose and (it < 10 or it % 500 == 0):
                        print(f"[ABC] Iter {it}: best hard={check.hard} soft={check.soft}")

    ev = fe.full_eval(best_sol)
    runtime = time.time() - t0
    if verbose:
        print(f"[ABC] {iters_done} iters, {runtime:.2f}s  "
              f"feasible={ev.feasible} hard={ev.hard} soft={ev.soft}")

    return {
        'solution': best_sol,
        'runtime': runtime,
        'evaluation': ev,
        'algorithm': 'ABC',
        'iterations': iters_done,
    }
