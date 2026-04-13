"""
Destroy-and-repair with adaptive operator selection.
3 destroy operators (random, worst, related) + 2 repair (greedy, random).
SA-like acceptance with adaptive operator weights.
"""

import time
import math
import random as _random
from core.models import ProblemInstance, Solution
from core.fast_eval import FastEvaluator
from algorithms.greedy import solve_greedy


def solve_alns(
    problem: ProblemInstance,
    max_iterations: int = 2000,
    destroy_pct: float = 0.15,
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

    n_destroy = max(1, int(n_e * destroy_pct))

    # SA acceptance
    temp = max(1.0, ev.soft * 0.02) if ev.soft > 0 else 50.0
    cooling = 0.9998

    # Operator weights (3 destroy, 2 repair)
    d_weights = [1.0, 1.0, 1.0]
    r_weights = [1.0, 1.0]
    d_scores = [0.0, 0.0, 0.0]
    r_scores = [0.0, 0.0]
    d_counts = [1, 1, 1]
    r_counts = [1, 1]

    if verbose:
        print(f"[ALNS] Init: feasible={ev.feasible} hard={ev.hard} soft={ev.soft}")

    iters_done = 0
    for it in range(max_iterations):
        iters_done = it + 1

        d_op = _roulette(d_weights, rng)
        r_op = _roulette(r_weights, rng)

        # Save positions of all exams
        saved = [(sol._period_of[e], sol._room_of[e]) for e in range(n_e)]

        # Destroy
        if d_op == 0:
            removed = _destroy_random(sol, n_e, n_destroy, rng)
        elif d_op == 1:
            removed = _destroy_worst(sol, fe, n_e, n_destroy, rng)
        else:
            removed = _destroy_related(sol, fe, n_e, n_destroy, rng)

        # Repair
        if r_op == 0:
            _repair_greedy(sol, fe, removed, valid_p, valid_r)
        else:
            _repair_random(sol, removed, valid_p, valid_r, rng)

        # Evaluate
        new_ev = fe.full_eval(sol)
        new_fitness = new_ev.fitness
        delta = new_fitness - current_fitness

        accept = delta < 0
        if not accept and temp > 1e-10:
            accept = rng.random() < math.exp(-delta / temp)

        score = 0.0
        if accept:
            current_fitness = new_fitness
            score = 1.0
            if new_fitness < best_fitness:
                best_sol = sol.copy()
                best_fitness = new_fitness
                score = 3.0
                if verbose and (it < 10 or it % 200 == 0):
                    print(f"[ALNS] Iter {it}: best hard={new_ev.hard} soft={new_ev.soft}")
        else:
            # Rollback only removed exams
            for e in removed:
                sol.assign(e, saved[e][0], saved[e][1])

        d_scores[d_op] += score
        r_scores[r_op] += score
        d_counts[d_op] += 1
        r_counts[r_op] += 1

        # Update weights every 100 iterations
        if (it + 1) % 100 == 0:
            for i in range(3):
                d_weights[i] = max(0.1, 0.7 * d_weights[i] + 0.3 * d_scores[i] / d_counts[i])
                d_scores[i] = 0.0
                d_counts[i] = 1
            for i in range(2):
                r_weights[i] = max(0.1, 0.7 * r_weights[i] + 0.3 * r_scores[i] / r_counts[i])
                r_scores[i] = 0.0
                r_counts[i] = 1

        temp *= cooling

    ev = fe.full_eval(best_sol)
    runtime = time.time() - t0
    if verbose:
        print(f"[ALNS] {iters_done} iters, {runtime:.2f}s  "
              f"feasible={ev.feasible} hard={ev.hard} soft={ev.soft}")

    return {
        'solution': best_sol,
        'runtime': runtime,
        'evaluation': ev,
        'algorithm': 'ALNS',
        'iterations': iters_done,
    }


def _roulette(weights, rng):
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


def _destroy_random(sol, n_e, n_destroy, rng):
    exams = list(range(n_e))
    rng.shuffle(exams)
    removed = []
    for e in exams:
        if len(removed) >= n_destroy:
            break
        if sol._period_of[e] >= 0:
            sol.unassign(e)
            removed.append(e)
    return removed


def _destroy_worst(sol, fe, n_e, n_destroy, rng):
    costs = []
    for e in range(n_e):
        pid = sol._period_of[e]
        if pid < 0:
            continue
        cost = 0
        for nb, _ in fe.adj[e]:
            nb_pid = sol._period_of[nb]
            if nb_pid == pid:
                cost += 100000
            elif nb_pid >= 0:
                gap = abs(pid - nb_pid)
                if 0 < gap <= fe.w_spread:
                    cost += 1
        costs.append((cost, e))

    costs.sort(reverse=True)
    pool = costs[:n_destroy * 2]
    rng.shuffle(pool)
    removed = []
    for _, e in pool:
        if len(removed) >= n_destroy:
            break
        if sol._period_of[e] >= 0:
            sol.unassign(e)
            removed.append(e)
    return removed


def _destroy_related(sol, fe, n_e, n_destroy, rng):
    start = rng.randint(0, n_e - 1)
    removed = set()
    queue = [start]
    while len(removed) < n_destroy and queue:
        e = queue.pop(0)
        if e in removed or sol._period_of[e] < 0:
            continue
        sol.unassign(e)
        removed.add(e)
        neighbors = [nb for nb, _ in fe.adj[e]]
        rng.shuffle(neighbors)
        queue.extend(neighbors)
    return list(removed)


def _repair_greedy(sol, fe, removed, valid_p, valid_r):
    # Most constrained first
    removed.sort(key=lambda e: len(fe.adj[e]), reverse=True)
    for eid in removed:
        vp = valid_p[eid]
        vr = valid_r[eid]
        if not vp:
            vp = list(range(fe.n_p))
        if not vr:
            vr = list(range(fe.n_r))

        best_pid, best_rid, best_cost = -1, -1, float('inf')
        for pid in vp[:15]:
            for rid in vr[:5]:
                cost = 0
                for nb, _ in fe.adj[eid]:
                    if sol._period_of[nb] == pid:
                        cost += 100000
                cost += fe.period_pen[pid] + fe.room_pen[rid]
                if cost < best_cost:
                    best_cost = cost
                    best_pid, best_rid = pid, rid

        if best_pid >= 0:
            sol.assign(eid, best_pid, best_rid)
        else:
            sol.assign(eid, vp[0], vr[0])


def _repair_random(sol, removed, valid_p, valid_r, rng):
    for eid in removed:
        vp = valid_p[eid]
        vr = valid_r[eid]
        if vp and vr:
            sol.assign(eid, rng.choice(vp), rng.choice(vr))
        else:
            sol.assign(eid, 0, 0)
