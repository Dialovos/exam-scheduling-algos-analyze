"""
Evolutionary algorithm with tournament selection,
uniform crossover, mutation, and elitism.
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


def _tournament_select(fitness, pop_size, k, rng):
    best = rng.randint(0, pop_size - 1)
    for _ in range(k - 1):
        cand = rng.randint(0, pop_size - 1)
        if fitness[cand] < fitness[best]:
            best = cand
    return best


def _crossover(a, b, problem, n_e, n_r, valid_r, rng):
    child = Solution(problem)

    for e in range(n_e):
        if rng.random() < 0.5 and a._period_of[e] >= 0:
            child.assign(e, a._period_of[e], a._room_of[e])
        elif b._period_of[e] >= 0:
            child.assign(e, b._period_of[e], b._room_of[e])
        elif a._period_of[e] >= 0:
            child.assign(e, a._period_of[e], a._room_of[e])
        else:
            child.assign(e, 0, 0)

    # Room repair
    for e in range(n_e):
        pid = child._period_of[e]
        if pid < 0:
            continue
        rid = child._room_of[e]
        if child.get_pr_enroll(pid, rid) > problem.rooms[rid].capacity:
            for r in valid_r[e]:
                if r == rid:
                    continue
                if child.get_pr_enroll(pid, r) + problem.exams[e].enrollment <= problem.rooms[r].capacity:
                    child.assign(e, pid, r)
                    break

    return child


def solve_ga(
    problem: ProblemInstance,
    pop_size: int = 50,
    max_generations: int = 500,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.15,
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

    pop_size = max(pop_size, 10)

    exam_dur = [e.duration for e in problem.exams]
    exam_enr = [e.enrollment for e in problem.exams]
    period_dur = [p.duration for p in problem.periods]
    room_cap = [r.capacity for r in problem.rooms]

    valid_p = [[p for p in range(n_p) if exam_dur[e] <= period_dur[p]] for e in range(n_e)]
    valid_r = [[r for r in range(n_r) if exam_enr[e] <= room_cap[r]] for e in range(n_e)]

    # Initialize population
    greedy_result = solve_greedy(problem, verbose=False, seed=seed)
    population = [greedy_result['solution']]
    fitness = [fe.full_eval(population[0]).fitness]

    for i in range(1, pop_size):
        sol = _random_solution(problem, fe, n_e, n_p, n_r, valid_p, valid_r, rng)
        fitness.append(fe.full_eval(sol).fitness)
        population.append(sol)

    # Global best
    bi = fitness.index(min(fitness))
    best_sol = population[bi].copy()
    best_fitness = fitness[bi]

    if verbose:
        ev = fe.full_eval(best_sol)
        print(f"[GA] Init: pop={pop_size} best hard={ev.hard} soft={ev.soft}")

    elite_count = max(2, pop_size // 10)
    iters_done = 0

    for gen in range(max_generations):
        iters_done = gen + 1

        rank = sorted(range(pop_size), key=lambda i: fitness[i])

        new_pop = []
        new_fit = []

        # Elitism
        for i in range(elite_count):
            new_pop.append(population[rank[i]].copy())
            new_fit.append(fitness[rank[i]])

        # Offspring
        while len(new_pop) < pop_size:
            if rng.random() < crossover_rate:
                p1 = _tournament_select(fitness, pop_size, 3, rng)
                p2 = _tournament_select(fitness, pop_size, 3, rng)
                child = _crossover(population[p1], population[p2],
                                   problem, n_e, n_r, valid_r, rng)
            else:
                p = _tournament_select(fitness, pop_size, 3, rng)
                child = population[p].copy()

            # Mutation
            if rng.random() < mutation_rate:
                n_mut = max(1, int(n_e * 0.05))
                for _ in range(n_mut):
                    eid = rng.randint(0, n_e - 1)
                    vp, vr = valid_p[eid], valid_r[eid]
                    if vp and vr:
                        child.assign(eid, rng.choice(vp), rng.choice(vr))

            new_fit.append(fe.full_eval(child).fitness)
            new_pop.append(child)

        population = new_pop
        fitness = new_fit

        # Update global best
        gen_best = fitness.index(min(fitness))
        if fitness[gen_best] < best_fitness - 0.5:
            check = fe.full_eval(population[gen_best])
            if check.fitness < best_fitness:
                best_sol = population[gen_best].copy()
                best_fitness = check.fitness
                if verbose and (gen < 10 or gen % 100 == 0):
                    print(f"[GA] Gen {gen}: best hard={check.hard} soft={check.soft}")

    ev = fe.full_eval(best_sol)
    runtime = time.time() - t0
    if verbose:
        print(f"[GA] {iters_done} gens, {runtime:.2f}s  "
              f"feasible={ev.feasible} hard={ev.hard} soft={ev.soft}")

    return {
        'solution': best_sol,
        'runtime': runtime,
        'evaluation': ev,
        'algorithm': 'Genetic Algorithm',
        'iterations': iters_done,
    }
