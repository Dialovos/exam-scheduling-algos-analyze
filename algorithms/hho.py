"""
Optimizations:
  1. FastEvaluator.move_delta() for O(k) move scoring instead of O(n*s) full eval
  2. Perturbations are sequences of single-exam moves with tracked delta
  3. Only 1 full_eval per hawk initialization; rest are deltas
  4. Smarter exploration: conflict-aware random moves, not blind random periods
"""

import random
import math
import time
import tracemalloc
from data.models import ProblemInstance, Solution
from data.fast_eval import FastEvaluator
from algorithms.greedy import solve_greedy


def _init_random_solution(problem: ProblemInstance, fe: FastEvaluator,
                          rng: random.Random) -> tuple[Solution, float]:
    """Build a randomized-greedy solution and return (solution, fitness)."""
    sol = Solution(problem)
    adj: list[list[tuple[int, int]]] = fe.adj
    period_ids = list(range(fe.n_p))
    room_ids = list(range(fe.n_r))

    pr_usage: dict[tuple[int, int], int] = {}

    order = list(range(fe.n_e))
    rng.shuffle(order)

    for eid in order:
        blocked = set()
        for nb, _ in adj[eid]:
            p = sol.get_period(nb)
            if p is not None:
                blocked.add(p)

        avail = [p for p in period_ids if p not in blocked
                 and fe.period_dur[p] >= fe.exam_dur[eid]]
        if not avail:
            avail = period_ids

        rng.shuffle(avail)
        placed = False
        for pid in avail:
            rooms = list(room_ids)
            rng.shuffle(rooms)
            for rid in rooms:
                usage = pr_usage.get((pid, rid), 0)
                if usage + fe.exam_enroll[eid] <= fe.room_cap[rid]:
                    sol.assign(eid, pid, rid)
                    pr_usage[(pid, rid)] = usage + fe.exam_enroll[eid]
                    placed = True
                    break
            if placed:
                break

        if not placed:
            sol.assign(eid, rng.choice(period_ids), rng.choice(room_ids))

    ev = fe.full_eval(sol)
    return sol, ev.fitness


def _perturb_with_delta(
    problem: ProblemInstance,
    fe: FastEvaluator,
    sol: Solution,
    current_fitness: float,
    intensity: float,
    rng: random.Random,
) -> tuple[Solution, float]:
    """
    Perturb a solution using incremental delta evaluation.
    Returns new solution and its fitness.
    """
    new_sol = sol.copy()
    fitness = current_fitness

    n_moves = max(1, int(fe.n_e * intensity))
    exams = rng.sample(range(fe.n_e), min(n_moves, fe.n_e))

    period_ids = list(range(fe.n_p))
    room_ids = list(range(fe.n_r))

    for eid in exams:
        dur = fe.exam_dur[eid]
        valid = [p for p in period_ids if fe.period_dur[p] >= dur]
        if not valid:
            continue

        new_pid = rng.choice(valid)
        old_rid = new_sol.get_room(eid)
        if old_rid is None:
            old_rid = rng.choice(room_ids)

        delta = fe.move_delta(new_sol, eid, new_pid, old_rid)
        # Always apply in exploration; only if improving in exploitation
        fe.apply_move(new_sol, eid, new_pid, old_rid)
        fitness += delta

    return new_sol, fitness


def _smart_perturb(
    problem: ProblemInstance,
    fe: FastEvaluator,
    sol: Solution,
    current_fitness: float,
    n_moves: int,
    rng: random.Random,
) -> tuple[Solution, float]:
    """
    Smart perturbation: try several random moves, keep only improving ones.
    This is like a mini local-search within the hawk update step.
    """
    new_sol = sol.copy()
    fitness = current_fitness

    period_ids = list(range(fe.n_p))
    room_ids = list(range(fe.n_r))
    exams = rng.sample(range(fe.n_e), min(n_moves * 3, fe.n_e))

    moves_applied = 0
    for eid in exams:
        if moves_applied >= n_moves:
            break

        dur = fe.exam_dur[eid]
        valid = [p for p in period_ids if fe.period_dur[p] >= dur]
        if not valid:
            continue

        new_pid = rng.choice(valid)
        old_rid = new_sol.get_room(eid)
        if old_rid is None:
            old_rid = 0

        delta = fe.move_delta(new_sol, eid, new_pid, old_rid)

        if delta < 0:  # only apply improvements
            fe.apply_move(new_sol, eid, new_pid, old_rid)
            fitness += delta
            moves_applied += 1

    return new_sol, fitness


def solve_hho(
    problem: ProblemInstance,
    population_size: int = 30,
    max_iterations: int = 100,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    rng = random.Random(seed)
    start_time = time.time()
    tracemalloc.start()

    if problem.conflict_matrix is None:
        problem.build_derived_data()

    fe = FastEvaluator(problem)

    # --- Initialize population ---
    greedy_result = solve_greedy(problem, verbose=False)
    population: list[Solution] = [greedy_result['solution'].copy()]
    greedy_eval = fe.full_eval(population[0])
    fitness: list[float] = [greedy_eval.fitness]

    for _ in range(population_size - 1):
        sol, fit = _init_random_solution(problem, fe, rng)
        population.append(sol)
        fitness.append(fit)

    # Find rabbit
    best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
    rabbit = population[best_idx].copy()
    rabbit_fitness = fitness[best_idx]

    if verbose:
        print(f"[HHO] Pop={population_size}, Iters={max_iterations}")
        print(f"[HHO] Initial best: {rabbit_fitness:.0f}")

    # Precompute Levy sigma
    beta = 1.5
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)

    iterations_done = 0
    for t in range(max_iterations):
        iterations_done = t + 1

        E0 = 2 * rng.random() - 1
        E = 2 * E0 * (1 - t / max_iterations)

        for i in range(population_size):
            if abs(E) >= 1:
                # === EXPLORATION ===
                if rng.random() < 0.5:
                    rand_idx = rng.randint(0, population_size - 1)
                    base = population[rand_idx]
                    base_fit = fitness[rand_idx]
                else:
                    base = rabbit
                    base_fit = rabbit_fitness

                new_sol, new_fit = _perturb_with_delta(
                    problem, fe, base, base_fit, 0.3, rng)
            else:
                # === EXPLOITATION ===
                r = rng.random()
                if abs(E) >= 0.5:
                    if r >= 0.5:
                        # Soft besiege: small perturbation
                        n_moves = max(1, int(fe.n_e * abs(E) * 0.15))
                        new_sol, new_fit = _smart_perturb(
                            problem, fe, rabbit, rabbit_fitness, n_moves, rng)
                    else:
                        # Soft besiege + Levy
                        new_sol, new_fit = _perturb_with_delta(
                            problem, fe, rabbit, rabbit_fitness, abs(E) * 0.12, rng)
                        # Additional improving search
                        levy = abs(rng.gauss(0, sigma_u) / (abs(rng.gauss(0, 1)) ** (1/beta)))
                        extra = max(1, int(levy * 2))
                        new_sol, new_fit = _smart_perturb(
                            problem, fe, new_sol, new_fit, extra, rng)
                else:
                    if r >= 0.5:
                        # Hard besiege: very focused search
                        n_moves = max(1, int(fe.n_e * abs(E) * 0.08))
                        new_sol, new_fit = _smart_perturb(
                            problem, fe, rabbit, rabbit_fitness, n_moves, rng)
                    else:
                        # Hard besiege + Levy
                        levy = abs(rng.gauss(0, sigma_u) / (abs(rng.gauss(0, 1)) ** (1/beta)))
                        intensity = min(0.15, levy * 0.02)
                        new_sol, new_fit = _perturb_with_delta(
                            problem, fe, rabbit, rabbit_fitness, intensity, rng)
                        new_sol, new_fit = _smart_perturb(
                            problem, fe, new_sol, new_fit,
                            max(1, int(fe.n_e * 0.05)), rng)

            # Update hawk
            if new_fit < fitness[i]:
                population[i] = new_sol
                fitness[i] = new_fit

        # Update rabbit
        cur_best_idx = min(range(len(fitness)), key=lambda i: fitness[i])

        # Periodic full re-sync every 10 iterations to correct drift
        if t % 10 == 0:
            for i in range(population_size):
                real_eval = fe.full_eval(population[i])
                fitness[i] = real_eval.fitness
            cur_best_idx = min(range(len(fitness)), key=lambda i: fitness[i])

        if fitness[cur_best_idx] < rabbit_fitness:
            rabbit = population[cur_best_idx].copy()
            rabbit_fitness = fitness[cur_best_idx]
            if verbose and t % 10 == 0:
                print(f"[HHO] Iter {t}: NEW BEST {rabbit_fitness:.0f}")
        elif verbose and t % 25 == 0:
            print(f"[HHO] Iter {t}: best={rabbit_fitness:.0f}")

    runtime = time.time() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    final_eval = fe.full_eval(rabbit)

    if verbose:
        print(f"[HHO] {iterations_done} iters in {runtime:.2f}s")
        print(f"[HHO] {final_eval.summary()}")

    return {
        'solution': rabbit,
        'runtime': runtime,
        'evaluation': final_eval,
        'algorithm': 'Harris Hawks Optimization',
        'iterations': iterations_done,
        'memory_peak_mb': peak_mem / (1024 * 1024),
    }
