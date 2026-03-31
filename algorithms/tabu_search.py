"""
Optimization:
  - Uses FastEvaluator.move_delta() → O(k) per candidate move instead of O(n*s) full eval
  - Explores more candidates per iteration since each is cheap
  - Full eval only at start and end for accurate scoring

Time Complexity: ~O(iterations * sample_exams * sample_periods * k)
"""

import random
import time
import tracemalloc
from data.models import ProblemInstance, Solution
from data.fast_eval import FastEvaluator
from algorithms.greedy import solve_greedy


def solve_tabu(
    problem: ProblemInstance,
    max_iterations: int = 200,
    tabu_tenure: int = 15,
    patience: int = 50,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    rng = random.Random(seed)
    start_time = time.time()
    tracemalloc.start()

    if problem.conflict_matrix is None:
        problem.build_derived_data()

    fe = FastEvaluator(problem)

    # --- Initial solution from Greedy ---
    greedy_result = solve_greedy(problem, verbose=False)
    current = greedy_result['solution'].copy()
    current_eval = fe.full_eval(current)
    current_fitness = current_eval.fitness

    best = current.copy()
    best_fitness = current_fitness

    if verbose:
        print(f"[Tabu] Initial: hard={current_eval.hard}, soft={current_eval.soft}, "
              f"fitness={current_fitness:.0f}")

    # Tabu list: (exam_id, old_period) -> expiry iteration
    tabu_list: dict[tuple[int, int], int] = {}
    no_improve = 0

    exam_ids = list(range(problem.num_exams()))
    period_ids = list(range(problem.num_periods()))

    iterations_done = 0

    for it in range(max_iterations):
        iterations_done = it + 1
        best_delta = float('inf')
        best_move = None

        # Sample exams — more for small instances, fewer for large
        n_sample_e = min(len(exam_ids), max(10, len(exam_ids) // 3))
        sample_e = rng.sample(exam_ids, n_sample_e)

        for eid in sample_e:
            old_pid = current.get_period(eid)
            old_rid = current.get_room(eid)
            if old_pid is None:
                continue

            dur = fe.exam_dur[eid]

            # Sample candidate periods
            cand_p = [p for p in period_ids if p != old_pid and fe.period_dur[p] >= dur]
            if len(cand_p) > 12:
                cand_p = rng.sample(cand_p, 12)

            for new_pid in cand_p:
                # O(k) delta evaluation!
                delta = fe.move_delta(current, eid, new_pid, old_rid)

                is_tabu = tabu_list.get((eid, new_pid), 0) > it
                aspiration = (current_fitness + delta) < best_fitness

                if (not is_tabu or aspiration) and delta < best_delta:
                    best_delta = delta
                    best_move = (eid, new_pid, old_rid, old_pid)

        # Apply best move
        if best_move is not None:
            eid, new_pid, new_rid, old_pid = best_move
            fe.apply_move(current, eid, new_pid, new_rid)
            current_fitness += best_delta
            tabu_list[(eid, old_pid)] = it + tabu_tenure

            # Periodic full re-sync to correct any accumulated drift
            if it % 15 == 0:
                real_eval = fe.full_eval(current)
                current_fitness = real_eval.fitness

            if current_fitness < best_fitness:
                best = current.copy()
                best_fitness = current_fitness
                no_improve = 0
                if verbose and it % 20 == 0:
                    print(f"[Tabu] Iter {it}: NEW BEST fitness={best_fitness:.0f}")
            else:
                no_improve += 1
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"[Tabu] Stopping at iter {it} (patience={patience})")
            break

    runtime = time.time() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Accurate final evaluation
    final_eval = fe.full_eval(best)

    if verbose:
        print(f"[Tabu] {iterations_done} iters in {runtime:.2f}s")
        print(f"[Tabu] {final_eval.summary()}")

    return {
        'solution': best,
        'runtime': runtime,
        'evaluation': final_eval,
        'algorithm': 'Tabu Search',
        'iterations': iterations_done,
        'memory_peak_mb': peak_mem / (1024 * 1024),
    }
