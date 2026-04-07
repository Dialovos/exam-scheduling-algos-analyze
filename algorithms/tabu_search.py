"""
Tabu Search — Feasibility-First
================================
Uses move_delta for fast neighbor evaluation.
Re-syncs with full_eval every 10 iterations to prevent drift.
When infeasible: focuses neighborhood on hard-violating exams.
"""

import time
import random as _random
from core.models import ProblemInstance, Solution
from core.fast_eval import FastEvaluator
from algorithms.greedy import solve_greedy


def solve_tabu(
    problem: ProblemInstance,
    max_iterations: int = 2000,
    tabu_tenure: int = 20,
    patience: int = 500,
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

    exam_dur = [e.duration for e in problem.exams]
    exam_enr = [e.enrollment for e in problem.exams]
    period_dur = [p.duration for p in problem.periods]
    room_cap = [r.capacity for r in problem.rooms]

    adj = [[] for _ in range(n_e)]
    for (e1, e2) in problem.conflict_matrix:
        adj[e1].append(e2)
        adj[e2].append(e1)

    valid_p = [[p for p in range(n_p) if exam_dur[e] <= period_dur[p]] for e in range(n_e)]
    valid_r = [[r for r in range(n_r) if exam_enr[e] <= room_cap[r]] for e in range(n_e)]

    # ── Initialize from greedy ──
    greedy_result = solve_greedy(problem, verbose=False, seed=seed)
    sol = greedy_result['solution']
    fe = FastEvaluator(problem)

    ev = fe.full_eval(sol)
    current_fitness = ev.fitness

    best_sol = sol.copy()
    best_fitness = current_fitness
    best_hard = ev.hard

    if verbose:
        print(f"[Tabu] Init: feasible={ev.feasible} hard={ev.hard} soft={ev.soft}")
    if ev.hard == 0 and max_iterations <= 0:
        return _result(best_sol, fe, t0, 0, verbose)

    # ── Tabu list ──
    tabu = {}
    no_improve = 0

    def get_bad_exams():
        bad = set()
        for eid in range(n_e):
            pid = sol._period_of[eid]
            if pid < 0:
                continue
            for nb in adj[eid]:
                if sol._period_of[nb] == pid:
                    bad.add(eid); bad.add(nb)
            rid = sol._room_of[eid]
            if sol.get_pr_enroll(pid, rid) > room_cap[rid]:
                bad.add(eid)
            if exam_dur[eid] > period_dur[pid]:
                bad.add(eid)
        for e1, ctype, e2 in fe.phcs:
            if e1 >= n_e or e2 >= n_e:
                continue
            p1, p2 = sol._period_of[e1], sol._period_of[e2]
            if p1 < 0 or p2 < 0:
                continue
            violated = False
            if ctype == "EXAM_COINCIDENCE" and p1 != p2: violated = True
            elif ctype == "EXCLUSION" and p1 == p2: violated = True
            elif ctype == "AFTER" and p1 <= p2: violated = True
            if violated:
                bad.add(e1); bad.add(e2)
        return bad

    # ── Main loop ──
    iters_done = 0
    for it in range(max_iterations):
        iters_done = it + 1

        # Re-sync every 10 iterations
        if it % 10 == 0:
            ev = fe.full_eval(sol)
            current_fitness = ev.fitness

        # Choose candidate exams
        bad = get_bad_exams()
        if bad:
            # When infeasible: focus on bad exams + some random
            candidates = list(bad)
            extra = min(20, n_e)
            for _ in range(extra):
                candidates.append(rng.randint(0, n_e - 1))
        else:
            # When feasible: sample broadly
            candidates = rng.sample(range(n_e), min(60, n_e))

        # Find best move
        best_eid, best_pid, best_rid, best_delta = -1, -1, -1, float('inf')

        for eid in candidates:
            cur_pid = sol._period_of[eid]
            targets = valid_p[eid]
            # When infeasible, search ALL periods for bad exams
            if eid not in bad and len(targets) > 12:
                targets = rng.sample(targets, 12)

            for pid in targets:
                if pid == cur_pid:
                    continue
                rooms = valid_r[eid] if valid_r[eid] else [0]
                for rid in rooms[:min(3, len(rooms))]:
                    d = fe.move_delta(sol, eid, pid, rid)
                    is_tabu = tabu.get((eid, cur_pid), 0) > it
                    if is_tabu and (current_fitness + d) >= best_fitness:
                        continue
                    if d < best_delta:
                        best_delta = d
                        best_eid, best_pid, best_rid = eid, pid, rid

        if best_eid < 0 or (best_delta >= 0 and bad):
            # No improving single move for infeasible state — try SWAP moves
            # Swap periods of two exams (chain move)
            overflow_exams = []
            for eid in bad:
                pid = sol._period_of[eid]
                rid = sol._room_of[eid]
                if pid >= 0 and rid >= 0 and sol.get_pr_enroll(pid, rid) > room_cap[rid]:
                    overflow_exams.append(eid)

            swap_candidates = overflow_exams if overflow_exams else list(bad)[:10]
            best_swap_delta = best_delta if best_eid >= 0 else float('inf')
            best_swap = None

            for ea in swap_candidates[:10]:
                pa = sol._period_of[ea]
                ra = sol._room_of[ea]
                # Try swapping with exams in other periods
                swap_targets = rng.sample(range(n_e), min(50, n_e))
                for eb in swap_targets:
                    if eb == ea:
                        continue
                    pb = sol._period_of[eb]
                    rb = sol._room_of[eb]
                    if pb == pa:
                        continue
                    if exam_dur[ea] > period_dur[pb] or exam_dur[eb] > period_dur[pa]:
                        continue
                    # Compute swap delta: move ea→pb, then eb→pa
                    d1 = fe.move_delta(sol, ea, pb, ra)
                    fe.apply_move(sol, ea, pb, ra)
                    d2 = fe.move_delta(sol, eb, pa, rb)
                    fe.apply_move(sol, eb, pa, rb)
                    # Undo
                    sol.assign(eb, pb, rb)
                    sol.assign(ea, pa, ra)
                    total_d = d1 + d2
                    if total_d < best_swap_delta:
                        best_swap_delta = total_d
                        best_swap = (ea, pb, ra, eb, pa, rb)

            if best_swap is not None and best_swap_delta < (best_delta if best_eid >= 0 else float('inf')):
                ea, pb, ra, eb, pa, rb = best_swap
                old_pa = sol._period_of[ea]
                old_pb = sol._period_of[eb]
                fe.apply_move(sol, ea, pb, ra)
                fe.apply_move(sol, eb, pa, rb)
                current_fitness += best_swap_delta
                tabu[(ea, old_pa)] = it + tabu_tenure
                tabu[(eb, old_pb)] = it + tabu_tenure
                best_eid = -2  # signal that a swap was applied

            if best_eid == -1:  # no single move AND no swap found
                no_improve += 1
                if no_improve > patience:
                    break
                continue

        if best_eid >= 0:
            # Apply single move
            old_pid = sol._period_of[best_eid]
            fe.apply_move(sol, best_eid, best_pid, best_rid)
            current_fitness += best_delta
            tabu[(best_eid, old_pid)] = it + tabu_tenure

        # Track best — verify with full_eval to prevent drift
        if current_fitness < best_fitness - 0.5:
            check = fe.full_eval(sol)
            actual_fitness = check.fitness
            # Only keep if ACTUALLY better
            if actual_fitness < best_fitness:
                best_sol = sol.copy()
                best_fitness = actual_fitness
                current_fitness = actual_fitness  # re-sync
                no_improve = 0
                if verbose and (it < 10 or it % 200 == 0):
                    print(f"[Tabu] Iter {it}: best hard={check.hard} soft={check.soft}")
            else:
                current_fitness = actual_fitness  # re-sync anyway
                no_improve += 1
        else:
            no_improve += 1
            if no_improve > patience:
                break

        # Perturbation when stuck
        if no_improve > 0 and no_improve % max(100, patience // 4) == 0:
            for _ in range(rng.randint(2, 5)):
                e = rng.randint(0, n_e - 1)
                vp = valid_p[e]
                vr = valid_r[e]
                if vp and vr:
                    d = fe.move_delta(sol, e, rng.choice(vp), rng.choice(vr))
                    fe.apply_move(sol, e, rng.choice(vp), rng.choice(vr))
            current_fitness = fe.full_eval(sol).fitness

    return _result(best_sol, fe, t0, iters_done, verbose)


def _result(sol, fe, t0, iters, verbose):
    ev = fe.full_eval(sol)
    runtime = time.time() - t0
    if verbose:
        print(f"[Tabu] {iters} iters, {runtime:.2f}s  "
              f"feasible={ev.feasible} hard={ev.hard} soft={ev.soft}")
    return {
        'solution': sol,
        'runtime': runtime,
        'evaluation': ev,
        'algorithm': 'Tabu Search',
        'iterations': iters,
    }