"""
DSatur Greedy with Constraint-Aware Placement & Repair
=======================================================
Root causes of infeasibility in the original greedy:
  1. Static degree ordering -> high-degree exams get stuck (Fix: DSatur)
  2. No period hard-constraint enforcement in fallback (Fix: coincidence groups)
  3. Room capacity overflow (Fix: best-fit + room rebalancing)
  4. Blind fallback to period 0 (Fix: least-damage + targeted repair)
"""
import time
import random as _random
from collections import defaultdict
from data.models import ProblemInstance, Solution
from data.fast_eval import FastEvaluator


def solve_greedy(problem: ProblemInstance, verbose: bool = False, seed: int = 42, **kw) -> dict:
    """Multi-start DSatur greedy. Tries seed 42 first, then random seeds if infeasible."""
    result = _solve_greedy_once(problem, verbose=verbose, seed=seed, **kw)
    if result['evaluation'].hard == 0:
        return result

    # Multi-start: try different seeds
    best = result
    t0 = time.time()
    max_attempts = 30
    if verbose:
        print(f"[Greedy] Infeasible (hard={result['evaluation'].hard}), trying {max_attempts} random seeds...")
    for s in range(max_attempts):
        r = _solve_greedy_once(problem, verbose=False, seed=s, **kw)
        ev = r['evaluation']
        bev = best['evaluation']
        if ev.hard < bev.hard or (ev.hard == bev.hard and ev.soft < bev.soft):
            best = r
            if ev.hard == 0:
                if verbose:
                    print(f"[Greedy] Feasible at seed {s}! soft={ev.soft} ({time.time()-t0:.1f}s)")
                break
    best['runtime'] = time.time() - t0 + result['runtime']
    return best


def _solve_greedy_once(problem: ProblemInstance, verbose: bool = False, seed: int = 42, **kw) -> dict:
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

    valid_periods = [[p for p in range(n_p) if exam_dur[e] <= period_dur[p]] for e in range(n_e)]
    valid_rooms = [sorted([r for r in range(n_r) if exam_enr[e] <= room_cap[r]],
                          key=lambda r: room_cap[r]) for e in range(n_e)]

    # ── Period hard constraints ─────────────────────────────
    coincidence = defaultdict(set)
    exclusion_of = defaultdict(set)     # exam -> set of exams that must NOT share period
    after_pairs = []                     # (e1, e2): e1 must be in a LATER period than e2
    after_of = defaultdict(list)         # exam -> [(other, direction)]  direction: 'after'/'before'

    for c in problem.period_hard_constraints:
        e1, e2 = c.exam1_id, c.exam2_id
        if e1 >= n_e or e2 >= n_e:
            continue
        if c.constraint_type == "EXAM_COINCIDENCE":
            coincidence[e1].add(e2)
            coincidence[e2].add(e1)
        elif c.constraint_type == "EXCLUSION":
            exclusion_of[e1].add(e2)
            exclusion_of[e2].add(e1)
        elif c.constraint_type == "AFTER":
            after_pairs.append((e1, e2))
            after_of[e1].append((e2, 'after'))   # e1 must come after e2
            after_of[e2].append((e1, 'before'))   # e2 must come before e1

    # Build coincidence groups
    coin_group = {}
    visited_coin = set()
    for e in range(n_e):
        if e in visited_coin or e not in coincidence:
            continue
        group = []
        stack = [e]
        while stack:
            cur = stack.pop()
            if cur in visited_coin:
                continue
            visited_coin.add(cur)
            group.append(cur)
            for nb in coincidence[cur]:
                if nb not in visited_coin:
                    stack.append(nb)
        for member in group:
            coin_group[member] = group

    sol = Solution(problem)
    assigned = [False] * n_e

    # ── Helpers ─────────────────────────────────────────────
    def blocked_periods(eid):
        blocked = set()
        for nb in adj[eid]:
            if assigned[nb]:
                p = sol._period_of[nb]
                if p >= 0:
                    blocked.add(p)
        for nb in exclusion_of[eid]:
            if assigned[nb]:
                p = sol._period_of[nb]
                if p >= 0:
                    blocked.add(p)
        return blocked

    def required_period(eid):
        for nb in coincidence.get(eid, set()):
            if assigned[nb]:
                return sol._period_of[nb]
        return None

    def after_ok(eid, pid):
        for other, direction in after_of[eid]:
            if not assigned[other]:
                continue
            o_pid = sol._period_of[other]
            if o_pid < 0:
                continue
            if direction == 'after' and pid <= o_pid:
                return False
            if direction == 'before' and pid >= o_pid:
                return False
        return True

    def find_room(eid, pid):
        for rid in valid_rooms[eid]:
            if sol.get_pr_enroll(pid, rid) + exam_enr[eid] <= room_cap[rid]:
                return rid
        return -1

    def try_place(eid):
        req = required_period(eid)
        if req is not None and req >= 0:
            rid = find_room(eid, req)
            if rid < 0:
                rid = valid_rooms[eid][0] if valid_rooms[eid] else 0
            sol.assign(eid, req, rid)
            return True

        blocked = blocked_periods(eid)
        for pid in valid_periods[eid]:
            if pid in blocked:
                continue
            if not after_ok(eid, pid):
                continue
            rid = find_room(eid, pid)
            if rid >= 0:
                sol.assign(eid, pid, rid)
                return True
        return False

    def try_displace(eid):
        blocked = blocked_periods(eid)
        valid_set = set(valid_periods[eid])
        candidates = sorted(valid_set & blocked,
                            key=lambda p: sum(1 for nb in adj[eid]
                                              if assigned[nb] and sol._period_of[nb] == p))
        for target in candidates:
            if not after_ok(eid, target):
                continue
            blockers = [nb for nb in adj[eid]
                        if assigned[nb] and sol._period_of[nb] == target]
            # Also check exclusion blockers
            for nb in exclusion_of[eid]:
                if assigned[nb] and sol._period_of[nb] == target and nb not in blockers:
                    blockers.append(nb)

            rollback = []
            all_moved = True
            for blocker in blockers:
                if blocker in coin_group:
                    group = coin_group[blocker]
                    if any(m != blocker and assigned[m] and sol._period_of[m] == target for m in group):
                        all_moved = False
                        break

                b_blocked = set()
                for bnb in adj[blocker]:
                    if assigned[bnb] and bnb != eid:
                        bp = sol._period_of[bnb]
                        if bp >= 0:
                            b_blocked.add(bp)
                for bnb in exclusion_of[blocker]:
                    if assigned[bnb] and bnb != eid:
                        bp = sol._period_of[bnb]
                        if bp >= 0:
                            b_blocked.add(bp)
                b_blocked.add(target)

                moved = False
                for b_pid in valid_periods[blocker]:
                    if b_pid in b_blocked:
                        continue
                    if not after_ok(blocker, b_pid):
                        continue
                    b_rid = find_room(blocker, b_pid)
                    if b_rid >= 0:
                        old_p, old_r = sol._period_of[blocker], sol._room_of[blocker]
                        sol.assign(blocker, b_pid, b_rid)
                        rollback.append((blocker, old_p, old_r))
                        moved = True
                        break
                if not moved:
                    all_moved = False
                    break

            if all_moved:
                rid = find_room(eid, target)
                if rid >= 0:
                    sol.assign(eid, target, rid)
                    return True

            for b_eid, b_op, b_or_ in reversed(rollback):
                sol.assign(b_eid, b_op, b_or_)
        return False

    def force_place(eid):
        req = required_period(eid)
        if req is not None and req >= 0:
            rid = find_room(eid, req)
            if rid < 0:
                rid = valid_rooms[eid][0] if valid_rooms[eid] else 0
            sol.assign(eid, req, rid)
            return

        best_pid, best_rid, best_cost = -1, -1, float('inf')
        for pid in valid_periods[eid]:
            conflicts = sum(1 for nb in adj[eid] if assigned[nb] and sol._period_of[nb] == pid)
            excl_viol = sum(1 for nb in exclusion_of[eid] if assigned[nb] and sol._period_of[nb] == pid)
            after_viol = 0 if after_ok(eid, pid) else 1
            rid = find_room(eid, pid)
            if rid >= 0:
                overflow = 0
            else:
                # Calculate actual overflow for best available room
                rid = valid_rooms[eid][0] if valid_rooms[eid] else 0
                overflow = sol.get_pr_enroll(pid, rid) + exam_enr[eid] - room_cap[rid]
                overflow = max(0, overflow)
            # Proportional costs: small overflow is much better than large overflow
            cost = (conflicts + excl_viol) * 100000 + after_viol * 50000 + overflow * 100
            if cost < best_cost:
                best_cost, best_pid, best_rid = cost, pid, rid
        if best_pid < 0:
            best_pid, best_rid = 0, 0
        sol.assign(eid, best_pid, best_rid)

    # ── DSatur ordering ─────────────────────────────────────
    degree = [len(adj[e]) + len(exclusion_of[e]) for e in range(n_e)]
    sat_counts = [0] * n_e
    # Room pressure: exams using >50% of max room cap should be placed early
    max_cap = max(room_cap) if room_cap else 1
    room_pressure = [int(10 * exam_enr[e] / max_cap) for e in range(n_e)]

    def pick_next():
        # Collect top candidates
        candidates = []
        for e in range(n_e):
            if assigned[e]:
                continue
            has_req = 1 if required_period(e) is not None else 0
            key = (has_req, sat_counts[e], degree[e] + room_pressure[e], exam_enr[e], e)
            candidates.append((key, e))
        if not candidates:
            return -1
        candidates.sort(key=lambda x: x[0], reverse=True)
        # Deterministic for seed=42, randomized for others
        if seed != 42 and len(candidates) > 1:
            top_k = min(3, len(candidates))
            return candidates[rng.randint(0, top_k - 1)][1]
        return candidates[0][1]

    def update_sat(eid):
        pid = sol._period_of[eid]
        if pid < 0:
            return
        for nb in adj[eid]:
            if not assigned[nb]:
                sat_counts[nb] += 1
        for nb in exclusion_of[eid]:
            if not assigned[nb]:
                sat_counts[nb] += 1

    # ── Pre-placement: room-dominating exams ──────────────
    stats = [0, 0, 0]
    # Exams using > 50% of room capacity need near-exclusive periods.
    # Place them first (sorted by enrollment desc) to claim the best slots.
    dominating = sorted(
        [e for e in range(n_e) if exam_enr[e] > max_cap * 0.5],
        key=lambda e: -exam_enr[e]
    )
    for eid in dominating:
        if assigned[eid]:
            continue
        if try_place(eid):
            assigned[eid] = True
            update_sat(eid)
            stats[0] += 1

    # ── Main loop ───────────────────────────────────────────
    for _ in range(n_e):
        eid = pick_next()
        if eid < 0:
            break
        if try_place(eid):
            assigned[eid] = True
            update_sat(eid)
            stats[0] += 1
        elif try_displace(eid):
            assigned[eid] = True
            update_sat(eid)
            stats[1] += 1
        else:
            force_place(eid)
            assigned[eid] = True
            update_sat(eid)
            stats[2] += 1

    # ── Repair pass ─────────────────────────────────────────
    fe = FastEvaluator(problem)
    ev = fe.full_eval(sol)
    initial_hard = ev.hard

    if ev.hard > 0:
        _repair(problem, sol, fe, adj, valid_periods, valid_rooms,
                exam_dur, exam_enr, period_dur, room_cap, n_p, n_r, verbose)
        ev = fe.full_eval(sol)

    runtime = time.time() - t0
    if verbose:
        print(f"[Greedy] {runtime:.3f}s  clean={stats[0]} displaced={stats[1]} forced={stats[2]}")
        print(f"[Greedy] feasible={ev.feasible} hard={ev.hard} soft={ev.soft}")
        if initial_hard > 0:
            print(f"[Greedy] Repair: {initial_hard} -> {ev.hard} hard violations")

    return {
        'solution': sol,
        'runtime': runtime,
        'evaluation': ev,
        'algorithm': 'Greedy',
        'iterations': 0,
    }


def _repair(problem, sol, fe, adj, valid_periods, valid_rooms,
            exam_dur, exam_enr, period_dur, room_cap, n_p, n_r, verbose):
    """Unified repair with chain-move room overflow fix."""
    n_e = problem.num_exams()

    # ── Phase 0: Chain-move room overflow repair ────────────
    # For each overflow, try to move the large exam to another period
    # while simultaneously moving small exams out of the target to make room.
    for _ in range(10):
        ev = fe.full_eval(sol)
        if ev.room_occupancy == 0:
            break

        for src_pid in range(n_p):
            for src_rid in range(n_r):
                total_enr = sol.get_pr_enroll(src_pid, src_rid)
                if total_enr <= room_cap[src_rid]:
                    continue

                # Find the largest exam causing overflow
                exams_here = [(e, exam_enr[e]) for e in range(n_e)
                              if sol._period_of[e] == src_pid and sol._room_of[e] == src_rid]
                exams_here.sort(key=lambda x: -x[1])
                if not exams_here:
                    continue

                large_eid, large_enr = exams_here[0]

                # Try moving large exam to each candidate period
                for tgt_pid in range(n_p):
                    if tgt_pid == src_pid:
                        continue
                    if exam_dur[large_eid] > period_dur[tgt_pid]:
                        continue
                    # Check student conflicts
                    has_conflict = any(sol._period_of[nb] == tgt_pid for nb in adj[large_eid])
                    if has_conflict:
                        continue

                    tgt_enr = sol.get_pr_enroll(tgt_pid, src_rid)
                    after_move = tgt_enr + large_enr
                    need_free = after_move - room_cap[src_rid]
                    if need_free <= 0:
                        # Direct move works! Do it.
                        sol.assign(large_eid, tgt_pid, src_rid)
                        break

                    # Need to free 'need_free' enrollment from tgt_pid
                    # Find small movable exams in tgt_pid
                    tgt_exams = [(e, exam_enr[e]) for e in range(n_e)
                                 if sol._period_of[e] == tgt_pid and sol._room_of[e] == src_rid
                                 and e != large_eid]
                    tgt_exams.sort(key=lambda x: x[1])  # smallest first

                    freed = 0
                    moves = []  # (eid, new_pid, new_rid)
                    for move_eid, move_enr in tgt_exams:
                        if freed >= need_free:
                            break
                        # Can this exam move to any other period?
                        for alt_pid in range(n_p):
                            if alt_pid == tgt_pid or alt_pid == src_pid:
                                continue
                            if exam_dur[move_eid] > period_dur[alt_pid]:
                                continue
                            mc = any(sol._period_of[nb] == alt_pid for nb in adj[move_eid])
                            if mc:
                                continue
                            # Check room capacity at alt_pid
                            for alt_rid in range(n_r):
                                if (exam_enr[move_eid] <= room_cap[alt_rid] and
                                    sol.get_pr_enroll(alt_pid, alt_rid) + exam_enr[move_eid] <= room_cap[alt_rid]):
                                    moves.append((move_eid, alt_pid, alt_rid))
                                    freed += move_enr
                                    break
                            if freed >= need_free or (moves and moves[-1][0] == move_eid):
                                break

                    if freed >= need_free:
                        # Execute chain: move small exams out, then move large exam in
                        for m_eid, m_pid, m_rid in moves:
                            sol.assign(m_eid, m_pid, m_rid)
                        sol.assign(large_eid, tgt_pid, src_rid)
                        if verbose:
                            print(f"[Repair] Chain move: exam {large_eid}(enr={large_enr}) "
                                  f"from period {src_pid}→{tgt_pid}, freed {freed} enrollment")
                        break
                else:
                    continue
                break  # restart overflow scan after a successful fix

    for round_i in range(200):
        ev = fe.full_eval(sol)
        if ev.hard == 0:
            return

        # Collect exams involved in ANY hard violation
        bad_exams = set()

        # Student conflicts
        for eid in range(n_e):
            pid = sol._period_of[eid]
            if pid < 0:
                continue
            for nb in adj[eid]:
                if sol._period_of[nb] == pid:
                    bad_exams.add(eid)
                    bad_exams.add(nb)

        # Room capacity
        for pid in range(n_p):
            for rid in range(n_r):
                if sol.get_pr_enroll(pid, rid) > room_cap[rid]:
                    for e in range(n_e):
                        if sol._period_of[e] == pid and sol._room_of[e] == rid:
                            bad_exams.add(e)

        # Period utilisation (exam too long for period)
        for eid in range(n_e):
            pid = sol._period_of[eid]
            if pid >= 0 and exam_dur[eid] > period_dur[pid]:
                bad_exams.add(eid)

        # Period hard constraints
        for c in problem.period_hard_constraints:
            e1, e2 = c.exam1_id, c.exam2_id
            if e1 >= n_e or e2 >= n_e:
                continue
            p1, p2 = sol._period_of[e1], sol._period_of[e2]
            if p1 < 0 or p2 < 0:
                continue
            violated = False
            if c.constraint_type == "EXAM_COINCIDENCE" and p1 != p2:
                violated = True
            elif c.constraint_type == "EXCLUSION" and p1 == p2:
                violated = True
            elif c.constraint_type == "AFTER" and p1 <= p2:
                violated = True
            if violated:
                bad_exams.add(e1)
                bad_exams.add(e2)

        if not bad_exams:
            return

        # For each bad exam, find its best improving move
        best_eid, best_pid, best_rid, best_delta = -1, -1, -1, 0.0

        for eid in bad_exams:
            cur_pid = sol._period_of[eid]
            for pid in range(n_p):
                if pid == cur_pid:
                    continue
                if exam_dur[eid] > period_dur[pid]:
                    continue
                for rid in range(n_r):
                    delta = fe.move_delta(sol, eid, pid, rid)
                    if delta < best_delta:
                        best_delta = delta
                        best_eid, best_pid, best_rid = eid, pid, rid

        if best_eid < 0 or best_delta >= 0:
            # No improving move found — try accepting sideways moves
            # (same hard, different soft) for diversification
            for eid in bad_exams:
                cur_pid = sol._period_of[eid]
                for pid in range(n_p):
                    if pid == cur_pid:
                        continue
                    if exam_dur[eid] > period_dur[pid]:
                        continue
                    for rid in range(n_r):
                        delta = fe.move_delta(sol, eid, pid, rid)
                        # Accept if it doesn't increase hard (delta < 100000)
                        if delta < best_delta and delta < 50000:
                            best_delta = delta
                            best_eid, best_pid, best_rid = eid, pid, rid
            if best_eid < 0:
                break

        fe.apply_move(sol, best_eid, best_pid, best_rid)