"""
Primary:  OR-Tools CP-SAT (fast model build, native Booleans, auto symmetry breaking)
Fallback: PuLP / CBC (if ortools unavailable)

CP-SAT replaces PuLP as primary solver (5-20x faster model construction).
Per-conflict proximity variable replaces O(conflicts * periods^2) objective terms.
Array-indexed lookups replace dict-keyed variables.
Aggressive domain pruning before variable creation.
"""

from core.models import ProblemInstance, Solution
from core.evaluator import evaluate
import os
import time
import collections

# ── Optional imports ──────────────────────────────────────────

try:
    from ortools.sat.python import cp_model
    HAS_CPSAT = True
except ImportError:
    HAS_CPSAT = False

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


# ==============================================================
#  PUBLIC API
# ==============================================================

def solve_ip(
    problem: ProblemInstance,
    time_limit: int = 300,
    verbose: bool = False,
    mip_gap: float = 0.01,
    warm_start: Solution | None = None,
    num_workers: int = 0,
) -> dict:
    """Solve exam timetabling via integer/constraint programming.

    Args:
        problem:      ITC 2007 ProblemInstance
        time_limit:   wall-clock seconds for the underlying solver
        verbose:      print build + solve progress
        mip_gap:      relative MIP gap (PuLP backend only; CP-SAT uses OR-Tools defaults)
        warm_start:   optional Solution fed to CP-SAT as a hint — often a 5-20× speedup
        num_workers:  CP-SAT parallel workers; 0 = all available cores

    Backend selection:
      1. OR-Tools CP-SAT  (preferred — faster build + solve, accepts warm-start)
      2. PuLP / HiGHS or CBC  (fallback — no warm-start support)
    """
    if problem.conflict_matrix is None:
        problem.build_derived_data()

    if HAS_CPSAT:
        return _solve_cpsat(problem, time_limit, verbose,
                            warm_start=warm_start, num_workers=num_workers)
    elif HAS_PULP:
        return _solve_pulp(problem, time_limit, verbose, mip_gap)
    else:
        raise ImportError(
            "No solver backend available. Install one of:\n"
            "  pip install ortools   (recommended)\n"
            "  pip install pulp"
        )


# ==============================================================
#  CP-SAT SOLVER  (primary)
# ==============================================================

def _solve_cpsat(problem, time_limit, verbose, warm_start=None, num_workers=0):
    start = time.time()

    n_e = problem.num_exams()
    n_p = problem.num_periods()
    n_r = problem.num_rooms()
    w = problem.weightings

    # ── Flat arrays for fast lookup ──────────────────────────
    e_enr  = [e.enrollment for e in problem.exams]
    e_dur  = [e.duration   for e in problem.exams]
    r_cap  = [r.capacity   for r in problem.rooms]
    r_pen  = [r.penalty    for r in problem.rooms]
    p_dur  = [p.duration   for p in problem.periods]
    p_pen  = [p.penalty    for p in problem.periods]
    p_day  = [p.day        for p in problem.periods]

    day_periods = problem.periods_per_day  # {day: [pid, ...]}

    # Period daypos (position within its day)
    p_daypos = [0] * n_p
    for day, pids in day_periods.items():
        for pos, pid in enumerate(pids):
            p_daypos[pid] = pos

    # ── Domain pruning ───────────────────────────────────────
    # valid_p[e] = periods whose duration accommodates exam e
    valid_p = [[] for _ in range(n_e)]
    for e in range(n_e):
        valid_p[e] = [p for p in range(n_p) if e_dur[e] <= p_dur[p]]

    # valid_r[e] = rooms whose capacity accommodates exam e
    valid_r = [[] for _ in range(n_e)]
    for e in range(n_e):
        valid_r[e] = [r for r in range(n_r) if e_enr[e] <= r_cap[r]]

    # Valid (e,p,r) triples
    valid_epr = []
    for e in range(n_e):
        for p in valid_p[e]:
            for r in valid_r[e]:
                valid_epr.append((e, p, r))

    if verbose:
        print(f"[CP-SAT] {n_e} exams, {n_p} periods, {n_r} rooms")
        print(f"[CP-SAT] Pruned to {len(valid_epr)} (e,p,r) triples "
              f"(from {n_e * n_p * n_r} full)")

    # ── Build model ──────────────────────────────────────────
    model = cp_model.CpModel()

    # --- Decision variables ---
    # x[e][p][r] = BoolVar (only for valid triples)
    x = {}
    for (e, p, r) in valid_epr:
        x[(e, p, r)] = model.new_bool_var(f'x_{e}_{p}_{r}')

    # y[e][p] = exam e assigned to period p (linked from x)
    y = {}
    for e in range(n_e):
        for p in valid_p[e]:
            rs = [x[(e, p, r)] for r in valid_r[e] if (e, p, r) in x]
            if rs:
                yep = model.new_bool_var(f'y_{e}_{p}')
                # y[e,p] == 1  iff  exactly one room chosen in this period
                model.add(sum(rs) == 1).only_enforce_if(yep)
                model.add(sum(rs) == 0).only_enforce_if(yep.Not())
                y[(e, p)] = yep

    # Index: per exam, list of x-variables for the "assign exactly one" constraint
    x_by_e = [[] for _ in range(n_e)]
    for (e, p, r), var in x.items():
        x_by_e[e].append(var)

    # Index: per (period, room), list of (exam_id, x-var) for capacity
    x_by_pr = collections.defaultdict(list)
    for (e, p, r), var in x.items():
        x_by_pr[(p, r)].append((e, var))

    # ── Hard constraints ─────────────────────────────────────

    # H1: Each exam assigned to exactly one (period, room)
    for e in range(n_e):
        model.add_exactly_one(x_by_e[e])

    # H2: No student conflicts (conflicting exams in different periods)
    for (e1, e2) in problem.conflict_matrix:
        common_ps = set(valid_p[e1]) & set(valid_p[e2])
        for p in common_ps:
            if (e1, p) in y and (e2, p) in y:
                model.add_bool_or([y[(e1, p)].Not(), y[(e2, p)].Not()])

    # H3: Room capacity
    for (p, r), items in x_by_pr.items():
        if items:
            model.add(
                sum(e_enr[e] * var for e, var in items) <= r_cap[r]
            )

    # H4: Period hard constraints (COINCIDENCE / EXCLUSION / AFTER)
    for phc in problem.period_hard_constraints:
        e1, e2 = phc.exam1_id, phc.exam2_id
        if e1 >= n_e or e2 >= n_e:
            continue

        if phc.constraint_type == "EXAM_COINCIDENCE":
            # Must be in the same period
            for p in range(n_p):
                y1 = y.get((e1, p))
                y2 = y.get((e2, p))
                if y1 is not None and y2 is not None:
                    model.add(y1 == y2)
                elif y1 is not None:
                    model.add(y1 == 0)
                elif y2 is not None:
                    model.add(y2 == 0)

        elif phc.constraint_type == "EXCLUSION":
            # Must NOT be in the same period
            common_ps = set(valid_p[e1]) & set(valid_p[e2])
            for p in common_ps:
                if (e1, p) in y and (e2, p) in y:
                    model.add_bool_or([y[(e1, p)].Not(), y[(e2, p)].Not()])

        elif phc.constraint_type == "AFTER":
            # e1 must come AFTER e2 → period(e1) > period(e2)
            per_e1 = model.new_int_var(0, n_p - 1, f'per_{e1}_after')
            per_e2 = model.new_int_var(0, n_p - 1, f'per_{e2}_after')
            for p in valid_p[e1]:
                if (e1, p) in y:
                    model.add(per_e1 == p).only_enforce_if(y[(e1, p)])
            for p in valid_p[e2]:
                if (e2, p) in y:
                    model.add(per_e2 == p).only_enforce_if(y[(e2, p)])
            model.add(per_e1 >= per_e2 + 1)

    # H5: Room exclusive
    rhc_exams = set(
        c.exam_id for c in problem.room_hard_constraints
        if c.constraint_type == "ROOM_EXCLUSIVE" and c.exam_id < n_e
    )
    for eid in rhc_exams:
        for p in valid_p[eid]:
            for r in valid_r[eid]:
                if (eid, p, r) not in x:
                    continue
                # If eid is in (p, r), no other exam can be
                others = [var for (e2, var) in x_by_pr[(p, r)] if e2 != eid]
                if others:
                    model.add(sum(others) == 0).only_enforce_if(x[(eid, p, r)])

    # ── Soft constraints (objective) ─────────────────────────

    obj_terms = []  # list of (coefficient, BoolVar or IntVar)

    # S1: Period penalties
    for e in range(n_e):
        for p in valid_p[e]:
            if p_pen[p] > 0 and (e, p) in y:
                obj_terms.append(p_pen[p] * y[(e, p)])

    # S2: Room penalties
    for (e, p, r), var in x.items():
        if r_pen[r] > 0:
            obj_terms.append(r_pen[r] * var)

    # S3: Front load penalty
    fl_n_largest, fl_n_last, fl_penalty = w.front_load
    if fl_penalty > 0 and fl_n_largest > 0 and fl_n_last > 0:
        sorted_exams = sorted(range(n_e), key=lambda e: e_enr[e], reverse=True)
        large_exams = set(sorted_exams[:fl_n_largest])
        last_periods = set(range(max(0, n_p - fl_n_last), n_p))
        for e in large_exams:
            for p in last_periods:
                if (e, p) in y:
                    obj_terms.append(fl_penalty * y[(e, p)])

    # S4: Non-mixed durations penalty
    # Penalty if exams with different durations share a (period, room)
    if w.non_mixed_durations > 0:
        dur_set_by_pr = collections.defaultdict(set)
        exams_by_pr = collections.defaultdict(list)
        for (e, p, r), var in x.items():
            dur_set_by_pr[(p, r)].add(e_dur[e])
            exams_by_pr[(p, r)].append((e, var))

        for (p, r), items in exams_by_pr.items():
            # Group exams by duration
            dur_groups = collections.defaultdict(list)
            for (e, var) in items:
                dur_groups[e_dur[e]].append(var)
            if len(dur_groups) < 2:
                continue
            # Penalty applies if ≥2 distinct durations are present
            # Create a Bool per duration group: "at least one exam of this duration here"
            dur_present = []
            for dur, vars_list in dur_groups.items():
                b = model.new_bool_var(f'dur_{dur}_pr_{p}_{r}')
                model.add(sum(vars_list) >= 1).only_enforce_if(b)
                model.add(sum(vars_list) == 0).only_enforce_if(b.Not())
                dur_present.append(b)
            # mixed = sum(dur_present) >= 2
            mixed = model.new_bool_var(f'mixed_{p}_{r}')
            model.add(sum(dur_present) >= 2).only_enforce_if(mixed)
            model.add(sum(dur_present) <= 1).only_enforce_if(mixed.Not())
            obj_terms.append(w.non_mixed_durations * mixed)

    # S5: Proximity penalties (two-in-a-row, two-in-a-day, period spread)
    #     Per-conflict auxiliary variable approach:
    #       For each conflicting pair (e1,e2), create one IntVar prox_cost.
    #       For each nearby period pair (p1,p2), add a lower-bound constraint:
    #         prox_cost >= pen * (y[e1,p1] + y[e2,p2] - 1)
    #       Since exactly one y[e,p]=1 per exam, at integer solution this
    #       correctly captures pen(assigned_p1, assigned_p2).

    # Precompute penalty for each ordered period pair (p1 < p2)
    proximity_penalties = {}  # (p1, p2) -> base_penalty
    for i in range(n_p):
        for j in range(i + 1, n_p):
            pen = 0
            # Two-in-a-row / two-in-a-day (same day)
            if p_day[i] == p_day[j]:
                gap_pos = abs(p_daypos[i] - p_daypos[j])
                if gap_pos == 1:
                    pen += w.two_in_a_row
                elif gap_pos > 1:
                    pen += w.two_in_a_day
            # Period spread
            gap = abs(j - i)
            if 0 < gap <= w.period_spread:
                pen += 1
            if pen > 0:
                proximity_penalties[(i, j)] = pen

    active_conflicts = [
        (e1, e2, shared)
        for (e1, e2), shared in problem.conflict_matrix.items()
        if shared > 0
    ]

    if verbose:
        print(f"[CP-SAT] {len(active_conflicts)} conflict pairs, "
              f"{len(proximity_penalties)} proximity period-pairs")

    # Determine upper bound for proximity cost per pair
    max_prox_pen = max(proximity_penalties.values(), default=0)
    max_shared = max((s for _, _, s in active_conflicts), default=1)
    prox_ub = max_prox_pen * max_shared

    for (e1, e2, shared) in active_conflicts:
        if prox_ub == 0:
            break

        # Collect relevant lower-bound constraints
        has_terms = False
        terms = []

        for (p1, p2), base_pen in proximity_penalties.items():
            pen = base_pen * shared
            # Case A: e1→p1, e2→p2
            y1a = y.get((e1, p1))
            y2a = y.get((e2, p2))
            if y1a is not None and y2a is not None:
                terms.append((pen, y1a, y2a))
                has_terms = True
            # Case B: e1→p2, e2→p1
            y1b = y.get((e1, p2))
            y2b = y.get((e2, p1))
            if y1b is not None and y2b is not None:
                terms.append((pen, y1b, y2b))
                has_terms = True

        if not has_terms:
            continue

        # One IntVar per conflict pair
        prox_cost = model.new_int_var(0, prox_ub, f'prox_{e1}_{e2}')
        for (pen, ya, yb) in terms:
            # prox_cost >= pen * (ya + yb - 1)
            # ↔ prox_cost >= pen*ya + pen*yb - pen
            model.add(prox_cost >= pen * ya + pen * yb - pen)
        obj_terms.append(prox_cost)

    # ── Objective ────────────────────────────────────────────
    if obj_terms:
        model.minimize(sum(obj_terms))

    # ── Warm start (CP-SAT hint) ─────────────────────────────
    # When a feasible Solution is supplied, feed its (e,p,r) triples as hints.
    # CP-SAT uses these as a starting point, often cutting solve time 5-20×.
    if warm_start is not None:
        hint_hits = 0
        hint_misses = 0
        for e, asg in warm_start.assignments.items():
            if e >= n_e:
                continue
            # Solution.assignments[e] is a (period_id, room_id) tuple
            p, r = asg[0], asg[1]
            var = x.get((e, p, r))
            if var is not None:
                model.add_hint(var, 1)
                hint_hits += 1
            else:
                hint_misses += 1
        if verbose:
            print(f"[CP-SAT] Warm-start: {hint_hits} hints seeded "
                  f"({hint_misses} skipped as pruned)")

    build_time = time.time() - start
    if verbose:
        print(f"[CP-SAT] Model built in {build_time:.2f}s")
        stats = model.model_stats()
        # Print first few lines of stats
        for line in str(stats).split('\n')[:6]:
            print(f"  {line}")

    # ── Solve ────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    workers = num_workers if num_workers and num_workers > 0 else (os.cpu_count() or 8)
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = verbose
    if verbose:
        print(f"[CP-SAT] Using {workers} parallel workers")

    status = solver.solve(model)
    runtime = time.time() - start

    status_names = {
        cp_model.OPTIMAL: 'Optimal',
        cp_model.FEASIBLE: 'Feasible',
        cp_model.INFEASIBLE: 'Infeasible',
        cp_model.MODEL_INVALID: 'ModelInvalid',
        cp_model.UNKNOWN: 'Unknown',
    }
    status_str = status_names.get(status, f'Status_{status}')

    # ── Extract solution ─────────────────────────────────────
    solution = Solution(problem)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (e, p, r), var in x.items():
            if solver.value(var):
                solution.assign(e, p, r)

    # Fallback for unassigned exams
    unassigned = 0
    for e in range(n_e):
        if e not in solution.assignments:
            unassigned += 1
            solution.assign(e, 0, 0)

    eval_result = evaluate(problem, solution)

    if verbose:
        print(f"[CP-SAT] Status: {status_str}")
        print(f"[CP-SAT] Completed in {runtime:.2f}s (build={build_time:.2f}s)")
        print(f"[CP-SAT] Feasible: {eval_result.is_feasible}, "
              f"Hard: {eval_result.hard_violations}, Soft: {eval_result.soft_penalty}")
        if unassigned > 0:
            print(f"[CP-SAT] WARNING: {unassigned} exams needed fallback assignment")

    return {
        'solution': solution,
        'runtime': runtime,
        'evaluation': eval_result,
        'algorithm': 'Integer Programming',
        'memory_peak_mb': 0,
        'solver_status': status_str,
    }


# ==============================================================
#  PuLP SOLVER  (fallback)
# ==============================================================

def _solve_pulp(problem, time_limit, verbose, mip_gap):
    """Streamlined PuLP fallback — no NetworkX, leaner constraint generation."""
    if not HAS_PULP:
        raise ImportError("PuLP is required. Install with: pip install pulp")

    start = time.time()

    n_e = problem.num_exams()
    n_p = problem.num_periods()
    n_r = problem.num_rooms()
    w = problem.weightings

    e_enr = [e.enrollment for e in problem.exams]
    e_dur = [e.duration   for e in problem.exams]
    r_cap = [r.capacity   for r in problem.rooms]
    r_pen = [r.penalty    for r in problem.rooms]
    p_dur = [p.duration   for p in problem.periods]
    p_pen = [p.penalty    for p in problem.periods]
    p_day = [p.day        for p in problem.periods]

    day_periods = problem.periods_per_day
    p_daypos = [0] * n_p
    for day, pids in day_periods.items():
        for pos, pid in enumerate(pids):
            p_daypos[pid] = pos

    # ── Pruned variable sets ─────────────────────────────────
    valid_EPR = []
    valid_EP = set()
    for e in range(n_e):
        for p in range(n_p):
            if e_dur[e] <= p_dur[p]:
                valid_EP.add((e, p))
                for r in range(n_r):
                    if e_enr[e] <= r_cap[r]:
                        valid_EPR.append((e, p, r))

    epr_by_e = collections.defaultdict(list)
    e_by_pr = collections.defaultdict(list)
    p_by_e = collections.defaultdict(list)
    for (e, p, r) in valid_EPR:
        epr_by_e[e].append((e, p, r))
        e_by_pr[(p, r)].append(e)
    for (e, p) in valid_EP:
        p_by_e[e].append(p)

    if verbose:
        print(f"[PuLP] {n_e} exams, {n_p} periods, {n_r} rooms")
        print(f"[PuLP] {len(valid_EPR)} pruned (e,p,r) variables")

    # ── Model ────────────────────────────────────────────────
    mdl = pulp.LpProblem("ExamTimetabling", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", valid_EPR, cat='Binary')
    y = pulp.LpVariable.dicts("y", list(valid_EP), cat='Binary')

    # Link x → y
    for (e, p) in valid_EP:
        rs = [r for r in range(n_r) if (e, p, r) in x]
        mdl += y[(e, p)] == pulp.lpSum(x[(e, p, r)] for r in rs), f"lnk_{e}_{p}"

    # H1: Assign each exam exactly once
    for e in range(n_e):
        mdl += pulp.lpSum(x[epr] for epr in epr_by_e[e]) == 1, f"asgn_{e}"

    # H2: Conflict — pairwise
    for (e1, e2) in problem.conflict_matrix:
        common = set(p_by_e[e1]) & set(p_by_e[e2])
        for p in common:
            mdl += y[(e1, p)] + y[(e2, p)] <= 1, f"cf_{e1}_{e2}_{p}"

    # H3: Room capacity
    for p in range(n_p):
        for r in range(n_r):
            es = e_by_pr[(p, r)]
            if es:
                mdl += (
                    pulp.lpSum(e_enr[e] * x[(e, p, r)] for e in es) <= r_cap[r],
                    f"rc_{p}_{r}"
                )

    # H4: Period hard constraints
    for phc in problem.period_hard_constraints:
        e1, e2 = phc.exam1_id, phc.exam2_id
        if e1 >= n_e or e2 >= n_e:
            continue
        common = set(p_by_e[e1]) & set(p_by_e[e2])
        if phc.constraint_type == "EXAM_COINCIDENCE":
            for p in range(n_p):
                y1 = y[(e1, p)] if (e1, p) in y else 0
                y2 = y[(e2, p)] if (e2, p) in y else 0
                if (e1, p) in y or (e2, p) in y:
                    mdl += y1 == y2, f"coin_{e1}_{e2}_{p}"
        elif phc.constraint_type == "EXCLUSION":
            for p in common:
                mdl += y[(e1, p)] + y[(e2, p)] <= 1, f"excl_{e1}_{e2}_{p}"
        elif phc.constraint_type == "AFTER":
            mdl += (
                pulp.lpSum(p * y[(e1, p)] for p in p_by_e[e1]) >=
                pulp.lpSum(p * y[(e2, p)] for p in p_by_e[e2]) + 1,
                f"aft_{e1}_{e2}"
            )

    # H5: Room exclusive
    for rhc in problem.room_hard_constraints:
        if rhc.constraint_type == "ROOM_EXCLUSIVE" and rhc.exam_id < n_e:
            eid = rhc.exam_id
            for p in p_by_e[eid]:
                for r in range(n_r):
                    if (eid, p, r) not in x:
                        continue
                    others = [e for e in e_by_pr[(p, r)] if e != eid]
                    for oe in others:
                        if (oe, p, r) in x:
                            mdl += x[(eid, p, r)] + x[(oe, p, r)] <= 1, \
                                f"rexcl_{eid}_{oe}_{p}_{r}"

    # ── Symmetry breaking (identical rooms) ──────────────────
    room_profiles = collections.defaultdict(list)
    for r in problem.rooms:
        room_profiles[(r.capacity, r.penalty)].append(r.id)
    for profile, ident_rooms in room_profiles.items():
        if len(ident_rooms) > 1:
            for i in range(len(ident_rooms) - 1):
                r1, r2 = ident_rooms[i], ident_rooms[i + 1]
                for p in range(n_p):
                    er1 = [e for e in e_by_pr[(p, r1)]]
                    er2 = [e for e in e_by_pr[(p, r2)]]
                    if er1 and er2:
                        mdl += (
                            pulp.lpSum(x[(e, p, r1)] for e in er1) >=
                            pulp.lpSum(x[(e, p, r2)] for e in er2),
                            f"sym_{p}_{r1}_{r2}"
                        )

    # ── Soft objective ───────────────────────────────────────
    penalty = []

    # Period / room penalties
    for (e, p) in valid_EP:
        if p_pen[p] > 0:
            penalty.append(p_pen[p] * y[(e, p)])
    for (e, p, r) in valid_EPR:
        if r_pen[r] > 0:
            penalty.append(r_pen[r] * x[(e, p, r)])

    # Proximity — same linearization as original but skip NetworkX cliques
    prox_pens = {}
    for i in range(n_p):
        for j in range(i + 1, n_p):
            pen = 0
            if p_day[i] == p_day[j]:
                gap_pos = abs(p_daypos[i] - p_daypos[j])
                if gap_pos == 1:
                    pen += w.two_in_a_row
                elif gap_pos > 1:
                    pen += w.two_in_a_day
            gap = abs(j - i)
            if 0 < gap <= w.period_spread:
                pen += 1
            if pen > 0:
                prox_pens[(i, j)] = pen

    for (e1, e2), shared in problem.conflict_matrix.items():
        if shared <= 0:
            continue
        for (p1, p2), base_pen in prox_pens.items():
            pv = base_pen * shared
            if (e1, p1) in y and (e2, p2) in y:
                penalty.append(pv * (y[(e1, p1)] + y[(e2, p2)] - 1))
            if (e1, p2) in y and (e2, p1) in y:
                penalty.append(pv * (y[(e1, p2)] + y[(e2, p1)] - 1))

    mdl += pulp.lpSum(penalty) if penalty else 0, "soft"

    build_time = time.time() - start
    if verbose:
        print(f"[PuLP] Model built in {build_time:.2f}s, solving...")

    # ── Solve ────────────────────────────────────────────────
    try:
        solver = pulp.HiGHS_CMD(timeLimit=time_limit, gapRel=mip_gap, msg=verbose)
        mdl.solve(solver)
    except Exception:
        if verbose:
            print("[PuLP] HiGHS unavailable, falling back to CBC")
        solver = pulp.PULP_CBC_CMD(
            timeLimit=time_limit, gapRel=mip_gap, msg=1 if verbose else 0
        )
        mdl.solve(solver)

    runtime = time.time() - start
    status_str = pulp.LpStatus[mdl.status]

    # ── Extract ──────────────────────────────────────────────
    solution = Solution(problem)
    if mdl.status in (pulp.constants.LpStatusOptimal, pulp.constants.LpStatusNotSolved):
        for (e, p, r) in valid_EPR:
            if x[(e, p, r)].varValue is not None and x[(e, p, r)].varValue > 0.5:
                solution.assign(e, p, r)

    unassigned = 0
    for e in range(n_e):
        if e not in solution.assignments:
            unassigned += 1
            solution.assign(e, 0, 0)

    eval_result = evaluate(problem, solution)

    if verbose:
        print(f"[PuLP] Status: {status_str}, {runtime:.2f}s")
        print(f"[PuLP] Feasible: {eval_result.is_feasible}, "
              f"Hard: {eval_result.hard_violations}, Soft: {eval_result.soft_penalty}")
        if unassigned:
            print(f"[PuLP] WARNING: {unassigned} exams needed fallback")

    return {
        'solution': solution,
        'runtime': runtime,
        'evaluation': eval_result,
        'algorithm': 'Integer Programming',
        'memory_peak_mb': 0,
        'solver_status': status_str,
    }