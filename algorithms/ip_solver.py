"""
Formulates the problem as a Mixed Integer Linear Program (MILP) and
solves using PuLP with the CBC solver (free, open-source).

Decision variables:
  x[e,p,r] = 1 if exam e is assigned to period p in room r

Hard constraints modeled as linear equalities/inequalities.
Soft constraints modeled in the objective with weighted penalties.

Time Complexity: Θ(2^n)

For large instances, a time limit is enforced.

Reference: Integer programming - Wikipedia (ref 5 in project plan)
           PuLP library documentation
"""

from data.models import ProblemInstance, Solution
from data.evaluator import evaluate
import time
import tracemalloc

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


def solve_ip(
    problem: ProblemInstance,
    time_limit: int = 300,
    verbose: bool = False,
    mip_gap: float = 0.05,
) -> dict:
    """
    Solve the exam timetabling problem using Integer Programming.

    Args:
        problem: The problem instance.
        time_limit: Maximum solver time in seconds (default 300s = 5 min).
        verbose: Print progress information.
        mip_gap: Acceptable MIP gap (0.05 = 5% of optimal).

    Returns:
        dict with keys: 'solution', 'runtime', 'evaluation', 'algorithm',
                        'memory_peak_mb', 'solver_status'
    """
    if not HAS_PULP:
        raise ImportError(
            "PuLP is required for the IP solver. Install with: pip install pulp"
        )

    start_time = time.time()
    tracemalloc.start()

    if problem.conflict_matrix is None:
        problem.build_derived_data()

    n_exams = problem.num_exams()
    n_periods = problem.num_periods()
    n_rooms = problem.num_rooms()
    w = problem.weightings

    if verbose:
        print(f"[IP] Problem size: {n_exams} exams, {n_periods} periods, {n_rooms} rooms")
        print(f"[IP] Variables: ~{n_exams * n_periods * n_rooms} binary")
        print(f"[IP] Time limit: {time_limit}s, MIP gap: {mip_gap}")

    # Check if problem is too large for IP
    total_vars = n_exams * n_periods * n_rooms
    if total_vars > 500_000:
        if verbose:
            print(f"[IP] WARNING: {total_vars} variables is very large. "
                  f"Consider reducing instance size or increasing time limit.")

    exam_by_id = {e.id: e for e in problem.exams}
    period_by_id = {p.id: p for p in problem.periods}
    room_by_id = {r.id: r for r in problem.rooms}

    # ==================== MODEL ====================
    model = pulp.LpProblem("ExamTimetabling", pulp.LpMinimize)

    # --- Decision variables ---
    # x[e][p][r] = 1 if exam e assigned to period p, room r
    E = [e.id for e in problem.exams]
    P = [p.id for p in problem.periods]
    R = [r.id for r in problem.rooms]

    x = pulp.LpVariable.dicts("x", (E, P, R), cat='Binary')

    # Helper: y[e][p] = 1 if exam e is in period p (sum over rooms)
    y = pulp.LpVariable.dicts("y", (E, P), cat='Binary')

    # Link x and y
    for e in E:
        for p in P:
            model += y[e][p] == pulp.lpSum(x[e][p][r] for r in R), f"link_{e}_{p}"

    # --- Hard Constraints ---

    # 1. Each exam assigned to exactly one (period, room)
    for e in E:
        model += (
            pulp.lpSum(x[e][p][r] for p in P for r in R) == 1,
            f"assign_{e}"
        )

    # 2. No student conflict: conflicting exams not in same period
    for (e1, e2), shared in problem.conflict_matrix.items():
        if shared > 0:
            for p in P:
                model += (
                    y[e1][p] + y[e2][p] <= 1,
                    f"conflict_{e1}_{e2}_p{p}"
                )

    # 3. Room capacity: total enrollment per (period, room) <= capacity
    for p in P:
        for r in R:
            model += (
                pulp.lpSum(
                    exam_by_id[e].enrollment * x[e][p][r] for e in E
                ) <= room_by_id[r].capacity,
                f"roomcap_{p}_{r}"
            )

    # 4. Period duration: exam duration <= period duration
    for e in E:
        for p in P:
            if exam_by_id[e].duration > period_by_id[p].duration:
                model += y[e][p] == 0, f"perdur_{e}_{p}"

    # 5. Period hard constraints
    for phc in problem.period_hard_constraints:
        e1, e2 = phc.exam1_id, phc.exam2_id
        if phc.constraint_type == "EXAM_COINCIDENCE":
            for p in P:
                model += y[e1][p] == y[e2][p], f"coincidence_{e1}_{e2}_{p}"
        elif phc.constraint_type == "EXCLUSION":
            for p in P:
                model += y[e1][p] + y[e2][p] <= 1, f"exclusion_{e1}_{e2}_{p}"
        elif phc.constraint_type == "AFTER":
            # exam1 must be after exam2: period(e1) > period(e2)
            # Linearize: sum(p * y[e1][p]) > sum(p * y[e2][p])
            model += (
                pulp.lpSum(p * y[e1][p] for p in P) >=
                pulp.lpSum(p * y[e2][p] for p in P) + 1,
                f"after_{e1}_{e2}"
            )

    # --- Soft Constraints (Objective) ---
    # We approximate the key soft constraints in the objective

    # Auxiliary variables for proximity penalties
    period_to_day = {p.id: p.day for p in problem.periods}
    day_periods = problem.periods_per_day

    # Simplified proximity penalty: for each conflicting pair,
    # penalize if they are in close periods
    penalty_terms = []

    # Period and room penalties (straightforward)
    for e in E:
        for p in P:
            pp = period_by_id[p].penalty
            if pp > 0:
                penalty_terms.append(pp * y[e][p])
            for r in R:
                rp = room_by_id[r].penalty
                if rp > 0:
                    penalty_terms.append(rp * x[e][p][r])

    # Proximity penalties (approximate): for conflicting exam pairs,
    # add penalty if both are in nearby periods
    spread_limit = w.period_spread
    for (e1, e2), shared in problem.conflict_matrix.items():
        if shared <= 0:
            continue
        for i, p1 in enumerate(P):
            for p2 in P[i+1:]:
                gap = abs(p2 - p1)
                d1 = period_to_day.get(p1)
                d2 = period_to_day.get(p2)

                penalty = 0
                if d1 == d2:
                    # Same day
                    day_pids = day_periods.get(d1, [])
                    if p1 in day_pids and p2 in day_pids:
                        pos1 = day_pids.index(p1)
                        pos2 = day_pids.index(p2)
                        if abs(pos1 - pos2) == 1:
                            penalty += w.two_in_a_row * shared
                        elif abs(pos1 - pos2) > 1:
                            penalty += w.two_in_a_day * shared

                if 0 < gap <= spread_limit:
                    penalty += shared  # period spread

                if penalty > 0:
                    # Linearize: penalty applies if both y[e1][p1]=1 and y[e2][p2]=1
                    # We use a common technique: z = y1 * y2 (though adding as
                    # penalty * (y1 + y2 - 1) is a valid lower bound relaxation)
                    penalty_terms.append(penalty * (y[e1][p1] + y[e2][p2] - 1))

    if penalty_terms:
        model += pulp.lpSum(penalty_terms), "soft_penalty"
    else:
        model += 0, "soft_penalty"

    # ==================== SOLVE ====================
    if verbose:
        print(f"[IP] Model built in {time.time() - start_time:.2f}s")
        print(f"[IP] Solving with CBC solver (time limit: {time_limit}s)...")

    solver = pulp.PULP_CBC_CMD(
        timeLimit=time_limit,
        gapRel=mip_gap,
        msg=1 if verbose else 0,
    )

    model.solve(solver)

    runtime = time.time() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_memory / (1024 * 1024)

    # ==================== EXTRACT SOLUTION ====================
    solution = Solution(problem)
    solver_status = pulp.LpStatus[model.status]

    if model.status in [pulp.constants.LpStatusOptimal, pulp.constants.LpStatusNotSolved]:
        # LpStatusNotSolved can mean time limit reached with a feasible solution
        for e in E:
            for p in P:
                for r in R:
                    if x[e][p][r].varValue is not None and x[e][p][r].varValue > 0.5:
                        solution.assign(e, p, r)
                        break

    # Fill any unassigned exams with greedy fallback
    unassigned_count = 0
    for e in E:
        if e not in solution.assignments:
            unassigned_count += 1
            solution.assign(e, P[0], R[0])

    eval_result = evaluate(problem, solution)

    if verbose:
        print(f"[IP] Solver status: {solver_status}")
        print(f"[IP] Completed in {runtime:.2f}s")
        print(f"[IP] Peak memory: {peak_mb:.1f} MB")
        print(f"[IP] Feasible: {eval_result.is_feasible}")
        print(f"[IP] Hard violations: {eval_result.hard_violations}")
        print(f"[IP] Soft penalty: {eval_result.soft_penalty}")
        if unassigned_count > 0:
            print(f"[IP] {unassigned_count} exams needed greedy fallback")

    return {
        'solution': solution,
        'runtime': runtime,
        'evaluation': eval_result,
        'algorithm': 'Integer Programming (PuLP/CBC)',
        'memory_peak_mb': peak_mb,
        'solver_status': solver_status,
    }
