from data.models import ProblemInstance, Solution
from data.evaluator import evaluate
import time
import tracemalloc
import collections

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


def solve_ip(
    problem: ProblemInstance,
    time_limit: int = 300,
    verbose: bool = False,
    mip_gap: float = 0.01,
) -> dict:
    
    if not HAS_PULP:
        raise ImportError("PuLP is required. Install with: pip install pulp")

    start_time = time.time()
    tracemalloc.start()

    if problem.conflict_matrix is None:
        problem.build_derived_data()

    n_exams = problem.num_exams()
    n_periods = problem.num_periods()
    n_rooms = problem.num_rooms()
    w = problem.weightings

    E = [e.id for e in problem.exams]
    P = [p.id for p in problem.periods]
    R = [r.id for r in problem.rooms]

    # Pre-fetch properties for fast lookup
    exam_enrollments = {e.id: e.enrollment for e in problem.exams}
    exam_durations = {e.id: e.duration for e in problem.exams}
    room_capacities = {r.id: r.capacity for r in problem.rooms}
    period_durations = {p.id: p.duration for p in problem.periods}
    period_penalties = {p.id: p.penalty for p in problem.periods}
    room_penalties = {r.id: r.penalty for r in problem.rooms}

    # ==================== VARIABLE PRUNING ====================
    valid_EPR = []
    valid_EP = set()
    
    for e in E:
        for p in P:
            if exam_durations[e] <= period_durations[p]:
                valid_EP.add((e, p))
                for r in R:
                    if exam_enrollments[e] <= room_capacities[r]:
                        valid_EPR.append((e, p, r))

    valid_epr_by_e = collections.defaultdict(list)
    valid_e_by_pr = collections.defaultdict(list)
    valid_p_by_e = collections.defaultdict(list)
    
    for (e, p, r) in valid_EPR:
        valid_epr_by_e[e].append((e, p, r))
        valid_e_by_pr[(p, r)].append(e)

    for (e, p) in valid_EP:
        valid_p_by_e[e].append(p)

    if verbose:
        print(f"[IP] Problem size: {n_exams} exams, {n_periods} periods, {n_rooms} rooms")
        print(f"[IP] Valid binary variables pruned to: {len(valid_EPR)}")

    # ==================== MODEL ====================
    model = pulp.LpProblem("ExamTimetabling", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", valid_EPR, cat='Binary')
    y = pulp.LpVariable.dicts("y", list(valid_EP), cat='Binary')

    # Link x and y
    for (e, p) in valid_EP:
        valid_rs = [r for r in R if (e, p, r) in x]
        model += y[(e, p)] == pulp.lpSum([x[(e, p, r)] for r in valid_rs]), f"link_{e}_{p}"

    # --- Hard Constraints ---
    # 1. Exactly one assignment per exam
    for e in E:
        model += pulp.lpSum([x[epr] for epr in valid_epr_by_e[e]]) == 1, f"assign_{e}"

    # 2. No student conflict (Pairwise & Clique Tightening)
    active_conflicts = [(e1, e2) for (e1, e2), shared in problem.conflict_matrix.items() if shared > 0]
    
    # Standard Pairwise
    for (e1, e2) in active_conflicts:
        common_ps = set(valid_p_by_e[e1]).intersection(valid_p_by_e[e2])
        for p in common_ps:
            model += y[(e1, p)] + y[(e2, p)] <= 1, f"conflict_{e1}_{e2}_p{p}"

    # Advanced Clique Tightening (Valid Inequalities via NetworkX)
    if HAS_NX:
        G = nx.Graph()
        G.add_nodes_from(E)
        G.add_edges_from(active_conflicts)
        
        cliques = [c for c in nx.find_cliques(G) if len(c) >= 3]
        if verbose and cliques:
            print(f"[IP] Applied {len(cliques)} constraint cliques to tighten mathematical bounds.")
            
        for i, clique in enumerate(cliques):
            for p in P:
                valid_clique_exams = [e for e in clique if (e, p) in y]
                if len(valid_clique_exams) >= 2:
                    model += pulp.lpSum([y[(e, p)] for e in valid_clique_exams]) <= 1, f"clique_{i}_p{p}"

    # 3. Room capacity
    for p in P:
        for r in R:
            exams_in_pr = valid_e_by_pr[(p, r)]
            if exams_in_pr:
                model += pulp.lpSum([exam_enrollments[e] * x[(e, p, r)] for e in exams_in_pr]) <= room_capacities[r], f"roomcap_{p}_{r}"

    # 4. Period Hard Constraints
    for phc in problem.period_hard_constraints:
        e1, e2 = phc.exam1_id, phc.exam2_id
        common_ps = set(valid_p_by_e[e1]).intersection(valid_p_by_e[e2])

        if phc.constraint_type == "EXAM_COINCIDENCE":
            for p in P:
                y1 = y[(e1, p)] if (e1, p) in y else 0
                y2 = y[(e2, p)] if (e2, p) in y else 0
                if (e1, p) in y or (e2, p) in y:
                    model += y1 == y2, f"coincidence_{e1}_{e2}_{p}"
        elif phc.constraint_type == "EXCLUSION":
            for p in common_ps:
                model += y[(e1, p)] + y[(e2, p)] <= 1, f"exclusion_{e1}_{e2}_{p}"
        elif phc.constraint_type == "AFTER":
            model += (
                pulp.lpSum([p * y[(e1, p)] for p in valid_p_by_e[e1]]) >=
                pulp.lpSum([p * y[(e2, p)] for p in valid_p_by_e[e2]]) + 1,
                f"after_{e1}_{e2}"
            )

    # --- Symmetry Breaking (Identical Rooms) ---
    room_profiles = collections.defaultdict(list)
    for r in problem.rooms:
        room_profiles[(r.capacity, r.penalty)].append(r.id)

    for profile, identical_rooms in room_profiles.items():
        if len(identical_rooms) > 1:
            for i in range(len(identical_rooms) - 1):
                r1, r2 = identical_rooms[i], identical_rooms[i+1]
                for p in P:
                    e_r1 = [e for e in valid_e_by_pr[(p, r1)]]
                    e_r2 = [e for e in valid_e_by_pr[(p, r2)]]
                    if e_r1 and e_r2:
                        model += pulp.lpSum([x[(e, p, r1)] for e in e_r1]) >= \
                                 pulp.lpSum([x[(e, p, r2)] for e in e_r2]), f"sym_break_p{p}_{r1}_{r2}"

    # --- Soft Constraints (Objective) ---
    penalty_terms = []

    for (e, p) in valid_EP:
        if period_penalties[p] > 0:
            penalty_terms.append(period_penalties[p] * y[(e, p)])
    for (e, p, r) in valid_EPR:
        if room_penalties[r] > 0:
            penalty_terms.append(room_penalties[r] * x[(e, p, r)])

    period_to_day = {p.id: p.day for p in problem.periods}
    day_periods = problem.periods_per_day
    spread_limit = w.period_spread
    
    precalc_proximity_penalties = {}
    for i, p1 in enumerate(P):
        for p2 in P[i+1:]:
            gap = abs(p2 - p1)
            d1, d2 = period_to_day.get(p1), period_to_day.get(p2)
            base_pen = 0

            if d1 == d2:
                day_pids = day_periods.get(d1, [])
                if p1 in day_pids and p2 in day_pids:
                    pos1, pos2 = day_pids.index(p1), day_pids.index(p2)
                    if abs(pos1 - pos2) == 1:
                        base_pen += w.two_in_a_row
                    elif abs(pos1 - pos2) > 1:
                        base_pen += w.two_in_a_day

            if 0 < gap <= spread_limit:
                base_pen += 1 

            if base_pen > 0:
                precalc_proximity_penalties[(p1, p2)] = base_pen

    for (e1, e2) in active_conflicts:
        shared = problem.conflict_matrix[(e1, e2)]
        for (p1, p2), base_pen in precalc_proximity_penalties.items():
            if (e1, p1) in y and (e2, p2) in y:
                penalty_val = base_pen * shared
                penalty_terms.append(penalty_val * (y[(e1, p1)] + y[(e2, p2)] - 1))

    if penalty_terms:
        model += pulp.lpSum(penalty_terms), "soft_penalty"
    else:
        model += 0, "soft_penalty"

    # ==================== SOLVE ====================
    if verbose:
        print(f"[IP] Model built in {time.time() - start_time:.2f}s")
        print(f"[IP] Attempting to solve with HiGHS (time limit: {time_limit}s)...")

    try:
        # NOTE: If you manually downloaded highs.exe, add path=r"C:\your\path\highs.exe" inside this command
        solver = pulp.HiGHS_CMD(timeLimit=time_limit, gapRel=mip_gap, msg=verbose)
        model.solve(solver)
    except (pulp.apis.core.PulpSolverError, Exception) as e:
        if verbose:
            print("[IP] WARNING: 'highs.exe' not found on system PATH. Falling back to built-in CBC solver.")
            print(f"[IP] Solving with CBC solver (time limit: {time_limit}s)...")
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, gapRel=mip_gap, msg=1 if verbose else 0)
        model.solve(solver)

    runtime = time.time() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_memory / (1024 * 1024)

    # ==================== EXTRACT SOLUTION ====================
    solution = Solution(problem)
    solver_status = pulp.LpStatus[model.status]

    if model.status in [pulp.constants.LpStatusOptimal, pulp.constants.LpStatusNotSolved]:
        for (e, p, r) in valid_EPR:
            if x[(e, p, r)].varValue is not None and x[(e, p, r)].varValue > 0.5:
                solution.assign(e, p, r)

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
            print(f"[IP] WARNING: {unassigned_count} exams needed greedy fallback.")

    return {
        'solution': solution,
        'runtime': runtime,
        'evaluation': eval_result,
        'algorithm': 'Integer Programming',
        'memory_peak_mb': peak_mb,
        'solver_status': solver_status,
    }