"""
Algorithm:
  1. Sort exams by conflict degree (descending) — largest degree first.
  2. For each exam, assign it to the first feasible (period, room) pair.
  3. Feasibility: no student conflict in that period, room has capacity,
     exam duration fits period duration.

Time Complexity: Θ(n^2)
"""

from data.models import ProblemInstance, Solution
from data.evaluator import evaluate
import time


def solve_greedy(problem: ProblemInstance, verbose: bool = False) -> dict:
    """
    Solve the exam timetabling problem using greedy graph coloring.

    Returns:
        dict with keys: 'solution', 'runtime', 'evaluation', 'algorithm'
    """
    start_time = time.time()

    if problem.conflict_matrix is None:
        problem.build_derived_data()

    solution = Solution(problem)
    exam_by_id = {e.id: e for e in problem.exams}
    period_by_id = {p.id: p for p in problem.periods}
    room_by_id = {r.id: r for r in problem.rooms}

    # --- Step 1: Sort exams by conflict degree (descending) ---
    # Higher degree = more conflicts = harder to place → schedule first
    exam_degrees = []
    for exam in problem.exams:
        degree = problem.get_exam_degree(exam.id)
        exam_degrees.append((exam.id, degree, exam.enrollment))

    # Sort: primary by degree desc, secondary by enrollment desc
    exam_degrees.sort(key=lambda x: (x[1], x[2]), reverse=True)
    ordering = [eid for eid, _, _ in exam_degrees]

    if verbose:
        print(f"[Greedy] Sorted {len(ordering)} exams by conflict degree")
        print(f"[Greedy] Max degree: {exam_degrees[0][1]}, "
              f"Min degree: {exam_degrees[-1][1]}")

    # --- Step 2: Track which periods are used by each exam's neighbors ---
    # Build adjacency list for fast neighbor lookup
    adj: dict[int, set[int]] = {e.id: set() for e in problem.exams}
    for (e1, e2) in problem.conflict_matrix:
        adj[e1].add(e2)
        adj[e2].add(e1)

    # Track: period -> total enrollment per room
    period_room_usage: dict[tuple[int, int], int] = {}
    # Track: period -> set of exam IDs (for constraint checking)
    period_exams: dict[int, set[int]] = {p.id: set() for p in problem.periods}

    unassigned = []

    # --- Step 3: Assign each exam to first feasible slot ---
    for eid in ordering:
        exam = exam_by_id[eid]

        # Find periods used by conflicting exams
        blocked_periods = set()
        for neighbor in adj[eid]:
            p = solution.get_period(neighbor)
            if p is not None:
                blocked_periods.add(p)

        assigned = False
        for period in problem.periods:
            pid = period.id
            if pid in blocked_periods:
                continue

            # Check period duration
            if exam.duration > period.duration:
                continue

            # Try each room in this period
            for room in problem.rooms:
                rid = room.id
                current_usage = period_room_usage.get((pid, rid), 0)
                if current_usage + exam.enrollment <= room.capacity:
                    # Feasible assignment found
                    solution.assign(eid, pid, rid)
                    period_room_usage[(pid, rid)] = current_usage + exam.enrollment
                    period_exams[pid].add(eid)
                    assigned = True
                    break

            if assigned:
                break

        if not assigned:
            # Fallback: assign to first period/room (will violate constraints)
            # This ensures completeness even if infeasible
            solution.assign(eid, problem.periods[0].id, problem.rooms[0].id)
            key = (problem.periods[0].id, problem.rooms[0].id)
            period_room_usage[key] = period_room_usage.get(key, 0) + exam.enrollment
            period_exams[problem.periods[0].id].add(eid)
            unassigned.append(eid)
            if verbose:
                print(f"[Greedy] WARNING: Exam {eid} could not be feasibly placed")

    runtime = time.time() - start_time
    eval_result = evaluate(problem, solution)

    if verbose:
        print(f"[Greedy] Completed in {runtime:.4f}s")
        print(f"[Greedy] Feasible: {eval_result.is_feasible}")
        print(f"[Greedy] Hard violations: {eval_result.hard_violations}")
        print(f"[Greedy] Soft penalty: {eval_result.soft_penalty}")
        if unassigned:
            print(f"[Greedy] {len(unassigned)} exams placed infeasibly as fallback")

    return {
        'solution': solution,
        'runtime': runtime,
        'evaluation': eval_result,
        'algorithm': 'Greedy Heuristic',
    }
