"""
OPTIMIZATION
"""

from __future__ import annotations
from dataclasses import dataclass
from core.models import ProblemInstance, Solution


@dataclass
class EvalBreakdown:
    """Itemized constraint costs."""
    conflicts: int = 0
    room_occupancy: int = 0
    period_utilisation: int = 0
    period_related: int = 0
    room_related: int = 0
    two_in_a_row: int = 0
    two_in_a_day: int = 0
    period_spread: int = 0
    non_mixed_durations: int = 0
    front_load: int = 0
    period_penalty: int = 0
    room_penalty: int = 0

    @property
    def hard(self) -> int:
        return (self.conflicts + self.room_occupancy + self.period_utilisation +
                self.period_related + self.room_related)

    # Compatibility aliases for old EvaluationResult interface
    @property
    def hard_violations(self) -> int:
        return self.hard

    @property
    def soft(self) -> int:
        return (self.two_in_a_row + self.two_in_a_day + self.period_spread +
                self.non_mixed_durations + self.front_load +
                self.period_penalty + self.room_penalty)

    @property
    def soft_penalty(self) -> int:
        return self.soft

    @property
    def feasible(self) -> bool:
        return self.hard == 0

    @property
    def is_feasible(self) -> bool:
        return self.hard == 0

    @property
    def fitness(self) -> float:
        return self.hard * 100000 + self.soft

    def summary(self) -> str:
        return (
            f"Feasible={self.feasible}  Hard={self.hard}  Soft={self.soft}\n"
            f"  conflicts={self.conflicts} room_occ={self.room_occupancy} "
            f"per_util={self.period_utilisation} per_rel={self.period_related} "
            f"room_rel={self.room_related}\n"
            f"  2row={self.two_in_a_row} 2day={self.two_in_a_day} "
            f"spread={self.period_spread} mixed={self.non_mixed_durations} "
            f"front={self.front_load} per_pen={self.period_penalty} "
            f"room_pen={self.room_penalty}"
        )


class FastEvaluator:
    """
    Usage:
        fe = FastEvaluator(problem)
        cost = fe.full_eval(solution)     # full evaluation
        delta = fe.move_delta(solution, exam_id, new_period, new_room)  # O(k)
    """

    def __init__(self, problem: ProblemInstance):
        if problem.conflict_matrix is None:
            problem.build_derived_data()

        self.p = problem
        n_e = len(problem.exams)
        n_p = len(problem.periods)
        n_r = len(problem.rooms)
        self.n_e = n_e
        self.n_p = n_p
        self.n_r = n_r

        w = problem.weightings
        self.w_2row = w.two_in_a_row
        self.w_2day = w.two_in_a_day
        self.w_spread = w.period_spread
        self.w_mixed = w.non_mixed_durations
        fl = w.front_load
        self.fl_n_largest = fl[0]
        self.fl_n_last = fl[1]
        self.fl_penalty = fl[2]

        # --- Precompute flat arrays ---
        # exam_dur[e] = duration of exam e
        self.exam_dur = [0] * n_e
        # exam_enroll[e] = enrollment count
        self.exam_enroll = [0] * n_e
        for e in problem.exams:
            self.exam_dur[e.id] = e.duration
            self.exam_enroll[e.id] = e.enrollment

        # period_dur[p], period_day[p], period_penalty[p]
        self.period_dur = [0] * n_p
        self.period_day = [0] * n_p
        self.period_pen = [0] * n_p
        for p in problem.periods:
            self.period_dur[p.id] = p.duration
            self.period_day[p.id] = p.day
            self.period_pen[p.id] = p.penalty

        # room_cap[r], room_penalty[r]
        self.room_cap = [0] * n_r
        self.room_pen = [0] * n_r
        for r in problem.rooms:
            self.room_cap[r.id] = r.capacity
            self.room_pen[r.id] = r.penalty

        # period_daypos[p] = position within its day (0, 1, 2, ...)
        self.period_daypos = [0] * n_p
        for day, pids in problem.periods_per_day.items():
            for pos, pid in enumerate(pids):
                self.period_daypos[pid] = pos

        # Adjacency list: adj[e] = list of (neighbor_exam, shared_students)
        self.adj: list[list[tuple[int, int]]] = [[] for _ in range(n_e)]
        for (e1, e2), shared in problem.conflict_matrix.items():
            self.adj[e1].append((e2, shared))
            self.adj[e2].append((e1, shared))

        # Exam -> student list (as tuple for speed)
        self.exam_students: list[tuple[int, ...]] = [() for _ in range(n_e)]
        for e in problem.exams:
            self.exam_students[e.id] = tuple(sorted(e.students))

        # Student -> exam list
        max_sid = 0
        for e in problem.exams:
            for s in e.students:
                if s > max_sid:
                    max_sid = s
        self.student_exams: list[list[int]] = [[] for _ in range(max_sid + 1)]
        for e in problem.exams:
            for s in e.students:
                self.student_exams[s].append(e.id)

        # Front load: set of large exam IDs
        sorted_exams = sorted(problem.exams, key=lambda e: e.enrollment, reverse=True)
        self.large_exams = set(e.id for e in sorted_exams[:self.fl_n_largest])
        all_pids = sorted(range(n_p))
        self.last_periods = set(all_pids[-self.fl_n_last:]) if self.fl_n_last > 0 else set()

        # Period hard constraints
        self.phcs = [(c.exam1_id, c.constraint_type, c.exam2_id)
                     for c in problem.period_hard_constraints]
        # Period hard constraints indexed by exam for O(k) delta lookup
        # phc_by_exam[eid] = [(other_eid, type_code)]
        #   type_code: 0=COINCIDENCE, 1=EXCLUSION, 2=AFTER(eid must come after other)
        self.phc_by_exam: list[list[tuple[int, int]]] = [[] for _ in range(n_e)]
        for c in problem.period_hard_constraints:
            e1, e2 = c.exam1_id, c.exam2_id
            if e1 >= n_e or e2 >= n_e:
                continue
            if c.constraint_type == "EXAM_COINCIDENCE":
                self.phc_by_exam[e1].append((e2, 0))
                self.phc_by_exam[e2].append((e1, 0))
            elif c.constraint_type == "EXCLUSION":
                self.phc_by_exam[e1].append((e2, 1))
                self.phc_by_exam[e2].append((e1, 1))
            elif c.constraint_type == "AFTER":
                self.phc_by_exam[e1].append((e2, 2))  # e1 must come after e2
        # Room hard constraints
        self.rhc_exams = set(c.exam_id for c in problem.room_hard_constraints
                             if c.constraint_type == "ROOM_EXCLUSIVE")

    def full_eval(self, sol: Solution) -> EvalBreakdown:
        """Full evaluation — O(n * avg_neighbors). Use for initial/final scoring."""
        r = EvalBreakdown()
        period_of = sol._period_of
        room_of = sol._room_of

        # Build period->exams and (period,room)->enrollment
        period_exams: list[list[int]] = [[] for _ in range(self.n_p)]
        pr_enroll: dict[tuple[int, int], int] = {}
        pr_exams: dict[tuple[int, int], list[int]] = {}

        for eid in range(self.n_e):
            pid = period_of[eid]
            if pid < 0:
                continue
            rid = room_of[eid]
            period_exams[pid].append(eid)
            key = (pid, rid)
            pr_enroll[key] = pr_enroll.get(key, 0) + self.exam_enroll[eid]
            if key not in pr_exams:
                pr_exams[key] = []
            pr_exams[key].append(eid)

        # Student period lists
        student_periods: dict[int, list[int]] = {}
        for eid in range(self.n_e):
            pid = period_of[eid]
            if pid < 0:
                continue
            for s in self.exam_students[eid]:
                if s not in student_periods:
                    student_periods[s] = []
                student_periods[s].append(pid)

        # === HARD ===
        # Conflicts
        for s, pids in student_periods.items():
            seen: dict[int, int] = {}
            for pid in pids:
                seen[pid] = seen.get(pid, 0) + 1
            for pid, cnt in seen.items():
                if cnt > 1:
                    r.conflicts += cnt - 1

        # Room occupancy
        for (pid, rid), enr in pr_enroll.items():
            if enr > self.room_cap[rid]:
                r.room_occupancy += 1

        # Period utilisation
        for eid in range(self.n_e):
            pid = period_of[eid]
            if pid >= 0 and self.exam_dur[eid] > self.period_dur[pid]:
                r.period_utilisation += 1

        # Period hard constraints
        for e1, ctype, e2 in self.phcs:
            p1 = period_of[e1]
            p2 = period_of[e2]
            if p1 < 0 or p2 < 0:
                continue
            if ctype == "EXAM_COINCIDENCE" and p1 != p2:
                r.period_related += 1
            elif ctype == "EXCLUSION" and p1 == p2:
                r.period_related += 1
            elif ctype == "AFTER" and p1 <= p2:
                r.period_related += 1

        # Room exclusive
        for eid in self.rhc_exams:
            pid = period_of[eid]
            if pid >= 0:
                rid = room_of[eid]
                if len(pr_exams.get((pid, rid), [])) > 1:
                    r.room_related += 1

        # === SOFT ===
        # Proximity: 2-in-a-row, 2-in-a-day, spread
        for s, pids in student_periods.items():
            unique = sorted(set(pids))
            for i in range(len(unique)):
                pi = unique[i]
                di = self.period_day[pi]
                posi = self.period_daypos[pi]
                for j in range(i + 1, len(unique)):
                    pj = unique[j]
                    dj = self.period_day[pj]
                    if di == dj:
                        posj = self.period_daypos[pj]
                        gap_pos = abs(posi - posj)
                        if gap_pos == 1:
                            r.two_in_a_row += self.w_2row
                        elif gap_pos > 1:
                            r.two_in_a_day += self.w_2day
                    gap = abs(pj - pi)
                    if 0 < gap <= self.w_spread:
                        r.period_spread += 1

        # Mixed durations
        for (pid, rid), eids in pr_exams.items():
            if len(eids) > 1:
                durs = set()
                for eid in eids:
                    durs.add(self.exam_dur[eid])
                if len(durs) > 1:
                    r.non_mixed_durations += self.w_mixed

        # Front load
        if self.fl_penalty > 0:
            for eid in self.large_exams:
                pid = period_of[eid]
                if pid >= 0 and pid in self.last_periods:
                    r.front_load += self.fl_penalty

        # Period/room penalties
        for eid in range(self.n_e):
            pid = period_of[eid]
            if pid >= 0:
                r.period_penalty += self.period_pen[pid]
                r.room_penalty += self.room_pen[room_of[eid]]

        return r

    def move_delta(
        self,
        sol: Solution,
        exam_id: int,
        new_pid: int,
        new_rid: int,
    ) -> float:
        """
        Compute the CHANGE in fitness if we move exam_id to (new_pid, new_rid).
        Returns delta fitness (negative = improvement).

        Iterates over students enrolled in exam_id (not all students).
        For each student, computes the exact hard-conflict and soft-proximity
        delta by checking their other exams' assigned periods.

        Cost: O(enrollment(exam) * avg_exams_per_student).
        Does NOT modify the solution.
        """
        period_of = sol._period_of
        old_pid = period_of[exam_id]
        if old_pid < 0:
            return 0.0
        old_rid = sol._room_of[exam_id]
        if old_pid == new_pid and old_rid == new_rid:
            return 0.0

        delta_hard = 0
        delta_soft = 0

        # --- Hard: period duration ---
        dur = self.exam_dur[exam_id]
        if dur > self.period_dur[old_pid]:
            delta_hard -= 1
        if dur > self.period_dur[new_pid]:
            delta_hard += 1

        # --- Hard: room occupancy (O(1) via tracked enrollment) ---
        enroll = self.exam_enroll[exam_id]
        old_cap = self.room_cap[old_rid]
        new_cap = self.room_cap[new_rid]
        # Current enrollment in old slot (includes our exam)
        old_total = sol.get_pr_enroll(old_pid, old_rid)
        # Current enrollment in new slot (excludes our exam)
        new_total = sol.get_pr_enroll(new_pid, new_rid)
        # Was old slot over capacity? Will it be after we leave?
        was_old_over = 1 if old_total > old_cap else 0
        will_old_over = 1 if (old_total - enroll) > old_cap else 0
        delta_hard -= (was_old_over - will_old_over)
        # Was new slot over capacity? Will it be after we arrive?
        was_new_over = 1 if new_total > new_cap else 0
        will_new_over = 1 if (new_total + enroll) > new_cap else 0
        delta_hard += (will_new_over - was_new_over)

        # --- Per-student exact delta for conflicts + proximity ---
        old_day = self.period_day[old_pid]
        old_daypos = self.period_daypos[old_pid]
        new_day = self.period_day[new_pid]
        new_daypos = self.period_daypos[new_pid]

        w_2row = self.w_2row
        w_2day = self.w_2day
        w_spread = self.w_spread
        pday = self.period_day
        pdaypos = self.period_daypos

        for s in self.exam_students[exam_id]:
            for other_eid in self.student_exams[s]:
                if other_eid == exam_id:
                    continue
                o_pid = period_of[other_eid]
                if o_pid < 0:
                    continue

                # --- Hard: conflict (same period) ---
                if o_pid == old_pid:
                    delta_hard -= 1  # removing a conflict for this student
                if o_pid == new_pid:
                    delta_hard += 1  # creating a conflict

                # --- Soft: proximity ---
                o_day = pday[o_pid]
                o_daypos = pdaypos[o_pid]

                # Remove old proximity
                if old_day == o_day:
                    gap_pos = old_daypos - o_daypos
                    if gap_pos < 0:
                        gap_pos = -gap_pos
                    if gap_pos == 1:
                        delta_soft -= w_2row
                    elif gap_pos > 1:
                        delta_soft -= w_2day
                old_gap = old_pid - o_pid
                if old_gap < 0:
                    old_gap = -old_gap
                if 0 < old_gap <= w_spread:
                    delta_soft -= 1

                # Add new proximity
                if new_day == o_day:
                    gap_pos = new_daypos - o_daypos
                    if gap_pos < 0:
                        gap_pos = -gap_pos
                    if gap_pos == 1:
                        delta_soft += w_2row
                    elif gap_pos > 1:
                        delta_soft += w_2day
                new_gap = new_pid - o_pid
                if new_gap < 0:
                    new_gap = -new_gap
                if 0 < new_gap <= w_spread:
                    delta_soft += 1

        # --- Soft: period/room penalties ---
        delta_soft += self.period_pen[new_pid] - self.period_pen[old_pid]
        delta_soft += self.room_pen[new_rid] - self.room_pen[old_rid]

        # --- Soft: front load ---
        if exam_id in self.large_exams and self.fl_penalty > 0:
            was_late = old_pid in self.last_periods
            will_late = new_pid in self.last_periods
            if was_late and not will_late:
                delta_soft -= self.fl_penalty
            elif not was_late and will_late:
                delta_soft += self.fl_penalty

        # --- Hard: period hard constraints (COINCIDENCE/EXCLUSION/AFTER) ---
        for other, tcode in self.phc_by_exam[exam_id]:
            o_pid = period_of[other]
            if o_pid < 0:
                continue
            if tcode == 0:        # COINCIDENCE: must share period
                if old_pid != o_pid:
                    delta_hard -= 1  # was violated
                if new_pid != o_pid:
                    delta_hard += 1  # will be violated
            elif tcode == 1:      # EXCLUSION: must NOT share period
                if old_pid == o_pid:
                    delta_hard -= 1  # was violated
                if new_pid == o_pid:
                    delta_hard += 1  # will be violated
            elif tcode == 2:      # AFTER: exam_id must come after other
                if old_pid <= o_pid:
                    delta_hard -= 1  # was violated
                if new_pid <= o_pid:
                    delta_hard += 1  # will be violated

        return delta_hard * 100000 + delta_soft

    def apply_move(self, sol: Solution, exam_id: int, new_pid: int, new_rid: int):
        """Apply a move to the solution (in-place)."""
        sol.assign(exam_id, new_pid, new_rid)