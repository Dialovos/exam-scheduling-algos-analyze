"""
Core data models for the Capacitated Examination Timetabling Problem.
Reference: McCollum et al. (2007) - "The Second International Timetabling Competition"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import copy


@dataclass
class Exam:
    """An examination to be scheduled."""
    id: int
    duration: int  # in minutes
    students: set[int] = field(default_factory=set)

    @property
    def enrollment(self) -> int:
        return len(self.students)


@dataclass
class Period:
    """A time period in the examination session."""
    id: int
    date: str  # e.g., "01:01:2007"
    time: str  # e.g., "09:00:00"
    duration: int  # in minutes
    penalty: int = 0  # soft penalty for using this period
    day: int = 0  # which day this period belongs to (derived)


@dataclass
class Room:
    """An examination room."""
    id: int
    capacity: int
    penalty: int = 0  # soft penalty for using this room


@dataclass
class PeriodHardConstraint:
    """Hard constraint relating two exams via periods."""
    exam1_id: int
    constraint_type: str  # "EXAM_COINCIDENCE", "EXCLUSION", "AFTER"
    exam2_id: int


@dataclass
class RoomHardConstraint:
    """Hard constraint relating an exam to room usage."""
    exam_id: int
    constraint_type: str  # "ROOM_EXCLUSIVE"


@dataclass
class InstitutionalWeightings:
    """Soft constraint weights (Institutional Model Index)."""
    two_in_a_row: int = 1
    two_in_a_day: int = 1
    period_spread: int = 1  # number of periods for spread calculation
    non_mixed_durations: int = 1
    front_load: tuple[int, int, int] = (0, 0, 0)  # (num_largest, num_last_periods, penalty)


@dataclass
class ProblemInstance:
    """Complete problem instance for exam timetabling."""
    exams: list[Exam] = field(default_factory=list)
    periods: list[Period] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    period_hard_constraints: list[PeriodHardConstraint] = field(default_factory=list)
    room_hard_constraints: list[RoomHardConstraint] = field(default_factory=list)
    weightings: InstitutionalWeightings = field(default_factory=InstitutionalWeightings)

    # Derived data (computed after loading)
    conflict_matrix: Optional[dict] = None  # {(e1, e2): num_shared_students}
    student_exams: Optional[dict] = None  # {student_id: set of exam_ids}
    periods_per_day: Optional[dict] = None  # {day: [period_ids]}
    exam_degree: Optional[list] = None  # precomputed conflict degrees

    def build_derived_data(self):
        """Compute conflict matrix and other derived structures."""
        # Build student -> exams mapping
        self.student_exams = {}
        for exam in self.exams:
            for s in exam.students:
                if s not in self.student_exams:
                    self.student_exams[s] = set()
                self.student_exams[s].add(exam.id)

        # Build conflict matrix (sparse: only conflicting pairs)
        self.conflict_matrix = {}
        for s, s_exams in self.student_exams.items():
            s_exams_list = sorted(s_exams)
            for i in range(len(s_exams_list)):
                for j in range(i + 1, len(s_exams_list)):
                    e1, e2 = s_exams_list[i], s_exams_list[j]
                    key = (e1, e2)
                    self.conflict_matrix[key] = self.conflict_matrix.get(key, 0) + 1

        # Precompute degree (O(1) lookup instead of O(|conflicts|) per exam)
        n = len(self.exams)
        self.exam_degree = [0] * n
        for (e1, e2) in self.conflict_matrix:
            self.exam_degree[e1] += 1
            self.exam_degree[e2] += 1

        # Group periods by day
        self.periods_per_day = {}
        for p in self.periods:
            if p.day not in self.periods_per_day:
                self.periods_per_day[p.day] = []
            self.periods_per_day[p.day].append(p.id)
        for day in self.periods_per_day:
            self.periods_per_day[day].sort()

    def num_exams(self) -> int:
        return len(self.exams)

    def num_periods(self) -> int:
        return len(self.periods)

    def num_rooms(self) -> int:
        return len(self.rooms)

    def num_students(self) -> int:
        if self.student_exams is None:
            self.build_derived_data()
        return len(self.student_exams)

    def conflict_density(self) -> float:
        """Fraction of exam pairs that share at least one student."""
        if self.conflict_matrix is None:
            self.build_derived_data()
        n = self.num_exams()
        max_pairs = n * (n - 1) / 2
        if max_pairs == 0:
            return 0.0
        return len(self.conflict_matrix) / max_pairs

    def get_conflict(self, e1: int, e2: int) -> int:
        """Return number of shared students between two exams."""
        if self.conflict_matrix is None:
            self.build_derived_data()
        key = (min(e1, e2), max(e1, e2))
        return self.conflict_matrix.get(key, 0)

    def get_exam_degree(self, exam_id: int) -> int:
        """Number of other exams conflicting with this exam."""
        if self.conflict_matrix is None:
            self.build_derived_data()
        return self.exam_degree[exam_id]

    def summary(self) -> str:
        if self.conflict_matrix is None:
            self.build_derived_data()
        return (
            f"Problem Instance Summary:\n"
            f"  Exams:    {self.num_exams()}\n"
            f"  Periods:  {self.num_periods()}\n"
            f"  Rooms:    {self.num_rooms()}\n"
            f"  Students: {self.num_students()}\n"
            f"  Conflicts: {len(self.conflict_matrix)} pairs "
            f"(density={self.conflict_density():.3f})\n"
            f"  Period constraints: {len(self.period_hard_constraints)}\n"
            f"  Room constraints:   {len(self.room_hard_constraints)}\n"
            f"  Weightings: 2row={self.weightings.two_in_a_row}, "
            f"2day={self.weightings.two_in_a_day}, "
            f"spread={self.weightings.period_spread}, "
            f"mixed={self.weightings.non_mixed_durations}, "
            f"front={self.weightings.front_load}"
        )


@dataclass
class Assignment:
    """Assignment of one exam to a (period, room) pair."""
    exam_id: int
    period_id: int
    room_id: int


class Solution:
    """
    Uses flat arrays for O(1) period/room lookup (critical for delta evaluation).
    Maintains _pr_enroll dict for O(1) room occupancy tracking.
    """

    __slots__ = ('problem', 'assignments', '_period_of', '_room_of', '_pr_enroll', '_enroll_cache')

    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        n = problem.num_exams()
        self.assignments: dict[int, tuple[int, int]] = {}
        self._period_of: list[int] = [-1] * n
        self._room_of: list[int] = [-1] * n
        # (period_id, room_id) -> total enrollment currently placed there
        self._pr_enroll: dict[tuple[int, int], int] = {}
        # Cache exam enrollments for O(1) access
        self._enroll_cache: list[int] = [e.enrollment for e in problem.exams]

    def assign(self, exam_id: int, period_id: int, room_id: int):
        enroll = self._enroll_cache[exam_id]
        # Remove from old slot if reassigning
        old_pid = self._period_of[exam_id]
        if old_pid >= 0:
            old_rid = self._room_of[exam_id]
            old_key = (old_pid, old_rid)
            self._pr_enroll[old_key] = self._pr_enroll.get(old_key, 0) - enroll
        # Add to new slot
        new_key = (period_id, room_id)
        self._pr_enroll[new_key] = self._pr_enroll.get(new_key, 0) + enroll
        # Update arrays
        self.assignments[exam_id] = (period_id, room_id)
        self._period_of[exam_id] = period_id
        self._room_of[exam_id] = room_id

    def unassign(self, exam_id: int):
        if exam_id in self.assignments:
            pid, rid = self.assignments[exam_id]
            enroll = self._enroll_cache[exam_id]
            key = (pid, rid)
            self._pr_enroll[key] = self._pr_enroll.get(key, 0) - enroll
            del self.assignments[exam_id]
            self._period_of[exam_id] = -1
            self._room_of[exam_id] = -1

    def get_period(self, exam_id: int) -> Optional[int]:
        p = self._period_of[exam_id]
        return p if p >= 0 else None

    def get_room(self, exam_id: int) -> Optional[int]:
        r = self._room_of[exam_id]
        return r if r >= 0 else None

    def get_pr_enroll(self, period_id: int, room_id: int) -> int:
        """Get total enrollment in a (period, room) slot. O(1)."""
        return self._pr_enroll.get((period_id, room_id), 0)

    def is_complete(self) -> bool:
        return len(self.assignments) == self.problem.num_exams()

    def copy(self) -> Solution:
        new_sol = Solution(self.problem)
        new_sol.assignments = dict(self.assignments)
        new_sol._period_of = list(self._period_of)
        new_sol._room_of = list(self._room_of)
        new_sol._pr_enroll = dict(self._pr_enroll)
        return new_sol

    def exams_in_period(self, period_id: int) -> list[int]:
        """Get all exam IDs assigned to a given period."""
        return [eid for eid, (pid, _) in self.assignments.items() if pid == period_id]

    def exams_in_period_room(self, period_id: int, room_id: int) -> list[int]:
        """Get all exam IDs in a given (period, room)."""
        return [eid for eid, (pid, rid) in self.assignments.items()
                if pid == period_id and rid == room_id]

    def to_output_lines(self) -> list[str]:
        """Generate ITC 2007 output format: one line per exam -> 'period, room'."""
        lines = []
        for eid in range(self.problem.num_exams()):
            if eid in self.assignments:
                pid, rid = self.assignments[eid]
                lines.append(f"{pid}, {rid}")
            else:
                lines.append("-1, -1")  # unassigned
        return lines
