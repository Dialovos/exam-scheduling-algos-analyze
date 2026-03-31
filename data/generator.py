"""
Generates instances with configurable size and conflict density,
following the project plan parameters:
  - Exam sizes: 50 to 1000
  - Students: ~7x exams
  - Conflict density: 5% to 50%
"""

import random
from data.models import (
    ProblemInstance, Exam, Period, Room,
    InstitutionalWeightings,
)


def generate_synthetic(
    num_exams: int = 50,
    student_ratio: float = 7.0,
    conflict_density: float = 0.15,
    num_rooms: int = 5,
    room_capacity: int = 200,
    periods_per_day: int = 3,
    num_days: int = 7,
    exam_durations: list[int] = None,
    seed: int = 42,
) -> ProblemInstance:
    """
    Args:
        num_exams: Number of exams to schedule.
        student_ratio: Students = num_exams * student_ratio.
        conflict_density: Target fraction of conflicting exam pairs (0 to 1).
        num_rooms: Number of available rooms.
        room_capacity: Capacity of each room.
        periods_per_day: Number of exam periods per day.
        num_days: Number of days in the exam session.
        exam_durations: Possible exam durations in minutes.
        seed: Random seed for reproducibility.

    Returns:
        A ProblemInstance ready for solving.
    """
    rng = random.Random(seed)
    if exam_durations is None:
        exam_durations = [60, 90, 120, 180]

    num_students = int(num_exams * student_ratio)
    num_periods = periods_per_day * num_days

    # Ensure we have enough period-room slots
    total_slots = num_periods * num_rooms
    if total_slots < num_exams:
        # Increase periods to accommodate
        num_days = (num_exams // (periods_per_day * num_rooms)) + 2
        num_periods = periods_per_day * num_days
        print(f"[Generator] Adjusted to {num_days} days ({num_periods} periods) to fit {num_exams} exams")

    problem = ProblemInstance()

    # --- Generate Exams ---
    # Assign students to exams to achieve target conflict density
    # Strategy: each student enrolls in k exams drawn from exam pool
    exams = []
    for eid in range(num_exams):
        dur = rng.choice(exam_durations)
        exams.append(Exam(id=eid, duration=dur, students=set()))

    # Calculate average exams per student to hit target conflict density
    # conflict_density ≈ 1 - (1 - k(k-1)/n^2)^S approximately
    # Simplified: use iterative enrollment approach
    avg_exams_per_student = max(2, int(2 + conflict_density * 6))

    for sid in range(num_students):
        # Each student enrolls in a random subset of exams
        k = max(1, rng.randint(
            max(1, avg_exams_per_student - 2),
            avg_exams_per_student + 2
        ))
        k = min(k, num_exams)
        enrolled = rng.sample(range(num_exams), k)
        for eid in enrolled:
            exams[eid].students.add(sid)

    problem.exams = exams

    # --- Generate Periods ---
    periods = []
    pid = 0
    for day in range(num_days):
        for slot in range(periods_per_day):
            hour = 9 + slot * 3  # 9:00, 12:00, 15:00
            dur = rng.choice(exam_durations)
            dur = max(dur, max(exam_durations))  # Period must accommodate longest exam
            periods.append(Period(
                id=pid,
                date=f"{day+1:02d}:01:2025",
                time=f"{hour:02d}:00:00",
                duration=dur,
                penalty=0,
                day=day,
            ))
            pid += 1
    problem.periods = periods

    # --- Generate Rooms ---
    rooms = []
    for rid in range(num_rooms):
        # Vary room capacities slightly
        cap = max(20, room_capacity + rng.randint(-room_capacity // 3, room_capacity // 3))
        rooms.append(Room(id=rid, capacity=cap, penalty=0))
    problem.rooms = rooms

    # --- Institutional Weightings (standard ITC 2007 style) ---
    problem.weightings = InstitutionalWeightings(
        two_in_a_row=7,
        two_in_a_day=5,
        period_spread=3,
        non_mixed_durations=2,
        front_load=(max(1, num_exams // 10), max(1, num_periods // 4), 5),
    )

    problem.build_derived_data()
    return problem


def generate_suite(sizes: list[int] = None, seed: int = 42) -> dict[str, ProblemInstance]:
    """Generate a suite of instances at different sizes for benchmarking."""
    if sizes is None:
        sizes = [50, 100, 200, 500]

    suite = {}
    for n in sizes:
        name = f"synthetic_{n}"
        suite[name] = generate_synthetic(
            num_exams=n,
            student_ratio=7.0,
            conflict_density=0.15,
            num_rooms=max(3, n // 20),
            room_capacity=max(50, int(n * 0.4)),
            seed=seed + n,
        )
    return suite


def write_itc2007_format(problem: ProblemInstance, filepath: str):
    """Write a ProblemInstance to ITC 2007 .exam format (for interoperability)."""
    with open(filepath, 'w') as f:
        # Exams section
        f.write(f"[Exams:{len(problem.exams)}]\n")
        for exam in problem.exams:
            students_str = ", ".join(str(s) for s in sorted(exam.students))
            if students_str:
                f.write(f"{exam.duration}, {students_str}\n")
            else:
                f.write(f"{exam.duration}\n")

        # Periods section
        f.write(f"[Periods:{len(problem.periods)}]\n")
        for p in problem.periods:
            f.write(f"{p.date}, {p.time}, {p.duration}, {p.penalty}\n")

        # Rooms section
        f.write(f"[Rooms:{len(problem.rooms)}]\n")
        for r in problem.rooms:
            f.write(f"{r.capacity}, {r.penalty}\n")

        # Period hard constraints
        f.write("[PeriodHardConstraints]\n")
        for phc in problem.period_hard_constraints:
            f.write(f"{phc.exam1_id}, {phc.constraint_type}, {phc.exam2_id}\n")

        # Room hard constraints
        f.write("[RoomHardConstraints]\n")
        for rhc in problem.room_hard_constraints:
            f.write(f"{rhc.exam_id}, {rhc.constraint_type}\n")

        # Institutional weightings
        f.write("[InstitutionalWeightings]\n")
        w = problem.weightings
        f.write(f"TWOINAROW:{w.two_in_a_row}\n")
        f.write(f"TWOINADAY:{w.two_in_a_day}\n")
        f.write(f"PERIODSPREAD:{w.period_spread}\n")
        f.write(f"NONMIXEDDURATIONS:{w.non_mixed_durations}\n")
        fl = w.front_load
        f.write(f"FRONTLOAD, {fl[0]}, {fl[1]}, {fl[2]}\n")
