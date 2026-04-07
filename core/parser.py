"""
Parser:
Format specification from the ITC 2007 technical report (McCollum et al., 2007).
File sections:
  [Exams:N]           - N lines: duration, student1, student2, ...
  [Periods:N]         - N lines: date, time, duration, penalty
  [Rooms:N]           - N lines: capacity, penalty
  [PeriodHardConstraints] - lines: exam1, type, exam2
  [RoomHardConstraints]   - lines: exam, ROOM_EXCLUSIVE
  [InstitutionalWeightings] - key:value pairs
"""

from core.models import (
    ProblemInstance, Exam, Period, Room,
    PeriodHardConstraint, RoomHardConstraint,
    InstitutionalWeightings,
)


def parse_itc2007_exam(filepath: str, limit: int = 0) -> ProblemInstance:
    """
    Args:
        filepath: Path to the .exam file.
        limit: If > 0, only load the first `limit` exams from the file.
               Constraints referencing exams beyond this limit are dropped.
               This lets you incrementally scale tests on large datasets:
                 limit=50  → fast smoke test
                 limit=200 → medium benchmark
                 limit=0   → full dataset
    """
    problem = ProblemInstance()
    problem.weightings = InstitutionalWeightings()

    with open(filepath, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    idx = 0

    while idx < len(lines):
        line = lines[idx]

        # --- [Exams:N] ---
        if line.startswith("[Exams:"):
            n = int(line.split(":")[1].rstrip("]"))
            actual_n = n if limit <= 0 else min(n, limit)
            idx += 1
            for eid in range(n):
                parts = lines[idx].split(",")
                parts = [p.strip() for p in parts]
                duration = int(parts[0])
                students = set()
                for s in parts[1:]:
                    s = s.strip()
                    if s:
                        students.add(int(s))
                if eid < actual_n:
                    problem.exams.append(Exam(id=eid, duration=duration, students=students))
                idx += 1
            continue

        # --- [Periods:N] ---
        elif line.startswith("[Periods:"):
            n = int(line.split(":")[1].rstrip("]"))
            idx += 1
            current_date = None
            current_day = -1
            for pid in range(n):
                parts = lines[idx].split(",")
                parts = [p.strip() for p in parts]
                date = parts[0]
                time = parts[1]
                duration = int(parts[2])
                penalty = int(parts[3]) if len(parts) > 3 else 0

                # Track day changes
                if date != current_date:
                    current_day += 1
                    current_date = date

                problem.periods.append(Period(
                    id=pid, date=date, time=time,
                    duration=duration, penalty=penalty,
                    day=current_day
                ))
                idx += 1
            continue

        # --- [Rooms:N] ---
        elif line.startswith("[Rooms:"):
            n = int(line.split(":")[1].rstrip("]"))
            idx += 1
            for rid in range(n):
                parts = lines[idx].split(",")
                parts = [p.strip() for p in parts]
                capacity = int(parts[0])
                penalty = int(parts[1]) if len(parts) > 1 else 0
                problem.rooms.append(Room(id=rid, capacity=capacity, penalty=penalty))
                idx += 1
            continue

        # --- [PeriodHardConstraints] ---
        elif line.startswith("[PeriodHardConstraints]"):
            idx += 1
            while idx < len(lines) and not lines[idx].startswith("["):
                if lines[idx].strip() == "":
                    idx += 1
                    continue
                parts = lines[idx].split(",")
                parts = [p.strip() for p in parts]
                if len(parts) >= 3:
                    e1 = int(parts[0])
                    ctype = parts[1]
                    e2 = int(parts[2])
                    max_eid = len(problem.exams)
                    if e1 < max_eid and e2 < max_eid:
                        problem.period_hard_constraints.append(
                            PeriodHardConstraint(e1, ctype, e2)
                        )
                idx += 1
            continue

        # --- [RoomHardConstraints] ---
        elif line.startswith("[RoomHardConstraints]"):
            idx += 1
            while idx < len(lines) and not lines[idx].startswith("["):
                if lines[idx].strip() == "":
                    idx += 1
                    continue
                parts = lines[idx].split(",")
                parts = [p.strip() for p in parts]
                if len(parts) >= 2:
                    eid = int(parts[0])
                    ctype = parts[1]
                    if eid < len(problem.exams):
                        problem.room_hard_constraints.append(
                            RoomHardConstraint(eid, ctype)
                        )
                idx += 1
            continue

        # --- [InstitutionalWeightings] ---
        elif line.startswith("[InstitutionalWeightings]"):
            idx += 1
            while idx < len(lines) and not lines[idx].startswith("["):
                if lines[idx].strip() == "":
                    idx += 1
                    continue
                raw = lines[idx].strip()
                if raw.startswith("TWOINAROW"):
                    val = raw.split(":")[1].strip() if ":" in raw else raw.split(",")[1].strip()
                    problem.weightings.two_in_a_row = int(val)
                elif raw.startswith("TWOINADAY"):
                    val = raw.split(":")[1].strip() if ":" in raw else raw.split(",")[1].strip()
                    problem.weightings.two_in_a_day = int(val)
                elif raw.startswith("PERIODSPREAD"):
                    val = raw.split(":")[1].strip() if ":" in raw else raw.split(",")[1].strip()
                    problem.weightings.period_spread = int(val)
                elif raw.startswith("NONMIXEDDURATIONS"):
                    val = raw.split(":")[1].strip() if ":" in raw else raw.split(",")[1].strip()
                    problem.weightings.non_mixed_durations = int(val)
                elif raw.startswith("FRONTLOAD"):
                    parts_fl = raw.split(",")
                    # FRONTLOAD, num_largest, num_last_periods, penalty
                    if len(parts_fl) >= 4:
                        problem.weightings.front_load = (
                            int(parts_fl[1].strip()),
                            int(parts_fl[2].strip()),
                            int(parts_fl[3].strip()),
                        )
                idx += 1
            continue
        else:
            idx += 1

    problem.build_derived_data()
    return problem


def write_solution_itc2007(solution, filepath: str):
    """Write solution in ITC 2007 output format."""
    lines = solution.to_output_lines()
    with open(filepath, 'w') as f:
        for line in lines:
            f.write(line + "\n")
