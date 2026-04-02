"""
Models real ITC 2007 characteristics analyzed from sets 4, 5, and 7:
  - Power-law enrollment (few massive exams, long tail of small ones)
  - Department-clustered student enrollment (realistic conflict patterns)
  - Tight period counts (max_degree >> num_periods, many exams per period)
  - Multiple exam/period durations, period and room penalties
  - COINCIDENCE, EXCLUSION, AFTER hard constraints
  - Varied room capacities (10×+ range)

Presets (n=200):
  "easy"        — 60 periods, 3 rooms, few constraints
  "medium"      — 30 periods, 2 rooms, moderate constraints
  "hard"        — 15 periods, 2 rooms, many constraints
  "competition" — 10 periods, 1 room, tightest
"""

import random
from data.models import (
    ProblemInstance, Exam, Period, Room,
    PeriodHardConstraint, RoomHardConstraint,
    InstitutionalWeightings,
)

PRESETS = {
    "easy": dict(
        student_ratio=4.0, period_ratio=0.30, periods_per_day=3,
        room_count_ratio=0.015, room_cap_base=300, room_cap_spread=0.3,
        enrollment_skew=1.5, dept_count_ratio=0.12, exams_per_student=3.0,
        period_penalty_frac=0.0, room_penalty_frac=0.0,
        phc_coincidence=0, phc_exclusion=0, phc_after=0,
        weightings=dict(two_in_a_row=5, two_in_a_day=3, period_spread=2,
                        non_mixed_durations=1, fl_frac=0.05, fl_penalty=3),
    ),
    "medium": dict(
        student_ratio=6.0, period_ratio=0.15, periods_per_day=3,
        room_count_ratio=0.01, room_cap_base=200, room_cap_spread=0.5,
        enrollment_skew=2.5, dept_count_ratio=0.10, exams_per_student=3.5,
        period_penalty_frac=0.15, room_penalty_frac=0.1,
        phc_coincidence=4, phc_exclusion=6, phc_after=2,
        weightings=dict(two_in_a_row=7, two_in_a_day=5, period_spread=3,
                        non_mixed_durations=5, fl_frac=0.10, fl_penalty=5),
    ),
    "hard": dict(
        student_ratio=8.0, period_ratio=0.12, periods_per_day=3,
        room_count_ratio=0.008, room_cap_base=200, room_cap_spread=0.7,
        enrollment_skew=3.0, dept_count_ratio=0.10, exams_per_student=4.0,
        period_penalty_frac=0.25, room_penalty_frac=0.15,
        phc_coincidence=6, phc_exclusion=10, phc_after=4,
        weightings=dict(two_in_a_row=9, two_in_a_day=7, period_spread=4,
                        non_mixed_durations=10, fl_frac=0.15, fl_penalty=10),
    ),
    "competition": dict(
        student_ratio=10.0, period_ratio=0.08, periods_per_day=3,
        room_count_ratio=0.004, room_cap_base=250, room_cap_spread=0.8,
        enrollment_skew=3.5, dept_count_ratio=0.08, exams_per_student=4.5,
        period_penalty_frac=0.20, room_penalty_frac=0.10,
        phc_coincidence=8, phc_exclusion=15, phc_after=5,
        weightings=dict(two_in_a_row=9, two_in_a_day=5, period_spread=2,
                        non_mixed_durations=10, fl_frac=0.18, fl_penalty=5),
    ),
}


def generate_synthetic(
    num_exams: int = 200,
    preset: str = "medium",
    seed: int = 42,
    **overrides,
) -> ProblemInstance:
    """Generate a competition-level exam scheduling instance.

    Args:
        num_exams: Number of exams.
        preset: "easy", "medium", "hard", or "competition".
        seed: Random seed.
        **overrides: Override any preset parameter.
    """
    rng = random.Random(seed)
    cfg = dict(PRESETS.get(preset, PRESETS["medium"]))
    cfg.update(overrides)

    ne = num_exams
    num_students = int(ne * cfg['student_ratio'])

    # ── Periods: tight count based on period_ratio ──
    ppd = cfg['periods_per_day']
    np_ = max(3, int(ne * cfg['period_ratio']))
    # Round up to multiple of ppd
    num_days = max(1, (np_ + ppd - 1) // ppd)
    np_ = num_days * ppd

    period_dur_pool = [120, 180]
    periods = []
    pid = 0
    for day in range(num_days):
        for slot in range(ppd):
            hour = 9 + slot * 3
            dur = rng.choice(period_dur_pool)
            periods.append(Period(id=pid, date=f"{day+1:02d}:01:2025",
                                  time=f"{hour:02d}:00:00", duration=dur,
                                  penalty=0, day=day))
            pid += 1

    # Period penalties
    n_pp = max(0, int(np_ * cfg['period_penalty_frac']))
    for _ in range(n_pp):
        periods[rng.randint(0, np_-1)].penalty = rng.choice([50, 100, 200, 500])

    # ── Rooms ──
    nr = max(1, int(ne * cfg['room_count_ratio']))
    rooms = []
    base = cfg['room_cap_base']
    spread = cfg['room_cap_spread']
    if nr == 1:
        # Single large room — capacity must hold many exams per period
        exams_per_period = ne / np_
        cap = max(500, int(num_students * 0.3), int(exams_per_period * base * 0.3))
        rooms.append(Room(id=0, capacity=cap, penalty=0))
    else:
        for rid in range(nr):
            if rid == 0:
                cap = int(base * (2.0 + rng.random()))
            else:
                cap = max(20, int(base * (0.15 + rng.random() * (1.0 + spread))))
            rooms.append(Room(id=rid, capacity=cap, penalty=0))
        rooms.sort(key=lambda r: -r.capacity)
        for i, r in enumerate(rooms): r.id = i

    n_rp = max(0, int(nr * cfg['room_penalty_frac']))
    for _ in range(n_rp):
        rooms[rng.randint(0, nr-1)].penalty = rng.choice([30, 50, 70, 100])

    max_cap = max(r.capacity for r in rooms)

    # ── Exams (power-law enrollment) ──
    dur_pool = [60, 90, 120, 150, 180]
    dur_w = [3, 2, 3, 1, 2]
    exams = []
    for eid in range(ne):
        dur = rng.choices(dur_pool, weights=dur_w, k=1)[0]
        exams.append(Exam(id=eid, duration=dur, students=set()))

    # ── Department-clustered enrollment ──
    n_depts = max(3, int(ne * cfg['dept_count_ratio']))
    dept_exams = [[] for _ in range(n_depts)]
    for eid in range(ne):
        d1 = rng.randint(0, n_depts - 1)
        dept_exams[d1].append(eid)
        if rng.random() < 0.2:
            dept_exams[rng.randint(0, n_depts - 1)].append(eid)

    for sid in range(num_students):
        dept = rng.randint(0, n_depts - 1)
        avail = dept_exams[dept] if dept_exams[dept] else list(range(ne))
        eps = cfg.get('exams_per_student', 4.0)
        k = max(1, min(int(rng.gauss(eps, 1.0)), len(avail), 8))
        for eid in rng.sample(avail, k):
            exams[eid].students.add(sid)

    # Boost top exams for power-law enrollment tail.
    # Use existing students (resampled) to avoid creating universal conflicts.
    all_sids = list(range(num_students))
    exam_by_enr = sorted(range(ne), key=lambda e: len(exams[e].students), reverse=True)
    n_boost = max(1, int(ne * 0.03))
    current_max = len(exams[exam_by_enr[0]].students) if ne > 0 else 1
    target_max = min(max_cap, int(current_max * (2.0 + cfg['enrollment_skew'])))
    for i in range(n_boost):
        eid = exam_by_enr[i]
        cur = len(exams[eid].students)
        extra = max(0, target_max - cur - i * (target_max // (n_boost + 1)))
        # Add students NOT currently enrolled in conflicting exams (reduces degree blow-up)
        candidates = [s for s in rng.sample(all_sids, min(extra * 3, num_students))
                      if s not in exams[eid].students]
        for s in candidates[:extra]:
            exams[eid].students.add(s)

    # ── Period hard constraints (scaled to instance size) ──
    phcs = []
    used = set()

    # Scale constraint counts with instance size (target: ~0.05-0.15 per exam)
    scale = ne / 200.0
    n_coin = max(0, int(cfg['phc_coincidence'] * scale))
    n_excl = max(0, int(cfg['phc_exclusion'] * scale))
    n_after = max(0, int(cfg['phc_after'] * scale))

    def _pair(avoid_conflict=False):
        for _ in range(500):
            a, b = rng.randint(0, ne-1), rng.randint(0, ne-1)
            if a == b: continue
            if (a,b) in used or (b,a) in used: continue
            if avoid_conflict and (exams[a].students & exams[b].students):
                continue  # Can't require coincidence for conflicting exams
            used.add((a,b))
            return a, b
        return None, None

    for _ in range(n_coin):
        a, b = _pair(avoid_conflict=True)  # COINCIDENCE only for non-conflicting pairs
        if a is not None: phcs.append(PeriodHardConstraint(a, "EXAM_COINCIDENCE", b))
    for _ in range(n_excl):
        a, b = _pair()
        if a is not None: phcs.append(PeriodHardConstraint(a, "EXCLUSION", b))
    for _ in range(n_after):
        a, b = _pair()
        if a is not None: phcs.append(PeriodHardConstraint(a, "AFTER", b))

    # ── Weightings ──
    w = cfg['weightings']
    fl_n = max(1, int(ne * w['fl_frac']))
    fl_last = max(1, np_ // 4)

    problem = ProblemInstance()
    problem.exams = exams
    problem.periods = periods
    problem.rooms = rooms
    problem.period_hard_constraints = phcs
    problem.room_hard_constraints = []
    problem.weightings = InstitutionalWeightings(
        two_in_a_row=w['two_in_a_row'], two_in_a_day=w['two_in_a_day'],
        period_spread=w['period_spread'], non_mixed_durations=w['non_mixed_durations'],
        front_load=(fl_n, fl_last, w['fl_penalty']),
    )
    problem.build_derived_data()
    return problem


def generate_suite(sizes=None, preset="medium", seed=42):
    if sizes is None: sizes = [50, 100, 200, 500]
    return {f"synthetic_{n}": generate_synthetic(n, preset, seed+n) for n in sizes}


def write_itc2007_format(problem: ProblemInstance, filepath: str):
    """Write a ProblemInstance to ITC 2007 .exam format."""
    with open(filepath, 'w') as f:
        f.write(f"[Exams:{len(problem.exams)}]\n")
        for exam in problem.exams:
            ss = ", ".join(str(s) for s in sorted(exam.students))
            f.write(f"{exam.duration}, {ss}\n" if ss else f"{exam.duration}\n")
        f.write(f"[Periods:{len(problem.periods)}]\n")
        for p in problem.periods:
            f.write(f"{p.date}, {p.time}, {p.duration}, {p.penalty}\n")
        f.write(f"[Rooms:{len(problem.rooms)}]\n")
        for r in problem.rooms:
            f.write(f"{r.capacity}, {r.penalty}\n")
        f.write("[PeriodHardConstraints]\n")
        for c in problem.period_hard_constraints:
            f.write(f"{c.exam1_id}, {c.constraint_type}, {c.exam2_id}\n")
        f.write("[RoomHardConstraints]\n")
        for c in problem.room_hard_constraints:
            f.write(f"{c.exam_id}, {c.constraint_type}\n")
        f.write("[InstitutionalWeightings]\n")
        w = problem.weightings
        f.write(f"TWOINAROW:{w.two_in_a_row}\nTWOINADAY:{w.two_in_a_day}\n")
        f.write(f"PERIODSPREAD:{w.period_spread}\nNONMIXEDDURATIONS:{w.non_mixed_durations}\n")
        fl = w.front_load
        f.write(f"FRONTLOAD, {fl[0]}, {fl[1]}, {fl[2]}\n")