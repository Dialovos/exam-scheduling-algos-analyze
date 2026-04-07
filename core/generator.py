"""
Models real ITC 2007 characteristics analyzed from sets 4, 5, and 7:
  - Power-law enrollment (few massive exams, long tail of small ones)
  - Department-clustered student enrollment (realistic conflict patterns)
  - Tight period counts (max_degree >> num_periods, many exams per period)
  - Multiple exam/period durations, period and room penalties
  - COINCIDENCE, EXCLUSION, AFTER hard constraints
  - Varied room capacities (10x+ range)

Presets (approximate at n=200, scales with instance size):
  "easy"        — many periods/rooms, few constraints
  "medium"      — moderate capacity, moderate constraints
  "hard"        — tight capacity, many constraints
  "competition" — tightest ratios, closest to ITC 2007 difficulty

A feasibility floor ensures enough room×period slots exist at any size.
"""

import random
import numpy as np
from core.models import (
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
    nprng = np.random.default_rng(seed)
    cfg = dict(PRESETS.get(preset, PRESETS["medium"]))
    cfg.update(overrides)

    ne = num_exams
    num_students = int(ne * cfg['student_ratio'])

    # ── Periods: tight count based on period_ratio ──
    ppd = cfg['periods_per_day']
    np_ = max(3, round(ne * cfg['period_ratio']))
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
    if n_pp > 0:
        pp_indices = nprng.integers(0, np_, size=n_pp)
        pp_vals = nprng.choice([50, 100, 200, 500], size=n_pp)
        for idx, val in zip(pp_indices, pp_vals):
            periods[idx].penalty = int(val)

    # ── Rooms ──
    nr = max(1, round(ne * cfg['room_count_ratio']))
    # Feasibility floor: ensure enough room×period slots for all exams.
    # Target ≤6 exams per slot — tighter ratios make greedy construction
    # nearly impossible regardless of algorithm quality.
    max_exams_per_slot = 6
    min_rooms = max(1, -(-ne // (np_ * max_exams_per_slot)))  # ceil div
    nr = max(nr, min_rooms)
    rooms = []
    base = cfg['room_cap_base']
    spread = cfg['room_cap_spread']
    if nr == 1:
        exams_per_period = ne / np_
        cap = max(500, int(num_students * 0.3), int(exams_per_period * base * 0.3))
        rooms.append(Room(id=0, capacity=cap, penalty=0))
    else:
        caps = np.empty(nr, dtype=int)
        caps[0] = int(base * (2.0 + nprng.random()))
        caps[1:] = np.maximum(20, (base * (0.15 + nprng.random(nr - 1) * (1.0 + spread))).astype(int))
        order = np.argsort(-caps)
        for i, idx in enumerate(order):
            rooms.append(Room(id=i, capacity=int(caps[idx]), penalty=0))

    n_rp = max(0, int(nr * cfg['room_penalty_frac']))
    if n_rp > 0:
        rp_indices = nprng.integers(0, nr, size=n_rp)
        rp_vals = nprng.choice([30, 50, 70, 100], size=n_rp)
        for idx, val in zip(rp_indices, rp_vals):
            rooms[idx].penalty = int(val)

    max_cap = max(r.capacity for r in rooms)

    # ── Exams (power-law enrollment) ──
    dur_pool = np.array([60, 90, 120, 150, 180])
    dur_w = np.array([3, 2, 3, 1, 2], dtype=float)
    dur_w /= dur_w.sum()
    exam_durs = nprng.choice(dur_pool, size=ne, p=dur_w)
    exams = [Exam(id=eid, duration=int(exam_durs[eid]), students=set()) for eid in range(ne)]

    # ── Department-clustered enrollment (vectorized) ──
    n_depts = max(3, int(ne * cfg['dept_count_ratio']))

    # Assign exams to departments
    dept_exams = [[] for _ in range(n_depts)]
    primary_depts = nprng.integers(0, n_depts, size=ne)
    for eid in range(ne):
        dept_exams[primary_depts[eid]].append(eid)
    # 20% chance of cross-listing
    cross_mask = nprng.random(ne) < 0.2
    cross_depts = nprng.integers(0, n_depts, size=ne)
    for eid in np.where(cross_mask)[0]:
        dept_exams[cross_depts[eid]].append(int(eid))

    # Pre-convert to arrays for fast sampling
    dept_exam_arrays = [np.array(d, dtype=np.int32) if d else np.arange(ne, dtype=np.int32)
                        for d in dept_exams]

    # Batch student enrollment — the main bottleneck for large instances
    eps = cfg.get('exams_per_student', 4.0)
    student_depts = nprng.integers(0, n_depts, size=num_students)
    student_k = np.clip(nprng.normal(eps, 1.0, size=num_students).astype(int), 1, 8)

    for sid in range(num_students):
        dept = student_depts[sid]
        avail = dept_exam_arrays[dept]
        k = min(int(student_k[sid]), len(avail))
        chosen = nprng.choice(avail, size=k, replace=False)
        for eid in chosen:
            exams[eid].students.add(sid)

    # Boost top exams for power-law enrollment tail
    exam_sizes = np.array([len(exams[e].students) for e in range(ne)])
    exam_by_enr = np.argsort(-exam_sizes)
    n_boost = max(1, int(ne * 0.03))
    current_max = int(exam_sizes[exam_by_enr[0]]) if ne > 0 else 1
    target_max = min(max_cap, int(current_max * (2.0 + cfg['enrollment_skew'])))

    for i in range(n_boost):
        eid = int(exam_by_enr[i])
        cur = len(exams[eid].students)
        extra = max(0, target_max - cur - i * (target_max // (n_boost + 1)))
        if extra <= 0:
            continue
        # Sample candidates not already enrolled, using set difference
        sample_size = min(extra * 3, num_students)
        candidates = nprng.choice(num_students, size=sample_size, replace=False)
        existing = exams[eid].students
        new_students = [int(s) for s in candidates if s not in existing]
        for s in new_students[:extra]:
            exams[eid].students.add(s)

    # ── Period hard constraints (scaled to instance size) ──
    phcs = []
    used = set()

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
                continue
            used.add((a,b))
            return a, b
        return None, None

    for _ in range(n_coin):
        a, b = _pair(avoid_conflict=True)
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
    """Write a ProblemInstance to ITC 2007 .exam format (buffered)."""
    lines = []
    lines.append(f"[Exams:{len(problem.exams)}]")
    for exam in problem.exams:
        if exam.students:
            lines.append(f"{exam.duration}, " + ", ".join(map(str, sorted(exam.students))))
        else:
            lines.append(str(exam.duration))
    lines.append(f"[Periods:{len(problem.periods)}]")
    for p in problem.periods:
        lines.append(f"{p.date}, {p.time}, {p.duration}, {p.penalty}")
    lines.append(f"[Rooms:{len(problem.rooms)}]")
    for r in problem.rooms:
        lines.append(f"{r.capacity}, {r.penalty}")
    lines.append("[PeriodHardConstraints]")
    for c in problem.period_hard_constraints:
        lines.append(f"{c.exam1_id}, {c.constraint_type}, {c.exam2_id}")
    lines.append("[RoomHardConstraints]")
    for c in problem.room_hard_constraints:
        lines.append(f"{c.exam_id}, {c.constraint_type}")
    lines.append("[InstitutionalWeightings]")
    w = problem.weightings
    lines.append(f"TWOINAROW:{w.two_in_a_row}")
    lines.append(f"TWOINADAY:{w.two_in_a_day}")
    lines.append(f"PERIODSPREAD:{w.period_spread}")
    lines.append(f"NONMIXEDDURATIONS:{w.non_mixed_durations}")
    fl = w.front_load
    lines.append(f"FRONTLOAD, {fl[0]}, {fl[1]}, {fl[2]}")

    with open(filepath, 'w') as f:
        f.write("\n".join(lines) + "\n")
