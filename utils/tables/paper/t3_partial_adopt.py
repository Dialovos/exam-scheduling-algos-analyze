"""T3 -- Partial-Adopt Golden Params.

Data is fixed at the values from the 2026-04-17 A/B benchmark on exam_comp_set2
(3 seeds). We hard-code these values: the A/B measurements are not regenerable
from the current batch, and re-running the A/B is not in scope for this work.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.tables.paper.t1_leaderboard import _to_booktabs


ROWS = [
    ("Kempe", "iters",    "10000", "16284", "+15%",  "1.52x", "adopted"),
    ("Tabu",  "iters",    "2000",  "3162",  "+6%",   "1.55x", "adopted"),
    ("Tabu",  "tenure",   "(bundled)", "(bundled)", "(part of tabu iters A/B)", "", "adopted"),
    ("Tabu",  "patience", "(bundled)", "(bundled)", "(part of tabu iters A/B)", "", "adopted"),
    ("ALNS",  "iters",    "1500",  "14341", "+7%",   "7.58x", "reverted"),
    ("GVNS",  "iters",    "5000",  "41117", "+1.6%", "9.08x", "reverted"),
    ("GVNS",  "budget",   "0",     "62",    "(part of gvns iters A/B)", "", "reverted"),
]


def make_t3(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(ROWS, columns=[
        "Algorithm", "Param", "Pre-Study", "Tuner-Proposed",
        "Quality Change", "Runtime Mult", "Verdict",
    ])

    csv_path = out_dir / "t3_partial_adopt.csv"
    tex_path = out_dir / "t3_partial_adopt.tex"
    out_df.to_csv(csv_path, index=False)
    tex_path.write_text(_to_booktabs(out_df))

    return csv_path, tex_path
