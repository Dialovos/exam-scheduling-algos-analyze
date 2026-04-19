"""T2 -- Chain Top-5 table (rank, chain string, score, n_evals, per-step params)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.plots.shared import load_batch018
from utils.tables.paper.t1_leaderboard import _to_booktabs


def _format_chain(chain_steps):
    return " -> ".join(step[0] for step in chain_steps)


def _format_per_step_params(chain_steps):
    parts = []
    for algo, params in chain_steps:
        if not params:
            parts.append(f"{algo}()")
        else:
            kv = ", ".join(f"{k}={v}" for k, v in params.items())
            parts.append(f"{algo}({kv})")
    return "; ".join(parts)


def make_t2(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b = load_batch018()
    top5 = b.chain_top5["top5"]

    rows = []
    for entry in top5:
        steps = entry["chain"]
        total_iters = 0
        for _, p in steps:
            for k, v in p.items():
                if k.endswith("_iters") and isinstance(v, (int, float)):
                    total_iters += int(v)
        rows.append({
            "Rank": entry["rank"],
            "Chain": _format_chain(steps),
            "Score": f"{entry['score']:.4f}",
            "n_evals": entry["n_evaluations"],
            "Total_Iters": total_iters,
            "Per-step Params": _format_per_step_params(steps),
        })

    out_df = pd.DataFrame(rows, columns=[
        "Rank", "Chain", "Score", "n_evals", "Total_Iters", "Per-step Params",
    ])

    csv_path = out_dir / "t2_chain_top5.csv"
    tex_path = out_dir / "t2_chain_top5.tex"
    out_df.to_csv(csv_path, index=False)
    tex_path.write_text(_to_booktabs(out_df))

    return csv_path, tex_path
