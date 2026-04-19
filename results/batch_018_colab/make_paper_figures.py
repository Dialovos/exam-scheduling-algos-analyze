"""Render every paper figure + table from batch_018 data.

Outputs go to ``graphs/`` (figures) and ``graphs/tables/`` (tables) at the
repo root. Safe to re-run; each call overwrites previous files.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from utils.plots.paper import (  # noqa: E402
    make_fig1, make_fig2, make_fig3, make_fig4, make_fig5, make_fig6,
    make_fig7, make_fig8,
)
from utils.tables.paper import make_t1, make_t2, make_t3  # noqa: E402


FIGS_DIR = REPO / "graphs"
TABLES_DIR = FIGS_DIR / "tables"


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("[paper] Rendering 8 figures + 3 tables...")

    make_fig1(FIGS_DIR / "fig1_pareto.png");              print("  [fig1] pareto done")
    make_fig2(FIGS_DIR / "fig2_family_heatmap.png");      print("  [fig2] family heatmap done")
    make_fig3(FIGS_DIR / "fig3_chain_methodology.png");   print("  [fig3] chain methodology done")
    make_fig4(FIGS_DIR / "fig4_scaling.png");             print("  [fig4] scaling done")
    make_fig5(FIGS_DIR / "fig5_sensitivity.png");         print("  [fig5] sensitivity done")
    make_fig6(FIGS_DIR / "fig6_ip_vs_heuristic.png");     print("  [fig6] ip vs heuristic done")
    make_fig7(FIGS_DIR / "fig7_gap_heatmap.png");         print("  [fig7] gap heatmap done")
    make_fig8(FIGS_DIR / "fig8_gap_leaderboard.png");     print("  [fig8] gap leaderboard done")

    make_t1(TABLES_DIR); print("  [t1] leaderboard done")
    make_t2(TABLES_DIR); print("  [t2] chain top-5 done")
    make_t3(TABLES_DIR); print("  [t3] partial-adopt done")

    elapsed = time.time() - t0
    print(f"[paper] All outputs in {FIGS_DIR} (took {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
