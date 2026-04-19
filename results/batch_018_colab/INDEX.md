# batch_018_colab — Results Index

Generated on Colab; flattened + polished on 2026-04-17.

## Layout

```
batch_018_colab/
├── aggregated.csv              Main batch: 13 algos × 8 ITC sets × 7 seeds
├── batch_meta.json             Batch metadata
├── make_paper_figures.py       Driver: renders 6 figures + 3 tables to graphs/
├── INDEX.md                    This file
├── colab_batch_ip/             CP-SAT IP runs (8 instances — set3/5/7 timed out)
├── colab_batch_scaling/        Synthetic n=50..1000 scaling sweep
├── colab_batch_sweep/          Parameter-sensitivity sweep (sensitivity.csv)
├── colab_batch_chain/          Chain-finder tournament
│   ├── top5_chains.json              ← programmatic top-5
│   ├── top5_chains.txt               ← human-readable version
│   └── tuning/                       ← checkpoints + per-trial logs
└── <algo>_<set>_<seed>/        729 main-batch run dirs (solutions + per-run logs)
```

## Paper figures + tables

Paper-grade outputs now live at the repo root:
- `graphs/fig1..fig6.png`
- `graphs/tables/t1..t3.{csv,tex}`

Regenerate with `python3 results/batch_018_colab/make_paper_figures.py`.

The old `figures/` directory and `make_extra_plots.py` driver have been
removed as superseded.

## Champion Chain

From `colab_batch_chain/top5_chains.json`:

| Rank | Score  | Chain                                          |
|------|--------|------------------------------------------------|
| 1    | 0.9282 | alns → kempe → tabu                            |
| 2    | 0.9319 | kempe → tabu → alns → kempe → alns → vns → tabu |
| 3    | 0.9334 | kempe → kempe → tabu → abc                     |
| 4    | 0.9489 | kempe → alns → kempe → alns → vns → tabu       |
| 5    | 0.9730 | gd → vns → tabu → gd → alns                    |

The new champion (**alns → kempe → tabu**) dethrones the previous tuned
chain (`kempe → alns → kempe → alns → vns → tabu` — now #4).

The champion is persisted to `tooling/tuned_params.json`; load via
`tooling.tuned_params.load_best_chain()` or the notebook's
*Champion Chain* section.

## Regenerate Figures

```bash
python3 results/batch_018_colab/make_extra_plots.py
```

Re-creates `scaling_overview.png`, `sensitivity_overview.png`,
`ip_vs_heuristic.png`, `top5_chains.png`. The other figures come from
`utils/plotting.py` invoked in the Colab notebook.

## Known Gaps

- CP-SAT IP produced no solution for set 3, 5, 7 within the 2-hour
  time limit; those instances are skipped in `ip_vs_heuristic.png`.
