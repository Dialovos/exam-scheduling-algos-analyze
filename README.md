# Exam Scheduling

Team: Hoang Le, Ian Cronin

## The Problem

This is a **Capacitated Examination Timetabling Problem** — an NP-hard variant of graph coloring. Given a set of exams, students, time periods, and rooms, the goal is to assign every exam to exactly one (period, room) pair such that:

- **Hard constraints** (must satisfy): No student sits two exams at the same time, room capacities are respected, and exam durations fit within their assigned periods.
- **Soft constraints** (minimize): Back-to-back exams for the same student, exams spread too closely together, mixed durations in the same room, front-loading penalties, and period/room usage costs.

We benchmark against the [ITC 2007 Examination Track](https://www.eeecs.qub.ac.uk/itc2007/examtrack/) specification and datasets.

## Quick Start

```bash
make                         # build the C++ solver
pip install -r requirements.txt
python main.py --dataset datasets/exam_comp_set4.exam
```

This runs all 7 C++ algorithms on set 4 (273 exams — fast, good for smoke tests). Results go into an auto-created batch directory under `results/`.

For interactive experiments, open `exam_scheduling.ipynb`.

## Datasets

| Set | Exams | Notes |
|-----|------:|-------|
| set4 | 273 | Small, fast — good for quick tests and parameter sweeps |
| set6 | 242 | Smallest set, minimal constraints |
| set8 | 598 | Medium, well-constrained |
| set1 | 607 | Medium, classic benchmark |
| set2 | 870 | Large, low constraint density |
| set3 | 934 | Hardest — 170 period constraints, only HHO achieves feasibility |
| set5 | 1018 | Large, tight room capacity |
| set7 | 1096 | Largest set |

All from the [ITC 2007 Examination Track](https://www.eeecs.qub.ac.uk/itc2007/examtrack/). The synthetic generator also outputs in ITC 2007 format.

## Command Reference

| Flag | Description |
|------|-------------|
| `--dataset FILE` | Run on an ITC 2007 `.exam` file |
| `--algo NAME` | Single algorithm: `greedy`, `tabu`, `hho`, `kempe`, `sa`, `alns`, `gd`, `ip` |
| `--mode MODE` | `demo` (default), `plot`, or `batches` |
| `--size N` | Exam count for synthetic demo mode |
| `--batch "name"` | Create a named batch for results |
| `--load-batch ID` | Write into an existing batch (by ID, name, or partial match) |
| `--no-batch` | Skip batching, write directly to `results/` |
| `--seed N` | Random seed (default: 42) |
| `--quiet` | Suppress progress output |
| `--tabu-iters` | Tabu iterations (default: 2000) |
| `--tabu-patience` | Tabu early-stop patience (default: 500) |
| `--hho-pop` | HHO population size (default: 50) |
| `--hho-iters` | HHO iterations (default: 500) |
| `--sa-iters` | SA iterations (default: 5000) |
| `--kempe-iters` | Kempe iterations (default: 3000) |
| `--alns-iters` | ALNS iterations (default: 2000) |
| `--gd-iters` | Great Deluge iterations (default: 5000) |

The C++ solver can also be called directly: `./cpp/exam_solver <file.exam> [same flags] -v`

## Algorithms

| Algorithm | Language | Description |
|---|---|---|
| **Greedy** | C++ | DSatur graph-coloring heuristic |
| **Tabu Search** | C++ | Feasibility-first local search with swap moves |
| **HHO** | C++ | Harris Hawks population metaheuristic with Levy flights |
| **Kempe Chain** | C++ | Conflict-chain period swaps (preserves feasibility by construction) |
| **Simulated Annealing** | C++ | Geometric cooling with probabilistic acceptance and reheat |
| **ALNS** | C++ | Adaptive destroy-and-repair with operator weight learning |
| **Great Deluge** | C++ | Linearly decaying acceptance level with escape mechanism |
| **IP** | Python | Exact constraint programming via OR-Tools CP-SAT |

All C++ algorithms are called from Python through `cpp_bridge.py`. If the C++ binary isn't compiled, equivalent Python fallbacks run automatically — just slower.

## Project Structure

```
exam_scheduling/
├── main.py                      # CLI entry point
├── exam_scheduling.ipynb        # Interactive experiment notebook
├── Makefile                     # Build C++ solver
├── requirements.txt
│
├── cpp/                         # C++ implementations (headers-only)
│   ├── main.cpp
│   ├── models.h                 # Exam, Period, Room, Solution, EvalResult
│   ├── parser.h                 # ITC 2007 .exam file parser
│   ├── evaluator.h              # Full eval + O(k) incremental move delta
│   ├── greedy.h
│   ├── tabu.h
│   ├── hho.h
│   ├── kempe.h
│   ├── sa.h
│   ├── alns.h
│   └── gd.h
│
├── algorithms/                  # Python implementations + bridge
│   ├── cpp_bridge.py            # Subprocess bridge to C++, auto-fallback
│   ├── ip_solver.py
│   ├── greedy.py
│   ├── tabu_search.py
│   ├── hho.py
│   ├── kempe_chain.py
│   ├── simulated_annealing.py
│   ├── alns.py
│   └── great_deluge.py
│
├── data/
│   ├── models.py                # ProblemInstance, Solution
│   ├── evaluator.py             # Reference evaluator (delegates to fast_eval)
│   ├── fast_eval.py             # Optimized evaluator with O(k) move_delta
│   ├── parser.py                # ITC 2007 format parser
│   └── generator.py             # Synthetic instance generator
│
├── utils/
│   ├── batch_manager.py         # Batch isolation (auto/manual/load previous)
│   ├── results_logger.py        # Structured run logging (JSONL + CSV)
│   ├── plotting.py              # 15 chart types for analysis
│   └── benchmark.py             # Batch benchmarking utilities
│
├── datasets/                    # ITC 2007 sets 1-8 + synthetic instances
└── results/
    └── batch_NNN_<name>/        # Each run in its own batch
        ├── batch_meta.json
        ├── run_log.jsonl
        ├── *.png                # Plots
        └── solutions/           # .sln files
```

## GenAI Usage Disclosure

AI-assisted code generation was used throughout this project. All result testing, experimental design, and technical reporting were done by the team.

## References

1. [ITC 2007 Exam Track — QUB](https://www.eeecs.qub.ac.uk/itc2007/examtrack/)
2. [Addressing Examination Timetabling — MDPI](https://www.mdpi.com/2079-3197/8/2/46)
3. [Tabu Search — Wikipedia](https://en.wikipedia.org/wiki/Tabu_search)
4. [Integer Programming — Wikipedia](https://en.wikipedia.org/wiki/Integer_programming)
5. [Greedy Coloring — Wikipedia](https://en.wikipedia.org/wiki/Greedy_coloring)
6. [Harris Hawks Optimization — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167739X18313530)
7. [Simulated Annealing — Wikipedia](https://en.wikipedia.org/wiki/Simulated_annealing)
8. [Adaptive Large Neighbourhood Search — Ropke & Pisinger (2006)](https://doi.org/10.1016/j.cor.2005.09.018)
9. [Great Deluge Algorithm — Dueck (1993)](https://doi.org/10.1007/BF01096763)
10. [Kempe Chains in Graph Coloring — Wikipedia](https://en.wikipedia.org/wiki/Kempe_chain)
