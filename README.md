# Exam Scheduling

Team: Hoang Le, Ian Cronin

## The Problem

This is a **Capacitated Examination Timetabling Problem** — an NP-hard variant of graph coloring. Given a set of exams, students, time periods, and rooms, the goal is to assign every exam to exactly one (period, room) pair such that:

- **Hard constraints** (must satisfy): No student sits two exams at the same time, room capacities are respected, and exam durations fit within their assigned periods.
- **Soft constraints** (minimize): Back-to-back exams for the same student, exams spread too closely together, mixed durations in the same room, front-loading penalties, and period/room usage costs.

We benchmark against the [ITC 2007 Examination Track](https://www.eeecs.qub.ac.uk/itc2007/examtrack/) specification and datasets.

## Quick Start

The Jupyter notebook (`exam_scheduling.ipynb`) is the easiest way to run experiments interactively. Otherwise, use the CLI:

```bash
# Build the C++ solver
make                    # Linux
# Windows: g++ -O3 -std=c++20 -o cpp/exam_solver.exe cpp/main.cpp

# Install Python dependencies
pip install -r requirements.txt

# Run all algorithms on a dataset
python main.py --dataset datasets/exam_comp_set4.exam

# Run a single algorithm
python main.py --dataset datasets/exam_comp_set4.exam --algo sa
python main.py --dataset datasets/exam_comp_set4.exam --algo ip --limit 100

# Tune parameters
python main.py --dataset datasets/exam_comp_set4.exam --tabu-iters 5000 --sa-iters 10000

# Run the C++ solver directly
./cpp/exam_solver datasets/exam_comp_set4.exam --algo all -v

# Generate and solve a synthetic instance
python main.py --mode demo --size 200
```

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
│   ├── results_logger.py        # Structured run logging (JSONL + CSV)
│   ├── plotting.py              # 8 chart types for analysis
│   └── benchmark.py             # Batch benchmarking utilities
│
├── datasets/                    # ITC 2007 sets 1-8 + synthetic instances
└── results/                     # Solutions, logs, and plots
```

## Datasets

We use the official ITC 2007 Examination Track datasets (sets 1 through 8), available from [QUB](https://www.eeecs.qub.ac.uk/itc2007/examtrack/). The synthetic generator also outputs in ITC 2007 format, so you can mix real and generated instances for testing.

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
