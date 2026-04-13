<p align="center">
  <img src="graphs/algo_radar.png" width="420"/>
</p>

<h1 align="center">Exam Scheduling</h1>

<p align="center">
  Metaheuristic comparison on the ITC 2007 Capacitated Examination Timetabling Problem
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=cplusplus" alt="C++20"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/Benchmark-ITC%202007-8B6914?style=flat-square" alt="ITC 2007"/>
  <img src="https://img.shields.io/badge/Algorithms-12-2E8B57?style=flat-square" alt="12 Algorithms"/>
</p>

<br/>

Twelve optimization algorithms — constructive, local search, population-based, and exact — compiled into a single C++ solver and benchmarked on all eight [ITC 2007](https://www.eeecs.qub.ac.uk/itc2007/examtrack/) datasets. Python fallbacks available when the binary is unavailable.

> **Contributors** &ensp; Hoang Le &ensp;·&ensp; Ian Cronin

---

## Problem

The Capacitated Examination Timetabling Problem is NP-hard (graph-coloring variant). Given exams, students, time slots, and rooms with limited seating:

- Assign every exam to exactly one (period, room) pair
- Hard constraints: no student sits two exams in the same period; room capacity respected; exam fits the period length
- Soft constraints: penalize back-to-back exams, same-day exams, exams too close together, mixed-duration rooms, and large exams late in the schedule
- Fitness: `hard_violations * 100000 + soft_penalty` (feasibility-first)

Exams are vertices, edges connect any pair sharing at least one student. No polynomial-time algorithm is known — everything here is heuristic or metaheuristic.

## Algorithms

| # | Algorithm | Type | Description |
|:---:|-----------|------|-------------|
| 1 | Greedy | Constructive | DSatur graph-coloring heuristic |
| 2 | Tabu Search | Local search | Feasibility-first with swap + room-only moves |
| 3 | Kempe Chain | Local search | Conflict-chain period swaps with SA acceptance |
| 4 | Simulated Annealing | Local search | Multi-neighbourhood with geometric cooling and reheat |
| 5 | ALNS | Hybrid | Adaptive destroy-and-repair with proximity-aware operators |
| 6 | Great Deluge | Local search | Linearly decaying acceptance level + swap moves |
| 7 | ABC | Swarm | Artificial Bee Colony with cost-weighted multi-move bees |
| 8 | GA | Evolutionary | Memetic GA: Kempe mutation + saturation-degree crossover |
| 9 | LAHC | Local search | Late Acceptance Hill Climbing with history list |
| 10 | WOA | Swarm | Whale Optimization with spiral + encircling |
| 11 | CP-SAT | Exact | Constraint programming via OR-Tools CP-SAT |
| 12 | GVNS | Hybrid | General Variable Neighbourhood Search with SA acceptance |

Delta evaluation (`move_delta()`, O(k) per move instead of O(n^2) full eval) drives every local search. Swap moves expand the neighbourhood by exchanging periods of two exams at once. A steepest-descent room post-processing pass runs on every final solution. Warm-start chaining (`--init-solution`) pipes one algorithm's output into the next.

## Datasets

| Set | Exams | Notes |
|:---:|------:|-------|
| set4 | 273 | Small, fast — good for quick tests and parameter sweeps |
| set6 | 242 | Smallest set, minimal constraints |
| set8 | 598 | Medium, well-constrained |
| set1 | 607 | Medium, classic benchmark |
| set2 | 870 | Large, low constraint density |
| set3 | 934 | Hardest — dense period constraints |
| set5 | 1018 | Large, tight room capacity |
| set7 | 1096 | Largest set |

All sourced from the [ITC 2007 Examination Track](https://www.eeecs.qub.ac.uk/itc2007/examtrack/). A synthetic generator writes ITC 2007 format for scalability testing.

---

## Results

### Aggregate performance

<table>
<tr>
<td width="50%">
<p align="center"><b>Soft penalty · Runtime · Memory</b></p>
<img src="graphs/algo_bars.png" width="100%"/>
<p align="center"><sub>Mean with error bars across all eight datasets</sub></p>
</td>
<td width="50%">
<p align="center"><b>Soft penalty distribution</b></p>
<img src="graphs/algo_boxes.png" width="100%"/>
<p align="center"><sub>Box plot spread across datasets per algorithm</sub></p>
</td>
</tr>
</table>

### Multi-dimensional view

<table>
<tr>
<td width="50%">
<p align="center"><b>Per-dataset heatmap</b></p>
<img src="graphs/algo_heatmap.png" width="100%"/>
<p align="center"><sub>Normalized soft penalty, algorithms x datasets. Cell values are actual penalties.</sub></p>
</td>
<td width="50%">
<p align="center"><b>Performance profile</b></p>
<img src="graphs/algo_radar.png" width="100%"/>
<p align="center"><sub>Memory, runtime, soft penalty, and constraint components. Smaller = better.</sub></p>
</td>
</tr>
</table>

### Quality vs cost

<table>
<tr>
<td width="50%">
<p align="center"><b>Runtime vs soft penalty</b></p>
<img src="graphs/algo_scatter.png" width="100%"/>
<p align="center"><sub>Bottom-left is the sweet spot</sub></p>
</td>
<td width="50%">
<p align="center"><b>Per-dataset trends</b></p>
<img src="graphs/summary_lines.png" width="100%"/>
<p align="center"><sub>Soft penalty, runtime, and peak memory across all sets</sub></p>
</td>
</tr>
</table>

### Scalability

<p align="center">
  <img src="graphs/scan_smoke.png" width="75%"/>
</p>
<p align="center"><sub>Chain(SA, Kempe, GD) quality and cost vs synthetic instance size</sub></p>

---

## Usage

### Prerequisites

- C++ compiler with C++20 support (g++ recommended)
- Python 3.10+
- pip packages: see `requirements.txt`

### Quick start

```bash
# Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build the C++ solver
make

# Run all algorithms on a dataset
python3 main.py --dataset instances/exam_comp_set4.exam

# Or open the notebook
jupyter notebook exam_scheduling.ipynb
```

### CLI examples

```bash
# Single algorithm
python3 main.py --dataset instances/exam_comp_set4.exam --algo sa

# Multiple algorithms
python3 main.py --dataset instances/exam_comp_set4.exam --algo sa,gd,vns

# Custom parameters
python3 main.py --dataset instances/exam_comp_set1.exam --sa-iters 10000 --seed 123

# Auto-tune across all datasets
python3 main.py --mode tune

# View active tuned parameters
python3 main.py --show-params
```

### Direct C++ usage

```bash
./cpp/build/exam_solver instances/exam_comp_set4.exam --algo all -v
```

<details>
<summary>Full flag reference</summary>
<br/>

| Flag | Description |
|------|-------------|
| `--dataset FILE` | ITC 2007 `.exam` file |
| `--algo NAME` | `greedy`, `tabu`, `kempe`, `sa`, `alns`, `gd`, `abc`, `ga`, `lahc`, `woa`, `cpsat`, `vns` |
| `--mode MODE` | `demo` (default), `plot`, `batches`, `tune` |
| `--size N` | Exam count for synthetic demo mode |
| `--seed N` | Random seed (default: 42) |
| `--tabu-iters` | Tabu iterations |
| `--sa-iters` | SA iterations |
| `--kempe-iters` | Kempe iterations |
| `--alns-iters` | ALNS iterations |
| `--gd-iters` | Great Deluge iterations |
| `--abc-pop` / `--abc-iters` | ABC colony size / iterations |
| `--ga-pop` / `--ga-iters` | GA population / generations |
| `--lahc-iters` / `--lahc-list` | LAHC iterations / history list length (0 = auto) |
| `--woa-pop` / `--woa-iters` | WOA population / iterations |
| `--cpsat-time` | CP-SAT time limit in seconds |
| `--vns-iters` / `--vns-budget` | GVNS iterations / scan budget per LS call (0 = auto) |
| `--show-params` | Print active param defaults and exit |
| `--rollback-params V` | Rollback tuned params to version V and exit |

</details>

## Auto-tuner

Automated parameter optimization and algorithm-chain discovery. Supports single-dataset tuning or global multi-dataset mode to avoid overfitting.

```bash
# Single dataset
python3 -m tooling.auto_tuner instances/exam_comp_set4.exam

# Global — all ITC 2007 sets (recommended)
python3 -m tooling.auto_tuner --all-sets
python3 -m tooling.auto_tuner --all-sets --max-time 20    # 20 min budget
python3 -m tooling.auto_tuner --all-sets --resume          # resume from checkpoint
```

Runs in phases: quick screen, parameter tuning, chain discovery, final validation. Winning parameters are saved to `tooling/tuned_params.json` with version history for rollback.

<details>
<summary>Project structure</summary>
<br/>

```
exam-scheduling/
├── README.md
├── Makefile
├── requirements.txt
├── main.py
├── exam_scheduling.ipynb
│
├── core/
│   ├── models.py
│   ├── parser.py
│   ├── generator.py
│   ├── fast_eval.py
│   └── evaluator.py
│
├── algorithms/
│   ├── cpp_bridge.py
│   ├── ip_solver.py
│   ├── greedy.py
│   ├── tabu_search.py
│   ├── kempe_chain.py
│   ├── simulated_annealing.py
│   ├── alns.py
│   ├── great_deluge.py
│   ├── abc.py
│   └── ga.py
│
├── cpp/
│   └── src/
│       ├── main.cpp
│       ├── models.h, parser.h, evaluator.h, greedy.h
│       ├── neighbourhoods.h
│       ├── tabu.h, kempe.h, sa.h, alns.h, gd.h
│       ├── abc.h, ga.h, lahc.h, woa.h
│       ├── cpsat.h, vns.h, feasibility.h
│       └── archive/
│
├── tooling/
│   ├── auto_tuner.py
│   ├── tuned_params.py
│   └── tuned_params.json
│
├── utils/
│   ├── batch_manager.py
│   ├── results_logger.py
│   └── plotting.py
│
├── instances/
├── results/
├── graphs/
├── report/
├── references/
└── tests/
```

</details>

## GenAI usage disclosure

AI-assisted coding (Claude) was used throughout development for algorithm implementation, debugging, and code refactoring. Experimental design, benchmarking methodology, parameter choices, and technical writing were done by the human authors.

## References

See [`references/references.md`](references/references.md) for the full annotated bibliography.

- [ITC 2007 Examination Track](https://www.eeecs.qub.ac.uk/itc2007/examtrack/) — benchmark datasets
- [Burke & Bykov (2008)](https://doi.org/10.1007/978-3-540-89439-1_26) — FastSA-ETP
- [Ropke & Pisinger (2006)](https://doi.org/10.1016/j.cor.2005.09.018) — ALNS
- [Hansen et al. (2010)](https://doi.org/10.1016/j.ejor.2008.10.012) — GVNS
- [Mirjalili & Lewis (2016)](https://doi.org/10.1016/j.advengsoft.2016.01.008) — WOA
- [Kirkpatrick et al. (1983)](https://doi.org/10.1126/science.220.4598.671) — Simulated Annealing
