# Code Layout

Per CSC 2400 rubric this folder documents the code organization. Source
lives in purpose-named directories at the repo root (not in a `/code/`
subdirectory) — modern Python/C++ projects colocate source with configs
and tests.

## Python modules
- `algorithms/` — Python entry points and C++ bridge adapter.
- `core/` — data structures (`Solution`, `Problem`, `EvalBreakdown`), parser,
  evaluator, synthetic generator.
- `utils/plots/` — plotting package split into themed submodules
  (shared, comparative, convergence, breakdown, tuning).
- `utils/batch_manager.py`, `utils/results_logger.py` — batch orchestration
  and per-run JSON/CSV logging.
- `tooling/` — `tuner/` package (auto-tuner) and `tuning_export.py`.
- `main.py` — CLI entry for single-run and batch modes.

## C++ source (headers-only solver)
- `cpp/src/main.cpp` — command parser and algorithm dispatcher.
- `cpp/src/` headers — one per algorithm (`tabu.h`, `sa.h`, `alns.h`, `gd.h`,
  `abc.h`, `ga.h`, `lahc.h`, `woa.h`, `cpsat.h`, `vns.h`, `kempe.h`,
  `greedy.h`, `hho.h`) plus shared infrastructure (`models.h`, `parser.h`,
  `evaluator.h`, `neighbourhoods.h`, `seeder.h`, `repair.h`).
- `cpp/build/exam_solver` — compiled binary (built by `make`).

## Data and outputs
- `instances/` — ITC 2007 public sets 1–8 and synthetic generator outputs.
- `results/` — per-batch runs (gitignored except `results/best/`).
- `graphs/` — final figures rendered for the paper.

## How to run
- Build: `make` (requires C++20).
- Single run: `python main.py --dataset instances/exam_comp_set1.exam --algo tabu`.
- Full batch (Colab): open `colab_runner.ipynb`.
