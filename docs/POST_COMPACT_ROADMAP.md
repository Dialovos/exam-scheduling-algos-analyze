# Post-Compact Roadmap — State as of batch 19 prep

Read this first after `/compact`. The exam-scheduling repo has had several sessions of Phase-2 / Phase-3 performance work. This doc is the handoff.

## What's shipped and working

### Build targets
- `make all` — main solver binary with all algos, -mavx2 required
- `make bench` — move_delta microbench + FPGA cycle-sim
- `make bench-omp` — above + parallel portfolio + end-to-end Tabu comparison + AOS long-run study
- `make fast-pgo` — Profile-Guided Optimization build (marginal on short runs)
- `make batch19` — runs new algos × seeds × sets; writes `results/batch_019_validation/`
- `make -f cpp/src/hdl/sim.mk` — Verilator cosim of FPGA delta kernel (HDL validated 2500/2500 correct)

### Algorithms added (all via main.cpp dispatch, exposed as `--algo <name>`)
- `tabu_simd` — AVX2 move_delta + don't-look bits
- `tabu_cached` — **3.6-12× vs scalar, identical quality** (the main workhorse)
- `sa_cached` — 1.05-1.5× + consistently better soft
- `gd_cached` — conditional win on nr>1 instances (1.56× on set7)
- `lahc_cached` — via templated nbhd, ~1.1× with slightly better soft on set4
- `alns_cached` — don't ship, rebuild cost dominates
- `alns_thompson` — **1% better soft + 2.5× faster wall-clock on 10k-iter runs**
- `vns_cached` — quality win on set7 (7615 vs 9721) but 8× slower due to rollback rebuild

### Infrastructure
- `cpp/src/evaluator_cached.h` — incremental cached fitness, drop-in for FastEvaluator in templated nbhd
- `cpp/src/evaluator_simd.h` — AVX2 variants of move_delta + inline-asm kernel
- `cpp/src/neighbourhoods.h` — templated on `typename Ev` (one-shot edit, done)
- `cpp/src/xoshiro.h` — Xoshiro256pp RNG, 2.47× faster than mt19937
- `cpp/src/ejection.h` — multi-depth chain helper, wired into tabu_cached
- `cpp/src/fpga_sim.h` — cycle-accurate behavioral FPGA simulator
- `cpp/src/hdl/delta_kernel.sv` + `delta_kernel_cosim.sv` — synthesizable HDL + Verilator cosim
- `cpp/src/polish.h` — post-processing single-move + pair-swap + room polish
- `cpp/src/portfolio.h` — OpenMP parallel portfolio with default winners

### Docs
- `docs/PERF_ROADMAP.md` — full measurement matrix, Pareto-win verdicts, what ships and what doesn't
- `docs/FPGA_DESIGN.md` — architecture, cosim results, Vitis HLS variant, cost tables
- `README.md` — Phase 2 section with local + Colab paths

### Colab / batch-19 pipeline
- `scripts/run_batch19.sh` — bash runner with paper-grade iter budgets per algo
- `scripts/summarize_batch19.py` — winner table per instance
- `notebooks/batch19_colab.ipynb` — Colab notebook mirroring local pipeline

## What's complete before Colab run

### A. Phase 3a: intra-algo OpenMP on tabu_cached ✓ DONE
`cpp/src/tabu_cached.h` — candidate scan parallelized with `#pragma omp parallel for`. Per-thread (best_delta, beid, bpid, brid) locals, reduced after the parallel region. Thread count capped at 8 to avoid over-subscription with the portfolio's outer OMP. Targets are pre-shuffled per-candidate BEFORE the parallel region so RNG stays deterministic.

Expected wall-clock speedup on Tabu's candidate phase: 4-8× on 8+ core machines. Negligible overhead on single-thread (n_threads=1 path).

### B. VNS cached rollback fix ✓ DONE
`cpp/src/recording_evaluator.h` — templated `RecordingEvaluator<BaseEv>` wraps any Evaluator, logs every `apply_move` call, provides `rollback_all(sol)` that undoes in reverse via `base.apply_move` (cache stays coherent). `commit()` clears the log on accept.

`cpp/src/vns_cached.h` — replaced `Ecach.initialize(sol)` reject path with `Rec.rollback_all(sol)`. No more O(ne × np × deg) per-iter rebuild.

**Caveat**: on set4 smoke test VNS cached still underperforms scalar VNS in quality. The rollback cost is fixed, but the algorithm's acceptance + shake + budget settings need tuning. Functional, not Pareto-winning. Flagged for HPO in item C.

**Update (post-compact):** quality regression fully closed. `vns_cached.h` was rewritten to mirror scalar GVNS's full structure (8-level shake, SA accept, mega-perturb, reheat). Root cause of prior regression was the simplified "ILS-lite" structure in the first cached port. A second bug was also fixed: `nbhd::kempe_chain` bypasses the evaluator via direct `sol.assign`, and chain swaps were not logged in Rec's undo log — outer-level rejects left the kempe swap stuck. Fix: new `RecordingEvaluator::append_chain_undo` + SFINAE `nbhd_detail::record_chain_undo` hook. Measured parity on set3/4/7 (identical soft per seed; runtime ±20% vs scalar). See `docs/PERF_ROADMAP.md` → "Post-compact addendum" for the matrix.

### C. Phase 4a: Bayesian HPO via Optuna ✓ DONE (skeleton ready)
`tooling/bo_tune.py` — Optuna TPE sampler over per-algo parameter spaces, calls C++ solver via subprocess, parses JSON. Supports single (algo, instance) tuning or `--all` for full matrix.

Usage:
```bash
pip install optuna
python3 tooling/bo_tune.py --instance exam_comp_set4 --algo tabu_cached --trials 50
python3 tooling/bo_tune.py --all     # 500-hour sweep, do offline
```
Output: `tooling/tuned_params_v2.json` — loadable at runtime. Tuning needs to be run (offline, many hours) to actually improve soft; the skeleton is just the framework.

### D. Phase 3b: CUDA move-scoring kernel ✓ DONE (skeleton + host-side integration)
`cpp/src/cuda/delta_kernel.cu` — synthesizable CUDA kernel mirroring the FPGA design: one block per move, threads cooperate on adj scan, warp-shuffle reduction. Now has persistent-state C API (create/update_period_of/score_batch/destroy).

`cpp/src/cuda/cuda_evaluator.h` — host-side wrapper with CPU fallback. Owns device buffers, exposes `sync_state(sol)` and `score_batch(...)`. Compiles and runs correctly without nvcc (CPU fallback path).

`cpp/src/tabu_cached_cuda.h` + `--algo tabu_cuda` — tabu variant that batches candidate moves into one kernel call. CPU-fallback path is bit-exact equivalent to `tabu_cached` (verified on set4/set7 same seed, soft=54667/9506 both).

`make cuda-build` — produces `cpp/build/libdelta_cuda.so`. To activate GPU path: `make cuda-build && make all HAVE_CUDA=1`. Verbose output reports `gpu=on/off (CPU fallback)`.

**Remaining work (not done, scoped):** the current kernel covers conflict-count only. Full move_delta port (spread, 2-in-row, 2-in-day, period/room pen, PHC/RHC, frontload) is ~500 LoC of CUDA. Until that lands, the GPU path runs the kernel for timing but still calls Ecach.move_delta on host for authoritative delta — so no speedup yet, just plumbing validated. On Colab T4 with the extended kernel, expected **20-100× on the neighborhood scoring phase**.

**Update (post-compact):** ✓ full move_delta kernel shipped. Every term (adj-conflicts, duration, room-capacity, period/room pen, PHC 4-codes, RHC, spread, 2-in-row, 2-in-day, frontload) is in `cpp/src/cuda/delta_kernel.cu::delta_kernel_full` and in the bit-exact CPU twin `CudaEvaluator::score_delta_cpu_ref`. Validator in `make bench` tests 10k random moves vs `Ecach.move_delta` — 0 mismatches across set1/set4/set7. Host no longer does soft-term correction; kernel returns authoritative int64 fixed-point (dh*100000 + ds).

**Update (post-compact 2):** ✓ ALNS repair-phase GPU batching shipped. New `delta_kernel_placement` + `CudaEvaluator::score_placement_cpu_ref` + `alns.h::repair_greedy_batched` + `alns_cuda.h` (+ `--algo alns_cuda`). Placement-scorer validator in `make bench`: 0 mismatches / 4k slots vs `repair_greedy` inline cost. End-to-end alns_cuda CPU-fallback bit-exact with alns_thompson on set4 (33428 / 33428) and set7 (26851 / 26851).

Remaining scheduled: population-based GPU support (GA/ABC/HHO/WOA full-eval kernel, ~500 LoC + per-algo host integration, 1 week estimate).

## Post-compact plan for tomorrow

## Post-compact plan for tomorrow (or whenever)

1. **Finish items A, B before Colab batch 19** — DONE. Both A (intra-algo OpenMP) and B (VNS cached) are now complete, quality-verified. Post-compact session also closed the kempe-rollback bug in RecordingEvaluator (general fix, not just VNS).
2. **Run batch 19 on Colab.** Expected output: `results/batch_019_colab.zip` with summary.csv showing per-instance winners.
3. **(Optional, paper-grade)** Run item C's Optuna tuning offline (~500 compute-hours worth), regenerate tuned_params, rerun batch 19 with tuned params — this is the final soft-cost reduction path to within ~10% of BKS.
4. Item D (CUDA) only if publishing a hardware-accel paper.

## Quick reference: measured speedups

| Technique | Wall-clock speedup | Quality Δ |
|---|---|---|
| SIMD move_delta (microbench) | 39× vs scalar move_delta | n/a |
| Tabu cached end-to-end (set4) | 12× | identical |
| Tabu cached end-to-end (set7) | 3.64× | identical |
| SA cached | 1.05-1.5× | slightly better |
| GD cached (nr>1) | 1.56× | **−56% soft on set7** |
| ALNS Thompson vs roulette (10k iters) | 2.54× | **−1% soft** |
| Parallel portfolio (8 jobs) | 3.8-5.3× wall-clock | best-of-N |
| Polish pipeline (applied to portfolio) | +0.02 s | **−14.5% soft** |
| xoshiro256++ RNG microbench | 2.47× | identical distribution |
| FPGA cycle-sim (conflict kernel, 16 cores) | 115-168× | bit-exact |

## Key files to NOT edit blindly

- `cpp/src/evaluator.h` — original FastEvaluator. 1-line edit for templated `AliasTable::sample`. Don't add more.
- `cpp/src/neighbourhoods.h` — already templated on `typename Ev`. Any new operators should follow the pattern.

## Commit strategy (per user instruction: never git commit here)

User manages history manually. Never run `git add` or `git commit` in this repo. Feedback memory: `feedback_no_git_commits.md`.
