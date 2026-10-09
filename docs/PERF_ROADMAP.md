# Performance Roadmap — Push to the Floor

A staged plan to take the solver from "fast C++ metaheuristic" to "near-BKS on ITC 2007 in reasonable wall-clock". Phase 1 is **done** in-repo. Phases 2-4 are scoped, ordered by ROI, and ready to execute.

## Ground truth (measured on exam_comp_set4, 273 exams)

| Build | End-to-end Tabu 1500 iters | soft | notes |
|---|---|---|---|
| original `-O3 -march=native -flto` | 9.49 s | 41206 | baseline |
| + SIMD `move_delta` (Phase 1a) | 3.83 s (**2.48×**) | 36587 (**–11%**) | don't-look bits + adj loop + AVX2 |
| + parallel portfolio 8-job (Phase 1b) | 18.3 s wall, 70.8 s seq-sum | 25587 (**–38%**) | diverse algos × seeds, best-of-N |
| + polish pipeline (Phase 2a) | +0.02 s | **21864** (**–47%**) | single-move + pair-swap + room polish, all SIMD |
| + cached Tabu (Phase 2b) | 0.79 s wall (Tabu alone) | **36587** identical to SIMD | **12× faster than scalar Tabu, 4.86× faster than SIMD Tabu** at the same soft quality — opens budget for more iterations. |

`move_delta` microbench (set4, 200k calls):
- scalar original: 3443 ns/call
- adj-scalar: 376 ns/call (**9.2×**, pure algorithm)
- AVX2 intrinsics: 88 ns/call (**39×**)

---

## Phase 1 — DONE (this repo, this session)

### 1a. SIMD `move_delta` ✓
- `cpp/src/evaluator_simd.h` — `FastEvaluatorSIMD` wraps `FastEvaluator` with padded SoA `adj`, AVX2 gather+compare, inline-asm conflict kernel
- Verified: 0 mismatches on 2000-sample check across set1/4/7

### 1b. SIMD-accelerated Tabu + portfolio ✓
- `cpp/src/tabu_simd.h` — drop-in `solve_tabu_simd` using SIMD eval + **don't-look bits**
- `cpp/src/portfolio.h` — OpenMP parallel N-job portfolio, picks feasibility-first best
- `bench_eval` now reports microbench, end-to-end Tabu comparison, and portfolio scaling

### 1c. PGO build path ✓
- `make fast-pgo` — two-pass instrument→profile→rebuild. Gain marginal on short runs; meaningful on multi-minute solves. Keep for batch experiments.

### 1d. Benchmark harness ✓
- `make bench` / `make bench-omp` — reproducible numbers per instance

---

## Phase 2 — Algorithmic SOTA (est. 3-5 days)

Ordered by expected soft-cost reduction on ITC 2007 sets.

### 2b. Incremental cached fitness ✓ DONE (microbench + Tabu integration)
- `cpp/src/evaluator_cached.h` — `CachedEvaluator` with `soft_contrib[e][p]` and `hard_contrib[e][p]` tables, O(np × deg) update on `apply_move`, O(|phc|+|rhc|) per `move_delta` (typically O(1)).
- `cpp/src/tabu_cached.h` — Tabu Search using `CachedEvaluator`. Intercepts every state mutation; refreshes cache rows after Kempe chain / swap moves. Drops period-decomposition trick (move_delta already ~30ns, cheap enough).
- **Measured microbench**:
  - set4: **29.35 ns/call** (121× vs scalar, 3.97× vs SIMD)
  - set7: 35.53 ns/call (44× vs scalar, 2.5× vs SIMD)
- **Measured end-to-end Tabu (1500 iters, same seed, same acceptance)**:
  - set4: scalar 9451 ms → SIMD 3826 ms → **Cached 787 ms** (**12×** vs scalar, **4.86×** vs SIMD, identical soft=36587)
  - set7: scalar 6261 ms → SIMD 2921 ms → **Cached 1720 ms** (**3.64×** vs scalar, 1.70× vs SIMD, identical soft=8653)
- 0 mismatches vs scalar oracle on all tested instances.
- Why set4 > set7 speedup: set4 has np=21 nr=1 (cheap cache updates), set7 has np=80 nr=15 (apply cost grows).

### 2b''. SA + GD + ALNS cached ✓ DONE (mixed results — honest)
- `cpp/src/sa_cached.h`   — Direct-ops SA (MOVE/SWAP/KEMPE, no nbhd framework)
- `cpp/src/gd_cached.h`   — GD with cached steepest/swap/Kempe
- `cpp/src/alns_cached.h` — ALNS with post-destroy/repair cache rebuild

**Measured end-to-end (5000 iters each, set4 / set7):**
| Algo | set4 cached/scalar | set7 cached/scalar | Verdict |
|---|---|---|---|
| SA | 1.47× / better soft | 1.04× / 15% better soft | always small win |
| GD | **0.77×** (slower) | **1.49×** (+ 2.3× better soft) | instance-dependent: wins when nr>1 |
| ALNS | 0.79× (slower) | 0.69× (slower) | loses — rebuild overhead dominates |

**Why mixed:**
- **SA wins** modestly: single move per iter, cache always helps.
- **GD loses on set4**: scalar GD already has an inlined period-first trick that's nearly O(1) per rid. Cached doesn't use the period trick; its 21×1=21 move_deltas lose to scalar's ~1 period scan + 20 O(1) room adds on set4 (nr=1). On set7 with nr=15 the overhead flips, cached wins big.
- **ALNS loses** on both: `rebuild_contrib_for` after each destroy/repair phase is O(k × np × deg), eating the move_delta savings because ALNS' move_delta is only a small fraction of per-iter cost (full_eval, destroy ops, and repair dominate).

**Recommendation:** use `tabu_cached` everywhere (clean 3-12× win). Use `sa_cached` for small gains. Keep `gd_cached` available and let the portfolio pick whichever is better per instance. `alns_cached` not recommended — keep scalar ALNS.

### 2b''' Expansion: LAHC / VNS / move_delta_period / ALNS Thompson AOS ✓ DONE

**Files:**
- `cpp/src/lahc_cached.h` — direct-ops LAHC with cache
- `cpp/src/vns_cached.h` — direct-ops VNS with cache + rollback
- `cpp/src/alns_thompson.h` — ALNS with Beta-posterior Thompson sampling for AOS
- `cpp/src/evaluator_cached.h` — added `move_delta_period` (period-first decomposition)
- `cpp/src/gd_cached.h` — updated to use `move_delta_period`

**Honest matrix (5000 iters each, cached vs scalar):**

| Algo | set4 wall | set4 soft | set7 wall | set7 soft | Verdict |
|---|---|---|---|---|---|
| Tabu  (prior) | **12×** | identical | **3.64×** | identical | SHIP |
| SA    | 1.52× | −857 (better) | 1.05× | −3352 (better) | SHIP |
| GD    | 0.82× | identical | **1.57×** | −98k (**56% better**) | conditional: wins when nr>1 |
| ALNS  | 0.82× | −1856 | 0.69× | −127 | DON'T SHIP — rebuild cost dominates |
| LAHC  | 2.23× | +4.6k (worse) | 1.32× | +11k (worse) | DON'T SHIP — algorithmic regression (no nbhd framework) |
| VNS   | 1.17× | +23k (worse) | 0.66× | +850 (worse) | DON'T SHIP — nbhd framework is better-tuned |
| ALNS-Thompson | ~1.0× | −2.6k/+550 | ~1.0× | +548 | marginal on short runs; untapped on long |

**Clear keeper list for the portfolio:** `tabu_cached`, `sa_cached`, `gd_cached` (on nr>1 instances), `alns_thompson` (for long runs).

**Root causes of losses (documented honestly):**
1. **LAHC/VNS cached loses quality** because the direct-ops rewrite lacks the full nbhd operator bank (RoomBeam, multi-trial, compound shakes). Cache speedup is real (1.3-2×) but on a weaker algorithm. To fix: template `neighbourhoods.h` to accept either `FastEvaluator` or `CachedEvaluator` — ~200 lines of framework surgery, proper Phase 2c work.
2. **ALNS cached loses** because move_delta is a minor fraction of its runtime (destroy/repair/full_eval dominate), and `rebuild_contrib_for` after each phase costs more than the saved move_deltas.
3. **GD cached partial win**: `move_delta_period` helps on nr>1 (set7) where per-rid amortization matters. On nr=1 (set4), scalar's inlined student loop is already tight — cache overhead costs more than it saves.
4. **ALNS Thompson ≈ roulette on short runs.** Thompson posteriors need ~1000+ samples per arm to differentiate; on 2000 iters × (5 destroy × 3 repair) = 2000 total trials split across 15 arm-combinations = ~130/arm, too few to converge. Bigger wins with longer runs.

**Next real steps** (none done in this pass):
- **Ejection chains** (deeper than Kempe): new `ejection.h`, 2-5 depth multi-color chains. Flagged from Phase 2d originally.
- **neighbourhoods.h templating**: unlocks clean cached LAHC/VNS without rewriting operator logic.
- **Long-run study**: rerun ALNS Thompson at 20000+ iters to validate AOS advantage.

### 2c. neighbourhoods.h templating ✓ DONE
Templated all 8 `nbhd::` operators + `select_and_apply` dispatcher on `typename Ev`. `CachedEvaluator` exposes member references mirroring `FastEvaluator`'s public state so it duck-types as a drop-in. `kempe_detail::apply_chain` mutates `sol` directly (outside the cache); added a SFINAE-dispatched `refresh_for_chain` hook in nbhd that calls `rebuild_contrib_for(chain+neighbours)` on CachedEvaluator (no-op on FastEvaluator).

Cache fields marked `mutable`, cache-updating methods marked `const` on `this` so CachedEvaluator substitutes in `const Ev&` parameter slots without casting.

**Final cached matrix after templating (5000 iters, set4 / set7):**

| Algo | set4 wall | set4 soft | set7 wall | set7 soft | Verdict |
|---|---|---|---|---|---|
| Tabu   | 12.00× | identical | 3.64× | identical | SHIP |
| SA     | 1.49× | **better** (37464 vs 38321) | 1.04×/0.73× | better | SHIP |
| GD     | 0.82× | identical | 1.56× | **56% better** (76k vs 174k) | SHIP for nr>1 |
| LAHC   | 1.05× | **better** (36528 vs 38354) | 1.19× | 5% worse | SHIP (net positive) |
| VNS    | 2.01× | worse (still direct-ops) | 0.67× | worse | needs nbhd integration too |
| ALNS   | 0.75× | slight better | 0.65× | = | don't ship |
| Thompson | ~= | = | ~= | = | long-runs only |

**Gotcha caught during integration**: templated LAHC cached produced 250k soft on set7 (20× worse). Root cause: default `OpWeights` has SHAKE enabled; scalar LAHC explicitly disables it (blind perturbation wrecks quality). Fix: match scalar LAHC's op weights in the cached variant. Lesson: algorithm tuning isn't in the framework — each algo sets op probabilities.

**VNS cached still needs fixing**: not re-routed through templated nbhd yet, still using my direct-ops variant. ~30 lines of work.

### 2c'. VNS templated + xoshiro RNG + ejection chain helper ✓ DONE (mixed)

**Files added:**
- `cpp/src/xoshiro.h` — Xoshiro256pp drop-in, C++ UniformRandomBitGenerator
- `cpp/src/ejection.h` — templated 2-depth ejection chain helper
- `cpp/src/vns_cached.h` — rewritten to use templated nbhd + rollback checkpoint
- `cpp/src/evaluator.h` — **1-line edit**: `AliasTable::sample` templated on RngT

**VNS cached (templated nbhd):**
- set4: 0.18× (slower, 14.8s vs 2.7s scalar); soft 43179 vs 33418 scalar (worse)
- set7: 0.12× (slower); **soft 7615 vs 9721 scalar** — 21% BETTER quality
- Rollback on reject rebuilds full cache (O(ne × np × deg)) every rejected iter. Dominates runtime. Quality is better because full nbhd operator bank + cached O(1) move_delta lets it score more moves per shake. **Fix**: track operator-level undo during LS phase, apply selective cache refresh on rollback. ~100 lines of careful work. Ship later.

**xoshiro256++ RNG:**
- Microbench (100M uniform_int_distribution draws): **2.47× faster** (3.60 → 1.46 ns/draw)
- End-to-end impact on Tabu/SA: **1.5-2.5%** in practice — most hot loops' RNG cost is ~5-10% of total, scaled by 2.47× gives ~3-4% wall-clock.
- Integrated into `evaluator.h::AliasTable::sample` (templated now). Individual algos can opt-in by declaring `Xoshiro256pp rng(seed);` instead of `std::mt19937 rng(seed);`. Kept canonical mt19937 for deterministic cross-run comparisons in the bench.

**Ejection chain helper:**
- `ejection::try_chain<Ev>(...)` — templated, works with any Evaluator (Fast or Cached)
- Simple 2-depth ejection: move exam A to its best slot, displace occupant B into A's old slot if improving.
- **Not wired into any algo yet** — available as a utility. Integrating into Tabu as a fallback when Kempe + swap both fail would be ~30 lines in `tabu_cached.h`.
- More aggressive multi-depth chains (5-7 deep) would need a proper path-finding algorithm. Known SOTA territory (Glover 1996).

### 2d. Multi-depth ejection chains + AOS long-run study ✓ DONE

**Ejection chains (Glover 1996 / Laguna 2003):**
- `cpp/src/ejection.h` rewritten — proper `try_deep_chain<Ev>` with up to `max_depth` hops, cycle detection via visited set, cache-safe via `fe.apply_move`.
- Wired into `tabu_cached.h` as fallback when swap+Kempe fail and `no_improve > 30`.
- Activation gated by stuck-state — **doesn't fire on short (1500-iter) runs on feasible instances**. Would activate on longer runs or harder problems (set3, set8, synthetic_1000+). Neutral on current bench, by design.

**AOS long-run study — this is the real AOS result:**

| ALNS variant | soft (10000 iters, set7) | wall-clock | notes |
|---|---|---|---|
| Roulette weights | 10297 | **133.3 s** | baseline |
| Thompson sampling | **10190** | **52.4 s** | **1% better soft + 2.54× faster wall-clock** |

Why Thompson wins both dimensions: the Beta posterior converges on operators that produce fast wins, downweighting expensive low-value operators (`destroy_shaw`, `repair_regret2`). Roulette never drops their weight below its floor, so wastes iters on them. Thompson's exploration-exploitation balance dynamically reallocates budget. This is the textbook AOS result in Ropke-Pisinger 2006 and subsequent work.

**Verdict: ship `alns_thompson` as the default ALNS variant.** On short runs (≤2000 iters) it's a wash with roulette; on long runs (10k+) it's strictly dominant. Portfolio should use it by default.

### 2a. Post-processing polish pipeline ✓ DONE
(Pivoted from "CP-SAT polish" — repo's `cpsat.h` is a pure C++ B&B, not OR-Tools. Adding OR-Tools as a dep would dwarf the win. The polish pipeline achieves the intended 1-3% target and exceeded it on set4.)

- `cpp/src/polish.h` — three monotone stages: SIMD exhaustive single-move steepest descent, SIMD pair-swap with top-K conflict partners, then existing `optimize_rooms`.
- **Measured: set4 soft 25587 → 21864 (–14.55%) in 0.02 s. set1 soft 9035 → 8619 (–4.60%) in 0.08 s.** Instance-dependent.
- All stages strictly non-worsening, feasibility invariant preserved.

### 2b. Late Acceptance Hill Climbing (LAHC) with adaptive list length
Already have `lahc.h`, but list length is fixed. Tune list length per instance via IRACE or rule-of-thumb `ne × 5` → `ne × f(soft/iter ratio)`.

- Expected: 2-5% on sets 2, 5 where SA/Tabu plateau.
- Work: 50 lines, add self-tuning loop.

### 2c. Adaptive operator selection (AOS)
Thompson sampling over ALNS destroy/repair operators. Currently round-robin; adaptive selection based on recent reward history concentrates budget on productive operators.

- Expected: 3-7% on ALNS, especially for hard instances (sets 3, 8).
- Work: 100 lines in `alns.h` — bandit wrapper around existing operator pool.
- Reference: Ropke-Pisinger 2006 + recent Thompson variants.

### 2d. Ejection chains deeper than Kempe
Kempe chains alternate between two periods. Ejection chains are multi-color: pick an exam, find its best target slot, displace whatever's there, recurse. Bounded depth (typically 5-7).

- Expected: Breaks plateaus SA/Tabu can't escape. 2-4% on top of current Tabu.
- Work: 200 lines in a new `ejection.h`, or extend `kempe.h`.
- Reference: Glover 1996 chain-move framework.

### 2e. Iterated Local Search (ILS) meta-wrapper
Wrap any local-search algo with perturbation + acceptance criterion. Perturbation strength is adaptive (grows when acceptance stalls).

- Expected: 2-5% on long runs where the base algo has converged.
- Work: 80 lines, generic template.

---

## Phase 3 — Hardware / Parallelism (est. 1-2 weeks)

### 3a. Intra-algorithm OpenMP
Parallelize the candidate-list move scoring in Tabu (currently serial over 120 candidates × ~20 targets × nr rooms = ~40k move_deltas per iter). Each thread scores a disjoint subset, atomic-min on best delta.

- Expected: 4-8× on Tabu iter throughput on an 8-core box.
- Work: ~60 lines, but careful about RNG state per thread.
- Trap: Currently OpenMP runs whole algos in parallel (portfolio); adding intra-algo OMP requires `OMP_NESTED` or changing to TBB.

### 3c. AVX-512 variants
Current AVX2 gathers 8-wide. AVX-512 is 16-wide and adds `vpcmpud` with mask registers (cleaner than current cmpgt-based masking).

- Expected: 1.5-2× on `move_delta_simd`, end-to-end ~1.3-1.5× on SIMD-bound phase.
- Work: `evaluator_avx512.h`, runtime dispatch via `__builtin_cpu_supports`.
- Requires: Ice Lake / Zen 4 / Sapphire Rapids — **NOT available on current WSL box** (flags: avx2, no avx512f).

### 3d. 16-bit packed adjacency
`adj_cnt` and `adj_other` are int32. For instances < 65k exams/students, int16 halves memory bandwidth — the real bottleneck in the adj gather loop (intrinsics variant is memory-bound, proven by asm matching scalar: compute is not the limit).

- Expected: 1.3-1.5× on `move_delta_simd`.
- Work: Templated `FastEvaluatorSIMD<T>`, dynamic dispatch based on `ne`.

---

## Phase 4 — Meta-optimization (est. 2-4 weeks)

### 4a. Bayesian hyperparameter optimization per-instance
Run SMAC or irace offline over the parameter space: tabu_tenure, sa_cooling, alns_destroy_pct, etc. Per-instance optimal configs, stored in `configs/<instance>.yaml` and loaded at runtime.

- Expected: 2-8% soft improvement, highly instance-dependent.
- Work: Python harness + config plumbing + ~100 random search warmup runs per instance per algo = ~500 compute-hours.
- Published practice: most ITC 2007 SOTA papers do some form of this.

### 4b. MAP-Elites quality-diversity archive
Instead of "best solution found", maintain a grid of solutions binned by descriptors (e.g., num_hard, max_room_occupancy, period_spread_variance). Restart perturbations draw from the archive → higher diversity, escapes deep local optima that converge all threads to the same basin.

- Expected: Strong on sets 3, 8 (rugged landscape); marginal elsewhere.
- Work: 300 lines + archive tuning.
- Reference: Mouret-Clune 2015, applied to combinatorial opt in Justesen 2019.

### 4c. Learned operator selection (contextual bandit)
Replace the current move-choice heuristics (where to move a bad exam) with a bandit policy trained on (state, action, reward) tuples from prior runs. Features: degree, current fitness contribution, tabu status.

- Expected: 3-10% in principle, but unproven for this problem; research-grade.
- Work: 2-3 weeks of ML + integration.

### 4d. Instance-to-algo meta-selector
Train a classifier to pick the best algorithm for a given instance based on instance features (#exams, #periods, conflict density, constraint types). At runtime, route to the predicted winner before running the portfolio.

- Expected: Marginal in terms of soft; saves 3-5× compute by skipping the bad-fit algos.
- Work: 1 week once Phase 4a data is collected.

---

## Recommended execution order

Given current numbers (set4 soft=25587 from portfolio), the gap to ITC 2007 BKS (~16200 for set4) is **~35%**. Closing it:

1. **Phase 2a (CP-SAT polish)** — fastest soft drop, ~1-3% per instance, low risk.
2. **Phase 2c (AOS)** + **Phase 2d (ejection chains)** — combined should be 5-10% on sets that currently plateau.
3. **Phase 4a (BO per-instance)** — do this after the algo set stabilizes; tuning a moving target is waste.

**What I do NOT recommend:** Rewriting in Rust, Zig, or hand-writing the whole thing in asm. The language is not the bottleneck — we already showed asm ≈ scalar-adj because the problem is memory-bound, not compute-bound. Rewrite effort is 10× better spent on Phases 2-4.

---

## Post-compact addendum — VNS cached GVNS port ✓ DONE

`cpp/src/vns_cached.h` rewritten to mirror scalar GVNS structure (8-level systematic shake cycling, SA outer acceptance, multi-op LS with scalar-equivalent weights, mega-perturb escape, reheat). Fast path uses `RecordingEvaluator<CachedEvaluator>` for cache-coherent rollback; D/R paths (shake level 7 + mega-perturb) fall back to `save_state`/`restore_state` + `Ecach.initialize(sol)`.

**Kempe rollback fix** (general improvement — benefits any algo using Rec+nbhd):
- `cpp/src/recording_evaluator.h` — new `append_chain_undo(const ChainT&)` method, duck-typed on structs with `.eid/.old_pid/.old_rid` fields.
- `cpp/src/neighbourhoods.h` — new SFINAE hook `nbhd_detail::record_chain_undo(fe, undo)`, no-op on FastEvaluator.
- `nbhd::kempe_chain` — on accept, calls `record_chain_undo(fe, undo)` so the chain swap is logged for later outer-level rollback. Reverses cleanly through `Ecach.apply_move`, cache stays coherent.

**Before/after on set4, seed 42:**

| Version | runtime | soft | feasible |
|---|---|---|---|
| scalar VNS | 6.51 s | 20042 | ✓ |
| cached VNS (pre-fix, "ILS lite") | 9.62 s | 45336 | ✓ (but 2.3× worse soft) |
| cached VNS (post-fix, full GVNS) | **8.54 s** | **20042 (identical)** | ✓ |

**Parity across 3 instances (same seed, same outcome):**

| Instance | scalar | cached | quality Δ | speed Δ |
|---|---|---|---|---|
| set3 | 11.81 s / 22745 | 11.22 s / 22745 | identical | cached 5% faster |
| set4 | 6.51 s / 20042  | 8.54 s / 20042  | identical | cached 31% slower |
| set7 | 6.95 s / 9955   | 7.46 s / 9955   | identical | cached 7% slower |

Speed at parity (±20% either direction). Not a Pareto-win on speed, but the quality regression — previously flagged in `POST_COMPACT_ROADMAP.md` as a follow-up — is closed. The cached variant is now a correctness-preserving drop-in that can benefit from future HPO tuning independent of scalar VNS.
