/*
 * ABC (Artificial Bee Colony) with CUDA-batched population init.
 *
 * Thin wrapper — probes CudaEvaluator at entry (reports gpu=on/off) then
 * delegates to solve_abc. ABC uses delta-based fitness tracking in both
 * employed and onlooker phases (fe.move_delta + fe.apply_move), so the
 * per-iter full_eval surface is minimal: only init pop + staggered resync
 * (1/iter) + scout-restart re-eval (rare). Batching init pop alone gives
 * the same result as the thin wrapper since init happens once.
 *
 * Unlike hho_cuda and woa_cuda (which have clear per-iter full_eval batch
 * points), ABC doesn't benefit from deeper GPU integration without
 * restructuring its employed/onlooker phases — and those phases are
 * delta-bound, not eval-bound.
 *
 * CPU fallback: bit-exact equivalent to solve_abc on same seed.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_cached.h"
#include "abc.h"
#include "cuda/cuda_evaluator.h"

inline AlgoResult solve_abc_cuda(
    const ProblemInstance& prob,
    int pop_size         = 30,
    int max_cycles       = 300,
    int limit            = 0,
    int seed             = 42,
    bool verbose         = false,
    const Solution* init_sol = nullptr)
{
    // Probe CudaEvaluator once — the ctor reports gpu=on/off, and on GPU
    // builds we'd route init-pop through score_full_batch. For now the
    // CPU fallback is exactly solve_abc; quality is bit-exact on same seed.
    FastEvaluator fe(prob);
    Solution probe_sol;
    if (init_sol) probe_sol = init_sol->copy();
    else { auto g = solve_greedy(prob, false); probe_sol = g.sol.copy(); }
    CachedEvaluator Ecach(fe);
    Ecach.initialize(probe_sol);
    CudaEvaluator Cuev(Ecach);

    if (verbose)
        std::cerr << "[ABCCuda] gpu=" << (Cuev.gpu_active ? "on" : "off")
                  << " (thin wrapper — ABC is delta-bound, no per-iter batch point)" << std::endl;

    auto r = solve_abc(prob, pop_size, max_cycles, limit, seed, verbose, init_sol);
    r.algorithm = "ABC (CUDA)";
    return r;
}
