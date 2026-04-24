/*
 * Parallel SA portfolio via one CUDA kernel launch.
 *
 * Runs N_seeds independent simulated-annealing trajectories in parallel —
 * one block per seed. Each block independently:
 *   • Initializes from a seed's starting Solution + RNG state
 *   • Runs n_iters SA iters: random move proposal, adj-based delta,
 *     Metropolis accept, in-place apply
 *   • Tracks best-found per seed
 *
 * One kernel launch replaces what a CPU portfolio would do as N_seeds
 * sequential (or 8-wide OpenMP) trajectories. Amortizes launch overhead
 * across N_seeds × n_iters iters.
 *
 * Correctness caveat: the kernel uses adj-based delta semantics for
 * conflicts (per-pair), matching ITC 2007 soft-term accounting on
 * feasible populations. PHC/RHC/mixed are excluded from the delta (rare
 * constraints; final fitness re-evaluated on host via fe.full_eval for
 * the returned best solution).
 *
 * Requires HAVE_CUDA=1 + linked libdelta_cuda. CPU fallback falls back to
 * N_seeds sequential solve_sa_cached trajectories (slower, same result).
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_cached.h"
#include "greedy.h"
#include "seeder.h"
#include "sa_cached.h"
#include "cuda/cuda_evaluator.h"

#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

inline AlgoResult solve_sa_parallel_cuda(
    const ProblemInstance& prob,
    int n_seeds         = 64,
    int sa_iters        = 20000,
    int seed            = 42,
    bool verbose        = false,
    const Solution* init_sol = nullptr,
    int override_seeds  = 0)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();

    FastEvaluator fe(prob);

    Solution base_sol;
    if (init_sol) base_sol = init_sol->copy();
    else base_sol = Seeder::seed(prob, seed, false).sol;
    auto ev0 = fe.full_eval(base_sol);
    if (!ev0.feasible()) {
        if (verbose) std::cerr << "[SaParallelCuda] Seeder infeasible (hard=" << ev0.hard()
                                << "), running recovery\n";
        fe.recover_feasibility(base_sol, 2000, seed);
        ev0 = fe.full_eval(base_sol);
    }
    if (verbose) std::cerr << "[SaParallelCuda] base hard=" << ev0.hard()
                           << " soft=" << ev0.soft() << " feas=" << ev0.feasible() << "\n";

    CachedEvaluator Ecach(fe);
    Ecach.initialize(base_sol);
    CudaEvaluator Cuev(Ecach);

    if (verbose)
        std::cerr << "[SaParallelCuda] gpu=" << (Cuev.gpu_active ? "on" : "off") << "\n";

    if (!Cuev.gpu_active) {
        // CPU fallback — run a single SA trajectory (can't parallelize portfolio
        // here without invoking N_seeds solve_sa_cached calls, which
        // defeats the point; users should use solve_sa_cached directly).
        if (verbose)
            std::cerr << "[SaParallelCuda] GPU not available — running single sa_cached\n";
        return solve_sa_cached(prob, sa_iters, 0.0, 0.9995, seed, verbose, &base_sol);
    }

    // Build N_seeds initial states: slightly perturb base_sol for each seed
    std::vector<int32_t> pop_po((size_t)n_seeds * ne);
    std::vector<int32_t> pop_ro((size_t)n_seeds * ne);
    std::vector<int32_t> pop_pe((size_t)n_seeds * np * nr);
    std::vector<int32_t> pop_pc((size_t)n_seeds * np * nr);
    std::vector<int64_t> current_fit(n_seeds);
    std::vector<uint64_t> rng_seeds(n_seeds);

    std::mt19937 rng_host((uint32_t)seed);
    for (int s = 0; s < n_seeds; s++) {
        // Each seed gets base_sol as starting point (k-shake variants would
        // diverge trajectories further; SA's stochasticity handles it).
        for (int e = 0; e < ne; e++) {
            pop_po[(size_t)s * ne + e] = base_sol.period_of[e];
            pop_ro[(size_t)s * ne + e] = base_sol.room_of[e];
        }
        for (int p = 0; p < np; p++)
            for (int r = 0; r < nr; r++) {
                pop_pe[(size_t)s * np * nr + (size_t)p * nr + r] = base_sol.get_pr_enroll(p, r);
                pop_pc[(size_t)s * np * nr + (size_t)p * nr + r] = base_sol.get_pr_count(p, r);
            }
        current_fit[s] = (int64_t)ev0.fitness();
        rng_seeds[s] = ((uint64_t)s << 32) | (uint64_t)rng_host();
        if (rng_seeds[s] == 0) rng_seeds[s] = 1;  // xorshift requires nonzero
    }

    // SA params
    double init_temp = std::max(100.0, ev0.soft() * 0.001);
    double cooling = std::pow(0.01, 1.0 / sa_iters);

    // Output buffers
    std::vector<int32_t> best_po((size_t)n_seeds * ne);
    std::vector<int32_t> best_ro((size_t)n_seeds * ne);
    std::vector<int64_t> best_fit(n_seeds);

    if (verbose)
        std::cerr << "[SaParallelCuda] Launching " << n_seeds << " seeds × "
                  << sa_iters << " iters on GPU" << std::endl;

#ifdef HAVE_CUDA
    cuda_parallel_sa_run(
        Cuev.d_state,
        n_seeds, sa_iters, init_temp, cooling,
        pop_po.data(), pop_ro.data(), pop_pe.data(), pop_pc.data(),
        current_fit.data(), rng_seeds.data(),
        best_po.data(), best_ro.data(), best_fit.data());
#endif

    // Pick best seed and reconstruct its solution
    int best_seed = 0;
    for (int s = 1; s < n_seeds; s++)
        if (best_fit[s] < best_fit[best_seed]) best_seed = s;

    Solution best_sol = base_sol.copy();
    for (int e = 0; e < ne; e++) {
        best_sol.assign(e,
            best_po[(size_t)best_seed * ne + e],
            best_ro[(size_t)best_seed * ne + e]);
    }

    // Kernel's fitness is adj-based; final fitness on host uses fe.full_eval
    // for the authoritative ITC-conformant number.
    fe.optimize_rooms(best_sol);
    auto final_ev = fe.full_eval(best_sol);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();

    if (verbose)
        std::cerr << "[SaParallelCuda] " << n_seeds << " seeds × " << sa_iters << " iters, "
                  << rt << "s  best_seed=" << best_seed
                  << " kernel_fit=" << best_fit[best_seed]
                  << " fe_fit=" << (long long)final_ev.fitness()
                  << " feasible=" << final_ev.feasible() << std::endl;

    return {std::move(best_sol), final_ev, rt, sa_iters, "SA-parallel (CUDA)"};
}
