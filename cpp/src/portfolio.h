/*
 * Parallel portfolio runner — OpenMP.
 *
 * Runs a user-selected set of (algorithm, seed) jobs in parallel across
 * threads and returns the best result (feasibility-first, then lowest soft).
 *
 * Design notes:
 *   • Each OpenMP thread runs an independent solve. No shared state except
 *     the input ProblemInstance (read-only) and the result collector
 *     (mutex-protected).
 *   • Wallclock speedup ≈ min(num_threads, num_jobs) on compute-bound
 *     metaheuristics.
 *   • Build: g++ -fopenmp -std=c++20 ... (see Makefile `exam_solver_portfolio`).
 *
 * Expose via a small CLI: --algo portfolio --portfolio-jobs tabu:42,tabu:1,
 *   sa:42,sa:1,alns:42,alns:1,kempe:42,kempe:1
 * or via direct call: run_portfolio(prob, spec).
 */

#pragma once

#include "models.h"
#include "greedy.h"
#include "tabu.h"
#include "tabu_simd.h"
#include "tabu_cached.h"
#include "sa.h"
#include "sa_cached.h"
#include "alns.h"
#include "alns_cached.h"
#include "alns_thompson.h"
#include "kempe.h"
#include "gd.h"
#include "gd_cached.h"
#include "lahc.h"
#include "lahc_cached.h"
#include "vns.h"
#include "vns_cached.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

struct PortfolioJob {
    std::string algo;   // "tabu", "tabu_simd", "sa", "alns", "kempe", "gd", "lahc", "vns"
    int seed;
};

// Parse spec string "tabu_simd:1,tabu_simd:2,sa:1,alns:1" into job list.
inline std::vector<PortfolioJob> parse_portfolio_spec(const std::string& spec) {
    std::vector<PortfolioJob> jobs;
    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        auto colon = tok.find(':');
        if (colon == std::string::npos) {
            jobs.push_back({tok, 42});
        } else {
            jobs.push_back({tok.substr(0, colon), std::stoi(tok.substr(colon + 1))});
        }
    }
    return jobs;
}

// Default diversified spec, post-batch-19: the proven-winner portfolio.
// 8 jobs — scales to 8-core machine. Increase when more cores available.
//
// Selection rationale (measured on set4/set7, see docs/PERF_ROADMAP.md):
//   tabu_cached  × 2 seeds  — 3.6-12× vs scalar Tabu, identical soft
//   sa_cached    × 2 seeds  — 1.05-1.5× + better soft on both instances
//   alns_thompson × 1 seed  — 1% better soft + 2.5× wall-clock on long runs
//   gd_cached    × 1 seed   — conditional winner on nr>1 instances
//   kempe        × 1 seed   — diversity; often finds the best-of-portfolio
//   lahc_cached  × 1 seed   — late-acceptance diversity
inline std::vector<PortfolioJob> default_portfolio_spec() {
    return {
        {"tabu_cached",   1}, {"tabu_cached",   2},
        {"sa_cached",     1}, {"sa_cached",     2},
        {"alns_thompson", 1},
        {"gd_cached",     1},
        {"kempe",         1},
        {"lahc_cached",   1},
    };
}

// Dispatcher — single-algo run. Extend as more SIMD variants are added.
inline AlgoResult run_single_algo(const ProblemInstance& prob, const PortfolioJob& j,
                                  bool verbose)
{
    if (j.algo == "tabu")        return solve_tabu(prob, 2000, 20, 500, j.seed, verbose);
    if (j.algo == "tabu_simd")   return solve_tabu_simd(prob, 2000, 20, 500, j.seed, verbose);
    if (j.algo == "tabu_cached") return solve_tabu_cached(prob, 2000, 20, 500, j.seed, verbose);
    if (j.algo == "sa")          return solve_sa(prob, 50000, 0.0, 0.9995, j.seed, verbose);
    if (j.algo == "sa_cached")   return solve_sa_cached(prob, 50000, 0.0, 0.9995, j.seed, verbose);
    if (j.algo == "alns")          return solve_alns(prob, 2000, 0.04, j.seed, verbose);
    if (j.algo == "alns_cached")   return solve_alns_cached(prob, 2000, 0.04, j.seed, verbose);
    if (j.algo == "alns_thompson") return solve_alns_thompson(prob, 2000, 0.04, j.seed, verbose);
    if (j.algo == "kempe")         return solve_kempe(prob, 3000, j.seed, verbose);
    if (j.algo == "gd")            return solve_great_deluge(prob, 50000, 0.0, j.seed, verbose);
    if (j.algo == "gd_cached")     return solve_great_deluge_cached(prob, 50000, 0.0, j.seed, verbose);
    if (j.algo == "lahc")          return solve_lahc(prob, 50000, 0, j.seed, verbose);
    if (j.algo == "lahc_cached")   return solve_lahc_cached(prob, 50000, 0, j.seed, verbose);
    if (j.algo == "vns")           return solve_vns(prob, 5000, 0, j.seed, verbose);
    if (j.algo == "vns_cached")    return solve_vns_cached(prob, 5000, 0, j.seed, verbose);
    // Default: DSatur greedy (cheap sanity).
    return solve_greedy(prob, verbose);
}

// Compare: feasibility first, then lower soft, then lower runtime.
inline bool better_result(const AlgoResult& a, const AlgoResult& b) {
    if (a.eval.feasible() != b.eval.feasible()) return a.eval.feasible();
    if (a.eval.hard()     != b.eval.hard())     return a.eval.hard() < b.eval.hard();
    if (a.eval.soft()     != b.eval.soft())     return a.eval.soft() < b.eval.soft();
    return a.runtime_sec < b.runtime_sec;
}

struct PortfolioResult {
    AlgoResult best;
    std::vector<AlgoResult> all;   // per-job, in original job order
    double wallclock_sec;
    int threads_used;
};

inline PortfolioResult run_portfolio(
    const ProblemInstance& prob,
    std::vector<PortfolioJob> jobs,
    bool verbose = false)
{
    PortfolioResult R;
    R.all.resize(jobs.size());
    auto t0 = std::chrono::high_resolution_clock::now();

#ifdef _OPENMP
    R.threads_used = omp_get_max_threads();
#else
    R.threads_used = 1;
#endif

    std::mutex log_mu;

    // OpenMP dynamic schedule — jobs may have different runtimes
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < (int)jobs.size(); i++) {
        auto rr = run_single_algo(prob, jobs[i], false);
        R.all[i] = rr;
        if (verbose) {
            std::lock_guard<std::mutex> lk(log_mu);
            std::cerr << "[portfolio] " << jobs[i].algo << ":" << jobs[i].seed
                      << "  hard=" << rr.eval.hard()
                      << "  soft=" << rr.eval.soft()
                      << "  " << rr.runtime_sec << "s\n";
        }
    }

    // Pick best
    int best_idx = 0;
    for (int i = 1; i < (int)R.all.size(); i++)
        if (better_result(R.all[i], R.all[best_idx])) best_idx = i;
    R.best = R.all[best_idx];

    auto t1 = std::chrono::high_resolution_clock::now();
    R.wallclock_sec = std::chrono::duration<double>(t1 - t0).count();
    return R;
}
