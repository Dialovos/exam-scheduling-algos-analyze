/*
 * HHO+ with batched per-hawk resync via CudaEvaluator.
 *
 * Fully integrated population-GPU variant — structure mirrors solve_hho,
 * but the per-iter "if variation didn't track fitness, re-full_eval"
 * branch is collected across all hawks and dispatched as ONE
 * score_full_batch call. On GPU this becomes one kernel launch per
 * iter; on CPU it loops fe.full_eval (bit-exact with solve_hho).
 *
 * Feasibility is computed via fe.count_hard_fast per resynced hawk
 * (CPU-side, ~100μs each — negligible vs the batched fitness path).
 *
 * CPU fallback: same seed → same result as solve_hho.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_cached.h"
#include "hho.h"
#include "cuda/cuda_evaluator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

inline AlgoResult solve_hho_cuda(
    const ProblemInstance& prob,
    int pop_size         = 20,
    int max_iterations   = 500,
    int seed             = 42,
    bool verbose         = false,
    const Solution* init_sol = nullptr)
{
    using namespace hho_detail;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    pop_size = std::max(pop_size, 5);
    FastEvaluator fe(prob);

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (fe.exam_dur[e] <= fe.period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (fe.exam_enroll[e] <= fe.room_cap[r]) valid_r[e].push_back(r);
    }
    std::vector<bool> is_rhc(ne, false);
    for (int e : fe.rhc_exams) if (e < ne) is_rhc[e] = true;

    std::vector<Hawk> pop(pop_size);
    if (init_sol) pop[0].sol = init_sol->copy();
    else          pop[0].sol = Seeder::seed(prob, seed, false).sol;

    // Init Ecach + CudaEvaluator. Ecach wants an initialized Solution; use pop[0] post-seed.
    CachedEvaluator Ecach(fe);
    Ecach.initialize(pop[0].sol);
    CudaEvaluator Cuev(Ecach);

    {
        auto ev = fe.full_eval(pop[0].sol);
        pop[0].fitness = ev.fitness();
        pop[0].feasible = ev.feasible();
    }
    for (int i = 1; i < pop_size; i++) {
        pop[i].sol = random_solution(prob, fe, ne, np, nr, valid_p, valid_r, rng);
    }

    // BATCH POINT: initial population eval via score_full_batch.
    {
        std::vector<Solution> init_sols;
        init_sols.reserve(pop_size - 1);
        for (int i = 1; i < pop_size; i++) init_sols.push_back(pop[i].sol.copy());
        std::vector<double> init_fit;
        Cuev.score_full_batch(init_sols, init_fit);
        for (int i = 1; i < pop_size; i++) {
            pop[i].fitness = init_fit[i - 1];
            pop[i].feasible = (fe.count_hard_fast(pop[i].sol) == 0);
        }
    }

    int prey_idx = 0;
    for (int i = 1; i < pop_size; i++)
        if (hawk_better(pop[i], pop[prey_idx])) prey_idx = i;

    Solution best_sol = pop[prey_idx].sol.copy();
    double best_fit = pop[prey_idx].fitness;
    bool best_feasible = pop[prey_idx].feasible;

    if (verbose) {
        auto ev = fe.full_eval(best_sol);
        std::cerr << "[HHOCuda] Init pop=" << pop_size
                  << " prey hard=" << ev.hard() << " soft=" << ev.soft()
                  << " gpu=" << (Cuev.gpu_active ? "on" : "off") << "\n";
    }

    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    const int KEMPE_POLISH_EVERY = std::max(25, max_iterations / 20);
    const int SAMPLE_SIZE        = std::min(ne, std::max(10, ne / 6));

    int total_iters = 0;

    // Scratch for per-iter batched resync
    std::vector<int> resync_idx;
    std::vector<Solution> resync_sols;
    std::vector<double> resync_fit;
    resync_idx.reserve(pop_size);
    resync_sols.reserve(pop_size);

    for (int t = 0; t < max_iterations; t++) {
        total_iters++;

        double E0 = unif(rng);
        double E  = 2.0 * E0 * (1.0 - (double)t / (double)max_iterations);

        resync_idx.clear();
        resync_sols.clear();

        for (int i = 0; i < pop_size; i++) {
            if (i == prey_idx) continue;

            Hawk& h = pop[i];
            const Solution& prey = pop[prey_idx].sol;
            double before = h.fitness;

            if (std::abs(E) >= 1.0) {
                int k = std::max(2, (int)std::ceil(std::abs(E) * 5.0));
                levy_perturb(h.sol, fe, valid_p, valid_r, is_rhc, rng, k);
            } else {
                double r = unif01(rng);
                if (r < 0.5 && std::abs(E) >= 0.5) {
                    h.fitness = steepest_single_move(
                        h.sol, fe, h.fitness, ne, valid_p, valid_r, is_rhc,
                        rng, SAMPLE_SIZE);
                } else if (r < 0.5 && std::abs(E) < 0.5) {
                    int k = std::max(3, ne / 40);
                    h.fitness = move_toward_prey(
                        h.sol, prey, h.fitness, fe, ne, valid_r, is_rhc, rng, k);
                } else if (r >= 0.5 && std::abs(E) >= 0.5) {
                    h.fitness = kempe_dive(h.sol, fe, prob, h.fitness, ne, np, rng);
                } else {
                    h.fitness = kempe_dive(h.sol, fe, prob, h.fitness, ne, np, rng);
                    int k = std::max(2, ne / 60);
                    h.fitness = move_toward_prey(
                        h.sol, prey, h.fitness, fe, ne, valid_r, is_rhc, rng, k);
                }
            }

            // Decision: does this hawk need a full resync?
            if (h.fitness == before || std::abs(E) >= 1.0) {
                // Defer to batched resync (collect index + sol snapshot)
                resync_idx.push_back(i);
                resync_sols.push_back(h.sol.copy());
            } else {
                // Fast path — soft already tracked; just count hard
                int hard = fe.count_hard_fast(h.sol);
                h.feasible = (hard == 0);
            }
        }

        // BATCH POINT: one score_full_batch call for all hawks needing resync
        if (!resync_idx.empty()) {
            Cuev.score_full_batch(resync_sols, resync_fit);
            for (size_t k = 0; k < resync_idx.size(); k++) {
                int i = resync_idx[k];
                pop[i].fitness = resync_fit[k];
                pop[i].feasible = (fe.count_hard_fast(pop[i].sol) == 0);
            }
        }

        // Prey update
        int new_prey = prey_idx;
        for (int i = 0; i < pop_size; i++)
            if (hawk_better(pop[i], pop[new_prey])) new_prey = i;
        if (new_prey != prey_idx) {
            prey_idx = new_prey;
            if (hawk_better(pop[prey_idx], {best_sol.copy(), best_fit, best_feasible})) {
                best_sol = pop[prey_idx].sol.copy();
                best_fit = pop[prey_idx].fitness;
                best_feasible = pop[prey_idx].feasible;
                if (verbose && (t < 10 || t % 100 == 0))
                    std::cerr << "[HHOCuda] t=" << t
                              << " new prey hard=" << fe.full_eval(best_sol).hard()
                              << " soft=" << fe.full_eval(best_sol).soft() << "\n";
            }
        }

        // Periodic Kempe polish (same as solve_hho)
        if (t > 0 && t % KEMPE_POLISH_EVERY == 0) {
            Solution polished = pop[prey_idx].sol.copy();
            if (!pop[prey_idx].feasible) {
                polished = Repair::kempe_repair(
                    prob, std::move(polished), /*iters=*/2000, /*restarts=*/1,
                    (uint64_t)seed * 31ull + (uint64_t)t);
            } else {
                std::uniform_int_distribution<int> dp(0, np - 1);
                double cur = fe.full_eval(polished).fitness();
                for (int attempt = 0; attempt < 20; attempt++) {
                    cur = kempe_dive(polished, fe, prob, cur, ne, np, rng);
                }
            }
            auto ev = fe.full_eval(polished);
            double fp = ev.fitness();
            bool feasp = ev.feasible();
            Hawk cand{polished.copy(), fp, feasp};
            if (hawk_better(cand, pop[prey_idx])) {
                pop[prey_idx].sol = std::move(polished);
                pop[prey_idx].fitness = fp;
                pop[prey_idx].feasible = feasp;
                if (hawk_better(pop[prey_idx], {best_sol.copy(), best_fit, best_feasible})) {
                    best_sol = pop[prey_idx].sol.copy();
                    best_fit = fp;
                    best_feasible = feasp;
                }
            }
        }
    }

    auto final_ev = fe.full_eval(best_sol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();

    if (verbose)
        std::cerr << "[HHOCuda] " << total_iters << " iters, " << rt << "s"
                  << " feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard() << " soft=" << final_ev.soft()
                  << " gpu=" << (Cuev.gpu_active ? "on" : "off") << "\n";

    return {std::move(best_sol), final_ev, rt, total_iters, "HHO (CUDA-deep)"};
}
