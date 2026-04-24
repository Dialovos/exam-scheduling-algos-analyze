/*
 * WOA with batched population eval via CudaEvaluator.
 *
 * Two batch points:
 *   1. Initial population (population_size full_evals → 1 call)
 *   2. Periodic resync every 100 iters (population_size full_evals → 1 call)
 *
 * Matches solve_woa on CPU fallback (same seed → same result).
 * On GPU build, fitness for the resync loop goes through score_full_batch;
 * minor semantic drift on infeasible whales (adj vs per-student conflict
 * counting) — documented in PERF_ROADMAP.md.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_cached.h"
#include "woa.h"  // reuse woa_detail::vnd, random_solution, Whale
#include "cuda/cuda_evaluator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

inline AlgoResult solve_woa_cuda(
    const ProblemInstance& prob,
    int population_size = 25,
    int max_iterations  = 3000,
    int seed            = 42,
    bool verbose        = false,
    const Solution* init_sol = nullptr)
{
    using namespace woa_detail;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    population_size = std::max(population_size, 5);
    FastEvaluator fe(prob);

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (fe.exam_dur[e] <= fe.period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (fe.exam_enroll[e] <= fe.room_cap[r]) valid_r[e].push_back(r);
    }

    std::vector<Whale> pop(population_size);
    if (init_sol) { pop[0].sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); pop[0].sol = g.sol.copy(); }
    auto ev0 = fe.full_eval(pop[0].sol);
    if (!ev0.feasible()) {
        fe.recover_feasibility(pop[0].sol, 500, seed);
        ev0 = fe.full_eval(pop[0].sol);
    }
    pop[0].fitness = ev0.fitness();
    pop[0].feasible = ev0.feasible();

    for (int i = 1; i < population_size; i++) {
        pop[i].sol = random_solution(prob, fe, ne, np, nr, valid_p, valid_r, rng);
    }

    CachedEvaluator Ecach(fe);
    Ecach.initialize(pop[0].sol);
    CudaEvaluator Cuev(Ecach);

    // BATCH POINT 1: initial population eval
    {
        std::vector<Solution> init_sols;
        init_sols.reserve(population_size - 1);
        for (int i = 1; i < population_size; i++) init_sols.push_back(pop[i].sol.copy());
        std::vector<double> init_fit;
        Cuev.score_full_batch(init_sols, init_fit);
        for (int i = 1; i < population_size; i++) {
            pop[i].fitness = init_fit[i - 1];
            pop[i].feasible = (fe.count_hard_fast(pop[i].sol) == 0);
        }
    }

    int leader_idx = 0;
    for (int i = 1; i < population_size; i++) {
        bool i_better = false;
        if (pop[i].feasible && !pop[leader_idx].feasible) i_better = true;
        else if (pop[i].feasible == pop[leader_idx].feasible && pop[i].fitness < pop[leader_idx].fitness)
            i_better = true;
        if (i_better) leader_idx = i;
    }

    Solution best_sol = pop[leader_idx].sol.copy();
    double best_fitness = pop[leader_idx].fitness;
    bool best_feasible = pop[leader_idx].feasible;

    if (verbose)
        std::cerr << "[WOACuda] Init pop=" << population_size
                  << " best fit=" << (long long)best_fitness
                  << " gpu=" << (Cuev.gpu_active ? "on" : "off") << std::endl;

    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::uniform_int_distribution<int> de(0, ne - 1);

    std::vector<std::vector<double>> whale_costs(population_size, std::vector<double>(ne, 1.0));
    std::vector<AliasTable> whale_alias(population_size);
    std::vector<bool> alias_dirty(population_size, true);

    auto recompute_alias = [&](int i) {
        for (int e = 0; e < ne; e++) {
            int pid = pop[i].sol.period_of[e]; if (pid < 0) { whale_costs[i][e] = 1.0; continue; }
            double c = 1.0;
            for (auto& [nb, _] : prob.adj[e]) {
                int np2 = pop[i].sol.period_of[nb]; if (np2 < 0) continue;
                if (np2 == pid) c += 100000;
                int gap = std::abs(pid - np2);
                if (gap > 0 && gap <= fe.w_spread) c += 1;
                if (fe.period_day[pid] == fe.period_day[np2]) {
                    int g = std::abs(fe.period_daypos[pid] - fe.period_daypos[np2]);
                    if (g == 1) c += fe.w_2row;
                    else if (g > 1) c += fe.w_2day;
                }
            }
            whale_costs[i][e] = c;
        }
        whale_alias[i].build(whale_costs[i]);
        alias_dirty[i] = false;
    };

    int vnd_budget = std::max(5, ne / 10);
    int iters_done = 0;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        double a = 2.0 * (1.0 - (double)it / max_iterations);

        if (it % 200 == 0)
            std::fill(alias_dirty.begin(), alias_dirty.end(), true);

        for (int i = 0; i < population_size; i++) {
            if (alias_dirty[i]) recompute_alias(i);

            if (i == leader_idx) {
                if (it % 5 == 0) {
                    pop[i].fitness = vnd(pop[i].sol, pop[i].fitness, fe, prob,
                                         ne, np, nr, valid_p, valid_r,
                                         whale_alias[i], vnd_budget, rng);
                    alias_dirty[i] = true;
                }
                continue;
            }

            double r1 = unif(rng);
            double A = 2.0 * a * r1 - a;
            double l = std::uniform_real_distribution<double>(-1.0, 1.0)(rng);
            double p = unif(rng);
            bool changed = false;

            if (p < 0.5 && std::abs(A) < 1.0) {
                int n_move = std::max(1, (int)(std::abs(A) * ne * 0.1));
                for (int m = 0; m < n_move; m++) {
                    int eid = whale_alias[i].sample(rng);
                    int lp = pop[leader_idx].sol.period_of[eid];
                    int lr = pop[leader_idx].sol.room_of[eid];
                    if (lp < 0) continue;
                    if (lp == pop[i].sol.period_of[eid] && lr == pop[i].sol.room_of[eid]) continue;

                    double d = fe.move_delta(pop[i].sol, eid, lp, lr);
                    bool accept = (d < 0) || (unif(rng) < std::exp(-d / (std::abs(A) * 1000.0 + 1.0)));
                    if (accept) {
                        fe.apply_move(pop[i].sol, eid, lp, lr);
                        pop[i].fitness += d;
                        changed = true;
                    }
                }

            } else if (p >= 0.5) {
                int n_chains = 1 + (int)(unif(rng) * 2);
                for (int c = 0; c < n_chains; c++) {
                    int eid = whale_alias[i].sample(rng);
                    int p1 = pop[i].sol.period_of[eid]; if (p1 < 0) continue;
                    int p2;
                    if (unif(rng) < 0.5 && pop[leader_idx].sol.period_of[eid] >= 0)
                        p2 = pop[leader_idx].sol.period_of[eid];
                    else
                        p2 = std::uniform_int_distribution<int>(0, np - 1)(rng);
                    if (p2 == p1) continue;

                    auto chain = kempe_detail::build_chain(pop[i].sol, prob.adj, ne, eid, p1, p2);
                    if (chain.empty() || (int)chain.size() > ne / 4) continue;

                    auto old_pe = fe.partial_eval(pop[i].sol, chain);
                    auto undo = kempe_detail::apply_chain(pop[i].sol, chain, p1, p2);
                    auto new_pe = fe.partial_eval(pop[i].sol, chain);
                    double kd = new_pe.fitness() - old_pe.fitness();

                    bool accept = (kd < 0) || (unif(rng) < 0.3 * std::exp(l));
                    if (accept) {
                        pop[i].fitness += kd;
                        changed = true;
                    } else {
                        kempe_detail::undo_chain(pop[i].sol, undo);
                    }
                }

            } else {
                int n_perturb = std::max(3, ne / 15);
                int rand_whale = rng() % population_size;
                for (int m = 0; m < n_perturb; m++) {
                    int eid = de(rng);
                    auto& vp = valid_p[eid]; auto& vr = valid_r[eid];
                    if (vp.empty() || vr.empty()) continue;

                    int new_pid, new_rid;
                    if (unif(rng) < 0.5 && pop[rand_whale].sol.period_of[eid] >= 0) {
                        new_pid = pop[rand_whale].sol.period_of[eid];
                        new_rid = vr[rng() % vr.size()];
                    } else {
                        new_pid = vp[rng() % vp.size()];
                        new_rid = vr[rng() % vr.size()];
                    }
                    double d = fe.move_delta(pop[i].sol, eid, new_pid, new_rid);
                    fe.apply_move(pop[i].sol, eid, new_pid, new_rid);
                    pop[i].fitness += d;
                }
                changed = true;
            }

            if (changed && it % 10 == 0) {
                alias_dirty[i] = true;
                recompute_alias(i);
                pop[i].fitness = vnd(pop[i].sol, pop[i].fitness, fe, prob,
                                      ne, np, nr, valid_p, valid_r,
                                      whale_alias[i], vnd_budget, rng);
            }
            if (changed) alias_dirty[i] = true;
        }

        // BATCH POINT 2: periodic resync every 100 iters (population_size full_evals → 1 call)
        if ((it + 1) % 100 == 0) {
            std::vector<Solution> resync_sols;
            resync_sols.reserve(population_size);
            for (int i = 0; i < population_size; i++) resync_sols.push_back(pop[i].sol.copy());
            std::vector<double> resync_fit;
            Cuev.score_full_batch(resync_sols, resync_fit);
            for (int i = 0; i < population_size; i++) {
                pop[i].fitness = resync_fit[i];
                pop[i].feasible = (fe.count_hard_fast(pop[i].sol) == 0);
            }
        }

        leader_idx = 0;
        for (int i = 1; i < population_size; i++) {
            bool i_better = false;
            if (pop[i].feasible && !pop[leader_idx].feasible) i_better = true;
            else if (pop[i].feasible == pop[leader_idx].feasible && pop[i].fitness < pop[leader_idx].fitness)
                i_better = true;
            if (i_better) leader_idx = i;
        }

        auto check = fe.full_eval(pop[leader_idx].sol);
        pop[leader_idx].fitness = check.fitness();
        pop[leader_idx].feasible = check.feasible();
        bool cf = check.feasible();
        bool dominated = (best_feasible && !cf);
        if (!dominated && check.fitness() < best_fitness) {
            best_sol = pop[leader_idx].sol.copy();
            best_fitness = check.fitness();
            best_feasible = cf;
            if (verbose && (it < 10 || it % 500 == 0))
                std::cerr << "[WOACuda] Iter " << it << ": best hard=" << check.hard()
                          << " soft=" << check.soft() << std::endl;
        }
    }

    fe.optimize_rooms(best_sol);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[WOACuda] " << iters_done << " iters, " << rt << "s"
                  << " feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard() << " soft=" << final_ev.soft()
                  << " gpu=" << (Cuev.gpu_active ? "on" : "off") << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "WOA (CUDA-deep)"};
}
