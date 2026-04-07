/*
 * natural_selection.h — Natural Selection Meta-Algorithm
 *
 * Competitive algorithm selection (no ML):
 *   Phase 1 — Trial:   run each metaheuristic at 20% budget, rank by fitness
 *   Phase 2 — Finals:  top-N finalists run at full budget
 *   Return:  best solution across all trials + finals (feasibility-first)
 *
 * Stepping stone toward a future dynamic hybrid that chains multiple algorithms.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"
#include "tabu.h"
#include "hho.h"
#include "kempe.h"
#include "sa.h"
#include "alns.h"
#include "gd.h"
#include "abc.h"
#include "ga.h"
#include "lahc.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

inline AlgoResult solve_natural_selection(
    const ProblemInstance& prob,
    int n_finalists     = 3,
    int tabu_iters      = 2000,
    int tabu_tenure     = 20,
    int tabu_patience   = 500,
    int hho_pop         = 30,
    int hho_iters       = 200,
    int sa_iters        = 5000,
    int kempe_iters     = 3000,
    int alns_iters      = 2000,
    int gd_iters        = 5000,
    int abc_pop         = 30,
    int abc_iters       = 3000,
    int ga_pop          = 50,
    int ga_iters        = 500,
    int lahc_iters      = 5000,
    int lahc_list       = 0,
    int seed            = 42,
    bool verbose        = false)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    const double trial_frac = 0.2;

    struct Candidate {
        std::string name;
        int id;
        double trial_fitness;
    };

    std::vector<AlgoResult> all_results;
    std::vector<Candidate> candidates;
    all_results.reserve(16);

    if (verbose)
        std::cerr << "[NS] Trial phase: testing 9 algorithms at "
                  << (int)(trial_frac * 100) << "% budget..." << std::endl;

    // ── Trial phase ────────────────────────────────────────

    auto run_trial = [&](auto fn, const std::string& name, int id) {
        auto r = fn();
        double fit = r.eval.fitness();
        if (verbose)
            std::cerr << "  " << std::left << std::setw(20) << name
                      << std::fixed << std::setprecision(0)
                      << " fitness=" << fit
                      << " (hard=" << r.eval.hard()
                      << " soft=" << r.eval.soft() << ")"
                      << std::fixed << std::setprecision(2)
                      << " " << r.runtime_sec << "s" << std::endl;
        candidates.push_back({name, id, fit});
        all_results.push_back(std::move(r));
    };

    run_trial([&]() {
        return solve_tabu(prob, std::max(100, (int)(tabu_iters * trial_frac)),
                          tabu_tenure, tabu_patience, seed, false);
    }, "Tabu Search", 0);

    run_trial([&]() {
        return solve_hho(prob, hho_pop, std::max(20, (int)(hho_iters * trial_frac)),
                         seed, false);
    }, "HHO", 1);

    run_trial([&]() {
        return solve_kempe(prob, std::max(100, (int)(kempe_iters * trial_frac)),
                           seed, false);
    }, "Kempe Chain", 2);

    run_trial([&]() {
        return solve_sa(prob, std::max(200, (int)(sa_iters * trial_frac)),
                        0.0, 0.9995, seed, false);
    }, "Simulated Annealing", 3);

    run_trial([&]() {
        return solve_alns(prob, std::max(100, (int)(alns_iters * trial_frac)),
                          0.15, seed, false);
    }, "ALNS", 4);

    run_trial([&]() {
        return solve_great_deluge(prob, std::max(200, (int)(gd_iters * trial_frac)),
                                  0.0, seed, false);
    }, "Great Deluge", 5);

    run_trial([&]() {
        return solve_abc(prob, abc_pop, std::max(100, (int)(abc_iters * trial_frac)),
                         0, seed, false);
    }, "ABC", 6);

    run_trial([&]() {
        return solve_ga(prob, ga_pop, std::max(50, (int)(ga_iters * trial_frac)),
                        0.8, 0.15, seed, false);
    }, "Genetic Algorithm", 7);

    run_trial([&]() {
        return solve_lahc(prob, std::max(200, (int)(lahc_iters * trial_frac)),
                          lahc_list, seed, false);
    }, "LAHC", 8);

    // ── Select finalists ───────────────────────────────────

    std::sort(candidates.begin(), candidates.end(),
              [](auto& a, auto& b) { return a.trial_fitness < b.trial_fitness; });

    int nf = std::max(1, std::min(n_finalists, (int)candidates.size()));

    if (verbose) {
        std::cerr << "[NS] Finalists (" << nf << "):";
        for (int i = 0; i < nf; i++)
            std::cerr << " " << candidates[i].name
                      << "(" << std::fixed << std::setprecision(0)
                      << candidates[i].trial_fitness << ")";
        std::cerr << std::endl;
    }

    // ── Finals phase ───────────────────────────────────────

    for (int i = 0; i < nf; i++) {
        int id = candidates[i].id;
        if (verbose)
            std::cerr << "[NS] Finals: running " << candidates[i].name
                      << " at full budget..." << std::endl;

        AlgoResult r;
        switch (id) {
            case 0: r = solve_tabu(prob, tabu_iters, tabu_tenure, tabu_patience, seed, verbose); break;
            case 1: r = solve_hho(prob, hho_pop, hho_iters, seed, verbose); break;
            case 2: r = solve_kempe(prob, kempe_iters, seed, verbose); break;
            case 3: r = solve_sa(prob, sa_iters, 0.0, 0.9995, seed, verbose); break;
            case 4: r = solve_alns(prob, alns_iters, 0.15, seed, verbose); break;
            case 5: r = solve_great_deluge(prob, gd_iters, 0.0, seed, verbose); break;
            case 6: r = solve_abc(prob, abc_pop, abc_iters, 0, seed, verbose); break;
            case 7: r = solve_ga(prob, ga_pop, ga_iters, 0.8, 0.15, seed, verbose); break;
            case 8: r = solve_lahc(prob, lahc_iters, lahc_list, seed, verbose); break;
        }

        if (verbose)
            std::cerr << "  -> " << r.algorithm << ": hard=" << r.eval.hard()
                      << " soft=" << r.eval.soft() << std::endl;

        all_results.push_back(std::move(r));
    }

    // ── Pick overall best (feasibility-first) ──────────────

    int best_idx = 0;
    for (int i = 1; i < (int)all_results.size(); i++) {
        bool bi = all_results[i].eval.feasible();
        bool bb = all_results[best_idx].eval.feasible();
        if ((bi && !bb) ||
            (bi == bb && all_results[i].eval.fitness() < all_results[best_idx].eval.fitness()))
            best_idx = i;
    }

    AlgoResult result = std::move(all_results[best_idx]);

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_rt = std::chrono::duration<double>(t1 - t0).count();

    std::string winner = result.algorithm;
    result.runtime_sec = total_rt;
    result.algorithm = "Natural Selection";

    if (verbose)
        std::cerr << "[NS] Winner: " << winner
                  << "  feasible=" << result.eval.feasible()
                  << " hard=" << result.eval.hard()
                  << " soft=" << result.eval.soft()
                  << "  total=" << std::fixed << std::setprecision(1)
                  << total_rt << "s" << std::endl;

    return result;
}
