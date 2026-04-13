/*
 * Kempe chain local search with SA-like acceptance.
 *
 * Standalone algorithm using kempe_detail from neighbourhoods.h.
 * Swaps exams between two periods along conflict chains,
 * preserving period-conflict feasibility by construction.
 *
 * Biased p2 selection (prefer periods with fewer exams / different days).
 * Adaptive chain cap: min(ne/4, 50) when feasible, ne/2 when not.
 * Multi-chain per iteration (try 2-3 chains, accept best delta).
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"
#include "neighbourhoods.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

inline AlgoResult solve_kempe(
    const ProblemInstance& prob,
    int max_iterations = 3000,
    int seed           = 42,
    bool verbose       = false,
    const Solution* init_sol = nullptr)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p();
    FastEvaluator fe(prob);

    // Init from greedy
    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); sol = g.sol.copy(); }

    EvalResult ev = fe.full_eval(sol);
    double current_fitness = ev.fitness();
    int current_hard = ev.hard();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();
    Solution best_sol = sol.copy();

    // SA-like temperature
    double temp = std::max(100.0, ev.soft() * 0.02);
    double cooling_rate = std::pow(0.01 / std::max(temp, 1.0), 1.0 / std::max(max_iterations, 1));
    cooling_rate = std::max(0.995, std::min(0.9999, cooling_rate));

    if (verbose)
        std::cerr << "[Kempe] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft()
                  << " T0=" << temp << std::endl;

    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_int_distribution<int> dp(0, np - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Per-exam cost for weighted seed selection
    std::vector<double> kempe_cost(ne, 1.0);
    AliasTable kempe_alias;
    auto recompute_kempe_costs = [&]() {
        for (int e = 0; e < ne; e++) {
            int pid = sol.period_of[e]; if (pid < 0) { kempe_cost[e] = 1.0; continue; }
            double c = 1.0;
            for (auto& [nb, _] : prob.adj[e]) {
                int np2 = sol.period_of[nb]; if (np2 < 0) continue;
                if (np2 == pid) c += 100000;
                int gap = std::abs(pid - np2);
                if (gap > 0 && gap <= fe.w_spread) c += 1;
                if (fe.period_day[pid] == fe.period_day[np2]) {
                    int g = std::abs(fe.period_daypos[pid] - fe.period_daypos[np2]);
                    if (g == 1) c += fe.w_2row;
                    else if (g > 1) c += fe.w_2day;
                }
            }
            kempe_cost[e] = c;
        }
        kempe_alias.build(kempe_cost);
    };
    recompute_kempe_costs();

    // Exams per period for biased p2 selection
    std::vector<int> period_load(np, 0);
    auto recompute_period_load = [&]() {
        std::fill(period_load.begin(), period_load.end(), 0);
        for (int e = 0; e < ne; e++)
            if (sol.period_of[e] >= 0) period_load[sol.period_of[e]]++;
    };
    recompute_period_load();

    // Biased p2 selection helper
    auto select_p2_biased = [&](int p1) -> int {
        double total = 0;
        std::vector<double> pw(np);
        for (int p = 0; p < np; p++) {
            if (p == p1) { pw[p] = 0; continue; }
            double day_bonus = (fe.period_day[p] != fe.period_day[p1]) ? 2.0 : 1.0;
            pw[p] = day_bonus / std::max(1, period_load[p]);
            total += pw[p];
        }
        if (total <= 0) return dp(rng);
        double r = unif(rng) * total;
        double acc = 0;
        for (int p = 0; p < np; p++) {
            acc += pw[p];
            if (r <= acc) return p;
        }
        return np - 1;
    };

    int no_improve = 0;
    int iters_done = 0;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;
        if (it % 500 == 0) { recompute_kempe_costs(); recompute_period_load(); }

        // Adaptive chain cap
        int chain_cap = ev.feasible() ? std::min(ne / 4, 50) : ne / 2;

        // Multi-chain: try 2-3 chains, keep best
        int n_tries = (no_improve > 200) ? 3 : 2;
        double best_delta = 1e18;
        std::vector<int> best_chain;
        int best_p1 = -1, best_p2 = -1;

        for (int tc = 0; tc < n_tries; tc++) {
            int eid = (unif(rng) < 0.7) ? kempe_alias.sample(rng) : de(rng);
            int p1 = sol.period_of[eid];
            if (p1 < 0) continue;

            // Biased p2 selection (60%) vs uniform (40%)
            int p2 = (unif(rng) < 0.6) ? select_p2_biased(p1) : dp(rng);
            if (p2 == p1) continue;

            auto chain = kempe_detail::build_chain(sol, prob.adj, ne, eid, p1, p2);
            if (chain.empty() || (int)chain.size() > chain_cap) continue;

            // Evaluate: partial_eval before, apply, partial_eval after, undo
            auto old_pe = fe.partial_eval(sol, chain);
            auto undo = kempe_detail::apply_chain(sol, chain, p1, p2);
            auto new_pe = fe.partial_eval(sol, chain);
            double delta = new_pe.fitness() - old_pe.fitness();
            kempe_detail::undo_chain(sol, undo);

            if (delta < best_delta) {
                best_delta = delta;
                best_chain = chain;
                best_p1 = p1;
                best_p2 = p2;
            }
        }

        if (best_chain.empty()) continue;

        // SA-like acceptance on best chain
        bool accept = (best_delta < 0);
        if (!accept && temp > 1e-10)
            accept = (unif(rng) < std::exp(-best_delta / temp));

        if (accept) {
            kempe_detail::apply_chain(sol, best_chain, best_p1, best_p2);
            current_fitness += best_delta;

            if (current_fitness < best_fitness - 0.5) {
                auto check = fe.full_eval(sol);
                current_fitness = check.fitness();
                current_hard = check.hard();
                bool nf = check.feasible();
                bool dominated = (best_feasible && !nf);
                if (!dominated && current_fitness < best_fitness) {
                    best_sol = sol.copy();
                    best_fitness = current_fitness;
                    best_feasible = nf;
                    no_improve = 0;
                    if (verbose && (it < 10 || it % 500 == 0))
                        std::cerr << "[Kempe] Iter " << it << ": best soft="
                                  << check.soft() << " T=" << temp << std::endl;
                }
            } else {
                no_improve++;
            }
        } else {
            no_improve++;
        }

        // Periodic resync
        if ((it + 1) % 200 == 0) {
            auto sync = fe.full_eval(sol);
            current_fitness = sync.fitness();
            current_hard = sync.hard();
        }

        temp *= cooling_rate;

        // Reheat when stuck
        if (no_improve > 0 && no_improve % 500 == 0)
            temp = std::max(temp, std::max(100.0, best_fitness * 0.005));
    }

    fe.optimize_rooms(best_sol);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[Kempe] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "Kempe Chain"};
}
