/*
 * kempe.h — Kempe Chain Local Search with SA-like Acceptance
 *
 * Swaps exams between two periods along conflict chains.
 * Preserves period-conflict feasibility by construction.
 * Accepts worse swaps with probability exp(-delta/T) for diversification.
 * Uses full_eval to assess chain swaps (chains touch many exams).
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <queue>
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

    // Flat adjacency
    std::vector<std::vector<int>> adj(ne);
    for (int e = 0; e < ne; e++)
        for (auto& [nb, _] : prob.adj[e]) adj[e].push_back(nb);

    // Init from greedy
    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); sol = g.sol.copy(); }

    EvalResult ev = fe.full_eval(sol);
    double current_fitness = ev.fitness();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();
    Solution best_sol = sol.copy();

    // SA-like temperature: calibrate from initial fitness
    double temp = std::max(100.0, ev.soft() * 0.02);
    double cooling = std::pow(0.01 / std::max(temp, 1.0), 1.0 / std::max(max_iterations, 1));
    // Ensure cooling is in reasonable range
    cooling = std::max(0.995, std::min(0.9999, cooling));

    if (verbose)
        std::cerr << "[Kempe] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft()
                  << " T0=" << temp << std::endl;

    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_int_distribution<int> dp(0, np - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    int no_improve = 0;
    int iters_done = 0;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        int eid = de(rng);
        int p1 = sol.period_of[eid];
        if (p1 < 0) continue;

        int p2 = dp(rng);
        if (p2 == p1) continue;

        // Build Kempe chain via BFS
        std::vector<int> chain;
        std::vector<bool> in_chain(ne, false);
        std::queue<int> q;
        q.push(eid);
        in_chain[eid] = true;

        while (!q.empty()) {
            int e = q.front(); q.pop();
            chain.push_back(e);
            int ep = sol.period_of[e];
            int target = (ep == p1) ? p2 : p1;
            for (int nb : adj[e]) {
                if (!in_chain[nb] && sol.period_of[nb] == target) {
                    in_chain[nb] = true;
                    q.push(nb);
                }
            }
        }

        if (chain.empty()) continue;

        // Save old assignments
        struct OldAssign { int eid, pid, rid; };
        std::vector<OldAssign> saved;
        saved.reserve(chain.size());
        for (int e : chain)
            saved.push_back({e, sol.period_of[e], sol.room_of[e]});

        // Swap: p1 <-> p2, keep rooms
        for (int e : chain) {
            int ep = sol.period_of[e];
            int er = sol.room_of[e];
            sol.assign(e, (ep == p1) ? p2 : p1, er);
        }

        auto new_ev = fe.full_eval(sol);
        double new_fitness = new_ev.fitness();
        double delta = new_fitness - current_fitness;

        // SA-like acceptance: always accept improvements, accept worse with probability
        bool accept = (delta < 0);
        if (!accept && temp > 1e-10)
            accept = (unif(rng) < std::exp(-delta / temp));

        if (accept) {
            current_fitness = new_fitness;
            bool nf = new_ev.feasible();
            bool dominated = (best_feasible && !nf);
            if (!dominated && new_fitness < best_fitness) {
                best_sol = sol.copy();
                best_fitness = new_fitness;
                best_feasible = nf;
                no_improve = 0;
                if (verbose && (it < 10 || it % 500 == 0))
                    std::cerr << "[Kempe] Iter " << it << ": best hard=" << new_ev.hard()
                              << " soft=" << new_ev.soft() << " T=" << temp << std::endl;
            } else {
                no_improve++;
            }
        } else {
            // Rollback
            for (auto& s : saved)
                sol.assign(s.eid, s.pid, s.rid);
            no_improve++;
        }

        temp *= cooling;

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
