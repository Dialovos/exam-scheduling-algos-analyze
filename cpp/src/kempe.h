/*
 * kempe.h — Kempe Chain Local Search
 *
 * Swaps exams between two periods along conflict chains.
 * Preserves period-conflict feasibility by construction.
 * Uses full_eval to assess chain swaps.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

inline AlgoResult solve_kempe(
    const ProblemInstance& prob,
    int max_iterations = 3000,
    int seed           = 42,
    bool verbose       = false)
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
    auto greedy_res = solve_greedy(prob, false);
    Solution sol = greedy_res.sol.copy();

    EvalResult ev = fe.full_eval(sol);
    double current_fitness = ev.fitness();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();
    Solution best_sol = sol.copy();

    if (verbose)
        std::cerr << "[Kempe] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft() << std::endl;

    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_int_distribution<int> dp(0, np - 1);

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

        if (new_fitness < current_fitness) {
            current_fitness = new_fitness;
            bool nf = new_ev.feasible();
            bool dominated = (best_feasible && !nf);
            if (!dominated && new_fitness < best_fitness) {
                best_sol = sol.copy();
                best_fitness = new_fitness;
                best_feasible = nf;
                no_improve = 0;
                if (verbose && (it < 10 || it % 500 == 0)) {
                    auto ev2 = fe.full_eval(sol);
                    std::cerr << "[Kempe] Iter " << it << ": best hard=" << ev2.hard()
                              << " soft=" << ev2.soft() << std::endl;
                }
            } else {
                no_improve++;
            }
        } else {
            // Rollback
            for (auto& s : saved)
                sol.assign(s.eid, s.pid, s.rid);
            no_improve++;
        }

        if (no_improve > max_iterations / 2) break;
    }

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