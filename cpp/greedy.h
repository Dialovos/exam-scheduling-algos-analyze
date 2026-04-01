/*
 * greedy.h — Greedy Graph Coloring Heuristic
 *
 * 1. Sort exams by conflict degree (descending)
 * 2. Assign each exam to the first feasible (period, room) pair
 *
 * Time: Θ(n²)   |   Deterministic baseline.
 */

#pragma once

#include "models.h"
#include "evaluator.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <set>
#include <vector>

inline AlgoResult solve_greedy(const ProblemInstance& prob, bool verbose = false) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();

    Solution sol;
    sol.init(prob);

    // Sort exams: highest conflict degree first, then largest enrollment
    std::vector<int> order(ne);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        int da = (int)prob.adj[a].size(), db = (int)prob.adj[b].size();
        if (da != db) return da > db;
        return prob.exams[a].enrollment() > prob.exams[b].enrollment();
    });

    int unassigned = 0;
    for (int eid : order) {
        int dur = prob.exams[eid].duration;

        // Blocked periods = periods already used by conflicting exams
        std::set<int> blocked;
        for (auto& [nb, _] : prob.adj[eid]) {
            int p = sol.period_of[nb];
            if (p >= 0) blocked.insert(p);
        }

        bool assigned = false;
        for (int pid = 0; pid < np && !assigned; pid++) {
            if (blocked.count(pid)) continue;
            if (dur > prob.periods[pid].duration) continue;

            for (int rid = 0; rid < nr; rid++) {
                if (sol.get_pr_enroll(pid, rid) + prob.exams[eid].enrollment()
                    <= prob.rooms[rid].capacity) {
                    sol.assign(eid, pid, rid);
                    assigned = true;
                    break;
                }
            }
        }

        if (!assigned) {
            sol.assign(eid, 0, 0); // fallback
            unassigned++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();

    FastEvaluator fe(prob);
    EvalResult ev = fe.full_eval(sol);

    if (verbose) {
        std::cerr << "[Greedy] " << rt << "s, feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft() << std::endl;
        if (unassigned > 0)
            std::cerr << "[Greedy] " << unassigned << " exams placed infeasibly" << std::endl;
    }

    return {std::move(sol), ev, rt, 0, "Greedy"};
}