/*
 * sa.h — Simulated Annealing
 *
 * Geometric cooling with probabilistic acceptance of worse moves.
 * Uses move_delta for O(k) neighbor evaluation.
 * Re-syncs with full_eval every 50 iterations.
 * Reheats when stuck.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

inline AlgoResult solve_sa(
    const ProblemInstance& prob,
    int max_iterations = 5000,
    double init_temp   = 0.0,
    double cooling     = 0.9995,
    int seed           = 42,
    bool verbose       = false)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    FastEvaluator fe(prob);

    std::vector<int> exam_dur(ne), exam_enr(ne), period_dur(np), room_cap(nr);
    for (auto& e : prob.exams) { exam_dur[e.id] = e.duration; exam_enr[e.id] = e.enrollment(); }
    for (auto& p : prob.periods) period_dur[p.id] = p.duration;
    for (auto& r : prob.rooms) room_cap[r.id] = r.capacity;

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (exam_dur[e] <= period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (exam_enr[e] <= room_cap[r]) valid_r[e].push_back(r);
    }

    // Init from greedy
    auto greedy_res = solve_greedy(prob, false);
    Solution sol = greedy_res.sol.copy();

    EvalResult ev = fe.full_eval(sol);
    double current_fitness = ev.fitness();
    Solution best_sol = sol.copy();
    double best_fitness = current_fitness;

    if (init_temp <= 0.0)
        init_temp = (ev.soft() > 0) ? std::max(1.0, ev.soft() * 0.05) : 100.0;
    double temp = init_temp;

    if (verbose)
        std::cerr << "[SA] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft()
                  << " T0=" << temp << std::endl;

    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    int no_improve = 0;
    int iters_done = 0;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        if (it % 50 == 0) {
            ev = fe.full_eval(sol);
            current_fitness = ev.fitness();
        }

        int eid = de(rng);
        auto& vp = valid_p[eid];
        auto& vr = valid_r[eid];
        if (vp.empty() || vr.empty()) continue;

        int new_pid = vp[rng() % vp.size()];
        int new_rid = vr[rng() % vr.size()];
        if (new_pid == sol.period_of[eid] && new_rid == sol.room_of[eid]) continue;

        double delta = fe.move_delta(sol, eid, new_pid, new_rid);

        bool accept = (delta < 0);
        if (!accept && temp > 1e-10)
            accept = (unif(rng) < std::exp(-delta / temp));

        if (accept) {
            fe.apply_move(sol, eid, new_pid, new_rid);
            current_fitness += delta;

            if (current_fitness < best_fitness - 0.5) {
                auto check = fe.full_eval(sol);
                double af = check.fitness();
                if (af < best_fitness) {
                    best_sol = sol.copy();
                    best_fitness = af;
                    no_improve = 0;
                    if (verbose && (it < 10 || it % 1000 == 0))
                        std::cerr << "[SA] Iter " << it << ": best hard=" << check.hard()
                                  << " soft=" << check.soft() << " T=" << temp << std::endl;
                }
                current_fitness = af;
            } else {
                no_improve++;
            }
        } else {
            no_improve++;
        }

        temp *= cooling;

        // Reheat when stuck
        if (no_improve > 0 && no_improve % 1000 == 0)
            temp = std::max(temp, init_temp * 0.1);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[SA] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "Simulated Annealing"};
}