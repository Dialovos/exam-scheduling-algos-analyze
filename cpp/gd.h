/*
 * gd.h — Great Deluge Algorithm
 *
 * Level-based acceptance: accepts any move where new fitness <= level.
 * Level decreases linearly from initial fitness.
 * Raises level when stuck to escape local optima.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <vector>

inline AlgoResult solve_great_deluge(
    const ProblemInstance& prob,
    int max_iterations = 5000,
    double decay_rate  = 0.0,
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

    double level = current_fitness;
    if (decay_rate <= 0.0)
        decay_rate = current_fitness / std::max(max_iterations, 1);

    if (verbose)
        std::cerr << "[GD] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft()
                  << " level=" << level << std::endl;

    std::uniform_int_distribution<int> de(0, ne - 1);
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
        double new_fitness = current_fitness + delta;

        if (new_fitness <= level) {
            fe.apply_move(sol, eid, new_pid, new_rid);
            current_fitness = new_fitness;

            if (current_fitness < best_fitness - 0.5) {
                auto check = fe.full_eval(sol);
                double af = check.fitness();
                if (af < best_fitness) {
                    best_sol = sol.copy();
                    best_fitness = af;
                    no_improve = 0;
                    if (verbose && (it < 10 || it % 1000 == 0))
                        std::cerr << "[GD] Iter " << it << ": best hard=" << check.hard()
                                  << " soft=" << check.soft() << " lvl=" << level << std::endl;
                }
                current_fitness = af;
            } else {
                no_improve++;
            }
        } else {
            no_improve++;
        }

        level -= decay_rate;

        // Raise level if stuck
        if (no_improve > 0 && no_improve % 1000 == 0)
            level = current_fitness * 1.02;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[GD] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "Great Deluge"};
}