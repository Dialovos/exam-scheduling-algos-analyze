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

    // Feasibility recovery if greedy started infeasible
    if (!ev.feasible()) {
        if (verbose)
            std::cerr << "[GD] Greedy infeasible (hard=" << ev.hard()
                      << "), running recovery..." << std::endl;
        fe.recover_feasibility(sol, 500, seed);
        ev = fe.full_eval(sol);
        if (verbose)
            std::cerr << "[GD] After recovery: feasible=" << ev.feasible()
                      << " hard=" << ev.hard() << " soft=" << ev.soft() << std::endl;
    }

    double current_fitness = ev.fitness();
    Solution best_sol = sol.copy();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();

    // Start level above current fitness for initial exploration, decay toward 30%
    double level = current_fitness * 1.1;
    double target_level = current_fitness * 0.3;
    // If infeasible, allow more room for exploration
    if (!ev.feasible()) {
        level = current_fitness * 1.2;
        target_level = current_fitness * 0.1;
    }
    if (decay_rate <= 0.0)
        decay_rate = (level - target_level) / std::max(max_iterations, 1);

    if (verbose)
        std::cerr << "[GD] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft()
                  << " level=" << level << " target=" << target_level << std::endl;

    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Per-exam cost for weighted selection
    std::vector<double> exam_cost(ne, 1.0);
    auto recompute_costs = [&]() {
        double total = 0;
        for (int e = 0; e < ne; e++) {
            int pid = sol.period_of[e]; if (pid < 0) { exam_cost[e] = 1.0; continue; }
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
            exam_cost[e] = c;
            total += c;
        }
        if (total > 0) for (int e = 0; e < ne; e++) exam_cost[e] /= total;
    };

    auto weighted_pick = [&]() -> int {
        double r = unif(rng);
        double acc = 0;
        for (int e = 0; e < ne; e++) {
            acc += exam_cost[e];
            if (r <= acc) return e;
        }
        return ne - 1;
    };

    recompute_costs();

    int no_improve = 0;
    int iters_done = 0;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        if (it % 50 == 0) {
            ev = fe.full_eval(sol);
            current_fitness = ev.fitness();
        }
        if (it % 500 == 0) recompute_costs();

        // Steepest descent within level: scan all moves for selected exam
        int eid = (unif(rng) < 0.7) ? weighted_pick() : de(rng);
        auto& vp = valid_p[eid];
        auto& vr = valid_r[eid];
        if (vp.empty() || vr.empty()) continue;

        int best_pid = -1, best_rid = -1;
        double best_delta = 1e18;
        int cp = sol.period_of[eid];

        for (int pid : vp) {
            if (pid == cp) continue;
            for (int rid : vr) {
                double d = fe.move_delta(sol, eid, pid, rid);
                if (d < best_delta) {
                    best_delta = d;
                    best_pid = pid;
                    best_rid = rid;
                }
            }
        }

        if (best_pid < 0) continue;
        double new_fitness = current_fitness + best_delta;

        if (new_fitness <= level) {
            fe.apply_move(sol, eid, best_pid, best_rid);
            current_fitness = new_fitness;

            if (current_fitness < best_fitness - 0.5) {
                auto check = fe.full_eval(sol);
                double af = check.fitness();
                bool af_feasible = check.feasible();
                bool dominated = (best_feasible && !af_feasible);
                if (!dominated && af < best_fitness) {
                    best_sol = sol.copy();
                    best_fitness = af;
                    best_feasible = af_feasible;
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

        // Raise level if stuck — 5% above current, every 500 iters
        if (no_improve > 0 && no_improve % 500 == 0)
            level = current_fitness * 1.05;
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