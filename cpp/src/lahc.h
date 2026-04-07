/*
 * lahc.h — Late Acceptance Hill Climbing
 *
 * Accept move if fitness(neighbor) <= fitness(current)
 * OR fitness(neighbor) <= history[iter % L].
 *
 * The history list acts as memory of recent fitness values, allowing
 * the search to accept moves no worse than L steps ago. This provides
 * automatic diversification without temperature tuning.
 *
 * Only one parameter: list length L (typically ne..ne*np).
 * Reported to outperform SA/GD on exam timetabling benchmarks
 * (Burke & Bykov, 2017).
 *
 * Uses move_delta for O(k) evaluation. Mixes period moves (70%)
 * and room-only moves (30%) for separate room optimization.
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

inline AlgoResult solve_lahc(
    const ProblemInstance& prob,
    int max_iterations = 5000,
    int list_length    = 0,    // 0 = auto (ne * 5)
    int seed           = 42,
    bool verbose       = false,
    const Solution* init_sol = nullptr)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    FastEvaluator fe(prob);

    // Auto list length: scale with problem size
    if (list_length <= 0) list_length = ne * 5;

    std::vector<int> exam_dur(ne), exam_enr(ne), period_dur(np), room_cap(nr);
    for (auto& e : prob.exams) { exam_dur[e.id] = e.duration; exam_enr[e.id] = e.enrollment(); }
    for (auto& p : prob.periods) period_dur[p.id] = p.duration;
    for (auto& r : prob.rooms) room_cap[r.id] = r.capacity;

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (exam_dur[e] <= period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (exam_enr[e] <= room_cap[r])  valid_r[e].push_back(r);
    }

    // Init from greedy
    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); sol = g.sol.copy(); }

    EvalResult ev = fe.full_eval(sol);

    // Feasibility recovery if needed
    if (!ev.feasible()) {
        if (verbose)
            std::cerr << "[LAHC] Greedy infeasible (hard=" << ev.hard()
                      << "), running recovery..." << std::endl;
        fe.recover_feasibility(sol, 500, seed);
        ev = fe.full_eval(sol);
    }

    double current_fitness = ev.fitness();
    Solution best_sol = sol.copy();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();

    // Initialize history list with current fitness
    std::vector<double> history(list_length, current_fitness);

    if (verbose)
        std::cerr << "[LAHC] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft()
                  << " L=" << list_length << std::endl;

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

        // Periodic resync
        if (it % 50 == 0) {
            ev = fe.full_eval(sol);
            current_fitness = ev.fitness();
        }
        if (it % 500 == 0) recompute_costs();

        int eid = (unif(rng) < 0.7) ? weighted_pick() : de(rng);
        auto& vp = valid_p[eid];
        auto& vr = valid_r[eid];
        if (vp.empty() || vr.empty()) continue;

        double delta;
        int move_pid, move_rid;
        bool is_feasible = (current_fitness < 100000);
        bool is_swap = false;
        int swap_e2 = -1, swap_e2_pid = -1, swap_e2_rid = -1;

        if (!is_feasible) {
            // Steepest descent for hard violations
            int cp = sol.period_of[eid];
            int best_pid = -1, best_rid = -1;
            double best_d = 1e18;
            for (int pid : vp) {
                if (pid == cp) continue;
                for (int rid : vr) {
                    double d = fe.move_delta(sol, eid, pid, rid);
                    if (d < best_d) { best_d = d; best_pid = pid; best_rid = rid; }
                }
            }
            if (best_pid < 0) continue;
            delta = best_d; move_pid = best_pid; move_rid = best_rid;
        } else {
            double r_move = unif(rng);
            if (r_move < 0.25 && ne > 1) {
                // Swap move: exchange periods of two exams (keep rooms)
                int e2 = (unif(rng) < 0.7) ? weighted_pick() : de(rng);
                if (e2 == eid) continue;
                int cp1 = sol.period_of[eid], cr1 = sol.room_of[eid];
                int cp2 = sol.period_of[e2], cr2 = sol.room_of[e2];
                if (cp1 < 0 || cp2 < 0 || cp1 == cp2) continue;
                if (exam_dur[eid] > period_dur[cp2] || exam_dur[e2] > period_dur[cp1]) continue;
                double d1 = fe.move_delta(sol, eid, cp2, cr1);
                fe.apply_move(sol, eid, cp2, cr1);
                double d2 = fe.move_delta(sol, e2, cp1, cr2);
                fe.apply_move(sol, eid, cp1, cr1); // undo
                delta = d1 + d2;
                move_pid = cp2; move_rid = cr1;
                is_swap = true;
                swap_e2 = e2; swap_e2_pid = cp1; swap_e2_rid = cr2;
            } else if (r_move < 0.40 && nr > 1) {
                // Room-only move
                move_pid = sol.period_of[eid];
                move_rid = vr[rng() % vr.size()];
                if (move_rid == sol.room_of[eid]) continue;
                delta = fe.move_delta(sol, eid, move_pid, move_rid);
            } else {
                // Single period+room move
                move_pid = vp[rng() % vp.size()];
                move_rid = vr[rng() % vr.size()];
                if (move_pid == sol.period_of[eid] && move_rid == sol.room_of[eid]) continue;
                delta = fe.move_delta(sol, eid, move_pid, move_rid);
            }
        }

        double new_fitness = current_fitness + delta;
        int hist_idx = it % list_length;

        // LAHC acceptance: better than current OR better than L steps ago
        bool accept = (new_fitness <= current_fitness) ||
                      (new_fitness <= history[hist_idx]);

        if (accept) {
            fe.apply_move(sol, eid, move_pid, move_rid);
            if (is_swap)
                fe.apply_move(sol, swap_e2, swap_e2_pid, swap_e2_rid);
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
                        std::cerr << "[LAHC] Iter " << it << ": best hard=" << check.hard()
                                  << " soft=" << check.soft() << std::endl;
                }
                current_fitness = af;
            } else {
                no_improve++;
            }
        } else {
            no_improve++;
        }

        // Update history with current fitness (whether accepted or not)
        history[hist_idx] = current_fitness;
    }

    fe.optimize_rooms(best_sol);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[LAHC] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "LAHC"};
}
