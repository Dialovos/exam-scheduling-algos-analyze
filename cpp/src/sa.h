/*
 * sa.h — Simulated Annealing
 *
 * Geometric cooling with probabilistic acceptance of worse moves.
 * Uses move_delta for O(k) neighbor evaluation.
 * Re-syncs with full_eval every 50 iterations.
 * Reheats when stuck.
 *
 * FastSA pruning: tracks accepted moves per exam per temperature bin.
 * Exams with zero accepted moves in the previous bin are skipped (90%),
 * saving move_delta evaluations on "frozen" exams at low T.
 *
 * Mixed neighborhood: 70% period+room moves, 30% room-only moves
 * for separate room optimization.
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
    bool verbose       = false,
    const Solution* init_sol = nullptr)
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
    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); sol = g.sol.copy(); }

    EvalResult ev = fe.full_eval(sol);

    // Feasibility recovery if greedy started infeasible
    if (!ev.feasible()) {
        if (verbose)
            std::cerr << "[SA] Greedy infeasible (hard=" << ev.hard()
                      << "), running recovery..." << std::endl;
        fe.recover_feasibility(sol, 500, seed);
        ev = fe.full_eval(sol);
        if (verbose)
            std::cerr << "[SA] After recovery: feasible=" << ev.feasible()
                      << " hard=" << ev.hard() << " soft=" << ev.soft() << std::endl;
    }

    double current_fitness = ev.fitness();
    Solution best_sol = sol.copy();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();

    // Temperature calibration: sample random moves, exclude hard-violation deltas
    if (init_temp <= 0.0) {
        double avg_worsen = 0; int n_w = 0;
        std::uniform_int_distribution<int> sde(0, ne - 1);
        for (int s = 0; s < 300; s++) {
            int eid = sde(rng);
            if (valid_p[eid].empty() || valid_r[eid].empty()) continue;
            int pid = valid_p[eid][rng() % valid_p[eid].size()];
            int rid = valid_r[eid][rng() % valid_r[eid].size()];
            double d = fe.move_delta(sol, eid, pid, rid);
            if (d > 0 && d < 50000) { avg_worsen += d; n_w++; }
        }
        double calib_temp = (n_w > 0) ? std::max(1.0, (avg_worsen / n_w) / 0.693) : 100.0;
        // Floor: small fraction of soft to ensure minimum exploration on large instances
        init_temp = std::max(calib_temp, ev.soft() * 0.005);
    }
    double temp = init_temp;

    if (verbose)
        std::cerr << "[SA] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft()
                  << " T0=" << temp << std::endl;

    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Per-exam cost for weighted selection (recomputed periodically)
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
        // Normalize to make a distribution
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

    // FastSA: temperature-bin activity tracking
    int bin_size = std::max(ne, 100);
    std::vector<int> active_cur(ne, 0);
    std::vector<int> active_prev(ne, 1);  // all active initially
    int bin_iter = 0;

    int no_improve = 0;
    int iters_done = 0;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        // FastSA: advance temperature bin
        bin_iter++;
        if (bin_iter >= bin_size) {
            active_prev.swap(active_cur);
            std::fill(active_cur.begin(), active_cur.end(), 0);
            bin_iter = 0;
        }

        if (it % 50 == 0) {
            ev = fe.full_eval(sol);
            current_fitness = ev.fitness();
        }
        // Recompute cost weights periodically
        if (it % 500 == 0) recompute_costs();

        int eid = (unif(rng) < 0.7) ? weighted_pick() : de(rng);
        auto& vp = valid_p[eid];
        auto& vr = valid_r[eid];
        if (vp.empty() || vr.empty()) continue;

        // FastSA: skip exams inactive in previous bin (90% skip rate)
        bool is_feasible = (current_fitness < 100000);
        if (is_feasible && active_prev[eid] == 0 && unif(rng) < 0.9)
            continue;

        double delta;
        int move_pid, move_rid;
        bool is_swap = false;
        int swap_e2 = -1, swap_e2_pid = -1, swap_e2_rid = -1;

        if (!is_feasible) {
            // Steepest descent scan — quickly reduce hard violations
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
                // Compute via temp-apply/undo
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

        bool accept = (delta < 0);
        if (!accept && temp > 1e-10)
            accept = (unif(rng) < std::exp(-delta / temp));

        if (accept) {
            fe.apply_move(sol, eid, move_pid, move_rid);
            if (is_swap) {
                fe.apply_move(sol, swap_e2, swap_e2_pid, swap_e2_rid);
                active_cur[swap_e2]++;
            }
            current_fitness += delta;
            active_cur[eid]++;

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

        // Reheat when stuck — stronger reheat at 30% of init
        if (no_improve > 0 && no_improve % 1000 == 0)
            temp = std::max(temp, init_temp * 0.3);
    }

    fe.optimize_rooms(best_sol);

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