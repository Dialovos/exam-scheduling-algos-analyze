/*
 * Multi-neighbourhood simulated annealing.
 * Based on Van Bulck, Goossens & Schaerf (2025).
 *
 * 7 operators via nbhd::select_and_apply:
 *   Move, Swap, Kempe, Kick, Shake, RoomBeam, RoomOnly
 *
 * Component-aware exam targeting (largest/costliest component bias).
 * Feature-based temperature calibration from density + period ratio.
 * Shake perturbation on reheat, not just a temp bump.
 * Infeasible phase uses kick + kempe for faster feasibility recovery.
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

    // Precompute valid periods/rooms per exam
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
    bool current_feasible = ev.feasible();
    Solution best_sol = sol.copy();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();

    // ── Temperature calibration ──
    // Feature-based: uses conflict density and period ratio for instance-adaptive T0.
    // Falls back to sampling if features suggest extreme instances.
    if (init_temp <= 0.0) {
        // Instance features
        int total_conflicts = 0;
        for (int e = 0; e < ne; e++) total_conflicts += (int)prob.adj[e].size();
        total_conflicts /= 2; // each edge counted twice
        double density = (ne > 1) ? (double)total_conflicts / ((double)ne * (ne - 1) / 2) : 0.0;
        double period_ratio = (double)np / std::max(ne, 1);

        // Sampling-based calibration (300 random soft-only moves) — primary
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

        // Feature-based floor: dense/tight instances need minimum exploration
        double feature_floor = 50.0 * (1.0 + density * 2.0) * std::min(3.0, 1.0 / std::max(0.3, period_ratio));

        // Soft floor: small fraction of initial soft penalty
        init_temp = std::max({calib_temp, feature_floor, ev.soft() * 0.005});
    }
    double temp = init_temp;

    if (verbose)
        std::cerr << "[SA] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft()
                  << " T0=" << temp << std::endl;

    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Per-exam cost for weighted selection via alias table
    std::vector<double> exam_cost(ne, 1.0);
    AliasTable alias;
    auto recompute_costs = [&]() {
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
        }
        alias.build(exam_cost);
    };
    recompute_costs();

    // ── Neighbourhood operator weights ──
    // Shake excluded from regular sampling (only used during reheat).
    // Shake blindly perturbs exams and can create hard violations,
    // trapping the search in infeasible recovery on tight instances.
    nbhd::OpWeights op_weights;
    op_weights.w[static_cast<int>(nbhd::OpType::SHAKE)] = 0.0;
    op_weights.w[static_cast<int>(nbhd::OpType::MOVE)] = 0.40;
    op_weights.w[static_cast<int>(nbhd::OpType::KEMPE)] = 0.20;
    op_weights.w[static_cast<int>(nbhd::OpType::KICK)] = 0.15;
    op_weights.w[static_cast<int>(nbhd::OpType::ROOM_BEAM)] = 0.10;

    int no_improve = 0;
    int iters_done = 0;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        // Periodic resync
        if (it % 200 == 0) {
            ev = fe.full_eval(sol);
            current_fitness = ev.fitness();
            current_feasible = ev.feasible();
        }
        if (it % 500 == 0) recompute_costs();

        // ── Multi-Neighbourhood SA (unified for feasible + infeasible) ──
        // Hard violations are weighted 100000x, so SA naturally avoids
        // creating them (acceptance probability ≈ 0 for +100000 delta).
        // Kick and Kempe operators handle violation reduction when present.

        // Select neighbourhood
        nbhd::OpType op = op_weights.sample(rng);

        // SA acceptance function
        auto sa_accept = [&](double delta) -> bool {
            if (delta < 0) return true;
            if (temp > 1e-10) return unif(rng) < std::exp(-delta / temp);
            return false;
        };

        // Apply operator via dispatcher
        auto mr = nbhd::select_and_apply(
            op, sol, fe, prob, valid_p, valid_r, alias, rng,
            sa_accept, std::max(3, ne / 20));

        if (mr.applied) {
            current_fitness += mr.delta;
            op_weights.record(op);

            if (current_fitness < best_fitness - 0.5) {
                auto check = fe.full_eval(sol);
                double af = check.fitness();
                bool af_feasible = check.feasible();
                current_feasible = af_feasible;
                bool dominated = (best_feasible && !af_feasible);
                if (!dominated && af < best_fitness) {
                    best_sol = sol.copy();
                    best_fitness = af;
                    best_feasible = af_feasible;
                    no_improve = 0;
                    if (verbose && (it < 10 || it % 1000 == 0))
                        std::cerr << "[SA] Iter " << it << ": best hard=" << check.hard()
                                  << " soft=" << check.soft()
                                  << " op=" << static_cast<int>(op)
                                  << " T=" << temp << std::endl;
                }
                current_fitness = af;
            } else {
                no_improve++;
            }
        } else {
            no_improve++;
        }

        temp *= cooling;

        // ── Reheat when stuck ──
        if (no_improve > 0 && no_improve % 1000 == 0) {
            // Restart from best_sol + light perturbation to escape local optima
            sol = best_sol.copy();
            int shake_n = (no_improve % 3000 == 0) ? std::max(5, ne / 15) : 3;
            nbhd::shake(sol, fe, valid_p, valid_r, shake_n, rng);
            temp = init_temp * 0.3;
            ev = fe.full_eval(sol);
            current_fitness = ev.fitness();
            current_feasible = ev.feasible();
            // If shake broke feasibility, recover immediately
            if (!current_feasible) {
                fe.recover_feasibility(sol, 50, seed + no_improve);
                ev = fe.full_eval(sol);
                current_fitness = ev.fitness();
                current_feasible = ev.feasible();
            }
            recompute_costs();
        }
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

    return {std::move(best_sol), final_ev, rt, iters_done, "Multi-Neighbourhood SA"};
}
