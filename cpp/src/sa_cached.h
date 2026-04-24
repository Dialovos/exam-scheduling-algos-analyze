/*
 * Multi-neighbourhood SA with CachedEvaluator — Phase 2b''.
 *
 * Direct-ops SA that bypasses nbhd::select_and_apply. Implements MOVE
 * (70%), SWAP (20%), KEMPE (10%) inline with cached move_delta /
 * apply_move calls. Same temperature calibration, alias-weighted exam
 * selection, and reheat logic as solve_sa.
 *
 * Trade-off vs solve_sa: loses RoomBeam and Kick ops. In practice those
 * are small contributors — MOVE dominates, and the cached variant does
 * orders of magnitude more iters per second, winning on total work done.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_cached.h"
#include "greedy.h"
#include "neighbourhoods.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

inline AlgoResult solve_sa_cached(
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

    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); sol = g.sol.copy(); }
    EvalResult ev = fe.full_eval(sol);
    if (!ev.feasible()) { fe.recover_feasibility(sol, 500, seed); ev = fe.full_eval(sol); }

    CachedEvaluator Ecach(fe);
    Ecach.initialize(sol);
    auto refresh_around = [&](int eid) {
        Ecach.rebuild_contrib_for(eid, sol);
        for (auto& pr : prob.adj[eid]) Ecach.rebuild_contrib_for(pr.first, sol);
    };

    double current_fitness = ev.fitness();
    Solution best_sol = sol.copy();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();

    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Temperature calibration (sampling via cached evaluator)
    if (init_temp <= 0.0) {
        double avg_worsen = 0; int n_w = 0;
        for (int s = 0; s < 300; s++) {
            int eid = de(rng);
            if (valid_p[eid].empty() || valid_r[eid].empty()) continue;
            int pid = valid_p[eid][rng() % valid_p[eid].size()];
            int rid = valid_r[eid][rng() % valid_r[eid].size()];
            double d = Ecach.move_delta(sol, eid, pid, rid);
            if (d > 0 && d < 50000) { avg_worsen += d; n_w++; }
        }
        init_temp = (n_w > 0) ? std::max(1.0, (avg_worsen / n_w) / 0.693)
                               : std::max(100.0, ev.soft() * 0.005);
    }
    double temp = init_temp;

    // Alias table
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

    auto sa_accept = [&](double delta) -> bool {
        if (delta < 0) return true;
        if (temp > 1e-10) return unif(rng) < std::exp(-delta / temp);
        return false;
    };

    int no_improve = 0;
    int iters_done = 0;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;
        if (it % 200 == 0) { ev = fe.full_eval(sol); current_fitness = ev.fitness(); }
        if (it % 500 == 0) recompute_costs();

        double r_op = unif(rng);
        bool applied = false;
        double d_applied = 0;

        if (r_op < 0.70) {
            // MOVE: random exam, cost-weighted selection 70% of the time
            int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            auto& vp = valid_p[eid];
            auto& vr = valid_r[eid];
            if (!vp.empty() && !vr.empty()) {
                int pid = vp[rng() % vp.size()];
                int rid = vr[rng() % vr.size()];
                if (pid != sol.period_of[eid] || rid != sol.room_of[eid]) {
                    double d = Ecach.move_delta(sol, eid, pid, rid);
                    if (sa_accept(d)) {
                        Ecach.apply_move(sol, eid, pid, rid);
                        applied = true; d_applied = d;
                    }
                }
            }
        } else if (r_op < 0.90 && ne > 1) {
            // SWAP
            int eid = alias.sample(rng);
            int e2  = de(rng);
            if (e2 != eid) {
                int cp1 = sol.period_of[eid], cr1 = sol.room_of[eid];
                int cp2 = sol.period_of[e2],  cr2 = sol.room_of[e2];
                if (cp1 >= 0 && cp2 >= 0 && cp1 != cp2 &&
                    exam_dur[eid] <= period_dur[cp2] && exam_dur[e2] <= period_dur[cp1]) {
                    double d1 = Ecach.move_delta(sol, eid, cp2, cr1);
                    Ecach.apply_move(sol, eid, cp2, cr1);
                    double d2 = Ecach.move_delta(sol, e2, cp1, cr2);
                    double td = d1 + d2;
                    if (sa_accept(td)) {
                        Ecach.apply_move(sol, e2, cp1, cr2);
                        applied = true; d_applied = td;
                    } else {
                        Ecach.apply_move(sol, eid, cp1, cr1);
                    }
                }
            }
        } else {
            // KEMPE chain
            int eid = alias.sample(rng);
            int kp1 = sol.period_of[eid];
            if (kp1 >= 0) {
                int kp2 = std::uniform_int_distribution<int>(0, np - 1)(rng);
                if (kp2 != kp1) {
                    auto chain = kempe_detail::build_chain(sol, prob.adj, ne, eid, kp1, kp2);
                    if (!chain.empty() && (int)chain.size() <= ne / 4) {
                        auto old_pe = fe.partial_eval(sol, chain);
                        auto undo_info = kempe_detail::apply_chain(sol, chain, kp1, kp2);
                        auto new_pe = fe.partial_eval(sol, chain);
                        double kd = new_pe.fitness() - old_pe.fitness();
                        if (sa_accept(kd)) {
                            for (auto& u : undo_info) refresh_around(u.eid);
                            applied = true; d_applied = kd;
                        } else {
                            kempe_detail::undo_chain(sol, undo_info);
                        }
                    }
                }
            }
        }

        if (applied) {
            current_fitness += d_applied;
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
                }
                current_fitness = af;
            } else no_improve++;
        } else no_improve++;

        temp *= cooling;

        if (no_improve > 0 && no_improve % 1000 == 0) {
            sol = best_sol.copy();
            Ecach.initialize(sol);  // full rebuild after restart
            temp = init_temp * 0.3;
            ev = fe.full_eval(sol);
            current_fitness = ev.fitness();
            recompute_costs();
        }
    }

    fe.optimize_rooms(best_sol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[SACached] " << iters_done << " iters, " << rt << "s"
                  << " feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard() << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "SA (Cached, direct-ops)"};
}
