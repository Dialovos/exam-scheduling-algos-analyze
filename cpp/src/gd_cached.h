/*
 * Great Deluge with CachedEvaluator — Phase 2b'' drop-in.
 *
 * Algorithm identical to solve_great_deluge. Only evaluator calls change:
 *   fe.move_delta → Ecach.move_delta     (O(1) per call)
 *   fe.apply_move → Ecach.apply_move     (updates cache)
 *   Kempe path still uses fe.partial_eval for the chain-fitness delta;
 *     after accept we refresh cache for all chain members.
 *
 * Steepest-period path's inlined student double-loop is gone — replaced
 * by a direct Ecach.move_delta(sol, eid, pid, rid) call per (pid,rid).
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

inline AlgoResult solve_great_deluge_cached(
    const ProblemInstance& prob,
    int max_iterations = 5000,
    double decay_rate  = 0.0,
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

    double hard_level = ev.hard() * 100000.0 * 1.1;
    double soft_level = ev.soft() * 1.2;
    double hard_decay = hard_level / (max_iterations * 0.3 + 1);
    double soft_decay_rate = decay_rate;
    if (soft_decay_rate <= 0.0)
        soft_decay_rate = (soft_level - ev.soft() * 0.3) / std::max(max_iterations, 1);

    double level = current_fitness * 1.1;
    if (decay_rate <= 0.0)
        decay_rate = (level - current_fitness * 0.3) / std::max(max_iterations, 1);

    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

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

    int no_improve = 0;
    int iters_done = 0;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;
        if (it % 200 == 0) { ev = fe.full_eval(sol); current_fitness = ev.fitness(); }
        if (it % 500 == 0) recompute_costs();

        double r_move = unif(rng);
        bool move_applied = false;

        if (r_move < 0.20) {
            // Kempe chain
            int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
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
                        double nf = current_fitness + kd;
                        double hard_val = (double)fe.count_hard_fast(sol) * 100000.0;
                        double soft_val = nf - hard_val;
                        bool accept = (hard_val <= hard_level && soft_val <= soft_level) || (nf <= level);
                        if (accept) {
                            // Refresh cache for chain + neighbours
                            for (auto& u : undo_info) refresh_around(u.eid);
                            current_fitness = nf;
                            move_applied = true;
                        } else {
                            kempe_detail::undo_chain(sol, undo_info);
                        }
                    }
                }
            }
        } else if (r_move < 0.45 && ne > 1) {
            // Swap
            int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            int e2  = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            if (e2 != eid) {
                int cp1 = sol.period_of[eid], cr1 = sol.room_of[eid];
                int cp2 = sol.period_of[e2],  cr2 = sol.room_of[e2];
                if (cp1 >= 0 && cp2 >= 0 && cp1 != cp2 &&
                    exam_dur[eid] <= period_dur[cp2] && exam_dur[e2] <= period_dur[cp1]) {
                    double d1 = Ecach.move_delta(sol, eid, cp2, cr1);
                    Ecach.apply_move(sol, eid, cp2, cr1);
                    double d2 = Ecach.move_delta(sol, e2, cp1, cr2);
                    double td = d1 + d2;
                    double nf = current_fitness + td;
                    if (nf <= level) {
                        Ecach.apply_move(sol, e2, cp1, cr2);
                        current_fitness = nf;
                        move_applied = true;
                    } else {
                        Ecach.apply_move(sol, eid, cp1, cr1); // undo
                    }
                }
            }
        } else {
            // Steepest period+room with period-first decomposition (cached)
            int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            auto& vp = valid_p[eid];
            auto& vr = valid_r[eid];
            if (!vp.empty() && !vr.empty()) {
                int cp = sol.period_of[eid];
                int enr = fe.exam_enroll[eid];
                double best_delta = 1e18;
                int best_pid = -1, best_rid = -1;
                for (int pid : vp) {
                    if (pid == cp) continue;
                    auto pd = Ecach.move_delta_period(sol, eid, pid);
                    for (int rid : vr) {
                        double dh = pd.dh;
                        int new_total = sol.get_pr_enroll(pid, rid);
                        dh += (((new_total + enr) > fe.room_cap[rid]) ? 1.0 : 0.0) -
                              ((new_total > fe.room_cap[rid]) ? 1.0 : 0.0);
                        double ds = pd.ds + fe.room_pen[rid];
                        double d = dh * 100000.0 + ds;
                        if (d < best_delta) { best_delta = d; best_pid = pid; best_rid = rid; }
                    }
                }
                if (best_pid >= 0) {
                    double nf = current_fitness + best_delta;
                    if (nf <= level) {
                        Ecach.apply_move(sol, eid, best_pid, best_rid);
                        current_fitness = nf;
                        move_applied = true;
                    }
                }
            }
        }

        if (move_applied) {
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

        double progress = (double)it / max_iterations;
        double effective_decay;
        if (progress < 0.3)      effective_decay = decay_rate * std::exp(-3.0 * progress);
        else if (progress < 0.7) effective_decay = decay_rate;
        else {
            double osc = std::sin(progress * 50.0) * decay_rate * 0.3;
            effective_decay = decay_rate * 0.5 + osc;
        }
        level -= effective_decay;
        hard_level = std::max(0.0, hard_level - hard_decay);
        soft_level = std::max(0.0, soft_level - soft_decay_rate);

        if (no_improve > 0 && no_improve % 500 == 0) {
            double dist = current_fitness / std::max(1.0, best_fitness) - 1.0;
            double raise_pct = std::min(0.15, 0.02 + dist * 0.1);
            level = current_fitness * (1.0 + raise_pct);
            soft_level = std::max(soft_level,
                (current_fitness - (double)fe.count_hard_fast(sol) * 100000.0) * (1.0 + raise_pct));
        }
    }

    fe.optimize_rooms(best_sol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[GDCached] " << iters_done << " iters, " << rt << "s"
                  << " feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard() << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "Great Deluge (Cached)"};
}
