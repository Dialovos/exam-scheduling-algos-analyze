/*
 * Great deluge with non-linear decay and dual-level tracking.
 *
 * Level-based acceptance: accepts any move where new fitness <= level.
 * 3-phase decay: exponential -> linear -> oscillating.
 *
 * Hard and soft levels tracked separately.
 * Kempe chain neighbourhood on 20% of moves.
 * Adaptive raise proportional to distance from best when stuck.
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

inline AlgoResult solve_great_deluge(
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

    // Init from greedy
    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); sol = g.sol.copy(); }

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

    // Dual levels: separate hard and soft tracking
    double hard_level = ev.hard() * 100000.0 * 1.1;
    double soft_level = ev.soft() * 1.2;
    double hard_decay = hard_level / (max_iterations * 0.3 + 1); // eliminate hard in first 30%
    double soft_decay_rate = decay_rate;
    if (soft_decay_rate <= 0.0)
        soft_decay_rate = (soft_level - ev.soft() * 0.3) / std::max(max_iterations, 1);

    // Combined level (for backward compat + simple acceptance path)
    double level = current_fitness * 1.1;
    if (decay_rate <= 0.0)
        decay_rate = (level - current_fitness * 0.3) / std::max(max_iterations, 1);

    if (verbose)
        std::cerr << "[GD] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft()
                  << " soft_level=" << soft_level << std::endl;

    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Per-exam cost for weighted selection via alias table (O(1) sample)
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

        if (it % 200 == 0) {
            ev = fe.full_eval(sol);
            current_fitness = ev.fitness();
        }
        if (it % 500 == 0) recompute_costs();

        // ── Neighbourhood selection: 20% Kempe, 25% swap, 55% steepest ──
        double r_move = unif(rng);
        bool move_applied = false;
        double move_delta_val = 0;

        if (r_move < 0.20) {
            // Kempe chain move
            int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            int kp1 = sol.period_of[eid]; if (kp1 < 0) goto skip_move;
            {
                int kp2 = std::uniform_int_distribution<int>(0, np - 1)(rng);
                if (kp2 == kp1) goto skip_move;

                auto chain = kempe_detail::build_chain(sol, prob.adj, ne, eid, kp1, kp2);
                if (chain.empty() || (int)chain.size() > ne / 4) goto skip_move;

                auto old_pe = fe.partial_eval(sol, chain);
                auto undo_info = kempe_detail::apply_chain(sol, chain, kp1, kp2);
                auto new_pe = fe.partial_eval(sol, chain);
                double kd = new_pe.fitness() - old_pe.fitness();

                double nf = current_fitness + kd;
                double new_hard_val = std::max(0.0, nf - (nf - (double)fe.count_hard_fast(sol) * 100000.0));
                double new_soft_val = nf - (double)fe.count_hard_fast(sol) * 100000.0;
                bool accept = (new_hard_val <= hard_level && new_soft_val <= soft_level) || (nf <= level);

                if (accept) {
                    current_fitness = nf;
                    move_applied = true;
                    move_delta_val = kd;
                } else {
                    kempe_detail::undo_chain(sol, undo_info);
                }
            }
        } else if (r_move < 0.45 && ne > 1) {
            // Swap move
            int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            int e2 = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            if (e2 == eid) goto skip_move;
            {
                int cp1 = sol.period_of[eid], cr1 = sol.room_of[eid];
                int cp2 = sol.period_of[e2], cr2 = sol.room_of[e2];
                if (cp1 < 0 || cp2 < 0 || cp1 == cp2) goto skip_move;
                if (exam_dur[eid] > period_dur[cp2] || exam_dur[e2] > period_dur[cp1]) goto skip_move;
                double d1 = fe.move_delta(sol, eid, cp2, cr1);
                fe.apply_move(sol, eid, cp2, cr1);
                double d2 = fe.move_delta(sol, e2, cp1, cr2);
                double td = d1 + d2;
                double nf = current_fitness + td;
                if (nf <= level) {
                    fe.apply_move(sol, e2, cp1, cr2);
                    current_fitness = nf;
                    move_applied = true;
                    move_delta_val = td;
                } else {
                    fe.apply_move(sol, eid, cp1, cr1); // undo first move
                }
            }
        } else {
            // Steepest period+room move (existing logic)
            int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            auto& vp = valid_p[eid];
            auto& vr = valid_r[eid];
            if (vp.empty() || vr.empty()) goto skip_move;

            int cp = sol.period_of[eid];
            int old_rid = sol.room_of[eid];
            int enr = fe.exam_enroll[eid];
            int old_total_pr = sol.get_pr_enroll(cp, old_rid);
            double dh_old_room = -(((old_total_pr > fe.room_cap[old_rid]) ? 1.0 : 0.0) -
                                   (((old_total_pr - enr) > fe.room_cap[old_rid]) ? 1.0 : 0.0));

            double best_delta = 1e18;
            int best_pid = -1, best_rid = -1;

            for (int pid : vp) {
                if (pid == cp) continue;
                double dh_p = 0, ds_p = 0;
                int dur_e = fe.exam_dur[eid];
                if (dur_e > fe.period_dur[cp])  dh_p -= 1;
                if (dur_e > fe.period_dur[pid]) dh_p += 1;
                dh_p += dh_old_room;

                int old_day = fe.period_day[cp], old_dpos = fe.period_daypos[cp];
                int new_day = fe.period_day[pid], new_dpos = fe.period_daypos[pid];
                for (int s : prob.exams[eid].students) {
                    for (int other : prob.student_exams[s]) {
                        if (other == eid) continue;
                        int o_pid = sol.period_of[other];
                        if (o_pid < 0) continue;
                        if (o_pid == cp)  dh_p -= 1;
                        if (o_pid == pid) dh_p += 1;
                        int o_day = fe.period_day[o_pid], o_dpos = fe.period_daypos[o_pid];
                        if (old_day == o_day) {
                            int g = std::abs(old_dpos - o_dpos);
                            if (g == 1) ds_p -= fe.w_2row; else if (g > 1) ds_p -= fe.w_2day;
                        }
                        int og = std::abs(cp - o_pid);
                        if (og > 0 && og <= fe.w_spread) ds_p -= 1;
                        if (new_day == o_day) {
                            int g = std::abs(new_dpos - o_dpos);
                            if (g == 1) ds_p += fe.w_2row; else if (g > 1) ds_p += fe.w_2day;
                        }
                        int ng = std::abs(pid - o_pid);
                        if (ng > 0 && ng <= fe.w_spread) ds_p += 1;
                    }
                }
                ds_p += fe.period_pen[pid] - fe.period_pen[cp];
                ds_p -= fe.room_pen[old_rid];
                if (fe.large_exams.count(eid) && fe.fl_penalty > 0) {
                    if (fe.last_periods.count(cp) && !fe.last_periods.count(pid)) ds_p -= fe.fl_penalty;
                    else if (!fe.last_periods.count(cp) && fe.last_periods.count(pid)) ds_p += fe.fl_penalty;
                }
                for (auto& [other, tcode] : fe.phc_by_exam[eid]) {
                    int o_pid = sol.period_of[other]; if (o_pid < 0) continue;
                    if (tcode == 0)      { if (cp != o_pid) dh_p -= 1; if (pid != o_pid) dh_p += 1; }
                    else if (tcode == 1) { if (cp == o_pid) dh_p -= 1; if (pid == o_pid) dh_p += 1; }
                    else if (tcode == 2) { if (cp <= o_pid) dh_p -= 1; if (pid <= o_pid) dh_p += 1; }
                }

                for (int rid : vr) {
                    double dh = dh_p, ds = ds_p;
                    int new_total = sol.get_pr_enroll(pid, rid);
                    dh += (((new_total + enr) > fe.room_cap[rid]) ? 1.0 : 0.0) -
                          ((new_total > fe.room_cap[rid]) ? 1.0 : 0.0);
                    ds += fe.room_pen[rid];
                    double d = dh * 100000.0 + ds;
                    if (d < best_delta) {
                        best_delta = d; best_pid = pid; best_rid = rid;
                    }
                }
            }

            if (best_pid < 0) goto skip_move;
            double nf = current_fitness + best_delta;
            if (nf <= level) {
                fe.apply_move(sol, eid, best_pid, best_rid);
                current_fitness = nf;
                move_applied = true;
                move_delta_val = best_delta;
            }
        }

        skip_move:
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

        // Non-linear 3-phase level decay
        double progress = (double)it / max_iterations;
        double effective_decay;
        if (progress < 0.3) {
            effective_decay = decay_rate * std::exp(-3.0 * progress);
        } else if (progress < 0.7) {
            effective_decay = decay_rate;
        } else {
            double osc = std::sin(progress * 50.0) * decay_rate * 0.3;
            effective_decay = decay_rate * 0.5 + osc;
        }
        level -= effective_decay;
        hard_level -= hard_decay;
        soft_level -= soft_decay_rate;
        hard_level = std::max(0.0, hard_level);
        soft_level = std::max(0.0, soft_level);

        // Adaptive raise when stuck (proportional to distance from best)
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
        std::cerr << "[GD] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "Great Deluge"};
}