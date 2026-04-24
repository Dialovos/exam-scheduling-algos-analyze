/*
 * ALNS with CachedEvaluator — Phase 2b'' drop-in.
 *
 * Strategy:
 *   • Destroy/repair phase rewrites many exam slots via sol.assign.
 *     After it completes, we rebuild the cache for every *affected* exam
 *     (the destroyed set + their 1-hop neighbourhood). This is O(k × np
 *     × deg) where k is n_destroy, typically ~4% of ne.
 *   • Local-search polish phase uses Ecach.move_delta (the hot path).
 *   • All intermediate state-changing calls route through Ecach.apply_move
 *     so the cache stays in sync.
 *
 * Note: ALNS' eval load is dominated by destroy/repair state ops and
 * full_eval, not move_delta, so the cache win here is smaller than Tabu
 * (~1.5-2× typical). Still worth integrating for the portfolio.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_cached.h"
#include "greedy.h"
#include "alns.h"  // reuse alns_detail helpers

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

inline AlgoResult solve_alns_cached(
    const ProblemInstance& prob,
    int max_iterations  = 2000,
    double destroy_pct  = 0.04,
    int seed            = 42,
    bool verbose        = false,
    const Solution* init_sol = nullptr)
{
    using namespace alns_detail;
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

    double current_fitness = ev.fitness();
    Solution best_sol = sol.copy();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();
    int current_hard = ev.hard();

    int n_destroy_base = std::max(1, (int)(ne * destroy_pct));
    int n_destroy = n_destroy_base;
    int no_improve_alns = 0;

    // Temperature calibration (cached)
    double temp;
    {
        double avg_worsen = 0; int n_w = 0;
        std::uniform_int_distribution<int> sde(0, ne - 1);
        for (int s = 0; s < 200; s++) {
            int eid = sde(rng);
            if (valid_p[eid].empty() || valid_r[eid].empty()) continue;
            int pid = valid_p[eid][rng() % valid_p[eid].size()];
            int rid = valid_r[eid][rng() % valid_r[eid].size()];
            double d = Ecach.move_delta(sol, eid, pid, rid);
            if (d > 0 && d < 50000) { avg_worsen += d; n_w++; }
        }
        double base_temp = (n_w > 0) ? (avg_worsen / n_w) / 0.693 : 100.0;
        temp = std::max(1.0, base_temp * std::sqrt((double)n_destroy));
    }
    double init_temp = temp;
    double cooling_rate = 0.999;

    std::vector<double> d_weights = {1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> r_weights = {1.0, 1.0, 1.0};
    std::vector<double> d_scores(5, 0); std::vector<double> r_scores(3, 0);
    std::vector<int> d_counts(5, 1);    std::vector<int> r_counts(3, 1);

    int lahc_len = std::max(ne / 10, 20);
    std::vector<double> lahc_history(lahc_len, current_fitness);
    bool use_lahc = false;
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    int iters_done = 0;
    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        int d_op = roulette(d_weights, rng);
        int r_op = roulette(r_weights, rng);

        auto saved = save_state(sol);

        // Destroy (bypasses cache — all exam changes are captured below)
        std::vector<int> removed;
        if (d_op == 0)      removed = destroy_random(sol, ne, n_destroy, rng);
        else if (d_op == 1) removed = destroy_worst(sol, fe, ne, n_destroy, rng);
        else if (d_op == 2) removed = destroy_related(sol, prob, ne, n_destroy, rng);
        else if (d_op == 3) removed = destroy_shaw(sol, prob, ne, n_destroy, rng);
        else                removed = destroy_period_strip(sol, ne, np, n_destroy, rng);

        // Repair (also bypasses cache)
        if (r_op == 0)      repair_greedy(sol, fe, removed, valid_p, valid_r);
        else if (r_op == 1) repair_random(sol, removed, valid_p, valid_r, rng);
        else                repair_regret2(sol, fe, removed, valid_p, valid_r);

        // Rebuild cache for affected exams: removed set + their neighbours.
        std::vector<uint8_t> touched(ne, 0);
        for (int e : removed) touched[e] = 1;
        for (int e : removed)
            for (auto& pr : prob.adj[e]) touched[pr.first] = 1;
        for (int e = 0; e < ne; e++)
            if (touched[e]) Ecach.rebuild_contrib_for(e, sol);

        // Local-search polish (cached hot path)
        for (int eid : removed) {
            const auto& vp = valid_p[eid];
            const auto& vr = valid_r[eid];
            if (vp.empty() || vr.empty()) continue;
            for (int t = 0; t < 3; t++) {
                int pid = vp[rng() % vp.size()];
                int rid = vr[rng() % vr.size()];
                if (pid == sol.period_of[eid] && rid == sol.room_of[eid]) continue;
                double d = Ecach.move_delta(sol, eid, pid, rid);
                if (d < -0.5) { Ecach.apply_move(sol, eid, pid, rid); break; }
            }
        }

        int new_hard = fe.count_hard_fast(sol);
        bool fast_reject = false;
        if (new_hard > current_hard) {
            double hard_delta = (double)(new_hard - current_hard) * 100000.0;
            if (temp < 1e-5 || std::exp(-hard_delta / temp) < 0.001) fast_reject = true;
        }

        double score = 0.0;
        if (fast_reject) {
            restore_state(sol, saved);
            // Restore → rebuild cache again (cheap: only touched exams)
            for (int e = 0; e < ne; e++) if (touched[e]) Ecach.rebuild_contrib_for(e, sol);
            no_improve_alns++;
        } else {
            auto new_ev = fe.full_eval(sol);
            double new_fitness = new_ev.fitness();
            double delta = new_fitness - current_fitness;
            bool accept;
            if (use_lahc) {
                int hi = it % lahc_len;
                accept = (new_fitness <= current_fitness) || (new_fitness <= lahc_history[hi]);
                lahc_history[hi] = current_fitness;
            } else {
                accept = (delta < 0);
                if (!accept && temp > 1e-10)
                    accept = (unif(rng) < std::exp(-delta / temp));
            }
            if (accept) {
                current_fitness = new_fitness;
                current_hard = new_ev.hard();
                score = 1.0;
                bool nf = new_ev.feasible();
                bool dominated = (best_feasible && !nf);
                if (!dominated && new_fitness < best_fitness) {
                    best_sol = sol.copy();
                    best_fitness = new_fitness;
                    best_feasible = nf;
                    score = 3.0;
                    no_improve_alns = 0;
                    n_destroy = n_destroy_base;
                } else no_improve_alns++;
            } else {
                restore_state(sol, saved);
                for (int e = 0; e < ne; e++) if (touched[e]) Ecach.rebuild_contrib_for(e, sol);
                no_improve_alns++;
            }
        }

        d_scores[d_op] += score; d_counts[d_op]++;
        r_scores[r_op] += score; r_counts[r_op]++;

        // Operator weight update (light)
        if (it > 0 && it % 50 == 0) {
            for (int k = 0; k < 5; k++) {
                d_weights[k] = 0.7 * d_weights[k] + 0.3 * (d_scores[k] / std::max(1, d_counts[k]));
                if (d_weights[k] < 0.1) d_weights[k] = 0.1;
            }
            for (int k = 0; k < 3; k++) {
                r_weights[k] = 0.7 * r_weights[k] + 0.3 * (r_scores[k] / std::max(1, r_counts[k]));
                if (r_weights[k] < 0.1) r_weights[k] = 0.1;
            }
        }

        temp *= cooling_rate;
        if (no_improve_alns > 0 && no_improve_alns % 300 == 0) {
            n_destroy = std::min(ne / 2, n_destroy * 2);
            temp = init_temp * 0.5;
        }
    }

    fe.optimize_rooms(best_sol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[ALNSCached] " << iters_done << " iters, " << rt << "s"
                  << " feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard() << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "ALNS (Cached)"};
}
