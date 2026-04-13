/*
 * General variable neighbourhood search (GVNS).
 *
 * Systematic shake cycling with SA outer acceptance and multi-operator
 * random descent for local search.
 *
 * Structure:
 *   Outer — 8 shake levels (room -> move -> swap -> kempe -> compound -> D/R)
 *   Inner — Random descent: move, swap, multi-trial move, kempe, room ops
 *   Accept — SA-based; improvements reset k=0, SA-accepted keep cycling
 *   Escape — Mega-perturbation on deep stagnation (destroy-worst + repair)
 *
 * Multi-trial move replaces exhaustive kick: O(15) vs O(np*nr) per call.
 * Kempe at 5% in LS (expensive on dense graphs, but effective).
 * SA outer acceptance prevents stalling from strict-improvement VNS.
 *
 * Reuses nbhd:: operators, kempe_detail:: chains, alns_detail:: D/R.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"
#include "neighbourhoods.h"
#include "alns.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

namespace vns_detail {

constexpr int N_SHAKE = 8;
constexpr int MULTI_TRIAL_N = 15;  // random placements per multi-trial move

// ── Shaking ──────────────────────────────────────────────────

inline double shake_perturb(
    Solution& sol, const FastEvaluator& fe,
    const ProblemInstance& prob,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    AliasTable& alias, std::mt19937& rng,
    int k, int ne)
{
    auto forced = [](double) { return true; };
    double cum = 0;

    switch (k) {
        case 0: { auto mr = nbhd::room_only(sol, fe, valid_r, alias, rng, forced);
                   if (mr.applied) cum += mr.delta; break; }
        case 1: { auto mr = nbhd::move_single(sol, fe, valid_p, valid_r, alias, rng, forced);
                   if (mr.applied) cum += mr.delta; break; }
        case 2: { auto mr = nbhd::swap_two(sol, fe, alias, rng, forced);
                   if (mr.applied) cum += mr.delta; break; }
        case 3: { auto mr = nbhd::kempe_chain(sol, fe, prob, alias, rng, forced);
                   if (mr.applied) cum += mr.delta; break; }
        case 4: { auto m1 = nbhd::move_single(sol, fe, valid_p, valid_r, alias, rng, forced);
                   if (m1.applied) cum += m1.delta;
                   auto m2 = nbhd::move_single(sol, fe, valid_p, valid_r, alias, rng, forced);
                   if (m2.applied) cum += m2.delta; break; }
        case 5: { auto m1 = nbhd::kempe_chain(sol, fe, prob, alias, rng, forced);
                   if (m1.applied) cum += m1.delta;
                   auto m2 = nbhd::move_single(sol, fe, valid_p, valid_r, alias, rng, forced);
                   if (m2.applied) cum += m2.delta; break; }
        case 6: { auto mr = nbhd::shake(sol, fe, valid_p, valid_r, 3, rng);
                   cum += mr.delta; break; }
        case 7: { int nd = std::max(1, (int)(ne * 0.04));
                   auto removed = alns_detail::destroy_random(sol, ne, nd, rng);
                   alns_detail::repair_greedy(sol, fe, removed, valid_p, valid_r);
                   return 0; }
    }
    return cum;
}

// ── Local search: multi-operator random descent ──────────────
// Operators:
//   Move (35%)        — random single-exam relocate
//   Swap (20%)        — exchange periods of two exams
//   Multi-trial (15%) — try N random placements for one exam, pick best improving
//   Kempe (5%)        — BFS chain swap
//   RoomBeam (15%)    — steepest-descent room for one exam
//   RoomOnly (10%)    — random room change

inline int local_search(
    Solution& sol, const FastEvaluator& fe,
    const ProblemInstance& prob,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    AliasTable& alias, std::mt19937& rng,
    int budget, double& fitness, int ne)
{
    int improvements = 0;
    auto improving = [](double d) { return d < -0.5; };
    std::uniform_real_distribution<double> r01(0.0, 1.0);
    std::uniform_int_distribution<int> de(0, ne - 1);

    for (int s = 0; s < budget; s++) {
        double r = r01(rng);

        if (r < 0.35) {
            auto mr = nbhd::move_single(sol, fe, valid_p, valid_r, alias, rng, improving);
            if (mr.applied) { fitness += mr.delta; improvements++; }

        } else if (r < 0.55) {
            auto mr = nbhd::swap_two(sol, fe, alias, rng, improving);
            if (mr.applied) { fitness += mr.delta; improvements++; }

        } else if (r < 0.70) {
            // Multi-trial move: try N random placements, pick best improving
            int eid = (r01(rng) < 0.7) ? alias.sample(rng) : de(rng);
            auto& vp = valid_p[eid]; auto& vr = valid_r[eid];
            if (!vp.empty() && !vr.empty() && sol.period_of[eid] >= 0) {
                double best_d = 0;
                int best_p = -1, best_r = -1;
                for (int t = 0; t < MULTI_TRIAL_N; t++) {
                    int pid = vp[rng() % vp.size()];
                    int rid = vr[rng() % vr.size()];
                    if (pid == sol.period_of[eid] && rid == sol.room_of[eid]) continue;
                    double d = fe.move_delta(sol, eid, pid, rid);
                    if (d < best_d) { best_d = d; best_p = pid; best_r = rid; }
                }
                if (best_p >= 0 && best_d < -0.5) {
                    fe.apply_move(sol, eid, best_p, best_r);
                    fitness += best_d;
                    improvements++;
                }
            }

        } else if (r < 0.75) {
            auto mr = nbhd::kempe_chain(sol, fe, prob, alias, rng, improving);
            if (mr.applied) { fitness += mr.delta; improvements++; }

        } else if (r < 0.90) {
            auto mr = nbhd::room_beam(sol, fe, valid_r, alias, rng, improving);
            if (mr.applied) { fitness += mr.delta; improvements++; }

        } else {
            auto mr = nbhd::room_only(sol, fe, valid_r, alias, rng, improving);
            if (mr.applied) { fitness += mr.delta; improvements++; }
        }
    }
    return improvements;
}

} // namespace vns_detail

// ── Public interface ─────────────────────────────────────────

inline AlgoResult solve_vns(
    const ProblemInstance& prob,
    int max_iterations = 5000,
    int scan_budget    = 0,
    int seed           = 42,
    bool verbose       = false,
    const Solution* init_sol = nullptr)
{
    using namespace vns_detail;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    FastEvaluator fe(prob);

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++)
            if (fe.exam_dur[e] <= fe.period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++)
            if (fe.exam_enroll[e] <= fe.room_cap[r]) valid_r[e].push_back(r);
    }

    if (scan_budget <= 0)
        scan_budget = std::clamp(ne / 10, 15, 50);

    // ── Init solution ──
    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); sol = g.sol.copy(); }

    EvalResult ev = fe.full_eval(sol);
    if (!ev.feasible()) {
        if (verbose)
            std::cerr << "[VNS] Greedy infeasible (hard=" << ev.hard()
                      << "), running recovery..." << std::endl;
        fe.recover_feasibility(sol, 500, seed);
        ev = fe.full_eval(sol);
    }

    double fitness = ev.fitness();
    bool feasible = ev.feasible();
    Solution best_sol = sol.copy();
    double best_fitness = fitness;
    bool best_feasible = feasible;

    // ── SA temperature calibration ──
    double sa_temp;
    {
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
        sa_temp = (n_w > 0) ? std::max(1.0, (avg_worsen / n_w) / 0.693) : 100.0;
        sa_temp = std::max(sa_temp, ev.soft() * 0.005);
    }
    double sa_init_temp = sa_temp;
    double sa_cooling = std::pow(0.01, 1.0 / max_iterations);

    // ── Alias table ──
    std::vector<double> exam_cost(ne, 1.0);
    AliasTable alias;
    auto recompute_costs = [&]() {
        for (int e = 0; e < ne; e++) {
            int pid = sol.period_of[e];
            if (pid < 0) { exam_cost[e] = 1.0; continue; }
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

    int k = 0;
    int stagnation = 0;
    int iters_done = 0;
    int total_ls_imps = 0;
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    if (verbose)
        std::cerr << "[VNS] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft()
                  << " budget=" << scan_budget
                  << " T0=" << (int)sa_temp << std::endl;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        auto saved = alns_detail::save_state(sol);
        double saved_fitness = fitness;

        // ── Phase 1: Shake at level k ──
        double shake_d = shake_perturb(
            sol, fe, prob, valid_p, valid_r, alias, rng, k, ne);
        fitness += shake_d;

        if (k == N_SHAKE - 1) {
            auto sev = fe.full_eval(sol);
            fitness = sev.fitness();
            feasible = sev.feasible();
        }

        // ── Phase 2: Local search descent ──
        int ls_imps = local_search(
            sol, fe, prob, valid_p, valid_r, alias, rng,
            scan_budget, fitness, ne);
        total_ls_imps += ls_imps;

        // Periodic resync
        if ((it + 1) % 200 == 0) {
            auto sev = fe.full_eval(sol);
            fitness = sev.fitness();
            feasible = sev.feasible();
        }

        // ── Phase 3: Accept or reject (SA outer acceptance) ──
        double delta = fitness - saved_fitness;
        bool improved = (delta < -0.5);

        if (improved) {
            k = 0;

            if (fitness < best_fitness - 0.5) {
                auto check = fe.full_eval(sol);
                fitness = check.fitness();
                feasible = check.feasible();
                bool dominated = (best_feasible && !feasible);
                if (!dominated && fitness < best_fitness) {
                    best_sol = sol.copy();
                    best_fitness = fitness;
                    best_feasible = feasible;
                    stagnation = 0;
                    if (verbose && (it < 20 || it % 500 == 0))
                        std::cerr << "[VNS] Iter " << it
                                  << ": best hard=" << check.hard()
                                  << " soft=" << check.soft()
                                  << " ls=" << ls_imps << std::endl;
                } else {
                    stagnation++;
                }
            } else {
                stagnation++;
            }
        } else if (delta > 0 && sa_temp > 1e-10 &&
                   unif(rng) < std::exp(-delta / sa_temp)) {
            k = (k + 1) % N_SHAKE;
            stagnation++;
        } else {
            alns_detail::restore_state(sol, saved);
            fitness = saved_fitness;
            k = (k + 1) % N_SHAKE;
            stagnation++;
        }

        sa_temp *= sa_cooling;

        // ── Maintenance ──

        if ((it + 1) % 300 == 0)
            recompute_costs();

        if ((it + 1) % 1500 == 0 && stagnation > 500) {
            sa_temp = sa_init_temp * 0.3;
            if (verbose)
                std::cerr << "[VNS] Reheat at iter " << it
                          << " stagnation=" << stagnation << std::endl;
        }

        // Mega-perturb threshold scales with problem size: small instances
        // trigger at 480 iters, large ones get more time before perturbing.
        int mega_thresh = std::max(N_SHAKE * 60, ne);
        if (stagnation > 0 && stagnation % mega_thresh == 0) {
            int nd = std::max(2, (int)(ne * 0.08));
            auto removed = alns_detail::destroy_worst(sol, fe, ne, nd, rng);
            alns_detail::repair_greedy(sol, fe, removed, valid_p, valid_r);
            auto re = fe.full_eval(sol);
            fitness = re.fitness();
            feasible = re.feasible();

            // If mega-perturb broke feasibility and we have a feasible best,
            // restart from best — don't let a bad perturb strand us in
            // infeasible territory permanently.
            if (!feasible && best_feasible) {
                sol = best_sol.copy();
                auto bev = fe.full_eval(sol);
                fitness = bev.fitness();
                feasible = bev.feasible();
                if (verbose)
                    std::cerr << "[VNS] Mega-perturb infeasible, restored best at iter " << it
                              << " (stagnation=" << stagnation << ")" << std::endl;
            } else if (verbose) {
                std::cerr << "[VNS] Mega-perturb at iter " << it
                          << " (stagnation=" << stagnation << ")"
                          << " fitness=" << (int)fitness << std::endl;
            }

            k = 0;
            stagnation = 1;
            recompute_costs();
        }
    }

    fe.optimize_rooms(best_sol);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[VNS] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft()
                  << "  ls_improvements=" << total_ls_imps << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "GVNS"};
}
