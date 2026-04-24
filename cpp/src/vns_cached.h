/*
 * VNS (General Variable Neighbourhood Search) with CachedEvaluator.
 *
 * Mirrors scalar vns.h's structure (8-level shake cycle, SA outer accept,
 * multi-op LS, mega-perturb escape, reheat) but routes the hot path
 * through CachedEvaluator + RecordingEvaluator for O(1) move_delta and
 * cache-coherent rollback.
 *
 * Fast path (shake levels 0-6 + LS):
 *   RecordingEvaluator wraps Ecach, logs every apply_move; on reject we
 *   iterate the log in reverse and apply inverse moves — cache stays
 *   coherent, no full rebuild.
 *
 * Slow path (shake level 7 D/R + mega-perturb):
 *   destroy_random / destroy_worst / repair_greedy mutate Solution directly
 *   via FastEvaluator. After these, Ecach.initialize(sol) rebuilds the
 *   cache. On reject we restore_state then rebuild. This matches what
 *   scalar VNS does, minus the cache resync.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_cached.h"
#include "recording_evaluator.h"
#include "greedy.h"
#include "neighbourhoods.h"
#include "alns.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

namespace vns_cached_detail {

constexpr int N_SHAKE = 8;
constexpr int MULTI_TRIAL_N = 15;

// Shake levels 0-6: templated on evaluator Ev (used with RecordingEvaluator).
// Level 7 (D/R) is hoisted to caller since it needs fe + cache resync.
template <typename Ev>
inline double shake_perturb_fast(
    Solution& sol, const Ev& ev,
    const ProblemInstance& prob,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    AliasTable& alias, std::mt19937& rng,
    int k)
{
    auto forced = [](double) { return true; };
    double cum = 0;

    switch (k) {
        case 0: { auto mr = nbhd::room_only(sol, ev, valid_r, alias, rng, forced);
                   if (mr.applied) cum += mr.delta; break; }
        case 1: { auto mr = nbhd::move_single(sol, ev, valid_p, valid_r, alias, rng, forced);
                   if (mr.applied) cum += mr.delta; break; }
        case 2: { auto mr = nbhd::swap_two(sol, ev, alias, rng, forced);
                   if (mr.applied) cum += mr.delta; break; }
        case 3: { auto mr = nbhd::kempe_chain(sol, ev, prob, alias, rng, forced);
                   if (mr.applied) cum += mr.delta; break; }
        case 4: { auto m1 = nbhd::move_single(sol, ev, valid_p, valid_r, alias, rng, forced);
                   if (m1.applied) cum += m1.delta;
                   auto m2 = nbhd::move_single(sol, ev, valid_p, valid_r, alias, rng, forced);
                   if (m2.applied) cum += m2.delta; break; }
        case 5: { auto m1 = nbhd::kempe_chain(sol, ev, prob, alias, rng, forced);
                   if (m1.applied) cum += m1.delta;
                   auto m2 = nbhd::move_single(sol, ev, valid_p, valid_r, alias, rng, forced);
                   if (m2.applied) cum += m2.delta; break; }
        case 6: { auto mr = nbhd::shake(sol, ev, valid_p, valid_r, 3, rng);
                   cum += mr.delta; break; }
        default: break;
    }
    return cum;
}

// Multi-operator LS descent, templated on Ev. Matches scalar VNS weights:
//   Move 35%, Swap 20%, Multi-trial 15%, Kempe 5%, RoomBeam 15%, RoomOnly 10%.
template <typename Ev>
inline int local_search(
    Solution& sol, const Ev& ev,
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
            auto mr = nbhd::move_single(sol, ev, valid_p, valid_r, alias, rng, improving);
            if (mr.applied) { fitness += mr.delta; improvements++; }

        } else if (r < 0.55) {
            auto mr = nbhd::swap_two(sol, ev, alias, rng, improving);
            if (mr.applied) { fitness += mr.delta; improvements++; }

        } else if (r < 0.70) {
            // Multi-trial move: try N random placements, pick best improving.
            int eid = (r01(rng) < 0.7) ? alias.sample(rng) : de(rng);
            auto& vp = valid_p[eid]; auto& vr = valid_r[eid];
            if (!vp.empty() && !vr.empty() && sol.period_of[eid] >= 0) {
                double best_d = 0;
                int best_p = -1, best_r = -1;
                for (int t = 0; t < MULTI_TRIAL_N; t++) {
                    int pid = vp[rng() % vp.size()];
                    int rid = vr[rng() % vr.size()];
                    if (pid == sol.period_of[eid] && rid == sol.room_of[eid]) continue;
                    double d = ev.move_delta(sol, eid, pid, rid);
                    if (d < best_d) { best_d = d; best_p = pid; best_r = rid; }
                }
                if (best_p >= 0 && best_d < -0.5) {
                    ev.apply_move(sol, eid, best_p, best_r);
                    fitness += best_d;
                    improvements++;
                }
            }

        } else if (r < 0.75) {
            auto mr = nbhd::kempe_chain(sol, ev, prob, alias, rng, improving);
            if (mr.applied) { fitness += mr.delta; improvements++; }

        } else if (r < 0.90) {
            auto mr = nbhd::room_beam(sol, ev, valid_r, alias, rng, improving);
            if (mr.applied) { fitness += mr.delta; improvements++; }

        } else {
            auto mr = nbhd::room_only(sol, ev, valid_r, alias, rng, improving);
            if (mr.applied) { fitness += mr.delta; improvements++; }
        }
    }
    return improvements;
}

} // namespace vns_cached_detail

inline AlgoResult solve_vns_cached(
    const ProblemInstance& prob,
    int max_iterations = 5000,
    int scan_budget    = 0,
    int seed           = 42,
    bool verbose       = false,
    const Solution* init_sol = nullptr)
{
    using namespace vns_cached_detail;
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

    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); sol = g.sol.copy(); }
    EvalResult ev = fe.full_eval(sol);
    if (!ev.feasible()) {
        if (verbose)
            std::cerr << "[VNSCached] Greedy infeasible (hard=" << ev.hard()
                      << "), running recovery..." << std::endl;
        fe.recover_feasibility(sol, 500, seed);
        ev = fe.full_eval(sol);
    }

    CachedEvaluator Ecach(fe);
    Ecach.initialize(sol);

    double fitness = ev.fitness();
    bool feasible = ev.feasible();
    Solution best_sol = sol.copy();
    double best_fitness = fitness;
    bool best_feasible = feasible;

    // SA temperature calibration (matches scalar VNS)
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
        std::cerr << "[VNSCached] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft()
                  << " budget=" << scan_budget
                  << " T0=" << (int)sa_temp << std::endl;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        double saved_fitness = fitness;
        bool used_dr = (k == N_SHAKE - 1);  // level 7 uses D/R, bypasses cache

        // Keep save_state as fallback for D/R iters (Rec can't rollback D/R
        // because destroy_random sets period_of to -1 via raw assignment).
        alns_detail::SavedState saved;
        RecordingEvaluator<CachedEvaluator> Rec(Ecach);

        if (used_dr) {
            saved = alns_detail::save_state(sol);
            int nd = std::max(1, (int)(ne * 0.04));
            auto removed = alns_detail::destroy_random(sol, ne, nd, rng);
            alns_detail::repair_greedy(sol, fe, removed, valid_p, valid_r);
            Ecach.initialize(sol);
            auto sev = fe.full_eval(sol);
            fitness = sev.fitness();
            feasible = sev.feasible();
        } else {
            double shake_d = shake_perturb_fast(
                sol, Rec, prob, valid_p, valid_r, alias, rng, k);
            fitness += shake_d;
        }

        // LS phase — always through Rec (fast rollback path)
        int ls_imps = local_search(
            sol, Rec, prob, valid_p, valid_r, alias, rng,
            scan_budget, fitness, ne);
        total_ls_imps += ls_imps;

        // Periodic resync
        if ((it + 1) % 200 == 0) {
            auto sev = fe.full_eval(sol);
            fitness = sev.fitness();
            feasible = sev.feasible();
        }

        // Accept or reject (SA outer acceptance)
        double delta = fitness - saved_fitness;
        bool improved = (delta < -0.5);

        if (improved) {
            Rec.commit();
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
                        std::cerr << "[VNSCached] Iter " << it
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
            Rec.commit();
            k = (k + 1) % N_SHAKE;
            stagnation++;
        } else {
            // Reject
            if (used_dr) {
                alns_detail::restore_state(sol, saved);
                Ecach.initialize(sol);
            } else {
                Rec.rollback_all(sol);  // cache-coherent fast rollback
            }
            fitness = saved_fitness;
            k = (k + 1) % N_SHAKE;
            stagnation++;
        }

        sa_temp *= sa_cooling;

        if ((it + 1) % 300 == 0)
            recompute_costs();

        if ((it + 1) % 1500 == 0 && stagnation > 500) {
            sa_temp = sa_init_temp * 0.3;
            if (verbose)
                std::cerr << "[VNSCached] Reheat at iter " << it
                          << " stagnation=" << stagnation << std::endl;
        }

        int mega_thresh = std::max(N_SHAKE * 60, ne);
        if (stagnation > 0 && stagnation % mega_thresh == 0) {
            int nd = std::max(2, (int)(ne * 0.08));
            auto removed = alns_detail::destroy_worst(sol, fe, ne, nd, rng);
            alns_detail::repair_greedy(sol, fe, removed, valid_p, valid_r);
            Ecach.initialize(sol);
            auto re = fe.full_eval(sol);
            fitness = re.fitness();
            feasible = re.feasible();

            if (!feasible && best_feasible) {
                sol = best_sol.copy();
                Ecach.initialize(sol);
                auto bev = fe.full_eval(sol);
                fitness = bev.fitness();
                feasible = bev.feasible();
                if (verbose)
                    std::cerr << "[VNSCached] Mega-perturb infeasible, restored best at iter " << it
                              << " (stagnation=" << stagnation << ")" << std::endl;
            } else if (verbose) {
                std::cerr << "[VNSCached] Mega-perturb at iter " << it
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
        std::cerr << "[VNSCached] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft()
                  << "  ls_improvements=" << total_ls_imps << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "VNS (Cached+GVNS)"};
}
