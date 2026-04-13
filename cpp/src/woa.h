/*
 * Discrete whale optimization + variable neighbourhood descent (VND).
 *
 * Bubble-net hunting adapted for discrete timetabling:
 *   1. Encircle: move exams toward leader's assignments
 *   2. Bubble-net spiral: Kempe chain swaps toward leader
 *   3. Random search: blind perturbation for exploration
 *
 * VND post-update on each whale:
 *   Move -> Swap -> Kempe -> Kick -> Room (first-improvement restarts from Move)
 *
 * Reference: 2025 IJACSA
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

namespace woa_detail {

struct Whale {
    Solution sol;
    double fitness;
    bool feasible;
};

// Random solution (same pattern as abc_detail)
inline Solution random_solution(
    const ProblemInstance& prob, const FastEvaluator& fe,
    int ne, int np, int nr,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    std::mt19937& rng)
{
    Solution sol;
    sol.init(prob);

    std::vector<int> order(ne);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    for (int eid : order) {
        std::vector<bool> blocked(np, false);
        for (auto& [nb, _] : prob.adj[eid])
            if (sol.period_of[nb] >= 0) blocked[sol.period_of[nb]] = true;

        std::vector<int> avail;
        for (int p : valid_p[eid])
            if (!blocked[p]) avail.push_back(p);
        if (avail.empty()) avail = valid_p[eid];
        if (avail.empty())
            for (int p = 0; p < np; p++) avail.push_back(p);

        std::shuffle(avail.begin(), avail.end(), rng);

        bool placed = false;
        for (int pid : avail) {
            for (int rid : valid_r[eid]) {
                if (sol.get_pr_enroll(pid, rid) + fe.exam_enroll[eid] <= fe.room_cap[rid]) {
                    sol.assign(eid, pid, rid);
                    placed = true; break;
                }
            }
            if (placed) break;
        }
        if (!placed) {
            int pid = avail[0];
            int rid = valid_r[eid].empty() ? 0 : valid_r[eid][rng() % valid_r[eid].size()];
            sol.assign(eid, pid, rid);
        }
    }
    return sol;
}

// VND: Move → Swap → Kempe → Kick → Room. First-improvement restarts from Move.
inline double vnd(
    Solution& sol, double fit,
    const FastEvaluator& fe, const ProblemInstance& prob,
    int ne, int np, int nr,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    AliasTable& alias,
    int budget,
    std::mt19937& rng)
{
    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    int attempts = 0;
    int level = 0; // 0=Move, 1=Swap, 2=Kempe, 3=Kick, 4=Room

    while (level < 5 && attempts < budget) {
        bool improved = false;
        attempts++;

        if (level == 0) {
            // Move: random exam to random valid (period, room)
            int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            auto& vp = valid_p[eid]; auto& vr = valid_r[eid];
            if (!vp.empty() && !vr.empty()) {
                int pid = vp[rng() % vp.size()];
                int rid = vr[rng() % vr.size()];
                double d = fe.move_delta(sol, eid, pid, rid);
                if (d < -0.5) {
                    fe.apply_move(sol, eid, pid, rid);
                    fit += d; improved = true;
                }
            }
        } else if (level == 1) {
            // Swap: exchange periods of two exams
            int ea = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            int eb = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            if (ea != eb) {
                int pa = sol.period_of[ea], ra = sol.room_of[ea];
                int pb = sol.period_of[eb], rb = sol.room_of[eb];
                if (pa >= 0 && pb >= 0 && pa != pb &&
                    fe.exam_dur[ea] <= fe.period_dur[pb] && fe.exam_dur[eb] <= fe.period_dur[pa]) {
                    double d1 = fe.move_delta(sol, ea, pb, ra);
                    fe.apply_move(sol, ea, pb, ra);
                    double d2 = fe.move_delta(sol, eb, pa, rb);
                    if (d1 + d2 < -0.5) {
                        fe.apply_move(sol, eb, pa, rb);
                        fit += d1 + d2; improved = true;
                    } else {
                        fe.apply_move(sol, ea, pa, ra); // undo
                    }
                }
            }
        } else if (level == 2) {
            // Kempe chain
            int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            int p1 = sol.period_of[eid]; if (p1 < 0) { level++; continue; }
            int p2 = std::uniform_int_distribution<int>(0, np - 1)(rng);
            if (p2 != p1) {
                auto chain = kempe_detail::build_chain(sol, prob.adj, ne, eid, p1, p2);
                if (!chain.empty() && (int)chain.size() <= ne / 4) {
                    auto old_pe = fe.partial_eval(sol, chain);
                    auto undo = kempe_detail::apply_chain(sol, chain, p1, p2);
                    auto new_pe = fe.partial_eval(sol, chain);
                    double kd = new_pe.fitness() - old_pe.fitness();
                    if (kd < -0.5) {
                        fit += kd; improved = true;
                    } else {
                        kempe_detail::undo_chain(sol, undo);
                    }
                }
            }
        } else if (level == 3) {
            // Kick: unassign exam, scan all valid slots, reinsert best
            int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            auto& vp = valid_p[eid]; auto& vr = valid_r[eid];
            if (!vp.empty() && !vr.empty()) {
                int old_p = sol.period_of[eid], old_r = sol.room_of[eid];
                double best_d = 0;
                int best_p = -1, best_r = -1;
                for (int pid : vp) {
                    for (int rid : vr) {
                        double d = fe.move_delta(sol, eid, pid, rid);
                        if (d < best_d) { best_d = d; best_p = pid; best_r = rid; }
                    }
                }
                if (best_p >= 0 && best_d < -0.5) {
                    fe.apply_move(sol, eid, best_p, best_r);
                    fit += best_d; improved = true;
                }
            }
        } else {
            // Room-only move
            if (nr <= 1) { level++; continue; }
            int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            auto& vr = valid_r[eid];
            if (!vr.empty()) {
                int pid = sol.period_of[eid]; if (pid < 0) { level++; continue; }
                int rid = vr[rng() % vr.size()];
                if (rid != sol.room_of[eid]) {
                    double d = fe.move_delta(sol, eid, pid, rid);
                    if (d < -0.5) {
                        fe.apply_move(sol, eid, pid, rid);
                        fit += d; improved = true;
                    }
                }
            }
        }

        if (improved) level = 0; // restart from Move on improvement
        else level++;
    }
    return fit;
}

} // namespace woa_detail

inline AlgoResult solve_woa(
    const ProblemInstance& prob,
    int population_size = 25,
    int max_iterations  = 3000,
    int seed            = 42,
    bool verbose        = false,
    const Solution* init_sol = nullptr)
{
    using namespace woa_detail;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    population_size = std::max(population_size, 5);
    FastEvaluator fe(prob);

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (fe.exam_dur[e] <= fe.period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (fe.exam_enroll[e] <= fe.room_cap[r]) valid_r[e].push_back(r);
    }

    // Initialize population
    std::vector<Whale> pop(population_size);
    if (init_sol) { pop[0].sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); pop[0].sol = g.sol.copy(); }
    auto ev0 = fe.full_eval(pop[0].sol);
    if (!ev0.feasible()) {
        fe.recover_feasibility(pop[0].sol, 500, seed);
        ev0 = fe.full_eval(pop[0].sol);
    }
    pop[0].fitness = ev0.fitness();
    pop[0].feasible = ev0.feasible();

    for (int i = 1; i < population_size; i++) {
        pop[i].sol = random_solution(prob, fe, ne, np, nr, valid_p, valid_r, rng);
        auto ev = fe.full_eval(pop[i].sol);
        pop[i].fitness = ev.fitness();
        pop[i].feasible = ev.feasible();
    }

    // Find leader (feasibility-first)
    int leader_idx = 0;
    for (int i = 1; i < population_size; i++) {
        bool i_better = false;
        if (pop[i].feasible && !pop[leader_idx].feasible) i_better = true;
        else if (pop[i].feasible == pop[leader_idx].feasible && pop[i].fitness < pop[leader_idx].fitness)
            i_better = true;
        if (i_better) leader_idx = i;
    }

    Solution best_sol = pop[leader_idx].sol.copy();
    double best_fitness = pop[leader_idx].fitness;
    bool best_feasible = pop[leader_idx].feasible;

    if (verbose) {
        auto ev = fe.full_eval(best_sol);
        std::cerr << "[WOA] Init: pop=" << population_size
                  << " best hard=" << ev.hard() << " soft=" << ev.soft() << std::endl;
    }

    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::uniform_int_distribution<int> de(0, ne - 1);

    // Per-whale alias tables (recomputed periodically)
    std::vector<std::vector<double>> whale_costs(population_size, std::vector<double>(ne, 1.0));
    std::vector<AliasTable> whale_alias(population_size);
    std::vector<bool> alias_dirty(population_size, true);

    auto recompute_alias = [&](int i) {
        for (int e = 0; e < ne; e++) {
            int pid = pop[i].sol.period_of[e]; if (pid < 0) { whale_costs[i][e] = 1.0; continue; }
            double c = 1.0;
            for (auto& [nb, _] : prob.adj[e]) {
                int np2 = pop[i].sol.period_of[nb]; if (np2 < 0) continue;
                if (np2 == pid) c += 100000;
                int gap = std::abs(pid - np2);
                if (gap > 0 && gap <= fe.w_spread) c += 1;
                if (fe.period_day[pid] == fe.period_day[np2]) {
                    int g = std::abs(fe.period_daypos[pid] - fe.period_daypos[np2]);
                    if (g == 1) c += fe.w_2row;
                    else if (g > 1) c += fe.w_2day;
                }
            }
            whale_costs[i][e] = c;
        }
        whale_alias[i].build(whale_costs[i]);
        alias_dirty[i] = false;
    };

    int vnd_budget = std::max(5, ne / 10);
    int iters_done = 0;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        // Decay parameter a: 2 → 0
        double a = 2.0 * (1.0 - (double)it / max_iterations);

        // Periodically recompute alias tables
        if (it % 200 == 0)
            std::fill(alias_dirty.begin(), alias_dirty.end(), true);

        for (int i = 0; i < population_size; i++) {
            if (alias_dirty[i]) recompute_alias(i);

            // Leader does VND-only (doesn't chase itself)
            if (i == leader_idx) {
                if (it % 5 == 0) {
                    pop[i].fitness = vnd(pop[i].sol, pop[i].fitness, fe, prob,
                                         ne, np, nr, valid_p, valid_r,
                                         whale_alias[i], vnd_budget, rng);
                    alias_dirty[i] = true;
                }
                continue;
            }

            double r1 = unif(rng);
            double A = 2.0 * a * r1 - a;
            double l = std::uniform_real_distribution<double>(-1.0, 1.0)(rng);
            double p = unif(rng);
            bool changed = false;

            if (p < 0.5 && std::abs(A) < 1.0) {
                // ── Phase 1: Encircle — move exams toward leader ──
                int n_move = std::max(1, (int)(std::abs(A) * ne * 0.1));
                for (int m = 0; m < n_move; m++) {
                    int eid = whale_alias[i].sample(rng);
                    int lp = pop[leader_idx].sol.period_of[eid];
                    int lr = pop[leader_idx].sol.room_of[eid];
                    if (lp < 0) continue;
                    if (lp == pop[i].sol.period_of[eid] && lr == pop[i].sol.room_of[eid]) continue;

                    double d = fe.move_delta(pop[i].sol, eid, lp, lr);
                    bool accept = (d < 0) || (unif(rng) < std::exp(-d / (std::abs(A) * 1000.0 + 1.0)));
                    if (accept) {
                        fe.apply_move(pop[i].sol, eid, lp, lr);
                        pop[i].fitness += d;
                        changed = true;
                    }
                }

            } else if (p >= 0.5) {
                // ── Phase 2: Bubble-net spiral — Kempe chains ──
                int n_chains = 1 + (int)(unif(rng) * 2);
                for (int c = 0; c < n_chains; c++) {
                    int eid = whale_alias[i].sample(rng);
                    int p1 = pop[i].sol.period_of[eid]; if (p1 < 0) continue;
                    int p2;
                    if (unif(rng) < 0.5 && pop[leader_idx].sol.period_of[eid] >= 0)
                        p2 = pop[leader_idx].sol.period_of[eid];
                    else
                        p2 = std::uniform_int_distribution<int>(0, np - 1)(rng);
                    if (p2 == p1) continue;

                    auto chain = kempe_detail::build_chain(pop[i].sol, prob.adj, ne, eid, p1, p2);
                    if (chain.empty() || (int)chain.size() > ne / 4) continue;

                    auto old_pe = fe.partial_eval(pop[i].sol, chain);
                    auto undo = kempe_detail::apply_chain(pop[i].sol, chain, p1, p2);
                    auto new_pe = fe.partial_eval(pop[i].sol, chain);
                    double kd = new_pe.fitness() - old_pe.fitness();

                    bool accept = (kd < 0) || (unif(rng) < 0.3 * std::exp(l));
                    if (accept) {
                        pop[i].fitness += kd;
                        changed = true;
                    } else {
                        kempe_detail::undo_chain(pop[i].sol, undo);
                    }
                }

            } else {
                // ── Phase 3: Random search — exploration ──
                int n_perturb = std::max(3, ne / 15);
                int rand_whale = rng() % population_size;
                for (int m = 0; m < n_perturb; m++) {
                    int eid = de(rng);
                    auto& vp = valid_p[eid]; auto& vr = valid_r[eid];
                    if (vp.empty() || vr.empty()) continue;

                    int new_pid, new_rid;
                    if (unif(rng) < 0.5 && pop[rand_whale].sol.period_of[eid] >= 0) {
                        new_pid = pop[rand_whale].sol.period_of[eid];
                        new_rid = vr[rng() % vr.size()];
                    } else {
                        new_pid = vp[rng() % vp.size()];
                        new_rid = vr[rng() % vr.size()];
                    }
                    double d = fe.move_delta(pop[i].sol, eid, new_pid, new_rid);
                    fe.apply_move(pop[i].sol, eid, new_pid, new_rid);
                    pop[i].fitness += d;
                }
                changed = true;
            }

            // VND on changed whales, every 10 iters to control runtime
            if (changed && it % 10 == 0) {
                alias_dirty[i] = true;
                recompute_alias(i);
                pop[i].fitness = vnd(pop[i].sol, pop[i].fitness, fe, prob,
                                      ne, np, nr, valid_p, valid_r,
                                      whale_alias[i], vnd_budget, rng);
            }
            if (changed) alias_dirty[i] = true;
        }

        // Periodic resync
        if ((it + 1) % 100 == 0) {
            for (int i = 0; i < population_size; i++) {
                auto ev = fe.full_eval(pop[i].sol);
                pop[i].fitness = ev.fitness();
                pop[i].feasible = ev.feasible();
            }
        }

        // Update leader and global best
        leader_idx = 0;
        for (int i = 1; i < population_size; i++) {
            bool i_better = false;
            if (pop[i].feasible && !pop[leader_idx].feasible) i_better = true;
            else if (pop[i].feasible == pop[leader_idx].feasible && pop[i].fitness < pop[leader_idx].fitness)
                i_better = true;
            if (i_better) leader_idx = i;
        }

        // Check improvement vs global best
        auto check = fe.full_eval(pop[leader_idx].sol);
        pop[leader_idx].fitness = check.fitness();
        pop[leader_idx].feasible = check.feasible();
        bool cf = check.feasible();
        bool dominated = (best_feasible && !cf);
        if (!dominated && check.fitness() < best_fitness) {
            best_sol = pop[leader_idx].sol.copy();
            best_fitness = check.fitness();
            best_feasible = cf;
            if (verbose && (it < 10 || it % 500 == 0))
                std::cerr << "[WOA] Iter " << it << ": best hard=" << check.hard()
                          << " soft=" << check.soft() << std::endl;
        }
    }

    fe.optimize_rooms(best_sol);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[WOA] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "WOA"};
}
