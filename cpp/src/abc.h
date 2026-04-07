/*
 * abc.h — Artificial Bee Colony
 *
 * Bio-inspired swarm optimization with three phases per iteration:
 *   1. Employed bees:  multi-move local search on each food source (k=3 moves)
 *   2. Onlooker bees:  fitness-proportional selection + crossover-inspired transfer
 *   3. Scout bees:     copy-and-perturb best source instead of random restart
 *
 * Uses move_delta for O(k) neighbor evaluation with periodic full_eval resync.
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

namespace abc_detail {

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

// Compute per-exam cost for targeted selection (higher cost = more worth moving)
inline void compute_exam_costs(
    const Solution& sol, const FastEvaluator& fe,
    const ProblemInstance& prob, int ne,
    std::vector<double>& costs)
{
    costs.assign(ne, 1.0);
    for (int e = 0; e < ne; e++) {
        int pid = sol.period_of[e]; if (pid < 0) continue;
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
        c += fe.period_pen[pid] + fe.room_pen[sol.room_of[e]];
        costs[e] = c;
    }
}

// Weighted random pick based on costs (higher cost = higher probability)
inline int cost_weighted_pick(
    const std::vector<double>& costs, int ne, std::mt19937& rng)
{
    double total = 0;
    for (int e = 0; e < ne; e++) total += costs[e];
    if (total <= 0) return rng() % ne;

    std::uniform_real_distribution<double> d(0.0, total);
    double r = d(rng);
    double acc = 0;
    for (int e = 0; e < ne; e++) {
        acc += costs[e];
        if (r <= acc) return e;
    }
    return ne - 1;
}

} // namespace abc_detail

inline AlgoResult solve_abc(
    const ProblemInstance& prob,
    int colony_size    = 30,
    int max_iterations = 3000,
    int abandon_limit  = 0,
    int seed           = 42,
    bool verbose       = false,
    const Solution* init_sol = nullptr)
{
    using namespace abc_detail;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    colony_size = std::max(colony_size, 5);
    if (abandon_limit <= 0) abandon_limit = colony_size * 3;
    FastEvaluator fe(prob);

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (fe.exam_dur[e] <= fe.period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (fe.exam_enroll[e] <= fe.room_cap[r]) valid_r[e].push_back(r);
    }

    // Initialize food sources: greedy + random
    std::vector<Solution> sources;
    std::vector<double> fitness;
    std::vector<int> trials;

    if (init_sol) { sources.push_back(init_sol->copy()); }
    else { auto g = solve_greedy(prob, false); sources.push_back(g.sol.copy()); }
    EvalResult ev0 = fe.full_eval(sources[0]);
    fitness.push_back(ev0.fitness());
    trials.push_back(0);

    // Feasibility recovery on greedy if needed
    if (!ev0.feasible()) {
        if (verbose)
            std::cerr << "[ABC] Greedy infeasible (hard=" << ev0.hard()
                      << "), running recovery..." << std::endl;
        fe.recover_feasibility(sources[0], 500, seed);
        ev0 = fe.full_eval(sources[0]);
        fitness[0] = ev0.fitness();
    }

    for (int i = 1; i < colony_size; i++) {
        auto sol = random_solution(prob, fe, ne, np, nr, valid_p, valid_r, rng);
        fitness.push_back(fe.full_eval(sol).fitness());
        sources.push_back(std::move(sol));
        trials.push_back(0);
    }

    // Global best (feasibility-first)
    int bi = (int)(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
    Solution best_sol = sources[bi].copy();
    double best_fitness = fitness[bi];
    bool best_feasible = fe.full_eval(best_sol).feasible();

    if (verbose) {
        auto ev = fe.full_eval(best_sol);
        std::cerr << "[ABC] Init: colony=" << colony_size
                  << " best hard=" << ev.hard() << " soft=" << ev.soft() << std::endl;
    }

    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    int moves_per_bee = 3;  // multi-move: each bee tries k moves
    int iters_done = 0;

    // Per-source exam costs (recomputed periodically)
    std::vector<std::vector<double>> source_costs(colony_size, std::vector<double>(ne, 1.0));

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        // Recompute exam costs periodically for targeted selection
        if (it % 100 == 0) {
            for (int i = 0; i < colony_size; i++)
                compute_exam_costs(sources[i], fe, prob, ne, source_costs[i]);
        }

        // ── Employed Bee Phase: multi-move local search ───────
        for (int i = 0; i < colony_size; i++) {
            bool improved = false;

            for (int m = 0; m < moves_per_bee; m++) {
                // Pick exam weighted by cost (target expensive exams)
                int eid = cost_weighted_pick(source_costs[i], ne, rng);
                auto& vp = valid_p[eid]; auto& vr = valid_r[eid];
                if (vp.empty() || vr.empty()) continue;

                // Knowledge transfer: take period from random other source
                int k = (int)(rng() % colony_size);
                if (k == i) k = (k + 1) % colony_size;

                int new_pid, new_rid;
                if (unif(rng) < 0.5 && sources[k].period_of[eid] >= 0) {
                    new_pid = sources[k].period_of[eid];
                    new_rid = vr[rng() % vr.size()];
                } else {
                    new_pid = vp[rng() % vp.size()];
                    new_rid = vr[rng() % vr.size()];
                }

                double delta = fe.move_delta(sources[i], eid, new_pid, new_rid);
                if (delta < 0) {
                    fe.apply_move(sources[i], eid, new_pid, new_rid);
                    fitness[i] += delta;
                    improved = true;
                }
            }

            if (improved) trials[i] = 0;
            else trials[i]++;
        }

        // ── Onlooker Bee Phase: fitness-proportional + multi-move ──
        double max_fit = *std::max_element(fitness.begin(), fitness.end());
        std::vector<double> probs(colony_size);
        double sum_prob = 0;
        for (int i = 0; i < colony_size; i++) {
            probs[i] = max_fit - fitness[i] + 1.0;
            sum_prob += probs[i];
        }

        for (int o = 0; o < colony_size; o++) {
            double r = unif(rng) * sum_prob;
            int sel = 0;
            double acc = 0;
            for (int i = 0; i < colony_size; i++) {
                acc += probs[i];
                if (r <= acc) { sel = i; break; }
            }

            bool improved = false;
            for (int m = 0; m < moves_per_bee; m++) {
                int eid = cost_weighted_pick(source_costs[sel], ne, rng);
                auto& vp = valid_p[eid]; auto& vr = valid_r[eid];
                if (vp.empty() || vr.empty()) continue;

                int k = (int)(rng() % colony_size);
                if (k == sel) k = (k + 1) % colony_size;

                int new_pid, new_rid;
                if (unif(rng) < 0.5 && sources[k].period_of[eid] >= 0) {
                    new_pid = sources[k].period_of[eid];
                    new_rid = vr[rng() % vr.size()];
                } else {
                    new_pid = vp[rng() % vp.size()];
                    new_rid = vr[rng() % vr.size()];
                }

                double delta = fe.move_delta(sources[sel], eid, new_pid, new_rid);
                if (delta < 0) {
                    fe.apply_move(sources[sel], eid, new_pid, new_rid);
                    fitness[sel] += delta;
                    improved = true;
                }
            }

            if (improved) trials[sel] = 0;
            else trials[sel]++;
        }

        // ── Scout Bee Phase: copy-and-perturb best instead of random restart ──
        for (int i = 0; i < colony_size; i++) {
            if (trials[i] > abandon_limit) {
                // Find current best source
                int cur_best = (int)(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
                // Copy best and apply random perturbation
                sources[i] = sources[cur_best].copy();
                fitness[i] = fitness[cur_best];

                int n_perturb = std::max(3, ne / 10);
                for (int p = 0; p < n_perturb; p++) {
                    int eid = de(rng);
                    auto& vp = valid_p[eid]; auto& vr = valid_r[eid];
                    if (vp.empty() || vr.empty()) continue;
                    int pid = vp[rng() % vp.size()];
                    int rid = vr[rng() % vr.size()];
                    double delta = fe.move_delta(sources[i], eid, pid, rid);
                    fe.apply_move(sources[i], eid, pid, rid);
                    fitness[i] += delta;
                }
                trials[i] = 0;
            }
        }

        // Periodic resync
        if ((it + 1) % 50 == 0)
            for (int i = 0; i < colony_size; i++)
                fitness[i] = fe.full_eval(sources[i]).fitness();

        // Update global best (feasibility-first)
        for (int i = 0; i < colony_size; i++) {
            if (fitness[i] < best_fitness - 0.5) {
                auto check = fe.full_eval(sources[i]);
                bool cf = check.feasible();
                bool dominated = (best_feasible && !cf);
                if (!dominated && check.fitness() < best_fitness) {
                    best_sol = sources[i].copy();
                    best_fitness = check.fitness();
                    best_feasible = cf;
                    if (verbose && (it < 10 || it % 500 == 0))
                        std::cerr << "[ABC] Iter " << it << ": best hard=" << check.hard()
                                  << " soft=" << check.soft() << std::endl;
                }
            }
        }
    }

    fe.optimize_rooms(best_sol);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[ABC] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "ABC"};
}
