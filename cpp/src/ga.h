/*
 * ga.h — Memetic Genetic Algorithm
 *
 * Delta-based evaluation throughout (no per-offspring full_eval):
 *   - Period-swap crossover: transfer random period groups from parent B
 *   - Local-search mutation: first-improvement moves via move_delta
 *   - Periodic full_eval resync every 25 generations
 *
 * Feasibility-first best tracking across all generations.
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

namespace ga_detail {

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

inline int tournament_select(
    const std::vector<double>& fitness, int pop_size, int k, std::mt19937& rng)
{
    std::uniform_int_distribution<int> di(0, pop_size - 1);
    int best = di(rng);
    for (int i = 1; i < k; i++) {
        int cand = di(rng);
        if (fitness[cand] < fitness[best]) best = cand;
    }
    return best;
}

// Delta-based crossover: start from parent A, selectively adopt parent B's assignments.
// Uses move_delta for incremental cost tracking — no full_eval needed.
inline double crossover_delta(
    Solution& child, double child_fit,
    const Solution& parent_b,
    const FastEvaluator& fe,
    int ne, int np,
    double swap_fraction,
    std::mt19937& rng)
{
    // Select random exams to transfer from parent B
    int n_swap = std::max(1, (int)(ne * swap_fraction));
    std::vector<int> exams(ne);
    std::iota(exams.begin(), exams.end(), 0);
    std::shuffle(exams.begin(), exams.end(), rng);
    if (n_swap < ne) exams.resize(n_swap);

    for (int eid : exams) {
        int b_pid = parent_b.period_of[eid];
        int b_rid = parent_b.room_of[eid];
        if (b_pid < 0) continue;
        if (b_pid == child.period_of[eid] && b_rid == child.room_of[eid]) continue;

        double delta = fe.move_delta(child, eid, b_pid, b_rid);
        fe.apply_move(child, eid, b_pid, b_rid);
        child_fit += delta;
    }
    return child_fit;
}

// Local-search mutation: first-improvement random moves via move_delta.
inline double local_search_mutation(
    Solution& sol, double fit,
    const FastEvaluator& fe,
    int ne,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    int n_steps,
    std::mt19937& rng)
{
    std::uniform_int_distribution<int> de(0, ne - 1);

    for (int step = 0; step < n_steps; step++) {
        int eid = de(rng);
        auto& vp = valid_p[eid];
        auto& vr = valid_r[eid];
        if (vp.empty() || vr.empty()) continue;

        // Try a few random (period, room) pairs, accept first improvement
        int tries = std::min(5, (int)(vp.size() * vr.size()));
        for (int t = 0; t < tries; t++) {
            int pid = vp[rng() % vp.size()];
            int rid = vr[rng() % vr.size()];
            if (pid == sol.period_of[eid] && rid == sol.room_of[eid]) continue;

            double delta = fe.move_delta(sol, eid, pid, rid);
            if (delta < -0.5) {
                fe.apply_move(sol, eid, pid, rid);
                fit += delta;
                break;
            }
        }
    }
    return fit;
}

} // namespace ga_detail

inline AlgoResult solve_ga(
    const ProblemInstance& prob,
    int pop_size          = 50,
    int max_generations   = 500,
    double crossover_rate = 0.8,
    double mutation_rate  = 0.15,
    int seed              = 42,
    bool verbose          = false,
    const Solution* init_sol = nullptr)
{
    using namespace ga_detail;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    pop_size = std::max(pop_size, 10);
    FastEvaluator fe(prob);

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (fe.exam_dur[e] <= fe.period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (fe.exam_enroll[e] <= fe.room_cap[r]) valid_r[e].push_back(r);
    }

    // Initialize: greedy + random
    std::vector<Solution> population;
    std::vector<double> fitness;

    Solution greedy_sol;
    if (init_sol) { greedy_sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); greedy_sol = g.sol.copy(); }
    EvalResult ev0 = fe.full_eval(greedy_sol);
    if (!ev0.feasible()) {
        if (verbose)
            std::cerr << "[GA] Greedy infeasible (hard=" << ev0.hard()
                      << "), running recovery..." << std::endl;
        fe.recover_feasibility(greedy_sol, 500, seed);
        ev0 = fe.full_eval(greedy_sol);
    }
    fitness.push_back(ev0.fitness());
    population.push_back(std::move(greedy_sol));

    for (int i = 1; i < pop_size; i++) {
        auto sol = random_solution(prob, fe, ne, np, nr, valid_p, valid_r, rng);
        fitness.push_back(fe.full_eval(sol).fitness());
        population.push_back(std::move(sol));
    }

    // Global best
    int bi = (int)(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
    Solution best_sol = population[bi].copy();
    double best_fitness = fitness[bi];
    bool best_feasible = fe.full_eval(best_sol).feasible();

    if (verbose) {
        auto ev = fe.full_eval(best_sol);
        std::cerr << "[GA] Init: pop=" << pop_size
                  << " best hard=" << ev.hard() << " soft=" << ev.soft() << std::endl;
    }

    std::uniform_real_distribution<double> unif(0.0, 1.0);
    int elite_count = std::max(2, pop_size / 10);
    int ls_steps = std::max(5, ne / 20);  // local search steps per offspring
    int iters_done = 0;

    for (int gen = 0; gen < max_generations; gen++) {
        iters_done = gen + 1;

        // Argsort by fitness
        std::vector<int> rank(pop_size);
        std::iota(rank.begin(), rank.end(), 0);
        std::sort(rank.begin(), rank.end(),
                  [&](int a, int b) { return fitness[a] < fitness[b]; });

        std::vector<Solution> new_pop;
        std::vector<double> new_fit;
        new_pop.reserve(pop_size);
        new_fit.reserve(pop_size);

        // Elitism
        for (int i = 0; i < elite_count; i++) {
            new_pop.push_back(population[rank[i]].copy());
            new_fit.push_back(fitness[rank[i]]);
        }

        // Generate offspring using delta-based evaluation
        while ((int)new_pop.size() < pop_size) {
            int p1 = tournament_select(fitness, pop_size, 3, rng);
            int p2 = tournament_select(fitness, pop_size, 3, rng);

            // Start from parent with better fitness
            int donor = (fitness[p1] <= fitness[p2]) ? p1 : p2;
            int other = (donor == p1) ? p2 : p1;

            Solution child = population[donor].copy();
            double child_fit = fitness[donor];

            // Crossover: transfer assignments from other parent via move_delta
            if (unif(rng) < crossover_rate) {
                double swap_frac = 0.2 + unif(rng) * 0.3; // 20-50% of exams
                child_fit = crossover_delta(
                    child, child_fit, population[other],
                    fe, ne, np, swap_frac, rng);
            }

            // Mutation: local search — always apply (this is what makes it memetic)
            child_fit = local_search_mutation(
                child, child_fit, fe, ne, valid_p, valid_r, ls_steps, rng);

            new_fit.push_back(child_fit);
            new_pop.push_back(std::move(child));
        }

        population = std::move(new_pop);
        fitness = std::move(new_fit);

        // Periodic resync with full_eval to correct delta drift
        if ((gen + 1) % 25 == 0) {
            for (int i = 0; i < pop_size; i++)
                fitness[i] = fe.full_eval(population[i]).fitness();
        }

        // Update global best
        int gen_best = (int)(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
        if (fitness[gen_best] < best_fitness - 0.5) {
            auto check = fe.full_eval(population[gen_best]);
            bool cf = check.feasible();
            bool dominated = (best_feasible && !cf);
            if (!dominated && check.fitness() < best_fitness) {
                best_sol = population[gen_best].copy();
                best_fitness = check.fitness();
                best_feasible = cf;
                if (verbose && (gen < 10 || gen % 100 == 0))
                    std::cerr << "[GA] Gen " << gen << ": best hard=" << check.hard()
                              << " soft=" << check.soft() << std::endl;
            }
        }
    }

    fe.optimize_rooms(best_sol);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[GA] " << iters_done << " gens, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "Genetic Algorithm"};
}
