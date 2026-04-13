/*
 * hho.h — Harris Hawks Optimization with incremental delta evaluation
 *
 * Population-based metaheuristic inspired by cooperative hunting behavior.
 * Phases: exploration (|E| >= 1) and exploitation (|E| < 1) with Lévy flights.
 *
 * Key optimization: move_delta() for O(k) perturbation scoring.
 */

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "models.h"
#include "evaluator.h"
#include "greedy.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace hho_detail {

// ── Random-greedy initialization ────────────────────────────

inline std::pair<Solution, double> init_random_solution(
    const ProblemInstance& prob, const FastEvaluator& fe, std::mt19937& rng)
{
    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    Solution sol;
    sol.init(prob);

    std::vector<int> order(ne);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    for (int eid : order) {
        std::vector<bool> blocked(np, false);
        for (auto& [nb, _] : prob.adj[eid]) {
            int p = sol.period_of[nb];
            if (p >= 0) blocked[p] = true;
        }

        int dur = prob.exams[eid].duration;
        std::vector<int> avail;
        for (int p = 0; p < np; p++)
            if (!blocked[p] && prob.periods[p].duration >= dur)
                avail.push_back(p);
        if (avail.empty())
            for (int p = 0; p < np; p++) avail.push_back(p);

        std::shuffle(avail.begin(), avail.end(), rng);

        bool placed = false;
        for (int pid : avail) {
            for (int rid = 0; rid < nr; rid++) {
                if (sol.get_pr_enroll(pid, rid) + fe.exam_enroll[eid]
                    <= fe.room_cap[rid]) {
                    sol.assign(eid, pid, rid);
                    placed = true;
                    break;
                }
            }
            if (placed) break;
        }
        if (!placed) {
            std::uniform_int_distribution<int> dp(0, np - 1), dr(0, nr - 1);
            sol.assign(eid, dp(rng), dr(rng));
        }
    }

    return {std::move(sol), fe.full_eval(sol).fitness()};
}

// ── Blind perturbation (exploration) ────────────────────────

inline std::pair<Solution, double> perturb_with_delta(
    const ProblemInstance& prob, const FastEvaluator& fe,
    const Solution& sol, double cur_fit, double intensity, std::mt19937& rng)
{
    int ne = prob.n_e(), np = prob.n_p();
    Solution ns = sol.copy();
    double fit = cur_fit;

    int n_moves = std::max(1, (int)(ne * intensity));
    std::vector<int> exams(ne);
    std::iota(exams.begin(), exams.end(), 0);
    std::shuffle(exams.begin(), exams.end(), rng);
    if (n_moves < ne) exams.resize(n_moves);

    for (int eid : exams) {
        int dur = fe.exam_dur[eid];
        std::vector<int> valid;
        for (int p = 0; p < np; p++)
            if (fe.period_dur[p] >= dur) valid.push_back(p);
        if (valid.empty()) continue;

        std::uniform_int_distribution<int> dv(0, (int)valid.size() - 1);
        int new_pid = valid[dv(rng)];
        int old_rid = ns.room_of[eid];
        if (old_rid < 0) old_rid = 0;

        double delta = fe.move_delta(ns, eid, new_pid, old_rid);
        fe.apply_move(ns, eid, new_pid, old_rid);
        fit += delta;
    }
    return {std::move(ns), fit};
}

// ── Smart perturbation (only apply improving moves) ─────────

inline std::pair<Solution, double> smart_perturb(
    const ProblemInstance& prob, const FastEvaluator& fe,
    const Solution& sol, double cur_fit, int n_moves, std::mt19937& rng)
{
    int ne = prob.n_e(), np = prob.n_p();
    Solution ns = sol.copy();
    double fit = cur_fit;

    int n_try = std::min(ne, n_moves * 3);
    std::vector<int> exams(ne);
    std::iota(exams.begin(), exams.end(), 0);
    std::shuffle(exams.begin(), exams.end(), rng);
    if (n_try < ne) exams.resize(n_try);

    int applied = 0;
    for (int eid : exams) {
        if (applied >= n_moves) break;
        int dur = fe.exam_dur[eid];
        std::vector<int> valid;
        for (int p = 0; p < np; p++)
            if (fe.period_dur[p] >= dur) valid.push_back(p);
        if (valid.empty()) continue;

        std::uniform_int_distribution<int> dv(0, (int)valid.size() - 1);
        int new_pid = valid[dv(rng)];
        int old_rid = ns.room_of[eid];
        if (old_rid < 0) old_rid = 0;

        double delta = fe.move_delta(ns, eid, new_pid, old_rid);
        if (delta < 0) {
            fe.apply_move(ns, eid, new_pid, old_rid);
            fit += delta;
            applied++;
        }
    }
    return {std::move(ns), fit};
}

} // namespace hho_detail

// ── Public interface ────────────────────────────────────────

inline AlgoResult solve_hho(
    const ProblemInstance& prob,
    int pop_size   = 30,
    int max_iters  = 100,
    int seed       = 42,
    bool verbose   = false,
    const Solution* init_sol = nullptr)
{
    using namespace hho_detail;
    auto t0 = std::chrono::high_resolution_clock::now();

    FastEvaluator fe(prob);
    int ne = prob.n_e();
    std::mt19937 rng(seed);

    // ── Initialize population ──
    std::vector<Solution> population;
    std::vector<double> fitness;

    if (init_sol) { population.push_back(init_sol->copy()); }
    else { auto g = solve_greedy(prob, false); population.push_back(g.sol.copy()); }
    fitness.push_back(fe.full_eval(population[0]).fitness());

    for (int i = 1; i < pop_size; i++) {
        auto [s, f] = init_random_solution(prob, fe, rng);
        population.push_back(std::move(s));
        fitness.push_back(f);
    }

    // Find rabbit (best solution)
    int best_idx = (int)(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
    Solution rabbit = population[best_idx].copy();
    double rabbit_fitness = fitness[best_idx];

    if (verbose)
        std::cerr << "[HHO] Pop=" << pop_size << " Iters=" << max_iters
                  << " Init best=" << rabbit_fitness << std::endl;

    // Lévy flight parameter
    double beta = 1.5;
    double sigma_u = std::pow(
        std::tgamma(1 + beta) * std::sin(M_PI * beta / 2) /
        (std::tgamma((1 + beta) / 2) * beta * std::pow(2.0, (beta - 1) / 2)),
        1.0 / beta);

    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double> norm(0.0, 1.0);

    int iters_done = 0;
    for (int t = 0; t < max_iters; t++) {
        iters_done = t + 1;

        double E0 = 2.0 * unif(rng) - 1.0;
        double E  = 2.0 * E0 * (1.0 - (double)t / max_iters);

        for (int i = 0; i < pop_size; i++) {
            Solution new_sol;
            double new_fit;

            if (std::abs(E) >= 1.0) {
                // ── EXPLORATION ──
                if (unif(rng) < 0.5) {
                    std::uniform_int_distribution<int> di(0, pop_size - 1);
                    int ri = di(rng);
                    std::tie(new_sol, new_fit) = perturb_with_delta(
                        prob, fe, population[ri], fitness[ri], 0.3, rng);
                } else {
                    std::tie(new_sol, new_fit) = perturb_with_delta(
                        prob, fe, rabbit, rabbit_fitness, 0.3, rng);
                }
            } else {
                // ── EXPLOITATION ──
                double r = unif(rng);
                double absE = std::abs(E);
                if (absE >= 0.5) {
                    if (r >= 0.5) {
                        int nm = std::max(1, (int)(ne * absE * 0.15));
                        std::tie(new_sol, new_fit) = smart_perturb(
                            prob, fe, rabbit, rabbit_fitness, nm, rng);
                    } else {
                        std::tie(new_sol, new_fit) = perturb_with_delta(
                            prob, fe, rabbit, rabbit_fitness, absE * 0.12, rng);
                        double levy = std::abs(norm(rng) * sigma_u /
                                               std::pow(std::abs(norm(rng)), 1.0 / beta));
                        int extra = std::max(1, (int)(levy * 2));
                        std::tie(new_sol, new_fit) = smart_perturb(
                            prob, fe, new_sol, new_fit, extra, rng);
                    }
                } else {
                    if (r >= 0.5) {
                        int nm = std::max(1, (int)(ne * absE * 0.08));
                        std::tie(new_sol, new_fit) = smart_perturb(
                            prob, fe, rabbit, rabbit_fitness, nm, rng);
                    } else {
                        double levy = std::abs(norm(rng) * sigma_u /
                                               std::pow(std::abs(norm(rng)), 1.0 / beta));
                        double intensity = std::min(0.15, levy * 0.02);
                        std::tie(new_sol, new_fit) = perturb_with_delta(
                            prob, fe, rabbit, rabbit_fitness, intensity, rng);
                        std::tie(new_sol, new_fit) = smart_perturb(
                            prob, fe, new_sol, new_fit,
                            std::max(1, (int)(ne * 0.05)), rng);
                    }
                }
            }

            // Accept if delta-tracked fitness improved; full_eval only on
            // new global best (rare) to avoid 30 full_evals per iteration.
            if (new_fit < fitness[i]) {
                population[i] = std::move(new_sol);
                fitness[i] = new_fit;
            }
        }

        // Staggered resync: 1 member per iteration to correct delta drift
        if (t % 3 == 0) {
            int slot = (t / 3) % pop_size;
            fitness[slot] = fe.full_eval(population[slot]).fitness();
        }

        int cur_best = (int)(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
        if (fitness[cur_best] < rabbit_fitness) {
            rabbit = population[cur_best].copy();
            rabbit_fitness = fitness[cur_best];
            if (verbose && t % 10 == 0)
                std::cerr << "[HHO] Iter " << t << ": NEW BEST " << rabbit_fitness << std::endl;
        } else if (verbose && t % 25 == 0) {
            std::cerr << "[HHO] Iter " << t << ": best=" << rabbit_fitness << std::endl;
        }
    }

    fe.optimize_rooms(rabbit);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(rabbit);

    if (verbose)
        std::cerr << "[HHO] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(rabbit), final_ev, rt, iters_done, "HHO"};
}