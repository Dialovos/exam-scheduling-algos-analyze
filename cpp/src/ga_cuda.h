/*
 * GA with batched full_eval via CudaEvaluator.
 *
 * Fully integrated pattern — demonstrates population-GPU batching:
 *   • Initial population (pop_size full-evals) routes through
 *     CudaEvaluator::score_full_batch. On CPU, this is equivalent to
 *     per-solution fe.full_eval (bit-exact); on GPU, it becomes one
 *     kernel launch per generation (full_eval_kernel is the scheduled
 *     follow-up — see PERF_ROADMAP.md).
 *   • The rest of GA (crossover/mutation/selection) is identical to
 *     solve_ga, using delta-based fitness tracking throughout.
 *
 * Same seed → same result as solve_ga on CPU-fallback builds.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_cached.h"
#include "ga.h"  // reuse ga_detail::tournament_select, crossovers, etc.
#include "cuda/cuda_evaluator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

inline AlgoResult solve_ga_cuda(
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
    CachedEvaluator Ecach(fe);
    // Ecach.initialize needs a Solution — we'll init after the greedy+pop build.

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (fe.exam_dur[e] <= fe.period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (fe.exam_enroll[e] <= fe.room_cap[r]) valid_r[e].push_back(r);
    }

    std::vector<Solution> population;
    std::vector<double> fitness;
    population.reserve(pop_size);
    fitness.resize(pop_size);

    Solution greedy_sol;
    if (init_sol) { greedy_sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); greedy_sol = g.sol.copy(); }
    EvalResult ev0 = fe.full_eval(greedy_sol);
    if (!ev0.feasible()) {
        if (verbose)
            std::cerr << "[GACuda] Greedy infeasible (hard=" << ev0.hard()
                      << "), running recovery..." << std::endl;
        fe.recover_feasibility(greedy_sol, 500, seed);
    }
    population.push_back(std::move(greedy_sol));
    for (int i = 1; i < pop_size; i++) {
        population.push_back(random_solution(prob, fe, ne, np, nr, valid_p, valid_r, rng));
    }

    // Lazy-init Ecach with the first member (only used for cached-path fallback
    // parity; not strictly needed for full_eval batch, but CudaEvaluator ctor
    // wants a CachedEvaluator that's been initialized).
    Ecach.initialize(population[0]);
    CudaEvaluator Cuev(Ecach);

    // BATCH POINT: evaluate all pop_size individuals in one call.
    // On CPU fallback this loops fe.full_eval — bit-exact with parent.
    // On GPU (when full_eval kernel ships) this becomes 1 kernel launch.
    Cuev.score_full_batch(population, fitness);

    int bi = (int)(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
    Solution best_sol = population[bi].copy();
    double best_fitness = fitness[bi];
    bool best_feasible = fe.full_eval(best_sol).feasible();

    if (verbose) {
        auto ev = fe.full_eval(best_sol);
        std::cerr << "[GACuda] Init: pop=" << pop_size
                  << " best hard=" << ev.hard() << " soft=" << ev.soft()
                  << " gpu=" << (Cuev.gpu_active ? "on" : "off") << std::endl;
    }

    std::uniform_real_distribution<double> unif(0.0, 1.0);
    int elite_count = std::max(2, pop_size / 10);
    int ls_steps = std::max(10, ne / 5);
    int iters_done = 0;

    std::vector<Solution> buf_pop;
    std::vector<double> buf_fit;
    buf_pop.reserve(pop_size);
    buf_fit.reserve(pop_size);
    int no_improve_gens = 0;

    for (int gen = 0; gen < max_generations; gen++) {
        iters_done = gen + 1;

        std::vector<int> rank(pop_size);
        std::iota(rank.begin(), rank.end(), 0);
        std::sort(rank.begin(), rank.end(),
                  [&](int a, int b) { return fitness[a] < fitness[b]; });

        buf_pop.clear();
        buf_fit.clear();

        for (int i = 0; i < elite_count; i++) {
            buf_pop.push_back(population[rank[i]].copy());
            buf_fit.push_back(fitness[rank[i]]);
        }

        while ((int)buf_pop.size() < pop_size) {
            int p1 = tournament_select(fitness, pop_size, 3, rng);
            int p2 = tournament_select(fitness, pop_size, 3, rng);

            int donor = (fitness[p1] <= fitness[p2]) ? p1 : p2;
            int other = (donor == p1) ? p2 : p1;

            Solution child = population[donor].copy();
            double child_fit = fitness[donor];

            if (unif(rng) < crossover_rate) {
                double swap_frac = 0.2 + unif(rng) * 0.3;
                if (unif(rng) < 0.7)
                    child_fit = crossover_satdegree(
                        child, child_fit, population[other],
                        fe, ne, np, swap_frac, rng);
                else
                    child_fit = crossover_random(
                        child, child_fit, population[other],
                        fe, ne, np, swap_frac, rng);
            }

            child_fit = local_search_mutation(
                child, child_fit, fe, ne, valid_p, valid_r, ls_steps, rng);

            if (unif(rng) < 0.3) {
                int kseed = rng() % ne;
                int kp1 = child.period_of[kseed];
                if (kp1 >= 0) {
                    int kp2 = rng() % np;
                    if (kp2 != kp1) {
                        auto chain = kempe_detail::build_chain(child, prob.adj, ne, kseed, kp1, kp2);
                        if (!chain.empty() && (int)chain.size() <= ne / 4) {
                            auto old_pe = fe.partial_eval(child, chain);
                            kempe_detail::apply_chain(child, chain, kp1, kp2);
                            auto new_pe = fe.partial_eval(child, chain);
                            child_fit += new_pe.fitness() - old_pe.fitness();
                        }
                    }
                }
            }

            bool too_similar = false;
            for (int i = 0; i < elite_count && !too_similar; i++) {
                int hamming = 0;
                for (int e = 0; e < ne; e++)
                    if (child.period_of[e] != buf_pop[i].period_of[e]) hamming++;
                if (hamming < ne / 10) too_similar = true;
            }
            if (too_similar) {
                buf_pop.push_back(population[donor].copy());
                buf_fit.push_back(fitness[donor]);
            } else {
                buf_fit.push_back(child_fit);
                buf_pop.push_back(std::move(child));
            }
        }

        std::swap(population, buf_pop);
        std::swap(fitness, buf_fit);

        if ((gen + 1) % 10 == 0) {
            int slot = ((gen + 1) / 5) % pop_size;
            fitness[slot] = fe.full_eval(population[slot]).fitness();
        }

        int gen_best = (int)(std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
        if (fitness[gen_best] < best_fitness - 0.5) {
            auto check = fe.full_eval(population[gen_best]);
            double af = check.fitness();
            bool af_feasible = check.feasible();
            bool dominated = (best_feasible && !af_feasible);
            if (!dominated && af < best_fitness) {
                best_sol = population[gen_best].copy();
                best_fitness = af;
                best_feasible = af_feasible;
                fitness[gen_best] = af;
                no_improve_gens = 0;
                if (verbose && (gen < 10 || gen % 100 == 0))
                    std::cerr << "[GACuda] Gen " << gen << ": best hard=" << check.hard()
                              << " soft=" << check.soft() << std::endl;
            } else {
                fitness[gen_best] = af;
                no_improve_gens++;
            }
        } else {
            no_improve_gens++;
        }

        if (no_improve_gens > 0 && no_improve_gens % 30 == 0) {
            int replaced = pop_size / 5;
            for (int i = pop_size - replaced; i < pop_size; i++) {
                population[i] = random_solution(prob, fe, ne, np, nr, valid_p, valid_r, rng);
                fitness[i] = fe.full_eval(population[i]).fitness();
            }
        }
    }

    fe.optimize_rooms(best_sol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[GACuda] " << iters_done << " gens, " << rt << "s"
                  << " feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard() << " soft=" << final_ev.soft()
                  << " gpu=" << (Cuev.gpu_active ? "on" : "off") << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "GA (CUDA)"};
}
