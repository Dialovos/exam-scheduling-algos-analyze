/*
 * tabu.h — Tabu Search with incremental delta evaluation
 *
 * 1. Start from Greedy solution
 * 2. Each iteration: sample exams × candidate periods, pick best non-tabu move
 * 3. Aspiration: accept tabu move if it beats global best
 * 4. Periodic full re-sync to correct accumulated drift
 *
 * Time per iteration: ~O(sample_exams * sample_periods * avg_students_per_exam)
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

inline AlgoResult solve_tabu(
    const ProblemInstance& prob,
    int max_iters    = 200,
    int tabu_tenure  = 15,
    int patience     = 50,
    int seed         = 42,
    bool verbose     = false)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    FastEvaluator fe(prob);
    int ne = prob.n_e(), np = prob.n_p();

    // ── Initial solution from Greedy ──
    auto greedy_res = solve_greedy(prob, false);
    Solution current = greedy_res.sol.copy();
    double cur_fitness = fe.full_eval(current).fitness();

    Solution best = current.copy();
    double best_fitness = cur_fitness;

    if (verbose)
        std::cerr << "[Tabu] Init: fitness=" << cur_fitness << std::endl;

    // Tabu list: (eid * np + pid) → expiry iteration
    std::unordered_map<int, int> tabu_list;
    int no_improve = 0;

    std::mt19937 rng(seed);
    int iters_done = 0;

    for (int it = 0; it < max_iters; it++) {
        iters_done = it + 1;
        double best_delta = 1e18;
        int mv_eid = -1, mv_new_pid = -1, mv_new_rid = -1, mv_old_pid = -1;

        // Sample a subset of exams
        int n_sample = std::min(ne, std::max(10, ne / 3));
        std::vector<int> sample_e(ne);
        std::iota(sample_e.begin(), sample_e.end(), 0);
        std::shuffle(sample_e.begin(), sample_e.end(), rng);
        sample_e.resize(n_sample);

        for (int eid : sample_e) {
            int old_pid = current.period_of[eid];
            int old_rid = current.room_of[eid];
            if (old_pid < 0) continue;

            int dur = fe.exam_dur[eid];

            // Candidate periods (exclude current, must fit duration)
            std::vector<int> cand;
            for (int p = 0; p < np; p++)
                if (p != old_pid && fe.period_dur[p] >= dur)
                    cand.push_back(p);
            if ((int)cand.size() > 12) {
                std::shuffle(cand.begin(), cand.end(), rng);
                cand.resize(12);
            }

            for (int new_pid : cand) {
                double delta = fe.move_delta(current, eid, new_pid, old_rid);

                bool is_tabu = tabu_list.count(eid * np + new_pid)
                               && tabu_list[eid * np + new_pid] > it;
                bool aspiration = (cur_fitness + delta) < best_fitness;

                if ((!is_tabu || aspiration) && delta < best_delta) {
                    best_delta = delta;
                    mv_eid = eid;
                    mv_new_pid = new_pid;
                    mv_new_rid = old_rid;
                    mv_old_pid = old_pid;
                }
            }
        }

        // Apply best move
        if (mv_eid >= 0) {
            fe.apply_move(current, mv_eid, mv_new_pid, mv_new_rid);
            cur_fitness += best_delta;
            tabu_list[mv_eid * np + mv_old_pid] = it + tabu_tenure;

            // Periodic full re-sync
            if (it % 15 == 0)
                cur_fitness = fe.full_eval(current).fitness();

            if (cur_fitness < best_fitness) {
                best = current.copy();
                best_fitness = cur_fitness;
                no_improve = 0;
                if (verbose && it % 20 == 0)
                    std::cerr << "[Tabu] Iter " << it << ": NEW BEST " << best_fitness << std::endl;
            } else {
                no_improve++;
            }
        } else {
            no_improve++;
        }

        if (no_improve >= patience) {
            if (verbose)
                std::cerr << "[Tabu] Stop at iter " << it << " (patience)" << std::endl;
            break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best);

    if (verbose)
        std::cerr << "[Tabu] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best), final_ev, rt, iters_done, "Tabu Search"};
}