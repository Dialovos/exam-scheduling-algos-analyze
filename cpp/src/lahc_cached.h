/*
 * LAHC with CachedEvaluator routed through the TEMPLATED nbhd framework.
 * Gets full operator bank (Move/Swap/Kempe/Kick/RoomBeam/RoomOnly) + LAHC
 * acceptance, with O(1) move_delta via cache. No algorithmic regression
 * vs scalar LAHC.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_cached.h"
#include "greedy.h"
#include "neighbourhoods.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

inline AlgoResult solve_lahc_cached(
    const ProblemInstance& prob,
    int max_iterations = 50000,
    int list_length    = 0,
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

    int L_max = (list_length > 0) ? list_length : std::max(50, ne * 5);
    int L_min = std::max(20, L_max / 10);
    int L_mid = (L_min + L_max) / 2;
    std::vector<double> history(L_min, current_fitness);
    int list_len = L_min;

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
            }
            exam_cost[e] = c;
        }
        alias.build(exam_cost);
    };
    recompute_costs();

    nbhd::OpWeights op_weights;
    op_weights.w[static_cast<int>(nbhd::OpType::SHAKE)] = 0.0;
    op_weights.w[static_cast<int>(nbhd::OpType::MOVE)] = 0.40;
    op_weights.w[static_cast<int>(nbhd::OpType::KEMPE)] = 0.20;
    op_weights.w[static_cast<int>(nbhd::OpType::KICK)] = 0.15;
    op_weights.w[static_cast<int>(nbhd::OpType::ROOM_BEAM)] = 0.10;

    int no_improve = 0;
    int iters_done = 0;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;
        if (it % 400 == 0) { ev = fe.full_eval(sol); current_fitness = ev.fitness(); }
        if (it % 500 == 0) recompute_costs();

        double progress = (double)it / max_iterations;
        int target_L = (progress < 0.33) ? L_min : (progress < 0.66 ? L_mid : L_max);
        if (target_L > (int)history.size()) {
            std::fill(history.begin(), history.end(), current_fitness);
            history.resize(target_L, current_fitness);
            list_len = target_L;
        }
        int hist_idx = it % list_len;

        auto lahc_accept = [&](double delta) -> bool {
            double nf = current_fitness + delta;
            return (nf <= current_fitness) || (nf <= history[hist_idx]);
        };

        nbhd::OpType op = op_weights.sample(rng);
        // Pass CachedEvaluator — duck-types as FastEvaluator in templated nbhd
        auto mr = nbhd::select_and_apply(
            op, sol, Ecach, prob, valid_p, valid_r, alias, rng,
            lahc_accept, std::max(3, ne / 20));

        if (mr.applied) {
            current_fitness += mr.delta;
            op_weights.record(op);

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
                    std::fill(history.begin(), history.end(), af);
                }
                current_fitness = af;
            } else no_improve++;
        } else no_improve++;

        history[hist_idx] = current_fitness;
    }

    fe.optimize_rooms(best_sol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[LAHCCached+nbhd] " << iters_done << " iters, " << rt << "s"
                  << " feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard() << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "LAHC (Cached+nbhd)"};
}
