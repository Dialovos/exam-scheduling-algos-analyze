/*
 * Phase 2a polish pipeline — SIMD-accelerated post-processing.
 *
 * Applied to a *feasible* solution (does nothing useful on infeasible). The
 * pipeline targets the 1-3% soft cost that SA/Tabu leaves on the table after
 * their acceptance criteria stall.
 *
 * Stages (each strictly non-worsening, monotone):
 *   1. polish_single_moves   — exhaustive single-exam steepest descent.
 *      For each exam × (valid period × valid room), evaluate move_delta
 *      via SIMD. Apply if strictly negative. Iterate until a full pass
 *      finds no improvement.
 *   2. polish_pair_swaps     — exhaustive 2-exam swap (candidate-limited).
 *      For each exam, try swapping with its top-K most-conflicting partners
 *      (K = 25 default). SIMD-scored. Budget-limited.
 *   3. polish_rooms          — delegate to FastEvaluator::optimize_rooms
 *      (existing per-period greedy).
 *
 * All stages preserve hard constraints: a move is only applied when both
 *  (a) hard delta ≤ 0 AND
 *  (b) fitness delta < 0.
 * This keeps feasibility invariant.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_simd.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace polish_detail {

struct PolishStats {
    int single_moves_applied = 0;
    int single_passes        = 0;
    int swaps_applied        = 0;
    int swap_pairs_scanned   = 0;
    double runtime_sec       = 0;
    int soft_before          = 0;
    int soft_after           = 0;
};

// Stage 1: single-exam exhaustive steepest descent.
// Strictly non-worsening. Stops when no move improves (or iter cap hit).
inline int polish_single_moves(
    Solution& sol, FastEvaluator& fe, const FastEvaluatorSIMD& fes,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    int max_passes, int& total_applied, int& passes_run)
{
    int ne = fe.ne;
    int applied_this_call = 0;
    for (int pass = 0; pass < max_passes; pass++) {
        passes_run++;
        bool any_improvement = false;
        for (int eid = 0; eid < ne; eid++) {
            if (sol.period_of[eid] < 0) continue;
            int best_pid = -1, best_rid = -1;
            double best_delta = -0.5;  // strict improvement
            int cp = sol.period_of[eid], cr = sol.room_of[eid];
            for (int pid : valid_p[eid]) {
                for (int rid : valid_r[eid]) {
                    if (pid == cp && rid == cr) continue;
                    double d = fes.move_delta_simd(sol, eid, pid, rid);
                    // Accept strictly improving moves only; reject any that
                    // introduces hard violations (dh contribution is in 1e5s).
                    if (d < best_delta) {
                        best_delta = d;
                        best_pid = pid; best_rid = rid;
                    }
                }
            }
            if (best_pid >= 0) {
                fe.apply_move(sol, eid, best_pid, best_rid);
                applied_this_call++;
                total_applied++;
                any_improvement = true;
            }
        }
        if (!any_improvement) break;
    }
    return applied_this_call;
}

// Stage 2: pair-swap exhaustive (candidate-limited to top-K neighbours by
// shared-student count). O(ne × K × 2) SIMD evals per pass.
inline int polish_pair_swaps(
    Solution& sol, FastEvaluator& fe, const FastEvaluatorSIMD& fes,
    const ProblemInstance& P, int top_k, int max_passes,
    std::chrono::high_resolution_clock::time_point deadline,
    int& total_applied, int& total_scanned)
{
    int ne = fe.ne;
    int applied_this_call = 0;

    // Per-exam ranked partners (descending shared-student count)
    std::vector<std::vector<int>> partners(ne);
    for (int e = 0; e < ne; e++) {
        auto v = P.adj[e];
        std::sort(v.begin(), v.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });
        for (int i = 0; i < std::min(top_k, (int)v.size()); i++)
            partners[e].push_back(v[i].first);
    }

    for (int pass = 0; pass < max_passes; pass++) {
        if (std::chrono::high_resolution_clock::now() >= deadline) break;
        bool any = false;
        for (int ea = 0; ea < ne; ea++) {
            int pa = sol.period_of[ea]; if (pa < 0) continue;
            int ra = sol.room_of[ea];
            for (int eb : partners[ea]) {
                if (eb == ea) continue;
                int pb = sol.period_of[eb]; if (pb < 0) continue;
                int rb = sol.room_of[eb];
                if (pa == pb && ra == rb) continue;
                // Must respect domain filters for both exams
                if (fe.exam_dur[ea] > fe.period_dur[pb]) continue;
                if (fe.exam_dur[eb] > fe.period_dur[pa]) continue;
                total_scanned++;

                double d1 = fes.move_delta_simd(sol, ea, pb, ra);
                fe.apply_move(sol, ea, pb, ra);
                double d2 = fes.move_delta_simd(sol, eb, pa, rb);
                // Undo in-place (state must be restored regardless of accept)
                sol.assign(eb, pb, rb);
                sol.assign(ea, pa, ra);
                double td = d1 + d2;
                if (td < -0.5) {
                    // Apply
                    fe.apply_move(sol, ea, pb, ra);
                    fe.apply_move(sol, eb, pa, rb);
                    applied_this_call++;
                    total_applied++;
                    any = true;
                    break;  // ea has moved; restart its partner loop next pass
                }
            }
            if (std::chrono::high_resolution_clock::now() >= deadline) break;
        }
        if (!any) break;
    }
    return applied_this_call;
}

} // namespace polish_detail

// Top-level: apply all three polish stages within a wall-clock budget.
// Preserves feasibility strictly. Returns stats struct for logging.
// Total budget is split: 40% single, 40% pair, 20% room (room is cheap).
inline polish_detail::PolishStats polish_solution(
    const ProblemInstance& P, Solution& sol, double budget_sec = 10.0,
    bool verbose = false)
{
    using clock = std::chrono::high_resolution_clock;
    polish_detail::PolishStats st;
    auto t0 = clock::now();

    FastEvaluator fe(P);
    FastEvaluatorSIMD fes(fe);
    int ne = fe.ne, np = fe.np, nr = fe.nr;

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (fe.exam_dur[e] <= fe.period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (fe.exam_enroll[e] <= fe.room_cap[r]) valid_r[e].push_back(r);
    }

    st.soft_before = fe.full_eval(sol).soft();

    auto single_deadline = t0 + std::chrono::duration_cast<clock::duration>(
        std::chrono::duration<double>(budget_sec * 0.4));
    auto pair_deadline = t0 + std::chrono::duration_cast<clock::duration>(
        std::chrono::duration<double>(budget_sec * 0.8));

    // Stage 1: single-move steepest descent until convergence or time out
    int single_passes_cap = 50;
    int passes = 0, applied = 0;
    while (clock::now() < single_deadline && passes < single_passes_cap) {
        int before = st.single_moves_applied;
        polish_detail::polish_single_moves(sol, fe, fes, valid_p, valid_r,
                                           1, st.single_moves_applied, st.single_passes);
        if (st.single_moves_applied == before) break;  // converged
        passes++;
    }

    // Stage 2: pair swaps
    polish_detail::polish_pair_swaps(sol, fe, fes, P, 25, 20,
                                     pair_deadline,
                                     st.swaps_applied, st.swap_pairs_scanned);

    // Stage 3: room polish (cheap)
    fe.optimize_rooms(sol);

    st.soft_after = fe.full_eval(sol).soft();
    auto t1 = clock::now();
    st.runtime_sec = std::chrono::duration<double>(t1 - t0).count();

    if (verbose) {
        std::cerr << "[polish] soft " << st.soft_before << " → " << st.soft_after
                  << " (Δ=" << (st.soft_after - st.soft_before) << ")"
                  << "  single=" << st.single_moves_applied
                  << " (in " << st.single_passes << " passes)"
                  << "  swaps=" << st.swaps_applied << "/" << st.swap_pairs_scanned
                  << "  " << st.runtime_sec << "s" << std::endl;
    }
    return st;
}
