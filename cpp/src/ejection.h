/*
 * Multi-depth ejection chains (Glover 1996).
 *
 * Starts from a seed exam, finds its best (period, room) slot. If that slot
 * is occupied by another exam, push the occupant to ITS best next slot, and
 * so on up to `max_depth` hops. Accept the full chain if the cumulative
 * delta is improving; otherwise roll back.
 *
 * Strictly more powerful than Kempe chains (2-color, period-only): ejection
 * chains are multi-color and move in (period, room) space, escaping plateaus
 * where Kempe can't find feasible swaps.
 *
 * Templated on Evaluator so FastEvaluator and CachedEvaluator both work.
 * Uses fe.apply_move — cache stays consistent when Ev = CachedEvaluator.
 */

#pragma once

#include "models.h"
#include "evaluator.h"

#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>

namespace ejection {

struct Step {
    int eid;
    int old_pid, old_rid;
    int new_pid, new_rid;
    double delta_at_step;
};

// Locate the one exam currently at (pid, rid). Returns -1 if empty.
// O(ne) in the worst case — fine for max_depth × rare calls.
inline int occupant_at(const Solution& sol, int pid, int rid, int exclude_eid) {
    for (int e = 0; e < (int)sol.period_of.size(); e++)
        if (e != exclude_eid && sol.period_of[e] == pid && sol.room_of[e] == rid)
            return e;
    return -1;
}

// Build and (optionally) apply a deep ejection chain.
//
// Returns cumulative delta of the chain as applied to sol. If
// apply_on_improve == true and delta < -threshold, the chain stays applied
// and chain_out contains the step list. Otherwise sol is rolled back to
// entry state and chain_out is cleared.
//
// Caller guarantees sol is in a consistent state that fe knows about
// (CachedEvaluator: initialize() has been called).
template <typename Ev, typename RngT>
inline double try_deep_chain(
    Solution& sol, const Ev& fe,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    int start_eid,
    int max_depth,
    int samples_per_step,
    RngT& rng,
    std::vector<Step>& chain_out,
    bool apply_on_improve = true,
    double improve_threshold = -0.5)
{
    chain_out.clear();
    std::unordered_set<int> visited;
    visited.insert(start_eid);

    double total_delta = 0;
    int cur_eid = start_eid;

    for (int d = 0; d < max_depth; d++) {
        int cp = sol.period_of[cur_eid];
        int cr = sol.room_of[cur_eid];
        if (cp < 0) break;

        const auto& vp = valid_p[cur_eid];
        const auto& vr = valid_r[cur_eid];
        if (vp.empty() || vr.empty()) break;

        // Sample candidates and pick best delta
        double best_d = 1e18;
        int best_p = -1, best_r = -1;
        int sp = std::min(samples_per_step, (int)vp.size());
        int sr = std::min(std::max(1, samples_per_step / 2), (int)vr.size());
        for (int i = 0; i < sp; i++) {
            int pid = vp[rng() % vp.size()];
            if (pid == cp) continue;
            for (int j = 0; j < sr; j++) {
                int rid = vr[rng() % vr.size()];
                double dd = fe.move_delta(sol, cur_eid, pid, rid);
                if (dd < best_d) { best_d = dd; best_p = pid; best_r = rid; }
            }
        }
        if (best_p < 0) break;

        // Find occupant (if any) at the target slot
        int next_eid = occupant_at(sol, best_p, best_r, cur_eid);

        // Apply the move — cache stays in sync via fe.apply_move
        fe.apply_move(sol, cur_eid, best_p, best_r);
        chain_out.push_back({cur_eid, cp, cr, best_p, best_r, best_d});
        total_delta += best_d;

        // If no displacement, chain ends
        if (next_eid < 0) break;
        // Cycle detection: don't revisit
        if (visited.count(next_eid)) break;
        visited.insert(next_eid);

        // Continue with the displaced occupant
        cur_eid = next_eid;
    }

    bool improved = (total_delta < improve_threshold);
    if (apply_on_improve && improved) {
        return total_delta;  // keep applied
    }

    // Roll back in reverse order
    for (auto it = chain_out.rbegin(); it != chain_out.rend(); ++it) {
        fe.apply_move(sol, it->eid, it->old_pid, it->old_rid);
    }
    if (!apply_on_improve || !improved) chain_out.clear();
    return total_delta;
}

} // namespace ejection
