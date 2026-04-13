/*
 * Shared neighbourhood operators: Move, Swap, Kempe, Kick, Shake, RoomBeam, RoomOnly.
 * Each takes a templated acceptance function for algorithm-agnostic use.
 * Used by SA, LAHC, GD, WOA, and GVNS.
 *
 * Also provides kempe_detail:: chain-building utilities reused by
 * standalone kempe.h, tabu.h, and the Kempe operator here.
 */

#pragma once

#include "models.h"
#include "evaluator.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <vector>

// ── Kempe chain utilities (shared with kempe.h, tabu.h, etc.) ─

namespace kempe_detail {

struct ChainUndo { int eid, old_pid, old_rid; };

// Build Kempe chain via BFS from seed exam between periods p1 and p2.
// Uses prob.adj format: vector<vector<pair<neighbor_id, shared_students>>>.
inline std::vector<int> build_chain(
    const Solution& sol,
    const std::vector<std::vector<std::pair<int,int>>>& adj,
    int ne, int seed_exam, int p1, int p2)
{
    std::vector<int> chain;
    std::vector<bool> in_chain(ne, false);
    std::queue<int> q;
    q.push(seed_exam);
    in_chain[seed_exam] = true;

    while (!q.empty()) {
        int e = q.front(); q.pop();
        chain.push_back(e);
        int ep = sol.period_of[e];
        int target = (ep == p1) ? p2 : p1;
        for (auto& [nb, _] : adj[e]) {
            if (!in_chain[nb] && sol.period_of[nb] == target) {
                in_chain[nb] = true;
                q.push(nb);
            }
        }
    }
    return chain;
}

// Apply chain swap (p1 <-> p2, keep rooms). Returns undo info.
inline std::vector<ChainUndo> apply_chain(
    Solution& sol, const std::vector<int>& chain, int p1, int p2)
{
    std::vector<ChainUndo> undo;
    undo.reserve(chain.size());
    for (int e : chain) {
        undo.push_back({e, sol.period_of[e], sol.room_of[e]});
        int ep = sol.period_of[e];
        sol.assign(e, (ep == p1) ? p2 : p1, sol.room_of[e]);
    }
    return undo;
}

// Undo a chain swap.
inline void undo_chain(Solution& sol, const std::vector<ChainUndo>& undo) {
    for (auto& u : undo)
        sol.assign(u.eid, u.old_pid, u.old_rid);
}

} // namespace kempe_detail

// ── Neighbourhood operators ────────────────────────────────────

namespace nbhd {

enum class OpType { MOVE, SWAP, KEMPE, KICK, SHAKE, ROOM_BEAM, ROOM_ONLY };
constexpr int N_OPS = 7;

struct MoveResult {
    double delta;
    bool applied;
    int exams_touched;
};

// ── 1. move_single ──────────────────────────────────────────
// Random exam -> random valid (period, room). 70% alias-weighted selection.

template<typename AcceptFn>
inline MoveResult move_single(
    Solution& sol, const FastEvaluator& fe,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    AliasTable& alias, std::mt19937& rng,
    AcceptFn accept_fn)
{
    int ne = fe.ne;
    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
    auto& vp = valid_p[eid];
    auto& vr = valid_r[eid];
    if (vp.empty() || vr.empty()) return {0, false, 0};

    int new_pid = vp[rng() % vp.size()];
    int new_rid = vr[rng() % vr.size()];
    if (new_pid == sol.period_of[eid] && new_rid == sol.room_of[eid])
        return {0, false, 0};

    double delta = fe.move_delta(sol, eid, new_pid, new_rid);
    if (accept_fn(delta)) {
        fe.apply_move(sol, eid, new_pid, new_rid);
        return {delta, true, 1};
    }
    return {delta, false, 0};
}

// ── 2. swap_two ─────────────────────────────────────────────
// Exchange periods of two exams (keep rooms). Duration-compatible check.
// Computes combined delta via temp-apply/undo.

template<typename AcceptFn>
inline MoveResult swap_two(
    Solution& sol, const FastEvaluator& fe,
    AliasTable& alias, std::mt19937& rng,
    AcceptFn accept_fn)
{
    int ne = fe.ne;
    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    int e1 = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
    int e2 = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
    if (e1 == e2) return {0, false, 0};

    int p1 = sol.period_of[e1], r1 = sol.room_of[e1];
    int p2 = sol.period_of[e2], r2 = sol.room_of[e2];
    if (p1 < 0 || p2 < 0 || p1 == p2) return {0, false, 0};

    // Duration compatibility
    if (fe.exam_dur[e1] > fe.period_dur[p2] || fe.exam_dur[e2] > fe.period_dur[p1])
        return {0, false, 0};

    // Compute combined delta: temp-apply e1->p2, measure e2->p1, undo
    double d1 = fe.move_delta(sol, e1, p2, r1);
    fe.apply_move(sol, e1, p2, r1);
    double d2 = fe.move_delta(sol, e2, p1, r2);
    fe.apply_move(sol, e1, p1, r1); // undo first move
    double delta = d1 + d2;

    if (accept_fn(delta)) {
        fe.apply_move(sol, e1, p2, r1);
        fe.apply_move(sol, e2, p1, r2);
        return {delta, true, 2};
    }
    return {delta, false, 0};
}

// ── 3. kempe_chain ──────────────────────────────────────────
// BFS Kempe chain swap between two periods. Uses partial_eval for delta.
// Chain is applied before evaluation and undone if not accepted.

template<typename AcceptFn>
inline MoveResult kempe_chain(
    Solution& sol, const FastEvaluator& fe,
    const ProblemInstance& prob,
    AliasTable& alias, std::mt19937& rng,
    AcceptFn accept_fn)
{
    int ne = fe.ne, np = fe.np;
    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_int_distribution<int> dp(0, np - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
    int p1 = sol.period_of[eid];
    if (p1 < 0) return {0, false, 0};
    int p2 = dp(rng);
    if (p2 == p1) return {0, false, 0};

    auto chain = kempe_detail::build_chain(sol, prob.adj, ne, eid, p1, p2);
    if (chain.empty() || (int)chain.size() > ne / 4)
        return {0, false, 0};

    // Partial eval before swap
    auto old_pe = fe.partial_eval(sol, chain);
    // Apply chain swap
    auto undo = kempe_detail::apply_chain(sol, chain, p1, p2);
    // Partial eval after swap
    auto new_pe = fe.partial_eval(sol, chain);
    double delta = new_pe.fitness() - old_pe.fitness();

    if (accept_fn(delta)) {
        return {delta, true, (int)chain.size()};
    }
    // Undo if not accepted
    kempe_detail::undo_chain(sol, undo);
    return {delta, false, 0};
}

// ── 4. kick ─────────────────────────────────────────────────
// Steepest-descent relocation: scan ALL valid (period, room) for one exam.
// Stronger than move_single because it exhaustively finds the best slot.

template<typename AcceptFn>
inline MoveResult kick(
    Solution& sol, const FastEvaluator& fe,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    AliasTable& alias, std::mt19937& rng,
    AcceptFn accept_fn)
{
    int ne = fe.ne;
    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
    auto& vp = valid_p[eid];
    auto& vr = valid_r[eid];
    if (vp.empty() || vr.empty()) return {0, false, 0};
    if (sol.period_of[eid] < 0) return {0, false, 0};

    // Exhaustive scan: find best (period, room)
    double best_delta = 0;
    int best_pid = -1, best_rid = -1;
    for (int pid : vp) {
        for (int rid : vr) {
            if (pid == sol.period_of[eid] && rid == sol.room_of[eid]) continue;
            double d = fe.move_delta(sol, eid, pid, rid);
            if (d < best_delta) {
                best_delta = d;
                best_pid = pid;
                best_rid = rid;
            }
        }
    }

    if (best_pid < 0) return {0, false, 0};

    if (accept_fn(best_delta)) {
        fe.apply_move(sol, eid, best_pid, best_rid);
        return {best_delta, true, 1};
    }
    return {best_delta, false, 0};
}

// ── 5. shake ────────────────────────────────────────────────
// Blind perturbation: randomly relocate `intensity` exams. Always applies.
// Used for diversification (reheat, escape local optima).

inline MoveResult shake(
    Solution& sol, const FastEvaluator& fe,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    int intensity, std::mt19937& rng)
{
    int ne = fe.ne;
    std::uniform_int_distribution<int> de(0, ne - 1);
    double cumulative_delta = 0;
    int applied_count = 0;

    for (int k = 0; k < intensity; k++) {
        int eid = de(rng);
        auto& vp = valid_p[eid];
        auto& vr = valid_r[eid];
        if (vp.empty() || vr.empty()) continue;

        int new_pid = vp[rng() % vp.size()];
        int new_rid = vr[rng() % vr.size()];
        double d = fe.move_delta(sol, eid, new_pid, new_rid);
        fe.apply_move(sol, eid, new_pid, new_rid);
        cumulative_delta += d;
        applied_count++;
    }
    return {cumulative_delta, true, applied_count};
}

// ── 6. room_beam ────────────────────────────────────────────
// Fixed period, steepest-descent room reassignment for one exam.

template<typename AcceptFn>
inline MoveResult room_beam(
    Solution& sol, const FastEvaluator& fe,
    const std::vector<std::vector<int>>& valid_r,
    AliasTable& alias, std::mt19937& rng,
    AcceptFn accept_fn)
{
    int ne = fe.ne;
    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
    int cur_pid = sol.period_of[eid];
    if (cur_pid < 0) return {0, false, 0};
    auto& vr = valid_r[eid];
    if (vr.size() <= 1) return {0, false, 0};

    // Evaluate all valid rooms, pick best
    double best_delta = 0;
    int best_rid = -1;
    for (int rid : vr) {
        if (rid == sol.room_of[eid]) continue;
        double d = fe.move_delta(sol, eid, cur_pid, rid);
        if (d < best_delta) {
            best_delta = d;
            best_rid = rid;
        }
    }

    if (best_rid < 0) return {0, false, 0};

    if (accept_fn(best_delta)) {
        fe.apply_move(sol, eid, cur_pid, best_rid);
        return {best_delta, true, 1};
    }
    return {best_delta, false, 0};
}

// ── 7. room_only ────────────────────────────────────────────
// Same period, random new room.

template<typename AcceptFn>
inline MoveResult room_only(
    Solution& sol, const FastEvaluator& fe,
    const std::vector<std::vector<int>>& valid_r,
    AliasTable& alias, std::mt19937& rng,
    AcceptFn accept_fn)
{
    int ne = fe.ne;
    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
    int cur_pid = sol.period_of[eid];
    if (cur_pid < 0) return {0, false, 0};
    auto& vr = valid_r[eid];
    if (vr.empty()) return {0, false, 0};

    int new_rid = vr[rng() % vr.size()];
    if (new_rid == sol.room_of[eid]) return {0, false, 0};

    double delta = fe.move_delta(sol, eid, cur_pid, new_rid);
    if (accept_fn(delta)) {
        fe.apply_move(sol, eid, cur_pid, new_rid);
        return {delta, true, 1};
    }
    return {delta, false, 0};
}

// ── Dispatcher ──────────────────────────────────────────────
// Select and apply a neighbourhood operator by type.

template<typename AcceptFn>
inline MoveResult select_and_apply(
    OpType op,
    Solution& sol, const FastEvaluator& fe,
    const ProblemInstance& prob,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    AliasTable& alias, std::mt19937& rng,
    AcceptFn accept_fn,
    int shake_intensity = 3)
{
    switch (op) {
        case OpType::MOVE:      return move_single(sol, fe, valid_p, valid_r, alias, rng, accept_fn);
        case OpType::SWAP:      return swap_two(sol, fe, alias, rng, accept_fn);
        case OpType::KEMPE:     return kempe_chain(sol, fe, prob, alias, rng, accept_fn);
        case OpType::KICK:      return kick(sol, fe, valid_p, valid_r, alias, rng, accept_fn);
        case OpType::SHAKE:     return shake(sol, fe, valid_p, valid_r, shake_intensity, rng);
        case OpType::ROOM_BEAM: return room_beam(sol, fe, valid_r, alias, rng, accept_fn);
        case OpType::ROOM_ONLY: return room_only(sol, fe, valid_r, alias, rng, accept_fn);
    }
    return {0, false, 0};
}

// ── Operator weights + activity tracking ────────────────────

struct OpWeights {
    // Default: Move=35%, Swap=15%, Kempe=20%, Kick=10%, Shake=5%, RoomBeam=10%, RoomOnly=5%
    double w[N_OPS] = {0.35, 0.15, 0.20, 0.10, 0.05, 0.10, 0.05};

    OpType sample(std::mt19937& rng) const {
        double total = 0;
        for (int i = 0; i < N_OPS; i++) total += w[i];
        std::uniform_real_distribution<double> d(0, total);
        double r = d(rng);
        double acc = 0;
        for (int i = 0; i < N_OPS; i++) {
            acc += w[i];
            if (r <= acc) return static_cast<OpType>(i);
        }
        return OpType::MOVE;
    }

    // Per-neighbourhood activity tracking (FastSA extension)
    int activity[N_OPS] = {};
    void record(OpType op) { activity[static_cast<int>(op)]++; }
    void reset() { std::fill(activity, activity + N_OPS, 0); }
};

} // namespace nbhd
