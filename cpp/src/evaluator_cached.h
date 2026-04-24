/*
 * Incremental cached-fitness evaluator — Phase 2b.
 *
 * Trades O(deg × np) work on apply_move for O(1) (+ O(|phc|) + O(nr)) work
 * per move_delta. Wins when the move:apply ratio is high (Tabu ~15000:1,
 * SA ~50:1). For Tabu this is typically 3-7× on iteration throughput on
 * top of SIMD.
 *
 * Caching strategy:
 *   soft_contrib[e][p]  — sum over adj[e] of soft contribution that exam e
 *                         would accrue if placed in period p, given every
 *                         other exam's current slot. Updated on apply_move
 *                         only for e and e's neighbours.
 *   hard_contrib[e][p]  — analogous for hard conflicts. Since hard dominates
 *                         (multiplied by 1e5), we keep it as an int delta.
 *
 * What's NOT cached (still computed lazily per move_delta call):
 *   • Room-occupancy, room-exclusive, period-utilisation — cheap, state-only
 *   • Period hard constraints (phcs) — per-exam list usually 0-3 items
 *   • Room penalty — O(1) lookup, no need to cache
 *   • Front-load — O(1) lookup
 *
 * Correctness oracle: move_delta_cached is validated against FastEvaluator
 * ::move_delta to within 1e-6 on every bench iteration.
 *
 * Invariant: the cache is valid iff the current Solution state matches the
 * state that last-called initialize() saw, updated only via apply_move().
 * Direct sol.assign() bypasses the cache — don't do that with this wrapper.
 */

#pragma once

#include "evaluator.h"
#include "evaluator_simd.h"

#include <cstdlib>
#include <vector>

class CachedEvaluator {
public:
    const FastEvaluator& E;
    const ProblemInstance& P;
    int ne, np, nr;

    // ── Public-member mirrors for duck-typed templated code ──
    // These aliases let CachedEvaluator stand in for FastEvaluator in any
    // templated nbhd operator that accesses members like fe.room_cap or
    // fe.period_day. References, so zero duplication, always in sync.
    const std::vector<int>& exam_dur;
    const std::vector<int>& exam_enroll;
    const std::vector<int>& period_dur;
    const std::vector<int>& period_day;
    const std::vector<int>& period_pen;
    const std::vector<int>& period_daypos;
    const std::vector<int>& room_cap;
    const std::vector<int>& room_pen;
    const std::unordered_set<int>& large_exams;
    const std::unordered_set<int>& last_periods;
    const std::unordered_set<int>& rhc_exams;
    const std::vector<std::vector<std::pair<int, int>>>& phc_by_exam;
    const int& w_2row;
    const int& w_2day;
    const int& w_spread;
    const int& w_mixed;
    const int& fl_n_largest;
    const int& fl_n_last;
    const int& fl_penalty;

    // soft_contrib[e][p] — soft cost if e placed in p (adj-summed).
    // hard_contrib[e][p] — weighted adj conflict count (shared students).
    // mutable: cache is conceptually side-state; apply_move stays const
    // externally so CachedEvaluator substitutes for FastEvaluator in
    // templated nbhd operators that take `const Ev& fe`.
    mutable std::vector<std::vector<int32_t>> soft_contrib;
    mutable std::vector<std::vector<int32_t>> hard_contrib;

    explicit CachedEvaluator(const FastEvaluator& e)
        : E(e), P(e.P),
          exam_dur(e.exam_dur), exam_enroll(e.exam_enroll),
          period_dur(e.period_dur), period_day(e.period_day),
          period_pen(e.period_pen), period_daypos(e.period_daypos),
          room_cap(e.room_cap), room_pen(e.room_pen),
          large_exams(e.large_exams), last_periods(e.last_periods),
          rhc_exams(e.rhc_exams), phc_by_exam(e.phc_by_exam),
          w_2row(e.w_2row), w_2day(e.w_2day), w_spread(e.w_spread),
          w_mixed(e.w_mixed),
          fl_n_largest(e.fl_n_largest), fl_n_last(e.fl_n_last),
          fl_penalty(e.fl_penalty)
    {
        ne = E.ne; np = E.np; nr = E.nr;
        soft_contrib.assign(ne, std::vector<int32_t>(np, 0));
        hard_contrib.assign(ne, std::vector<int32_t>(np, 0));
    }

    // Expose partial_eval / full_eval / count_hard_fast / optimize_rooms /
    // move_delta_period / move_delta_hard via forwarding so templated code
    // can call them on CachedEvaluator the same way as FastEvaluator.
    EvalResult full_eval(const Solution& sol) const { return E.full_eval(sol); }
    EvalResult partial_eval(const Solution& sol, const std::vector<int>& a) const { return E.partial_eval(sol, a); }
    int count_hard_fast(const Solution& sol) const { return E.count_hard_fast(sol); }
    void optimize_rooms(Solution& sol) const { E.optimize_rooms(sol); }
    int move_delta_hard(const Solution& sol, int eid, int p, int r) const { return E.move_delta_hard(sol, eid, p, r); }
    FastEvaluator::PeriodDelta move_delta_period_scalar(const Solution& sol, int eid, int p) const { return E.move_delta_period(sol, eid, p); }

    // Build both tables from scratch given a fully-assigned Solution.
    // O(ne * np * avg_deg). Called once per run.
    void initialize(const Solution& sol) const {
        for (int e = 0; e < ne; e++)
            rebuild_contrib_for(e, sol);
    }

    // Single-exam rebuild: recompute contribution[e][*] by walking adj[e].
    // Use when e's neighborhood has entirely changed (init, large restart).
    void rebuild_contrib_for(int e, const Solution& sol) const {
        const auto& po = sol.period_of;
        int w_2row = E.w_2row, w_2day = E.w_2day, w_spread = E.w_spread;

        // Zero both rows
        std::fill(soft_contrib[e].begin(), soft_contrib[e].end(), 0);
        std::fill(hard_contrib[e].begin(), hard_contrib[e].end(), 0);

        for (auto& pr : P.adj[e]) {
            int other = pr.first;
            int cnt   = pr.second;
            int o_pid = po[other];
            if (o_pid < 0) continue;
            int o_day  = E.period_day[o_pid];
            int o_dpos = E.period_daypos[o_pid];

            for (int p = 0; p < np; p++) {
                int d_p = E.period_day[p];
                int pos_p = E.period_daypos[p];
                // hard: conflict if same period
                if (p == o_pid) hard_contrib[e][p] += cnt;
                // soft: proximity
                if (d_p == o_day) {
                    int g = std::abs(pos_p - o_dpos);
                    if (g == 1)      soft_contrib[e][p] += w_2row * cnt;
                    else if (g > 1)  soft_contrib[e][p] += w_2day * cnt;
                }
                int og = std::abs(p - o_pid);
                if (og > 0 && og <= w_spread) soft_contrib[e][p] += cnt;
            }
        }
    }

    // Incremental: one neighbor changed from old_nbp to new_nbp. Update
    // all entries of contrib[e][*] that depend on that neighbor.
    // O(np) per call.
    void update_contrib_for_neighbor_move(int e, int nb, int cnt,
                                          int old_nbp, int new_nbp) const
    {
        int w_2row = E.w_2row, w_2day = E.w_2day, w_spread = E.w_spread;

        int old_day = (old_nbp >= 0) ? E.period_day[old_nbp] : -1;
        int old_dpos = (old_nbp >= 0) ? E.period_daypos[old_nbp] : -1;
        int new_day = (new_nbp >= 0) ? E.period_day[new_nbp] : -1;
        int new_dpos = (new_nbp >= 0) ? E.period_daypos[new_nbp] : -1;

        for (int p = 0; p < np; p++) {
            int d_p = E.period_day[p];
            int pos_p = E.period_daypos[p];

            // Remove old contribution (nb was at old_nbp)
            if (old_nbp >= 0) {
                if (p == old_nbp) hard_contrib[e][p] -= cnt;
                if (d_p == old_day) {
                    int g = std::abs(pos_p - old_dpos);
                    if (g == 1)     soft_contrib[e][p] -= w_2row * cnt;
                    else if (g > 1) soft_contrib[e][p] -= w_2day * cnt;
                }
                int og = std::abs(p - old_nbp);
                if (og > 0 && og <= w_spread) soft_contrib[e][p] -= cnt;
            }

            // Add new contribution (nb is at new_nbp)
            if (new_nbp >= 0) {
                if (p == new_nbp) hard_contrib[e][p] += cnt;
                if (d_p == new_day) {
                    int g = std::abs(pos_p - new_dpos);
                    if (g == 1)     soft_contrib[e][p] += w_2row * cnt;
                    else if (g > 1) soft_contrib[e][p] += w_2day * cnt;
                }
                int og = std::abs(p - new_nbp);
                if (og > 0 && og <= w_spread) soft_contrib[e][p] += cnt;
            }
        }
    }

    // Intercepting apply_move: updates cache, delegates to FastEvaluator.
    // After this call, all cached contribs reflect the new solution state.
    // const on `this` (cache is mutable) so CachedEvaluator drops into
    // templated nbhd signatures that take `const Ev& fe`.
    void apply_move(Solution& sol, int e, int new_pid, int new_rid) const {
        int old_pid = sol.period_of[e];

        // Delegate the actual state mutation
        E.apply_move(sol, e, new_pid, new_rid);

        // e's own contrib: full rebuild (its own neighbours didn't move,
        // but caching "contribution if *I* move to each period" doesn't
        // depend on e's current position — it's a lookup of what I'd
        // contribute at any potential slot. Rebuilding costs O(|adj[e]|×np)
        // but is NOT always needed if we track e's position separately.
        // We rebuild to be safe — e's neighbours haven't changed so the
        // result is actually the same as before. Skip this rebuild.)
        //
        // Instead: the neighbours of e now see e at a new slot — update
        // their contrib vectors incrementally.
        for (auto& pr : P.adj[e]) {
            int nb = pr.first;
            int cnt = pr.second;
            update_contrib_for_neighbor_move(nb, e, cnt, old_pid, new_pid);
        }
    }

    // Period-only delta using cache. Matches FastEvaluator::move_delta_period.
    // Returns (dh, ds) EXCLUDING new-room contributions — caller adds:
    //   dh += room_occupancy_delta(pid, rid)     -- O(1)
    //   ds += room_pen[rid]                       -- O(1)
    // RHC new-room contribution is NOT included (caller handles if needed).
    struct PeriodDelta { double dh, ds; };

    PeriodDelta move_delta_period(const Solution& sol, int eid, int new_pid) const {
        int old_pid = sol.period_of[eid];
        int old_rid = sol.room_of[eid];
        double dh = 0, ds = 0;

        dh += hard_contrib[eid][new_pid] - hard_contrib[eid][old_pid];
        ds += soft_contrib[eid][new_pid] - soft_contrib[eid][old_pid];

        int dur = E.exam_dur[eid];
        if (dur > E.period_dur[old_pid]) dh -= 1;
        if (dur > E.period_dur[new_pid]) dh += 1;

        int enr = E.exam_enroll[eid];
        int old_total = sol.get_pr_enroll(old_pid, old_rid);
        dh -= ((old_total > E.room_cap[old_rid]) ? 1.0 : 0.0) -
              (((old_total - enr) > E.room_cap[old_rid]) ? 1.0 : 0.0);

        ds += E.period_pen[new_pid] - E.period_pen[old_pid];
        ds -= E.room_pen[old_rid];

        if (E.large_exams.count(eid) && E.fl_penalty > 0) {
            bool was = E.last_periods.count(old_pid) > 0;
            bool will = E.last_periods.count(new_pid) > 0;
            if (was && !will)      ds -= E.fl_penalty;
            else if (!was && will) ds += E.fl_penalty;
        }

        for (auto& pc : E.phc_by_exam[eid]) {
            int other = pc.first, tc = pc.second;
            int o_pid = sol.period_of[other]; if (o_pid < 0) continue;
            if (tc == 0)      { if (old_pid != o_pid) dh -= 1; if (new_pid != o_pid) dh += 1; }
            else if (tc == 1) { if (old_pid == o_pid) dh -= 1; if (new_pid == o_pid) dh += 1; }
            else if (tc == 2) { if (old_pid <= o_pid) dh -= 1; if (new_pid <= o_pid) dh += 1; }
            else if (tc == 3) { if (old_pid >= o_pid) dh -= 1; if (new_pid >= o_pid) dh += 1; }
        }

        if (!E.rhc_exams.empty()) {
            int oc = sol.get_pr_count(old_pid, old_rid);
            if (E.rhc_exams.count(eid) && oc > 1) dh -= 1;
            for (int re : E.rhc_exams) {
                if (re == eid || re >= ne) continue;
                if (sol.period_of[re] == old_pid && sol.room_of[re] == old_rid && oc == 2)
                    dh -= 1;
            }
        }

        return {dh, ds};
    }

    // The fast path: O(|phc_by_exam[e]|) + O(|rhc_exams|) worst case,
    // usually O(1) because phcs/rhcs are rare.
    double move_delta(const Solution& sol, int eid, int new_pid, int new_rid) const {
        int old_pid = sol.period_of[eid];
        if (old_pid < 0) return E.move_delta(sol, eid, new_pid, new_rid);
        int old_rid = sol.room_of[eid];
        if (old_pid == new_pid && old_rid == new_rid) return 0.0;

        // ── Cached portion (was the student/adj double-loop) ──
        double dh = hard_contrib[eid][new_pid] - hard_contrib[eid][old_pid];
        double ds = soft_contrib[eid][new_pid] - soft_contrib[eid][old_pid];

        // ── Non-cached: duration, room occupancy, phcs, rhcs, penalties ──
        int dur = E.exam_dur[eid];
        if (dur > E.period_dur[old_pid]) dh -= 1;
        if (dur > E.period_dur[new_pid]) dh += 1;

        int enr = E.exam_enroll[eid];
        int old_total = sol.get_pr_enroll(old_pid, old_rid);
        int new_total = sol.get_pr_enroll(new_pid, new_rid);
        dh -= ((old_total > E.room_cap[old_rid]) ? 1 : 0) -
              (((old_total - enr) > E.room_cap[old_rid]) ? 1 : 0);
        dh += (((new_total + enr) > E.room_cap[new_rid]) ? 1 : 0) -
              ((new_total > E.room_cap[new_rid]) ? 1 : 0);

        ds += E.period_pen[new_pid] - E.period_pen[old_pid];
        ds += E.room_pen[new_rid] - E.room_pen[old_rid];

        if (E.large_exams.count(eid) && E.fl_penalty > 0) {
            bool was = E.last_periods.count(old_pid) > 0;
            bool will = E.last_periods.count(new_pid) > 0;
            if (was && !will)      ds -= E.fl_penalty;
            else if (!was && will) ds += E.fl_penalty;
        }

        for (auto& pc : E.phc_by_exam[eid]) {
            int other = pc.first, tcode = pc.second;
            int o_pid = sol.period_of[other]; if (o_pid < 0) continue;
            if (tcode == 0)      { if (old_pid != o_pid) dh -= 1; if (new_pid != o_pid) dh += 1; }
            else if (tcode == 1) { if (old_pid == o_pid) dh -= 1; if (new_pid == o_pid) dh += 1; }
            else if (tcode == 2) { if (old_pid <= o_pid) dh -= 1; if (new_pid <= o_pid) dh += 1; }
            else if (tcode == 3) { if (old_pid >= o_pid) dh -= 1; if (new_pid >= o_pid) dh += 1; }
        }

        if (!E.rhc_exams.empty()) {
            int oc = sol.get_pr_count(old_pid, old_rid);
            int nc = sol.get_pr_count(new_pid, new_rid);
            bool eid_rhc = E.rhc_exams.count(eid) > 0;
            if (eid_rhc) {
                if (oc > 1) dh -= 1;
                if (nc > 0) dh += 1;
            }
            for (int re : E.rhc_exams) {
                if (re == eid || re >= ne) continue;
                int rp = sol.period_of[re]; if (rp < 0) continue;
                int rr = sol.room_of[re];
                if (rp == old_pid && rr == old_rid && oc == 2) dh -= 1;
                if (rp == new_pid && rr == new_rid && nc == 1) dh += 1;
            }
        }

        return dh * 100000.0 + ds;
    }
};
