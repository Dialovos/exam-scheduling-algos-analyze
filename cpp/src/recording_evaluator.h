/*
 * RecordingEvaluator — wraps any Evaluator (FastEvaluator or CachedEvaluator)
 * and records every apply_move into an undo log. On rollback, iterates the
 * log in reverse and applies the inverse moves through the underlying
 * evaluator, keeping its cache (if any) coherent without a full rebuild.
 *
 * Purpose: eliminate the O(ne × np × deg) `Ecach.initialize(sol)` cost on
 * VNS reject paths. Same fix pattern works for any LS that needs "try a
 * sequence of moves, possibly undo the batch."
 *
 * API mirror: provides move_delta, apply_move, and the public member
 * aliases so RecordingEvaluator substitutes for the base Ev in any
 * templated nbhd call.
 */

#pragma once

#include "models.h"
#include "evaluator.h"

#include <cstdint>
#include <vector>

template <typename BaseEv>
class RecordingEvaluator {
public:
    const BaseEv& base;
    struct Undo { int eid, old_pid, old_rid; };
    mutable std::vector<Undo> log;

    // Public member mirrors for drop-in substitutability
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
    const int ne, np, nr;
    const ProblemInstance& P;

    explicit RecordingEvaluator(const BaseEv& b)
        : base(b),
          exam_dur(b.exam_dur), exam_enroll(b.exam_enroll),
          period_dur(b.period_dur), period_day(b.period_day),
          period_pen(b.period_pen), period_daypos(b.period_daypos),
          room_cap(b.room_cap), room_pen(b.room_pen),
          large_exams(b.large_exams), last_periods(b.last_periods),
          rhc_exams(b.rhc_exams), phc_by_exam(b.phc_by_exam),
          w_2row(b.w_2row), w_2day(b.w_2day), w_spread(b.w_spread),
          ne(b.ne), np(b.np), nr(b.nr), P(b.P)
    {}

    double move_delta(const Solution& sol, int eid, int new_pid, int new_rid) const {
        return base.move_delta(sol, eid, new_pid, new_rid);
    }

    // Intercept: log the pre-state before delegating.
    void apply_move(Solution& sol, int eid, int new_pid, int new_rid) const {
        log.push_back({eid, sol.period_of[eid], sol.room_of[eid]});
        base.apply_move(sol, eid, new_pid, new_rid);
    }

    // Undo every logged move in reverse, through base's apply_move
    // (so base's cache stays coherent without full rebuild).
    void rollback_all(Solution& sol) const {
        for (auto it = log.rbegin(); it != log.rend(); ++it)
            base.apply_move(sol, it->eid, it->old_pid, it->old_rid);
        log.clear();
    }

    // Accept and clear the log — state stays as-is, log is discarded.
    void commit() const { log.clear(); }

    // Append a chain-undo sequence to the log (used by kempe_chain on accept
    // so a later outer-level reject can rollback through Ecach.apply_move).
    // Duck-typed on any struct with .eid, .old_pid, .old_rid fields.
    template <typename ChainT>
    void append_chain_undo(const ChainT& undo) const {
        for (auto& u : undo) log.push_back({u.eid, u.old_pid, u.old_rid});
    }

    // Forwarded utility methods
    EvalResult full_eval(const Solution& sol) const { return base.full_eval(sol); }
    EvalResult partial_eval(const Solution& sol, const std::vector<int>& a) const {
        return base.partial_eval(sol, a);
    }
    int count_hard_fast(const Solution& sol) const { return base.count_hard_fast(sol); }
    int move_delta_hard(const Solution& sol, int eid, int p, int r) const {
        return base.move_delta_hard(sol, eid, p, r);
    }

    // For CachedEvaluator as base: allow bulk-mutation refresh hook visibility
    template <typename B = BaseEv>
    auto rebuild_contrib_for(int e, const Solution& sol) const
        -> decltype(std::declval<const B&>().rebuild_contrib_for(e, sol), void())
    {
        base.rebuild_contrib_for(e, sol);
    }
};
