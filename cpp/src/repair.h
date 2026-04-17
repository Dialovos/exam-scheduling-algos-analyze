/*
 * repair.h — Bounded feasibility restoration and room-level cleanup.
 *
 * One public function, one philosophy:
 *   - Repair::kempe_repair(prob, sol, budget) — given an "almost feasible"
 *     schedule, close the gap to hard=0 inside a small budget. Kempe chains,
 *     targeted hard-only relocations, two-move ejection chains, and a
 *     stagnation-driven perturbation kick. No pretty-soft-penalty work: that
 *     is SA's job. This is the "surgeon, not doctor" routine.
 *
 *   - Repair::fix_room_overflows(prob, sol) — scan for room overflow /
 *     room_exclusive violations and reassign rooms in-place (period stays).
 *     Cheap enough to run in a hot loop after every period move.
 *
 * This file is intentionally a bounded distillation of what feasibility.h
 * does — same tactics, smaller budget, self-contained. Task 6 will retire
 * feasibility.h entirely once every algo warm-starts from Seeder::seed().
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "neighbourhoods.h"

#include <algorithm>
#include <random>
#include <set>
#include <vector>

namespace Repair {

namespace detail {

// ── Violation scan ──────────────────────────────────────────
// Mirror of feasibility_detail::find_violating_exams — kept local so repair.h
// has no cross-file dependency beyond evaluator/neighbourhoods/models.
inline std::vector<int> violating_exams(
    const Solution& sol, const FastEvaluator& fe, const ProblemInstance& prob)
{
    int ne = fe.ne;
    std::vector<char> bad(ne, 0);
    for (int e = 0; e < ne; e++) {
        int p = sol.period_of[e]; if (p < 0) { bad[e] = 1; continue; }
        int r = sol.room_of[e];
        for (auto& [nb, _] : prob.adj[e])
            if (sol.period_of[nb] == p) { bad[e] = 1; bad[nb] = 1; }
        if (sol.get_pr_enroll(p, r) > fe.room_cap[r]) bad[e] = 1;
        if (fe.exam_dur[e] > fe.period_dur[p]) bad[e] = 1;
        if (fe.rhc_exams.count(e) && sol.get_pr_count(p, r) > 1) bad[e] = 1;
    }
    for (auto& c : prob.phcs) {
        if (c.exam1 >= ne || c.exam2 >= ne) continue;
        int p1 = sol.period_of[c.exam1], p2 = sol.period_of[c.exam2];
        if (p1 < 0 || p2 < 0) continue;
        bool v = (c.type == "EXAM_COINCIDENCE" && p1 != p2) ||
                 (c.type == "EXCLUSION" && p1 == p2) ||
                 (c.type == "AFTER" && p1 <= p2);
        if (v) { bad[c.exam1] = 1; bad[c.exam2] = 1; }
    }
    std::vector<int> out;
    out.reserve(ne / 8);
    for (int e = 0; e < ne; e++) if (bad[e]) out.push_back(e);
    return out;
}

// ── Room picker ─────────────────────────────────────────────
// RHC exams strongly prefer empty rooms; otherwise best-fit by slack.
inline int pick_room(const Solution& sol, const FastEvaluator& fe,
                     int eid, int pid, bool is_rhc)
{
    int enr = fe.exam_enroll[eid];
    if (is_rhc) {
        int best_r = -1, best_slack = -1;
        for (int rid = 0; rid < fe.nr; rid++) {
            if (sol.get_pr_count(pid, rid) > 0) continue;
            int slack = fe.room_cap[rid] - sol.get_pr_enroll(pid, rid);
            if (slack >= enr && slack > best_slack) { best_slack = slack; best_r = rid; }
        }
        if (best_r >= 0) return best_r;
    }
    int best_r = -1, best_slack = -1;
    for (int rid = 0; rid < fe.nr; rid++) {
        int slack = fe.room_cap[rid] - sol.get_pr_enroll(pid, rid);
        if (slack >= enr && slack > best_slack) { best_slack = slack; best_r = rid; }
    }
    return best_r;
}

// Count conflict neighbours of eid in period pid — how many room-mates
// we'd be tripping over if we moved there. Cheap scan over adj.
inline int conflicts_in_period(
    const Solution& sol, const ProblemInstance& prob, int eid, int pid)
{
    int c = 0;
    for (auto& [nb, _] : prob.adj[eid])
        if (sol.period_of[nb] == pid) c++;
    return c;
}

} // namespace detail

// ── fix_room_overflows: one-pass room-only cleanup ─────────
// Useful after any move that left a room overflow or rhc violation. Period
// stays put — this only reassigns rooms. Exits silently if no room works.
inline void fix_room_overflows(const ProblemInstance& prob, Solution& sol) {
    FastEvaluator fe(prob);
    int ne = fe.ne;
    std::vector<bool> is_rhc(ne, false);
    for (int e : fe.rhc_exams) if (e < ne) is_rhc[e] = true;

    for (int e = 0; e < ne; e++) {
        int p = sol.period_of[e], r = sol.room_of[e];
        if (p < 0 || r < 0) continue;
        bool overflow = sol.get_pr_enroll(p, r) > fe.room_cap[r];
        bool rhc_viol = is_rhc[e] && sol.get_pr_count(p, r) > 1;
        if (!overflow && !rhc_viol) continue;
        int nr = detail::pick_room(sol, fe, e, p, is_rhc[e]);
        if (nr >= 0 && nr != r) sol.assign(e, p, nr);
    }
}

// ── kempe_repair: bounded, budget-capped feasibility restoration ──
// Strategy (lifted from feasibility.h, narrowed):
//   55% Kempe chain swaps seeded from violating exams
//   45% targeted hard-only single relocation
//   Every `eject_every` iters: RHC repair + two-move ejection chain
//   Stagnation >= stag_limit: heavy perturbation around bad exams, retry
//
// Budget lives in iterations, not passes — the old pass-based loop had no way
// to cap total work. Returns as soon as hard=0 is reached.
inline Solution kempe_repair(
    const ProblemInstance& prob, Solution sol,
    int iters_budget = 8000,
    int restarts = 3,
    uint64_t rng_seed = 42)
{
    FastEvaluator fe(prob);
    int ne = fe.ne, np = fe.np, nr = fe.nr;
    std::mt19937 rng(rng_seed);

    int best_hard = fe.count_hard_fast(sol);
    if (best_hard == 0) return sol;
    Solution best_sol = sol.copy();

    // Precompute valid_p / valid_r / is_rhc for hot loop.
    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++)
            if (fe.exam_dur[e] <= fe.period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++)
            if (fe.exam_enroll[e] <= fe.room_cap[r]) valid_r[e].push_back(r);
    }
    std::vector<bool> is_rhc(ne, false);
    for (int e : fe.rhc_exams) if (e < ne) is_rhc[e] = true;

    std::uniform_int_distribution<int> dp(0, np - 1);
    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    const int STAG_LIMIT = std::max(300, iters_budget / 20);
    const int EJECT_FREQ = 250;

    std::vector<int> bad_cache;
    int cache_age = 999;
    auto refresh_bad = [&]() {
        bad_cache = detail::violating_exams(sol, fe, prob);
        cache_age = 0;
    };

    for (int restart = 0; restart <= restarts && best_hard > 0; restart++) {
        int cur_hard = fe.count_hard_fast(sol);
        int stagnation = 0;
        refresh_bad();

        for (int it = 0; it < iters_budget && best_hard > 0; it++) {
            cache_age++;
            if (cache_age >= 100) refresh_bad();

            if (unif(rng) < 0.55 && !bad_cache.empty()) {
                // ── Kempe chain move ─────────────────────────────
                int eid = bad_cache[rng() % bad_cache.size()];
                int p1 = sol.period_of[eid];
                if (p1 < 0) continue;

                int p2;
                if (unif(rng) < 0.5) {
                    // Look for a low-conflict target — amortises the chain cost.
                    int best_p2 = -1, min_conf = ne;
                    for (int trial = 0; trial < 6; trial++) {
                        int cp = dp(rng);
                        if (cp == p1) continue;
                        int c = detail::conflicts_in_period(sol, prob, eid, cp);
                        if (c < min_conf) { min_conf = c; best_p2 = cp; }
                    }
                    p2 = best_p2 >= 0 ? best_p2 : dp(rng);
                } else {
                    p2 = dp(rng);
                }
                if (p2 == p1) continue;

                auto chain = kempe_detail::build_chain(sol, prob.adj, ne, eid, p1, p2);
                if (chain.empty() || (int)chain.size() > ne / 2) continue;

                auto undo = kempe_detail::apply_chain(sol, chain, p1, p2);
                int new_h = fe.count_hard_fast(sol);

                if (new_h < cur_hard ||
                    (new_h == cur_hard && unif(rng) < 0.20)) {
                    cur_hard = new_h;
                    if (new_h < best_hard) {
                        best_hard = new_h;
                        best_sol = sol.copy();
                        stagnation = 0;
                        refresh_bad();
                        if (best_hard == 0) break;
                    } else {
                        stagnation++;
                    }
                } else {
                    kempe_detail::undo_chain(sol, undo);
                    stagnation++;
                }
            } else {
                // ── Targeted hard-only single relocation ────────
                if (bad_cache.empty()) { stagnation++; continue; }
                int eid = bad_cache[rng() % bad_cache.size()];
                int cur_p = sol.period_of[eid];
                if (cur_p < 0) { stagnation++; continue; }

                int best_tp = -1, best_tr = -1, best_d = 0;
                for (int pid : valid_p[eid]) {
                    if (pid == cur_p) continue;
                    int rid = detail::pick_room(sol, fe, eid, pid, is_rhc[eid]);
                    if (rid < 0) rid = 0;
                    int d = fe.move_delta_hard(sol, eid, pid, rid);
                    if (d < best_d) { best_d = d; best_tp = pid; best_tr = rid; }
                }

                if (best_tp >= 0 && best_d < 0) {
                    fe.apply_move(sol, eid, best_tp, best_tr);
                    cur_hard += best_d;
                    if (cur_hard < best_hard) {
                        best_hard = cur_hard;
                        best_sol = sol.copy();
                        stagnation = 0;
                        refresh_bad();
                        if (best_hard == 0) break;
                    } else {
                        stagnation++;
                    }
                } else {
                    stagnation++;
                }
            }

            // ── Periodic ejection / room-repair sweep ──────────
            if (cur_hard > 0 && cur_hard <= 6 && it > 0 && it % EJECT_FREQ == 0) {
                fix_room_overflows(prob, sol);
                int h = fe.count_hard_fast(sol);
                if (h < best_hard) {
                    best_hard = h; best_sol = sol.copy(); cur_hard = h;
                    stagnation = 0; refresh_bad();
                }
                if (best_hard == 0) break;

                // Two-move ejection chain on ≤5 violations
                if (cur_hard > 0 && cur_hard <= 5) {
                    auto bad = detail::violating_exams(sol, fe, prob);
                    std::shuffle(bad.begin(), bad.end(), rng);
                    for (int eid : bad) {
                        int cp = sol.period_of[eid];
                        if (cp < 0) continue;
                        for (int tp : valid_p[eid]) {
                            if (tp == cp) continue;
                            std::vector<int> blockers;
                            for (auto& [nb, _] : prob.adj[eid])
                                if (sol.period_of[nb] == tp) blockers.push_back(nb);
                            if (blockers.empty() || blockers.size() > 3) continue;

                            // Snapshot + try to evict blockers to neutral periods.
                            std::vector<std::tuple<int,int,int>> snap;
                            for (int b : blockers)
                                snap.emplace_back(b, sol.period_of[b], sol.room_of[b]);

                            bool all_moved = true;
                            for (int b : blockers) {
                                int bp_best = -1, br_best = -1, mc = 999;
                                for (int bp : valid_p[b]) {
                                    if (bp == tp || bp == cp) continue;
                                    int nc = detail::conflicts_in_period(sol, prob, b, bp);
                                    if (nc < mc) {
                                        int br = detail::pick_room(sol, fe, b, bp, is_rhc[b]);
                                        if (br >= 0) { mc = nc; bp_best = bp; br_best = br; }
                                    }
                                }
                                if (bp_best >= 0 && mc == 0)
                                    fe.apply_move(sol, b, bp_best, br_best);
                                else { all_moved = false; break; }
                            }

                            if (all_moved) {
                                int rid = detail::pick_room(sol, fe, eid, tp, is_rhc[eid]);
                                if (rid < 0) rid = 0;
                                fe.apply_move(sol, eid, tp, rid);
                                int nh = fe.count_hard_fast(sol);
                                if (nh < best_hard) {
                                    best_hard = nh; best_sol = sol.copy(); cur_hard = nh;
                                    stagnation = 0; refresh_bad();
                                    if (best_hard == 0) break;
                                }
                                if (nh >= cur_hard) {
                                    // Restore — ejection didn't pan out.
                                    for (auto& [b, bp, br] : snap)
                                        fe.apply_move(sol, b, bp, br);
                                    // Restore eid too (its snapshot wasn't saved; back out manually)
                                    // We know eid was at cp; put it back.
                                    int er = detail::pick_room(sol, fe, eid, cp, is_rhc[eid]);
                                    if (er < 0) er = 0;
                                    fe.apply_move(sol, eid, cp, er);
                                }
                            } else {
                                for (auto& [b, bp, br] : snap)
                                    fe.apply_move(sol, b, bp, br);
                            }
                            if (best_hard == 0) break;
                        }
                        if (best_hard == 0) break;
                    }
                }
            }

            // Cache drift recovery — full recount every 2k iters
            if (it > 0 && it % 2000 == 0) cur_hard = fe.count_hard_fast(sol);

            // ── Perturbation on stagnation ──────────────────────
            if (stagnation >= STAG_LIMIT) {
                refresh_bad();
                int n_perturb = std::max(5, (int)(ne * 0.07));
                std::set<int> pool(bad_cache.begin(), bad_cache.end());
                for (int b : bad_cache) {
                    for (auto& [nb, _] : prob.adj[b]) {
                        pool.insert(nb);
                        if ((int)pool.size() >= n_perturb) break;
                    }
                    if ((int)pool.size() >= n_perturb) break;
                }
                while ((int)pool.size() < n_perturb) pool.insert(de(rng));

                for (int e : pool) {
                    if (valid_p[e].empty() || valid_r[e].empty()) continue;
                    int np2 = valid_p[e][rng() % valid_p[e].size()];
                    int nr2 = valid_r[e][rng() % valid_r[e].size()];
                    fe.apply_move(sol, e, np2, nr2);
                }
                cur_hard = fe.count_hard_fast(sol);
                stagnation = 0;
                refresh_bad();
            }
        }

        if (best_hard == 0) break;

        // Between restarts: resume from best + extra perturbation
        sol = best_sol.copy();
        auto bad = detail::violating_exams(sol, fe, prob);
        int n_perturb = std::max(15, (int)(ne * 0.12));
        std::set<int> pool(bad.begin(), bad.end());
        for (int b : bad)
            for (auto& [nb, _] : prob.adj[b]) pool.insert(nb);
        while ((int)pool.size() < n_perturb) pool.insert(de(rng));
        for (int e : pool) {
            if (valid_p[e].empty() || valid_r[e].empty()) continue;
            int np2 = valid_p[e][rng() % valid_p[e].size()];
            int nr2 = valid_r[e][rng() % valid_r[e].size()];
            fe.apply_move(sol, e, np2, nr2);
        }
    }

    return best_sol;
}

} // namespace Repair
