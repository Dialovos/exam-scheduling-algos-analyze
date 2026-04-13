/*
 * Targeted feasibility solver for hard instances (e.g., ITC 2007 set3) where
 * greedy produces infeasible solutions and metaheuristics stall at hard > 0.
 *
 * Three phases:
 *   1. Hard-only Kempe + targeted relocations (pure hard-count objective,
 *      no soft penalty — all effort on feasibility)
 *   2. Ejection chains for last 1-5 violations (multi-move coordination)
 *   3. Multi-restart with strategic perturbation
 *
 * move_delta_hard() via adj list + pr_count: O(degree + |rhc|), handles
 * room_exclusive natively. Room selection prefers empty rooms for RHC exams.
 * Dedicated rhc_repair pass for stubborn room_exclusive violations.
 *
 * Called automatically when greedy is infeasible; output feeds downstream algos.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"
#include "neighbourhoods.h"
#include "alns.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <set>
#include <vector>

namespace feasibility_detail {

// ── Find exams involved in any hard violation ──────────���────
inline std::vector<int> find_violating_exams(
    const Solution& sol, const FastEvaluator& fe, const ProblemInstance& prob)
{
    int ne = fe.ne, nr = fe.nr;
    std::vector<bool> bad(ne, false);

    for (int e = 0; e < ne; e++) {
        int p = sol.period_of[e]; if (p < 0) continue;
        for (auto& [nb, _] : prob.adj[e])
            if (sol.period_of[nb] == p) { bad[e] = true; bad[nb] = true; }
        if (sol.get_pr_enroll(p, sol.room_of[e]) > fe.room_cap[sol.room_of[e]])
            bad[e] = true;
        if (fe.exam_dur[e] > fe.period_dur[p])
            bad[e] = true;
        if (fe.rhc_exams.count(e) && sol.get_pr_count(p, sol.room_of[e]) > 1)
            bad[e] = true;
    }
    for (auto& c : prob.phcs) {
        if (c.exam1 >= ne || c.exam2 >= ne) continue;
        int p1 = sol.period_of[c.exam1], p2 = sol.period_of[c.exam2];
        if (p1 < 0 || p2 < 0) continue;
        bool v = (c.type == "EXAM_COINCIDENCE" && p1 != p2) ||
                 (c.type == "EXCLUSION" && p1 == p2) ||
                 (c.type == "AFTER" && p1 <= p2);
        if (v) { bad[c.exam1] = true; bad[c.exam2] = true; }
    }

    std::vector<int> result;
    for (int e = 0; e < ne; e++) if (bad[e]) result.push_back(e);
    return result;
}

// ── Count conflict neighbors of exam in a specific period ───
inline int conflicts_in_period(
    const Solution& sol, const ProblemInstance& prob, int eid, int pid)
{
    int c = 0;
    for (auto& [nb, _] : prob.adj[eid])
        if (sol.period_of[nb] == pid) c++;
    return c;
}

// ── Find best room for exam, respecting room_exclusive ──────
inline int find_room(const Solution& sol, const FastEvaluator& fe,
                     const std::vector<int>& vr, int eid, int pid, bool is_rhc)
{
    // For rhc exams: strongly prefer empty rooms
    if (is_rhc) {
        int best_r = -1, best_slack = -1;
        for (int rid : vr) {
            if (sol.get_pr_count(pid, rid) > 0) continue;
            int slack = fe.room_cap[rid] - sol.get_pr_enroll(pid, rid);
            if (slack >= fe.exam_enroll[eid] && slack > best_slack) {
                best_slack = slack; best_r = rid;
            }
        }
        if (best_r >= 0) return best_r;
    }

    // Standard: pick room with most slack that fits
    int best_r = -1, best_slack = -1;
    for (int rid : vr) {
        int slack = fe.room_cap[rid] - sol.get_pr_enroll(pid, rid);
        if (slack >= fe.exam_enroll[eid] && slack > best_slack) {
            best_slack = slack; best_r = rid;
        }
    }
    return (best_r >= 0) ? best_r : (vr.empty() ? 0 : vr.back());
}

// ── Room-exclusive repair pass ──────────────────────────────
inline int rhc_repair(
    Solution& sol, FastEvaluator& fe, const ProblemInstance& prob,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    bool verbose)
{
    int ne = fe.ne, nr = fe.nr;
    int fixed = 0;

    for (int round = 0; round < 30; round++) {
        bool progress = false;

        for (int eid : fe.rhc_exams) {
            if (eid >= ne || sol.period_of[eid] < 0) continue;
            int pid = sol.period_of[eid], rid = sol.room_of[eid];
            if (sol.get_pr_count(pid, rid) <= 1) continue;

            // Strategy 1: empty room in same period
            for (int r : valid_r[eid]) {
                if (r == rid) continue;
                if (sol.get_pr_count(pid, r) == 0 &&
                    fe.room_cap[r] >= fe.exam_enroll[eid]) {
                    fe.apply_move(sol, eid, pid, r);
                    fixed++; progress = true;
                    if (verbose)
                        std::cerr << "[Feasibility] RHC fix: exam " << eid
                                  << " room " << rid << "->" << r << "\n";
                    goto next_rhc;
                }
            }

            // Strategy 2: different period with empty room, no new conflicts
            for (int p : valid_p[eid]) {
                if (p == pid) continue;
                if (conflicts_in_period(sol, prob, eid, p) > 0) continue;
                // Check period hard constraints
                {
                    int phc_delta = 0;
                    for (auto& [other, tc] : fe.phc_by_exam[eid]) {
                        int op = sol.period_of[other]; if (op < 0) continue;
                        if (tc == 0) { if (pid != op) phc_delta--; if (p != op) phc_delta++; }
                        else if (tc == 1) { if (pid == op) phc_delta--; if (p == op) phc_delta++; }
                        else if (tc == 2) { if (pid <= op) phc_delta--; if (p <= op) phc_delta++; }
                    }
                    if (phc_delta > 0) continue;
                }
                for (int r : valid_r[eid]) {
                    if (sol.get_pr_count(p, r) == 0 &&
                        fe.room_cap[r] >= fe.exam_enroll[eid]) {
                        fe.apply_move(sol, eid, p, r);
                        fixed++; progress = true;
                        if (verbose)
                            std::cerr << "[Feasibility] RHC fix: exam " << eid
                                      << " -> period " << p << " room " << r << "\n";
                        goto next_rhc;
                    }
                }
            }

            // Strategy 3: move non-rhc roommates to other rooms in same period
            {
                std::vector<int> roommates;
                for (int e2 = 0; e2 < ne; e2++) {
                    if (e2 == eid || sol.period_of[e2] != pid || sol.room_of[e2] != rid)
                        continue;
                    if (fe.rhc_exams.count(e2)) continue;
                    roommates.push_back(e2);
                }
                for (int rm : roommates) {
                    int best_r2 = -1, best_s2 = -1;
                    for (int r2 : valid_r[rm]) {
                        if (r2 == rid) continue;
                        int s2 = fe.room_cap[r2] - sol.get_pr_enroll(pid, r2);
                        if (s2 >= fe.exam_enroll[rm] && s2 > best_s2) {
                            best_s2 = s2; best_r2 = r2;
                        }
                    }
                    if (best_r2 >= 0) {
                        fe.apply_move(sol, rm, pid, best_r2);
                        fixed++;
                        if (sol.get_pr_count(pid, rid) <= 1) {
                            progress = true;
                            if (verbose)
                                std::cerr << "[Feasibility] RHC fix: moved roommate "
                                          << rm << " to room " << best_r2 << "\n";
                            goto next_rhc;
                        }
                    }
                }
            }
            next_rhc:;
        }

        if (!progress) break;
    }
    return fixed;
}

// ── Phase 2: Ejection chain repair ──────────────────────────
inline bool ejection_repair(
    Solution& sol, FastEvaluator& fe, const ProblemInstance& prob,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    const std::vector<bool>& is_rhc,
    std::mt19937& rng, int max_rounds, bool verbose)
{
    int ne = fe.ne, np = fe.np;

    for (int round = 0; round < max_rounds; round++) {
        int hard = fe.count_hard_fast(sol);
        if (hard == 0) return true;

        auto bad = find_violating_exams(sol, fe, prob);
        if (bad.empty()) return true;
        std::shuffle(bad.begin(), bad.end(), rng);

        bool improved = false;

        // ── Level 1: steepest single relocation ──
        for (int eid : bad) {
            int cur_p = sol.period_of[eid];
            int best_pid = -1, best_rid = -1;
            int best_delta = 0;

            for (int pid : valid_p[eid]) {
                if (pid == cur_p) continue;
                int rid = find_room(sol, fe, valid_r[eid], eid, pid, is_rhc[eid]);
                int d = fe.move_delta_hard(sol, eid, pid, rid);
                if (d < best_delta) {
                    best_delta = d; best_pid = pid; best_rid = rid;
                }
            }

            if (best_pid >= 0 && best_delta < 0) {
                fe.apply_move(sol, eid, best_pid, best_rid);
                improved = true;
            }
        }

        hard = fe.count_hard_fast(sol);
        if (hard == 0) return true;
        if (improved) continue;

        // ── Level 2: two-move ejection chains ──
        bad = find_violating_exams(sol, fe, prob);
        std::shuffle(bad.begin(), bad.end(), rng);

        for (int eid : bad) {
            int cur_p = sol.period_of[eid];
            if (cur_p < 0) continue;

            for (int tp : valid_p[eid]) {
                if (tp == cur_p) continue;

                std::vector<int> blockers;
                for (auto& [nb, _] : prob.adj[eid])
                    if (sol.period_of[nb] == tp) blockers.push_back(nb);

                if (blockers.empty() || blockers.size() > 4) continue;

                auto saved = alns_detail::save_state(sol);
                bool all_moved = true;

                for (int b : blockers) {
                    int best_bp = -1, best_br = -1;
                    int min_nc = 999;

                    for (int bp : valid_p[b]) {
                        if (bp == tp || bp == cur_p) continue;
                        int nc = conflicts_in_period(sol, prob, b, bp);
                        if (nc < min_nc) {
                            int br = find_room(sol, fe, valid_r[b], b, bp, is_rhc[b]);
                            min_nc = nc; best_bp = bp; best_br = br;
                        }
                    }

                    if (best_bp >= 0 && min_nc == 0) {
                        fe.apply_move(sol, b, best_bp, best_br);
                    } else {
                        all_moved = false; break;
                    }
                }

                if (all_moved) {
                    int rid = find_room(sol, fe, valid_r[eid], eid, tp, is_rhc[eid]);
                    fe.apply_move(sol, eid, tp, rid);
                    int new_hard = fe.count_hard_fast(sol);

                    if (new_hard < hard) {
                        if (verbose)
                            std::cerr << "[Feasibility] L2 ejection: exam " << eid
                                      << " -> period " << tp << " (moved "
                                      << blockers.size() << " blockers) hard="
                                      << new_hard << "\n";
                        improved = true;
                        break;
                    }
                }

                alns_detail::restore_state(sol, saved);
            }

            if (improved) break;
        }

        if (!improved) break;
    }

    return fe.count_hard_fast(sol) == 0;
}

} // namespace feasibility_detail


// ── Public interface ────────────────────────────────────────

inline AlgoResult solve_feasibility(
    const ProblemInstance& prob,
    int seed     = 42,
    bool verbose = false,
    const Solution* init_sol = nullptr)
{
    using namespace feasibility_detail;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    FastEvaluator fe(prob);

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++)
            if (fe.exam_dur[e] <= fe.period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++)
            if (fe.exam_enroll[e] <= fe.room_cap[r]) valid_r[e].push_back(r);
    }

    std::vector<bool> is_rhc(ne, false);
    for (int e : fe.rhc_exams) if (e < ne) is_rhc[e] = true;

    // ── Init solution ��─
    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); sol = g.sol.copy(); }

    int best_hard = fe.count_hard_fast(sol);
    Solution best_sol = sol.copy();

    if (best_hard == 0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        return {std::move(best_sol), fe.full_eval(best_sol),
                std::chrono::duration<double>(t1 - t0).count(), 0, "Feasibility"};
    }

    if (verbose)
        std::cerr << "[Feasibility] Init hard=" << best_hard << std::endl;

    std::uniform_int_distribution<int> de(0, ne - 1);
    std::uniform_int_distribution<int> dp(0, np - 1);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    constexpr int KEMPE_ITERS   = 30000;
    constexpr int MAX_RESTARTS  = 15;
    constexpr int STAG_LIMIT    = 2000;
    constexpr int EJECTION_FREQ = 400;

    int total_iters = 0;

    std::vector<int> bad_cache;
    int cache_age = 999;
    auto refresh_bad = [&]() {
        bad_cache = find_violating_exams(sol, fe, prob);
        cache_age = 0;
    };

    for (int restart = 0; restart < MAX_RESTARTS && best_hard > 0; restart++) {
        int current_hard = fe.count_hard_fast(sol);
        int stagnation = 0;
        refresh_bad();

        for (int it = 0; it < KEMPE_ITERS && best_hard > 0; it++) {
            total_iters++;
            cache_age++;
            if (cache_age >= 150) refresh_bad();

            if (unif(rng) < 0.55 && !bad_cache.empty()) {
                // ── Kempe chain ──
                int eid = (unif(rng) < 0.85 && !bad_cache.empty())
                    ? bad_cache[rng() % bad_cache.size()]
                    : de(rng);
                int p1 = sol.period_of[eid];
                if (p1 < 0) continue;

                int p2;
                if (unif(rng) < 0.5) {
                    int best_p2 = -1, min_conf = ne;
                    for (int trial = 0; trial < 8; trial++) {
                        int cp = dp(rng);
                        if (cp == p1) continue;
                        int c = conflicts_in_period(sol, prob, eid, cp);
                        if (c < min_conf) { min_conf = c; best_p2 = cp; }
                    }
                    p2 = (best_p2 >= 0) ? best_p2 : dp(rng);
                } else {
                    p2 = dp(rng);
                }
                if (p2 == p1) continue;

                auto chain = kempe_detail::build_chain(sol, prob.adj, ne, eid, p1, p2);
                if (chain.empty() || (int)chain.size() > ne / 2) continue;

                int old_h = current_hard;
                auto undo = kempe_detail::apply_chain(sol, chain, p1, p2);
                int new_h = fe.count_hard_fast(sol);

                if (new_h < old_h || (new_h == old_h && unif(rng) < 0.20)) {
                    current_hard = new_h;
                    if (new_h < best_hard) {
                        best_hard = new_h;
                        best_sol = sol.copy();
                        stagnation = 0;
                        refresh_bad();
                        if (verbose)
                            std::cerr << "[Feasibility] Kempe iter " << total_iters
                                      << " (r" << restart << "): hard=" << best_hard
                                      << " chain=" << chain.size() << std::endl;
                        if (best_hard == 0) break;
                    } else {
                        stagnation++;
                    }
                } else {
                    kempe_detail::undo_chain(sol, undo);
                    stagnation++;
                }
            } else {
                // ── Targeted single relocation with hard-only delta ──
                if (bad_cache.empty()) { stagnation++; continue; }
                int eid = bad_cache[rng() % bad_cache.size()];
                int cur_p = sol.period_of[eid];
                if (cur_p < 0) { stagnation++; continue; }

                int best_tp = -1, best_tr = -1;
                int best_delta = 0;

                for (int pid : valid_p[eid]) {
                    if (pid == cur_p) continue;
                    int rid = find_room(sol, fe, valid_r[eid], eid, pid, is_rhc[eid]);
                    int d = fe.move_delta_hard(sol, eid, pid, rid);
                    if (d < best_delta) {
                        best_delta = d; best_tp = pid; best_tr = rid;
                    }
                }

                if (best_tp >= 0 && best_delta < 0) {
                    fe.apply_move(sol, eid, best_tp, best_tr);
                    current_hard += best_delta;
                    if (current_hard < best_hard) {
                        best_hard = current_hard;
                        best_sol = sol.copy();
                        stagnation = 0;
                        refresh_bad();
                        if (verbose)
                            std::cerr << "[Feasibility] Reloc iter " << total_iters
                                      << " (r" << restart << "): hard=" << best_hard
                                      << std::endl;
                        if (best_hard == 0) break;
                    } else {
                        stagnation++;
                    }
                } else if (best_tp >= 0 && best_delta == 0 && unif(rng) < 0.15) {
                    fe.apply_move(sol, eid, best_tp, best_tr);
                    stagnation++;
                } else {
                    stagnation++;
                }
            }

            // ── RHC repair + ejection when hard is low ──
            if (current_hard > 0 && current_hard <= 8 &&
                it > 0 && it % EJECTION_FREQ == 0) {
                int rf = rhc_repair(sol, fe, prob, valid_p, valid_r, verbose);
                if (rf > 0) {
                    current_hard = fe.count_hard_fast(sol);
                    if (current_hard < best_hard) {
                        best_hard = current_hard;
                        best_sol = sol.copy();
                        stagnation = 0;
                        refresh_bad();
                        if (verbose)
                            std::cerr << "[Feasibility] RHC iter " << total_iters
                                      << " (r" << restart << "): hard=" << best_hard
                                      << std::endl;
                    }
                    if (best_hard == 0) break;
                }

                if (current_hard > 0 && current_hard <= 5) {
                    if (ejection_repair(sol, fe, prob, valid_p, valid_r, is_rhc, rng, 20, verbose)) {
                        best_hard = 0;
                        best_sol = sol.copy();
                        break;
                    }
                    int h = fe.count_hard_fast(sol);
                    if (h < best_hard) {
                        best_hard = h;
                        best_sol = sol.copy();
                        current_hard = h;
                        refresh_bad();
                    }
                }
            }

            // Periodic verification
            if (it > 0 && it % 2000 == 0)
                current_hard = fe.count_hard_fast(sol);

            // ── Perturb when stuck ──
            if (stagnation >= STAG_LIMIT) {
                refresh_bad();
                int n_perturb = std::max(5, (int)(ne * 0.07));
                std::set<int> to_perturb(bad_cache.begin(), bad_cache.end());
                for (int b : bad_cache) {
                    for (auto& [nb, _] : prob.adj[b]) {
                        to_perturb.insert(nb);
                        if ((int)to_perturb.size() >= n_perturb) break;
                    }
                    if ((int)to_perturb.size() >= n_perturb) break;
                }
                while ((int)to_perturb.size() < n_perturb)
                    to_perturb.insert(de(rng));

                for (int e : to_perturb) {
                    if (valid_p[e].empty() || valid_r[e].empty()) continue;
                    int np2 = valid_p[e][rng() % valid_p[e].size()];
                    int nr2 = valid_r[e][rng() % valid_r[e].size()];
                    fe.apply_move(sol, e, np2, nr2);
                }
                current_hard = fe.count_hard_fast(sol);
                stagnation = 0;
                refresh_bad();
            }
        }

        if (best_hard == 0) break;

        // ── Post-loop: RHC repair + ejection on best solution ──
        sol = best_sol.copy();
        rhc_repair(sol, fe, prob, valid_p, valid_r, verbose);
        int h = fe.count_hard_fast(sol);
        if (h < best_hard) {
            best_hard = h;
            best_sol = sol.copy();
            if (verbose)
                std::cerr << "[Feasibility] Post-RHC (r" << restart
                          << "): hard=" << best_hard << std::endl;
        }
        if (best_hard == 0) break;

        if (best_hard <= 5) {
            if (ejection_repair(sol, fe, prob, valid_p, valid_r, is_rhc, rng, 50, verbose)) {
                best_hard = 0;
                best_sol = sol.copy();
                break;
            }
            h = fe.count_hard_fast(sol);
            if (h < best_hard) {
                best_hard = h;
                best_sol = sol.copy();
            }
        }

        // ── Phase 3: Restart from best + heavy perturbation ──
        sol = best_sol.copy();
        auto bad = find_violating_exams(sol, fe, prob);
        int n_perturb = std::max(15, (int)(ne * 0.12));
        std::set<int> to_perturb(bad.begin(), bad.end());
        for (int b : bad)
            for (auto& [nb, _] : prob.adj[b])
                to_perturb.insert(nb);
        while ((int)to_perturb.size() < n_perturb)
            to_perturb.insert(de(rng));

        for (int e : to_perturb) {
            if (valid_p[e].empty() || valid_r[e].empty()) continue;
            int np2 = valid_p[e][rng() % valid_p[e].size()];
            int nr2 = valid_r[e][rng() % valid_r[e].size()];
            fe.apply_move(sol, e, np2, nr2);
        }

        if (verbose)
            std::cerr << "[Feasibility] Restart " << (restart + 1)
                      << ": best_hard=" << best_hard
                      << " perturbed=" << to_perturb.size() << std::endl;
    }

    // Skip optimize_rooms — it can break room_exclusive on some datasets.
    // Downstream algorithms handle room optimization themselves.

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[Feasibility] " << rt << "s  iters=" << total_iters
                  << " feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, total_iters, "Feasibility"};
}
