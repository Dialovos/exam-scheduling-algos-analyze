/*
 * evaluator.h — Full evaluation + O(k) incremental delta evaluation
 *
 * FastEvaluator precomputes flat arrays for cache-friendly access.
 *   full_eval(sol)                  → O(students * avg_exams_per_student)
 *   move_delta(sol, eid, pid, rid)  → O(enrollment * avg_exams_per_student)
 *   apply_move(sol, eid, pid, rid)  → O(1)
 */

#pragma once

#include "models.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>

class FastEvaluator {
public:
    const ProblemInstance& P;
    int ne, np, nr;

    // Flat arrays — indexed by exam/period/room id
    std::vector<int> exam_dur, exam_enroll;
    std::vector<int> period_dur, period_day, period_pen, period_daypos;
    std::vector<int> room_cap, room_pen;

    // Soft constraint weights
    int w_2row, w_2day, w_spread, w_mixed;
    int fl_n_largest, fl_n_last, fl_penalty;

    // Precomputed sets
    std::set<int> large_exams;
    std::set<int> last_periods;
    std::set<int> rhc_exams;

    // Period hard constraints indexed by exam for O(k) delta lookup
    // phc_by_exam[eid] = {(other_eid, type_code)}
    //   type_code: 0 = COINCIDENCE, 1 = EXCLUSION, 2 = AFTER (eid must come after other)
    std::vector<std::vector<std::pair<int, int>>> phc_by_exam;

    explicit FastEvaluator(const ProblemInstance& p) : P(p) {
        ne = p.n_e(); np = p.n_p(); nr = p.n_r();

        // Exam arrays
        exam_dur.resize(ne);
        exam_enroll.resize(ne);
        for (auto& e : p.exams) {
            exam_dur[e.id] = e.duration;
            exam_enroll[e.id] = e.enrollment();
        }

        // Period arrays
        period_dur.resize(np);
        period_day.resize(np);
        period_pen.resize(np);
        period_daypos.resize(np);
        for (auto& pp : p.periods) {
            period_dur[pp.id] = pp.duration;
            period_day[pp.id] = pp.day;
            period_pen[pp.id] = pp.penalty;
            period_daypos[pp.id] = p.period_daypos[pp.id];
        }

        // Room arrays
        room_cap.resize(nr);
        room_pen.resize(nr);
        for (auto& r : p.rooms) {
            room_cap[r.id] = r.capacity;
            room_pen[r.id] = r.penalty;
        }

        // Weights
        w_2row = p.w.two_in_a_row;
        w_2day = p.w.two_in_a_day;
        w_spread = p.w.period_spread;
        w_mixed = p.w.non_mixed_durations;
        fl_n_largest = p.w.fl_n_largest;
        fl_n_last = p.w.fl_n_last;
        fl_penalty = p.w.fl_penalty;

        // Large exams (for front load penalty)
        std::vector<int> sorted_idx(ne);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&](int a, int b) { return exam_enroll[a] > exam_enroll[b]; });
        for (int i = 0; i < std::min(fl_n_largest, ne); i++)
            large_exams.insert(sorted_idx[i]);

        // Last periods (for front load penalty)
        for (int i = std::max(0, np - fl_n_last); i < np; i++)
            last_periods.insert(i);

        // Room exclusive constraints
        for (auto& rc : p.rhcs)
            if (rc.type == "ROOM_EXCLUSIVE")
                rhc_exams.insert(rc.exam_id);

        // Period hard constraints indexed by exam
        phc_by_exam.resize(ne);
        for (auto& c : p.phcs) {
            if (c.exam1 >= ne || c.exam2 >= ne) continue;
            if (c.type == "EXAM_COINCIDENCE")
                phc_by_exam[c.exam1].push_back({c.exam2, 0});
            else if (c.type == "EXCLUSION")
                phc_by_exam[c.exam1].push_back({c.exam2, 1});
            else if (c.type == "AFTER")
                phc_by_exam[c.exam1].push_back({c.exam2, 2});
        }
    }

    // ── Full evaluation ─────────────────────────────────────

    EvalResult full_eval(const Solution& sol) const {
        EvalResult r;
        const auto& po = sol.period_of;
        const auto& ro = sol.room_of;

        // Build per-(period,room) data
        std::map<int64_t, std::vector<int>> pr_exams;
        std::map<int64_t, int> pr_enr;
        for (int e = 0; e < ne; e++) {
            if (po[e] < 0) continue;
            int64_t key = (int64_t)po[e] * nr + ro[e];
            pr_exams[key].push_back(e);
            pr_enr[key] += exam_enroll[e];
        }

        // Per-student analysis: conflicts + proximity
        int max_sid = (int)P.student_exams.size();
        for (int s = 0; s < max_sid; s++) {
            const auto& sexams = P.student_exams[s];
            if (sexams.empty()) continue;

            std::vector<int> pids;
            pids.reserve(sexams.size());
            for (int eid : sexams)
                if (eid < ne && po[eid] >= 0)
                    pids.push_back(po[eid]);
            if (pids.empty()) continue;

            std::sort(pids.begin(), pids.end());

            // Hard: conflicts (duplicate periods)
            for (int i = 1; i < (int)pids.size(); i++)
                if (pids[i] == pids[i - 1])
                    r.conflicts++;

            // Soft: proximity (unique periods only)
            std::vector<int> unique_pids;
            unique_pids.reserve(pids.size());
            for (int p : pids)
                if (unique_pids.empty() || unique_pids.back() != p)
                    unique_pids.push_back(p);

            for (int i = 0; i < (int)unique_pids.size(); i++) {
                int pi = unique_pids[i];
                int di = period_day[pi];
                int posi = period_daypos[pi];
                for (int j = i + 1; j < (int)unique_pids.size(); j++) {
                    int pj = unique_pids[j];
                    int dj = period_day[pj];
                    if (di == dj) {
                        int gap_pos = std::abs(posi - period_daypos[pj]);
                        if (gap_pos == 1)
                            r.two_in_a_row += w_2row;
                        else if (gap_pos > 1)
                            r.two_in_a_day += w_2day;
                    }
                    int gap = std::abs(pj - pi);
                    if (gap > 0 && gap <= w_spread)
                        r.period_spread += 1;
                }
            }
        }

        // Hard: room occupancy
        for (auto& [key, enr] : pr_enr)
            if (enr > room_cap[key % nr])
                r.room_occupancy++;

        // Hard: period utilisation
        for (int e = 0; e < ne; e++)
            if (po[e] >= 0 && exam_dur[e] > period_dur[po[e]])
                r.period_utilisation++;

        // Hard: period constraints
        for (auto& c : P.phcs) {
            if (c.exam1 >= ne || c.exam2 >= ne) continue;
            int p1 = po[c.exam1], p2 = po[c.exam2];
            if (p1 < 0 || p2 < 0) continue;
            if (c.type == "EXAM_COINCIDENCE" && p1 != p2)       r.period_related++;
            else if (c.type == "EXCLUSION" && p1 == p2)         r.period_related++;
            else if (c.type == "AFTER" && p1 <= p2)             r.period_related++;
        }

        // Hard: room exclusive
        for (int eid : rhc_exams) {
            if (eid >= ne) continue;
            int pid = po[eid];
            if (pid < 0) continue;
            int64_t key = (int64_t)pid * nr + ro[eid];
            auto it = pr_exams.find(key);
            if (it != pr_exams.end() && it->second.size() > 1)
                r.room_related++;
        }

        // Soft: mixed durations
        for (auto& [key, eids] : pr_exams) {
            if (eids.size() > 1) {
                std::set<int> durs;
                for (int e : eids) durs.insert(exam_dur[e]);
                if (durs.size() > 1)
                    r.non_mixed_durations += w_mixed;
            }
        }

        // Soft: front load
        if (fl_penalty > 0)
            for (int eid : large_exams)
                if (eid < ne && po[eid] >= 0 && last_periods.count(po[eid]))
                    r.front_load += fl_penalty;

        // Soft: period/room penalties
        for (int e = 0; e < ne; e++) {
            if (po[e] >= 0) {
                r.period_penalty += period_pen[po[e]];
                r.room_penalty += room_pen[ro[e]];
            }
        }

        return r;
    }

    // ── Incremental delta evaluation ────────────────────────

    double move_delta(const Solution& sol, int eid, int new_pid, int new_rid) const {
        const auto& po = sol.period_of;
        int old_pid = po[eid];
        if (old_pid < 0) return 0.0;
        int old_rid = sol.room_of[eid];
        if (old_pid == new_pid && old_rid == new_rid) return 0.0;

        double dh = 0, ds = 0;

        // ── Period duration ──
        int dur = exam_dur[eid];
        if (dur > period_dur[old_pid]) dh -= 1;
        if (dur > period_dur[new_pid]) dh += 1;

        // ── Room occupancy ──
        int enr = exam_enroll[eid];
        int old_total = sol.get_pr_enroll(old_pid, old_rid);
        int new_total = sol.get_pr_enroll(new_pid, new_rid);
        dh -= ((old_total > room_cap[old_rid]) ? 1 : 0) -
              (((old_total - enr) > room_cap[old_rid]) ? 1 : 0);
        dh += (((new_total + enr) > room_cap[new_rid]) ? 1 : 0) -
              ((new_total > room_cap[new_rid]) ? 1 : 0);

        // ── Per-student: conflicts + proximity ──
        int old_day = period_day[old_pid], old_dpos = period_daypos[old_pid];
        int new_day = period_day[new_pid], new_dpos = period_daypos[new_pid];

        for (int s : P.exams[eid].students) {
            for (int other : P.student_exams[s]) {
                if (other == eid) continue;
                int o_pid = po[other];
                if (o_pid < 0) continue;

                // Hard: conflict
                if (o_pid == old_pid) dh -= 1;
                if (o_pid == new_pid) dh += 1;

                // Soft: proximity
                int o_day = period_day[o_pid];
                int o_dpos = period_daypos[o_pid];

                // Remove old proximity
                if (old_day == o_day) {
                    int g = std::abs(old_dpos - o_dpos);
                    if (g == 1) ds -= w_2row;
                    else if (g > 1) ds -= w_2day;
                }
                int og = std::abs(old_pid - o_pid);
                if (og > 0 && og <= w_spread) ds -= 1;

                // Add new proximity
                if (new_day == o_day) {
                    int g = std::abs(new_dpos - o_dpos);
                    if (g == 1) ds += w_2row;
                    else if (g > 1) ds += w_2day;
                }
                int ng = std::abs(new_pid - o_pid);
                if (ng > 0 && ng <= w_spread) ds += 1;
            }
        }

        // ── Period/room penalties ──
        ds += period_pen[new_pid] - period_pen[old_pid];
        ds += room_pen[new_rid] - room_pen[old_rid];

        // ── Front load ──
        if (large_exams.count(eid) && fl_penalty > 0) {
            bool was_late = last_periods.count(old_pid) > 0;
            bool will_late = last_periods.count(new_pid) > 0;
            if (was_late && !will_late)      ds -= fl_penalty;
            else if (!was_late && will_late)  ds += fl_penalty;
        }

        // ── Period hard constraints ──
        for (auto& [other, tcode] : phc_by_exam[eid]) {
            int o_pid = po[other];
            if (o_pid < 0) continue;
            if (tcode == 0) {        // COINCIDENCE
                if (old_pid != o_pid) dh -= 1;
                if (new_pid != o_pid) dh += 1;
            } else if (tcode == 1) { // EXCLUSION
                if (old_pid == o_pid) dh -= 1;
                if (new_pid == o_pid) dh += 1;
            } else if (tcode == 2) { // AFTER: eid must be after other
                if (old_pid <= o_pid) dh -= 1;
                if (new_pid <= o_pid) dh += 1;
            }
        }

        return dh * 100000.0 + ds;
    }

    // ── Apply move in-place ─────────────────────────────────

    void apply_move(Solution& sol, int eid, int new_pid, int new_rid) const {
        sol.assign(eid, new_pid, new_rid);
    }

    // ── Feasibility recovery phase ──────────────────────────
    // Aggressive hard-violation reduction for infeasible initial solutions.
    // Returns true if a feasible solution was found.

    bool recover_feasibility(Solution& sol, int max_rounds = 1000, int seed = 42) const {
        std::mt19937 rng(seed);

        // Random-greedy construction: builds a new solution from scratch
        // avoiding conflicts where possible, using random order
        auto random_greedy_init = [&](int rseed) -> Solution {
            std::mt19937 lrng(rseed);
            Solution ns;
            ns.init(P);
            std::vector<int> order(ne);
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), lrng);

            for (int eid : order) {
                // Find periods where no conflict neighbor is placed
                std::set<int> blocked;
                for (auto& [nb, _] : P.adj[eid])
                    if (ns.period_of[nb] >= 0) blocked.insert(ns.period_of[nb]);
                // Also block exclusion periods
                for (auto& [other, tcode] : phc_by_exam[eid]) {
                    int op = ns.period_of[other]; if (op < 0) continue;
                    if (tcode == 1) blocked.insert(op); // EXCLUSION
                }

                std::vector<int> avail;
                for (int p = 0; p < np; p++)
                    if (!blocked.count(p) && exam_dur[eid] <= period_dur[p])
                        avail.push_back(p);
                if (avail.empty())
                    for (int p = 0; p < np; p++)
                        if (exam_dur[eid] <= period_dur[p]) avail.push_back(p);

                std::shuffle(avail.begin(), avail.end(), lrng);

                bool placed = false;
                for (int pid : avail) {
                    for (int rid = 0; rid < nr; rid++) {
                        if (ns.get_pr_enroll(pid, rid) + exam_enroll[eid] <= room_cap[rid]) {
                            ns.assign(eid, pid, rid);
                            placed = true; break;
                        }
                    }
                    if (placed) break;
                }
                if (!placed) {
                    int pid = avail.empty() ? 0 : avail[0];
                    ns.assign(eid, pid, lrng() % nr);
                }
            }
            return ns;
        };

        auto get_bad = [&]() -> std::vector<int> {
            std::vector<bool> is_bad(ne, false);
            for (int e = 0; e < ne; e++) {
                int p = sol.period_of[e]; if (p < 0) continue;
                for (auto& [nb, _] : P.adj[e])
                    if (sol.period_of[nb] == p) { is_bad[e] = true; is_bad[nb] = true; }
                if (sol.get_pr_enroll(p, sol.room_of[e]) > room_cap[sol.room_of[e]]) is_bad[e] = true;
                if (exam_dur[e] > period_dur[p]) is_bad[e] = true;
            }
            for (auto& c : P.phcs) {
                if (c.exam1 >= ne || c.exam2 >= ne) continue;
                int p1 = sol.period_of[c.exam1], p2 = sol.period_of[c.exam2];
                if (p1 < 0 || p2 < 0) continue;
                bool v = (c.type == "EXAM_COINCIDENCE" && p1 != p2) ||
                         (c.type == "EXCLUSION" && p1 == p2) ||
                         (c.type == "AFTER" && p1 <= p2);
                if (v) { is_bad[c.exam1] = true; is_bad[c.exam2] = true; }
            }
            std::vector<int> bad;
            for (int e = 0; e < ne; e++) if (is_bad[e]) bad.push_back(e);
            return bad;
        };

        double best_fit = full_eval(sol).fitness();
        Solution best_sol = sol.copy();

        for (int rnd = 0; rnd < max_rounds; rnd++) {
            auto bad = get_bad();
            if (bad.empty()) return true;

            std::shuffle(bad.begin(), bad.end(), rng);

            bool improved = false;

            // Phase 1: steepest single moves for each bad exam
            for (int eid : bad) {
                int best_pid = -1, best_rid = -1;
                double best_delta = 0;
                for (int pid = 0; pid < np; pid++) {
                    if (exam_dur[eid] > period_dur[pid]) continue;
                    if (pid == sol.period_of[eid]) continue;
                    for (int rid = 0; rid < nr; rid++) {
                        if (exam_enroll[eid] > room_cap[rid]) continue;
                        double d = move_delta(sol, eid, pid, rid);
                        if (d < best_delta) {
                            best_delta = d; best_pid = pid; best_rid = rid;
                        }
                    }
                }
                if (best_pid >= 0) {
                    apply_move(sol, eid, best_pid, best_rid);
                    improved = true;
                }
            }

            // Phase 2: swap moves for pairs of bad exams
            if (!improved || (rnd % 5 == 0 && !bad.empty())) {
                int swap_limit = std::min((int)bad.size(), 10);
                for (int i = 0; i < swap_limit; i++) {
                    int ea = bad[i];
                    int pa = sol.period_of[ea], ra = sol.room_of[ea];
                    if (pa < 0) continue;

                    // Try swapping with random exams in conflicting periods
                    for (auto& [nb, _] : P.adj[ea]) {
                        if (sol.period_of[nb] != pa) continue; // only conflicting
                        int pb = sol.period_of[nb], rb = sol.room_of[nb];
                        // Try moving ea to nb's period won't work (same period), so find alt periods
                        for (int tp = 0; tp < np; tp++) {
                            if (tp == pa) continue;
                            if (exam_dur[ea] > period_dur[tp]) continue;
                            double d1 = move_delta(sol, ea, tp, ra);
                            if (d1 < 0) {
                                apply_move(sol, ea, tp, ra);
                                improved = true;
                                break;
                            }
                        }
                        if (sol.period_of[ea] != pa) break; // moved successfully
                    }
                }
            }

            double fit = full_eval(sol).fitness();
            if (fit < best_fit) {
                best_fit = fit;
                best_sol = sol.copy();
            }

            if (!improved) {
                // Stronger perturbation: kick random exams
                std::uniform_int_distribution<int> de(0, ne - 1);
                int kicks = std::max(5, ne / 30);
                for (int k = 0; k < kicks; k++) {
                    int e = de(rng);
                    std::vector<int> vp, vr;
                    for (int p = 0; p < np; p++)
                        if (exam_dur[e] <= period_dur[p]) vp.push_back(p);
                    for (int r = 0; r < nr; r++)
                        if (exam_enroll[e] <= room_cap[r]) vr.push_back(r);
                    if (vp.empty() || vr.empty()) continue;
                    apply_move(sol, e, vp[rng() % vp.size()], vr[rng() % vr.size()]);
                }
            }

            // Every 100 rounds, try a random-greedy restart
            if (rnd > 0 && rnd % 100 == 0) {
                Solution rs = random_greedy_init(seed + rnd);
                double rf = full_eval(rs).fitness();
                if (rf < best_fit) {
                    best_fit = rf;
                    best_sol = rs.copy();
                    sol = std::move(rs);
                } else if (full_eval(rs).hard() < full_eval(sol).hard()) {
                    sol = std::move(rs);
                }
            }
        }

        // Restore best found
        sol = best_sol.copy();
        return full_eval(sol).feasible();
    }
};