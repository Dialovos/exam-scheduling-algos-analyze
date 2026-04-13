/*
 * Full evaluation and O(k) incremental delta evaluation.
 * FastEvaluator precomputes flat arrays for cache-friendly access.
 *   full_eval(sol)                  → O(students * avg_exams_per_student)
 *   move_delta(sol, eid, pid, rid)  → O(enrollment * avg_exams_per_student)
 *   apply_move(sol, eid, pid, rid)  → O(1)
 */

#pragma once

#include "models.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ── Walker's Alias Table: O(n) build, O(1) sample ──────────
struct AliasTable {
    int n = 0;
    std::vector<double> prob;
    std::vector<int> alias;

    void build(const std::vector<double>& weights) {
        n = (int)weights.size();
        prob.resize(n); alias.resize(n);
        double total = 0;
        for (double w : weights) total += w;
        if (n == 0 || total <= 0) {
            for (int i = 0; i < n; i++) { prob[i] = 1.0; alias[i] = i; }
            return;
        }
        std::vector<double> sc(n);
        for (int i = 0; i < n; i++) sc[i] = weights[i] * n / total;
        std::vector<int> sm, lg;
        for (int i = 0; i < n; i++) {
            if (sc[i] < 1.0) sm.push_back(i); else lg.push_back(i);
        }
        while (!sm.empty() && !lg.empty()) {
            int s = sm.back(); sm.pop_back();
            int l = lg.back(); lg.pop_back();
            prob[s] = sc[s]; alias[s] = l;
            sc[l] -= (1.0 - sc[s]);
            if (sc[l] < 1.0) sm.push_back(l); else lg.push_back(l);
        }
        while (!lg.empty()) { int l = lg.back(); lg.pop_back(); prob[l] = 1.0; alias[l] = l; }
        while (!sm.empty()) { int s = sm.back(); sm.pop_back(); prob[s] = 1.0; alias[s] = s; }
    }

    int sample(std::mt19937& rng) const {
        int i = std::uniform_int_distribution<int>(0, n - 1)(rng);
        return (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < prob[i]) ? i : alias[i];
    }
};

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

    // Precomputed sets (unordered for O(1) lookup in hot paths)
    std::unordered_set<int> large_exams;
    std::unordered_set<int> last_periods;
    std::unordered_set<int> rhc_exams;

    // Period hard constraints indexed by exam for O(k) delta lookup
    // phc_by_exam[eid] = {(other_eid, type_code)}
    //   type_code: 0 = COINCIDENCE, 1 = EXCLUSION, 2 = AFTER (eid must come after other),
    //              3 = BEFORE (eid must come before other — reverse of AFTER)
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

        // Period hard constraints indexed by exam — bidirectional for
        // symmetric constraints so move_delta sees them from both sides.
        phc_by_exam.resize(ne);
        for (auto& c : p.phcs) {
            if (c.exam1 >= ne || c.exam2 >= ne) continue;
            if (c.type == "EXAM_COINCIDENCE") {
                phc_by_exam[c.exam1].push_back({c.exam2, 0});
                phc_by_exam[c.exam2].push_back({c.exam1, 0});  // symmetric
            } else if (c.type == "EXCLUSION") {
                phc_by_exam[c.exam1].push_back({c.exam2, 1});
                phc_by_exam[c.exam2].push_back({c.exam1, 1});  // symmetric
            } else if (c.type == "AFTER") {
                phc_by_exam[c.exam1].push_back({c.exam2, 2});  // exam1 must be AFTER exam2
                phc_by_exam[c.exam2].push_back({c.exam1, 3});  // exam2 must be BEFORE exam1
            }
        }
    }

    // ── Full evaluation ─────────────────────────────────────

    EvalResult full_eval(const Solution& sol) const {
        EvalResult r;
        const auto& po = sol.period_of;
        const auto& ro = sol.room_of;

        // Build per-(period,room) data
        std::unordered_map<int64_t, std::vector<int>> pr_exams;
        std::unordered_map<int64_t, int> pr_enr;
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

    // ── Partial evaluation (for chain-swap delta) ─────────
    // Same as full_eval but student loop only covers students
    // of affected_exams.  Non-student parts computed fully (cheap).
    // Use ONLY for delta: new_partial.fitness() - old_partial.fitness()
    // gives the exact fitness change of a multi-exam swap.

    EvalResult partial_eval(const Solution& sol,
                            const std::vector<int>& affected_exams) const {
        EvalResult r;
        const auto& po = sol.period_of;
        const auto& ro = sol.room_of;

        // Build per-(period,room) data — full, O(ne)
        std::unordered_map<int64_t, std::vector<int>> pr_exams;
        for (int e = 0; e < ne; e++) {
            if (po[e] < 0) continue;
            int64_t key = (int64_t)po[e] * nr + ro[e];
            pr_exams[key].push_back(e);
        }

        // Student loop — restricted to students of affected exams
        std::unordered_set<int> students;
        for (int e : affected_exams)
            for (int s : P.exams[e].students)
                students.insert(s);

        for (int s : students) {
            const auto& sexams = P.student_exams[s];
            if (sexams.empty()) continue;
            std::vector<int> pids;
            pids.reserve(sexams.size());
            for (int eid : sexams)
                if (eid < ne && po[eid] >= 0)
                    pids.push_back(po[eid]);
            if (pids.empty()) continue;
            std::sort(pids.begin(), pids.end());
            for (int i = 1; i < (int)pids.size(); i++)
                if (pids[i] == pids[i - 1]) r.conflicts++;
            std::vector<int> unique_pids;
            unique_pids.reserve(pids.size());
            for (int p : pids)
                if (unique_pids.empty() || unique_pids.back() != p)
                    unique_pids.push_back(p);
            for (int i = 0; i < (int)unique_pids.size(); i++) {
                int pi = unique_pids[i];
                int di = period_day[pi], posi = period_daypos[pi];
                for (int j = i + 1; j < (int)unique_pids.size(); j++) {
                    int pj = unique_pids[j];
                    if (di == period_day[pj]) {
                        int g = std::abs(posi - period_daypos[pj]);
                        if (g == 1)      r.two_in_a_row += w_2row;
                        else if (g > 1)  r.two_in_a_day += w_2day;
                    }
                    int gap = std::abs(pj - pi);
                    if (gap > 0 && gap <= w_spread) r.period_spread += 1;
                }
            }
        }

        // ── Non-student parts (full, same as full_eval) ──

        for (int p = 0; p < np; p++)
            for (int rr = 0; rr < nr; rr++)
                if (sol.get_pr_enroll(p, rr) > room_cap[rr])
                    r.room_occupancy++;

        for (int e = 0; e < ne; e++)
            if (po[e] >= 0 && exam_dur[e] > period_dur[po[e]])
                r.period_utilisation++;

        for (auto& c : P.phcs) {
            if (c.exam1 >= ne || c.exam2 >= ne) continue;
            int p1 = po[c.exam1], p2 = po[c.exam2];
            if (p1 < 0 || p2 < 0) continue;
            if (c.type == "EXAM_COINCIDENCE" && p1 != p2)       r.period_related++;
            else if (c.type == "EXCLUSION" && p1 == p2)         r.period_related++;
            else if (c.type == "AFTER" && p1 <= p2)             r.period_related++;
        }

        for (int eid : rhc_exams) {
            if (eid >= ne) continue;
            int pid = po[eid]; if (pid < 0) continue;
            int64_t key = (int64_t)pid * nr + ro[eid];
            auto it = pr_exams.find(key);
            if (it != pr_exams.end() && it->second.size() > 1)
                r.room_related++;
        }

        for (auto& [key, eids] : pr_exams) {
            if (eids.size() > 1) {
                std::set<int> durs;
                for (int e : eids) durs.insert(exam_dur[e]);
                if (durs.size() > 1) r.non_mixed_durations += w_mixed;
            }
        }

        if (fl_penalty > 0)
            for (int eid : large_exams)
                if (eid < ne && po[eid] >= 0 && last_periods.count(po[eid]))
                    r.front_load += fl_penalty;

        for (int e = 0; e < ne; e++) {
            if (po[e] >= 0) {
                r.period_penalty += period_pen[po[e]];
                r.room_penalty += room_pen[ro[e]];
            }
        }
        return r;
    }

    // ── Fast hard-violation count (no student loop) ─────────
    // Uses adjacency list for conflicts instead of iterating all students.
    // O(edges + np*nr + ne + |phcs|) — much faster than full_eval.

    int count_hard_fast(const Solution& sol) const {
        int hard = 0;
        // Conflicts via adjacency (nb > e avoids double-counting)
        for (int e = 0; e < ne; e++) {
            int p = sol.period_of[e]; if (p < 0) continue;
            for (auto& [nb, _] : P.adj[e])
                if (nb > e && sol.period_of[nb] == p) hard++;
        }
        // Room occupancy via flat pr_enroll
        for (int p = 0; p < np; p++)
            for (int r = 0; r < nr; r++)
                if (sol.get_pr_enroll(p, r) > room_cap[r]) hard++;
        // Period utilisation
        for (int e = 0; e < ne; e++)
            if (sol.period_of[e] >= 0 && exam_dur[e] > period_dur[sol.period_of[e]]) hard++;
        // Period constraints
        for (auto& c : P.phcs) {
            if (c.exam1 >= ne || c.exam2 >= ne) continue;
            int p1 = sol.period_of[c.exam1], p2 = sol.period_of[c.exam2];
            if (p1 < 0 || p2 < 0) continue;
            if (c.type == "EXAM_COINCIDENCE" && p1 != p2)   hard++;
            else if (c.type == "EXCLUSION" && p1 == p2)     hard++;
            else if (c.type == "AFTER" && p1 <= p2)         hard++;
        }
        // Room exclusive — use maintained pr_count from Solution
        for (int eid : rhc_exams) {
            if (eid >= ne || sol.period_of[eid] < 0) continue;
            if (sol.get_pr_count(sol.period_of[eid], sol.room_of[eid]) > 1) hard++;
        }
        return hard;
    }

    // ── Incremental delta evaluation ────────────────────────

    // Delta for removing an exam from its current slot (assigned → unassigned).
    double unassign_delta(const Solution& sol, int eid) const {
        const auto& po = sol.period_of;
        int old_pid = po[eid];
        if (old_pid < 0) return 0.0;
        int old_rid = sol.room_of[eid];
        double dh = 0, ds = 0;

        if (exam_dur[eid] > period_dur[old_pid]) dh -= 1;

        int enr = exam_enroll[eid];
        int old_total = sol.get_pr_enroll(old_pid, old_rid);
        dh -= ((old_total > room_cap[old_rid]) ? 1 : 0) -
              (((old_total - enr) > room_cap[old_rid]) ? 1 : 0);

        int old_day = period_day[old_pid], old_dpos = period_daypos[old_pid];
        for (int s : P.exams[eid].students) {
            for (int other : P.student_exams[s]) {
                if (other == eid) continue;
                int o_pid = po[other]; if (o_pid < 0) continue;
                if (o_pid == old_pid) dh -= 1;
                if (old_day == period_day[o_pid]) {
                    int g = std::abs(old_dpos - period_daypos[o_pid]);
                    if (g == 1) ds -= w_2row; else if (g > 1) ds -= w_2day;
                }
                int og = std::abs(old_pid - o_pid);
                if (og > 0 && og <= w_spread) ds -= 1;
            }
        }
        ds -= period_pen[old_pid] + room_pen[old_rid];
        if (large_exams.count(eid) && fl_penalty > 0 && last_periods.count(old_pid))
            ds -= fl_penalty;
        for (auto& [other, tcode] : phc_by_exam[eid]) {
            int o_pid = po[other]; if (o_pid < 0) continue;
            if (tcode == 0 && old_pid != o_pid)  dh -= 1;
            else if (tcode == 1 && old_pid == o_pid) dh -= 1;
            else if (tcode == 2 && old_pid <= o_pid) dh -= 1;
            else if (tcode == 3 && old_pid >= o_pid) dh -= 1;
        }
        // Room exclusive for unassign
        if (!rhc_exams.empty()) {
            int oc = sol.get_pr_count(old_pid, old_rid);
            if (rhc_exams.count(eid) && oc > 1) dh -= 1;
            for (int re : rhc_exams) {
                if (re == eid || re >= ne) continue;
                if (po[re] == old_pid && sol.room_of[re] == old_rid && oc == 2)
                    dh -= 1;
            }
        }
        return dh * 100000.0 + ds;
    }

    double move_delta(const Solution& sol, int eid, int new_pid, int new_rid) const {
        const auto& po = sol.period_of;
        int old_pid = po[eid];
        if (old_pid < 0) {
            // Assign from unassigned — only add new costs
            double dh = 0, ds = 0;
            if (exam_dur[eid] > period_dur[new_pid]) dh += 1;
            int enr = exam_enroll[eid];
            int new_total = sol.get_pr_enroll(new_pid, new_rid);
            dh += (((new_total + enr) > room_cap[new_rid]) ? 1 : 0) -
                  ((new_total > room_cap[new_rid]) ? 1 : 0);
            int new_day = period_day[new_pid], new_dpos = period_daypos[new_pid];
            for (int s : P.exams[eid].students) {
                for (int other : P.student_exams[s]) {
                    if (other == eid) continue;
                    int o_pid = po[other]; if (o_pid < 0) continue;
                    if (o_pid == new_pid) dh += 1;
                    if (new_day == period_day[o_pid]) {
                        int g = std::abs(new_dpos - period_daypos[o_pid]);
                        if (g == 1) ds += w_2row; else if (g > 1) ds += w_2day;
                    }
                    int ng = std::abs(new_pid - o_pid);
                    if (ng > 0 && ng <= w_spread) ds += 1;
                }
            }
            ds += period_pen[new_pid] + room_pen[new_rid];
            if (large_exams.count(eid) && fl_penalty > 0 && last_periods.count(new_pid))
                ds += fl_penalty;
            for (auto& [other, tcode] : phc_by_exam[eid]) {
                int o_pid = po[other]; if (o_pid < 0) continue;
                if (tcode == 0 && new_pid != o_pid) dh += 1;
                else if (tcode == 1 && new_pid == o_pid) dh += 1;
                else if (tcode == 2 && new_pid <= o_pid) dh += 1;
                else if (tcode == 3 && new_pid >= o_pid) dh += 1;
            }
            // Room exclusive for assign-from-unassigned
            if (!rhc_exams.empty()) {
                int nc = sol.get_pr_count(new_pid, new_rid);
                if (rhc_exams.count(eid) && nc > 0) dh += 1;
                for (int re : rhc_exams) {
                    if (re == eid || re >= ne) continue;
                    if (po[re] == new_pid && sol.room_of[re] == new_rid && nc == 1)
                        dh += 1;
                }
            }
            return dh * 100000.0 + ds;
        }
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
            } else if (tcode == 3) { // BEFORE: eid must be before other
                if (old_pid >= o_pid) dh -= 1;
                if (new_pid >= o_pid) dh += 1;
            }
        }

        // ── Room exclusive ──  O(|rhc_exams|) — typically 0-15 per dataset
        if (!rhc_exams.empty()) {
            int os = old_pid * nr + old_rid;
            int ns = new_pid * nr + new_rid;
            int oc = sol.get_pr_count(old_pid, old_rid);
            int nc = sol.get_pr_count(new_pid, new_rid);
            bool eid_rhc = rhc_exams.count(eid) > 0;
            if (eid_rhc) {
                if (oc > 1) dh -= 1;   // was sharing → violation removed
                if (nc > 0) dh += 1;   // will share → violation added
            }
            for (int re : rhc_exams) {
                if (re == eid || re >= ne) continue;
                int rp = po[re]; if (rp < 0) continue;
                int rr = sol.room_of[re];
                if (rp == old_pid && rr == old_rid && oc == 2)
                    dh -= 1;  // re was sharing only with eid → violation removed
                if (rp == new_pid && rr == new_rid && nc == 1)
                    dh += 1;  // re was alone → gains violation
            }
        }

        return dh * 100000.0 + ds;
    }

    // ── Period-only delta (for period-first steepest descent) ──
    // Returns (dh, ds) for changing exam eid to new_pid, excluding new-room
    // contributions. Caller adds room occupancy + room_pen per room in O(1).
    // old_room release is included; subtract room_pen[old_rid] is included.

    struct PeriodDelta { double dh, ds; };

    PeriodDelta move_delta_period(const Solution& sol, int eid, int new_pid) const {
        const auto& po = sol.period_of;
        int old_pid = po[eid];
        int old_rid = sol.room_of[eid];
        double dh = 0, ds = 0;

        int dur = exam_dur[eid];
        if (dur > period_dur[old_pid]) dh -= 1;
        if (dur > period_dur[new_pid]) dh += 1;

        // Old room occupancy release
        int enr = exam_enroll[eid];
        int old_total = sol.get_pr_enroll(old_pid, old_rid);
        dh -= ((old_total > room_cap[old_rid]) ? 1.0 : 0.0) -
              (((old_total - enr) > room_cap[old_rid]) ? 1.0 : 0.0);

        int old_day = period_day[old_pid], old_dpos = period_daypos[old_pid];
        int new_day = period_day[new_pid], new_dpos = period_daypos[new_pid];
        for (int s : P.exams[eid].students) {
            for (int other : P.student_exams[s]) {
                if (other == eid) continue;
                int o_pid = po[other]; if (o_pid < 0) continue;
                if (o_pid == old_pid) dh -= 1;
                if (o_pid == new_pid) dh += 1;
                int o_day = period_day[o_pid], o_dpos = period_daypos[o_pid];
                if (old_day == o_day) {
                    int g = std::abs(old_dpos - o_dpos);
                    if (g == 1) ds -= w_2row; else if (g > 1) ds -= w_2day;
                }
                int og = std::abs(old_pid - o_pid);
                if (og > 0 && og <= w_spread) ds -= 1;
                if (new_day == o_day) {
                    int g = std::abs(new_dpos - o_dpos);
                    if (g == 1) ds += w_2row; else if (g > 1) ds += w_2day;
                }
                int ng = std::abs(new_pid - o_pid);
                if (ng > 0 && ng <= w_spread) ds += 1;
            }
        }

        ds += period_pen[new_pid] - period_pen[old_pid];
        ds -= room_pen[old_rid]; // caller adds room_pen[new_rid]

        if (large_exams.count(eid) && fl_penalty > 0) {
            if (last_periods.count(old_pid) && !last_periods.count(new_pid)) ds -= fl_penalty;
            else if (!last_periods.count(old_pid) && last_periods.count(new_pid)) ds += fl_penalty;
        }

        for (auto& [other, tcode] : phc_by_exam[eid]) {
            int o_pid = po[other]; if (o_pid < 0) continue;
            if (tcode == 0)      { if (old_pid != o_pid) dh -= 1; if (new_pid != o_pid) dh += 1; }
            else if (tcode == 1) { if (old_pid == o_pid) dh -= 1; if (new_pid == o_pid) dh += 1; }
            else if (tcode == 2) { if (old_pid <= o_pid) dh -= 1; if (new_pid <= o_pid) dh += 1; }
            else if (tcode == 3) { if (old_pid >= o_pid) dh -= 1; if (new_pid >= o_pid) dh += 1; }
        }

        // Room exclusive: old room release only (caller handles new room)
        if (!rhc_exams.empty()) {
            int oc = sol.get_pr_count(old_pid, old_rid);
            if (rhc_exams.count(eid) && oc > 1) dh -= 1;
            for (int re : rhc_exams) {
                if (re == eid || re >= ne) continue;
                if (po[re] == old_pid && sol.room_of[re] == old_rid && oc == 2)
                    dh -= 1;
            }
        }

        return {dh, ds};
    }

    // ── Hard-only delta using adjacency list ─────────────────
    // O(degree(eid) + |rhc_exams|) — fast hard-only evaluation.
    // Includes room_exclusive via pr_count.

    int move_delta_hard(const Solution& sol, int eid, int new_pid, int new_rid) const {
        const auto& po = sol.period_of;
        int old_pid = po[eid];
        if (old_pid < 0) {
            int dh = 0;
            if (exam_dur[eid] > period_dur[new_pid]) dh++;
            int enr = exam_enroll[eid];
            int nt = sol.get_pr_enroll(new_pid, new_rid);
            dh += (((nt + enr) > room_cap[new_rid]) ? 1 : 0)
                - ((nt > room_cap[new_rid]) ? 1 : 0);
            for (auto& [nb, _] : P.adj[eid])
                if (po[nb] == new_pid) dh++;
            for (auto& [other, tc] : phc_by_exam[eid]) {
                int op = po[other]; if (op < 0) continue;
                if      (tc == 0 && new_pid != op) dh++;
                else if (tc == 1 && new_pid == op) dh++;
                else if (tc == 2 && new_pid <= op) dh++;
                else if (tc == 3 && new_pid >= op) dh++;
            }
            if (!rhc_exams.empty()) {
                int nc = sol.get_pr_count(new_pid, new_rid);
                if (rhc_exams.count(eid) && nc > 0) dh++;
                for (int re : rhc_exams) {
                    if (re == eid || re >= ne) continue;
                    if (po[re] == new_pid && sol.room_of[re] == new_rid && nc == 1)
                        dh++;
                }
            }
            return dh;
        }
        int old_rid = sol.room_of[eid];
        if (old_pid == new_pid && old_rid == new_rid) return 0;

        int dh = 0;
        if (exam_dur[eid] > period_dur[old_pid]) dh--;
        if (exam_dur[eid] > period_dur[new_pid]) dh++;
        int enr = exam_enroll[eid];
        int ot = sol.get_pr_enroll(old_pid, old_rid);
        int nt = sol.get_pr_enroll(new_pid, new_rid);
        dh -= ((ot > room_cap[old_rid]) ? 1 : 0)
            - (((ot - enr) > room_cap[old_rid]) ? 1 : 0);
        if (old_pid != new_pid || old_rid != new_rid)
            dh += (((nt + enr) > room_cap[new_rid]) ? 1 : 0)
                - ((nt > room_cap[new_rid]) ? 1 : 0);
        if (old_pid != new_pid) {
            for (auto& [nb, _] : P.adj[eid]) {
                int op = po[nb]; if (op < 0) continue;
                if (op == old_pid) dh--;
                if (op == new_pid) dh++;
            }
        }
        for (auto& [other, tc] : phc_by_exam[eid]) {
            int op = po[other]; if (op < 0) continue;
            if      (tc == 0) { if (old_pid != op) dh--; if (new_pid != op) dh++; }
            else if (tc == 1) { if (old_pid == op) dh--; if (new_pid == op) dh++; }
            else if (tc == 2) { if (old_pid <= op) dh--; if (new_pid <= op) dh++; }
            else if (tc == 3) { if (old_pid >= op) dh--; if (new_pid >= op) dh++; }
        }
        if (!rhc_exams.empty()) {
            int oc = sol.get_pr_count(old_pid, old_rid);
            int nc = sol.get_pr_count(new_pid, new_rid);
            bool eid_rhc = rhc_exams.count(eid) > 0;
            if (eid_rhc) {
                if (oc > 1) dh--;
                if (nc > 0) dh++;
            }
            for (int re : rhc_exams) {
                if (re == eid || re >= ne) continue;
                int rp = po[re]; if (rp < 0) continue;
                int rr = sol.room_of[re];
                if (rp == old_pid && rr == old_rid && oc == 2) dh--;
                if (rp == new_pid && rr == new_rid && nc == 1) dh++;
            }
        }
        return dh;
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
                std::vector<bool> blocked(np, false);
                for (auto& [nb, _] : P.adj[eid])
                    if (ns.period_of[nb] >= 0) blocked[ns.period_of[nb]] = true;
                // Also block exclusion periods
                for (auto& [other, tcode] : phc_by_exam[eid]) {
                    int op = ns.period_of[other]; if (op < 0) continue;
                    if (tcode == 1) blocked[op] = true; // EXCLUSION
                }

                std::vector<int> avail;
                for (int p = 0; p < np; p++)
                    if (!blocked[p] && exam_dur[eid] <= period_dur[p])
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

    // ── Graph decomposition ────────────────────────────────
    // BFS-based connected components of the conflict graph.
    // Used by CP-SAT (per-component solving) and Multi-Neighbourhood SA
    // (component-aware operator targeting).

    std::vector<std::vector<int>> find_connected_components() const {
        std::vector<bool> visited(ne, false);
        std::vector<std::vector<int>> components;
        for (int e = 0; e < ne; e++) {
            if (visited[e]) continue;
            std::vector<int> comp;
            std::queue<int> q;
            q.push(e); visited[e] = true;
            while (!q.empty()) {
                int cur = q.front(); q.pop();
                comp.push_back(cur);
                for (auto& [nb, _] : P.adj[cur]) {
                    if (!visited[nb]) { visited[nb] = true; q.push(nb); }
                }
            }
            components.push_back(std::move(comp));
        }
        return components;
    }

    // ── Room post-processing ────────────────────────────────
    // Per-period greedy room reassignment: moves each exam to its
    // best room (steepest descent) until no improvement. Targets
    // room_penalty, mixed_duration, room_occupancy.

    void optimize_rooms(Solution& sol) const {
        if (nr <= 1) return;
        for (int p = 0; p < np; p++) {
            for (int pass = 0; pass < 5; pass++) {
                bool changed = false;
                for (int e = 0; e < ne; e++) {
                    if (sol.period_of[e] != p) continue;
                    int cur_r = sol.room_of[e];
                    double best_d = -0.5;
                    int best_r = -1;
                    for (int r = 0; r < nr; r++) {
                        if (r == cur_r) continue;
                        double d = move_delta(sol, e, p, r);
                        if (d < best_d) { best_d = d; best_r = r; }
                    }
                    if (best_r >= 0) {
                        apply_move(sol, e, p, best_r);
                        changed = true;
                    }
                }
                if (!changed) break;
            }
        }
    }
};