/*
 * greedy.h — DSatur Greedy with Multi-Start & Chain Repair
 *
 * 1. Pre-place room-dominating exams (>50% of max room capacity)
 * 2. DSatur ordering: pick exam with highest saturation + room pressure
 * 3. Displacement repair when stuck (move blockers)
 * 4. Chain-move room overflow repair
 * 5. Delta-based general repair for remaining violations
 * 6. Multi-start: retry with randomized tiebreaking if infeasible
 */

#pragma once

#include "models.h"
#include "evaluator.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <set>
#include <vector>

namespace greedy_detail {

struct GreedyCtx {
    const ProblemInstance& P;
    int ne, np, nr;

    std::vector<int> exam_dur, exam_enr, period_dur, room_cap;
    std::vector<std::vector<int>> adj_flat;      // conflict neighbors (IDs only)
    std::vector<std::vector<int>> valid_periods;  // periods where duration fits
    std::vector<std::vector<int>> valid_rooms;    // rooms where enrollment fits (sorted by cap asc)
    std::vector<std::set<int>> exclusion_of;      // EXCLUSION partners
    std::vector<std::vector<int>> coin_group;     // coincidence group (empty if none)
    // after_of[e] = {(other, dir)}  dir: 0=must_come_after, 1=must_come_before
    std::vector<std::vector<std::pair<int,int>>> after_of;
    int max_cap;

    explicit GreedyCtx(const ProblemInstance& p) : P(p) {
        ne = p.n_e(); np = p.n_p(); nr = p.n_r();

        exam_dur.resize(ne); exam_enr.resize(ne);
        for (auto& e : p.exams) { exam_dur[e.id] = e.duration; exam_enr[e.id] = e.enrollment(); }

        period_dur.resize(np);
        for (auto& pp : p.periods) period_dur[pp.id] = pp.duration;

        room_cap.resize(nr);
        for (auto& r : p.rooms) room_cap[r.id] = r.capacity;
        max_cap = *std::max_element(room_cap.begin(), room_cap.end());

        // Conflict adjacency (IDs only)
        adj_flat.resize(ne);
        for (int e = 0; e < ne; e++)
            for (auto& [nb, _] : p.adj[e])
                adj_flat[e].push_back(nb);

        // Valid periods/rooms per exam
        valid_periods.resize(ne);
        valid_rooms.resize(ne);
        for (int e = 0; e < ne; e++) {
            for (int pp = 0; pp < np; pp++)
                if (exam_dur[e] <= period_dur[pp])
                    valid_periods[e].push_back(pp);
            std::vector<int> vr;
            for (int r = 0; r < nr; r++)
                if (exam_enr[e] <= room_cap[r])
                    vr.push_back(r);
            std::sort(vr.begin(), vr.end(), [&](int a, int b){ return room_cap[a] < room_cap[b]; });
            valid_rooms[e] = std::move(vr);
        }

        // Period hard constraints
        exclusion_of.resize(ne);
        coin_group.resize(ne);
        after_of.resize(ne);
        std::vector<std::set<int>> coincidence(ne);

        for (auto& c : p.phcs) {
            int e1 = c.exam1, e2 = c.exam2;
            if (e1 >= ne || e2 >= ne) continue;
            if (c.type == "EXAM_COINCIDENCE") {
                coincidence[e1].insert(e2);
                coincidence[e2].insert(e1);
            } else if (c.type == "EXCLUSION") {
                exclusion_of[e1].insert(e2);
                exclusion_of[e2].insert(e1);
            } else if (c.type == "AFTER") {
                after_of[e1].push_back({e2, 0}); // e1 after e2
                after_of[e2].push_back({e1, 1}); // e2 before e1
            }
        }

        // Build coincidence groups (connected components)
        std::vector<bool> vis(ne, false);
        for (int e = 0; e < ne; e++) {
            if (vis[e] || coincidence[e].empty()) continue;
            std::vector<int> group;
            std::vector<int> stk = {e};
            while (!stk.empty()) {
                int cur = stk.back(); stk.pop_back();
                if (vis[cur]) continue;
                vis[cur] = true;
                group.push_back(cur);
                for (int nb : coincidence[cur])
                    if (!vis[nb]) stk.push_back(nb);
            }
            for (int m : group) coin_group[m] = group;
        }
    }
};

inline AlgoResult solve_greedy_once(
    const ProblemInstance& prob, const GreedyCtx& ctx, int seed, bool verbose)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = ctx.ne, np = ctx.np, nr = ctx.nr;
    Solution sol; sol.init(prob);
    std::vector<bool> assigned(ne, false);
    std::vector<int> sat_counts(ne, 0);

    // Degree = conflict neighbors + exclusion partners
    std::vector<int> degree(ne);
    for (int e = 0; e < ne; e++)
        degree[e] = (int)ctx.adj_flat[e].size() + (int)ctx.exclusion_of[e].size();

    // Room pressure: exams using >50% of max room cap get boosted priority
    std::vector<int> room_pressure(ne);
    for (int e = 0; e < ne; e++)
        room_pressure[e] = (int)(10.0 * ctx.exam_enr[e] / ctx.max_cap);

    // ── Helpers ──
    auto blocked_periods = [&](int eid) -> std::set<int> {
        std::set<int> blocked;
        for (int nb : ctx.adj_flat[eid])
            if (assigned[nb] && sol.period_of[nb] >= 0)
                blocked.insert(sol.period_of[nb]);
        for (int nb : ctx.exclusion_of[eid])
            if (assigned[nb] && sol.period_of[nb] >= 0)
                blocked.insert(sol.period_of[nb]);
        return blocked;
    };

    auto required_period = [&](int eid) -> int {
        for (int nb : ctx.coin_group[eid])
            if (nb != eid && assigned[nb]) return sol.period_of[nb];
        return -1;
    };

    auto after_ok = [&](int eid, int pid) -> bool {
        for (auto& [other, dir] : ctx.after_of[eid]) {
            if (!assigned[other]) continue;
            int o = sol.period_of[other]; if (o < 0) continue;
            if (dir == 0 && pid <= o) return false; // must come after
            if (dir == 1 && pid >= o) return false; // must come before
        }
        return true;
    };

    auto find_room = [&](int eid, int pid) -> int {
        for (int rid : ctx.valid_rooms[eid])
            if (sol.get_pr_enroll(pid, rid) + ctx.exam_enr[eid] <= ctx.room_cap[rid])
                return rid;
        return -1;
    };

    auto try_place = [&](int eid) -> bool {
        int req = required_period(eid);
        if (req >= 0) {
            int rid = find_room(eid, req);
            if (rid < 0) rid = ctx.valid_rooms[eid].empty() ? 0 : ctx.valid_rooms[eid][0];
            sol.assign(eid, req, rid);
            return true;
        }
        auto blocked = blocked_periods(eid);
        for (int pid : ctx.valid_periods[eid]) {
            if (blocked.count(pid)) continue;
            if (!after_ok(eid, pid)) continue;
            int rid = find_room(eid, pid);
            if (rid >= 0) { sol.assign(eid, pid, rid); return true; }
        }
        return false;
    };

    auto try_displace = [&](int eid) -> bool {
        auto blocked = blocked_periods(eid);
        std::set<int> vset(ctx.valid_periods[eid].begin(), ctx.valid_periods[eid].end());
        std::vector<int> cands;
        for (int p : vset) if (blocked.count(p)) cands.push_back(p);
        std::sort(cands.begin(), cands.end(), [&](int a, int b){
            int ca = 0, cb = 0;
            for (int nb : ctx.adj_flat[eid]) { if (assigned[nb] && sol.period_of[nb]==a) ca++; if (assigned[nb] && sol.period_of[nb]==b) cb++; }
            return ca < cb;
        });

        for (int target : cands) {
            if (!after_ok(eid, target)) continue;
            std::vector<int> blockers;
            for (int nb : ctx.adj_flat[eid])
                if (assigned[nb] && sol.period_of[nb] == target) blockers.push_back(nb);
            for (int nb : ctx.exclusion_of[eid])
                if (assigned[nb] && sol.period_of[nb] == target) {
                    bool dup = false; for (int b : blockers) if (b == nb) { dup = true; break; }
                    if (!dup) blockers.push_back(nb);
                }

            std::vector<std::tuple<int,int,int>> rollback;
            bool all_moved = true;
            for (int blocker : blockers) {
                if (!ctx.coin_group[blocker].empty()) {
                    bool has_coinc = false;
                    for (int m : ctx.coin_group[blocker])
                        if (m != blocker && assigned[m] && sol.period_of[m] == target) { has_coinc = true; break; }
                    if (has_coinc) { all_moved = false; break; }
                }
                std::set<int> bb;
                for (int bnb : ctx.adj_flat[blocker])
                    if (assigned[bnb] && bnb != eid && sol.period_of[bnb] >= 0) bb.insert(sol.period_of[bnb]);
                for (int bnb : ctx.exclusion_of[blocker])
                    if (assigned[bnb] && bnb != eid && sol.period_of[bnb] >= 0) bb.insert(sol.period_of[bnb]);
                bb.insert(target);

                bool moved = false;
                for (int bp : ctx.valid_periods[blocker]) {
                    if (bb.count(bp)) continue;
                    if (!after_ok(blocker, bp)) continue;
                    int br = find_room(blocker, bp);
                    if (br >= 0) {
                        rollback.push_back({blocker, sol.period_of[blocker], sol.room_of[blocker]});
                        sol.assign(blocker, bp, br);
                        moved = true; break;
                    }
                }
                if (!moved) { all_moved = false; break; }
            }
            if (all_moved) {
                int rid = find_room(eid, target);
                if (rid >= 0) { sol.assign(eid, target, rid); return true; }
            }
            for (int i = (int)rollback.size()-1; i >= 0; i--) {
                auto [be, bp, br] = rollback[i];
                sol.assign(be, bp, br);
            }
        }
        return false;
    };

    auto force_place = [&](int eid) {
        int req = required_period(eid);
        if (req >= 0) {
            int rid = find_room(eid, req);
            if (rid < 0) rid = ctx.valid_rooms[eid].empty() ? 0 : ctx.valid_rooms[eid][0];
            sol.assign(eid, req, rid); return;
        }
        int best_pid = -1, best_rid = -1;
        long long best_cost = (long long)1e18;
        for (int pid : ctx.valid_periods[eid]) {
            int conflicts = 0;
            for (int nb : ctx.adj_flat[eid]) if (assigned[nb] && sol.period_of[nb] == pid) conflicts++;
            int excl = 0;
            for (int nb : ctx.exclusion_of[eid]) if (assigned[nb] && sol.period_of[nb] == pid) excl++;
            int av = after_ok(eid, pid) ? 0 : 1;
            int rid = find_room(eid, pid);
            int overflow = 0;
            if (rid < 0) {
                rid = ctx.valid_rooms[eid].empty() ? 0 : ctx.valid_rooms[eid][0];
                overflow = std::max(0, sol.get_pr_enroll(pid, rid) + ctx.exam_enr[eid] - ctx.room_cap[rid]);
            }
            long long cost = (long long)(conflicts + excl) * 100000LL + (long long)av * 50000LL + (long long)overflow * 100LL;
            if (cost < best_cost) { best_cost = cost; best_pid = pid; best_rid = rid; }
        }
        if (best_pid < 0) { best_pid = 0; best_rid = 0; }
        sol.assign(eid, best_pid, best_rid);
    };

    auto update_sat = [&](int eid) {
        int pid = sol.period_of[eid]; if (pid < 0) return;
        for (int nb : ctx.adj_flat[eid]) if (!assigned[nb]) sat_counts[nb]++;
        for (int nb : ctx.exclusion_of[eid]) if (!assigned[nb]) sat_counts[nb]++;
    };

    auto pick_next = [&]() -> int {
        struct Key { int has_req, sat, deg, enr, id; };
        auto cmp = [](const Key& a, const Key& b) {
            if (a.has_req != b.has_req) return a.has_req > b.has_req;
            if (a.sat != b.sat) return a.sat > b.sat;
            if (a.deg != b.deg) return a.deg > b.deg;
            if (a.enr != b.enr) return a.enr > b.enr;
            return a.id < b.id;
        };
        std::vector<Key> cands;
        for (int e = 0; e < ne; e++) {
            if (assigned[e]) continue;
            int hr = (required_period(e) >= 0) ? 1 : 0;
            cands.push_back({hr, sat_counts[e], degree[e] + room_pressure[e], ctx.exam_enr[e], e});
        }
        if (cands.empty()) return -1;
        std::sort(cands.begin(), cands.end(), cmp);
        // Randomized tiebreaking for non-default seeds
        if (seed != 42 && cands.size() > 1) {
            int top_k = std::min(3, (int)cands.size());
            return cands[rng() % top_k].id;
        }
        return cands[0].id;
    };

    int stats[3] = {0, 0, 0};

    // ── Pre-place room-dominating exams ──
    std::vector<int> dominating;
    for (int e = 0; e < ne; e++)
        if (ctx.exam_enr[e] > ctx.max_cap / 2) dominating.push_back(e);
    std::sort(dominating.begin(), dominating.end(), [&](int a, int b){ return ctx.exam_enr[a] > ctx.exam_enr[b]; });
    for (int eid : dominating) {
        if (assigned[eid]) continue;
        if (try_place(eid)) { assigned[eid] = true; update_sat(eid); stats[0]++; }
    }

    // ── DSatur main loop ──
    for (int step = 0; step < ne; step++) {
        int eid = pick_next();
        if (eid < 0) break;
        if (try_place(eid))           { assigned[eid] = true; update_sat(eid); stats[0]++; }
        else if (try_displace(eid))   { assigned[eid] = true; update_sat(eid); stats[1]++; }
        else { force_place(eid);        assigned[eid] = true; update_sat(eid); stats[2]++; }
    }

    // ── Repair ──
    FastEvaluator fe(prob);
    EvalResult ev = fe.full_eval(sol);
    int initial_hard = ev.hard();

    // Phase 0: Chain-move room overflow repair
    for (int round = 0; round < 10 && ev.room_occupancy > 0; round++) {
        bool fixed = false;
        for (int sp = 0; sp < np && !fixed; sp++) {
            for (int sr = 0; sr < nr && !fixed; sr++) {
                if (sol.get_pr_enroll(sp, sr) <= ctx.room_cap[sr]) continue;
                // Find largest exam in overflowing slot
                int large_e = -1, large_enr = 0;
                for (int e = 0; e < ne; e++)
                    if (sol.period_of[e] == sp && sol.room_of[e] == sr && ctx.exam_enr[e] > large_enr)
                        { large_e = e; large_enr = ctx.exam_enr[e]; }
                if (large_e < 0) continue;

                for (int tp = 0; tp < np; tp++) {
                    if (tp == sp) continue;
                    if (ctx.exam_dur[large_e] > ctx.period_dur[tp]) continue;
                    bool hc = false;
                    for (int nb : ctx.adj_flat[large_e]) if (sol.period_of[nb] == tp) { hc = true; break; }
                    if (hc) continue;

                    int tgt_enr = sol.get_pr_enroll(tp, sr);
                    int need_free = tgt_enr + large_enr - ctx.room_cap[sr];
                    if (need_free <= 0) {
                        sol.assign(large_e, tp, sr); fixed = true; break;
                    }
                    // Chain: free enrollment from target period
                    std::vector<std::pair<int,int>> tgt_exams;
                    for (int e = 0; e < ne; e++)
                        if (sol.period_of[e] == tp && sol.room_of[e] == sr && e != large_e)
                            tgt_exams.push_back({ctx.exam_enr[e], e});
                    std::sort(tgt_exams.begin(), tgt_exams.end());

                    int freed = 0;
                    std::vector<std::tuple<int,int,int>> moves;
                    for (auto& [me, mid] : tgt_exams) {
                        if (freed >= need_free) break;
                        for (int ap = 0; ap < np; ap++) {
                            if (ap == tp || ap == sp) continue;
                            if (ctx.exam_dur[mid] > ctx.period_dur[ap]) continue;
                            bool mc = false;
                            for (int nb : ctx.adj_flat[mid]) if (sol.period_of[nb] == ap) { mc = true; break; }
                            if (mc) continue;
                            for (int ar = 0; ar < nr; ar++) {
                                if (ctx.exam_enr[mid] <= ctx.room_cap[ar] &&
                                    sol.get_pr_enroll(ap, ar) + ctx.exam_enr[mid] <= ctx.room_cap[ar]) {
                                    moves.push_back({mid, ap, ar});
                                    freed += me; break;
                                }
                            }
                            if (!moves.empty() && std::get<0>(moves.back()) == mid) break;
                        }
                    }
                    if (freed >= need_free) {
                        for (auto& [me2, mp, mr] : moves) sol.assign(me2, mp, mr);
                        sol.assign(large_e, tp, sr); fixed = true; break;
                    }
                }
            }
        }
        ev = fe.full_eval(sol);
    }

    // Phase 1: Delta-based general repair
    for (int round = 0; round < 200; round++) {
        ev = fe.full_eval(sol);
        if (ev.hard() == 0) break;

        // Collect bad exams
        std::set<int> bad;
        for (int e = 0; e < ne; e++) {
            int p = sol.period_of[e]; if (p < 0) continue;
            for (int nb : ctx.adj_flat[e]) if (sol.period_of[nb] == p) { bad.insert(e); bad.insert(nb); }
            if (sol.get_pr_enroll(p, sol.room_of[e]) > ctx.room_cap[sol.room_of[e]]) bad.insert(e);
            if (ctx.exam_dur[e] > ctx.period_dur[p]) bad.insert(e);
        }
        for (auto& c : prob.phcs) {
            if (c.exam1 >= ne || c.exam2 >= ne) continue;
            int p1 = sol.period_of[c.exam1], p2 = sol.period_of[c.exam2];
            if (p1 < 0 || p2 < 0) continue;
            bool viol = (c.type == "EXAM_COINCIDENCE" && p1 != p2) ||
                        (c.type == "EXCLUSION" && p1 == p2) ||
                        (c.type == "AFTER" && p1 <= p2);
            if (viol) { bad.insert(c.exam1); bad.insert(c.exam2); }
        }
        if (bad.empty()) break;

        int beid = -1, bpid = -1, brid = -1; double bdelta = 0.0;
        for (int eid : bad) {
            int cp = sol.period_of[eid];
            for (int pid = 0; pid < np; pid++) {
                if (pid == cp || ctx.exam_dur[eid] > ctx.period_dur[pid]) continue;
                for (int rid = 0; rid < nr; rid++) {
                    double d = fe.move_delta(sol, eid, pid, rid);
                    if (d < bdelta) { bdelta = d; beid = eid; bpid = pid; brid = rid; }
                }
            }
        }
        if (beid < 0 || bdelta >= 0) {
            // Accept sideways
            for (int eid : bad) {
                int cp = sol.period_of[eid];
                for (int pid = 0; pid < np; pid++) {
                    if (pid == cp || ctx.exam_dur[eid] > ctx.period_dur[pid]) continue;
                    for (int rid = 0; rid < nr; rid++) {
                        double d = fe.move_delta(sol, eid, pid, rid);
                        if (d < bdelta && d < 50000) { bdelta = d; beid = eid; bpid = pid; brid = rid; }
                    }
                }
            }
            if (beid < 0) break;
        }
        fe.apply_move(sol, beid, bpid, brid);
    }

    ev = fe.full_eval(sol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();

    if (verbose)
        std::cerr << "[Greedy] " << rt << "s  clean=" << stats[0]
                  << " displaced=" << stats[1] << " forced=" << stats[2]
                  << "  feasible=" << ev.feasible() << " hard=" << ev.hard()
                  << " soft=" << ev.soft() << std::endl;

    return {std::move(sol), ev, rt, 0, "Greedy"};
}

} // namespace greedy_detail

inline AlgoResult solve_greedy(const ProblemInstance& prob, bool verbose = false) {
    greedy_detail::GreedyCtx ctx(prob);

    auto r = greedy_detail::solve_greedy_once(prob, ctx, 42, verbose);
    if (r.eval.feasible()) return r;

    // Multi-start
    if (verbose)
        std::cerr << "[Greedy] Infeasible (hard=" << r.eval.hard()
                  << "), trying random seeds..." << std::endl;

    AlgoResult best = std::move(r);
    for (int s = 0; s < 30; s++) {
        auto r2 = greedy_detail::solve_greedy_once(prob, ctx, s, false);
        if (r2.eval.hard() < best.eval.hard() ||
            (r2.eval.hard() == best.eval.hard() && r2.eval.soft() < best.eval.soft())) {
            best = std::move(r2);
            if (best.eval.feasible()) {
                if (verbose)
                    std::cerr << "[Greedy] Feasible at seed " << s << "!" << std::endl;
                break;
            }
        }
    }
    best.algorithm = "Greedy";
    return best;
}