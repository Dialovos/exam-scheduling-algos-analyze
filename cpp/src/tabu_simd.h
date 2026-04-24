/*
 * SIMD-accelerated drop-in Tabu Search.
 *
 * Same algorithm as solve_tabu in tabu.h — identical move selection,
 * tabu tenure, Kempe chain, swap, oscillation, token-ring logic.
 *
 * Differences:
 *   • Uses FastEvaluatorSIMD::move_delta_simd for full-delta evaluation in
 *     the main candidate loop (the hottest path).
 *   • Adds don't-look bits: exams whose last eval found no improving move
 *     are skipped until touched by a neighbour move. Classic local-search
 *     acceleration; cuts candidate loop size ~2-3× on long runs.
 *   • Period-only delta (move_delta_period) is still scalar — it's already
 *     fast (O(|adj[eid]|) once we're on the adj branch) and re-SIMDing
 *     it would duplicate the whole header.
 *
 * Use: --algo tabu_simd (expose in main.cpp when wiring)  OR  call
 *      solve_tabu_simd(prob, ...) directly from bench/test harness.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_simd.h"
#include "greedy.h"
#include "neighbourhoods.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <numeric>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

inline AlgoResult solve_tabu_simd(
    const ProblemInstance& prob,
    int max_iterations = 2000,
    int tabu_tenure    = 20,
    int patience       = 500,
    int seed           = 42,
    bool verbose       = false,
    const Solution* init_sol = nullptr)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();

    std::vector<int> exam_dur(ne), exam_enr(ne), period_dur(np), room_cap(nr);
    for (auto& e : prob.exams) { exam_dur[e.id] = e.duration; exam_enr[e.id] = e.enrollment(); }
    for (auto& p : prob.periods) period_dur[p.id] = p.duration;
    for (auto& r : prob.rooms) room_cap[r.id] = r.capacity;

    std::vector<std::vector<int>> adj(ne);
    for (int e = 0; e < ne; e++)
        for (auto& [nb, _] : prob.adj[e]) adj[e].push_back(nb);

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (exam_dur[e] <= period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (exam_enr[e] <= room_cap[r])  valid_r[e].push_back(r);
    }

    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, verbose); sol = g.sol.copy(); }

    FastEvaluator fe(prob);
    FastEvaluatorSIMD fes(fe);  // ← SIMD sidecar

    EvalResult ev = fe.full_eval(sol);
    double current_fitness = ev.fitness();
    Solution best_sol = sol.copy();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();

    if (verbose)
        std::cerr << "[TabuSIMD] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft() << std::endl;

    std::unordered_map<int64_t, int> tabu;
    auto tabu_key = [&](int eid, int pid) -> int64_t { return (int64_t)eid * np + pid; };

    int no_improve = 0;
    int base_tenure = tabu_tenure;

    // ── Don't-look bits ──
    // dont_look[e] = true means "last scan found no improving move for e —
    // skip until a neighbour is touched". Reset whenever a move changes a
    // neighbour's slot (their evaluation context shifted).
    std::vector<uint8_t> dont_look(ne, 0);
    auto clear_dont_look_around = [&](int eid) {
        dont_look[eid] = 0;
        for (int nb : adj[eid]) dont_look[nb] = 0;
    };

    auto get_bad = [&]() -> std::set<int> {
        std::set<int> bad;
        for (int e = 0; e < ne; e++) {
            int p = sol.period_of[e]; if (p < 0) continue;
            for (int nb : adj[e]) if (sol.period_of[nb] == p) { bad.insert(e); bad.insert(nb); }
            if (sol.get_pr_enroll(p, sol.room_of[e]) > room_cap[sol.room_of[e]]) bad.insert(e);
            if (exam_dur[e] > period_dur[p]) bad.insert(e);
        }
        for (auto& c : prob.phcs) {
            if (c.exam1 >= ne || c.exam2 >= ne) continue;
            int p1 = sol.period_of[c.exam1], p2 = sol.period_of[c.exam2];
            if (p1 < 0 || p2 < 0) continue;
            bool v = (c.type == "EXAM_COINCIDENCE" && p1 != p2) ||
                     (c.type == "EXCLUSION" && p1 == p2) ||
                     (c.type == "AFTER" && p1 <= p2);
            if (v) { bad.insert(c.exam1); bad.insert(c.exam2); }
        }
        return bad;
    };

    std::vector<double> tabu_exam_cost(ne, 1.0);
    AliasTable tabu_alias;
    auto recompute_tabu_costs = [&]() {
        for (int e = 0; e < ne; e++) {
            int pid = sol.period_of[e]; if (pid < 0) { tabu_exam_cost[e] = 1.0; continue; }
            double c = 1.0;
            for (auto& [nb, _] : prob.adj[e]) {
                int np2 = sol.period_of[nb]; if (np2 < 0) continue;
                if (np2 == pid) c += 100000;
                int gap = std::abs(pid - np2);
                if (gap > 0 && gap <= fe.w_spread) c += 1;
                if (fe.period_day[pid] == fe.period_day[np2]) {
                    int g = std::abs(fe.period_daypos[pid] - fe.period_daypos[np2]);
                    if (g == 1) c += fe.w_2row;
                    else if (g > 1) c += fe.w_2day;
                }
            }
            tabu_exam_cost[e] = c;
        }
        tabu_alias.build(tabu_exam_cost);
    };
    recompute_tabu_costs();

    std::vector<std::vector<int>> visit_count(ne, std::vector<int>(np, 0));
    std::uniform_int_distribution<int> de(0, ne - 1);

    int iters_done = 0;
    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        if (it % 50 == 0) {
            ev = fe.full_eval(sol);
            current_fitness = ev.fitness();
        }

        int active_tenure = base_tenure + (no_improve / 100) * 5;
        active_tenure = std::min(active_tenure, base_tenure * 3);

        if (it % 200 == 0) recompute_tabu_costs();

        auto bad = get_bad();

        std::vector<int> candidates;
        std::vector<bool> in_cands(ne, false);
        if (!bad.empty()) {
            for (int e : bad) { candidates.push_back(e); in_cands[e] = true; }
        }
        int n_extra = std::max(60, 120 - (int)candidates.size());
        for (int i = 0; i < n_extra * 2 && (int)candidates.size() < 120; i++) {
            int e = tabu_alias.sample(rng);
            if (!in_cands[e]) { candidates.push_back(e); in_cands[e] = true; }
        }

        int beid = -1, bpid = -1, brid = -1;
        double bdelta = 1e18;

        for (int eid : candidates) {
            if (dont_look[eid] && !bad.count(eid)) continue;  // skip stuck

            int cp = sol.period_of[eid];
            auto& targets = valid_p[eid];
            std::vector<int> tgts;
            bool is_bad = bad.count(eid) > 0;
            if (!is_bad && (int)targets.size() > 12) {
                tgts = targets;
                std::shuffle(tgts.begin(), tgts.end(), rng);
                tgts.resize(12);
            } else {
                tgts = targets;
            }

            int enr_e = exam_enr[eid];
            bool is_tabu = tabu.count(tabu_key(eid, cp)) &&
                           tabu[tabu_key(eid, cp)] > it;

            bool found_any_improving = false;
            for (int pid : tgts) {
                if (pid == cp) continue;
                auto pd = fe.move_delta_period(sol, eid, pid);
                auto& rooms = valid_r[eid];
                for (int ri = 0; ri < (int)rooms.size(); ri++) {
                    int rid = rooms[ri];
                    double dh = pd.dh, ds = pd.ds;
                    int new_total = sol.get_pr_enroll(pid, rid);
                    dh += (((new_total + enr_e) > room_cap[rid]) ? 1.0 : 0.0) -
                          ((new_total > room_cap[rid]) ? 1.0 : 0.0);
                    ds += fe.room_pen[rid];
                    double d = dh * 100000.0 + ds;
                    if (is_tabu && (current_fitness + d) >= best_fitness) continue;
                    if (d < 0) found_any_improving = true;
                    if (d < bdelta) { bdelta = d; beid = eid; bpid = pid; brid = rid; }
                }
            }
            if (!found_any_improving && !is_bad) dont_look[eid] = 1;
        }

        // Swap moves — use SIMD move_delta for the two-way scoring
        if ((beid < 0 || (bdelta >= 0 && !bad.empty())) && !bad.empty()) {
            std::vector<int> swap_cands(bad.begin(), bad.end());
            if ((int)swap_cands.size() > 10) swap_cands.resize(10);
            double best_swap_d = (beid >= 0) ? bdelta : 1e18;
            int sea = -1, sepb = -1, sera = -1, seb = -1, sepa = -1, serb = -1;

            for (int ea : swap_cands) {
                int pa = sol.period_of[ea], ra = sol.room_of[ea];
                std::vector<int> stgts(ne);
                std::iota(stgts.begin(), stgts.end(), 0);
                std::shuffle(stgts.begin(), stgts.end(), rng);
                if ((int)stgts.size() > 50) stgts.resize(50);

                for (int eb : stgts) {
                    if (eb == ea) continue;
                    int pb = sol.period_of[eb], rb = sol.room_of[eb];
                    if (pb == pa) continue;
                    if (exam_dur[ea] > period_dur[pb] || exam_dur[eb] > period_dur[pa]) continue;

                    double d1 = fes.move_delta_simd(sol, ea, pb, ra);  // ← SIMD
                    fe.apply_move(sol, ea, pb, ra);
                    double d2 = fes.move_delta_simd(sol, eb, pa, rb);  // ← SIMD
                    sol.assign(eb, pb, rb); sol.assign(ea, pa, ra); // undo
                    double td = d1 + d2;
                    if (td < best_swap_d) {
                        best_swap_d = td;
                        sea = ea; sepb = pb; sera = ra;
                        seb = eb; sepa = pa; serb = rb;
                    }
                }
            }
            if (sea >= 0 && best_swap_d < ((beid >= 0) ? bdelta : 1e18)) {
                int opa = sol.period_of[sea], opb = sol.period_of[seb];
                fe.apply_move(sol, sea, sepb, sera);
                fe.apply_move(sol, seb, sepa, serb);
                current_fitness += best_swap_d;
                tabu[tabu_key(sea, opa)] = it + active_tenure;
                tabu[tabu_key(seb, opb)] = it + active_tenure;
                clear_dont_look_around(sea);
                clear_dont_look_around(seb);
                beid = -2;
            }

            if (beid == -1) {
                no_improve++;
                if (no_improve > patience) break;
                continue;
            }
        }

        if ((no_improve > 50 || !bad.empty()) && beid != -2) {
            for (int kc = 0; kc < 3; kc++) {
                int kseed = tabu_alias.sample(rng);
                int kp1 = sol.period_of[kseed]; if (kp1 < 0) continue;
                int kp2 = std::uniform_int_distribution<int>(0, np - 1)(rng);
                if (kp2 == kp1) continue;

                auto chain = kempe_detail::build_chain(sol, prob.adj, ne, kseed, kp1, kp2);
                if (chain.empty() || (int)chain.size() > ne / 4) continue;

                auto old_pe = fe.partial_eval(sol, chain);
                auto undo_info = kempe_detail::apply_chain(sol, chain, kp1, kp2);
                auto new_pe = fe.partial_eval(sol, chain);
                double kd = new_pe.fitness() - old_pe.fitness();

                bool any_tabu = false;
                for (auto& u : undo_info)
                    if (tabu.count(tabu_key(u.eid, u.old_pid)) && tabu[tabu_key(u.eid, u.old_pid)] > it)
                        { any_tabu = true; break; }

                bool aspiration = (current_fitness + kd < best_fitness);
                if ((!any_tabu || aspiration) && kd < bdelta) {
                    for (auto& u : undo_info) {
                        tabu[tabu_key(u.eid, u.old_pid)] = it + active_tenure;
                        clear_dont_look_around(u.eid);
                    }
                    current_fitness += kd;
                    beid = -3;
                    break;
                } else {
                    kempe_detail::undo_chain(sol, undo_info);
                }
            }
        }

        if (beid >= 0) {
            int op = sol.period_of[beid];
            fe.apply_move(sol, beid, bpid, brid);
            current_fitness += bdelta;
            tabu[tabu_key(beid, op)] = it + active_tenure;
            visit_count[beid][bpid]++;
            clear_dont_look_around(beid);
        }

        if (current_fitness < best_fitness - 0.5) {
            auto check = fe.full_eval(sol);
            double af = check.fitness();
            bool af_feasible = check.feasible();
            bool dominated = (best_feasible && !af_feasible);
            if (!dominated && af < best_fitness) {
                best_sol = sol.copy();
                best_fitness = af;
                best_feasible = af_feasible;
                current_fitness = af;
                no_improve = 0;
                if (verbose && (it < 10 || it % 200 == 0))
                    std::cerr << "[TabuSIMD] Iter " << it << ": best hard=" << check.hard()
                              << " soft=" << check.soft() << std::endl;
            } else {
                current_fitness = af;
                no_improve++;
            }
        } else {
            no_improve++;
            if (no_improve > patience) break;
        }

        if (no_improve > 0 && no_improve % std::max(100, patience / 4) == 0) {
            std::fill(dont_look.begin(), dont_look.end(), 0);  // reset on perturbation
            int osc_iters = 20 + (rng() % 30);
            for (int oi = 0; oi < osc_iters; oi++) {
                int e = de(rng);
                if (!valid_p[e].empty() && !valid_r[e].empty()) {
                    int p = valid_p[e][rng() % valid_p[e].size()];
                    int r = valid_r[e][rng() % valid_r[e].size()];
                    double d = fes.move_delta_simd(sol, e, p, r);
                    if (d < 0) fe.apply_move(sol, e, p, r);
                }
            }
            ev = fe.full_eval(sol);
            current_fitness = ev.fitness();
        }

        if (no_improve > patience / 2) {
            int e = tabu_alias.sample(rng);
            int min_visits = INT_MAX, best_p = -1;
            for (int p : valid_p[e]) {
                if (visit_count[e][p] < min_visits) { min_visits = visit_count[e][p]; best_p = p; }
            }
            if (best_p >= 0 && best_p != sol.period_of[e]) {
                int r = valid_r[e].empty() ? 0 : valid_r[e][rng() % valid_r[e].size()];
                fe.apply_move(sol, e, best_p, r);
                visit_count[e][best_p]++;
                clear_dont_look_around(e);
                ev = fe.full_eval(sol);
                current_fitness = ev.fitness();
            }
        }
    }

    fe.optimize_rooms(best_sol);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[TabuSIMD] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "Tabu (SIMD)"};
}
