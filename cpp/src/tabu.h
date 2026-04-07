/*
 * tabu.h — Feasibility-First Tabu Search
 *
 * Starts from DSatur greedy, optimizes with move_delta.
 * Re-syncs with full_eval every 10 iterations.
 * When infeasible: focuses on hard-violating exams + swap moves.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

inline AlgoResult solve_tabu(
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

    // Flat adjacency
    std::vector<std::vector<int>> adj(ne);
    for (int e = 0; e < ne; e++)
        for (auto& [nb, _] : prob.adj[e]) adj[e].push_back(nb);

    // Valid periods/rooms
    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (exam_dur[e] <= period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (exam_enr[e] <= room_cap[r])  valid_r[e].push_back(r);
    }

    // ── Init from greedy ──
    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, verbose); sol = g.sol.copy(); }
    FastEvaluator fe(prob);

    EvalResult ev = fe.full_eval(sol);
    double current_fitness = ev.fitness();
    Solution best_sol = sol.copy();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();

    if (verbose)
        std::cerr << "[Tabu] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft() << std::endl;

    // ── Tabu map: (eid * np + old_pid) -> expiry iteration ──
    std::unordered_map<int64_t, int> tabu;
    auto tabu_key = [&](int eid, int pid) -> int64_t { return (int64_t)eid * np + pid; };

    int no_improve = 0;

    // ── Find hard-violating exams ──
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

    // ── Main loop ──
    int iters_done = 0;
    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        // Re-sync every 10 iterations
        if (it % 10 == 0) {
            ev = fe.full_eval(sol);
            current_fitness = ev.fitness();
        }

        auto bad = get_bad();

        // Build candidate list
        std::vector<int> candidates;
        if (!bad.empty()) {
            candidates.assign(bad.begin(), bad.end());
            std::uniform_int_distribution<int> de(0, ne - 1);
            for (int i = 0; i < std::min(20, ne); i++) candidates.push_back(de(rng));
        } else {
            candidates.resize(ne);
            std::iota(candidates.begin(), candidates.end(), 0);
            std::shuffle(candidates.begin(), candidates.end(), rng);
            if ((int)candidates.size() > 60) candidates.resize(60);
        }

        // Find best single move
        int beid = -1, bpid = -1, brid = -1;
        double bdelta = 1e18;

        for (int eid : candidates) {
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

            for (int pid : tgts) {
                if (pid == cp) continue;
                auto& rooms = valid_r[eid];
                for (int ri = 0; ri < (int)rooms.size(); ri++) {
                    int rid = rooms[ri];
                    double d = fe.move_delta(sol, eid, pid, rid);
                    bool is_tabu = tabu.count(tabu_key(eid, cp)) &&
                                   tabu[tabu_key(eid, cp)] > it;
                    if (is_tabu && (current_fitness + d) >= best_fitness) continue;
                    if (d < bdelta) { bdelta = d; beid = eid; bpid = pid; brid = rid; }
                }
            }
        }

        // Swap moves when single moves can't fix infeasibility
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

                    double d1 = fe.move_delta(sol, ea, pb, ra);
                    fe.apply_move(sol, ea, pb, ra);
                    double d2 = fe.move_delta(sol, eb, pa, rb);
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
                tabu[tabu_key(sea, opa)] = it + tabu_tenure;
                tabu[tabu_key(seb, opb)] = it + tabu_tenure;
                beid = -2; // signal swap applied
            }

            if (beid == -1) {
                no_improve++;
                if (no_improve > patience) break;
                continue;
            }
        }

        if (beid >= 0) {
            int op = sol.period_of[beid];
            fe.apply_move(sol, beid, bpid, brid);
            current_fitness += bdelta;
            tabu[tabu_key(beid, op)] = it + tabu_tenure;
        }

        // Track best — verify with full_eval, feasibility-first
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
                    std::cerr << "[Tabu] Iter " << it << ": best hard=" << check.hard()
                              << " soft=" << check.soft() << std::endl;
            } else {
                current_fitness = af;
                no_improve++;
            }
        } else {
            no_improve++;
            if (no_improve > patience) break;
        }

        // Perturbation
        if (no_improve > 0 && no_improve % std::max(100, patience / 4) == 0) {
            std::uniform_int_distribution<int> de(0, ne-1);
            for (int k = 0; k < 3; k++) {
                int e = de(rng);
                if (!valid_p[e].empty() && !valid_r[e].empty()) {
                    int p = valid_p[e][rng() % valid_p[e].size()];
                    int r = valid_r[e][rng() % valid_r[e].size()];
                    fe.apply_move(sol, e, p, r);
                }
            }
            current_fitness = fe.full_eval(sol).fitness();
        }
    }

    fe.optimize_rooms(best_sol);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[Tabu] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "Tabu Search"};
}