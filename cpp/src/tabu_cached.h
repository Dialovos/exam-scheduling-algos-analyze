/*
 * Phase 2b' — Tabu Search with incremental cached fitness.
 *
 * Same algorithm as solve_tabu_simd. Only change: every move_delta and
 * apply_move routes through CachedEvaluator. Period-decomposition trick
 * is dropped because move_delta_cached is already cheap enough (~30 ns)
 * that the 2-phase trick stops paying off.
 *
 * Invariant critical to correctness: every state mutation of `sol` MUST
 * go through Ecach.apply_move, else the cache desyncs. In particular,
 * Kempe chain and swap moves do their own assign — we rebuild the
 * affected exams' cache rows after those.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_simd.h"
#include "evaluator_cached.h"
#include "greedy.h"
#include "neighbourhoods.h"
#include "ejection.h"
#include "xoshiro.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <climits>
#include <numeric>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

inline AlgoResult solve_tabu_cached(
    const ProblemInstance& prob,
    int max_iterations = 2000,
    int tabu_tenure    = 20,
    int patience       = 500,
    int seed           = 42,
    bool verbose       = false,
    const Solution* init_sol = nullptr)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);   // keep canonical for deterministic cross-run comparison;
                              // swap to Xoshiro256pp for 1.5-2.5× RNG speedup (different trajectory)

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
    CachedEvaluator Ecach(fe);
    Ecach.initialize(sol);

    // Helper: rebuild the cache rows for an exam and all its neighbours.
    // Use after Kempe chain or swap mutations that didn't go through
    // Ecach.apply_move.
    auto refresh_around = [&](int eid) {
        Ecach.rebuild_contrib_for(eid, sol);
        for (auto& pr : prob.adj[eid])
            Ecach.rebuild_contrib_for(pr.first, sol);
    };

    EvalResult ev = fe.full_eval(sol);
    double current_fitness = ev.fitness();
    Solution best_sol = sol.copy();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();

    if (verbose)
        std::cerr << "[TabuCached] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft() << std::endl;

    std::unordered_map<int64_t, int> tabu;
    auto tabu_key = [&](int eid, int pid) -> int64_t { return (int64_t)eid * np + pid; };

    int no_improve = 0;
    int base_tenure = tabu_tenure;

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

        // ── Phase 3a: intra-algo OpenMP on candidate scan ──
        // Each thread scans a disjoint subset of candidates, maintains its
        // own best; final reduction picks global best. Shuffling of targets
        // is pre-computed per-candidate BEFORE the parallel region so RNG
        // stays deterministic.
        int n_cand = (int)candidates.size();
        std::vector<std::vector<int>> cand_tgts(n_cand);
        std::vector<int> cand_cp(n_cand);
        std::vector<uint8_t> cand_is_bad(n_cand, 0);
        std::vector<uint8_t> cand_is_tabu(n_cand, 0);
        std::vector<uint8_t> cand_skip(n_cand, 0);
        for (int ci = 0; ci < n_cand; ci++) {
            int eid = candidates[ci];
            if (dont_look[eid] && !bad.count(eid)) { cand_skip[ci] = 1; continue; }
            int cp = sol.period_of[eid];
            cand_cp[ci] = cp;
            bool is_bad = bad.count(eid) > 0;
            cand_is_bad[ci] = is_bad ? 1 : 0;
            auto& targets = valid_p[eid];
            if (!is_bad && (int)targets.size() > 12) {
                cand_tgts[ci] = targets;
                std::shuffle(cand_tgts[ci].begin(), cand_tgts[ci].end(), rng);
                cand_tgts[ci].resize(12);
            } else {
                cand_tgts[ci] = targets;
            }
            cand_is_tabu[ci] = (tabu.count(tabu_key(eid, cp)) &&
                                tabu[tabu_key(eid, cp)] > it) ? 1 : 0;
        }

        int n_threads = 1;
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
        if (n_threads > 8) n_threads = 8;  // cap to avoid over-subscription with outer portfolio
#endif
        std::vector<double> local_bdelta(n_threads, 1e18);
        std::vector<int>    local_beid(n_threads, -1);
        std::vector<int>    local_bpid(n_threads, -1);
        std::vector<int>    local_brid(n_threads, -1);
        std::vector<uint8_t> local_imp(n_cand, 0);  // per-candidate: found any improving move

        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int ci = 0; ci < n_cand; ci++) {
            if (cand_skip[ci]) continue;
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            int eid = candidates[ci];
            int cp  = cand_cp[ci];
            bool is_tabu = cand_is_tabu[ci];
            bool found_any_improving = false;
            for (int pid : cand_tgts[ci]) {
                if (pid == cp) continue;
                for (int rid : valid_r[eid]) {
                    double d = Ecach.move_delta(sol, eid, pid, rid);
                    if (is_tabu && (current_fitness + d) >= best_fitness) continue;
                    if (d < 0) found_any_improving = true;
                    if (d < local_bdelta[tid]) {
                        local_bdelta[tid] = d;
                        local_beid[tid] = eid;
                        local_bpid[tid] = pid;
                        local_brid[tid] = rid;
                    }
                }
            }
            local_imp[ci] = found_any_improving ? 1 : 0;
        }

        // Reduce thread-local bests → global best
        for (int t = 0; t < n_threads; t++) {
            if (local_bdelta[t] < bdelta) {
                bdelta = local_bdelta[t];
                beid   = local_beid[t];
                bpid   = local_bpid[t];
                brid   = local_brid[t];
            }
        }
        // Update don't-look bits post-parallel
        for (int ci = 0; ci < n_cand; ci++) {
            if (!cand_skip[ci] && !local_imp[ci] && !cand_is_bad[ci])
                dont_look[candidates[ci]] = 1;
        }

        // ── Swap moves ──  (direct sol.assign → refresh cache)
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

                    // Score swap: d1 (ea→pb,ra), then hypothetically apply,
                    // score d2 (eb→pa,rb), then undo. Cache stays consistent
                    // because we route BOTH intermediate moves through Ecach.
                    double d1 = Ecach.move_delta(sol, ea, pb, ra);
                    Ecach.apply_move(sol, ea, pb, ra);
                    double d2 = Ecach.move_delta(sol, eb, pa, rb);
                    // Undo: route back through Ecach to restore cache
                    Ecach.apply_move(sol, ea, pa, ra);
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
                Ecach.apply_move(sol, sea, sepb, sera);
                Ecach.apply_move(sol, seb, sepa, serb);
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

        // ── Kempe chain ── mutates sol directly via kempe_detail; rebuild cache after
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
                    // Cache: all exams in chain + their neighbours need refresh
                    for (auto& u : undo_info) refresh_around(u.eid);
                    current_fitness += kd;
                    beid = -3;
                    break;
                } else {
                    kempe_detail::undo_chain(sol, undo_info);
                    // Cache stays in sync because kempe_detail::undo_chain
                    // restored state to what cache already reflects (no
                    // refresh needed).
                }
            }
        }

        if (beid >= 0) {
            int op = sol.period_of[beid];
            Ecach.apply_move(sol, beid, bpid, brid);
            current_fitness += bdelta;
            tabu[tabu_key(beid, op)] = it + active_tenure;
            visit_count[beid][bpid]++;
            clear_dont_look_around(beid);
        }

        // ── Ejection chain fallback ──
        // Trigger when swap + kempe both rejected AND we're stuck for a while.
        // Uses multi-depth chain from a bad exam to escape local optimum.
        if (beid < 0 && no_improve > 30 && !bad.empty() && no_improve % 10 == 0) {
            std::vector<int> bads(bad.begin(), bad.end());
            int seed_exam = bads[rng() % bads.size()];
            std::vector<ejection::Step> chain;
            double ejd = ejection::try_deep_chain(
                sol, Ecach, valid_p, valid_r,
                seed_exam, /*max_depth=*/5, /*samples_per_step=*/8,
                rng, chain, /*apply_on_improve=*/true, /*threshold=*/-0.5);
            if (!chain.empty()) {
                // Chain was applied (total delta < -0.5)
                current_fitness += ejd;
                for (auto& step : chain) {
                    tabu[tabu_key(step.eid, step.old_pid)] = it + active_tenure;
                    clear_dont_look_around(step.eid);
                }
                beid = -4;  // signal ejection applied
            }
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
                    std::cerr << "[TabuCached] Iter " << it << ": best hard=" << check.hard()
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
            std::fill(dont_look.begin(), dont_look.end(), 0);
            int osc_iters = 20 + (rng() % 30);
            for (int oi = 0; oi < osc_iters; oi++) {
                int e = de(rng);
                if (!valid_p[e].empty() && !valid_r[e].empty()) {
                    int p = valid_p[e][rng() % valid_p[e].size()];
                    int r = valid_r[e][rng() % valid_r[e].size()];
                    double d = Ecach.move_delta(sol, e, p, r);
                    if (d < 0) Ecach.apply_move(sol, e, p, r);
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
                Ecach.apply_move(sol, e, best_p, r);
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
        std::cerr << "[TabuCached] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "Tabu (Cached)"};
}
