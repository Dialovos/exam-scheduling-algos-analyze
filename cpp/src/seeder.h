/*
 * seeder.h — Universal starting-point generator.
 *
 * Every iterative algo in this project used to grab greedy's output and pray.
 * On "benign" instances (set1, set2, set4-8) that was fine; on set3 the greedy
 * walked out with hard > 0 and every downstream metaheuristic spent their
 * entire budget climbing out of the hole. feasibility.h was the duct-tape
 * answer: when greedy fails, run a dedicated hard-only solver first.
 *
 * This header generalises that pattern. It's a staircase: cheap at the bottom,
 * expensive at the top, and every step is one commitment to "do I really need
 * to climb higher?" Most instances fall off at Layer 1; the hard ones escalate.
 *
 *   Layer 1 — DSatur greedy   (fast, ~90% of the public instances clear here)
 *   Layer 2 — Carter LWD      (static conflict-weighted ordering, better on
 *                              dense graphs where saturation ties hide work)
 *   Layer 3 — Multi-start     (7 seeds through Layer 1 — brute the luck out)
 *   Layer 4 — Kempe repair    (surgical; swap periods around violations)
 *
 * Public API:
 *   solve_seeder(prob, seed, verbose) -> AlgoResult    (for --algo seeder)
 *   Seeder::seed(prob, seed, verbose) -> SeedResult    (for other algos to
 *                                                       consume as warm-start)
 *
 * The bouncer doesn't care what the algo does next. Its only job is: you do
 * NOT get past this door with hard_violations > 0 if a feasible start exists.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"
#include "repair.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

namespace seeder_detail {

// Layer 2 — Carter largest-weighted-degree static ordering.
// DSatur adapts priorities as it assigns; Carter commits to an ordering up
// front by total conflict weight (Σ shared_students across neighbours). On
// dense instances where many exams have identical saturation counts, the
// static tie-break from weighted-degree often unblocks the assignment.
inline Solution carter_lwd(
    const ProblemInstance& prob, const greedy_detail::GreedyCtx& ctx,
    std::mt19937& rng, FastEvaluator& fe)
{
    int ne = ctx.ne;
    Solution sol; sol.init(prob);

    // Weighted degree = Σ shared_students over all conflict neighbours.
    // prob.adj[e] already carries {neighbour, shared_count} pairs.
    std::vector<long long> wdeg(ne, 0);
    for (int e = 0; e < ne; e++)
        for (auto& [nb, w] : prob.adj[e]) wdeg[e] += w;

    std::vector<int> order(ne);
    std::iota(order.begin(), order.end(), 0);
    // Largest weighted degree first; random tie-break so re-runs explore.
    std::shuffle(order.begin(), order.end(), rng);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b){ return wdeg[a] > wdeg[b]; });

    std::vector<bool> assigned(ne, false);

    auto coin_ready_period = [&](int eid) -> int {
        // If any member of this exam's coincidence group is already placed,
        // we must use that period.
        for (int nb : ctx.coin_group[eid])
            if (nb != eid && assigned[nb]) return sol.period_of[nb];
        return -1;
    };

    auto feasible_period = [&](int eid, int pid) -> bool {
        if (ctx.exam_dur[eid] > ctx.period_dur[pid]) return false;
        for (int nb : ctx.adj_flat[eid])
            if (assigned[nb] && sol.period_of[nb] == pid) return false;
        for (int nb : ctx.exclusion_of[eid])
            if (assigned[nb] && sol.period_of[nb] == pid) return false;
        for (auto& [other, dir] : ctx.after_of[eid])
            if (assigned[other]) {
                int op = sol.period_of[other];
                if (dir == 0 && pid <= op) return false; // eid after other
                if (dir == 1 && pid >= op) return false; // eid before other
            }
        return true;
    };

    auto pick_room = [&](int eid, int pid, bool rhc) -> int {
        int best_r = -1, best_slack = INT32_MAX;
        for (int rid : ctx.valid_rooms[eid]) {
            int occ = sol.get_pr_enroll(pid, rid);
            int count = sol.get_pr_count(pid, rid);
            if (rhc && count > 0) continue;               // room_exclusive: solo
            int slack = ctx.room_cap[rid] - occ - ctx.exam_enr[eid];
            if (slack < 0) continue;
            if (slack < best_slack) { best_slack = slack; best_r = rid; }
        }
        return best_r;
    };

    // Room-exclusive exams get extra care; detect once.
    std::vector<bool> is_rhc(ne, false);
    for (int e : fe.rhc_exams) if (e < ne) is_rhc[e] = true;

    for (int eid : order) {
        if (assigned[eid]) continue;  // coincidence group fills siblings together

        int forced = coin_ready_period(eid);
        if (forced >= 0) {
            // Follow the group — this and all its unplaced siblings go here.
            for (int sib : ctx.coin_group[eid]) {
                if (assigned[sib]) continue;
                if (!feasible_period(sib, forced)) continue;  // group is doomed; skip
                int rid = pick_room(sib, forced, is_rhc[sib]);
                if (rid < 0) continue;
                sol.assign(sib, forced, rid);
                assigned[sib] = true;
            }
            continue;
        }

        // Pick earliest feasible period that admits the full coincidence group.
        int chosen_p = -1, chosen_r = -1;
        for (int pid : ctx.valid_periods[eid]) {
            if (!feasible_period(eid, pid)) continue;
            int rid = pick_room(eid, pid, is_rhc[eid]);
            if (rid < 0) continue;
            // Sanity: the rest of the coincidence group must also fit here.
            bool group_ok = true;
            for (int sib : ctx.coin_group[eid]) {
                if (sib == eid || assigned[sib]) continue;
                if (!feasible_period(sib, pid)) { group_ok = false; break; }
            }
            if (!group_ok) continue;
            chosen_p = pid; chosen_r = rid; break;
        }
        if (chosen_p < 0) continue;  // give up on this exam; final repair pass catches it

        sol.assign(eid, chosen_p, chosen_r);
        assigned[eid] = true;
        // Place coincidence siblings in same period.
        for (int sib : ctx.coin_group[eid]) {
            if (sib == eid || assigned[sib]) continue;
            int rid = pick_room(sib, chosen_p, is_rhc[sib]);
            if (rid < 0) continue;
            sol.assign(sib, chosen_p, rid);
            assigned[sib] = true;
        }
    }

    // Drop anything still unassigned into whatever period/room it fits —
    // may add hard violations, but that's the next layer's problem.
    for (int e = 0; e < ne; e++) {
        if (assigned[e]) continue;
        int bp = -1, br = -1, best_cost = INT32_MAX;
        for (int pid : ctx.valid_periods[e]) {
            for (int rid : ctx.valid_rooms[e]) {
                int cost = 0;
                for (int nb : ctx.adj_flat[e])
                    if (sol.period_of[nb] == pid) cost++;
                int slack = ctx.room_cap[rid] - sol.get_pr_enroll(pid, rid) - ctx.exam_enr[e];
                if (slack < 0) cost += 10;
                if (cost < best_cost) { best_cost = cost; bp = pid; br = rid; }
            }
        }
        if (bp >= 0) { sol.assign(e, bp, br); assigned[e] = true; }
    }

    return sol;
}

// Layer 4 now delegates to Repair::kempe_repair (see repair.h). Kept this
// alias for local readability — the call site below is cleaner with
// `seeder_detail::kempe_repair_final` than the full namespaced form.
inline Solution kempe_repair_final(
    const ProblemInstance& prob, Solution sol,
    int iters_budget = 8000, int restarts = 3, uint64_t rng_seed = 42)
{
    // First clean any orphaned room overflows — often the Kempe chain logic
    // can't see room-level issues because it only flips periods.
    Repair::fix_room_overflows(prob, sol);
    return Repair::kempe_repair(
        prob, std::move(sol), iters_budget, restarts, rng_seed);
}

// Pick the solution with the fewest hard violations (tie-break: lowest soft).
inline Solution pick_best(
    FastEvaluator& fe, std::vector<Solution>&& candidates)
{
    Solution* best = nullptr;
    double best_fit = 1e18;
    for (auto& s : candidates) {
        auto ev = fe.full_eval(s);
        double fit = (double)ev.hard() * 1e8 + (double)ev.soft();
        if (fit < best_fit) { best_fit = fit; best = &s; }
    }
    return best ? std::move(*best) : std::move(candidates[0]);
}

} // namespace seeder_detail

// ============================================================
//  Public API
// ============================================================

namespace Seeder {

struct Result {
    Solution sol;
    EvalResult eval;
    int layer_used = 0;   // 1..4
    double runtime_sec = 0.0;
};

// Entry point other algos will call once Task 6 lands. Until then `solve_seeder`
// (below) is how the seeder is exercised via the CLI / tests.
inline Result seed(const ProblemInstance& prob, int rng_seed = 42, bool verbose = false) {
    auto t0 = std::chrono::high_resolution_clock::now();

    greedy_detail::GreedyCtx ctx(prob);
    FastEvaluator fe(prob);
    std::mt19937 rng(rng_seed);

    auto finish = [&](Solution s, int layer) -> Result {
        auto ev = fe.full_eval(s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double rt = std::chrono::duration<double>(t1 - t0).count();
        if (verbose)
            std::cerr << "[Seeder] layer=" << layer
                      << " hard=" << ev.hard()
                      << " soft=" << ev.soft()
                      << " rt=" << rt << "s\n";
        return Result{std::move(s), ev, layer, rt};
    };

    // ── Layer 1 — DSatur ────────────────────────────────────
    // One shot. If this clears, skip the rest and go home.
    auto l1 = greedy_detail::solve_greedy_once(prob, ctx, rng_seed, verbose);
    if (l1.eval.feasible()) return finish(std::move(l1.sol), 1);

    // ── Layer 2 — Carter largest-weighted-degree ────────────
    auto l2_sol = seeder_detail::carter_lwd(prob, ctx, rng, fe);
    auto l2_ev = fe.full_eval(l2_sol);
    if (l2_ev.feasible()) return finish(std::move(l2_sol), 2);

    // ── Layer 3 — Multi-start DSatur (7 seeds, keep best) ───
    std::vector<Solution> candidates;
    candidates.push_back(std::move(l1.sol));
    candidates.push_back(std::move(l2_sol));
    for (int s = 0; s < 7; s++) {
        int trial_seed = (int)(rng() & 0x7fffffff);
        auto r = greedy_detail::solve_greedy_once(prob, ctx, trial_seed, false);
        if (r.eval.feasible()) return finish(std::move(r.sol), 3);
        candidates.push_back(std::move(r.sol));
    }
    Solution best = seeder_detail::pick_best(fe, std::move(candidates));
    auto best_ev = fe.full_eval(best);
    if (best_ev.feasible()) return finish(std::move(best), 3);

    // ── Layer 4 — Kempe repair on the least-bad candidate ────
    // `seeder_layer=4` means we had to call the surgeon. Try a few RNG seeds
    // inside the bounded-budget repair — empirically set3 shows ~20% of
    // seeds hit dead-ends where a fresh one closes cleanly. Cheap insurance.
    Solution best_repaired = best.copy();
    int best_repaired_hard = fe.full_eval(best_repaired).hard();
    for (int attempt = 0; attempt < 4 && best_repaired_hard > 0; attempt++) {
        uint64_t sub_seed = (uint64_t)rng_seed * 31ull + (uint64_t)attempt * 1009ull;
        Solution trial = seeder_detail::kempe_repair_final(
            prob, best.copy(),
            /*iters_budget=*/8000, /*restarts=*/3, /*rng_seed=*/sub_seed);
        int h = fe.full_eval(trial).hard();
        if (h < best_repaired_hard) {
            best_repaired_hard = h;
            best_repaired = std::move(trial);
            if (best_repaired_hard == 0) break;
        }
    }
    return finish(std::move(best_repaired), 4);
}

} // namespace Seeder

// Drop-in equivalent to solve_greedy/solve_tabu/etc so main.cpp can dispatch
// it from the normal `if (want("seeder")) { ... }` block. AlgoResult.iterations
// is repurposed here to carry the layer number — it's the only free slot on
// the struct and the semantics ("how far did we go?") are close enough.
inline AlgoResult solve_seeder(
    const ProblemInstance& prob, int seed = 42, bool verbose = false)
{
    auto r = Seeder::seed(prob, seed, verbose);
    AlgoResult out;
    out.sol = std::move(r.sol);
    out.eval = r.eval;
    out.runtime_sec = r.runtime_sec;
    out.iterations = r.layer_used;   // abuse of field; see write_result_json
    out.algorithm = "Seeder";
    return out;
}
