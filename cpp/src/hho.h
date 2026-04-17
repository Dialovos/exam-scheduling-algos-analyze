/*
 * hho.h — Harris Hawks Optimization, discrete-lattice "Plus" variant.
 *
 * Classical HHO (Heidari et al. 2019) was designed for continuous spaces:
 * hawks move in R^n toward a prey position, switching between exploration
 * and four exploitation modes driven by the prey's "escape energy" E.
 * For exam timetabling the decision space is discrete, so we reinterpret
 * the movement primitives:
 *
 *   Exploration (|E| ≥ 1):         Lévy-ish k-period-swap perturbation
 *   Soft besiege (r<0.5, |E|≥0.5): steepest 1-move toward prey's period
 *   Hard besiege (r<0.5, |E|<0.5): steepest + k-move chain toward prey
 *   Soft + dive  (r≥0.5, |E|≥0.5): Kempe chain swap near prey
 *   Hard + dive  (r≥0.5, |E|<0.5): Kempe chain + best-fit room re-pack
 *
 * The "Plus" is twofold:
 *   1) Seeder warm-start for hawk[0] (the leader) + random for the rest;
 *   2) Periodic Kempe-polish of the prey — every HHO paper on continuous
 *      spaces gets away without this because their landscape is smooth, but
 *      ours is combinatorial and the prey stalls without an intensification
 *      kick. The polish is cheap: one pass of Repair::kempe_repair at hard
 *      budget when already feasible, else a soft-delta-only Kempe sweep.
 *
 * Was in the roster, got cut for runtime, now re-entering the ring.
 * Still not expected to dethrone SA — but it gives a very different
 * exploration/exploitation balance and is worth having for the ablation.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "neighbourhoods.h"
#include "seeder.h"
#include "repair.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace hho_detail {

struct Hawk {
    Solution sol;
    double fitness = 1e18;
    bool feasible = false;
};

// Diversified random solution. Same pattern as woa_detail::random_solution —
// duplicated here instead of cross-including because WOA's random_solution
// is privately scoped and I don't want to force it public just for HHO+.
inline Solution random_solution(
    const ProblemInstance& prob, const FastEvaluator& fe,
    int ne, int np, int /*nr*/,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    std::mt19937& rng)
{
    Solution sol; sol.init(prob);
    std::vector<int> order(ne);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    for (int eid : order) {
        std::vector<bool> blocked(np, false);
        for (auto& [nb, _] : prob.adj[eid])
            if (sol.period_of[nb] >= 0) blocked[sol.period_of[nb]] = true;

        std::vector<int> avail;
        for (int p : valid_p[eid]) if (!blocked[p]) avail.push_back(p);
        if (avail.empty()) avail = valid_p[eid];
        if (avail.empty()) for (int p = 0; p < np; p++) avail.push_back(p);
        std::shuffle(avail.begin(), avail.end(), rng);

        bool placed = false;
        for (int pid : avail) {
            for (int rid : valid_r[eid]) {
                if (sol.get_pr_enroll(pid, rid) + fe.exam_enroll[eid] <= fe.room_cap[rid]) {
                    sol.assign(eid, pid, rid);
                    placed = true; break;
                }
            }
            if (placed) break;
        }
        if (!placed) {
            int pid = avail[0];
            int rid = valid_r[eid].empty() ? 0 : valid_r[eid][rng() % valid_r[eid].size()];
            sol.assign(eid, pid, rid);
        }
    }
    return sol;
}

// Pick best room for (eid, pid) among those with capacity and no rhc clash.
inline int best_room_for(
    const Solution& sol, const FastEvaluator& fe,
    int eid, int pid,
    const std::vector<std::vector<int>>& valid_r,
    bool is_rhc)
{
    int best_r = -1, best_slack = -1;
    int enr = fe.exam_enroll[eid];
    for (int rid : valid_r[eid]) {
        if (is_rhc && sol.get_pr_count(pid, rid) > 0) continue;
        int slack = fe.room_cap[rid] - sol.get_pr_enroll(pid, rid);
        if (slack >= enr && slack > best_slack) { best_slack = slack; best_r = rid; }
    }
    return best_r >= 0 ? best_r : (valid_r[eid].empty() ? 0 : valid_r[eid].front());
}

// Lévy-ish perturbation: k random single-exam period moves. HHO's original
// Lévy flight samples from a heavy-tailed distribution; the discrete analog
// is "most moves are small, occasionally a big jump" — implemented as a
// burst of k uniform re-assignments drawn from a geometric-ish distribution.
inline void levy_perturb(
    Solution& sol, FastEvaluator& fe,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    const std::vector<bool>& is_rhc,
    std::mt19937& rng, int k)
{
    std::uniform_int_distribution<int> de(0, fe.ne - 1);
    for (int i = 0; i < k; i++) {
        int eid = de(rng);
        if (valid_p[eid].empty()) continue;
        int pid = valid_p[eid][rng() % valid_p[eid].size()];
        int rid = best_room_for(sol, fe, eid, pid, valid_r, is_rhc[eid]);
        fe.apply_move(sol, eid, pid, rid);
    }
}

// Steepest single-move descent: find the move that most reduces fitness
// (hard*1e8 + soft), apply it, return new fitness. No change if none found.
inline double steepest_single_move(
    Solution& sol, FastEvaluator& fe, double cur_fit,
    int ne,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    const std::vector<bool>& is_rhc,
    std::mt19937& rng, int sample_size)
{
    std::uniform_int_distribution<int> de(0, ne - 1);
    int best_eid = -1, best_pid = -1, best_rid = -1;
    double best_delta = 0;

    // Sample `sample_size` exams rather than scan all ne — HHO runs many
    // iterations, we don't need an exhaustive sweep each call.
    for (int i = 0; i < sample_size; i++) {
        int eid = de(rng);
        if (valid_p[eid].empty() || valid_r[eid].empty()) continue;
        int cur_p = sol.period_of[eid];
        for (int pid : valid_p[eid]) {
            if (pid == cur_p) continue;
            int rid = best_room_for(sol, fe, eid, pid, valid_r, is_rhc[eid]);
            double d = fe.move_delta(sol, eid, pid, rid);
            if (d < best_delta) {
                best_delta = d; best_eid = eid; best_pid = pid; best_rid = rid;
            }
        }
    }
    if (best_eid >= 0 && best_delta < 0) {
        fe.apply_move(sol, best_eid, best_pid, best_rid);
        return cur_fit + best_delta;
    }
    return cur_fit;
}

// Move exams in `hawk` toward the prey's period assignments. Picks `k` exams
// whose period differs from prey's and moves them (with best-fit room). If
// the resulting state is worse than `hawk_fit`, roll back.
inline double move_toward_prey(
    Solution& hawk, const Solution& prey, double hawk_fit,
    FastEvaluator& fe, int ne,
    const std::vector<std::vector<int>>& valid_r,
    const std::vector<bool>& is_rhc,
    std::mt19937& rng, int k)
{
    std::vector<int> diff;
    diff.reserve(ne);
    for (int e = 0; e < ne; e++)
        if (hawk.period_of[e] != prey.period_of[e]) diff.push_back(e);
    if (diff.empty()) return hawk_fit;

    std::shuffle(diff.begin(), diff.end(), rng);
    k = std::min(k, (int)diff.size());

    // Snapshot moved exams so we can roll back if fitness rises.
    std::vector<std::tuple<int,int,int>> snap;
    snap.reserve(k);
    for (int i = 0; i < k; i++) {
        int eid = diff[i];
        snap.emplace_back(eid, hawk.period_of[eid], hawk.room_of[eid]);
        int target_p = prey.period_of[eid];
        int rid = best_room_for(hawk, fe, eid, target_p, valid_r, is_rhc[eid]);
        fe.apply_move(hawk, eid, target_p, rid);
    }
    auto ev = fe.full_eval(hawk);
    double new_fit = ev.fitness();
    if (new_fit < hawk_fit) return new_fit;

    // Roll back (reverse order to avoid intermediate phantom conflicts).
    for (auto it = snap.rbegin(); it != snap.rend(); ++it) {
        auto& [eid, op, orv] = *it;
        fe.apply_move(hawk, eid, op, orv);
    }
    return hawk_fit;
}

// Kempe chain swap: pick a random exam, try a low-conflict target period,
// accept if hard-or-fitness improves.
inline double kempe_dive(
    Solution& sol, FastEvaluator& fe, const ProblemInstance& prob,
    double cur_fit, int ne, int np, std::mt19937& rng)
{
    std::uniform_int_distribution<int> dp(0, np - 1);
    std::uniform_int_distribution<int> de(0, ne - 1);

    int eid = de(rng);
    int p1 = sol.period_of[eid];
    if (p1 < 0) return cur_fit;

    int p2 = dp(rng);
    if (p2 == p1) p2 = (p2 + 1) % np;

    auto chain = kempe_detail::build_chain(sol, prob.adj, ne, eid, p1, p2);
    if (chain.empty() || (int)chain.size() > ne / 3) return cur_fit;

    auto undo = kempe_detail::apply_chain(sol, chain, p1, p2);
    auto ev = fe.full_eval(sol);
    double nf = ev.fitness();
    if (nf < cur_fit) return nf;
    kempe_detail::undo_chain(sol, undo);
    return cur_fit;
}

// Feasibility-first fitness comparator: feasible beats infeasible regardless
// of raw fitness value, since soft-penalty arithmetic isn't meaningful until
// hard=0.
inline bool hawk_better(const Hawk& a, const Hawk& b) {
    if (a.feasible && !b.feasible) return true;
    if (!a.feasible && b.feasible) return false;
    return a.fitness < b.fitness;
}

} // namespace hho_detail

inline AlgoResult solve_hho(
    const ProblemInstance& prob,
    int pop_size         = 20,
    int max_iterations   = 500,
    int seed             = 42,
    bool verbose         = false,
    const Solution* init_sol = nullptr)
{
    using namespace hho_detail;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    pop_size = std::max(pop_size, 5);
    FastEvaluator fe(prob);

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (fe.exam_dur[e] <= fe.period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (fe.exam_enroll[e] <= fe.room_cap[r]) valid_r[e].push_back(r);
    }
    std::vector<bool> is_rhc(ne, false);
    for (int e : fe.rhc_exams) if (e < ne) is_rhc[e] = true;

    // ── Population init ────────────────────────────────────
    // Hawk 0 = warm start (init_sol if given, else Seeder). Others = random.
    std::vector<Hawk> pop(pop_size);
    if (init_sol) pop[0].sol = init_sol->copy();
    else          pop[0].sol = Seeder::seed(prob, seed, false).sol;

    {
        auto ev = fe.full_eval(pop[0].sol);
        pop[0].fitness = ev.fitness();
        pop[0].feasible = ev.feasible();
    }
    for (int i = 1; i < pop_size; i++) {
        pop[i].sol = random_solution(prob, fe, ne, np, nr, valid_p, valid_r, rng);
        auto ev = fe.full_eval(pop[i].sol);
        pop[i].fitness = ev.fitness();
        pop[i].feasible = ev.feasible();
    }

    // Find initial prey (best hawk).
    int prey_idx = 0;
    for (int i = 1; i < pop_size; i++)
        if (hawk_better(pop[i], pop[prey_idx])) prey_idx = i;

    Solution best_sol = pop[prey_idx].sol.copy();
    double best_fit = pop[prey_idx].fitness;
    bool best_feasible = pop[prey_idx].feasible;

    if (verbose) {
        auto ev = fe.full_eval(best_sol);
        std::cerr << "[HHO+] Init pop=" << pop_size
                  << " prey hard=" << ev.hard() << " soft=" << ev.soft() << "\n";
    }

    std::uniform_real_distribution<double> unif(-1.0, 1.0);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    const int KEMPE_POLISH_EVERY = std::max(25, max_iterations / 20);
    const int SAMPLE_SIZE        = std::min(ne, std::max(10, ne / 6));

    int total_iters = 0;

    for (int t = 0; t < max_iterations; t++) {
        total_iters++;

        // Prey decays its escape energy; all hawks share the same t-schedule.
        double E0 = unif(rng);                        // (-1, 1)
        double E  = 2.0 * E0 * (1.0 - (double)t / (double)max_iterations);

        for (int i = 0; i < pop_size; i++) {
            if (i == prey_idx) continue;              // prey doesn't move

            Hawk& h = pop[i];
            const Solution& prey = pop[prey_idx].sol;
            double before = h.fitness;

            if (std::abs(E) >= 1.0) {
                // ── Exploration — Lévy-style burst ───────────
                int k = std::max(2, (int)std::ceil(std::abs(E) * 5.0));
                levy_perturb(h.sol, fe, valid_p, valid_r, is_rhc, rng, k);
            } else {
                double r = unif01(rng);
                if (r < 0.5 && std::abs(E) >= 0.5) {
                    // Soft besiege — single-move descent
                    h.fitness = steepest_single_move(
                        h.sol, fe, h.fitness, ne, valid_p, valid_r, is_rhc,
                        rng, SAMPLE_SIZE);
                } else if (r < 0.5 && std::abs(E) < 0.5) {
                    // Hard besiege — converge k moves toward prey
                    int k = std::max(3, ne / 40);
                    h.fitness = move_toward_prey(
                        h.sol, prey, h.fitness, fe, ne, valid_r, is_rhc, rng, k);
                } else if (r >= 0.5 && std::abs(E) >= 0.5) {
                    // Soft besiege with progressive rapid dive — Kempe chain
                    h.fitness = kempe_dive(h.sol, fe, prob, h.fitness, ne, np, rng);
                } else {
                    // Hard besiege with rapid dive — Kempe + toward-prey mini chain
                    h.fitness = kempe_dive(h.sol, fe, prob, h.fitness, ne, np, rng);
                    int k = std::max(2, ne / 60);
                    h.fitness = move_toward_prey(
                        h.sol, prey, h.fitness, fe, ne, valid_r, is_rhc, rng, k);
                }
            }

            // Resync fitness/feasible (levy_perturb doesn't maintain them).
            if (h.fitness == before || std::abs(E) >= 1.0) {
                auto ev = fe.full_eval(h.sol);
                h.fitness = ev.fitness();
                h.feasible = ev.feasible();
            } else {
                // Hack: if we accepted a move, we updated fitness but not feasible.
                // Recompute cheaply via a fast hard-count — soft is already in fit.
                int hard = fe.count_hard_fast(h.sol);
                h.feasible = (hard == 0);
            }
        }

        // ── Update prey ─────────────────────────────────────
        int new_prey = prey_idx;
        for (int i = 0; i < pop_size; i++)
            if (hawk_better(pop[i], pop[new_prey])) new_prey = i;
        if (new_prey != prey_idx) {
            prey_idx = new_prey;
            if (hawk_better(pop[prey_idx], {best_sol.copy(), best_fit, best_feasible})) {
                best_sol = pop[prey_idx].sol.copy();
                best_fit = pop[prey_idx].fitness;
                best_feasible = pop[prey_idx].feasible;
                if (verbose)
                    std::cerr << "[HHO+] t=" << t
                              << " new prey hard=" << fe.full_eval(best_sol).hard()
                              << " soft=" << fe.full_eval(best_sol).soft() << "\n";
            }
        }

        // ── Periodic Kempe polish on the prey ───────────────
        // This is the "+" in HHO+: intensification kick the continuous-space
        // paper doesn't need. Keeps the prey from stagnating mid-search.
        if (t > 0 && t % KEMPE_POLISH_EVERY == 0) {
            Solution polished = pop[prey_idx].sol.copy();
            if (!pop[prey_idx].feasible) {
                // If still infeasible, use Repair's bounded kempe_repair.
                polished = Repair::kempe_repair(
                    prob, std::move(polished), /*iters=*/2000, /*restarts=*/1,
                    (uint64_t)seed * 31ull + (uint64_t)t);
            } else {
                // Feasible: soft-improving Kempe sweep.
                std::uniform_int_distribution<int> dp(0, np - 1);
                double cur = fe.full_eval(polished).fitness();
                for (int attempt = 0; attempt < 20; attempt++) {
                    cur = kempe_dive(polished, fe, prob, cur, ne, np, rng);
                }
            }
            auto ev = fe.full_eval(polished);
            double fp = ev.fitness();
            bool feasp = ev.feasible();
            // Accept only if strictly better (feasibility-first).
            Hawk cand{polished.copy(), fp, feasp};
            if (hawk_better(cand, pop[prey_idx])) {
                pop[prey_idx].sol = std::move(polished);
                pop[prey_idx].fitness = fp;
                pop[prey_idx].feasible = feasp;
                if (hawk_better(pop[prey_idx], {best_sol.copy(), best_fit, best_feasible})) {
                    best_sol = pop[prey_idx].sol.copy();
                    best_fit = fp;
                    best_feasible = feasp;
                }
            }
        }
    }

    // Final eval & return.
    auto final_ev = fe.full_eval(best_sol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();

    if (verbose)
        std::cerr << "[HHO+] " << rt << "s iters=" << total_iters
                  << " feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << "\n";

    return {std::move(best_sol), final_ev, rt, total_iters, "HHO+"};
}
