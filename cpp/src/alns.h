/*
 * Adaptive large neighborhood search (ALNS).
 *
 * Destroy-and-repair with adaptive operator selection.
 * 3 destroy operators (random, worst, related) + 2 repair (greedy, random).
 * SA-like acceptance criterion.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

namespace alns_detail {

// ── Helpers ────────────────────────────────────────────────────

inline int roulette(const std::vector<double>& weights, std::mt19937& rng) {
    double total = 0;
    for (double w : weights) total += w;
    std::uniform_real_distribution<double> d(0, total);
    double r = d(rng);
    double acc = 0;
    for (int i = 0; i < (int)weights.size(); i++) {
        acc += weights[i];
        if (r <= acc) return i;
    }
    return (int)weights.size() - 1;
}

struct SavedState {
    std::vector<int> period_of;
    std::vector<int> room_of;
    std::vector<int> pr_enroll;
    std::vector<int> pr_count;
};

inline SavedState save_state(const Solution& sol) {
    return {sol.period_of, sol.room_of, sol.pr_enroll, sol.pr_count};
}

inline void restore_state(Solution& sol, const SavedState& s) {
    sol.period_of = s.period_of;
    sol.room_of = s.room_of;
    sol.pr_enroll = s.pr_enroll;
    sol.pr_count = s.pr_count;
}

inline void unassign(Solution& sol, int eid) {
    int pid = sol.period_of[eid];
    if (pid >= 0) {
        int rid = sol.room_of[eid];
        int key = sol.pr_key(pid, rid);
        sol.pr_enroll[key] -= sol.enroll_cache[eid];
        sol.pr_count[key]--;
        sol.period_of[eid] = -1;
        sol.room_of[eid] = -1;
    }
}


// ── Destroy operators ──────────────────────────────────────────

inline std::vector<int> destroy_random(
    Solution& sol, int ne, int n_destroy, std::mt19937& rng)
{
    std::vector<int> exams(ne);
    std::iota(exams.begin(), exams.end(), 0);
    std::shuffle(exams.begin(), exams.end(), rng);

    std::vector<int> removed;
    for (int e : exams) {
        if ((int)removed.size() >= n_destroy) break;
        if (sol.period_of[e] >= 0) {
            unassign(sol, e);
            removed.push_back(e);
        }
    }
    return removed;
}

inline std::vector<int> destroy_worst(
    Solution& sol, const FastEvaluator& fe, int ne, int n_destroy, std::mt19937& rng)
{
    std::vector<std::pair<int, int>> costs;
    for (int e = 0; e < ne; e++) {
        int pid = sol.period_of[e];
        if (pid < 0) continue;
        int cost = 0;
        for (auto& [nb, _] : fe.P.adj[e]) {
            int nb_pid = sol.period_of[nb];
            if (nb_pid == pid)
                cost += 100000;
            else if (nb_pid >= 0) {
                if (fe.period_day[pid] == fe.period_day[nb_pid]) {
                    int g = std::abs(fe.period_daypos[pid] - fe.period_daypos[nb_pid]);
                    if (g == 1) cost += fe.w_2row;
                    else if (g > 1) cost += fe.w_2day;
                }
                int gap = std::abs(pid - nb_pid);
                if (gap > 0 && gap <= fe.w_spread) cost += 1;
            }
        }
        cost += fe.period_pen[pid] + fe.room_pen[sol.room_of[e]];
        costs.push_back({cost, e});
    }
    std::sort(costs.begin(), costs.end(), [](auto& a, auto& b) { return a.first > b.first; });

    int pool_size = std::min((int)costs.size(), n_destroy * 2);
    std::vector<std::pair<int,int>> pool(costs.begin(), costs.begin() + pool_size);
    std::shuffle(pool.begin(), pool.end(), rng);

    std::vector<int> removed;
    for (auto& [_, e] : pool) {
        if ((int)removed.size() >= n_destroy) break;
        if (sol.period_of[e] >= 0) {
            unassign(sol, e);
            removed.push_back(e);
        }
    }
    return removed;
}

inline std::vector<int> destroy_related(
    Solution& sol, const ProblemInstance& prob, int ne, int n_destroy, std::mt19937& rng)
{
    std::uniform_int_distribution<int> de(0, ne - 1);
    int start = de(rng);

    std::vector<int> removed;
    std::vector<bool> visited(ne, false);
    std::queue<int> q;
    q.push(start);
    visited[start] = true;

    while (!q.empty() && (int)removed.size() < n_destroy) {
        int e = q.front(); q.pop();
        if (sol.period_of[e] >= 0) {
            unassign(sol, e);
            removed.push_back(e);
        }
        std::vector<int> neighbors;
        for (auto& [nb, _] : prob.adj[e])
            if (!visited[nb]) neighbors.push_back(nb);
        std::shuffle(neighbors.begin(), neighbors.end(), rng);
        for (int nb : neighbors) {
            if (!visited[nb]) {
                visited[nb] = true;
                q.push(nb);
            }
        }
    }
    return removed;
}

// Shaw removal: destroy by relatedness (shared students + proximity + enrollment)
inline std::vector<int> destroy_shaw(
    Solution& sol, const ProblemInstance& prob, int ne, int n_destroy, std::mt19937& rng)
{
    std::uniform_int_distribution<int> de(0, ne - 1);
    int seed = de(rng);
    while (sol.period_of[seed] < 0 && n_destroy > 0) seed = de(rng);
    int seed_pid = sol.period_of[seed];

    // Compute relatedness to seed for all assigned exams
    std::vector<std::pair<double, int>> relatedness;
    for (int e = 0; e < ne; e++) {
        if (e == seed || sol.period_of[e] < 0) continue;
        double r = 0;
        for (auto& [nb, shared] : prob.adj[seed])
            if (nb == e) { r += shared * 10.0; break; }
        int pdiff = std::abs(sol.period_of[e] - seed_pid);
        if (pdiff == 0) r += 5.0;
        else if (pdiff <= 3) r += 3.0;
        int ediff = std::abs(prob.exams[e].enrollment() - prob.exams[seed].enrollment());
        if (ediff < 20) r += 2.0;
        relatedness.push_back({r, e});
    }
    std::sort(relatedness.begin(), relatedness.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    std::vector<int> removed;
    unassign(sol, seed);
    removed.push_back(seed);
    for (auto& [_, e] : relatedness) {
        if ((int)removed.size() >= n_destroy) break;
        if (sol.period_of[e] >= 0) { unassign(sol, e); removed.push_back(e); }
    }
    return removed;
}

// Period-strip removal: remove all exams in a random period
inline std::vector<int> destroy_period_strip(
    Solution& sol, int ne, int np, int n_destroy, std::mt19937& rng)
{
    std::uniform_int_distribution<int> dp(0, np - 1);
    int target_p = dp(rng);

    std::vector<int> in_period;
    for (int e = 0; e < ne; e++)
        if (sol.period_of[e] == target_p) in_period.push_back(e);

    std::shuffle(in_period.begin(), in_period.end(), rng);
    std::vector<int> removed;
    for (int e : in_period) {
        if ((int)removed.size() >= n_destroy) break;
        unassign(sol, e);
        removed.push_back(e);
    }
    return removed;
}

// ── Repair operators ───────────────────────────────────────────

inline void repair_greedy(
    Solution& sol, const FastEvaluator& fe,
    const std::vector<int>& removed,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r)
{
    std::vector<int> order = removed;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return fe.P.adj[a].size() > fe.P.adj[b].size();
    });

    for (int eid : order) {
        const auto& vp = valid_p[eid];
        const auto& vr = valid_r[eid];
        if (vp.empty() || vr.empty()) {
            if (!vp.empty() && !vr.empty()) sol.assign(eid, vp[0], vr[0]);
            continue;
        }

        int best_pid = -1, best_rid = -1;
        long long best_cost = (long long)1e18;

        // Period-first scan: compute adjacency cost ONCE per period,
        // then add cheap room cost. Avoids redundant adjacency scans.
        for (int pid : vp) {
            long long pcost = 0;
            for (auto& [nb, _] : fe.P.adj[eid]) {
                int nb_pid = sol.period_of[nb];
                if (nb_pid < 0) continue;
                if (nb_pid == pid) pcost += 100000;
                else {
                    int gap = std::abs(pid - nb_pid);
                    if (gap > 0 && gap <= fe.w_spread) pcost += 1;
                    if (fe.period_day[pid] == fe.period_day[nb_pid]) {
                        int g = std::abs(fe.period_daypos[pid] - fe.period_daypos[nb_pid]);
                        if (g == 1) pcost += fe.w_2row;
                        else if (g > 1) pcost += fe.w_2day;
                    }
                }
            }
            pcost += fe.period_pen[pid];
            if (fe.large_exams.count(eid) && fe.fl_penalty > 0 && fe.last_periods.count(pid))
                pcost += fe.fl_penalty;

            // Room scan: only capacity + penalty (no adjacency)
            for (int rid : vr) {
                long long cost = pcost;
                if (sol.get_pr_enroll(pid, rid) + fe.exam_enroll[eid] > fe.room_cap[rid])
                    cost += 100000;
                cost += fe.room_pen[rid];
                if (cost < best_cost) {
                    best_cost = cost;
                    best_pid = pid;
                    best_rid = rid;
                }
            }
        }

        if (best_pid >= 0)
            sol.assign(eid, best_pid, best_rid);
        else
            sol.assign(eid, vp[0], vr[0]);
    }
}

// Batched-scoring variant of repair_greedy. Identical semantics (same sort,
// same cost formula), but routes all (pid, rid) scoring through a
// `ScorerT` that exposes `score_placement_batch(eids, pids, rids, costs)`.
// On GPU-enabled builds, this becomes one kernel launch per unplaced exam
// covering np × nr candidates; on CPU, it's equivalent to repair_greedy
// (verified bit-exact in `make bench` placement-scorer validator).
template <typename ScorerT>
inline void repair_greedy_batched(
    Solution& sol, const FastEvaluator& fe,
    ScorerT& scorer,            // must have sync_state(sol) + score_placement_batch
    const std::vector<int>& removed,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r)
{
    std::vector<int> order = removed;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return fe.P.adj[a].size() > fe.P.adj[b].size();
    });

    // Scratch buffers reused across all placements
    std::vector<int32_t> mv_eid, mv_pid, mv_rid;
    std::vector<long long> costs;
    mv_eid.reserve(2048); mv_pid.reserve(2048); mv_rid.reserve(2048);

    for (int eid : order) {
        const auto& vp = valid_p[eid];
        const auto& vr = valid_r[eid];
        if (vp.empty() || vr.empty()) {
            if (!vp.empty() && !vr.empty()) sol.assign(eid, vp[0], vr[0]);
            continue;
        }

        // Enumerate all (pid, rid) candidates for this exam
        mv_eid.clear(); mv_pid.clear(); mv_rid.clear();
        for (int pid : vp)
            for (int rid : vr) {
                mv_eid.push_back(eid);
                mv_pid.push_back(pid);
                mv_rid.push_back(rid);
            }

        // State has changed since last placement (previous exam was assigned)
        // so re-sync before scoring.
        scorer.sync_state(sol);
        scorer.score_placement_batch(mv_eid, mv_pid, mv_rid, costs);

        long long best_cost = (long long)1e18;
        int best_pid = -1, best_rid = -1;
        for (size_t k = 0; k < costs.size(); k++) {
            if (costs[k] < best_cost) {
                best_cost = costs[k];
                best_pid = mv_pid[k];
                best_rid = mv_rid[k];
            }
        }
        if (best_pid >= 0)
            sol.assign(eid, best_pid, best_rid);
        else
            sol.assign(eid, vp[0], vr[0]);
    }
}

inline void repair_random(
    Solution& sol,
    const std::vector<int>& removed,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r,
    std::mt19937& rng)
{
    for (int eid : removed) {
        const auto& vp = valid_p[eid];
        const auto& vr = valid_r[eid];
        if (!vp.empty() && !vr.empty())
            sol.assign(eid, vp[rng() % vp.size()], vr[rng() % vr.size()]);
        else
            sol.assign(eid, 0, 0);
    }
}

// Regret-2 repair: place exam with highest regret (best - 2nd best cost gap) first
inline void repair_regret2(
    Solution& sol, const FastEvaluator& fe,
    const std::vector<int>& removed,
    const std::vector<std::vector<int>>& valid_p,
    const std::vector<std::vector<int>>& valid_r)
{
    std::vector<int> unplaced = removed;

    while (!unplaced.empty()) {
        int max_regret_exam = -1;
        double max_regret = -1e18;
        int best_pid = -1, best_rid = -1;

        for (int eid : unplaced) {
            double best_cost = 1e18, second_cost = 1e18;
            int bp = -1, br = -1;

            for (int pid : valid_p[eid]) {
                // Period adjacency cost (once per period)
                long long pcost = 0;
                for (auto& [nb, _] : fe.P.adj[eid]) {
                    int nb_pid = sol.period_of[nb]; if (nb_pid < 0) continue;
                    if (nb_pid == pid) pcost += 100000;
                    else {
                        int gap = std::abs(pid - nb_pid);
                        if (gap > 0 && gap <= fe.w_spread) pcost += 1;
                        if (fe.period_day[pid] == fe.period_day[nb_pid]) {
                            int g = std::abs(fe.period_daypos[pid] - fe.period_daypos[nb_pid]);
                            if (g == 1) pcost += fe.w_2row;
                            else if (g > 1) pcost += fe.w_2day;
                        }
                    }
                }
                pcost += fe.period_pen[pid];
                if (fe.large_exams.count(eid) && fe.fl_penalty > 0 && fe.last_periods.count(pid))
                    pcost += fe.fl_penalty;

                for (int rid : valid_r[eid]) {
                    long long cost = pcost;
                    if (sol.get_pr_enroll(pid, rid) + fe.exam_enroll[eid] > fe.room_cap[rid])
                        cost += 100000;
                    cost += fe.room_pen[rid];
                    if (cost < best_cost) {
                        second_cost = best_cost;
                        best_cost = cost; bp = pid; br = rid;
                    } else if (cost < second_cost) {
                        second_cost = cost;
                    }
                }
            }

            double regret = second_cost - best_cost;
            if (regret > max_regret) {
                max_regret = regret; max_regret_exam = eid;
                best_pid = bp; best_rid = br;
            }
        }

        if (max_regret_exam >= 0 && best_pid >= 0) {
            sol.assign(max_regret_exam, best_pid, best_rid);
            unplaced.erase(std::remove(unplaced.begin(), unplaced.end(), max_regret_exam), unplaced.end());
        } else {
            break;
        }
    }

    // Fallback for any remaining unplaced
    if (!unplaced.empty())
        repair_greedy(sol, fe, unplaced, valid_p, valid_r);
}

} // namespace alns_detail

// ── Public interface ───────────────────────────────────────────

inline AlgoResult solve_alns(
    const ProblemInstance& prob,
    int max_iterations  = 2000,
    double destroy_pct  = 0.04,
    int seed            = 42,
    bool verbose        = false,
    const Solution* init_sol = nullptr)
{
    using namespace alns_detail;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    FastEvaluator fe(prob);

    std::vector<int> exam_dur(ne), exam_enr(ne), period_dur(np), room_cap(nr);
    for (auto& e : prob.exams) { exam_dur[e.id] = e.duration; exam_enr[e.id] = e.enrollment(); }
    for (auto& p : prob.periods) period_dur[p.id] = p.duration;
    for (auto& r : prob.rooms) room_cap[r.id] = r.capacity;

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (exam_dur[e] <= period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (exam_enr[e] <= room_cap[r]) valid_r[e].push_back(r);
    }

    // Init from greedy
    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); sol = g.sol.copy(); }

    EvalResult ev = fe.full_eval(sol);

    // Feasibility recovery if greedy started infeasible
    if (!ev.feasible()) {
        if (verbose)
            std::cerr << "[ALNS] Greedy infeasible (hard=" << ev.hard()
                      << "), running recovery..." << std::endl;
        fe.recover_feasibility(sol, 500, seed);
        ev = fe.full_eval(sol);
        if (verbose)
            std::cerr << "[ALNS] After recovery: feasible=" << ev.feasible()
                      << " hard=" << ev.hard() << " soft=" << ev.soft() << std::endl;
    }

    double current_fitness = ev.fitness();
    Solution best_sol = sol.copy();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();

    int n_destroy_base = std::max(1, (int)(ne * destroy_pct));
    int n_destroy = n_destroy_base;
    int no_improve_alns = 0;

    // SA acceptance — calibrate from random soft-only move deltas
    double temp;
    {
        double avg_worsen = 0; int n_w = 0;
        std::uniform_int_distribution<int> sde(0, ne - 1);
        for (int s = 0; s < 200; s++) {
            int eid = sde(rng);
            if (valid_p[eid].empty() || valid_r[eid].empty()) continue;
            int pid = valid_p[eid][rng() % valid_p[eid].size()];
            int rid = valid_r[eid][rng() % valid_r[eid].size()];
            double d = fe.move_delta(sol, eid, pid, rid);
            if (d > 0 && d < 50000) { avg_worsen += d; n_w++; }
        }
        // Scale: destroy/repair affects n_destroy exams, so deltas are ~sqrt(n_destroy) larger
        double base_temp = (n_w > 0) ? (avg_worsen / n_w) / 0.693 : 100.0;
        temp = std::max(1.0, base_temp * std::sqrt((double)n_destroy));
    }
    double init_temp = temp;
    double cooling_rate = 0.999;

    // Operator weights: random, worst, related, shaw, period-strip
    std::vector<double> d_weights = {1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> r_weights = {1.0, 1.0, 1.0}; // greedy, random, regret-2
    std::vector<double> d_scores(5, 0);
    std::vector<double> r_scores(3, 0);
    std::vector<int> d_counts(5, 1);
    std::vector<int> r_counts(3, 1);

    // LAHC history for alternating acceptance
    int lahc_len = std::max(ne / 10, 20);
    std::vector<double> lahc_history(lahc_len, current_fitness);
    bool use_lahc = false;

    std::uniform_real_distribution<double> unif(0.0, 1.0);

    int current_hard = ev.hard();

    if (verbose)
        std::cerr << "[ALNS] Init: feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft() << std::endl;

    int iters_done = 0;
    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        int d_op = roulette(d_weights, rng);
        int r_op = roulette(r_weights, rng);

        // Save state
        auto saved = save_state(sol);

        // Destroy
        std::vector<int> removed;
        if (d_op == 0)
            removed = destroy_random(sol, ne, n_destroy, rng);
        else if (d_op == 1)
            removed = destroy_worst(sol, fe, ne, n_destroy, rng);
        else if (d_op == 2)
            removed = destroy_related(sol, prob, ne, n_destroy, rng);
        else if (d_op == 3)
            removed = destroy_shaw(sol, prob, ne, n_destroy, rng);
        else
            removed = destroy_period_strip(sol, ne, np, n_destroy, rng);

        // Repair
        if (r_op == 0)
            repair_greedy(sol, fe, removed, valid_p, valid_r);
        else if (r_op == 1)
            repair_random(sol, removed, valid_p, valid_r, rng);
        else
            repair_regret2(sol, fe, removed, valid_p, valid_r);

        // Local search polish on repaired exams (1 pass, light)
        for (int eid : removed) {
            const auto& vp = valid_p[eid];
            const auto& vr = valid_r[eid];
            if (vp.empty() || vr.empty()) continue;
            for (int t = 0; t < 3; t++) {
                int pid = vp[rng() % vp.size()];
                int rid = vr[rng() % vr.size()];
                if (pid == sol.period_of[eid] && rid == sol.room_of[eid]) continue;
                double d = fe.move_delta(sol, eid, pid, rid);
                if (d < -0.5) { fe.apply_move(sol, eid, pid, rid); break; }
            }
        }

        // Fast hard-violation check: skip full_eval when clearly worse
        int new_hard = fe.count_hard_fast(sol);
        bool fast_reject = false;
        if (new_hard > current_hard) {
            double hard_delta = (double)(new_hard - current_hard) * 100000.0;
            if (temp < 1e-5 || std::exp(-hard_delta / temp) < 0.001)
                fast_reject = true;
        }

        double score = 0.0;
        if (fast_reject) {
            restore_state(sol, saved);
            no_improve_alns++;
        } else {
            auto new_ev = fe.full_eval(sol);
            double new_fitness = new_ev.fitness();
            double delta = new_fitness - current_fitness;

            bool accept;
            if (use_lahc) {
                int hi = it % lahc_len;
                accept = (new_fitness <= current_fitness) || (new_fitness <= lahc_history[hi]);
                lahc_history[hi] = current_fitness;
            } else {
                accept = (delta < 0);
                if (!accept && temp > 1e-10)
                    accept = (unif(rng) < std::exp(-delta / temp));
            }

            if (accept) {
                current_fitness = new_fitness;
                current_hard = new_ev.hard();
                score = 1.0;
                bool nf = new_ev.feasible();
                bool dominated = (best_feasible && !nf);
                if (!dominated && new_fitness < best_fitness) {
                    best_sol = sol.copy();
                    best_fitness = new_fitness;
                    best_feasible = nf;
                    score = 3.0;
                    no_improve_alns = 0;
                    n_destroy = n_destroy_base;
                    if (verbose && (it < 10 || it % 200 == 0))
                        std::cerr << "[ALNS] Iter " << it << ": best hard=" << new_ev.hard()
                                  << " soft=" << new_ev.soft() << std::endl;
                } else {
                    no_improve_alns++;
                }
            } else {
                restore_state(sol, saved);
                no_improve_alns++;
            }
        }

        // Adaptive destroy size: grow when stuck, cap at 12%
        if (no_improve_alns > 0 && no_improve_alns % 100 == 0) {
            n_destroy = std::min(std::max(1, (int)(ne * 0.12)),
                                 n_destroy + std::max(1, ne / 50));
        }

        d_scores[d_op] += score;
        r_scores[r_op] += score;
        d_counts[d_op]++;
        r_counts[r_op]++;

        if ((it + 1) % 100 == 0) {
            for (int i = 0; i < 5; i++) {
                d_weights[i] = std::max(0.1, 0.7 * d_weights[i] + 0.3 * d_scores[i] / d_counts[i]);
                d_scores[i] = 0; d_counts[i] = 1;
            }
            for (int i = 0; i < 3; i++) {
                r_weights[i] = std::max(0.1, 0.7 * r_weights[i] + 0.3 * r_scores[i] / r_counts[i]);
                r_scores[i] = 0; r_counts[i] = 1;
            }
        }

        temp *= cooling_rate;

        // Toggle SA/LAHC acceptance every 200 iters
        if ((it + 1) % 200 == 0) {
            use_lahc = !use_lahc;
            if (current_fitness >= best_fitness)
                temp = std::max(temp, init_temp * 0.3);
        }
    }

    fe.optimize_rooms(best_sol);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[ALNS] " << iters_done << " iters, " << rt << "s"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "ALNS"};
}