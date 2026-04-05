/*
 * alns.h — Adaptive Large Neighborhood Search
 *
 * Destroy-and-repair framework with adaptive operator selection.
 * 3 destroy operators (random, worst, related) + 2 repair operators (greedy, random).
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
    std::unordered_map<int64_t, int> pr_enroll;
};

inline SavedState save_state(const Solution& sol) {
    return {sol.period_of, sol.room_of, sol.pr_enroll};
}

inline void restore_state(Solution& sol, const SavedState& s) {
    sol.period_of = s.period_of;
    sol.room_of = s.room_of;
    sol.pr_enroll = s.pr_enroll;
}

inline void unassign(Solution& sol, int eid) {
    int pid = sol.period_of[eid];
    if (pid >= 0) {
        int rid = sol.room_of[eid];
        int enr = sol.enroll_cache[eid];
        sol.pr_enroll[sol.pr_key(pid, rid)] -= enr;
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
                int gap = std::abs(pid - nb_pid);
                if (gap > 0 && gap <= fe.w_spread) cost += 1;
            }
        }
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
        int best_pid = -1, best_rid = -1;
        long long best_cost = (long long)1e18;

        for (int pid : vp) {
            for (int rid : vr) {
                long long cost = 0;
                // Hard: student conflicts
                for (auto& [nb, _] : fe.P.adj[eid])
                    if (sol.period_of[nb] == pid) cost += 100000;
                // Hard: room capacity
                if (sol.get_pr_enroll(pid, rid) + fe.exam_enroll[eid] > fe.room_cap[rid])
                    cost += 100000;
                // Soft: proximity
                for (auto& [nb, _] : fe.P.adj[eid]) {
                    int nb_pid = sol.period_of[nb];
                    if (nb_pid < 0 || nb_pid == pid) continue;
                    int gap = std::abs(pid - nb_pid);
                    if (gap > 0 && gap <= fe.w_spread) cost += 1;
                    if (fe.period_day[pid] == fe.period_day[nb_pid]) {
                        int g = std::abs(fe.period_daypos[pid] - fe.period_daypos[nb_pid]);
                        if (g == 1) cost += fe.w_2row;
                        else if (g > 1) cost += fe.w_2day;
                    }
                }
                // Soft: penalties
                cost += fe.period_pen[pid] + fe.room_pen[rid];
                // Soft: front load
                if (fe.large_exams.count(eid) && fe.fl_penalty > 0 && fe.last_periods.count(pid))
                    cost += fe.fl_penalty;
                if (cost < best_cost) {
                    best_cost = cost;
                    best_pid = pid;
                    best_rid = rid;
                }
            }
        }

        if (best_pid >= 0)
            sol.assign(eid, best_pid, best_rid);
        else if (!vp.empty() && !vr.empty())
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

} // namespace alns_detail

// ── Public interface ───────────────────────────────────────────

inline AlgoResult solve_alns(
    const ProblemInstance& prob,
    int max_iterations  = 2000,
    double destroy_pct  = 0.15,
    int seed            = 42,
    bool verbose        = false)
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
    auto greedy_res = solve_greedy(prob, false);
    Solution sol = greedy_res.sol.copy();

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

    int n_destroy = std::max(1, (int)(ne * destroy_pct));

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

    // Operator weights
    std::vector<double> d_weights = {1.0, 1.0, 1.0};
    std::vector<double> r_weights = {1.0, 1.0};
    std::vector<double> d_scores = {0, 0, 0};
    std::vector<double> r_scores = {0, 0};
    std::vector<int> d_counts = {1, 1, 1};
    std::vector<int> r_counts = {1, 1};

    std::uniform_real_distribution<double> unif(0.0, 1.0);

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
        else
            removed = destroy_related(sol, prob, ne, n_destroy, rng);

        // Repair
        if (r_op == 0)
            repair_greedy(sol, fe, removed, valid_p, valid_r);
        else
            repair_random(sol, removed, valid_p, valid_r, rng);

        // Evaluate
        auto new_ev = fe.full_eval(sol);
        double new_fitness = new_ev.fitness();
        double delta = new_fitness - current_fitness;

        bool accept = (delta < 0);
        if (!accept && temp > 1e-10)
            accept = (unif(rng) < std::exp(-delta / temp));

        double score = 0.0;
        if (accept) {
            current_fitness = new_fitness;
            score = 1.0;
            bool nf = new_ev.feasible();
            bool dominated = (best_feasible && !nf);
            if (!dominated && new_fitness < best_fitness) {
                best_sol = sol.copy();
                best_fitness = new_fitness;
                best_feasible = nf;
                score = 3.0;
                if (verbose && (it < 10 || it % 200 == 0))
                    std::cerr << "[ALNS] Iter " << it << ": best hard=" << new_ev.hard()
                              << " soft=" << new_ev.soft() << std::endl;
            }
        } else {
            restore_state(sol, saved);
        }

        d_scores[d_op] += score;
        r_scores[r_op] += score;
        d_counts[d_op]++;
        r_counts[r_op]++;

        if ((it + 1) % 100 == 0) {
            for (int i = 0; i < 3; i++) {
                d_weights[i] = std::max(0.1, 0.7 * d_weights[i] + 0.3 * d_scores[i] / d_counts[i]);
                d_scores[i] = 0; d_counts[i] = 1;
            }
            for (int i = 0; i < 2; i++) {
                r_weights[i] = std::max(0.1, 0.7 * r_weights[i] + 0.3 * r_scores[i] / r_counts[i]);
                r_scores[i] = 0; r_counts[i] = 1;
            }
        }

        temp *= cooling_rate;

        // Reheat when stuck
        if ((it + 1) % 200 == 0 && current_fitness >= best_fitness)
            temp = std::max(temp, init_temp * 0.3);
    }

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