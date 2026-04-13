/*
 * Graph decomposition + branch-and-bound solver (pure STL).
 *
 *   1. Decompose conflict graph into connected components via BFS
 *   2. Small components (<=15): exhaustive backtracking + forward checking
 *   3. Medium components (16-80): best-first B&B with node limit
 *   4. Large components (>80): keep incumbent (greedy/init)
 *   5. SA polish if time remains
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "greedy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

namespace cpsat_detail {

struct Domain {
    std::vector<bool> allowed;
    int count;

    void init(int np) { allowed.assign(np, true); count = np; }
    void remove(int p) { if (p >= 0 && p < (int)allowed.size() && allowed[p]) { allowed[p] = false; count--; } }
    bool has(int p) const { return p >= 0 && p < (int)allowed.size() && allowed[p]; }
    bool empty() const { return count == 0; }
    int first() const {
        for (int i = 0; i < (int)allowed.size(); i++)
            if (allowed[i]) return i;
        return -1;
    }
};

struct Component {
    std::vector<int> exams;   // global exam IDs
    int size() const { return (int)exams.size(); }
};

// Decompose conflict graph into connected components
inline std::vector<Component> decompose(const ProblemInstance& prob, int ne) {
    std::vector<bool> visited(ne, false);
    std::vector<Component> comps;

    for (int start = 0; start < ne; start++) {
        if (visited[start]) continue;
        Component comp;
        std::queue<int> q;
        q.push(start);
        visited[start] = true;
        while (!q.empty()) {
            int e = q.front(); q.pop();
            comp.exams.push_back(e);
            for (auto& [nb, _] : prob.adj[e]) {
                if (!visited[nb]) {
                    visited[nb] = true;
                    q.push(nb);
                }
            }
        }
        comps.push_back(std::move(comp));
    }
    // Sort by size descending
    std::sort(comps.begin(), comps.end(),
              [](const Component& a, const Component& b) { return a.size() > b.size(); });
    return comps;
}

// Cost of assigning exam eid to period pid (soft cost relative to current partial assignment)
inline long long assignment_cost(
    int eid, int pid,
    const std::vector<int>& assignment, // global exam -> period (-1 = unassigned)
    const FastEvaluator& fe, const ProblemInstance& prob)
{
    long long cost = 0;
    for (auto& [nb, _] : prob.adj[eid]) {
        int nb_pid = assignment[nb];
        if (nb_pid < 0) continue;
        if (nb_pid == pid) cost += 100000;
        else {
            int gap = std::abs(pid - nb_pid);
            if (gap > 0 && gap <= fe.w_spread) cost += 1;
            if (fe.period_day[pid] == fe.period_day[nb_pid]) {
                int g = std::abs(fe.period_daypos[pid] - fe.period_daypos[nb_pid]);
                if (g == 1) cost += fe.w_2row;
                else if (g > 1) cost += fe.w_2day;
            }
        }
    }
    cost += fe.period_pen[pid];
    if (fe.large_exams.count(eid) && fe.fl_penalty > 0 && fe.last_periods.count(pid))
        cost += fe.fl_penalty;
    return cost;
}

// Forward checking: propagate assignment[var]=val, return removed entries for undo
struct Removal { int exam; int period; };

inline std::vector<Removal> forward_check(
    int var, int val,
    const std::vector<int>& comp_exams,
    std::vector<Domain>& domains,
    const std::vector<int>& local_idx, // global exam -> local index (-1 if not in component)
    const ProblemInstance& prob)
{
    std::vector<Removal> removals;
    // Remove val from all conflict neighbors in this component
    for (auto& [nb, _] : prob.adj[comp_exams[var]]) {
        int nb_local = local_idx[nb];
        if (nb_local < 0) continue; // not in this component
        if (domains[nb_local].has(val)) {
            domains[nb_local].remove(val);
            removals.push_back({nb_local, val});
        }
    }
    return removals;
}

inline void undo_propagation(std::vector<Domain>& domains, const std::vector<Removal>& removals) {
    for (auto& r : removals) {
        domains[r.exam].allowed[r.period] = true;
        domains[r.exam].count++;
    }
}

// MRV heuristic: pick unassigned variable with smallest domain
inline int pick_mrv(const std::vector<int>& local_assignment, const std::vector<Domain>& domains, int n) {
    int best = -1, best_count = INT_MAX;
    for (int i = 0; i < n; i++) {
        if (local_assignment[i] >= 0) continue;
        if (domains[i].count < best_count) {
            best_count = domains[i].count;
            best = i;
        }
    }
    return best;
}

// Backtracking solver for small components
inline void backtrack_solve(
    int depth, int n,
    std::vector<int>& local_assignment,
    std::vector<Domain>& domains,
    const std::vector<int>& comp_exams,
    const std::vector<int>& local_idx,
    const std::vector<int>& global_assignment,
    const FastEvaluator& fe, const ProblemInstance& prob,
    long long& best_cost,
    std::vector<int>& best_assignment,
    int& nodes_explored,
    int node_limit,
    const std::chrono::high_resolution_clock::time_point& deadline)
{
    if (nodes_explored >= node_limit) return;
    if (std::chrono::high_resolution_clock::now() > deadline) return;
    nodes_explored++;

    if (depth == n) {
        // Evaluate full assignment
        long long cost = 0;
        // Build temp global assignment for cost computation
        std::vector<int> temp_global = global_assignment;
        for (int i = 0; i < n; i++)
            temp_global[comp_exams[i]] = local_assignment[i];

        for (int i = 0; i < n; i++)
            cost += assignment_cost(comp_exams[i], local_assignment[i], temp_global, fe, prob);
        cost /= 2; // each pair counted twice

        if (cost < best_cost) {
            best_cost = cost;
            best_assignment = local_assignment;
        }
        return;
    }

    int var = pick_mrv(local_assignment, domains, n);
    if (var < 0 || domains[var].empty()) return;

    // Try values sorted by cost (cheapest first)
    std::vector<std::pair<long long, int>> candidates;
    // Build temp assignment for cost eval
    std::vector<int> temp_global = global_assignment;
    for (int i = 0; i < n; i++)
        if (local_assignment[i] >= 0) temp_global[comp_exams[i]] = local_assignment[i];

    int np = (int)domains[var].allowed.size();
    for (int p = 0; p < np; p++) {
        if (!domains[var].has(p)) continue;
        long long c = assignment_cost(comp_exams[var], p, temp_global, fe, prob);
        candidates.push_back({c, p});
    }
    std::sort(candidates.begin(), candidates.end());

    for (auto& [_, val] : candidates) {
        local_assignment[var] = val;
        auto saved_domains = domains;
        auto removals = forward_check(var, val, comp_exams, domains, local_idx, prob);

        // Check no domain wiped out
        bool feasible = true;
        for (int i = 0; i < n; i++) {
            if (local_assignment[i] < 0 && domains[i].empty()) { feasible = false; break; }
        }

        if (feasible) {
            backtrack_solve(depth + 1, n, local_assignment, domains,
                           comp_exams, local_idx, global_assignment,
                           fe, prob, best_cost, best_assignment,
                           nodes_explored, node_limit, deadline);
        }

        domains = saved_domains;
        local_assignment[var] = -1;
    }
}

} // namespace cpsat_detail

inline AlgoResult solve_cpsat(
    const ProblemInstance& prob,
    double time_limit_sec = 60.0,
    int seed             = 42,
    bool verbose         = false,
    const Solution* init_sol = nullptr)
{
    using namespace cpsat_detail;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto deadline = t0 + std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
        std::chrono::duration<double>(time_limit_sec));
    std::mt19937 rng(seed);

    int ne = prob.n_e(), np = prob.n_p(), nr = prob.n_r();
    FastEvaluator fe(prob);

    std::vector<std::vector<int>> valid_p(ne), valid_r(ne);
    for (int e = 0; e < ne; e++) {
        for (int p = 0; p < np; p++) if (fe.exam_dur[e] <= fe.period_dur[p]) valid_p[e].push_back(p);
        for (int r = 0; r < nr; r++) if (fe.exam_enroll[e] <= fe.room_cap[r]) valid_r[e].push_back(r);
    }

    // Incumbent from greedy
    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); sol = g.sol.copy(); }
    EvalResult ev = fe.full_eval(sol);
    if (!ev.feasible()) {
        fe.recover_feasibility(sol, 500, seed);
        ev = fe.full_eval(sol);
    }

    // Global assignment vector (for cost computation)
    std::vector<int> global_assignment = sol.period_of;

    // Decompose
    auto comps = decompose(prob, ne);

    if (verbose) {
        std::cerr << "[CPSAT] " << comps.size() << " components: ";
        int small = 0, med = 0, large = 0;
        for (auto& c : comps) {
            if (c.size() <= 15) small++;
            else if (c.size() <= 80) med++;
            else large++;
        }
        std::cerr << small << " small, " << med << " medium, " << large << " large" << std::endl;
    }

    // Build local index for each component
    std::vector<int> local_idx(ne, -1);

    int total_nodes = 0;
    int comps_improved = 0;

    for (auto& comp : comps) {
        if (std::chrono::high_resolution_clock::now() > deadline) break;

        int n = comp.size();

        // Build local index
        for (int i = 0; i < n; i++) local_idx[comp.exams[i]] = i;

        if (n <= 15) {
            // ── Small: exhaustive backtracking ──
            std::vector<Domain> domains(n);
            for (int i = 0; i < n; i++) {
                int eid = comp.exams[i];
                domains[i].init(np);
                // Restrict to valid periods
                for (int p = 0; p < np; p++) {
                    bool valid = false;
                    for (int vp : valid_p[eid]) if (vp == p) { valid = true; break; }
                    if (!valid) domains[i].remove(p);
                }
            }

            std::vector<int> local_assignment(n, -1);
            long long best_cost = LLONG_MAX;
            std::vector<int> best_local;
            int nodes_explored = 0;

            // Compute incumbent cost for this component
            std::vector<int> incumbent_local(n);
            for (int i = 0; i < n; i++) incumbent_local[i] = global_assignment[comp.exams[i]];
            best_local = incumbent_local;
            {
                long long inc_cost = 0;
                for (int i = 0; i < n; i++)
                    inc_cost += assignment_cost(comp.exams[i], incumbent_local[i],
                                                global_assignment, fe, prob);
                inc_cost /= 2;
                best_cost = inc_cost;
            }

            auto comp_deadline = std::min(deadline,
                std::chrono::high_resolution_clock::now() +
                std::chrono::milliseconds(std::max(100, (int)(time_limit_sec * 1000 * n / ne))));

            backtrack_solve(0, n, local_assignment, domains,
                           comp.exams, local_idx, global_assignment,
                           fe, prob, best_cost, best_local,
                           nodes_explored, 50000, comp_deadline);

            total_nodes += nodes_explored;

            // Apply best if found
            if (!best_local.empty()) {
                bool changed = false;
                for (int i = 0; i < n; i++) {
                    if (best_local[i] != global_assignment[comp.exams[i]]) changed = true;
                    global_assignment[comp.exams[i]] = best_local[i];
                }
                if (changed) comps_improved++;
            }

        } else if (n <= 80) {
            // ── Medium: limited backtracking with aggressive pruning ──
            std::vector<Domain> domains(n);
            for (int i = 0; i < n; i++) {
                int eid = comp.exams[i];
                domains[i].init(np);
                for (int p = 0; p < np; p++) {
                    bool valid = false;
                    for (int vp : valid_p[eid]) if (vp == p) { valid = true; break; }
                    if (!valid) domains[i].remove(p);
                }
            }

            std::vector<int> local_assignment(n, -1);
            long long best_cost = LLONG_MAX;
            std::vector<int> best_local;
            int nodes_explored = 0;

            // Incumbent
            std::vector<int> incumbent_local(n);
            for (int i = 0; i < n; i++) incumbent_local[i] = global_assignment[comp.exams[i]];
            best_local = incumbent_local;
            {
                long long inc_cost = 0;
                for (int i = 0; i < n; i++)
                    inc_cost += assignment_cost(comp.exams[i], incumbent_local[i],
                                                global_assignment, fe, prob);
                inc_cost /= 2;
                best_cost = inc_cost;
            }

            auto comp_deadline = std::min(deadline,
                std::chrono::high_resolution_clock::now() +
                std::chrono::milliseconds(std::max(500, (int)(time_limit_sec * 1000 * n / ne))));

            backtrack_solve(0, n, local_assignment, domains,
                           comp.exams, local_idx, global_assignment,
                           fe, prob, best_cost, best_local,
                           nodes_explored, 100000, comp_deadline);

            total_nodes += nodes_explored;

            if (!best_local.empty()) {
                bool changed = false;
                for (int i = 0; i < n; i++) {
                    if (best_local[i] != global_assignment[comp.exams[i]]) changed = true;
                    global_assignment[comp.exams[i]] = best_local[i];
                }
                if (changed) comps_improved++;
            }
        }
        // Large components: keep incumbent

        // Clear local index
        for (int i = 0; i < n; i++) local_idx[comp.exams[i]] = -1;
    }

    // Apply optimized period assignments back to solution
    for (int e = 0; e < ne; e++) {
        if (global_assignment[e] != sol.period_of[e] && global_assignment[e] >= 0) {
            // Find best room for the new period
            int pid = global_assignment[e];
            int best_rid = sol.room_of[e]; // keep current room as fallback
            int best_excess = INT_MAX;
            for (int rid : valid_r[e]) {
                int excess = sol.get_pr_enroll(pid, rid) + fe.exam_enroll[e] - fe.room_cap[rid];
                if (excess < best_excess) { best_excess = excess; best_rid = rid; }
            }
            sol.assign(e, pid, best_rid);
        }
    }

    // Recover feasibility if B&B introduced PHC violations
    {
        auto check = fe.full_eval(sol);
        if (!check.feasible()) {
            fe.recover_feasibility(sol, 1000, seed);
        }
    }

    // SA polish if time remains
    auto t_polish = std::chrono::high_resolution_clock::now();
    double remaining = std::chrono::duration<double>(deadline - t_polish).count();
    if (remaining > 1.0) {
        ev = fe.full_eval(sol);
        double fitness = ev.fitness();

        // Calibrate SA temp
        double temp;
        {
            double avg_w = 0; int nw = 0;
            std::uniform_int_distribution<int> sde(0, ne - 1);
            for (int s = 0; s < 200; s++) {
                int eid = sde(rng);
                if (valid_p[eid].empty() || valid_r[eid].empty()) continue;
                int pid = valid_p[eid][rng() % valid_p[eid].size()];
                int rid = valid_r[eid][rng() % valid_r[eid].size()];
                double d = fe.move_delta(sol, eid, pid, rid);
                if (d > 0 && d < 50000) { avg_w += d; nw++; }
            }
            temp = (nw > 0) ? std::max(1.0, (avg_w / nw) / 0.693) : 100.0;
        }

        std::uniform_int_distribution<int> de(0, ne - 1);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        int sa_iters = std::max(500, (int)(remaining * 2000));

        // Per-exam cost + alias
        std::vector<double> exam_cost(ne, 1.0);
        AliasTable alias;
        for (int e = 0; e < ne; e++) {
            int pid = sol.period_of[e]; if (pid < 0) continue;
            double c = 1.0;
            for (auto& [nb, _] : prob.adj[e]) {
                int np2 = sol.period_of[nb]; if (np2 < 0) continue;
                if (np2 == pid) c += 100000;
            }
            exam_cost[e] = c;
        }
        alias.build(exam_cost);

        Solution best_sol_sa = sol.copy();
        double best_fitness_sa = fitness;

        for (int it = 0; it < sa_iters; it++) {
            if (std::chrono::high_resolution_clock::now() > deadline) break;

            int eid = (unif(rng) < 0.7) ? alias.sample(rng) : de(rng);
            auto& vp = valid_p[eid]; auto& vr = valid_r[eid];
            if (vp.empty() || vr.empty()) continue;

            int pid = vp[rng() % vp.size()];
            int rid = vr[rng() % vr.size()];
            double d = fe.move_delta(sol, eid, pid, rid);

            bool accept = (d < 0);
            if (!accept && temp > 1e-10)
                accept = (unif(rng) < std::exp(-d / temp));

            if (accept) {
                fe.apply_move(sol, eid, pid, rid);
                fitness += d;
                if (fitness < best_fitness_sa - 0.5) {
                    auto check = fe.full_eval(sol);
                    fitness = check.fitness();
                    if (fitness < best_fitness_sa) {
                        best_sol_sa = sol.copy();
                        best_fitness_sa = fitness;
                    }
                }
            }
            temp *= 0.9995;
        }
        sol = std::move(best_sol_sa);
    }

    fe.optimize_rooms(sol);

    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(sol);

    if (verbose)
        std::cerr << "[CPSAT] " << rt << "s, " << total_nodes << " nodes, "
                  << comps_improved << " components improved"
                  << "  feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard()
                  << " soft=" << final_ev.soft() << std::endl;

    return {std::move(sol), final_ev, rt, total_nodes, "CP-SAT B&B"};
}
