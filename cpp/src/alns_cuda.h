/*
 * ALNS (Thompson) with batched repair scoring via CudaEvaluator.
 *
 * Same algorithm as solve_alns_thompson. Only difference: the greedy repair
 * operator routes its (np × nr) candidate scoring through
 * CudaEvaluator::score_placement_batch. CPU fallback is bit-exact
 * equivalent to repair_greedy (validated by `make bench` placement-scorer
 * test); GPU path uses delta_kernel_placement, one kernel launch per
 * unplaced exam.
 *
 * CPU-fallback quality/runtime is identical to alns_thompson on same seed.
 * GPU speedup materialises when nvcc + libdelta_cuda.so are linked.
 */

#pragma once

#include "models.h"
#include "evaluator.h"
#include "evaluator_cached.h"
#include "greedy.h"
#include "alns.h"
#include "cuda/cuda_evaluator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

namespace alns_cuda_detail {

inline double beta_sample(double a, double b, std::mt19937& rng) {
    std::gamma_distribution<double> ga(a, 1.0), gb(b, 1.0);
    double x = ga(rng), y = gb(rng);
    return (x + y > 0) ? x / (x + y) : 0.5;
}

} // namespace alns_cuda_detail

inline AlgoResult solve_alns_cuda(
    const ProblemInstance& prob,
    int max_iterations  = 2000,
    double destroy_pct  = 0.04,
    int seed            = 42,
    bool verbose        = false,
    const Solution* init_sol = nullptr)
{
    using namespace alns_detail;
    using alns_cuda_detail::beta_sample;
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

    Solution sol;
    if (init_sol) { sol = init_sol->copy(); }
    else { auto g = solve_greedy(prob, false); sol = g.sol.copy(); }
    EvalResult ev = fe.full_eval(sol);
    if (!ev.feasible()) { fe.recover_feasibility(sol, 500, seed); ev = fe.full_eval(sol); }

    CachedEvaluator Ecach(fe);
    Ecach.initialize(sol);
    CudaEvaluator Cuev(Ecach);  // GPU state if available, else CPU fallback

    double current_fitness = ev.fitness();
    Solution best_sol = sol.copy();
    double best_fitness = current_fitness;
    bool best_feasible = ev.feasible();
    int current_hard = ev.hard();

    int n_destroy_base = std::max(1, (int)(ne * destroy_pct));
    int n_destroy = n_destroy_base;
    int no_improve_alns = 0;

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
        double base_temp = (n_w > 0) ? (avg_worsen / n_w) / 0.693 : 100.0;
        temp = std::max(1.0, base_temp * std::sqrt((double)n_destroy));
    }
    double init_temp = temp;
    double cooling_rate = 0.999;

    const int N_D = 5, N_R = 3;
    std::vector<double> d_alpha(N_D, 1.0), d_beta(N_D, 1.0);
    std::vector<double> r_alpha(N_R, 1.0), r_beta(N_R, 1.0);
    auto thompson_pick = [&](const std::vector<double>& a,
                             const std::vector<double>& b) -> int {
        int best = 0; double best_p = -1;
        for (int k = 0; k < (int)a.size(); k++) {
            double p = beta_sample(a[k], b[k], rng);
            if (p > best_p) { best_p = p; best = k; }
        }
        return best;
    };

    std::uniform_real_distribution<double> unif(0.0, 1.0);
    int iters_done = 0;

    if (verbose)
        std::cerr << "[ALNSCuda] Init feasible=" << ev.feasible()
                  << " hard=" << ev.hard() << " soft=" << ev.soft()
                  << " gpu=" << (Cuev.gpu_active ? "on" : "off") << std::endl;

    for (int it = 0; it < max_iterations; it++) {
        iters_done = it + 1;

        int d_op = thompson_pick(d_alpha, d_beta);
        int r_op = thompson_pick(r_alpha, r_beta);

        auto saved = save_state(sol);

        std::vector<int> removed;
        if (d_op == 0)      removed = destroy_random(sol, ne, n_destroy, rng);
        else if (d_op == 1) removed = destroy_worst(sol, fe, ne, n_destroy, rng);
        else if (d_op == 2) removed = destroy_related(sol, prob, ne, n_destroy, rng);
        else if (d_op == 3) removed = destroy_shaw(sol, prob, ne, n_destroy, rng);
        else                removed = destroy_period_strip(sol, ne, np, n_destroy, rng);

        if (r_op == 0)      repair_greedy_batched(sol, fe, Cuev, removed, valid_p, valid_r);
        else if (r_op == 1) repair_random(sol, removed, valid_p, valid_r, rng);
        else                repair_regret2(sol, fe, removed, valid_p, valid_r);

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

        int new_hard = fe.count_hard_fast(sol);
        double reward = 0;
        bool fast_reject = false;
        if (new_hard > current_hard) {
            double hard_delta = (double)(new_hard - current_hard) * 100000.0;
            if (temp < 1e-5 || std::exp(-hard_delta / temp) < 0.001) fast_reject = true;
        }

        if (fast_reject) {
            restore_state(sol, saved);
            no_improve_alns++;
        } else {
            auto new_ev = fe.full_eval(sol);
            double new_fitness = new_ev.fitness();
            double delta = new_fitness - current_fitness;
            bool accept = (delta < 0);
            if (!accept && temp > 1e-10)
                accept = (unif(rng) < std::exp(-delta / temp));
            if (accept) {
                current_fitness = new_fitness;
                current_hard = new_ev.hard();
                bool nf = new_ev.feasible();
                bool dominated = (best_feasible && !nf);
                if (!dominated && new_fitness < best_fitness) {
                    best_sol = sol.copy();
                    best_fitness = new_fitness;
                    best_feasible = nf;
                    reward = 3.0;
                    no_improve_alns = 0;
                    n_destroy = n_destroy_base;
                } else {
                    reward = (delta < 0) ? 1.5 : 0.5;
                    no_improve_alns++;
                }
            } else {
                restore_state(sol, saved);
                no_improve_alns++;
            }
        }

        double r01 = std::min(1.0, reward / 3.0);
        d_alpha[d_op] += r01;      d_beta[d_op] += (1.0 - r01);
        r_alpha[r_op] += r01;      r_beta[r_op] += (1.0 - r01);

        temp *= cooling_rate;
        if (no_improve_alns > 0 && no_improve_alns % 300 == 0) {
            n_destroy = std::min(ne / 2, n_destroy * 2);
            temp = init_temp * 0.5;
        }
    }

    fe.optimize_rooms(best_sol);
    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double>(t1 - t0).count();
    EvalResult final_ev = fe.full_eval(best_sol);

    if (verbose)
        std::cerr << "[ALNSCuda] " << iters_done << " iters, " << rt << "s"
                  << " feasible=" << final_ev.feasible()
                  << " hard=" << final_ev.hard() << " soft=" << final_ev.soft()
                  << "  gpu=" << (Cuev.gpu_active ? "on" : "off") << std::endl;

    return {std::move(best_sol), final_ev, rt, iters_done, "ALNS (Thompson+CUDA)"};
}
