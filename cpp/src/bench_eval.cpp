/*
 * Benchmark: scalar vs adj-scalar vs SIMD vs asm-kernel for move_delta.
 *
 * Usage:  ./bench_eval <instance.exam> [iters]
 *
 * Steps:
 *   1. Parse instance, build derived, greedy-construct a solution.
 *   2. Generate N random (eid, new_pid, new_rid) move proposals.
 *   3. For each variant, time apply of all N proposals (no state change:
 *      move_delta is read-only).
 *   4. Verify each variant agrees with the scalar baseline within epsilon.
 *
 * Does not modify any existing header. Links only against header-only code.
 */

#include "parser.h"
#include "models.h"
#include "evaluator.h"
#include "greedy.h"
#include "evaluator_simd.h"
#include "tabu.h"
#include "tabu_simd.h"
#include "tabu_cached.h"
#include "lahc_cached.h"
#include "vns_cached.h"
#include "alns.h"
#include "alns_thompson.h"
#include "xoshiro.h"
#include "portfolio.h"
#include "polish.h"
#include "fpga_sim.h"
#include "evaluator_cached.h"
#include "cuda/cuda_evaluator.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <string>
#include <cmath>

struct Proposal { int eid; int new_pid; int new_rid; };

static std::vector<Proposal> make_proposals(
    const ProblemInstance& P, const Solution& sol, int n, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> de(0, P.n_e() - 1);
    std::uniform_int_distribution<int> dp(0, P.n_p() - 1);
    std::uniform_int_distribution<int> dr(0, P.n_r() - 1);
    std::vector<Proposal> v;
    v.reserve(n);
    // Only include eids that are currently assigned — all four variants
    // delegate unassigned→assigned to the original scalar path, which
    // makes those cases uninteresting for timing comparison.
    std::vector<int> assigned_eids;
    for (int e = 0; e < P.n_e(); e++)
        if (sol.period_of[e] >= 0) assigned_eids.push_back(e);
    std::uniform_int_distribution<int> daes(0, (int)assigned_eids.size() - 1);
    for (int i = 0; i < n; i++)
        v.push_back({assigned_eids[daes(rng)], dp(rng), dr(rng)});
    return v;
}

template <typename F>
static double time_ms(F&& f, int reps = 1) {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <instance.exam> [iters=200000]\n", argv[0]);
        return 1;
    }
    std::string path = argv[1];
    int iters = (argc >= 3) ? std::atoi(argv[2]) : 200000;

    std::printf("[bench] loading %s ...\n", path.c_str());
    auto P = parser::parse_exam_file(path);
    P.build_derived();

    std::printf("[bench] instance: ne=%d np=%d nr=%d  (adj pairs: %zu)\n",
                P.n_e(), P.n_p(), P.n_r(), [&]{
                    size_t s = 0; for (auto& a : P.adj) s += a.size(); return s;
                }());

    std::printf("[bench] running greedy to get a real solution ...\n");
    auto gres = solve_greedy(P, false);
    Solution sol = gres.sol;
    std::printf("[bench] greedy: hard=%d soft=%d  runtime=%.2fs\n",
                gres.eval.hard(), gres.eval.soft(), gres.runtime_sec);

    FastEvaluator E(P);
    FastEvaluatorSIMD Esimd(E);

    auto props = make_proposals(P, sol, iters, 12345);
    std::printf("[bench] %d random proposals generated.\n", iters);

    // ── Cached evaluator (Phase 2b) — initialize once per Solution ──
    CachedEvaluator Ecach(E);
    Ecach.initialize(sol);

    // ── Correctness check on a small sample ──
    int mismatch_adj = 0, mismatch_simd = 0, mismatch_cached = 0;
    const double eps = 1e-6;
    int check_n = std::min(iters, 2000);
    for (int i = 0; i < check_n; i++) {
        const auto& p = props[i];
        double ref = E.move_delta(sol, p.eid, p.new_pid, p.new_rid);
        double a   = Esimd.move_delta_adj (sol, p.eid, p.new_pid, p.new_rid);
        if (std::fabs(ref - a) > eps) mismatch_adj++;
#ifdef EVAL_SIMD_AVX2
        double s   = Esimd.move_delta_simd(sol, p.eid, p.new_pid, p.new_rid);
        if (std::fabs(ref - s) > eps) mismatch_simd++;
#endif
        double c   = Ecach.move_delta(sol, p.eid, p.new_pid, p.new_rid);
        if (std::fabs(ref - c) > eps) mismatch_cached++;
    }
    std::printf("[bench] correctness (first %d): adj=%d, simd=%d, cached=%d mismatches\n",
                check_n, mismatch_adj, mismatch_simd, mismatch_cached);
    if (mismatch_adj > 0 || mismatch_simd > 0 || mismatch_cached > 0) {
        std::fprintf(stderr, "[bench] WARNING: mismatches detected — do not trust timings.\n");
    }

    // ── Timings ──
    volatile double sink = 0;  // prevent dead-code elimination
    int reps = 3;

    double t_scalar = time_ms([&]{
        double s = 0;
        for (auto& p : props) s += E.move_delta(sol, p.eid, p.new_pid, p.new_rid);
        sink += s;
    }, reps);

    double t_adj = time_ms([&]{
        double s = 0;
        for (auto& p : props) s += Esimd.move_delta_adj(sol, p.eid, p.new_pid, p.new_rid);
        sink += s;
    }, reps);

#ifdef EVAL_SIMD_AVX2
    double t_simd = time_ms([&]{
        double s = 0;
        for (auto& p : props) s += Esimd.move_delta_simd(sol, p.eid, p.new_pid, p.new_rid);
        sink += s;
    }, reps);

    double t_asm = time_ms([&]{
        double s = 0;
        for (auto& p : props) s += Esimd.move_delta_asm(sol, p.eid, p.new_pid, p.new_rid);
        sink += s;
    }, reps);
#endif

    // Cached timing (Phase 2b) — Solution state unchanged, cache still valid
    double t_cached = time_ms([&]{
        double s = 0;
        for (auto& p : props) s += Ecach.move_delta(sol, p.eid, p.new_pid, p.new_rid);
        sink += s;
    }, reps);

    (void)sink;

    std::printf("\n=== move_delta timing (%d calls, %d reps avg) ===\n", iters, reps);
    std::printf("  %-18s %10.2f ms   %6.2f ns/call   %5.2fx\n",
                "scalar (orig)", t_scalar, t_scalar * 1e6 / iters, 1.0);
    std::printf("  %-18s %10.2f ms   %6.2f ns/call   %5.2fx\n",
                "adj-scalar",    t_adj,    t_adj    * 1e6 / iters, t_scalar / t_adj);
#ifdef EVAL_SIMD_AVX2
    std::printf("  %-18s %10.2f ms   %6.2f ns/call   %5.2fx\n",
                "AVX2 intrinsics", t_simd, t_simd * 1e6 / iters, t_scalar / t_simd);
    std::printf("  %-18s %10.2f ms   %6.2f ns/call   %5.2fx\n",
                "asm kernel+scalar", t_asm, t_asm * 1e6 / iters, t_scalar / t_asm);
#else
    std::printf("  (AVX2 not compiled in — rebuild with -mavx2)\n");
#endif
    std::printf("  %-18s %10.2f ms   %6.2f ns/call   %5.2fx\n",
                "cached (Phase 2b)", t_cached, t_cached * 1e6 / iters,
                t_scalar / t_cached);

    // ═══════════════════════════════════════════════════════════
    //  END-TO-END: Tabu vs Tabu-SIMD vs Portfolio
    // ═══════════════════════════════════════════════════════════
    std::printf("\n=== end-to-end solver comparison ===\n");
    std::printf("(same init, same iter cap — SIMD should do more work per sec)\n");

    int tabu_iters = 1500;
    auto tt0 = std::chrono::steady_clock::now();
    auto rt_scalar = solve_tabu(P, tabu_iters, 20, 500, 42, false);
    auto tt1 = std::chrono::steady_clock::now();
    auto rt_simd   = solve_tabu_simd(P, tabu_iters, 20, 500, 42, false);
    auto tt2 = std::chrono::steady_clock::now();
    auto rt_cach   = solve_tabu_cached(P, tabu_iters, 20, 500, 42, false);
    auto tt3 = std::chrono::steady_clock::now();

    double ms_scalar = std::chrono::duration<double, std::milli>(tt1 - tt0).count();
    double ms_simd   = std::chrono::duration<double, std::milli>(tt2 - tt1).count();
    double ms_cach   = std::chrono::duration<double, std::milli>(tt3 - tt2).count();

    std::printf("  %-14s feasible=%d hard=%d soft=%6d  %8.1f ms  iters=%d\n",
                "Tabu (scalar)", rt_scalar.eval.feasible(), rt_scalar.eval.hard(),
                rt_scalar.eval.soft(), ms_scalar, rt_scalar.iterations);
    std::printf("  %-14s feasible=%d hard=%d soft=%6d  %8.1f ms  iters=%d   (%.2fx)\n",
                "Tabu (SIMD)",   rt_simd.eval.feasible(),   rt_simd.eval.hard(),
                rt_simd.eval.soft(),   ms_simd,   rt_simd.iterations,
                ms_scalar / ms_simd);
    std::printf("  %-14s feasible=%d hard=%d soft=%6d  %8.1f ms  iters=%d   (%.2fx)\n",
                "Tabu (Cached)", rt_cach.eval.feasible(),   rt_cach.eval.hard(),
                rt_cach.eval.soft(),   ms_cach,   rt_cach.iterations,
                ms_scalar / ms_cach);

    // ── Phase 2b'' — SA / GD / ALNS cached comparison ──
    std::printf("\n=== Phase 2b'' cached vs scalar (5000 iters each) ===\n");
    auto sa0  = std::chrono::steady_clock::now();
    auto rsa  = solve_sa(P, 5000, 0.0, 0.9995, 42, false);
    auto sa1  = std::chrono::steady_clock::now();
    auto rsac = solve_sa_cached(P, 5000, 0.0, 0.9995, 42, false);
    auto sa2  = std::chrono::steady_clock::now();
    double ms_sa  = std::chrono::duration<double, std::milli>(sa1 - sa0).count();
    double ms_sac = std::chrono::duration<double, std::milli>(sa2 - sa1).count();
    std::printf("  %-16s hard=%d soft=%6d  %8.1f ms\n",
                "SA (scalar)", rsa.eval.hard(), rsa.eval.soft(), ms_sa);
    std::printf("  %-16s hard=%d soft=%6d  %8.1f ms   (%.2fx)\n",
                "SA (cached)", rsac.eval.hard(), rsac.eval.soft(), ms_sac, ms_sa / ms_sac);

    auto gd0  = std::chrono::steady_clock::now();
    auto rgd  = solve_great_deluge(P, 5000, 0.0, 42, false);
    auto gd1  = std::chrono::steady_clock::now();
    auto rgdc = solve_great_deluge_cached(P, 5000, 0.0, 42, false);
    auto gd2  = std::chrono::steady_clock::now();
    double ms_gd  = std::chrono::duration<double, std::milli>(gd1 - gd0).count();
    double ms_gdc = std::chrono::duration<double, std::milli>(gd2 - gd1).count();
    std::printf("  %-16s hard=%d soft=%6d  %8.1f ms\n",
                "GD (scalar)", rgd.eval.hard(), rgd.eval.soft(), ms_gd);
    std::printf("  %-16s hard=%d soft=%6d  %8.1f ms   (%.2fx)\n",
                "GD (cached)", rgdc.eval.hard(), rgdc.eval.soft(), ms_gdc, ms_gd / ms_gdc);

    auto al0  = std::chrono::steady_clock::now();
    auto ral  = solve_alns(P, 1000, 0.04, 42, false);
    auto al1  = std::chrono::steady_clock::now();
    auto ralc = solve_alns_cached(P, 1000, 0.04, 42, false);
    auto al2  = std::chrono::steady_clock::now();
    double ms_al  = std::chrono::duration<double, std::milli>(al1 - al0).count();
    double ms_alc = std::chrono::duration<double, std::milli>(al2 - al1).count();
    std::printf("  %-16s hard=%d soft=%6d  %8.1f ms\n",
                "ALNS (scalar)", ral.eval.hard(), ral.eval.soft(), ms_al);
    std::printf("  %-16s hard=%d soft=%6d  %8.1f ms   (%.2fx)\n",
                "ALNS (cached)", ralc.eval.hard(), ralc.eval.soft(), ms_alc, ms_al / ms_alc);

    // ── LAHC, VNS (cached), and ALNS Thompson (AOS) ──
    auto la0  = std::chrono::steady_clock::now();
    auto rla  = solve_lahc(P, 5000, 0, 42, false);
    auto la1  = std::chrono::steady_clock::now();
    auto rlac = solve_lahc_cached(P, 5000, 0, 42, false);
    auto la2  = std::chrono::steady_clock::now();
    double ms_la  = std::chrono::duration<double, std::milli>(la1 - la0).count();
    double ms_lac = std::chrono::duration<double, std::milli>(la2 - la1).count();
    std::printf("  %-16s hard=%d soft=%6d  %8.1f ms\n",
                "LAHC (scalar)", rla.eval.hard(), rla.eval.soft(), ms_la);
    std::printf("  %-16s hard=%d soft=%6d  %8.1f ms   (%.2fx)\n",
                "LAHC (cached)", rlac.eval.hard(), rlac.eval.soft(), ms_lac, ms_la / ms_lac);

    auto vn0  = std::chrono::steady_clock::now();
    auto rvn  = solve_vns(P, 2000, 0, 42, false);
    auto vn1  = std::chrono::steady_clock::now();
    auto rvnc = solve_vns_cached(P, 2000, 0, 42, false);
    auto vn2  = std::chrono::steady_clock::now();
    double ms_vn  = std::chrono::duration<double, std::milli>(vn1 - vn0).count();
    double ms_vnc = std::chrono::duration<double, std::milli>(vn2 - vn1).count();
    std::printf("  %-16s hard=%d soft=%6d  %8.1f ms\n",
                "VNS (scalar)", rvn.eval.hard(), rvn.eval.soft(), ms_vn);
    std::printf("  %-16s hard=%d soft=%6d  %8.1f ms   (%.2fx)\n",
                "VNS (cached)", rvnc.eval.hard(), rvnc.eval.soft(), ms_vnc, ms_vn / ms_vnc);

    auto at0  = std::chrono::steady_clock::now();
    auto ralt = solve_alns_thompson(P, 1000, 0.04, 42, false);
    auto at1  = std::chrono::steady_clock::now();
    double ms_at = std::chrono::duration<double, std::milli>(at1 - at0).count();
    std::printf("  %-16s hard=%d soft=%6d  %8.1f ms   (AOS Thompson)\n",
                "ALNS (thompson)", ralt.eval.hard(), ralt.eval.soft(), ms_at);

    // ── Long-run ALNS Thompson vs roulette (10000 iters — AOS needs time) ──
    std::printf("\n=== AOS long-run study (10000 iters) ===\n");
    {
        auto L0 = std::chrono::steady_clock::now();
        auto rL_roulette = solve_alns(P, 10000, 0.04, 42, false);
        auto L1 = std::chrono::steady_clock::now();
        auto rL_thompson = solve_alns_thompson(P, 10000, 0.04, 42, false);
        auto L2 = std::chrono::steady_clock::now();
        double ms_rouL = std::chrono::duration<double, std::milli>(L1 - L0).count();
        double ms_thoL = std::chrono::duration<double, std::milli>(L2 - L1).count();
        std::printf("  ALNS (roulette)  soft=%6d  %.1f ms\n",
                    rL_roulette.eval.soft(), ms_rouL);
        std::printf("  ALNS (thompson)  soft=%6d  %.1f ms   (Δsoft=%+d)\n",
                    rL_thompson.eval.soft(), ms_thoL,
                    rL_thompson.eval.soft() - rL_roulette.eval.soft());
    }

    // ── RNG microbench: mt19937 vs xoshiro256++ ──
    std::printf("\n=== RNG microbench (100M uniform_int draws) ===\n");
    {
        const int N = 100'000'000;
        auto mt0 = std::chrono::steady_clock::now();
        {
            std::mt19937 mtrng(12345);
            std::uniform_int_distribution<int> dist(0, 1000);
            int64_t s = 0;
            for (int i = 0; i < N; i++) s += dist(mtrng);
            sink += s;
        }
        auto mt1 = std::chrono::steady_clock::now();
        {
            Xoshiro256pp xrng(12345);
            std::uniform_int_distribution<int> dist(0, 1000);
            int64_t s = 0;
            for (int i = 0; i < N; i++) s += dist(xrng);
            sink += s;
        }
        auto mt2 = std::chrono::steady_clock::now();
        double ms_mt = std::chrono::duration<double, std::milli>(mt1 - mt0).count();
        double ms_xo = std::chrono::duration<double, std::milli>(mt2 - mt1).count();
        std::printf("  %-12s %7.1f ms  (%.2f ns/draw)\n",
                    "mt19937", ms_mt, ms_mt * 1e6 / N);
        std::printf("  %-12s %7.1f ms  (%.2f ns/draw)   (%.2fx faster)\n",
                    "xoshiro++", ms_xo, ms_xo * 1e6 / N, ms_mt / ms_xo);
    }

#ifdef _OPENMP
    std::printf("\n=== parallel portfolio (OpenMP) ===\n");
    auto pspec = default_portfolio_spec();
    std::printf("  jobs: %zu   threads: %d\n", pspec.size(), omp_get_max_threads());
    auto pres = run_portfolio(P, pspec, false);
    std::printf("  wall: %.1f s   best: %s  hard=%d soft=%d\n",
                pres.wallclock_sec, pres.best.algorithm.c_str(),
                pres.best.eval.hard(), pres.best.eval.soft());
    double sum_seq = 0;
    for (auto& r : pres.all) sum_seq += r.runtime_sec;
    std::printf("  sequential-sum: %.1f s   parallel-speedup: %.2fx\n",
                sum_seq, sum_seq / pres.wallclock_sec);

    // ═══════════════════════════════════════════════════════════
    //  Phase 2a polish pipeline (applied to portfolio winner)
    // ═══════════════════════════════════════════════════════════
    if (pres.best.eval.feasible()) {
        std::printf("\n=== Phase 2a polish pipeline ===\n");
        Solution polished = pres.best.sol.copy();
        auto pstat = polish_solution(P, polished, 10.0, false);
        double pct = pstat.soft_before > 0
                     ? 100.0 * (pstat.soft_before - pstat.soft_after) / pstat.soft_before
                     : 0.0;
        std::printf("  soft: %d → %d  (Δ=%d, %.2f%%)  single=%d (%d passes) swaps=%d  %.2f s\n",
                    pstat.soft_before, pstat.soft_after,
                    pstat.soft_before - pstat.soft_after, pct,
                    pstat.single_moves_applied, pstat.single_passes,
                    pstat.swaps_applied, pstat.runtime_sec);
    } else {
        std::printf("\n  (polish skipped — portfolio result infeasible)\n");
    }
#else
    std::printf("\n  (portfolio requires -fopenmp — build with `make bench-omp`)\n");
#endif

    // ═══════════════════════════════════════════════════════════
    //  FPGA cycle-accurate behavioral simulation
    //  (matches SystemVerilog DeltaKernel; runs today, no tools needed)
    // ═══════════════════════════════════════════════════════════
    std::printf("\n=== FPGA cycle-sim (conflict-delta kernel) ===\n");
    std::printf("target: Alveo U55C @ 350 MHz, LANES=8 per kernel\n");

    // Correctness gate first — any mismatch and we bail loudly.
    DeltaKernelSim sim(/*lanes=*/8, /*clock_mhz=*/350.0, /*pipe_depth=*/6);
    int fpga_mismatch = 0;
    int fpga_check_n = std::min(iters, 5000);
    for (int i = 0; i < fpga_check_n; i++) {
        const auto& p = props[i];
        int old_pid = sol.period_of[p.eid]; if (old_pid < 0) continue;
        int padded = (int)Esimd.adj_other[p.eid].size();
        int32_t ref = conflict_delta_scalar(
            Esimd.adj_other[p.eid].data(), Esimd.adj_cnt[p.eid].data(),
            padded, sol.period_of.data(), old_pid, p.new_pid);
#ifdef EVAL_SIMD_AVX2
        int32_t simd = conflict_delta_simd_isolated(
            Esimd.adj_other[p.eid].data(), Esimd.adj_cnt[p.eid].data(),
            padded, sol.period_of.data(), old_pid, p.new_pid);
        if (simd != ref) { fpga_mismatch++; continue; }
#endif
        int32_t got = sim.process_move(
            Esimd.adj_other[p.eid].data(), Esimd.adj_cnt[p.eid].data(),
            padded, sol.period_of.data(), old_pid, p.new_pid);
        if (got != ref) {
            fpga_mismatch++;
            if (fpga_mismatch <= 3) {
                std::fprintf(stderr,
                    "[fpga] MISMATCH eid=%d old=%d new=%d  ref=%d got=%d\n",
                    p.eid, old_pid, p.new_pid, ref, got);
            }
        }
    }
    if (fpga_mismatch != 0) {
        std::fprintf(stderr,
            "[fpga] FAIL: %d/%d mismatches — cycle-sim results NOT reliable.\n",
            fpga_mismatch, fpga_check_n);
    } else {
        std::printf("  correctness: %d/%d moves match scalar oracle [OK]\n",
                    fpga_check_n, fpga_check_n);
    }

    // Also add pipeline fill once per full batch (amortized over all moves)
    sim.add_pipeline_fill();

    double ns_iter = sim.stats.ns_per_move_iterative();
    double ns_pipe = sim.stats.ns_per_move_pipelined();
    int kernels = 16;
    double ns_pipe_n = sim.stats.ns_per_move_multi_kernel(kernels);

    std::printf("  moves simulated:       %llu\n",
                (unsigned long long)sim.stats.moves_processed);
    std::printf("  total cycles (iter):   %llu\n",
                (unsigned long long)sim.stats.total_cycles_iterative);
    std::printf("  cycles/move avg:       %.2f\n",
                (double)sim.stats.total_cycles_iterative / sim.stats.moves_processed);
    std::printf("  ns/move, 1 kernel iter:      %.2f\n", ns_iter);
    std::printf("  ns/move, 1 kernel pipelined: %.2f  (II=1 steady state)\n", ns_pipe);
    std::printf("  ns/move, %d kernels parallel: %.2f  (realistic Alveo U55C)\n",
                kernels, ns_pipe_n);

    // Fair compare: conflict-count-only, both sides
#ifdef EVAL_SIMD_AVX2
    // Reuse timing harness: just the conflict-count portion on SIMD
    double t_simd_cc = time_ms([&]{
        int32_t s = 0;
        for (auto& p : props) {
            int op = sol.period_of[p.eid]; if (op < 0) continue;
            int padded = (int)Esimd.adj_other[p.eid].size();
            s += conflict_delta_simd_isolated(
                Esimd.adj_other[p.eid].data(), Esimd.adj_cnt[p.eid].data(),
                padded, sol.period_of.data(), op, p.new_pid);
        }
        sink += s;
    }, reps);
    double ns_simd_cc = t_simd_cc * 1e6 / iters;
    std::printf("\n  AVX2 conflict-count (isolated): %.2f ns/move  (wall-clock CPU)\n",
                ns_simd_cc);
    std::printf("  FPGA advantage (1 kernel, pipelined):  %.2fx\n", ns_simd_cc / ns_pipe);
    std::printf("  FPGA advantage (%d kernels, Alveo):     %.2fx\n",
                kernels, ns_simd_cc / ns_pipe_n);
#endif

    // ────────────────────────────────────────────────────────────
    // Section: CUDA CPU-twin validator
    // Compares CudaEvaluator::score_delta_cpu_ref against
    // CachedEvaluator::move_delta on 10k random moves. When mismatches == 0,
    // the CPU twin is bit-exact; the CUDA kernel is a straight transcription
    // of the same math.
    // ────────────────────────────────────────────────────────────
    {
        std::printf("\n=== CUDA CPU-twin validator ===\n");

        FastEvaluator fe_val(P);
        Solution sol_val;
        {
            auto g = solve_greedy(P, false);
            sol_val = g.sol.copy();
        }
        auto ev_val = fe_val.full_eval(sol_val);
        if (!ev_val.feasible()) fe_val.recover_feasibility(sol_val, 500, 42);

        CachedEvaluator Ecach_val(fe_val);
        Ecach_val.initialize(sol_val);
        CudaEvaluator Cuev_val(Ecach_val);
        Cuev_val.sync_state(sol_val);

        std::mt19937 rng_val(42);
        std::uniform_int_distribution<int> de_e(0, P.n_e() - 1);
        std::uniform_int_distribution<int> de_p(0, P.n_p() - 1);
        std::uniform_int_distribution<int> de_r(0, P.n_r() - 1);

        int n_val = 10000;
        int mismatches = 0, n_tested = 0, n_nonzero = 0;
        double max_abs_err = 0.0;
        for (int i = 0; i < n_val; i++) {
            int e = de_e(rng_val);
            if (sol_val.period_of[e] < 0) continue;
            int dur = fe_val.exam_dur[e];
            int enr = fe_val.exam_enroll[e];
            int tp = -1, tr = -1;
            for (int t = 0; t < 10; t++) {
                int cp = de_p(rng_val);
                int cr = de_r(rng_val);
                if (dur <= fe_val.period_dur[cp] && enr <= fe_val.room_cap[cr]) { tp = cp; tr = cr; break; }
            }
            if (tp < 0) continue;
            double d_cached = Ecach_val.move_delta(sol_val, e, tp, tr);
            double d_cpu    = Cuev_val.score_delta_cpu_ref(e, tp, tr);
            double err = std::fabs(d_cached - d_cpu);
            if (err > 1e-6) mismatches++;
            if (err > max_abs_err) max_abs_err = err;
            if (std::fabs(d_cached) > 0.5) n_nonzero++;
            n_tested++;
        }
        std::printf("  moves tested:     %d (nonzero delta: %d)\n", n_tested, n_nonzero);
        std::printf("  mismatches:       %d\n", mismatches);
        std::printf("  max abs error:    %.6f\n", max_abs_err);
        if (mismatches == 0)
            std::printf("  VERDICT:          PASS (CPU twin bit-exact; kernel is safe to port)\n");
        else
            std::printf("  VERDICT:          FAIL — CPU twin diverges from Ecach.move_delta\n");
    }

    // ────────────────────────────────────────────────────────────
    // Section: Placement scorer validator
    // Verifies score_placement_cpu_ref matches the inline cost formula
    // used in alns.h::repair_greedy, bit-exact. Tests the scorer that
    // the ALNS-GPU path will use.
    // ────────────────────────────────────────────────────────────
    {
        std::printf("\n=== Placement-scorer validator ===\n");

        FastEvaluator fe_pl(P);
        Solution sol_pl = solve_greedy(P, false).sol;
        auto ev_pl = fe_pl.full_eval(sol_pl);
        if (!ev_pl.feasible()) fe_pl.recover_feasibility(sol_pl, 500, 42);

        // Destroy K exams to get a sol with unplaced slots; match ALNS setup.
        std::mt19937 rng_pl(99);
        int n_destroy = std::max(2, (int)(P.n_e() * 0.05));
        auto removed = alns_detail::destroy_random(sol_pl, P.n_e(), n_destroy, rng_pl);

        CachedEvaluator Ecach_pl(fe_pl);
        Ecach_pl.initialize(sol_pl);
        CudaEvaluator Cuev_pl(Ecach_pl);
        Cuev_pl.sync_state(sol_pl);

        int n_val = 4000, mismatches = 0, n_tested = 0;
        long long max_abs_err = 0;
        std::uniform_int_distribution<int> de_p(0, P.n_p() - 1);
        std::uniform_int_distribution<int> de_r(0, P.n_r() - 1);

        for (int i = 0; i < n_val && !removed.empty(); i++) {
            int eid = removed[rng_pl() % removed.size()];
            int dur = fe_pl.exam_dur[eid];
            int enr = fe_pl.exam_enroll[eid];
            int tp = -1, tr = -1;
            for (int t = 0; t < 10; t++) {
                int cp = de_p(rng_pl);
                int cr = de_r(rng_pl);
                if (dur <= fe_pl.period_dur[cp] && enr <= fe_pl.room_cap[cr]) { tp = cp; tr = cr; break; }
            }
            if (tp < 0) continue;

            // Inline cost (mirrors repair_greedy exactly)
            long long pcost_ref = 0;
            for (auto& [nb, _] : P.adj[eid]) {
                int nb_pid = sol_pl.period_of[nb];
                if (nb_pid < 0) continue;
                if (nb_pid == tp) pcost_ref += 100000;
                else {
                    int gap = std::abs(tp - nb_pid);
                    if (gap > 0 && gap <= fe_pl.w_spread) pcost_ref += 1;
                    if (fe_pl.period_day[tp] == fe_pl.period_day[nb_pid]) {
                        int g = std::abs(fe_pl.period_daypos[tp] - fe_pl.period_daypos[nb_pid]);
                        if (g == 1) pcost_ref += fe_pl.w_2row;
                        else if (g > 1) pcost_ref += fe_pl.w_2day;
                    }
                }
            }
            pcost_ref += fe_pl.period_pen[tp];
            if (fe_pl.large_exams.count(eid) && fe_pl.fl_penalty > 0 && fe_pl.last_periods.count(tp))
                pcost_ref += fe_pl.fl_penalty;
            if (sol_pl.get_pr_enroll(tp, tr) + fe_pl.exam_enroll[eid] > fe_pl.room_cap[tr])
                pcost_ref += 100000;
            pcost_ref += fe_pl.room_pen[tr];

            long long pcost_cpu = Cuev_pl.score_placement_cpu_ref(eid, tp, tr);
            long long err = std::abs(pcost_cpu - pcost_ref);
            if (err != 0) mismatches++;
            if (err > max_abs_err) max_abs_err = err;
            n_tested++;
        }
        std::printf("  moves tested:     %d\n", n_tested);
        std::printf("  mismatches:       %d\n", mismatches);
        std::printf("  max abs error:    %lld\n", max_abs_err);
        if (mismatches == 0)
            std::printf("  VERDICT:          PASS (placement scorer matches repair_greedy)\n");
        else
            std::printf("  VERDICT:          FAIL\n");
    }

    // ────────────────────────────────────────────────────────────
    // Section: GPU-vs-CPU end-to-end kernel validator
    // Only runs when built with HAVE_CUDA=1 and GPU is present at runtime.
    // Compares kernel output against score_*_cpu_ref for delta, placement,
    // and full-eval. This is the check that the CUDA kernels are faithful
    // transcriptions of the bit-exact-validated CPU twins.
    // ────────────────────────────────────────────────────────────
#ifdef HAVE_CUDA
    {
        std::printf("\n=== GPU-vs-CPU end-to-end kernel validator ===\n");
        FastEvaluator fe_g(P);
        Solution sol_g = solve_greedy(P, false).sol;
        if (!fe_g.full_eval(sol_g).feasible()) fe_g.recover_feasibility(sol_g, 500, 42);
        CachedEvaluator Ecach_g(fe_g);
        Ecach_g.initialize(sol_g);
        CudaEvaluator Cuev_g(Ecach_g);

        std::printf("  GPU active:       %s\n", Cuev_g.gpu_active ? "yes" : "no (skipping)");
        if (!Cuev_g.gpu_active) { std::printf("  (skipped)\n"); }
        else {
            Cuev_g.sync_state(sol_g);

            // Generate a batch of random moves
            std::mt19937 rng_g(7);
            std::uniform_int_distribution<int> de_e(0, P.n_e() - 1);
            std::uniform_int_distribution<int> de_p(0, P.n_p() - 1);
            std::uniform_int_distribution<int> de_r(0, P.n_r() - 1);

            std::vector<int32_t> mv_eid, mv_new_pid, mv_new_rid;
            for (int i = 0; i < 1000; i++) {
                int e = de_e(rng_g);
                if (sol_g.period_of[e] < 0) continue;
                int pp = -1, rr = -1;
                for (int t = 0; t < 10; t++) {
                    int cp = de_p(rng_g); int cr = de_r(rng_g);
                    if (fe_g.exam_dur[e] <= fe_g.period_dur[cp] &&
                        fe_g.exam_enroll[e] <= fe_g.room_cap[cr]) { pp = cp; rr = cr; break; }
                }
                if (pp < 0) continue;
                mv_eid.push_back(e); mv_new_pid.push_back(pp); mv_new_rid.push_back(rr);
            }
            int N = (int)mv_eid.size();

            // GPU batch
            std::vector<double> gpu_out;
            Cuev_g.score_batch(mv_eid, mv_new_pid, mv_new_rid, gpu_out);

            // CPU twin
            int diff = 0; double max_err = 0;
            for (int i = 0; i < N; i++) {
                double d_cpu = Cuev_g.score_delta_cpu_ref(mv_eid[i], mv_new_pid[i], mv_new_rid[i]);
                double err = std::fabs(gpu_out[i] - d_cpu);
                if (err > 1e-3) diff++;
                if (err > max_err) max_err = err;
            }
            std::printf("  move-delta: tested=%d  mismatches=%d  max_err=%.3f  %s\n",
                        N, diff, max_err, (diff == 0 ? "PASS" : "FAIL"));

            // Placement scorer
            std::vector<long long> gpu_pcosts;
            Cuev_g.score_placement_batch(mv_eid, mv_new_pid, mv_new_rid, gpu_pcosts);
            int pdiff = 0; long long pmax = 0;
            for (int i = 0; i < N; i++) {
                long long p_cpu = Cuev_g.score_placement_cpu_ref(mv_eid[i], mv_new_pid[i], mv_new_rid[i]);
                long long err = std::abs(gpu_pcosts[i] - p_cpu);
                if (err != 0) pdiff++;
                if (err > pmax) pmax = err;
            }
            std::printf("  placement:  tested=%d  mismatches=%d  max_err=%lld  %s\n",
                        N, pdiff, pmax, (pdiff == 0 ? "PASS" : "FAIL"));

            // Full-eval batch (N=3 sols — take sol_g three times to test batching)
            std::vector<Solution> sols = {sol_g.copy(), sol_g.copy(), sol_g.copy()};
            std::vector<double> fb;
            Cuev_g.score_full_batch(sols, fb);
            double cpu_ref = Cuev_g.score_full_cpu_ref_adj(sol_g);
            double fe_ref = fe_g.full_eval(sol_g).fitness();
            std::printf("  full-eval:  gpu[0]=%.1f  cpu_twin=%.1f  fe.full_eval=%.1f\n",
                        fb[0], cpu_ref, fe_ref);
            bool match = std::fabs(fb[0] - cpu_ref) < 1e-3 &&
                         std::fabs(fb[1] - cpu_ref) < 1e-3 &&
                         std::fabs(fb[2] - cpu_ref) < 1e-3;
            std::printf("  VERDICT:    %s (kernel==twin across N=3 batch)\n",
                        match ? "PASS" : "FAIL");
        }
    }
#else
    std::printf("\n=== GPU-vs-CPU validator ===\n  HAVE_CUDA not defined — rebuild with HAVE_CUDA=1 to test GPU kernel\n");
#endif

    // ────────────────────────────────────────────────────────────
    // Section: Full-eval CPU-twin validators
    // Tests both twins against fe.full_eval on feasible + infeasible inputs:
    //   • score_full_cpu_ref (per-student): matches on BOTH
    //   • score_full_cpu_ref_adj (adj-based): matches on feasible only
    // ────────────────────────────────────────────────────────────
    {
        std::printf("\n=== Full-eval CPU-twin validators ===\n");

        FastEvaluator fe_fe(P);
        Solution sol_feas = solve_greedy(P, false).sol;
        if (!fe_fe.full_eval(sol_feas).feasible()) fe_fe.recover_feasibility(sol_feas, 500, 42);
        CachedEvaluator Ecach_fe(fe_fe);
        Ecach_fe.initialize(sol_feas);
        CudaEvaluator Cuev_fe(Ecach_fe);

        // Build an INFEASIBLE sol: take feasible and corrupt it by swapping
        // several exams into conflicting periods.
        Solution sol_infeas = sol_feas.copy();
        {
            std::mt19937 rcorrupt(99);
            std::uniform_int_distribution<int> de(0, P.n_e() - 1);
            std::uniform_int_distribution<int> dp(0, P.n_p() - 1);
            for (int k = 0; k < 50; k++) {
                int e = de(rcorrupt);
                int p = dp(rcorrupt);
                int r = sol_infeas.room_of[e]; if (r < 0) r = 0;
                sol_infeas.assign(e, p, r);
            }
        }

        auto test_sol = [&](const Solution& sol, const char* label) {
            Cuev_fe.sync_state(sol);
            double fit_fe  = fe_fe.full_eval(sol).fitness();
            double fit_st  = Cuev_fe.score_full_cpu_ref(sol);
            double fit_adj = Cuev_fe.score_full_cpu_ref_adj(sol);
            std::printf("  %s:\n", label);
            std::printf("    fe.full_eval:           %.1f\n", fit_fe);
            std::printf("    score_full_cpu_ref:     %.1f  diff=%.1f  %s\n",
                        fit_st, std::fabs(fit_fe - fit_st),
                        std::fabs(fit_fe - fit_st) < 1e-3 ? "PASS" : "FAIL");
            std::printf("    score_full_cpu_ref_adj: %.1f  diff=%.1f  %s\n",
                        fit_adj, std::fabs(fit_fe - fit_adj),
                        std::fabs(fit_fe - fit_adj) < 1e-3 ? "(match)" : "(differs — expected on infeasible)");
        };

        test_sol(sol_feas, "feasible sol");
        test_sol(sol_infeas, "infeasible sol (corrupted, 50 random moves)");
    }

    return 0;
}
