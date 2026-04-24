// ═════════════════════════════════════════════════════════════════════
//  Verilator testbench for DeltaKernelCosim.
//
//  Flow per move:
//    1. Load a move (eid, old_pid, new_pid).
//    2. Pulse start, then drive LANES-wide groups of adj data each cycle,
//       looking up po[adj_other[lane]] from the host-side shadow.
//    3. Count cycles between start and out_valid.
//    4. Compare DUT out_delta against conflict_delta_scalar.
//  Fails loudly with nonzero exit code on any mismatch.
// ═════════════════════════════════════════════════════════════════════

// -I cpp/src passed by sim.mk
#include "parser.h"
#include "models.h"
#include "evaluator.h"
#include "evaluator_simd.h"
#include "greedy.h"
#include "fpga_sim.h"

#include "VDeltaKernelCosim.h"
#include "verilated.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>

static inline void tick(VDeltaKernelCosim* dut, uint64_t& cycles) {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    cycles++;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    std::unique_ptr<VDeltaKernelCosim> dut(new VDeltaKernelCosim);

    std::string path = argc >= 2 ? argv[1] : "instances/exam_comp_set4.exam";
    int n_moves      = argc >= 3 ? std::atoi(argv[2]) : 1000;
    const int LANES  = 8;

    auto P = parser::parse_exam_file(path);
    P.build_derived();
    auto g = solve_greedy(P, false);
    Solution sol = g.sol;
    FastEvaluator fe(P);
    FastEvaluatorSIMD fes(fe);

    std::mt19937 rng(42);
    std::vector<int> assigned;
    for (int e = 0; e < P.n_e(); e++) if (sol.period_of[e] >= 0) assigned.push_back(e);
    std::uniform_int_distribution<int> de(0, (int)assigned.size() - 1);
    std::uniform_int_distribution<int> dp(0, P.n_p() - 1);

    // Reset
    dut->clk = 0; dut->rst_n = 0;
    dut->start = 0; dut->in_group_valid = 0;
    for (int i = 0; i < 4; i++) {
        dut->clk = !dut->clk; dut->eval();
    }
    dut->rst_n = 1;
    dut->clk = 0; dut->eval();

    uint64_t cycles = 0;
    int ok = 0, bad = 0;

    auto t0 = std::chrono::steady_clock::now();
    for (int m = 0; m < n_moves; m++) {
        int e = assigned[de(rng)];
        int old_pid = sol.period_of[e];
        int new_pid = dp(rng);

        const int32_t* other = fes.adj_other[e].data();
        const int32_t* cnt   = fes.adj_cnt[e].data();
        int padded           = (int)fes.adj_other[e].size();
        int len              = padded;   // already padded; DUT uses adj_len

        int32_t ref = conflict_delta_scalar(other, cnt, padded,
                                            sol.period_of.data(),
                                            old_pid, new_pid);

        // Pulse start
        dut->start      = 1;
        dut->in_old_pid = old_pid;
        dut->in_new_pid = new_pid;
        dut->in_adj_len = len;
        dut->in_group_valid = 0;
        tick(dut.get(), cycles);
        dut->start = 0;

        // Feed groups until out_valid
        int group_idx = 0;
        int guard = 0;
        while (!dut->out_valid && guard < 4096) {
            // Drive next group
            for (int L = 0; L < LANES; L++) {
                int i = group_idx * LANES + L;
                int oth_val = (i < padded) ? other[i] : 0;
                int cnt_val = (i < padded) ? cnt[i]   : 0;
                int po_val  = (oth_val >= 0 && oth_val < P.n_e())
                                ? sol.period_of[oth_val] : -1;
                dut->in_adj_other[L] = oth_val;
                dut->in_adj_cnt[L]   = cnt_val;
                dut->in_po_of[L]     = po_val;
            }
            dut->in_group_valid = 1;
            tick(dut.get(), cycles);
            group_idx++;
            guard++;
        }
        dut->in_group_valid = 0;

        int32_t got = dut->out_delta;
        if (got == ref) ok++; else {
            bad++;
            if (bad <= 3) std::fprintf(stderr,
                "MISMATCH eid=%d old=%d new=%d  ref=%d got=%d\n",
                e, old_pid, new_pid, ref, got);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(t1 - t0).count();

    std::printf("Verilator cosim: %d/%d correct\n", ok, n_moves);
    std::printf("  total sim cycles:    %llu\n", (unsigned long long)cycles);
    std::printf("  avg cycles/move:     %.2f\n", (double)cycles / n_moves);
    std::printf("  @ 350 MHz ns/move:   %.2f\n", (double)cycles * 1000.0 / 350.0 / n_moves);
    std::printf("  wall sim time:       %.2f s  (simulator overhead)\n", wall_s);
    return bad ? 1 : 0;
}
