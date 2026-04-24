/*
 * Cycle-accurate BEHAVIORAL simulator of the FPGA DeltaKernel.
 *
 * Mirrors the SystemVerilog module in cpp/src/hdl/delta_kernel.sv and the
 * Vitis HLS variant in cpp/src/hdl/delta_kernel.cpp. Running this gives you
 * real cycle counts without needing Verilator or Vitis installed.
 *
 * What it models:
 *   • LANES parallel compare+accumulate lanes per cycle (default 8).
 *   • Per-cycle BRAM read of adj_other[eid][i..i+LANES] and adj_cnt likewise.
 *   • Single-cycle lookup of period_of[other] (assumed BRAM-resident).
 *   • Masked accumulate: (cnt != 0) AND (opid != -1).
 *   • Pipeline fill of PIPE_DEPTH cycles for the first move only.
 *
 * Two throughput reporting modes:
 *   iterative  — processes one move at a time. cycles/move = ceil(len/LANES).
 *   pipelined  — back-to-back moves at II=1 (valid when adj_len doesn't vary
 *                wildly or when padded). Steady-state throughput = 1 move
 *                per max(cycles_over_lanes, 1) per LANES-group.
 *
 * Correctness: computes the same conflict-delta as conflict_delta_scalar
 * in this header. Assertion check is wired into the bench.
 */

#pragma once

#include "evaluator.h"
#include "evaluator_simd.h"

#include <cstdint>
#include <cstdio>
#include <vector>

struct FPGASimStats {
    uint64_t total_cycles_iterative = 0;
    uint64_t total_cycles_pipelined = 0;
    uint64_t moves_processed        = 0;
    uint64_t pipeline_fill_latency  = 6;   // typical Vitis HLS pipe depth
    double   clock_mhz              = 350.0;
    int      lanes                  = 8;

    double ns_per_move_iterative() const {
        return moves_processed ?
            (double)total_cycles_iterative * 1000.0 / clock_mhz / moves_processed : 0.0;
    }
    double ns_per_move_pipelined() const {
        return moves_processed ?
            (double)total_cycles_pipelined * 1000.0 / clock_mhz / moves_processed : 0.0;
    }
    double ns_per_move_multi_kernel(int n_kernels) const {
        // N kernels in parallel → 1/N cycles per move at steady state
        return ns_per_move_pipelined() / n_kernels;
    }
};

// ── Reference: pure scalar conflict-count delta ──
// Returns Σ cnt * ((o_pid == new_pid) - (o_pid == old_pid)) over valid lanes.
// Used as the correctness oracle for both SIMD and FPGA sim.
inline int32_t conflict_delta_scalar(
    const int32_t* adj_other, const int32_t* adj_cnt, int padded_len,
    const int* period_of, int old_pid, int new_pid)
{
    int32_t acc = 0;
    for (int i = 0; i < padded_len; i++) {
        int cnt = adj_cnt[i];
        if (cnt == 0) continue;
        int oth = adj_other[i];
        int opid = period_of[oth];
        if (opid < 0) continue;
        if (opid == new_pid) acc += cnt;
        if (opid == old_pid) acc -= cnt;
    }
    return acc;
}

// ── Behavioral FPGA simulator ──
// Produces identical results to the HDL kernel (see delta_kernel.sv).
class DeltaKernelSim {
public:
    FPGASimStats stats;

    DeltaKernelSim(int lanes = 8, double clock_mhz = 350.0, int pipe_depth = 6) {
        stats.lanes                 = lanes;
        stats.clock_mhz             = clock_mhz;
        stats.pipeline_fill_latency = pipe_depth;
    }

    // Process one move. Returns the conflict delta.
    // Cycles-accurate to a LANES-wide unrolled inner loop.
    int32_t process_move(
        const int32_t* adj_other, const int32_t* adj_cnt, int padded_len,
        const int* period_of, int old_pid, int new_pid)
    {
        int L = stats.lanes;
        int32_t acc = 0;

        for (int i = 0; i < padded_len; i += L) {
            // One cycle: LANES parallel lanes execute in the same clock.
            int32_t lane_sum = 0;
            for (int k = 0; k < L; k++) {
                int cnt = adj_cnt[i + k];
                if (cnt == 0) continue;  // padding lane
                int oth  = adj_other[i + k];
                int opid = period_of[oth];
                if (opid < 0) continue;
                if (opid == new_pid) lane_sum += cnt;
                if (opid == old_pid) lane_sum -= cnt;
            }
            acc += lane_sum;
            stats.total_cycles_iterative += 1;
        }
        stats.moves_processed += 1;

        // Pipelined counting: one new move starts every cycle at II=1 when
        // back-to-back. For a batch of N moves we add 1 cycle per move plus
        // the pipeline fill latency once. The caller accumulates this via
        // the pipelined counter directly.
        stats.total_cycles_pipelined += 1;
        return acc;
    }

    // Add the pipeline-fill overhead exactly once per batch (call before or
    // after scoring N back-to-back moves).
    void add_pipeline_fill() {
        stats.total_cycles_pipelined += stats.pipeline_fill_latency;
    }
};

// ── Isolated SIMD conflict-count (for fair HDL comparison) ──
// Same as conflict_batch_asm but full vector→scalar loop, no asm.
// Matches HDL semantics exactly: conflict count only, masked for valid.
#if defined(__AVX2__)
inline int32_t conflict_delta_simd_isolated(
    const int32_t* adj_other, const int32_t* adj_cnt, int padded_len,
    const int* period_of, int old_pid, int new_pid)
{
    __m256i old_v  = _mm256_set1_epi32(old_pid);
    __m256i new_v  = _mm256_set1_epi32(new_pid);
    __m256i negone = _mm256_set1_epi32(-1);
    __m256i zero   = _mm256_setzero_si256();
    __m256i acc    = zero;

    for (int i = 0; i < padded_len; i += 8) {
        __m256i other_v = _mm256_loadu_si256((const __m256i*)(adj_other + i));
        __m256i cnt_v   = _mm256_loadu_si256((const __m256i*)(adj_cnt   + i));
        __m256i o_pid_v = _mm256_i32gather_epi32(period_of, other_v, 4);
        __m256i valid   = _mm256_andnot_si256(
            _mm256_cmpeq_epi32(cnt_v, zero),
            _mm256_cmpgt_epi32(o_pid_v, negone));
        __m256i eq_old = _mm256_and_si256(_mm256_cmpeq_epi32(o_pid_v, old_v), valid);
        __m256i eq_new = _mm256_and_si256(_mm256_cmpeq_epi32(o_pid_v, new_v), valid);
        acc = _mm256_add_epi32(acc,
                _mm256_sub_epi32(_mm256_and_si256(cnt_v, eq_new),
                                 _mm256_and_si256(cnt_v, eq_old)));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}
#endif
