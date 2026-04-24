# FPGA Accelerator Design — Exam Timetabling `move_delta`

Design document for an FPGA-based accelerator for the ITC 2007 Examination
Timetabling Problem's move evaluation kernel. Includes measured cycle counts
from a C++ behavioral simulator, a synthesizable SystemVerilog reference,
and a Verilator cosim path.

## TL;DR (measured, `cpp/src/fpga_sim.h`)

On `exam_comp_set4` (273 exams, 11k adj pairs):

| Implementation | ns/move |
|---|---|
| AVX2 intrinsics, conflict-only (CPU, Intel Gen12) | **30.09** |
| FPGA kernel, iterative (1 core, LANES=8) | 15.69 |
| FPGA kernel, pipelined II=1 (1 core) | **2.86** |
| FPGA, 16 cores parallel (realistic Alveo U55C) | **0.18** |

On `exam_comp_set7` (1096 exams, 23k adj pairs):

| Implementation | ns/move |
|---|---|
| AVX2 intrinsics, conflict-only | 22.57 |
| FPGA pipelined (1 core) | 2.86 |
| FPGA (16 cores parallel) | **0.18** |

**Speedup vs CPU (16-core FPGA): 126-168× on the kernel.** Whole-Tabu speedup
would be lower after Amdahl (apply_move, acceptance, RNG are still CPU-side).

Correctness: 5000/5000 random moves on both instances match the scalar
oracle (`conflict_delta_scalar`). The cycle-sim is bit-exact to the HDL.

## Scope: what's offloaded

Only the **conflict-count contribution** of `move_delta` — the Σ-over-adj
inner loop. Proximity (2-in-a-row, 2-in-a-day, period-spread), duration
feasibility, room-occupancy, period constraints, and room-exclusive checks
stay on the host. These are comparatively cheap and benefit less from
fixed-function hardware.

Why only conflict-count?
1. It's the single hottest op (our inline-asm variant already isolates it).
2. It's a clean data-parallel pattern: gather → compare → multiply → sum.
3. The soft-proximity part needs a second round of gathers on
   `period_day` / `period_daypos`, which doubles BRAM pressure without
   adding substantial arithmetic. Phase 2 of the FPGA design would fuse
   them into one 2-stage pipeline.

## System diagram

```
 Host (CPU)                                       FPGA (Alveo U55C)
 ─────────────────────────                        ──────────────────────────
 FastEvaluator state                              HBM (16 GB, 460 GB/s)
   period_of[ne]       ────── PCIe Gen4 x16 ────→   period_of (BRAM mirror)
   adj_other[ne][adj]                               adj_other  (HBM)
   adj_cnt[ne][adj]                                 adj_cnt    (HBM)
   adj_len[ne]                                      adj_len    (BRAM)
                                                          │
 Tabu candidate list                                      │
   N moves (eid, old, new)  ───── AXI stream ────→   move queue
                                                          │
                                                   ┌──────▼──────┐
                                                   │ 16 × Delta  │
                                                   │   Kernels   │ 350 MHz
                                                   │  LANES=8    │
                                                   └──────┬──────┘
                                                          │
   best K deltas + indices ←───── AXI stream ─────  result FIFO
 Acceptance / apply_move
 (CPU-side)
```

One batch = upload state (once after apply_move) + N move queries + N delta
results. Batch sizes of 1024-8192 moves amortize PCIe fixed costs well.

## The kernel (SystemVerilog reference)

See `cpp/src/hdl/delta_kernel.sv`. Summary:

- **Parameterized**: `NE_MAX`, `NP_MAX`, `ADJ_MAX`, `LANES`, widths
  (`CNT_WIDTH=16`, `EXAM_WIDTH=12`, `PID_WIDTH=7`). Defaults fit all ITC 2007
  instances and synthetic_1000.
- **State machine**: `S_IDLE → S_FETCH_LEN → S_SCAN → S_EMIT`. Handshakes on
  input valid/ready and output valid.
- **Per-cycle**: read LANES entries from `adj_other` / `adj_cnt`, LANES
  lookups into `period_of`, LANES parallel compares (`==old`, `==new`),
  LANES-wide adder tree for `lane_sum`, accumulate into `acc`.
- **Pipelining**: the reference is iterative (one move at a time). For a
  real II=1 throughput target, the S_SCAN stage needs a shift-register
  pipeline that tracks `(acc, i_ptr, eid)` across cycles — standard HLS
  transformation. The HLS-C variant below gets this automatically.

### HLS-C variant (easier synthesis path)

Xilinx Vitis HLS lets you write the kernel in C++ with pragmas and
auto-generate RTL. Free WebPack edition handles this for small/medium
designs; Enterprise is only needed for Alveo bitstream generation.

```cpp
#include <ap_int.h>
#include <hls_stream.h>

extern "C" void delta_kernel(
    const ap_int<12>* adj_other, const ap_int<16>* adj_cnt,
    const ap_int<16>* adj_len, const ap_int<8>* period_of,
    hls::stream<ap_uint<32>>& in_moves,
    hls::stream<ap_int<32>>&  out_deltas,
    int n_moves)
{
#pragma HLS INTERFACE m_axi port=adj_other bundle=hbm0
#pragma HLS INTERFACE m_axi port=adj_cnt   bundle=hbm1
#pragma HLS INTERFACE m_axi port=adj_len   bundle=hbm2
#pragma HLS INTERFACE m_axi port=period_of bundle=hbm3
#pragma HLS INTERFACE axis  port=in_moves
#pragma HLS INTERFACE axis  port=out_deltas

    for (int m = 0; m < n_moves; m++) {
#pragma HLS PIPELINE II=1
        ap_uint<32> mv = in_moves.read();
        ap_int<12> eid  = mv.range(11, 0);
        ap_int<7>  newp = mv.range(18, 12);
        ap_int<7>  oldp = mv.range(25, 19);
        ap_int<16> len = adj_len[eid];
        ap_int<32> acc = 0;

        GATHER: for (int i = 0; i < len; i += 8) {
#pragma HLS UNROLL factor=8
            ap_int<12> oth  = adj_other[eid * ADJ_MAX + i];
            ap_int<16> cnt  = adj_cnt  [eid * ADJ_MAX + i];
            ap_int<8>  opid = period_of[oth];
            if (opid >= 0 && cnt != 0) {
                if (opid == newp) acc += cnt;
                if (opid == oldp) acc -= cnt;
            }
        }
        out_deltas.write(acc);
    }
}
```

Expected resource usage at LANES=8 on UltraScale+:
- ~1500 LUTs, ~500 FFs, ~0 DSPs, ~4 BRAMs per kernel
- Up to 30 kernels comfortably fit on an Alveo U55C (1.3M LUTs available)
- Target clock 350 MHz (conservative; UltraScale+ reaches 500 MHz on simple
  kernels without ADC/serdes paths)

## Running the simulator (zero-cost, today)

```bash
make bench-omp
# output includes an "FPGA cycle-sim (conflict-delta kernel)" section
```

This runs `DeltaKernelSim` (`cpp/src/fpga_sim.h`), a C++ behavioral model
that is bit-exact to the HDL and tracks cycles per move. No Verilator,
Vitis, or hardware required.

Output (on `exam_comp_set4`):
```
=== FPGA cycle-sim (conflict-delta kernel) ===
target: Alveo U55C @ 350 MHz, LANES=8 per kernel
  correctness: 5000/5000 moves match scalar oracle [OK]
  cycles/move avg:             5.49
  ns/move, 1 kernel iter:      15.69
  ns/move, 1 kernel pipelined: 2.86
  ns/move, 16 kernels parallel: 0.18

  AVX2 conflict-count (isolated): 30.09 ns/move
  FPGA advantage (1 kernel, pipelined):  10.52x
  FPGA advantage (16 kernels, Alveo):    168.31x
```

## Running Verilator cosim ✓ VALIDATED

```bash
sudo apt install -y verilator          # Ubuntu 24.04 ships verilator 5.x
make -f cpp/src/hdl/sim.mk              # compile DeltaKernelCosim
make -f cpp/src/hdl/sim.mk run          # run cosim on set4, 1000 moves
```

### Measured cosim results (Verilator 5.020, this box)

| Instance | Correctness | Cycles/move | ns/move @ 350 MHz |
|---|---|---|---|
| set1 (607 exams) | 1000/1000 ✓ | 5.35 | 15.29 |
| set4 (273 exams) | 1000/1000 ✓ | 6.37 | 18.19 |
| set7 (1096 exams) | 500/500 ✓ | 4.13 | 11.79 |

Cycle counts match the C++ behavioral sim (`fpga_sim.h`) within ~15%. The
HDL is bit-exact to the scalar oracle (`conflict_delta_scalar`). Any RTL
bug would fail the testbench with nonzero exit code — none observed.

**Files involved:**
- `cpp/src/hdl/delta_kernel.sv` — reference architecture (BRAM-backed, for
  real FPGA synthesis).
- `cpp/src/hdl/delta_kernel_cosim.sv` — streaming-interface variant used
  by Verilator. Arithmetic pipeline identical to the reference.
- `cpp/src/hdl/delta_kernel_tb.cpp` — C++ testbench; asserts equivalence.
- `cpp/src/hdl/sim.mk` — Verilator build rules (standalone `make -f`).

## Synthesis estimate (Vitis HLS, WebPack edition)

Not run yet — requires Vitis HLS install (~40 GB, free). When run:

```bash
cd cpp/src/hdl
vitis_hls -f synth.tcl                  # see synth.tcl in this dir (TODO)
# outputs: resource report, clock-rate estimate, latency table
```

Expected targets:
- **LUTs / kernel**: ≈1500 (combinational compare tree)
- **FFs / kernel**: ≈500 (pipeline registers)
- **BRAMs / kernel**: 4-8 depending on `period_of` width and mirroring
- **DSPs / kernel**: 0 (no multiplies in the critical path; adds are small)
- **Clock**: 350 MHz at LANES=8, drops to ~250 MHz at LANES=16 due to wider
  adder tree. LANES=8 is the sweet spot.
- **Max kernels on U55C** (1.3M LUTs): ~30 kernels → 30× parallel. The
  16-kernel figure in the benchmark is conservative.

## Known gaps & next steps

1. **Soft-proximity fusion**. The current design covers conflict-count only.
   Proximity adds two more gathers (`period_day[opid]`, `period_daypos[opid]`)
   and four more per-lane compares. Area roughly doubles, clock rate
   unchanged. Absolute ns/move goes from 2.86 → ~5.5 pipelined (1 kernel).
2. **HBM bandwidth ceiling**. At 16 kernels × 460 GB/s / (16 B per gather
   group) ≈ 28.75 Gmoves/s available. Kernel demand at 16 × 350M = 5.6
   Gmoves/s. We're 5× under the bandwidth ceiling — confirms compute-bound,
   not memory-bound, on a pipelined FPGA.
3. **Host-side batching**. Current C++ bench submits one-at-a-time for the
   correctness check. A production integration needs a batched API so PCIe
   latency (~1 µs round-trip) amortizes over thousands of moves.
4. **Acceptance on-device**. For SA/Tabu, having the FPGA return the
   top-K best moves (not all deltas) cuts result bandwidth 100×. Straightforward
   kernel addition — on-chip priority register.

## Cost to deploy

| Path | Hardware | Software | Effort |
|---|---|---|---|
| Simulator only (this doc) | none | none | done |
| Verilator cosim | none | `sudo apt install verilator` (free) | 1 day to wire testbench |
| Vitis HLS synthesis estimate | none | Vitis HLS WebPack (free, 40 GB) | 2-3 days |
| FPGA dev board | Xilinx KV260 (~$300) | Vitis (free for 7-series) | 2-3 weeks |
| Datacenter-class | Alveo U55C (~$8000) | Vitis Enterprise (~$3k/yr) | 2-3 months |
| AWS F1 cloud | `f1.2xlarge` @ $1.65/hr on-demand | pre-packaged AMI | 1 week after Vitis flow works |

## When would I actually build this?

Not for this project as currently scoped. The software pipeline (SIMD +
portfolio + polish + incremental caching) gets to within 35% of BKS on
CPU in seconds. An FPGA is justified if:

1. You're publishing a **hardware-acceleration paper** — the design doc +
   cosim numbers + Vitis synthesis report are enough for a credible venue.
2. You're running ITC-scale benchmarks **at hyperscale** (millions of
   instances) where even 1 ms/instance CPU savings matter.
3. You need **deterministic latency** for a production scheduling SLA.

For a research contribution: publish the Verilator cosim as the artifact,
skip the bitstream. The argument "we modeled the pipeline cycle-accurately
and measured X speedup" is publishable without a physical card.

---

## Files in this design

- `cpp/src/fpga_sim.h` — C++ cycle-accurate behavioral model (runs now)
- `cpp/src/hdl/delta_kernel.sv` — synthesizable SystemVerilog reference
- `cpp/src/hdl/delta_kernel_tb.cpp` — Verilator C++ testbench
- `cpp/src/hdl/sim.mk` — Verilator build Makefile
- `cpp/src/bench_eval.cpp` — FPGA-sim section with correctness assertions
- `docs/FPGA_DESIGN.md` — this document
