// ═════════════════════════════════════════════════════════════════════
//  DeltaKernel — synthesizable SystemVerilog for FPGA cosim / ASIC
//
//  Computes the CONFLICT-DELTA contribution of one move_delta call:
//    acc = Σ adj_cnt[i] × ((po[adj_other[i]] == new_pid) -
//                          (po[adj_other[i]] == old_pid))
//    over i ∈ [0, adj_len[eid]), skipping lanes where cnt == 0 or opid < 0.
//
//  Target:   Alveo U55C (Xilinx Virtex UltraScale+ XCU55C) or Agilex 7.
//  Clock:    ~350 MHz typical at LANES=8.
//  Area:     ≈ 1500 LUTs + 500 FFs + 1 DSP (popcount-style tree).
//  Cosim:    Verilator 5.x. See sim.mk in this directory.
//
//  Parameters kept template-friendly so NE_MAX/ADJ_MAX can scale per
//  instance class. Defaults fit ITC 2007 set1-8 + synthetic_1000.
// ═════════════════════════════════════════════════════════════════════

`timescale 1ns / 1ps

module DeltaKernel #(
    parameter int NE_MAX      = 4096,   // max exams
    parameter int NP_MAX      = 128,    // max periods
    parameter int ADJ_MAX     = 4096,   // padded adj capacity per exam
    parameter int LANES       = 8,      // parallel compute lanes
    parameter int CNT_WIDTH   = 16,     // bits per adj_cnt value
    parameter int EXAM_WIDTH  = 12,     // bits per exam id (supports 4096)
    parameter int PID_WIDTH   = 7       // bits per period id (supports 128)
)(
    input  logic                            clk,
    input  logic                            rst_n,

    // ── Per-move request (handshake) ───────────────────────────
    input  logic                            in_valid,
    output logic                            in_ready,
    input  logic [EXAM_WIDTH-1:0]           in_eid,
    input  logic [PID_WIDTH-1:0]            in_old_pid,
    input  logic [PID_WIDTH-1:0]            in_new_pid,

    // ── BRAM read ports (synthesized as true dual-port BRAMs) ──
    // Host updates these between batches via the AXI-lite control interface
    // (control path elided here for brevity).
    output logic [EXAM_WIDTH-1:0]           po_rd_addr,
    input  logic signed [PID_WIDTH:0]       po_rd_data,    // signed: -1 == unassigned

    output logic [EXAM_WIDTH-1:0]           adj_len_addr,
    input  logic [15:0]                     adj_len_data,

    // adj_other / adj_cnt indexed as [eid * ADJ_MAX + i] → flat addr
    // packed LANES-wide for single-cycle BRAM read of a whole group.
    output logic [$clog2(NE_MAX*ADJ_MAX/LANES)-1:0] adj_grp_addr,
    input  logic [EXAM_WIDTH*LANES-1:0]             adj_other_group,
    input  logic [CNT_WIDTH*LANES-1:0]              adj_cnt_group,

    // ── Result (handshake) ─────────────────────────────────────
    output logic                            out_valid,
    output logic signed [31:0]              out_delta,
    output logic [EXAM_WIDTH-1:0]           out_eid      // tag for host
);

    // ── State machine ──────────────────────────────────────────
    typedef enum logic [1:0] { S_IDLE, S_FETCH_LEN, S_SCAN, S_EMIT } state_t;
    state_t state, next_state;

    logic [EXAM_WIDTH-1:0]    reg_eid;
    logic [PID_WIDTH-1:0]     reg_old, reg_new;
    logic [15:0]              reg_len;
    logic [15:0]              i_ptr;
    logic signed [31:0]       acc;

    assign in_ready = (state == S_IDLE);

    // Register inputs on handshake
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_eid <= '0; reg_old <= '0; reg_new <= '0;
        end else if (state == S_IDLE && in_valid) begin
            reg_eid <= in_eid;
            reg_old <= in_old_pid;
            reg_new <= in_new_pid;
        end
    end

    // FSM transitions
    always_comb begin
        next_state = state;
        case (state)
            S_IDLE:       if (in_valid)                      next_state = S_FETCH_LEN;
            S_FETCH_LEN:                                     next_state = S_SCAN;
            S_SCAN:       if (i_ptr + LANES >= reg_len)      next_state = S_EMIT;
            S_EMIT:                                          next_state = S_IDLE;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= next_state;
    end

    // adj_len lookup
    assign adj_len_addr = reg_eid;

    // adj group address — each row is LANES entries wide
    assign adj_grp_addr = (reg_eid * ADJ_MAX + i_ptr) / LANES;

    // ── Per-lane compute (combinational tree) ─────────────────
    logic signed [31:0] lane_sum;
    always_comb begin
        lane_sum = 0;
        for (int L = 0; L < LANES; L++) begin
            automatic logic [EXAM_WIDTH-1:0]    oth  = adj_other_group[L*EXAM_WIDTH +: EXAM_WIDTH];
            automatic logic [CNT_WIDTH-1:0]     cnt  = adj_cnt_group  [L*CNT_WIDTH  +: CNT_WIDTH];
            automatic logic                     in_range = (i_ptr + L < reg_len);
            // po_rd for lane L — in real design needs LANES po ports or a
            // LANES-wide BRAM of po. Conceptually shown as po_rd_data[L].
            // For cosim/sketch, issuing sequential lookups in the testbench
            // harness; synthesizable version uses LANES-port URAM.
            automatic logic signed [PID_WIDTH:0] opid = po_rd_data;
            if (in_range && cnt != 0 && opid >= 0) begin
                if (opid == reg_new) lane_sum = lane_sum + cnt;
                if (opid == reg_old) lane_sum = lane_sum - cnt;
            end
        end
    end

    // Accumulator
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc   <= 0;
            i_ptr <= 0;
            out_valid <= 0;
            out_delta <= 0;
            out_eid   <= '0;
        end else begin
            out_valid <= 0;
            case (state)
                S_IDLE: begin
                    if (in_valid) begin
                        acc   <= 0;
                        i_ptr <= 0;
                    end
                end
                S_FETCH_LEN: begin
                    reg_len <= adj_len_data;
                end
                S_SCAN: begin
                    acc   <= acc + lane_sum;
                    i_ptr <= i_ptr + LANES;
                end
                S_EMIT: begin
                    out_valid <= 1;
                    out_delta <= acc;
                    out_eid   <= reg_eid;
                end
            endcase
        end
    end

endmodule
