// ═════════════════════════════════════════════════════════════════════
//  DeltaKernelCosim — Verilator-friendly, self-contained variant.
//
//  Purpose: cosim against C++ scalar oracle to validate the arithmetic
//  pipeline. Testbench feeds one LANES-wide group per cycle; DUT
//  accumulates and emits the final delta when 'done' is asserted.
//
//  For real FPGA synthesis, see delta_kernel.sv (BRAM-backed variant).
//  The arithmetic pipeline here is identical; only the memory topology
//  differs.
// ═════════════════════════════════════════════════════════════════════

`timescale 1ns / 1ps

module DeltaKernelCosim #(
    parameter int LANES = 8
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // Start pulse: capture old/new pid, zero accumulator.
    input  logic                        start,
    output logic                        busy,

    // Per-move scalars (held stable during compute)
    input  logic signed [7:0]           in_old_pid,
    input  logic signed [7:0]           in_new_pid,
    input  logic [15:0]                 in_adj_len,

    // Stream interface — testbench drives one group per cycle when busy.
    // The testbench looks up po_of_lane from its own shadow state.
    input  logic                        in_group_valid,
    input  logic signed [11:0]          in_adj_other [0:LANES-1],
    input  logic        [15:0]          in_adj_cnt   [0:LANES-1],
    input  logic signed [7:0]           in_po_of     [0:LANES-1],

    // Result
    output logic                        out_valid,
    output logic signed [31:0]          out_delta
);

    logic signed [31:0]  acc;
    logic [15:0]         i_ptr;
    logic signed [7:0]   reg_old, reg_new;
    logic [15:0]         reg_len;
    logic                scanning;

    // Combinational lane sum
    logic signed [31:0] lane_sum;
    always_comb begin
        lane_sum = 0;
        for (int L = 0; L < LANES; L++) begin
            logic signed [31:0] cnt_ext;
            cnt_ext = {{16{1'b0}}, in_adj_cnt[L]};
            if ((i_ptr + L < reg_len) &&
                (in_adj_cnt[L] != 16'd0) &&
                (in_po_of[L]   >= 8'sd0)) begin
                if (in_po_of[L] == reg_new) lane_sum = lane_sum + cnt_ext;
                if (in_po_of[L] == reg_old) lane_sum = lane_sum - cnt_ext;
            end
        end
    end

    assign busy = scanning;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc       <= 32'sd0;
            i_ptr     <= 16'd0;
            reg_old   <= 8'sd0;
            reg_new   <= 8'sd0;
            reg_len   <= 16'd0;
            scanning  <= 1'b0;
            out_valid <= 1'b0;
            out_delta <= 32'sd0;
        end else begin
            out_valid <= 1'b0;
            if (start && !scanning) begin
                acc      <= 32'sd0;
                i_ptr    <= 16'd0;
                reg_old  <= in_old_pid;
                reg_new  <= in_new_pid;
                reg_len  <= in_adj_len;
                scanning <= (in_adj_len != 16'd0);
                if (in_adj_len == 16'd0) begin
                    out_valid <= 1'b1;
                    out_delta <= 32'sd0;
                end
            end else if (scanning && in_group_valid) begin
                acc   <= acc + lane_sum;
                i_ptr <= i_ptr + 16'(LANES);
                if (i_ptr + 16'(LANES) >= reg_len) begin
                    scanning  <= 1'b0;
                    out_valid <= 1'b1;
                    out_delta <= acc + lane_sum;
                end
            end
        end
    end

endmodule
