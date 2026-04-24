// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VDeltaKernelCosim.h for the primary calling header

#ifndef VERILATED_VDELTAKERNELCOSIM___024ROOT_H_
#define VERILATED_VDELTAKERNELCOSIM___024ROOT_H_  // guard

#include "verilated.h"


class VDeltaKernelCosim__Syms;

class alignas(VL_CACHE_LINE_BYTES) VDeltaKernelCosim___024root final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VL_IN8(clk,0,0);
    VL_IN8(rst_n,0,0);
    VL_IN8(start,0,0);
    VL_OUT8(busy,0,0);
    VL_IN8(in_old_pid,7,0);
    VL_IN8(in_new_pid,7,0);
    VL_IN8(in_group_valid,0,0);
    VL_OUT8(out_valid,0,0);
    CData/*7:0*/ DeltaKernelCosim__DOT__reg_old;
    CData/*7:0*/ DeltaKernelCosim__DOT__reg_new;
    CData/*0:0*/ DeltaKernelCosim__DOT__scanning;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VicoFirstIteration;
    CData/*0:0*/ __Vtrigprevexpr___TOP__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__rst_n__0;
    CData/*0:0*/ __VactContinue;
    VL_IN16(in_adj_len,15,0);
    SData/*15:0*/ DeltaKernelCosim__DOT__i_ptr;
    SData/*15:0*/ DeltaKernelCosim__DOT__reg_len;
    VL_OUT(out_delta,31,0);
    IData/*31:0*/ DeltaKernelCosim__DOT__acc;
    IData/*31:0*/ DeltaKernelCosim__DOT__lane_sum;
    IData/*31:0*/ __VactIterCount;
    VL_IN16(in_adj_other[8],11,0);
    VL_IN16(in_adj_cnt[8],15,0);
    VL_IN8(in_po_of[8],7,0);
    VlTriggerVec<1> __VstlTriggered;
    VlTriggerVec<1> __VicoTriggered;
    VlTriggerVec<1> __VactTriggered;
    VlTriggerVec<1> __VnbaTriggered;

    // INTERNAL VARIABLES
    VDeltaKernelCosim__Syms* const vlSymsp;

    // CONSTRUCTORS
    VDeltaKernelCosim___024root(VDeltaKernelCosim__Syms* symsp, const char* v__name);
    ~VDeltaKernelCosim___024root();
    VL_UNCOPYABLE(VDeltaKernelCosim___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
