// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VDeltaKernelCosim.h for the primary calling header

#include "VDeltaKernelCosim__pch.h"
#include "VDeltaKernelCosim___024root.h"

VL_ATTR_COLD void VDeltaKernelCosim___024root___eval_static(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_static\n"); );
}

VL_ATTR_COLD void VDeltaKernelCosim___024root___eval_initial(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_initial\n"); );
    // Body
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = vlSelf->clk;
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = vlSelf->rst_n;
}

VL_ATTR_COLD void VDeltaKernelCosim___024root___eval_final(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_final\n"); );
}

#ifdef VL_DEBUG
VL_ATTR_COLD void VDeltaKernelCosim___024root___dump_triggers__stl(VDeltaKernelCosim___024root* vlSelf);
#endif  // VL_DEBUG
VL_ATTR_COLD bool VDeltaKernelCosim___024root___eval_phase__stl(VDeltaKernelCosim___024root* vlSelf);

VL_ATTR_COLD void VDeltaKernelCosim___024root___eval_settle(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_settle\n"); );
    // Init
    IData/*31:0*/ __VstlIterCount;
    CData/*0:0*/ __VstlContinue;
    // Body
    __VstlIterCount = 0U;
    vlSelf->__VstlFirstIteration = 1U;
    __VstlContinue = 1U;
    while (__VstlContinue) {
        if (VL_UNLIKELY((0x64U < __VstlIterCount))) {
#ifdef VL_DEBUG
            VDeltaKernelCosim___024root___dump_triggers__stl(vlSelf);
#endif
            VL_FATAL_MT("/home/hoang/Developer/cli/claude/personal_proj/exam-scheduling/cpp/src/hdl/delta_kernel_cosim.sv", 15, "", "Settle region did not converge.");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        __VstlContinue = 0U;
        if (VDeltaKernelCosim___024root___eval_phase__stl(vlSelf)) {
            __VstlContinue = 1U;
        }
        vlSelf->__VstlFirstIteration = 0U;
    }
}

#ifdef VL_DEBUG
VL_ATTR_COLD void VDeltaKernelCosim___024root___dump_triggers__stl(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VstlTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        VL_DBG_MSGF("         'stl' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void VDeltaKernelCosim___024root___stl_sequent__TOP__0(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___stl_sequent__TOP__0\n"); );
    // Init
    IData/*31:0*/ DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext;
    DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext = 0;
    // Body
    vlSelf->busy = vlSelf->DeltaKernelCosim__DOT__scanning;
    vlSelf->DeltaKernelCosim__DOT__lane_sum = 0U;
    DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext 
        = vlSelf->in_adj_cnt[0U];
    if (((((IData)(vlSelf->DeltaKernelCosim__DOT__i_ptr) 
           < (IData)(vlSelf->DeltaKernelCosim__DOT__reg_len)) 
          & (0U != vlSelf->in_adj_cnt[0U])) & VL_LTES_III(8, 0U, 
                                                          vlSelf->in_po_of
                                                          [0U]))) {
        if ((vlSelf->in_po_of[0U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_new))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   + DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
        if ((vlSelf->in_po_of[0U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_old))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   - DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
    }
    DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext 
        = vlSelf->in_adj_cnt[1U];
    if ((((((IData)(1U) + (IData)(vlSelf->DeltaKernelCosim__DOT__i_ptr)) 
           < (IData)(vlSelf->DeltaKernelCosim__DOT__reg_len)) 
          & (0U != vlSelf->in_adj_cnt[1U])) & VL_LTES_III(8, 0U, 
                                                          vlSelf->in_po_of
                                                          [1U]))) {
        if ((vlSelf->in_po_of[1U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_new))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   + DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
        if ((vlSelf->in_po_of[1U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_old))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   - DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
    }
    DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext 
        = vlSelf->in_adj_cnt[2U];
    if ((((((IData)(2U) + (IData)(vlSelf->DeltaKernelCosim__DOT__i_ptr)) 
           < (IData)(vlSelf->DeltaKernelCosim__DOT__reg_len)) 
          & (0U != vlSelf->in_adj_cnt[2U])) & VL_LTES_III(8, 0U, 
                                                          vlSelf->in_po_of
                                                          [2U]))) {
        if ((vlSelf->in_po_of[2U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_new))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   + DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
        if ((vlSelf->in_po_of[2U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_old))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   - DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
    }
    DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext 
        = vlSelf->in_adj_cnt[3U];
    if ((((((IData)(3U) + (IData)(vlSelf->DeltaKernelCosim__DOT__i_ptr)) 
           < (IData)(vlSelf->DeltaKernelCosim__DOT__reg_len)) 
          & (0U != vlSelf->in_adj_cnt[3U])) & VL_LTES_III(8, 0U, 
                                                          vlSelf->in_po_of
                                                          [3U]))) {
        if ((vlSelf->in_po_of[3U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_new))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   + DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
        if ((vlSelf->in_po_of[3U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_old))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   - DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
    }
    DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext 
        = vlSelf->in_adj_cnt[4U];
    if ((((((IData)(4U) + (IData)(vlSelf->DeltaKernelCosim__DOT__i_ptr)) 
           < (IData)(vlSelf->DeltaKernelCosim__DOT__reg_len)) 
          & (0U != vlSelf->in_adj_cnt[4U])) & VL_LTES_III(8, 0U, 
                                                          vlSelf->in_po_of
                                                          [4U]))) {
        if ((vlSelf->in_po_of[4U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_new))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   + DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
        if ((vlSelf->in_po_of[4U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_old))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   - DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
    }
    DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext 
        = vlSelf->in_adj_cnt[5U];
    if ((((((IData)(5U) + (IData)(vlSelf->DeltaKernelCosim__DOT__i_ptr)) 
           < (IData)(vlSelf->DeltaKernelCosim__DOT__reg_len)) 
          & (0U != vlSelf->in_adj_cnt[5U])) & VL_LTES_III(8, 0U, 
                                                          vlSelf->in_po_of
                                                          [5U]))) {
        if ((vlSelf->in_po_of[5U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_new))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   + DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
        if ((vlSelf->in_po_of[5U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_old))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   - DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
    }
    DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext 
        = vlSelf->in_adj_cnt[6U];
    if ((((((IData)(6U) + (IData)(vlSelf->DeltaKernelCosim__DOT__i_ptr)) 
           < (IData)(vlSelf->DeltaKernelCosim__DOT__reg_len)) 
          & (0U != vlSelf->in_adj_cnt[6U])) & VL_LTES_III(8, 0U, 
                                                          vlSelf->in_po_of
                                                          [6U]))) {
        if ((vlSelf->in_po_of[6U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_new))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   + DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
        if ((vlSelf->in_po_of[6U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_old))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   - DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
    }
    DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext 
        = vlSelf->in_adj_cnt[7U];
    if ((((((IData)(7U) + (IData)(vlSelf->DeltaKernelCosim__DOT__i_ptr)) 
           < (IData)(vlSelf->DeltaKernelCosim__DOT__reg_len)) 
          & (0U != vlSelf->in_adj_cnt[7U])) & VL_LTES_III(8, 0U, 
                                                          vlSelf->in_po_of
                                                          [7U]))) {
        if ((vlSelf->in_po_of[7U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_new))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   + DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
        if ((vlSelf->in_po_of[7U] == (IData)(vlSelf->DeltaKernelCosim__DOT__reg_old))) {
            vlSelf->DeltaKernelCosim__DOT__lane_sum 
                = (vlSelf->DeltaKernelCosim__DOT__lane_sum 
                   - DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext);
        }
    }
}

VL_ATTR_COLD void VDeltaKernelCosim___024root___eval_stl(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_stl\n"); );
    // Body
    if ((1ULL & vlSelf->__VstlTriggered.word(0U))) {
        VDeltaKernelCosim___024root___stl_sequent__TOP__0(vlSelf);
    }
}

VL_ATTR_COLD void VDeltaKernelCosim___024root___eval_triggers__stl(VDeltaKernelCosim___024root* vlSelf);

VL_ATTR_COLD bool VDeltaKernelCosim___024root___eval_phase__stl(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_phase__stl\n"); );
    // Init
    CData/*0:0*/ __VstlExecute;
    // Body
    VDeltaKernelCosim___024root___eval_triggers__stl(vlSelf);
    __VstlExecute = vlSelf->__VstlTriggered.any();
    if (__VstlExecute) {
        VDeltaKernelCosim___024root___eval_stl(vlSelf);
    }
    return (__VstlExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void VDeltaKernelCosim___024root___dump_triggers__ico(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___dump_triggers__ico\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VicoTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        VL_DBG_MSGF("         'ico' region trigger index 0 is active: Internal 'ico' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void VDeltaKernelCosim___024root___dump_triggers__act(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VactTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VactTriggered.word(0U))) {
        VL_DBG_MSGF("         'act' region trigger index 0 is active: @(posedge clk or negedge rst_n)\n");
    }
}
#endif  // VL_DEBUG

#ifdef VL_DEBUG
VL_ATTR_COLD void VDeltaKernelCosim___024root___dump_triggers__nba(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___dump_triggers__nba\n"); );
    // Body
    if ((1U & (~ (IData)(vlSelf->__VnbaTriggered.any())))) {
        VL_DBG_MSGF("         No triggers active\n");
    }
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VL_DBG_MSGF("         'nba' region trigger index 0 is active: @(posedge clk or negedge rst_n)\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void VDeltaKernelCosim___024root___ctor_var_reset(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___ctor_var_reset\n"); );
    // Body
    vlSelf->clk = VL_RAND_RESET_I(1);
    vlSelf->rst_n = VL_RAND_RESET_I(1);
    vlSelf->start = VL_RAND_RESET_I(1);
    vlSelf->busy = VL_RAND_RESET_I(1);
    vlSelf->in_old_pid = VL_RAND_RESET_I(8);
    vlSelf->in_new_pid = VL_RAND_RESET_I(8);
    vlSelf->in_adj_len = VL_RAND_RESET_I(16);
    vlSelf->in_group_valid = VL_RAND_RESET_I(1);
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->in_adj_other[__Vi0] = VL_RAND_RESET_I(12);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->in_adj_cnt[__Vi0] = VL_RAND_RESET_I(16);
    }
    for (int __Vi0 = 0; __Vi0 < 8; ++__Vi0) {
        vlSelf->in_po_of[__Vi0] = VL_RAND_RESET_I(8);
    }
    vlSelf->out_valid = VL_RAND_RESET_I(1);
    vlSelf->out_delta = VL_RAND_RESET_I(32);
    vlSelf->DeltaKernelCosim__DOT__acc = VL_RAND_RESET_I(32);
    vlSelf->DeltaKernelCosim__DOT__i_ptr = VL_RAND_RESET_I(16);
    vlSelf->DeltaKernelCosim__DOT__reg_old = VL_RAND_RESET_I(8);
    vlSelf->DeltaKernelCosim__DOT__reg_new = VL_RAND_RESET_I(8);
    vlSelf->DeltaKernelCosim__DOT__reg_len = VL_RAND_RESET_I(16);
    vlSelf->DeltaKernelCosim__DOT__scanning = VL_RAND_RESET_I(1);
    vlSelf->DeltaKernelCosim__DOT__lane_sum = VL_RAND_RESET_I(32);
    vlSelf->__Vtrigprevexpr___TOP__clk__0 = VL_RAND_RESET_I(1);
    vlSelf->__Vtrigprevexpr___TOP__rst_n__0 = VL_RAND_RESET_I(1);
}
