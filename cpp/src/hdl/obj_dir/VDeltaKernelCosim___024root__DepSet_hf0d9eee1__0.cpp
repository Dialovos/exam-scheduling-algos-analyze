// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VDeltaKernelCosim.h for the primary calling header

#include "VDeltaKernelCosim__pch.h"
#include "VDeltaKernelCosim___024root.h"

VL_INLINE_OPT void VDeltaKernelCosim___024root___ico_sequent__TOP__0(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___ico_sequent__TOP__0\n"); );
    // Init
    IData/*31:0*/ DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext;
    DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext = 0;
    // Body
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

void VDeltaKernelCosim___024root___eval_ico(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_ico\n"); );
    // Body
    if ((1ULL & vlSelf->__VicoTriggered.word(0U))) {
        VDeltaKernelCosim___024root___ico_sequent__TOP__0(vlSelf);
    }
}

void VDeltaKernelCosim___024root___eval_triggers__ico(VDeltaKernelCosim___024root* vlSelf);

bool VDeltaKernelCosim___024root___eval_phase__ico(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_phase__ico\n"); );
    // Init
    CData/*0:0*/ __VicoExecute;
    // Body
    VDeltaKernelCosim___024root___eval_triggers__ico(vlSelf);
    __VicoExecute = vlSelf->__VicoTriggered.any();
    if (__VicoExecute) {
        VDeltaKernelCosim___024root___eval_ico(vlSelf);
    }
    return (__VicoExecute);
}

void VDeltaKernelCosim___024root___eval_act(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_act\n"); );
}

VL_INLINE_OPT void VDeltaKernelCosim___024root___nba_sequent__TOP__0(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___nba_sequent__TOP__0\n"); );
    // Init
    IData/*31:0*/ DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext;
    DeltaKernelCosim__DOT__unnamedblk1__DOT__unnamedblk2__DOT__cnt_ext = 0;
    IData/*31:0*/ __Vdly__DeltaKernelCosim__DOT__acc;
    __Vdly__DeltaKernelCosim__DOT__acc = 0;
    SData/*15:0*/ __Vdly__DeltaKernelCosim__DOT__i_ptr;
    __Vdly__DeltaKernelCosim__DOT__i_ptr = 0;
    SData/*15:0*/ __Vdly__DeltaKernelCosim__DOT__reg_len;
    __Vdly__DeltaKernelCosim__DOT__reg_len = 0;
    CData/*0:0*/ __Vdly__DeltaKernelCosim__DOT__scanning;
    __Vdly__DeltaKernelCosim__DOT__scanning = 0;
    // Body
    __Vdly__DeltaKernelCosim__DOT__acc = vlSelf->DeltaKernelCosim__DOT__acc;
    __Vdly__DeltaKernelCosim__DOT__scanning = vlSelf->DeltaKernelCosim__DOT__scanning;
    __Vdly__DeltaKernelCosim__DOT__reg_len = vlSelf->DeltaKernelCosim__DOT__reg_len;
    __Vdly__DeltaKernelCosim__DOT__i_ptr = vlSelf->DeltaKernelCosim__DOT__i_ptr;
    if (vlSelf->rst_n) {
        vlSelf->out_valid = 0U;
        if (((IData)(vlSelf->start) & (~ (IData)(vlSelf->DeltaKernelCosim__DOT__scanning)))) {
            __Vdly__DeltaKernelCosim__DOT__acc = 0U;
            __Vdly__DeltaKernelCosim__DOT__i_ptr = 0U;
            vlSelf->DeltaKernelCosim__DOT__reg_old 
                = vlSelf->in_old_pid;
            vlSelf->DeltaKernelCosim__DOT__reg_new 
                = vlSelf->in_new_pid;
            __Vdly__DeltaKernelCosim__DOT__reg_len 
                = vlSelf->in_adj_len;
            __Vdly__DeltaKernelCosim__DOT__scanning 
                = (0U != (IData)(vlSelf->in_adj_len));
            if ((0U == (IData)(vlSelf->in_adj_len))) {
                vlSelf->out_valid = 1U;
                vlSelf->out_delta = 0U;
            }
        } else if (((IData)(vlSelf->DeltaKernelCosim__DOT__scanning) 
                    & (IData)(vlSelf->in_group_valid))) {
            __Vdly__DeltaKernelCosim__DOT__acc = (vlSelf->DeltaKernelCosim__DOT__acc 
                                                  + vlSelf->DeltaKernelCosim__DOT__lane_sum);
            __Vdly__DeltaKernelCosim__DOT__i_ptr = 
                (0xffffU & ((IData)(8U) + (IData)(vlSelf->DeltaKernelCosim__DOT__i_ptr)));
            if (((0xffffU & ((IData)(8U) + (IData)(vlSelf->DeltaKernelCosim__DOT__i_ptr))) 
                 >= (IData)(vlSelf->DeltaKernelCosim__DOT__reg_len))) {
                __Vdly__DeltaKernelCosim__DOT__scanning = 0U;
                vlSelf->out_valid = 1U;
                vlSelf->out_delta = (vlSelf->DeltaKernelCosim__DOT__acc 
                                     + vlSelf->DeltaKernelCosim__DOT__lane_sum);
            }
        }
    } else {
        __Vdly__DeltaKernelCosim__DOT__acc = 0U;
        __Vdly__DeltaKernelCosim__DOT__i_ptr = 0U;
        vlSelf->DeltaKernelCosim__DOT__reg_old = 0U;
        vlSelf->DeltaKernelCosim__DOT__reg_new = 0U;
        __Vdly__DeltaKernelCosim__DOT__reg_len = 0U;
        __Vdly__DeltaKernelCosim__DOT__scanning = 0U;
        vlSelf->out_valid = 0U;
        vlSelf->out_delta = 0U;
    }
    vlSelf->DeltaKernelCosim__DOT__acc = __Vdly__DeltaKernelCosim__DOT__acc;
    vlSelf->DeltaKernelCosim__DOT__scanning = __Vdly__DeltaKernelCosim__DOT__scanning;
    vlSelf->DeltaKernelCosim__DOT__i_ptr = __Vdly__DeltaKernelCosim__DOT__i_ptr;
    vlSelf->DeltaKernelCosim__DOT__reg_len = __Vdly__DeltaKernelCosim__DOT__reg_len;
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

void VDeltaKernelCosim___024root___eval_nba(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_nba\n"); );
    // Body
    if ((1ULL & vlSelf->__VnbaTriggered.word(0U))) {
        VDeltaKernelCosim___024root___nba_sequent__TOP__0(vlSelf);
    }
}

void VDeltaKernelCosim___024root___eval_triggers__act(VDeltaKernelCosim___024root* vlSelf);

bool VDeltaKernelCosim___024root___eval_phase__act(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_phase__act\n"); );
    // Init
    VlTriggerVec<1> __VpreTriggered;
    CData/*0:0*/ __VactExecute;
    // Body
    VDeltaKernelCosim___024root___eval_triggers__act(vlSelf);
    __VactExecute = vlSelf->__VactTriggered.any();
    if (__VactExecute) {
        __VpreTriggered.andNot(vlSelf->__VactTriggered, vlSelf->__VnbaTriggered);
        vlSelf->__VnbaTriggered.thisOr(vlSelf->__VactTriggered);
        VDeltaKernelCosim___024root___eval_act(vlSelf);
    }
    return (__VactExecute);
}

bool VDeltaKernelCosim___024root___eval_phase__nba(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_phase__nba\n"); );
    // Init
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = vlSelf->__VnbaTriggered.any();
    if (__VnbaExecute) {
        VDeltaKernelCosim___024root___eval_nba(vlSelf);
        vlSelf->__VnbaTriggered.clear();
    }
    return (__VnbaExecute);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void VDeltaKernelCosim___024root___dump_triggers__ico(VDeltaKernelCosim___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void VDeltaKernelCosim___024root___dump_triggers__nba(VDeltaKernelCosim___024root* vlSelf);
#endif  // VL_DEBUG
#ifdef VL_DEBUG
VL_ATTR_COLD void VDeltaKernelCosim___024root___dump_triggers__act(VDeltaKernelCosim___024root* vlSelf);
#endif  // VL_DEBUG

void VDeltaKernelCosim___024root___eval(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval\n"); );
    // Init
    IData/*31:0*/ __VicoIterCount;
    CData/*0:0*/ __VicoContinue;
    IData/*31:0*/ __VnbaIterCount;
    CData/*0:0*/ __VnbaContinue;
    // Body
    __VicoIterCount = 0U;
    vlSelf->__VicoFirstIteration = 1U;
    __VicoContinue = 1U;
    while (__VicoContinue) {
        if (VL_UNLIKELY((0x64U < __VicoIterCount))) {
#ifdef VL_DEBUG
            VDeltaKernelCosim___024root___dump_triggers__ico(vlSelf);
#endif
            VL_FATAL_MT("/home/hoang/Developer/cli/claude/personal_proj/exam-scheduling/cpp/src/hdl/delta_kernel_cosim.sv", 15, "", "Input combinational region did not converge.");
        }
        __VicoIterCount = ((IData)(1U) + __VicoIterCount);
        __VicoContinue = 0U;
        if (VDeltaKernelCosim___024root___eval_phase__ico(vlSelf)) {
            __VicoContinue = 1U;
        }
        vlSelf->__VicoFirstIteration = 0U;
    }
    __VnbaIterCount = 0U;
    __VnbaContinue = 1U;
    while (__VnbaContinue) {
        if (VL_UNLIKELY((0x64U < __VnbaIterCount))) {
#ifdef VL_DEBUG
            VDeltaKernelCosim___024root___dump_triggers__nba(vlSelf);
#endif
            VL_FATAL_MT("/home/hoang/Developer/cli/claude/personal_proj/exam-scheduling/cpp/src/hdl/delta_kernel_cosim.sv", 15, "", "NBA region did not converge.");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        __VnbaContinue = 0U;
        vlSelf->__VactIterCount = 0U;
        vlSelf->__VactContinue = 1U;
        while (vlSelf->__VactContinue) {
            if (VL_UNLIKELY((0x64U < vlSelf->__VactIterCount))) {
#ifdef VL_DEBUG
                VDeltaKernelCosim___024root___dump_triggers__act(vlSelf);
#endif
                VL_FATAL_MT("/home/hoang/Developer/cli/claude/personal_proj/exam-scheduling/cpp/src/hdl/delta_kernel_cosim.sv", 15, "", "Active region did not converge.");
            }
            vlSelf->__VactIterCount = ((IData)(1U) 
                                       + vlSelf->__VactIterCount);
            vlSelf->__VactContinue = 0U;
            if (VDeltaKernelCosim___024root___eval_phase__act(vlSelf)) {
                vlSelf->__VactContinue = 1U;
            }
        }
        if (VDeltaKernelCosim___024root___eval_phase__nba(vlSelf)) {
            __VnbaContinue = 1U;
        }
    }
}

#ifdef VL_DEBUG
void VDeltaKernelCosim___024root___eval_debug_assertions(VDeltaKernelCosim___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    VDeltaKernelCosim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    VDeltaKernelCosim___024root___eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((vlSelf->clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((vlSelf->rst_n & 0xfeU))) {
        Verilated::overWidthError("rst_n");}
    if (VL_UNLIKELY((vlSelf->start & 0xfeU))) {
        Verilated::overWidthError("start");}
    if (VL_UNLIKELY((vlSelf->in_group_valid & 0xfeU))) {
        Verilated::overWidthError("in_group_valid");}
}
#endif  // VL_DEBUG
