// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "VDeltaKernelCosim__pch.h"

//============================================================
// Constructors

VDeltaKernelCosim::VDeltaKernelCosim(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new VDeltaKernelCosim__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , rst_n{vlSymsp->TOP.rst_n}
    , start{vlSymsp->TOP.start}
    , busy{vlSymsp->TOP.busy}
    , in_old_pid{vlSymsp->TOP.in_old_pid}
    , in_new_pid{vlSymsp->TOP.in_new_pid}
    , in_group_valid{vlSymsp->TOP.in_group_valid}
    , out_valid{vlSymsp->TOP.out_valid}
    , in_adj_len{vlSymsp->TOP.in_adj_len}
    , out_delta{vlSymsp->TOP.out_delta}
    , in_adj_other{vlSymsp->TOP.in_adj_other}
    , in_adj_cnt{vlSymsp->TOP.in_adj_cnt}
    , in_po_of{vlSymsp->TOP.in_po_of}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

VDeltaKernelCosim::VDeltaKernelCosim(const char* _vcname__)
    : VDeltaKernelCosim(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

VDeltaKernelCosim::~VDeltaKernelCosim() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void VDeltaKernelCosim___024root___eval_debug_assertions(VDeltaKernelCosim___024root* vlSelf);
#endif  // VL_DEBUG
void VDeltaKernelCosim___024root___eval_static(VDeltaKernelCosim___024root* vlSelf);
void VDeltaKernelCosim___024root___eval_initial(VDeltaKernelCosim___024root* vlSelf);
void VDeltaKernelCosim___024root___eval_settle(VDeltaKernelCosim___024root* vlSelf);
void VDeltaKernelCosim___024root___eval(VDeltaKernelCosim___024root* vlSelf);

void VDeltaKernelCosim::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate VDeltaKernelCosim::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    VDeltaKernelCosim___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        VDeltaKernelCosim___024root___eval_static(&(vlSymsp->TOP));
        VDeltaKernelCosim___024root___eval_initial(&(vlSymsp->TOP));
        VDeltaKernelCosim___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    VDeltaKernelCosim___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool VDeltaKernelCosim::eventsPending() { return false; }

uint64_t VDeltaKernelCosim::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "%Error: No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* VDeltaKernelCosim::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void VDeltaKernelCosim___024root___eval_final(VDeltaKernelCosim___024root* vlSelf);

VL_ATTR_COLD void VDeltaKernelCosim::final() {
    VDeltaKernelCosim___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* VDeltaKernelCosim::hierName() const { return vlSymsp->name(); }
const char* VDeltaKernelCosim::modelName() const { return "VDeltaKernelCosim"; }
unsigned VDeltaKernelCosim::threads() const { return 1; }
void VDeltaKernelCosim::prepareClone() const { contextp()->prepareClone(); }
void VDeltaKernelCosim::atClone() const {
    contextp()->threadPoolpOnClone();
}

//============================================================
// Trace configuration

VL_ATTR_COLD void VDeltaKernelCosim::trace(VerilatedVcdC* tfp, int levels, int options) {
    vl_fatal(__FILE__, __LINE__, __FILE__,"'VDeltaKernelCosim::trace()' called on model that was Verilated without --trace option");
}
