// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VDELTAKERNELCOSIM__SYMS_H_
#define VERILATED_VDELTAKERNELCOSIM__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "VDeltaKernelCosim.h"

// INCLUDE MODULE CLASSES
#include "VDeltaKernelCosim___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES)VDeltaKernelCosim__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    VDeltaKernelCosim* const __Vm_modelp;
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    VDeltaKernelCosim___024root    TOP;

    // CONSTRUCTORS
    VDeltaKernelCosim__Syms(VerilatedContext* contextp, const char* namep, VDeltaKernelCosim* modelp);
    ~VDeltaKernelCosim__Syms();

    // METHODS
    const char* name() { return TOP.name(); }
};

#endif  // guard
