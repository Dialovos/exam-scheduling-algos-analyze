# Verilator build — runs after `sudo apt install verilator`.
# Produces obj_dir/VDeltaKernel + a C++ testbench linked against the
# project's parser/evaluator headers.
#
# Usage:
#   sudo apt install -y verilator
#   make -f cpp/src/hdl/sim.mk
#   make -f cpp/src/hdl/sim.mk run

HDL_DIR  = cpp/src/hdl
TB_SRC   = $(HDL_DIR)/delta_kernel_tb.cpp
SV_SRC   = $(HDL_DIR)/delta_kernel_cosim.sv
TOP      = DeltaKernelCosim
OBJ_DIR  = $(HDL_DIR)/obj_dir

VERILATOR = verilator
PROJ_ROOT := $(abspath .)
SRC_INC   := $(PROJ_ROOT)/cpp/src
VFLAGS    = --cc --build -Wall -Wno-UNUSED -Wno-MODDUP -Wno-WIDTHEXPAND \
            -Wno-WIDTHTRUNC -Wno-UNDRIVEN -Wno-DECLFILENAME \
            -CFLAGS "-std=c++20 -O2 -I$(SRC_INC)" \
            -LDFLAGS "-pthread" \
            --top-module $(TOP)

.PHONY: all run clean check-verilator

all: check-verilator $(OBJ_DIR)/V$(TOP)

check-verilator:
	@command -v $(VERILATOR) >/dev/null 2>&1 || { \
	  echo "ERROR: verilator not found. Install with: sudo apt install verilator"; \
	  exit 1; }

$(OBJ_DIR)/V$(TOP): $(SV_SRC) $(TB_SRC)
	$(VERILATOR) $(VFLAGS) $(abspath $(SV_SRC)) --exe $(abspath $(TB_SRC)) \
	    -o V$(TOP) --Mdir $(OBJ_DIR)

run: $(OBJ_DIR)/V$(TOP)
	$(OBJ_DIR)/V$(TOP) instances/exam_comp_set4.exam 1000

clean:
	rm -rf $(OBJ_DIR)
