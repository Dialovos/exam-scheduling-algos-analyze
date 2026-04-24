CXX      = g++
CXXFLAGS = -O3 -std=c++20 -march=native -flto
SRC_DIR  = cpp/src
BUILD_DIR = cpp/build
SRC      = $(SRC_DIR)/main.cpp
BIN      = $(BUILD_DIR)/exam_solver
HEADERS  = $(SRC_DIR)/models.h $(SRC_DIR)/parser.h $(SRC_DIR)/evaluator.h \
           $(SRC_DIR)/greedy.h $(SRC_DIR)/tabu.h \
           $(SRC_DIR)/kempe.h $(SRC_DIR)/sa.h $(SRC_DIR)/alns.h \
           $(SRC_DIR)/gd.h $(SRC_DIR)/abc.h $(SRC_DIR)/ga.h \
           $(SRC_DIR)/lahc.h $(SRC_DIR)/woa.h $(SRC_DIR)/cpsat.h \
           $(SRC_DIR)/vns.h $(SRC_DIR)/neighbourhoods.h \
           $(SRC_DIR)/seeder.h $(SRC_DIR)/repair.h \
           $(SRC_DIR)/hho.h \
           $(SRC_DIR)/evaluator_simd.h $(SRC_DIR)/evaluator_cached.h \
           $(SRC_DIR)/tabu_simd.h $(SRC_DIR)/tabu_cached.h \
           $(SRC_DIR)/tabu_cached_cuda.h $(SRC_DIR)/cuda/cuda_evaluator.h \
           $(SRC_DIR)/sa_cached.h $(SRC_DIR)/gd_cached.h \
           $(SRC_DIR)/lahc_cached.h $(SRC_DIR)/alns_cached.h \
           $(SRC_DIR)/alns_thompson.h $(SRC_DIR)/alns_cuda.h $(SRC_DIR)/vns_cached.h \
           $(SRC_DIR)/ga_cuda.h $(SRC_DIR)/abc_cuda.h \
           $(SRC_DIR)/hho_cuda.h $(SRC_DIR)/woa_cuda.h \
           $(SRC_DIR)/sa_parallel_cuda.h \
           $(SRC_DIR)/xoshiro.h $(SRC_DIR)/ejection.h

.PHONY: all clean test reproduce bench bench-omp fast-pgo pgo-clean batch19 batch19-colab

BENCH_SRC = $(SRC_DIR)/bench_eval.cpp
BENCH_BIN = $(BUILD_DIR)/bench_eval
BENCH_HDR = $(HEADERS) $(SRC_DIR)/evaluator_simd.h $(SRC_DIR)/tabu_simd.h \
            $(SRC_DIR)/portfolio.h $(SRC_DIR)/polish.h $(SRC_DIR)/fpga_sim.h \
            $(SRC_DIR)/evaluator_cached.h $(SRC_DIR)/tabu_cached.h \
            $(SRC_DIR)/sa_cached.h $(SRC_DIR)/gd_cached.h $(SRC_DIR)/alns_cached.h \
            $(SRC_DIR)/lahc_cached.h $(SRC_DIR)/vns_cached.h $(SRC_DIR)/alns_thompson.h \
            $(SRC_DIR)/xoshiro.h $(SRC_DIR)/ejection.h \
            $(SRC_DIR)/recording_evaluator.h

all: $(BIN)

# Optional CUDA linkage — set HAVE_CUDA=1 when invoking make AND build
# libdelta_cuda.so first via `make cuda-build`. On CPU-only machines omit
# HAVE_CUDA; tabu_cuda will run the CPU fallback path (correct, no speedup).
CUDA_FLAGS =
CUDA_LINK =
# CUDA_LIBDIR: directory containing libcudart.so. Defaults try standard
# locations; user can override: `make all HAVE_CUDA=1 CUDA_LIBDIR=/path`.
CUDA_LIBDIR ?= $(shell test -f /usr/local/cuda/lib64/libcudart.so && echo /usr/local/cuda/lib64 || \
                         (test -f /usr/lib/x86_64-linux-gnu/libcudart.so && echo /usr/lib/x86_64-linux-gnu || \
                          echo /usr/lib/cuda/lib64))
ifeq ($(HAVE_CUDA),1)
  CUDA_FLAGS += -DHAVE_CUDA
  CUDA_LINK  += -L$(BUILD_DIR) -ldelta_cuda -Wl,-rpath,'$$ORIGIN/../build' \
                -L$(CUDA_LIBDIR) -lcudart
endif

$(BIN): $(SRC) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -mavx2 -fopenmp $(CUDA_FLAGS) -o $@ $(SRC) $(CUDA_LINK)
	@echo "Built: $@"

clean:
	rm -f $(BIN) $(BENCH_BIN)

# ── Alt-route SIMD/asm benchmark ──
# Compares FastEvaluator::move_delta against:
#   - adj-scalar  (algorithmic rewrite using adj[] — same complexity class)
#   - AVX2 intrinsics  (8-wide gather+compare)
#   - inline asm kernel  (hand-rolled AT&T AVX2 conflict-count)
# Does NOT modify any existing header. Build with: make bench
$(BENCH_BIN): $(BENCH_SRC) $(BENCH_HDR)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -mavx2 $(CUDA_FLAGS) -o $@ $(BENCH_SRC) $(CUDA_LINK)
	@echo "Built: $@"

BENCH_INSTANCE ?= instances/exam_comp_set4.exam
BENCH_ITERS    ?= 200000
bench: $(BENCH_BIN)
	$(BENCH_BIN) $(BENCH_INSTANCE) $(BENCH_ITERS)

# OpenMP-enabled bench — adds parallel portfolio timing section
BENCH_OMP_BIN = $(BUILD_DIR)/bench_eval_omp
$(BENCH_OMP_BIN): $(BENCH_SRC) $(BENCH_HDR)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -mavx2 -fopenmp -o $@ $(BENCH_SRC)
	@echo "Built: $@"
bench-omp: $(BENCH_OMP_BIN)
	$(BENCH_OMP_BIN) $(BENCH_INSTANCE) $(BENCH_ITERS)

# ── Profile-Guided Optimization build ──
# Two-pass: instrument → run training workload → rebuild with profile data.
# Target exam_comp_set4 as training because it has the richest adj pattern.
# Typical extra gain: +10-20% on top of -O3 -march=native -flto.
PGO_DIR = $(BUILD_DIR)/pgo-profile
FAST_BIN = $(BUILD_DIR)/exam_solver_fast
fast-pgo:
	@mkdir -p $(PGO_DIR) $(BUILD_DIR)
	@echo "=== PGO pass 1: instrumented build ==="
	$(CXX) $(CXXFLAGS) -mavx2 -fopenmp -fprofile-generate=$(PGO_DIR) \
	    -o $(FAST_BIN) $(SRC)
	@echo "=== PGO pass 2: collect profile (set4, tabu + sa) ==="
	$(FAST_BIN) instances/exam_comp_set4.exam --algo tabu --tabu-iters 500 >/dev/null || true
	$(FAST_BIN) instances/exam_comp_set4.exam --algo sa   --sa-iters  10000 >/dev/null || true
	@echo "=== PGO pass 3: optimized rebuild ==="
	$(CXX) $(CXXFLAGS) -mavx2 -fopenmp -fprofile-use=$(PGO_DIR) \
	    -fprofile-correction -o $(FAST_BIN) $(SRC)
	@echo "Built PGO binary: $(FAST_BIN)"
pgo-clean:
	rm -rf $(PGO_DIR) $(FAST_BIN)

# ── CUDA move-scoring kernel (Phase 3b — optional, requires nvcc + GPU) ──
CUDA_SRC = $(SRC_DIR)/cuda/delta_kernel.cu
CUDA_LIB = $(BUILD_DIR)/libdelta_cuda.so
cuda-build:
	@command -v nvcc >/dev/null 2>&1 || { \
	  echo "error: nvcc not found. Install CUDA toolkit to use --algo cuda*"; exit 1; }
	@mkdir -p $(BUILD_DIR)
	nvcc -O3 -std=c++17 --compiler-options -fPIC -shared \
	    -o $(CUDA_LIB) $(CUDA_SRC)
	@echo "Built CUDA kernel: $(CUDA_LIB)"
cuda-clean:
	rm -f $(CUDA_LIB)

# ── Batch 19: validate post-Phase-2 algos on all ITC 2007 sets ──
# Runs the cached/Thompson/SIMD variants across sets 1-8 with 3 seeds each.
# Same script works locally (this target) and on Colab (scripts/run_batch19.sh).
# Output: results/batch_019_validation/<algo>_<instance>_seed<S>/
BATCH19_DIR ?= results/batch_019_validation
BATCH19_SEEDS ?= 42 43 44
BATCH19_ALGOS ?= tabu_cached sa_cached gd_cached lahc_cached alns_thompson
BATCH19_SETS  ?= exam_comp_set1 exam_comp_set2 exam_comp_set3 exam_comp_set4 \
                 exam_comp_set5 exam_comp_set6 exam_comp_set7 exam_comp_set8
batch19: $(BIN)
	@bash scripts/run_batch19.sh "$(BATCH19_DIR)" "$(BATCH19_SEEDS)" \
	      "$(BATCH19_ALGOS)" "$(BATCH19_SETS)"

# Colab variant: same script, but runs via Python subprocess for progress display
# and uploads zip to Drive. See notebooks/batch19_colab.ipynb.
batch19-colab:
	@echo "Run notebooks/batch19_colab.ipynb on Colab. Locally use 'make batch19'."

test: $(BIN)
	@echo "=== exam_comp_set4 ==="
	$(BIN) instances/exam_comp_set4.exam --algo all --tabu-iters 2000 -v
	@echo ""
	@echo "=== exam_comp_set5 ==="
	$(BIN) instances/exam_comp_set5.exam --algo all --tabu-iters 2000 -v
	@echo ""
	@echo "=== exam_comp_set7 ==="
	$(BIN) instances/exam_comp_set7.exam --algo all --tabu-iters 2000 -v

# Smoke-run the solver and regenerate every paper figure from the cached
# batch. REPRO_BATCH points at the batch directory whose
# `make_paper_figures.py` is invoked — override on the CLI to target a
# different cached batch:
#   make reproduce REPRO_BATCH=results/batch_018_colab
REPRO_BATCH ?= results/batch_018_colab
reproduce: $(BIN)
	@echo "=== Smoke: Tabu on exam_comp_set1 ==="
	@$(BIN) instances/exam_comp_set1.exam --algo tabu --seed 42 --tabu-iters 200 2>/dev/null | tail -n +1 | head -25
	@echo ""
	@echo "=== Regenerating paper figures from $(REPRO_BATCH) into graphs/ ==="
	python3 $(REPRO_BATCH)/make_paper_figures.py
	@echo "=== Done. Figures written to graphs/ ==="
