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
           $(SRC_DIR)/hho.h

.PHONY: all clean test reproduce

all: $(BIN)

$(BIN): $(SRC) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)
	@echo "Built: $@"

clean:
	rm -f $(BIN)

test: $(BIN)
	@echo "=== exam_comp_set4 ==="
	$(BIN) instances/exam_comp_set4.exam --algo all --tabu-iters 2000 -v
	@echo ""
	@echo "=== exam_comp_set5 ==="
	$(BIN) instances/exam_comp_set5.exam --algo all --tabu-iters 2000 -v
	@echo ""
	@echo "=== exam_comp_set7 ==="
	$(BIN) instances/exam_comp_set7.exam --algo all --tabu-iters 2000 -v

# Smoke-run the solver and regenerate every paper figure from the latest
# batch. REPRO_BATCH picks which run_log.csv feeds the plots — override
# it on the CLI if you want to target a different cached batch:
#   make reproduce REPRO_BATCH=results/batch_017_tunning8
REPRO_BATCH ?= results/batch_017_tunning8
reproduce: $(BIN)
	@echo "=== Smoke: Tabu on exam_comp_set1 ==="
	@$(BIN) instances/exam_comp_set1.exam --algo tabu --seed 42 --tabu-iters 200 2>/dev/null | tail -n +1 | head -25
	@echo ""
	@echo "=== Regenerating figures from $(REPRO_BATCH) into graphs/ ==="
	python3 -m tooling.regen_figures --from $(REPRO_BATCH) --out graphs/
	@echo "=== Done. Figures written to graphs/ ==="
