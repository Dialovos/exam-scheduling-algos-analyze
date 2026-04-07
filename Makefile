CXX      = g++
CXXFLAGS = -O3 -std=c++20
SRC_DIR  = cpp/src
BUILD_DIR = cpp/build
SRC      = $(SRC_DIR)/main.cpp
BIN      = $(BUILD_DIR)/exam_solver
HEADERS  = $(SRC_DIR)/models.h $(SRC_DIR)/parser.h $(SRC_DIR)/evaluator.h \
           $(SRC_DIR)/greedy.h $(SRC_DIR)/tabu.h $(SRC_DIR)/hho.h \
           $(SRC_DIR)/kempe.h $(SRC_DIR)/sa.h $(SRC_DIR)/alns.h \
           $(SRC_DIR)/gd.h $(SRC_DIR)/abc.h $(SRC_DIR)/ga.h \
           $(SRC_DIR)/lahc.h $(SRC_DIR)/natural_selection.h

.PHONY: all clean test

all: $(BIN)

$(BIN): $(SRC) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)
	@echo "Built: $@"

clean:
	rm -f $(BIN)

test: $(BIN)
	@echo "=== exam_comp_set4 ==="
	$(BIN) instances/exam_comp_set4.exam --algo all --tabu-iters 2000 --hho-pop 30 --hho-iters 200 -v
	@echo ""
	@echo "=== exam_comp_set5 ==="
	$(BIN) instances/exam_comp_set5.exam --algo all --tabu-iters 2000 --hho-pop 30 --hho-iters 200 -v
	@echo ""
	@echo "=== exam_comp_set7 ==="
	$(BIN) instances/exam_comp_set7.exam --algo all --tabu-iters 2000 --hho-pop 30 --hho-iters 200 -v
