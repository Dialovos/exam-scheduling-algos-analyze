CXX      = g++
CXXFLAGS = -O3 -std=c++20
SRC      = cpp/main.cpp
BIN      = cpp/exam_solver

.PHONY: all clean test

all: $(BIN)

$(BIN): cpp/main.cpp cpp/models.h cpp/parser.h cpp/evaluator.h cpp/greedy.h cpp/tabu.h cpp/hho.h
	$(CXX) $(CXXFLAGS) -o $@ cpp/main.cpp
	@echo "Built: $@"

clean:
	rm -f $(BIN)

test: $(BIN)
	@echo "=== exam_comp_set4 ==="
	$(BIN) datasets/exam_comp_set4.exam --algo all --tabu-iters 2000 --hho-pop 30 --hho-iters 200 -v
	@echo ""
	@echo "=== exam_comp_set5 ==="
	$(BIN) datasets/exam_comp_set5.exam --algo all --tabu-iters 2000 --hho-pop 30 --hho-iters 200 -v
	@echo ""
	@echo "=== exam_comp_set7 ==="
	$(BIN) datasets/exam_comp_set7.exam --algo all --tabu-iters 2000 --hho-pop 30 --hho-iters 200 -v