CXX      = g++
CXXFLAGS = -O3 -std=c++20
LDFLAGS  = -lm
SRC      = cpp/exam_solver.cpp
BIN      = cpp/exam_solver

.PHONY: all clean test demo

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

clean:
	rm -f $(BIN)

test: $(BIN)
	@echo "Running on exam_comp_set4.exam..."
	$(BIN) datasets/exam_comp_set4.exam --algo all \
		--tabu-iters 2000 --tabu-patience 500 \
		--hho-pop 50 --hho-iters 500 -v

demo: $(BIN)
	python main.py --mode demo --size 50