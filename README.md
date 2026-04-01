# Exam Scheduling

Team: Hoang Le, Ian Cronin

## NP-hard Problem (graph coloring):

Given exams, students, time periods, and rooms: assign every exam to exactly one (period, room) pair such that:

- Hard constraints: No student sits two exams simultaneously; not exceeding room capacity; exam duration fits the period
- Soft constraints: Minimize penalties, mixed durations, and resource penalties

## Quick Start:

```bash
# Build the C++
g++ -O3 -std=c++20 -o cpp/exam_solver cpp/main.cpp

#  Install py deps
pip install -r requirements.txt

# ---- Guide ----

# All algo
python main.py --dataset datasets/exam_comp_set4.exam

# Single algo
python main.py --dataset datasets/exam_comp_set4.exam --algo tabu
python main.py --dataset datasets/exam_comp_set4.exam --algo ip --limit 100

# Higher iter
python main.py --dataset datasets/exam_comp_set4.exam --tabu-iters 5000 --hho-pop 100 --hho-iters 1000

# Run C++ directly
./cpp/exam_solver datasets/exam_comp_set4.exam --algo all --tabu-iters 2000 --hho-pop 50 --hho-iters 500 -v

# synthetic instance
python main.py --mode demo --size 200
```

## Project Structure

```
exam_scheduling/
├── main.py
├── Makefile
├── requirements.txt
│
├── cpp/
│   ├── main.cpp
│   ├── models.h
│   ├── parser.h
│   ├── evaluator.h
│   ├── greedy.h
│   ├── tabu.h
│   └── hho.h
│
├── algorithms/
│   ├── cpp_bridge.py
│   └── ip_solver.py
│
├── data/
│   ├── models.py
│   ├── evaluator.py
│   ├── fast_eval.py
│   ├── parser.py
│   └── generator.py
│
├── utils/
│   ├── plotting.py
│   └── benchmark.py
│
├── datasets/
├── results/
└── .vscode/
```

## ITC 2007 Dataset Support:

Official datasets from: https://www.eeecs.qub.ac.uk/itc2007/examtrack/
The synthetic generator also outputs in this format, so you can mix real and synthetic instances.

## References:

1. [ITC 2007 Exam Track — QUB](https://www.eeecs.qub.ac.uk/itc2007/examtrack/)
2. [Addressing Examination Timetabling — MDPI](https://www.mdpi.com/2079-3197/8/2/46)
3. [Tabu Search — Wikipedia](https://en.wikipedia.org/wiki/Tabu_search)
4. [Integer Programming — Wikipedia](https://en.wikipedia.org/wiki/Integer_programming)
5. [Greedy Coloring — Wikipedia](https://en.wikipedia.org/wiki/Greedy_coloring)
6. [Harris Hawks Optimization — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167739X18313530)
