`# Exam Scheduling

Team: Hoang Le, Ian Cronin

NP-hard Problem (graph coloring):

Given exams, students, time periods, and rooms: assign every exam to exactly one (period, room) pair such that:

- Hard constraints: No student sits two exams simultaneously; not exceeding room capacity; exam duration fits the period
- Soft constraints: Minimize penalties, mixed durations, and resource penalties

Quick Start:

```bash
pip install -r requirements.txt

# quick test
python main.py --dataset datasets/synthetic_50.exam
```

Project Structure

```
exam_scheduling/
├── main.py
├── requirements.txt
├── README.md
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── data/
│   ├── models.py
│   ├── evaluator.py
│   ├── parser.py
│   └── generator.py
├── algorithms/
│   ├── greedy.py
│   ├── ip_solver.py
│   ├── tabu_search.py
│   └── hho.py
├── utils/
│   ├── benchmark.py
│   └── plotting.py
├── datasets/
└── results/
```

ITC 2007 Dataset Support:

Official datasets from: https://www.eeecs.qub.ac.uk/itc2007/examtrack/
The synthetic generator also outputs in this format, so you can mix real and synthetic instances.

References:

1. [ITC 2007 Exam Track — QUB](https://www.eeecs.qub.ac.uk/itc2007/examtrack/)
2. [Addressing Examination Timetabling — MDPI](https://www.mdpi.com/2079-3197/8/2/46)
3. [Tabu Search — Wikipedia](https://en.wikipedia.org/wiki/Tabu_search)
4. [Integer Programming — Wikipedia](https://en.wikipedia.org/wiki/Integer_programming)
5. [Greedy Coloring — Wikipedia](https://en.wikipedia.org/wiki/Greedy_coloring)
6. [Harris Hawks Optimization — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167739X18313530)
