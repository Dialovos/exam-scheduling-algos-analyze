"""
Generates publication-quality plots addressing the project's research questions:
  Q1: Runtime scaling across algorithms
  Q2: Solution quality vs execution time
  Q3: Parameter tuning effects
  Q4: Memory usage scaling
"""

import json
import os
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


ALGO_COLORS = {
    'Greedy': '#2196F3',
    'IP': '#F44336',
    'Tabu Search': '#4CAF50',
    'HHO': '#FF9800',
}

ALGO_MARKERS = {
    'Greedy': 'o',
    'IP': 's',
    'Tabu Search': '^',
    'HHO': 'D',
}


def _aggregate_results(results: list[dict]) -> dict:
    """
    Aggregate trial results: compute mean and std for each (algorithm, num_exams).
    Returns: {(algo, n): {'runtime_mean', 'runtime_std', 'soft_mean', 'soft_std',
                          'memory_mean', 'feasible_rate'}}
    """
    groups = defaultdict(list)
    for r in results:
        if r.get('error'):
            continue
        key = (r['algorithm'], r['num_exams'])
        groups[key].append(r)

    agg = {}
    for key, trials in groups.items():
        runtimes = [t['runtime'] for t in trials]
        softs = [t['soft_penalty'] for t in trials if t['soft_penalty'] != float('inf')]
        mems = [t.get('memory_peak_mb', 0) for t in trials]
        feasible = [t['feasible'] for t in trials]

        n = len(runtimes)
        rt_mean = sum(runtimes) / n if n else 0
        rt_std = (sum((x - rt_mean)**2 for x in runtimes) / max(1, n-1)) ** 0.5 if n > 1 else 0

        sf_mean = sum(softs) / len(softs) if softs else float('inf')
        sf_std = (sum((x - sf_mean)**2 for x in softs) / max(1, len(softs)-1)) ** 0.5 if len(softs) > 1 else 0

        mem_mean = sum(mems) / len(mems) if mems else 0

        agg[key] = {
            'runtime_mean': rt_mean,
            'runtime_std': rt_std,
            'soft_mean': sf_mean,
            'soft_std': sf_std,
            'memory_mean': mem_mean,
            'feasible_rate': sum(feasible) / len(feasible) if feasible else 0,
            'num_trials': n,
        }
    return agg


def plot_runtime_scaling(results: list[dict], output_dir: str = "results"):
    """
    Q1: How does runtime scale with input size?
    Plots runtime (y) vs num_exams (x) for each algorithm.
    """
    if not HAS_MATPLOTLIB:
        print("[Plot] matplotlib not available, skipping plots")
        return

    agg = _aggregate_results(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    algos = sorted(set(r['algorithm'] for r in results if not r.get('error')))
    for algo in algos:
        points = [(n, agg[(algo, n)]) for (a, n) in agg if a == algo]
        points.sort(key=lambda x: x[0])
        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1]['runtime_mean'] for p in points]
        errs = [p[1]['runtime_std'] for p in points]

        color = ALGO_COLORS.get(algo, '#999999')
        marker = ALGO_MARKERS.get(algo, 'o')

        ax.errorbar(xs, ys, yerr=errs, label=algo, color=color, marker=marker,
                     linewidth=2, markersize=8, capsize=4)

    ax.set_xlabel('Number of Exams', fontsize=13)
    ax.set_ylabel('Runtime (seconds)', fontsize=13)
    ax.set_title('Q1: Runtime Scaling by Algorithm', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    fig.tight_layout()
    filepath = os.path.join(output_dir, "q1_runtime_scaling.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: {filepath}")


def plot_quality_vs_time(results: list[dict], output_dir: str = "results"):
    """
    Q2: Solution quality vs execution time trade-off.
    Scatter plot: runtime (x) vs soft penalty (y) colored by algorithm.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    algos = sorted(set(r['algorithm'] for r in results if not r.get('error')))
    for algo in algos:
        pts = [(r['runtime'], r['soft_penalty'])
               for r in results if r['algorithm'] == algo
               and not r.get('error') and r['feasible']
               and r['soft_penalty'] != float('inf')]
        if not pts:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        color = ALGO_COLORS.get(algo, '#999999')
        marker = ALGO_MARKERS.get(algo, 'o')
        ax.scatter(xs, ys, label=algo, color=color, marker=marker, s=60, alpha=0.7)

    ax.set_xlabel('Runtime (seconds)', fontsize=13)
    ax.set_ylabel('Soft Constraint Penalty', fontsize=13)
    ax.set_title('Q2: Solution Quality vs Execution Time', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath = os.path.join(output_dir, "q2_quality_vs_time.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: {filepath}")


def plot_memory_scaling(results: list[dict], output_dir: str = "results"):
    """
    Q4: Memory usage scaling across algorithms.
    """
    if not HAS_MATPLOTLIB:
        return

    agg = _aggregate_results(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    algos = sorted(set(r['algorithm'] for r in results if not r.get('error')))
    for algo in algos:
        points = [(n, agg[(algo, n)]) for (a, n) in agg if a == algo]
        points.sort(key=lambda x: x[0])
        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1]['memory_mean'] for p in points]

        color = ALGO_COLORS.get(algo, '#999999')
        marker = ALGO_MARKERS.get(algo, 'o')
        ax.plot(xs, ys, label=algo, color=color, marker=marker,
                linewidth=2, markersize=8)

    ax.set_xlabel('Number of Exams', fontsize=13)
    ax.set_ylabel('Peak Memory (MB)', fontsize=13)
    ax.set_title('Q4: Memory Usage Scaling', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    filepath = os.path.join(output_dir, "q4_memory_scaling.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: {filepath}")


def plot_parameter_study(results: list[dict], output_dir: str = "results"):
    """
    Q3: Parameter tuning effects on Tabu Search and HHO.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Tabu: iterations vs quality ---
    tabu_pts = [(r['param_value'], r['runtime'], r['soft_penalty'])
                for r in results if r.get('param_name') == 'tabu_iterations'
                and not r.get('error')]
    tabu_pts.sort()

    if tabu_pts:
        ax = axes[0]
        xs = [p[0] for p in tabu_pts]
        ys_quality = [p[2] for p in tabu_pts]
        ys_time = [p[1] for p in tabu_pts]

        color1 = '#4CAF50'
        color2 = '#E91E63'

        ax.plot(xs, ys_quality, color=color1, marker='^', linewidth=2, markersize=8,
                label='Soft Penalty')
        ax.set_xlabel('Tabu Iterations', fontsize=12)
        ax.set_ylabel('Soft Penalty', fontsize=12, color=color1)
        ax.tick_params(axis='y', labelcolor=color1)

        ax2 = ax.twinx()
        ax2.plot(xs, ys_time, color=color2, marker='s', linewidth=2, markersize=8,
                 linestyle='--', label='Runtime')
        ax2.set_ylabel('Runtime (s)', fontsize=12, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        ax.set_title('Q3: Tabu Search Parameter Tuning', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

    # --- HHO: population vs quality ---
    hho_pts = [(r['param_value'], r['runtime'], r['soft_penalty'])
               for r in results if r.get('param_name') == 'hho_population'
               and not r.get('error')]
    hho_pts.sort()

    if hho_pts:
        ax = axes[1]
        xs = [p[0] for p in hho_pts]
        ys_quality = [p[2] for p in hho_pts]
        ys_time = [p[1] for p in hho_pts]

        color1 = '#FF9800'
        color2 = '#E91E63'

        ax.plot(xs, ys_quality, color=color1, marker='D', linewidth=2, markersize=8,
                label='Soft Penalty')
        ax.set_xlabel('HHO Population Size', fontsize=12)
        ax.set_ylabel('Soft Penalty', fontsize=12, color=color1)
        ax.tick_params(axis='y', labelcolor=color1)

        ax2 = ax.twinx()
        ax2.plot(xs, ys_time, color=color2, marker='s', linewidth=2, markersize=8,
                 linestyle='--', label='Runtime')
        ax2.set_ylabel('Runtime (s)', fontsize=12, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        ax.set_title('Q3: HHO Parameter Tuning', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

    fig.tight_layout()
    filepath = os.path.join(output_dir, "q3_parameter_study.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: {filepath}")


def plot_feasibility_comparison(results: list[dict], output_dir: str = "results"):
    """Bar chart comparing feasibility rates across algorithms and sizes."""
    if not HAS_MATPLOTLIB:
        return

    agg = _aggregate_results(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    algos = sorted(set(a for (a, _) in agg))
    sizes = sorted(set(n for (_, n) in agg))

    bar_width = 0.18
    x_positions = range(len(sizes))

    for i, algo in enumerate(algos):
        rates = []
        for n in sizes:
            key = (algo, n)
            if key in agg:
                rates.append(agg[key]['feasible_rate'] * 100)
            else:
                rates.append(0)

        color = ALGO_COLORS.get(algo, '#999999')
        offset = (i - len(algos)/2 + 0.5) * bar_width
        bars = ax.bar([x + offset for x in x_positions], rates,
                      bar_width, label=algo, color=color, alpha=0.85)

    ax.set_xlabel('Number of Exams', fontsize=13)
    ax.set_ylabel('Feasibility Rate (%)', fontsize=13)
    ax.set_title('Feasibility Rate by Algorithm and Problem Size', fontsize=15, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sizes)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    filepath = os.path.join(output_dir, "feasibility_comparison.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: {filepath}")


def generate_all_plots(results_dir: str = "results"):
    """Load results and generate all plots."""
    bench_file = os.path.join(results_dir, "benchmark_results.json")
    param_file = os.path.join(results_dir, "parameter_study.json")

    if os.path.exists(bench_file):
        with open(bench_file) as f:
            bench_results = json.load(f)
        plot_runtime_scaling(bench_results, results_dir)
        plot_quality_vs_time(bench_results, results_dir)
        plot_memory_scaling(bench_results, results_dir)
        plot_feasibility_comparison(bench_results, results_dir)
    else:
        print(f"[Plot] No benchmark results found at {bench_file}")

    if os.path.exists(param_file):
        with open(param_file) as f:
            param_results = json.load(f)
        plot_parameter_study(param_results, results_dir)
    else:
        print(f"[Plot] No parameter study found at {param_file}")

    # Soft constraint breakdown if available
    breakdown_file = os.path.join(results_dir, "soft_breakdown.json")
    if os.path.exists(breakdown_file):
        with open(breakdown_file) as f:
            breakdown = json.load(f)
        plot_soft_constraint_breakdown(breakdown, results_dir)


def plot_soft_constraint_breakdown(breakdown: dict, output_dir: str = "results"):
    """
    Stacked bar chart showing each soft constraint's contribution per algorithm.

    Args:
        breakdown: {algo_name: {constraint_name: value, ...}, ...}
    """
    if not HAS_MATPLOTLIB:
        return

    SOFT_KEYS = ['two_in_a_row', 'two_in_a_day', 'period_spread',
                 'non_mixed_durations', 'front_load', 'period_penalty', 'room_penalty']
    SOFT_LABELS = ['2-in-a-Row', '2-in-a-Day', 'Period Spread',
                   'Mixed Duration', 'Front Load', 'Period Pen.', 'Room Pen.']
    SOFT_COLORS = ['#E53935', '#FB8C00', '#FDD835', '#43A047',
                   '#1E88E5', '#8E24AA', '#6D4C41']

    algos = list(breakdown.keys())
    n = len(algos)

    fig, ax = plt.subplots(figsize=(max(8, n * 2.2), 6))

    x = range(n)
    bottoms = [0] * n

    for i, (key, label, color) in enumerate(zip(SOFT_KEYS, SOFT_LABELS, SOFT_COLORS)):
        vals = [breakdown[a].get(key, 0) for a in algos]
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.88, width=0.6)
        # Add value labels for significant components
        for j, v in enumerate(vals):
            if v > 0:
                y_pos = bottoms[j] + v / 2
                if v > sum(breakdown[algos[j]].get(k, 0) for k in SOFT_KEYS) * 0.05:
                    ax.text(j, y_pos, str(v), ha='center', va='center',
                            fontsize=8, fontweight='bold', color='white')
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    # Total labels on top
    for j, a in enumerate(algos):
        total = sum(breakdown[a].get(k, 0) for k in SOFT_KEYS)
        ax.text(j, bottoms[j] + max(bottoms) * 0.02, str(total),
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=12)
    ax.set_ylabel('Soft Constraint Penalty', fontsize=13)
    ax.set_title('Soft Constraint Breakdown by Algorithm', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    filepath = os.path.join(output_dir, "soft_constraint_breakdown.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: {filepath}")
