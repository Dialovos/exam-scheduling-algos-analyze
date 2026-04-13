"""
Plotting for exam scheduling experiments.

Chart types:
  1. Algorithm comparison (horizontal bars)
  2. Soft constraint breakdown (stacked bar)
  3. Runtime vs quality (scatter)
  4. Multi-dataset heatmap
  5. Feasibility rates (horizontal bars)
  6. Scaling analysis (line)
  7. Parameter sensitivity (dual-axis line)
  8. Summary dashboard (2x2 grid)
  9. Box plot comparison
 10. Radar / performance profile
 11. Ranking table
 12. Convergence overlay (line)
 13. Line comparison across datasets
 14. Soft component lines
"""
import os
from collections import defaultdict

__all__ = [
    'ALGO_COLORS', 'ALGO_MARKERS', 'ALGO_SHORT', 'ALGO_ORDER',
    'SOFT_KEYS', 'SOFT_LABELS', 'SOFT_COLORS',
    'plot_algorithm_comparison', 'plot_soft_breakdown',
    'plot_runtime_vs_quality', 'plot_multi_dataset_heatmap',
    'plot_feasibility_rates', 'plot_scaling',
    'plot_parameter_sensitivity', 'plot_summary_dashboard',
    'plot_box_comparison', 'plot_radar', 'plot_rank_table',
    'plot_convergence', 'plot_line_across_datasets', 'plot_soft_lines',
    'plot_continuous_scan', 'plot_experiment_summary',
    'plot_algo_bars', 'plot_algo_boxes', 'plot_algo_radar', 'plot_algo_scatter',
    'plot_algo_heatmap',
    'generate_all_plots', 'plot_soft_constraint_breakdown',
]

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Colorblind-friendly palette for all algorithms ───────────────────────────
ALGO_COLORS = {
    'Greedy':                  '#4E79A7',
    'Feasibility':             '#7B7B7B',
    'Tabu Search':             '#F28E2B',
    'Kempe Chain':             '#76B7B2',
    'Simulated Annealing':     '#59A14F',
    'Multi-Neighbourhood SA':  '#59A14F',
    'ALNS':                    '#EDC948',
    'Great Deluge':            '#B07AA1',
    'ABC':                     '#FF9DA7',
    'Genetic Algorithm':       '#9C755F',
    'LAHC':                    '#BAB0AC',
    'WOA':                     '#E15759',
    'HHO':                     '#E15759',
    'CP-SAT':                  '#D37295',
    'CP-SAT B&B':              '#D37295',
    'GVNS':                    '#4B0082',
}
ALGO_MARKERS = {
    'Greedy': 'o', 'Feasibility': 'o',
    'Tabu Search': '^', 'Kempe Chain': 'v',
    'Simulated Annealing': 's', 'Multi-Neighbourhood SA': 's',
    'ALNS': 'P', 'Great Deluge': 'X',
    'ABC': 'h', 'Genetic Algorithm': 'p', 'LAHC': 'H',
    'WOA': 'D', 'HHO': 'D',
    'CP-SAT': 'd', 'CP-SAT B&B': 'd', 'GVNS': '*',
}
ALGO_SHORT = {
    'Greedy': 'Greedy', 'Feasibility': 'Feas',
    'Tabu Search': 'Tabu', 'Kempe Chain': 'Kempe',
    'Simulated Annealing': 'SA', 'Multi-Neighbourhood SA': 'SA',
    'ALNS': 'ALNS', 'Great Deluge': 'GD',
    'ABC': 'ABC', 'Genetic Algorithm': 'GA', 'LAHC': 'LAHC',
    'WOA': 'WOA', 'HHO': 'HHO',
    'CP-SAT': 'CP-SAT', 'CP-SAT B&B': 'CP-SAT', 'GVNS': 'GVNS',
}
ALGO_ORDER = ['Greedy', 'Feasibility', 'Tabu Search', 'Kempe Chain',
              'Simulated Annealing', 'Multi-Neighbourhood SA',
              'ALNS', 'Great Deluge',
              'ABC', 'Genetic Algorithm', 'LAHC',
              'WOA', 'HHO', 'CP-SAT', 'CP-SAT B&B', 'GVNS']

SOFT_KEYS   = ['two_in_a_row', 'two_in_a_day', 'period_spread',
               'non_mixed_durations', 'front_load', 'period_penalty', 'room_penalty']
SOFT_LABELS = ['2-in-a-Row', '2-in-a-Day', 'Period Spread',
               'Mixed Dur.', 'Front Load', 'Period Pen.', 'Room Pen.']
SOFT_COLORS = ['#E53935', '#FB8C00', '#FDD835', '#43A047',
               '#1E88E5', '#8E24AA', '#6D4C41']

METRIC_LABELS = {
    'soft_penalty': 'Soft Penalty', 'runtime': 'Runtime (s)',
    'hard_violations': 'Hard Violations', 'memory_peak_mb': 'Peak Memory (MB)',
}


def _c(algo):
    return ALGO_COLORS.get(algo, '#888888')

def _m(algo):
    return ALGO_MARKERS.get(algo, 'o')

def _short(algo):
    return ALGO_SHORT.get(algo, algo)

def _order(algos):
    """Sort algorithms in canonical display order."""
    idx = {a: i for i, a in enumerate(ALGO_ORDER)}
    return sorted(algos, key=lambda a: idx.get(a, 99))

def _style():
    plt.rcParams.update({
        'figure.dpi': 130, 'font.size': 11,
        'font.family': 'sans-serif',
        'axes.titlesize': 13, 'axes.labelsize': 11,
        'axes.grid': True, 'grid.alpha': 0.22, 'grid.linestyle': '--',
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.fontsize': 9, 'legend.framealpha': 0.85,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
    })

def _kfmt(ax, axis='y'):
    """Smart thousands formatter — only fires for values >= 1000."""
    def _fmt(x, _):
        if abs(x) >= 1e6:  return f'{x/1e6:,.1f}M'
        if abs(x) >= 1e3:  return f'{x/1e3:,.1f}k'
        if abs(x) >= 10:   return f'{x:,.0f}'
        return f'{x:.2f}'
    fmt = ticker.FuncFormatter(_fmt)
    if axis in ('y', 'both'): ax.yaxis.set_major_formatter(fmt)
    if axis in ('x', 'both'): ax.xaxis.set_major_formatter(fmt)

def _apply_xlabels(ax, names, *, short=True):
    """Set x-tick labels with auto-rotation to avoid overlap."""
    labels = [_short(n) for n in names] if short else list(names)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    if len(labels) >= 5:
        ax.tick_params(axis='x', rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')

def _save(fig, path):
    if path:
        fig.savefig(path, dpi=150, bbox_inches='tight')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Algorithm Comparison — horizontal bars (no label clipping)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_algorithm_comparison(df, dataset=None, metric='soft_penalty',
                              title=None, save_path=None, ax=None):
    """Horizontal bar chart — values on the right, names on the left.
    Eliminates label overlap entirely."""
    if not HAS_MPL: return
    _style()
    data = df[df['dataset'] == dataset] if dataset else df.copy()
    grouped = (data.groupby('algorithm')[metric]
               .agg(['mean', 'std', 'min', 'count'])
               .reindex(_order(data['algorithm'].unique()))
               .dropna(subset=['mean']))
    if grouped.empty: return

    own_fig = ax is None
    n = len(grouped)
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, max(3, n * 0.65)))
    else:
        fig = ax.figure

    y = np.arange(n)
    colors = [_c(a) for a in grouped.index]
    bars = ax.barh(y, grouped['mean'], xerr=grouped['std'],
                   color=colors, alpha=0.88, height=0.6,
                   capsize=3, edgecolor='white', linewidth=0.5,
                   error_kw={'linewidth': 1})

    # Value labels — always to the right of the bar, never clipped
    x_max = (grouped['mean'] + grouped['std']).max()
    for i, (algo, row) in enumerate(grouped.iterrows()):
        lbl = f"{row['mean']:,.0f}" if row['mean'] >= 10 else f"{row['mean']:.2f}"
        if row['count'] > 1:
            lbl += f"  (n={int(row['count'])})"
        ax.text(row['mean'] + row['std'] + x_max * 0.02, i,
                lbl, va='center', ha='left', fontsize=9, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels([_short(a) for a in grouped.index], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, x_max * 1.30)  # 30 % headroom for labels
    _kfmt(ax, 'x')
    mlabel = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel(mlabel)
    ds = f" — {dataset}" if dataset else ""
    ax.set_title(title or f"{mlabel}{ds}", fontweight='bold')

    if own_fig:
        fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Soft Constraint Breakdown (stacked vertical bars, clean labels)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_soft_breakdown(df_or_dict, dataset=None, title=None, save_path=None):
    if not HAS_MPL: return
    _style()
    import pandas as pd
    if isinstance(df_or_dict, pd.DataFrame):
        data = df_or_dict[df_or_dict['dataset'] == dataset] if dataset else df_or_dict
        breakdown = {a: {k: g[k].mean() for k in SOFT_KEYS}
                     for a, g in data.groupby('algorithm')}
    else:
        breakdown = df_or_dict

    algos = _order(breakdown.keys())
    n = len(algos)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.3), 6))
    x = np.arange(n)
    bottoms = np.zeros(n)
    bw = min(0.7, 5.0 / max(n, 1))

    for key, label, color in zip(SOFT_KEYS, SOFT_LABELS, SOFT_COLORS):
        vals = np.array([breakdown[a].get(key, 0) for a in algos])
        ax.bar(x, vals, bottom=bottoms, label=label, color=color,
               alpha=0.90, width=bw, edgecolor='white', linewidth=0.3)
        bottoms += vals

    # Only totals on top — no in-bar segment text to avoid clutter
    top = bottoms.max()
    for j in range(n):
        ax.text(j, bottoms[j] + top * 0.015, f"{bottoms[j]:,.0f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    _apply_xlabels(ax, algos)
    _kfmt(ax)
    ax.set_ylim(0, top * 1.12)
    ax.set_ylabel('Soft Penalty')
    ax.set_title(title or f"Soft Constraint Breakdown"
                 f"{' — ' + dataset if dataset else ''}", fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Runtime vs Quality Scatter
# ═══════════════════════════════════════════════════════════════════════════════
def plot_runtime_vs_quality(df, dataset=None, title=None, save_path=None):
    if not HAS_MPL: return
    _style()
    data = df[df['feasible'] == True].copy()
    if dataset: data = data[data['dataset'] == dataset]
    if data.empty: return

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in _order(data['algorithm'].unique()):
        adf = data[data['algorithm'] == algo]
        ax.scatter(adf['runtime'], adf['soft_penalty'], label=_short(algo),
                   color=_c(algo), marker=_m(algo), s=100, alpha=0.78,
                   edgecolors='white', linewidth=0.7, zorder=3)

    ax.set_xlabel('Runtime (s)')
    ax.set_ylabel('Soft Penalty')
    _kfmt(ax)
    ax.set_title(title or f"Quality vs Runtime"
                 f"{' — ' + dataset if dataset else ''}", fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.85)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Multi-Dataset Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def plot_multi_dataset_heatmap(df, metric='soft_penalty', title=None, save_path=None):
    if not HAS_MPL: return
    _style()
    pivot = df.pivot_table(values=metric, index='algorithm', columns='dataset', aggfunc='mean')
    if pivot.empty: return
    ordered = [a for a in ALGO_ORDER if a in pivot.index]
    leftover = [a for a in pivot.index if a not in ordered]
    pivot = pivot.reindex(ordered + leftover)

    nr, nc = pivot.shape
    fig, ax = plt.subplots(figsize=(max(6, nc * 2.2), max(3, nr * 0.7)))
    im = ax.imshow(pivot.values, cmap='YlOrRd_r', aspect='auto')
    ax.set_xticks(range(nc))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(nr))
    ax.set_yticklabels([_short(a) for a in pivot.index], fontsize=10)

    thresh = np.nanmean(pivot.values)
    for i in range(nr):
        for j in range(nc):
            v = pivot.values[i, j]
            if not np.isnan(v):
                c = 'white' if v > thresh else 'black'
                ax.text(j, i, f"{v:,.0f}", ha='center', va='center',
                        fontsize=9, color=c, fontweight='bold')

    fig.colorbar(im, ax=ax, shrink=0.8,
                 label=METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"Cross-Dataset: "
                 f"{METRIC_LABELS.get(metric, metric)}", fontweight='bold')
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Feasibility Rates (horizontal bars)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_feasibility_rates(df, title=None, save_path=None):
    if not HAS_MPL: return
    _style()
    ordered = _order(df['algorithm'].unique())
    rates = df.groupby('algorithm')['feasible'].mean().reindex(ordered).dropna() * 100
    n = len(rates)

    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.55)))
    y = np.arange(n)
    bars = ax.barh(y, rates.values, color=[_c(a) for a in rates.index],
                   alpha=0.88, height=0.55, edgecolor='white', linewidth=0.5)
    for i, v in enumerate(rates.values):
        ax.text(v + 1, i, f"{v:.0f}%", va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels([_short(a) for a in rates.index], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel('Feasibility Rate (%)')
    ax.set_title(title or 'Feasibility Rate by Algorithm', fontweight='bold')
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Scaling Analysis (line chart)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_scaling(df, x_col='num_exams', y_col='runtime', title=None, save_path=None):
    if not HAS_MPL: return
    _style()
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in _order(df['algorithm'].unique()):
        adf = df[df['algorithm'] == algo]
        g = (adf.groupby(x_col)[y_col]
             .agg(['mean', 'std']).reset_index().sort_values(x_col))
        ax.errorbar(g[x_col], g['mean'], yerr=g['std'], label=_short(algo),
                    color=_c(algo), marker=_m(algo), linewidth=2.2,
                    markersize=8, capsize=4, alpha=0.85)

    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(METRIC_LABELS.get(y_col, y_col.replace('_', ' ').title()))
    _kfmt(ax)
    ax.set_title(title or f"{METRIC_LABELS.get(y_col, y_col.replace('_', ' ').title())} "
                 f"vs {x_col.replace('_', ' ').title()}", fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Parameter Sensitivity (dual-axis line)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_parameter_sensitivity(df, param_col, metric='soft_penalty',
                               algorithm=None, title=None, save_path=None):
    if not HAS_MPL: return
    _style()
    data = df[df['algorithm'] == algorithm] if algorithm else df.copy()
    g = (data.groupby(param_col)
         .agg(soft_mean=(metric, 'mean'), soft_std=(metric, 'std'),
              rt_mean=('runtime', 'mean'), rt_std=('runtime', 'std'))
         .reset_index().sort_values(param_col))
    if g.empty: return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    c1, c2 = '#59A14F', '#E15759'
    ax1.errorbar(g[param_col], g['soft_mean'], yerr=g['soft_std'],
                 color=c1, marker='^', linewidth=2.2, capsize=4,
                 label=METRIC_LABELS.get(metric, metric))
    ax1.set_xlabel(param_col.replace('_', ' ').title())
    ax1.set_ylabel(METRIC_LABELS.get(metric, metric), color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)
    _kfmt(ax1)

    ax2 = ax1.twinx()
    ax2.errorbar(g[param_col], g['rt_mean'], yerr=g['rt_std'],
                 color=c2, marker='s', linewidth=2, capsize=4,
                 linestyle='--', label='Runtime')
    ax2.set_ylabel('Runtime (s)', color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)
    ax2.spines['right'].set_visible(True)

    al = f" ({algorithm})" if algorithm else ""
    ax1.set_title(title or f"Parameter Sensitivity{al}", fontweight='bold')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Summary Dashboard (2x2)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_summary_dashboard(df, dataset=None, save_path=None):
    if not HAS_MPL: return
    _style()
    data = df[df['dataset'] == dataset] if dataset else df
    algos = _order(data['algorithm'].unique())
    n = len(algos)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top-left: soft penalty (horizontal bars)
    plot_algorithm_comparison(df, dataset=dataset, metric='soft_penalty', ax=axes[0, 0])

    # Top-right: runtime (horizontal bars)
    plot_algorithm_comparison(df, dataset=dataset, metric='runtime', ax=axes[0, 1])

    # Bottom-left: soft breakdown (compact stacked bar)
    ax = axes[1, 0]
    breakdown = {a: {k: g[k].mean() for k in SOFT_KEYS}
                 for a, g in data.groupby('algorithm')}
    x = np.arange(n)
    bottoms = np.zeros(n)
    bw = min(0.7, 5.0 / max(n, 1))
    for key, label, color in zip(SOFT_KEYS, SOFT_LABELS, SOFT_COLORS):
        vals = np.array([breakdown.get(a, {}).get(key, 0) for a in algos])
        ax.bar(x, vals, bottom=bottoms, label=label, color=color,
               alpha=0.90, width=bw, edgecolor='white', linewidth=0.3)
        bottoms += vals
    _apply_xlabels(ax, algos)
    _kfmt(ax)
    top = bottoms.max() if len(bottoms) else 1
    ax.set_ylim(0, top * 1.08)
    ax.set_ylabel('Soft Penalty')
    ax.set_title('Soft Constraint Breakdown', fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='upper right')

    # Bottom-right: feasibility
    ax = axes[1, 1]
    rates = data.groupby('algorithm')['feasible'].mean().reindex(algos).fillna(0) * 100
    y = np.arange(n)
    ax.barh(y, rates.values, color=[_c(a) for a in algos],
            alpha=0.88, height=0.55, edgecolor='white', linewidth=0.5)
    for i, v in enumerate(rates.values):
        ax.text(v + 1, i, f"{v:.0f}%", va='center', fontsize=9, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels([_short(a) for a in algos], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel('Feasibility (%)')
    ax.set_title('Feasibility Rate', fontweight='bold')

    ds_label = f" — {dataset}" if dataset else ""
    fig.suptitle(f"Results Dashboard{ds_label}",
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Box Plot Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def plot_box_comparison(df, dataset=None, metric='soft_penalty',
                        title=None, save_path=None):
    """Horizontal box + jittered strip plot — readable with 10+ algorithms.

    Design choices (publication-quality):
      - Horizontal: labels never clip or rotate, works for any algo count
      - Jittered strip overlay: shows every data point (crucial for n<20)
      - Mean diamond: distinguishes mean from median line
      - Sorted by median: best algorithms on top for easy scanning
      - Subdued box fill + clean spines: minimal chart-junk
    """
    if not HAS_MPL: return
    _style()
    data = df[df['dataset'] == dataset] if dataset else df.copy()
    algos = _order(data['algorithm'].unique())
    box_data, valid_algos = [], []
    for a in algos:
        vals = data[data['algorithm'] == a][metric].dropna().values
        if len(vals) > 0:
            box_data.append(vals)
            valid_algos.append(a)
    if not valid_algos: return

    # Sort by median (best on top after invert)
    medians = [np.median(d) for d in box_data]
    order = np.argsort(medians)[::-1]  # reversed because yaxis inverts
    box_data = [box_data[i] for i in order]
    valid_algos = [valid_algos[i] for i in order]

    n = len(valid_algos)
    fig, ax = plt.subplots(figsize=(10, max(3.5, n * 0.6 + 1)))
    positions = list(range(n))

    # Horizontal box: thin, transparent fill, no fliers (strip shows them)
    bp = ax.boxplot(
        box_data, positions=positions, vert=False, patch_artist=True,
        widths=0.45, showfliers=False,
        medianprops=dict(color='#222222', linewidth=2),
        whiskerprops=dict(color='#666666', linewidth=1, linestyle='--'),
        capprops=dict(color='#666666', linewidth=1),
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='#E03030',
                       markeredgecolor='white', markersize=5.5,
                       markeredgewidth=0.8, zorder=5),
    )
    for patch, algo in zip(bp['boxes'], valid_algos):
        color = _c(algo)
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    # Jittered strip overlay — show every individual data point
    rng = np.random.RandomState(0)
    for i, (vals, algo) in enumerate(zip(box_data, valid_algos)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(vals, i + jitter, color=_c(algo), alpha=0.65,
                   s=22, edgecolor='white', linewidth=0.4, zorder=4)

    # Annotation: median value to the right
    x_max = max(np.max(d) for d in box_data)
    x_min = min(np.min(d) for d in box_data)
    pad = (x_max - x_min) * 0.02
    for i, vals in enumerate(box_data):
        med = np.median(vals)
        ax.text(x_max + pad, i, f'{med:,.0f}', va='center', ha='left',
                fontsize=8.5, color='#444444', fontstyle='italic')

    ax.set_yticks(positions)
    ax.set_yticklabels([_short(a) for a in valid_algos], fontsize=10)
    ax.set_xlim(x_min - (x_max - x_min) * 0.05,
                x_max + (x_max - x_min) * 0.18)
    _kfmt(ax, 'x')
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))

    # Legend for mean marker
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#E03030',
               markeredgecolor='white', markersize=5.5, label='Mean'),
        Line2D([0], [0], color='#222222', linewidth=2, label='Median'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
              framealpha=0.8)

    ax.set_title(title or f"Distribution: {METRIC_LABELS.get(metric, metric)}"
                 f"{' — ' + dataset if dataset else ''}", fontweight='bold')
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Radar / Performance Profile
# ═══════════════════════════════════════════════════════════════════════════════
def plot_radar(df, dataset=None, title=None, save_path=None):
    """Radar chart — each axis is a soft component normalized to [0, 1].
    Smaller polygon = better."""
    if not HAS_MPL: return
    _style()
    data = df[df['dataset'] == dataset] if dataset else df.copy()
    means = data.groupby('algorithm')[SOFT_KEYS].mean()
    algos = [a for a in _order(means.index) if a in means.index]
    if len(algos) < 2: return

    vals = means.loc[algos].values
    col_max = vals.max(axis=0)
    col_max[col_max == 0] = 1
    norm = vals / col_max

    n_cats = len(SOFT_LABELS)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(30)

    for i, algo in enumerate(algos):
        values = norm[i].tolist() + [norm[i][0]]
        ax.plot(angles, values, 'o-', linewidth=2, label=_short(algo),
                color=_c(algo), markersize=5, alpha=0.85)
        ax.fill(angles, values, alpha=0.06, color=_c(algo))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(SOFT_LABELS, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7, alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
    ax.set_title(title or f"Performance Profile"
                 f"{' — ' + dataset if dataset else ''}",
                 fontweight='bold', pad=20)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Ranking Table
# ═══════════════════════════════════════════════════════════════════════════════
def plot_rank_table(df, dataset=None, title=None, save_path=None):
    """Color-coded ranking table: soft penalty, feasibility, runtime, rank."""
    if not HAS_MPL: return
    _style()
    data = df[df['dataset'] == dataset] if dataset else df.copy()
    if data.empty: return

    stats = data.groupby('algorithm').agg(
        soft_mean=('soft_penalty', 'mean'),
        soft_std=('soft_penalty', 'std'),
        feasible_pct=('feasible', lambda x: x.mean() * 100),
        runtime_mean=('runtime', 'mean'),
        n_runs=('soft_penalty', 'count'),
    ).fillna(0).sort_values('soft_mean')
    stats['rank'] = range(1, len(stats) + 1)

    algos = stats.index.tolist()
    col_labels = ['Rank', 'Algorithm', 'Soft Penalty', 'Std Dev',
                  'Feasible %', 'Runtime (s)', 'Runs']

    cell_text = []
    for algo in algos:
        r = stats.loc[algo]
        cell_text.append([
            f"{int(r['rank'])}", _short(algo),
            f"{r['soft_mean']:,.0f}", f"{r['soft_std']:,.0f}",
            f"{r['feasible_pct']:.0f}%", f"{r['runtime_mean']:.2f}",
            f"{int(r['n_runs'])}",
        ])

    n = len(algos)
    fig, ax = plt.subplots(figsize=(11, max(2.5, n * 0.5 + 1.5)))
    ax.axis('off')

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')

    for i in range(n):
        frac = i / max(n - 1, 1)
        rgb = (int(50 + frac * 180), int(200 - frac * 130), int(80 - frac * 30))
        bg = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(bg)
            table[i + 1, j].set_alpha(0.25)

    ax.set_title(title or f"Algorithm Ranking"
                 f"{' — ' + dataset if dataset else ''}",
                 fontweight='bold', fontsize=13, pad=20)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Convergence Overlay (line chart)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_convergence(traces, title=None, save_path=None):
    """Line chart of convergence curves.
    traces: {algo_name: [(iteration, fitness), ...]}"""
    if not HAS_MPL or not traces: return
    _style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in _order(traces.keys()):
        data = traces[algo]
        if not data: continue
        iters, fitness = zip(*data)
        ax.plot(iters, fitness, label=_short(algo), color=_c(algo),
                linewidth=2.2, alpha=0.85)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness (Hard * 100k + Soft)')
    _kfmt(ax)
    ax.set_title(title or 'Convergence Comparison', fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Line Comparison Across Datasets
# ═══════════════════════════════════════════════════════════════════════════════
def plot_line_across_datasets(df, metric='soft_penalty', title=None, save_path=None):
    """Line chart — one line per algorithm, x = dataset, y = metric mean.
    Better than bar chart when comparing trends across multiple datasets."""
    if not HAS_MPL: return
    _style()
    datasets = list(df['dataset'].unique())
    if len(datasets) < 2: return

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.5), 6))
    x = np.arange(len(datasets))

    for algo in _order(df['algorithm'].unique()):
        adf = df[df['algorithm'] == algo]
        means = [adf[adf['dataset'] == ds][metric].mean() for ds in datasets]
        stds  = [adf[adf['dataset'] == ds][metric].std()  for ds in datasets]
        # Replace NaN std with 0
        stds = [0 if np.isnan(s) else s for s in stds]
        ax.errorbar(x, means, yerr=stds, label=_short(algo),
                    color=_c(algo), marker=_m(algo), linewidth=2.2,
                    markersize=8, capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9)
    if len(datasets) >= 4:
        ax.tick_params(axis='x', rotation=25)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
    _kfmt(ax)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)} Across Datasets",
                 fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Soft Component Lines (line chart per component across algorithms)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_soft_lines(df, dataset=None, title=None, save_path=None):
    """Line chart — x = algorithm, y = soft component value, one line per
    component. Good for spotting which penalties dominate each algorithm."""
    if not HAS_MPL: return
    _style()
    data = df[df['dataset'] == dataset] if dataset else df.copy()
    means = data.groupby('algorithm')[SOFT_KEYS].mean()
    algos = _order(means.index)
    means = means.reindex(algos)

    fig, ax = plt.subplots(figsize=(max(8, len(algos) * 1.2), 6))
    x = np.arange(len(algos))

    for key, label, color in zip(SOFT_KEYS, SOFT_LABELS, SOFT_COLORS):
        vals = means[key].values
        ax.plot(x, vals, 'o-', label=label, color=color,
                linewidth=2, markersize=7, alpha=0.85)

    _apply_xlabels(ax, algos)
    _kfmt(ax)
    ax.set_ylabel('Penalty Value')
    ax.set_title(title or f"Soft Components by Algorithm"
                 f"{' — ' + dataset if dataset else ''}", fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Continuous Scan (size / parameter sweep with numeric x-axis)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_continuous_scan(df, x_col='num_exams', title=None, save_path=None):
    """Continuous-axis line chart for size/parameter scans.

    Layout: two subplots side-by-side.
      Left  — soft_penalty vs x_col, single line with ±std ribbon.
      Right — runtime (left y-axis) + memory_peak_mb (right y-axis),
              two lines with ±std ribbons, dual y-axis.

    Aggregates rows by x_col using mean + std. Numeric x-axis (not categorical).

    Expected df columns:
        x_col, 'soft_penalty', 'runtime', 'memory_peak_mb'
    Multiple rows per x value (e.g., multiple seeds) are aggregated.
    """
    if not HAS_MPL:
        return
    _style()

    required = {x_col, 'soft_penalty', 'runtime', 'memory_peak_mb'}
    missing = required - set(df.columns)
    if missing:
        print(f"[plot_continuous_scan] missing columns: {missing}")
        return

    grouped = df.groupby(x_col).agg(
        soft_mean=('soft_penalty', 'mean'),
        soft_std =('soft_penalty', 'std'),
        rt_mean  =('runtime', 'mean'),
        rt_std   =('runtime', 'std'),
        mem_mean =('memory_peak_mb', 'mean'),
        mem_std  =('memory_peak_mb', 'std'),
    ).reset_index().fillna(0)

    x = grouped[x_col].values

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: soft penalty ───────────────────────────────────
    soft_color = '#E15759'  # HHO red
    ax_l.plot(x, grouped['soft_mean'], '-o', color=soft_color,
              linewidth=2.2, markersize=7, label='Soft Penalty')
    ax_l.fill_between(x,
                      grouped['soft_mean'] - grouped['soft_std'],
                      grouped['soft_mean'] + grouped['soft_std'],
                      color=soft_color, alpha=0.2)
    ax_l.set_xlabel(x_col.replace('_', ' ').title())
    ax_l.set_ylabel(METRIC_LABELS.get('soft_penalty', 'Soft Penalty'))
    ax_l.set_title('Quality', fontweight='bold')
    ax_l.grid(True, alpha=0.3)
    _kfmt(ax_l)

    # ── Right: runtime (left axis) + memory (right axis) ─────
    rt_color  = '#F28E2B'  # Tabu orange
    mem_color = '#4E79A7'  # Greedy blue

    l1 = ax_r.plot(x, grouped['rt_mean'], '-o', color=rt_color,
                   linewidth=2.2, markersize=7, label='Runtime (s)')
    ax_r.fill_between(x,
                      grouped['rt_mean'] - grouped['rt_std'],
                      grouped['rt_mean'] + grouped['rt_std'],
                      color=rt_color, alpha=0.2)
    ax_r.set_xlabel(x_col.replace('_', ' ').title())
    ax_r.set_ylabel('Runtime (s)', color=rt_color)
    ax_r.tick_params(axis='y', labelcolor=rt_color)
    ax_r.grid(True, alpha=0.3)

    ax_r2 = ax_r.twinx()
    mscale, munit, _ = _mem_unit(grouped['mem_mean'].tolist())
    mem_scaled = grouped['mem_mean'] * mscale
    mem_std_scaled = grouped['mem_std'] * mscale
    l2 = ax_r2.plot(x, mem_scaled, '-s', color=mem_color,
                    linewidth=2.2, markersize=7, label=f'Peak Memory ({munit})')
    ax_r2.fill_between(x,
                       mem_scaled - mem_std_scaled,
                       mem_scaled + mem_std_scaled,
                       color=mem_color, alpha=0.2)
    ax_r2.set_ylabel(f'Peak Memory ({munit})', color=mem_color)
    ax_r2.tick_params(axis='y', labelcolor=mem_color)
    ax_r2.grid(False)

    ax_r.set_title('Cost', fontweight='bold')

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax_r.legend(lines, labels, loc='upper left', fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive experiment summary (Plotly)
# ═══════════════════════════════════════════════════════════════════════════════

def _hex_to_rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip('#')
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    return f'rgba({int(h[:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})'

_PLOTLY_MARKERS = {
    'o': 'circle', '^': 'triangle-up', 'v': 'triangle-down',
    's': 'square', 'P': 'cross', 'X': 'x', 'h': 'hexagon',
    'p': 'pentagon', 'H': 'hexagon2', 'D': 'diamond',
    'd': 'diamond-wide', '*': 'star',
}

def _save_plotly(fig, save_path):
    """Save a Plotly figure — HTML (no deps) or image (needs kaleido)."""
    if save_path.endswith('.html'):
        fig.write_html(save_path, include_plotlyjs='cdn')
    else:
        fig.write_image(save_path)


def _mem_unit(mb_values):
    """Pick best memory unit based on max value. Returns (scale, label, fmt)."""
    mx = max(mb_values) if mb_values else 0
    if mx >= 1.0:
        return 1.0, 'MB', '.1f'
    if mx >= 0.001:
        return 1024.0, 'KB', '.0f'
    return 1024.0 * 1024.0, 'B', '.0f'


def plot_experiment_summary(df, save_path=None):
    """Interactive summary: soft penalty, runtime, and memory across datasets.

    Three-panel line chart with categorical (evenly-spaced) x-axis so datasets
    with similar sizes don't overlap.  One line per algorithm with ±1σ ribbon.
    Averages across seeds/trials.  Only feasible runs plotted.

    Requires: ``pip install plotly``
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as _np
    except ImportError:
        print("[plot_experiment_summary] pip install plotly")
        return None

    feasible = df[df['feasible'] == True].copy() if 'feasible' in df.columns else df.copy()
    if feasible.empty:
        print("[plot_experiment_summary] No feasible runs to plot")
        return None

    has_mem = 'memory_peak_mb' in feasible.columns
    agg_dict = {
        'soft_m': ('soft_penalty', 'mean'), 'soft_s': ('soft_penalty', 'std'),
        'rt_m': ('runtime', 'mean'),        'rt_s': ('runtime', 'std'),
    }
    if has_mem:
        agg_dict['mem_m'] = ('memory_peak_mb', 'mean')
        agg_dict['mem_s'] = ('memory_peak_mb', 'std')

    # Build categorical x labels sorted by size
    ds_size = (feasible.groupby('dataset')['num_exams'].first()
                       .sort_values().reset_index())
    ds_size['label'] = ds_size.apply(
        lambda r: f"{r['dataset'].replace('exam_comp_', '')} ({r['num_exams']})", axis=1)
    label_order = ds_size['label'].tolist()
    ds_to_label = dict(zip(ds_size['dataset'], ds_size['label']))

    feasible['ds_label'] = feasible['dataset'].map(ds_to_label)

    g = feasible.groupby(['ds_label', 'algorithm']).agg(**agg_dict).reset_index().fillna(0)

    # Auto-scale memory unit
    if has_mem:
        mscale, munit, mfmt = _mem_unit(g['mem_m'].tolist())
        g['mem_m'] = g['mem_m'] * mscale
        g['mem_s'] = g['mem_s'] * mscale

    algos = _order([a for a in g['algorithm'].unique()])

    ncols = 3 if has_mem else 2
    titles = ['<b>Soft Penalty</b>', '<b>Runtime</b>']
    if has_mem:
        titles.append(f'<b>Peak Memory</b>')

    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles,
                        horizontal_spacing=0.07)

    panels = [
        (1, 'soft_m', 'soft_s', 'Soft', '%{y:,.0f}'),
        (2, 'rt_m',   'rt_s',   'Time', '%{y:.2f}s'),
    ]
    if has_mem:
        panels.append((3, 'mem_m', 'mem_s', 'Mem', f'%{{y:{mfmt}}} {munit}'))

    for algo in algos:
        ad = g[g['algorithm'] == algo].copy()
        if ad.empty:
            continue
        # Sort by the categorical order
        ad['_ord'] = ad['ds_label'].map({l: i for i, l in enumerate(label_order)})
        ad = ad.sort_values('_ord')

        x = ad['ds_label'].values
        c = ALGO_COLORS.get(algo, '#888888')
        band = _hex_to_rgba(c, 0.10)
        short = ALGO_SHORT.get(algo, algo)
        sym = _PLOTLY_MARKERS.get(ALGO_MARKERS.get(algo, 'o'), 'circle')

        for col, ym, ys, label, fmt in panels:
            fig.add_trace(go.Scatter(
                x=x, y=ad[ym].values,
                name=short, legendgroup=algo,
                mode='lines+markers', showlegend=(col == 1),
                line=dict(color=c, width=2.4, shape='spline'),
                marker=dict(size=7, symbol=sym),
                hovertemplate=f'{short}: {fmt}<extra></extra>',
            ), row=1, col=col)

            upper = (ad[ym] + ad[ys]).values
            lower = _np.maximum(0, (ad[ym] - ad[ys]).values)
            fig.add_trace(go.Scatter(
                x=list(x) + list(x[::-1]),
                y=_np.concatenate([upper, lower[::-1]]),
                fill='toself', fillcolor=band, line=dict(width=0),
                showlegend=False, legendgroup=algo, hoverinfo='skip',
            ), row=1, col=col)

    fig.update_layout(
        width=1200, height=480,
        template='plotly_white',
        font=dict(family='Inter, -apple-system, system-ui, sans-serif', size=12),
        legend=dict(
            orientation='h', yanchor='top', y=-0.22,
            xanchor='center', x=0.5,
            font=dict(size=11), tracegroupgap=5,
        ),
        margin=dict(t=50, b=110, l=60, r=30),
        hovermode='x unified',
    )

    for col in range(1, ncols + 1):
        fig.update_xaxes(
            categoryorder='array', categoryarray=label_order,
            tickangle=-35, row=1, col=col,
            gridcolor='rgba(0,0,0,0.06)',
        )
    fig.update_yaxes(title_text='Soft Penalty', row=1, col=1,
                     gridcolor='rgba(0,0,0,0.06)')
    fig.update_yaxes(title_text='Seconds', row=1, col=2,
                     gridcolor='rgba(0,0,0,0.06)')
    if has_mem:
        fig.update_yaxes(title_text=munit, row=1, col=3,
                         gridcolor='rgba(0,0,0,0.06)')

    if save_path:
        try: _save_plotly(fig, save_path)
        except Exception: pass
    fig.show()
    return fig


def plot_algo_bars(df, save_path=None):
    """Horizontal bar chart: soft penalty, runtime, and memory per algorithm."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[plot_algo_bars] pip install plotly"); return None

    feas = df[df['feasible'] == True].copy() if 'feasible' in df.columns else df.copy()
    if feas.empty:
        print("[plot_algo_bars] No feasible runs"); return None

    has_mem = 'memory_peak_mb' in feas.columns
    agg_dict = {
        'soft_m': ('soft_penalty', 'mean'), 'soft_s': ('soft_penalty', 'std'),
        'rt_m': ('runtime', 'mean'), 'rt_s': ('runtime', 'std'),
        'n': ('soft_penalty', 'count'),
    }
    if has_mem:
        agg_dict['mem_m'] = ('memory_peak_mb', 'mean')
        agg_dict['mem_s'] = ('memory_peak_mb', 'std')

    g = feas.groupby('algorithm').agg(**agg_dict).reset_index().fillna(0)

    # Auto-scale memory unit
    if has_mem:
        mscale, munit, mfmt = _mem_unit(g['mem_m'].tolist())
        g['mem_m'] = g['mem_m'] * mscale
        g['mem_s'] = g['mem_s'] * mscale

    ncols = 3 if has_mem else 2
    titles = ['<b>Soft Penalty</b>', '<b>Runtime</b>']
    if has_mem:
        titles.append('<b>Peak Memory</b>')

    panels = [
        ('soft_m', 'soft_s', 'Soft Penalty', '%{x:,.0f}'),
        ('rt_m',   'rt_s',   'Seconds',      '%{x:.2f}s'),
    ]
    if has_mem:
        panels.append(('mem_m', 'mem_s', munit, f'%{{x:{mfmt}}} {munit}'))

    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles,
                        horizontal_spacing=0.14)

    for col, (ym, ys, xlabel, fmt) in enumerate(panels, 1):
        gs = g.sort_values(ym, ascending=True)
        shorts = [ALGO_SHORT.get(a, a) for a in gs['algorithm']]
        colors = [ALGO_COLORS.get(a, '#888888') for a in gs['algorithm']]
        fig.add_trace(go.Bar(
            y=shorts, x=gs[ym].values, orientation='h',
            error_x=dict(type='data', array=gs[ys].values, thickness=1.2),
            marker_color=colors, showlegend=False,
            hovertemplate=f'%{{y}}: {fmt}<extra></extra>',
        ), row=1, col=col)
        fig.update_xaxes(title_text=xlabel, row=1, col=col,
                         gridcolor='rgba(0,0,0,0.06)')

    h = max(350, len(g) * 38)
    fig.update_layout(
        width=1150 if has_mem else 1050, height=h,
        template='plotly_white',
        font=dict(family='Inter, -apple-system, system-ui, sans-serif', size=12),
        margin=dict(t=45, b=30, l=80, r=30),
    )

    if save_path:
        try: _save_plotly(fig, save_path)
        except Exception: pass
    fig.show()
    return fig


def plot_algo_boxes(df, save_path=None):
    """Box plot: soft penalty distribution per algorithm, all datasets pooled."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[plot_algo_boxes] pip install plotly"); return None

    feas = df[df['feasible'] == True].copy() if 'feasible' in df.columns else df.copy()
    if feas.empty:
        print("[plot_algo_boxes] No feasible runs"); return None

    medians = feas.groupby('algorithm')['soft_penalty'].median().sort_values()
    algos = medians.index.tolist()

    fig = go.Figure()
    for algo in algos:
        ad = feas[feas['algorithm'] == algo]
        short = ALGO_SHORT.get(algo, algo)
        fig.add_trace(go.Box(
            y=ad['soft_penalty'].values, name=short,
            marker_color=ALGO_COLORS.get(algo, '#888888'),
            boxpoints='all', jitter=0.4, pointpos=-1.5,
            marker=dict(size=4, opacity=0.5),
            line=dict(width=1.8),
            hoverinfo='y+name',
        ))

    fig.update_layout(
        width=1050, height=480, template='plotly_white',
        font=dict(family='Inter, -apple-system, system-ui, sans-serif', size=12),
        yaxis_title='Soft Penalty', showlegend=False,
        title=dict(text='<b>Soft Penalty Distribution</b>', x=0.5, font_size=14),
        margin=dict(t=50, b=40, l=65, r=30),
        yaxis=dict(gridcolor='rgba(0,0,0,0.06)'),
    )

    if save_path:
        try: _save_plotly(fig, save_path)
        except Exception: pass
    fig.show()
    return fig


def plot_algo_radar(df, save_path=None):
    """Radar chart: normalised performance profile per algorithm (smaller = better)."""
    try:
        import plotly.graph_objects as go
        import numpy as _np
    except ImportError:
        print("[plot_algo_radar] pip install plotly"); return None

    feas = df[df['feasible'] == True].copy() if 'feasible' in df.columns else df.copy()
    if feas.empty:
        print("[plot_algo_radar] No feasible runs"); return None

    metrics = [('soft_penalty', 'Soft Penalty', False),
               ('runtime', 'Runtime', False),
               ('memory_peak_mb', 'Memory', False),
               ('two_in_a_row', '2-in-a-Row', False),
               ('two_in_a_day', '2-in-a-Day', False),
               ('period_spread', 'Period Spread', False),
               ('non_mixed_durations', 'Mixed Dur.', False),
               ('front_load', 'Front Load', False)]
    # Keep only metrics that exist in df
    metrics = [(k, l, inv) for k, l, inv in metrics if k in feas.columns]
    if len(metrics) < 3:
        print("[plot_algo_radar] Need >= 3 metrics"); return None

    g = feas.groupby('algorithm')[[m[0] for m in metrics]].mean()
    algos = _order([a for a in g.index])

    # Normalise 0–1 per metric (0 = best)
    normed = g.copy()
    for col, _, inv in metrics:
        mn, mx = g[col].min(), g[col].max()
        if mx > mn:
            normed[col] = (g[col] - mn) / (mx - mn)
            if inv:
                normed[col] = 1.0 - normed[col]
        else:
            normed[col] = 0.0

    cats = [m[1] for m in metrics]

    fig = go.Figure()
    for algo in algos:
        if algo not in normed.index:
            continue
        vals = [normed.loc[algo, m[0]] for m in metrics]
        vals.append(vals[0])  # close the polygon
        short = ALGO_SHORT.get(algo, algo)
        c = ALGO_COLORS.get(algo, '#888888')
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            name=short, line=dict(color=c, width=2.2),
            fill='toself', fillcolor=_hex_to_rgba(c, 0.08),
        ))

    fig.update_layout(
        width=650, height=520, template='plotly_white',
        font=dict(family='Inter, -apple-system, system-ui, sans-serif', size=11),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05],
                            gridcolor='rgba(0,0,0,0.08)', tickfont_size=9),
            angularaxis=dict(gridcolor='rgba(0,0,0,0.08)'),
        ),
        title=dict(text='<b>Performance Profile</b> <sub>(smaller = better)</sub>',
                   x=0.5, font_size=14),
        legend=dict(font_size=10),
        margin=dict(t=70, b=40, l=65, r=65),
    )

    if save_path:
        try: _save_plotly(fig, save_path)
        except Exception: pass
    fig.show()
    return fig


def _spread_labels(xs, ys, min_gap_frac=0.04):
    """Pick per-point textposition to reduce overlap.

    Alternates between positions around each marker based on proximity
    to neighbours so labels don't pile up.
    """
    import numpy as _np
    n = len(xs)
    if n == 0:
        return []
    positions = ['top center'] * n
    opts = ['top center', 'bottom center', 'top right', 'top left',
            'bottom right', 'bottom left', 'middle right', 'middle left']
    xs, ys = _np.array(xs, dtype=float), _np.array(ys, dtype=float)
    xr = xs.max() - xs.min() if xs.max() != xs.min() else 1.0
    yr = ys.max() - ys.min() if ys.max() != ys.min() else 1.0
    for i in range(n):
        best, best_min = opts[0], -1.0
        for pos in opts:
            min_d = float('inf')
            for j in range(n):
                if j == i:
                    continue
                d = (((xs[i]-xs[j])/xr)**2 + ((ys[i]-ys[j])/yr)**2)**0.5
                min_d = min(min_d, d)
            if min_d > best_min:
                best, best_min = pos, min_d
        positions[i] = best
    return positions


def plot_algo_scatter(df, save_path=None):
    """Scatter: quality (y) vs runtime (x), one point per algorithm."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[plot_algo_scatter] pip install plotly"); return None

    feas = df[df['feasible'] == True].copy() if 'feasible' in df.columns else df.copy()
    if feas.empty:
        print("[plot_algo_scatter] No feasible runs"); return None

    g = feas.groupby('algorithm').agg(
        soft_m=('soft_penalty', 'mean'), rt_m=('runtime', 'mean'),
    ).reset_index()

    algos = g['algorithm'].tolist()
    shorts = [ALGO_SHORT.get(a, a) for a in algos]
    colors = [ALGO_COLORS.get(a, '#888888') for a in algos]
    syms = [_PLOTLY_MARKERS.get(ALGO_MARKERS.get(a, 'o'), 'circle') for a in algos]

    fig = go.Figure()
    for i, algo in enumerate(algos):
        r = g[g['algorithm'] == algo].iloc[0]
        fig.add_trace(go.Scatter(
            x=[r['rt_m']], y=[r['soft_m']],
            mode='markers',
            marker=dict(size=13, color=colors[i], symbol=syms[i],
                        line=dict(width=1.2, color='white')),
            name=shorts[i], showlegend=True,
            hovertemplate=(f'<b>{shorts[i]}</b><br>Soft: %{{y:,.0f}}<br>'
                           f'Time: %{{x:.2f}}s<extra></extra>'),
        ))

    fig.update_layout(
        width=700, height=520,
        template='plotly_white',
        title=dict(text='<b>Quality vs Runtime</b>', x=0.5, font_size=14),
        font=dict(family='Inter, -apple-system, system-ui, sans-serif', size=12),
        margin=dict(t=55, b=50, l=70, r=140),
        xaxis=dict(title='Mean Runtime (s)', gridcolor='rgba(0,0,0,0.06)'),
        yaxis=dict(title='Mean Soft Penalty', gridcolor='rgba(0,0,0,0.06)'),
        legend=dict(yanchor='top', y=0.98, xanchor='left', x=1.02,
                    font_size=10),
    )

    if save_path:
        try: _save_plotly(fig, save_path)
        except Exception: pass
    fig.show()
    return fig


def plot_algo_heatmap(df, save_path=None):
    """Heatmap: algorithm x dataset, coloured by mean soft penalty.

    Cells with no feasible result show 'n/f' (not feasible) in grey.
    Algorithms that only appear on a subset of datasets (e.g. Feasibility,
    Greedy) are excluded to keep the grid clean.
    """
    try:
        import plotly.graph_objects as go
        import numpy as _np
    except ImportError:
        print("[plot_algo_heatmap] pip install plotly"); return None

    feas = df[df['feasible'] == True].copy() if 'feasible' in df.columns else df.copy()
    if feas.empty:
        print("[plot_algo_heatmap] No feasible runs"); return None

    # Exclude init-only algorithms (they don't run on every set)
    skip = {'Greedy', 'Feasibility'}
    feas = feas[~feas['algorithm'].isin(skip)]
    if feas.empty:
        print("[plot_algo_heatmap] No search-algorithm data"); return None

    all_algos = sorted(df[~df['algorithm'].isin(skip)]['algorithm'].unique())
    all_datasets = sorted(df['dataset'].unique())

    g = feas.groupby(['algorithm', 'dataset'])['soft_penalty'].mean().reset_index()
    pivot = g.pivot(index='algorithm', columns='dataset', values='soft_penalty')
    pivot = pivot.reindex(index=all_algos, columns=all_datasets)

    # Build infeasible lookup: algorithm ran but never reached feasible
    infeas_set = set()
    for algo in all_algos:
        for ds in all_datasets:
            sub = df[(df['algorithm'] == algo) & (df['dataset'] == ds)]
            if len(sub) > 0 and sub['feasible'].sum() == 0:
                infeas_set.add((algo, ds))

    # Order: datasets by num_exams, algorithms by overall mean soft
    ds_size = df.groupby('dataset')['num_exams'].first().sort_values()
    ds_order = [d for d in ds_size.index if d in pivot.columns]
    ds_labels = [d.replace('exam_comp_', '') for d in ds_order]

    algo_means = pivot[ds_order].mean(axis=1).sort_values()
    algo_order = [a for a in algo_means.index if a in pivot.index]
    algo_labels = [ALGO_SHORT.get(a, a) for a in algo_order]

    z = pivot.loc[algo_order, ds_order].values

    # Per-column normalisation
    z_norm = _np.full_like(z, dtype=float, fill_value=_np.nan)
    for j in range(z.shape[1]):
        col = z[:, j]
        valid = ~_np.isnan(col)
        if valid.any():
            mn, mx = _np.nanmin(col), _np.nanmax(col)
            if mx > mn:
                z_norm[:, j] = (col - mn) / (mx - mn)
            else:
                z_norm[valid, j] = 0.0

    # Annotation text: raw values or 'n/f'
    text = []
    for i, algo in enumerate(algo_order):
        row_text = []
        for j, ds in enumerate(ds_order):
            v = z[i, j]
            if not _np.isnan(v):
                row_text.append(f'{v:,.0f}')
            elif (algo, ds) in infeas_set:
                row_text.append('n/f')
            else:
                row_text.append('-')
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z_norm, x=ds_labels, y=algo_labels,
        text=text, texttemplate='%{text}', textfont=dict(size=10),
        colorscale='RdYlGn_r', showscale=True,
        colorbar=dict(title='Relative', tickvals=[0, 0.5, 1],
                      ticktext=['Best', 'Mid', 'Worst']),
        hovertemplate=('<b>%{y}</b> on %{x}<br>'
                       'Soft: %{text}<extra></extra>'),
        xgap=2, ygap=2,
    ))

    fig.update_layout(
        width=max(550, len(ds_order) * 85 + 180),
        height=max(400, len(algo_order) * 38 + 120),
        template='plotly_white',
        font=dict(family='Inter, -apple-system, system-ui, sans-serif', size=12),
        title=dict(text='<b>Soft Penalty Heatmap</b> <sub>(per-dataset normalised, n/f = not feasible)</sub>',
                   x=0.5, font_size=14),
        xaxis=dict(title='', side='bottom'),
        yaxis=dict(title='', autorange='reversed'),
        margin=dict(t=65, b=40, l=80, r=30),
    )

    if save_path:
        try: _save_plotly(fig, save_path)
        except Exception: pass
    fig.show()
    return fig


def save_all_plotly(df, output_dir="graphs"):
    """Save all Plotly summary charts as PNG to graphs/.

    Suppresses fig.show() so charts are written to disk without
    duplicating the notebook display output.
    """
    import os, plotly.graph_objects as go
    os.makedirs(output_dir, exist_ok=True)
    _orig = go.Figure.show
    go.Figure.show = lambda self, *a, **k: None
    try:
        p = lambda name: os.path.join(output_dir, name)
        plot_experiment_summary(df, save_path=p('summary_lines.png'))
        plot_algo_bars(df,              save_path=p('algo_bars.png'))
        plot_algo_boxes(df,             save_path=p('algo_boxes.png'))
        plot_algo_radar(df,             save_path=p('algo_radar.png'))
        plot_algo_scatter(df,           save_path=p('algo_scatter.png'))
        plot_algo_heatmap(df,           save_path=p('algo_heatmap.png'))
    finally:
        go.Figure.show = _orig
    print(f"[Plot] Saved 6 charts to {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# Batch generation + legacy
# ═══════════════════════════════════════════════════════════════════════════════
def generate_all_plots(logger_or_df, output_dir="results"):
    """Generate the full suite of plots and save to disk."""
    if not HAS_MPL:
        print("[Plot] matplotlib not available"); return
    import pandas as pd
    df = (logger_or_df.to_dataframe()
          if hasattr(logger_or_df, 'to_dataframe') else logger_or_df)
    if df.empty:
        print("[Plot] No data"); return
    os.makedirs(output_dir, exist_ok=True)

    for ds in df['dataset'].unique():
        s = ds.replace(' ', '_')
        plot_algorithm_comparison(df, dataset=ds, metric='soft_penalty',
                                  save_path=f"{output_dir}/{s}_soft_comparison.png")
        plot_algorithm_comparison(df, dataset=ds, metric='runtime',
                                  save_path=f"{output_dir}/{s}_runtime_comparison.png")
        plot_soft_breakdown(df, dataset=ds,
                            save_path=f"{output_dir}/{s}_soft_breakdown.png")
        plot_soft_lines(df, dataset=ds,
                        save_path=f"{output_dir}/{s}_soft_lines.png")
        plot_runtime_vs_quality(df, dataset=ds,
                                save_path=f"{output_dir}/{s}_quality_vs_time.png")
        plot_box_comparison(df, dataset=ds,
                            save_path=f"{output_dir}/{s}_box_comparison.png")
        plot_radar(df, dataset=ds,
                   save_path=f"{output_dir}/{s}_radar.png")
        plot_rank_table(df, dataset=ds,
                        save_path=f"{output_dir}/{s}_ranking.png")

    if len(df['dataset'].unique()) > 1:
        plot_multi_dataset_heatmap(df, save_path=f"{output_dir}/heatmap_soft.png")
        plot_line_across_datasets(df, save_path=f"{output_dir}/line_across_datasets.png")
        plot_feasibility_rates(df, save_path=f"{output_dir}/feasibility_rates.png")

    plot_summary_dashboard(df, save_path=f"{output_dir}/dashboard.png")
    print(f"[Plot] All plots saved to {output_dir}/")


def plot_soft_constraint_breakdown(breakdown: dict, output_dir: str = "results"):
    """Legacy interface for main.py."""
    plot_soft_breakdown(breakdown,
                        save_path=os.path.join(output_dir, "soft_constraint_breakdown.png"))
