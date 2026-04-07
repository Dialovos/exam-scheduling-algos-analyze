"""
Research-quality plotting for exam scheduling experiments.

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
    'Greedy':              '#4E79A7',
    'Tabu Search':         '#F28E2B',
    'HHO':                 '#E15759',
    'Kempe Chain':         '#76B7B2',
    'Simulated Annealing': '#59A14F',
    'ALNS':                '#EDC948',
    'Great Deluge':        '#B07AA1',
    'ABC':                 '#FF9DA7',
    'Genetic Algorithm':   '#9C755F',
    'LAHC':                '#BAB0AC',
    'Natural Selection':   '#4B0082',
    'IP':                  '#D37295',
}
ALGO_MARKERS = {
    'Greedy': 'o', 'Tabu Search': '^', 'HHO': 'D', 'Kempe Chain': 'v',
    'Simulated Annealing': 's', 'ALNS': 'P', 'Great Deluge': 'X',
    'ABC': 'h', 'Genetic Algorithm': 'p', 'LAHC': 'H',
    'Natural Selection': '*', 'IP': 'd',
}
ALGO_SHORT = {
    'Greedy': 'Greedy', 'Tabu Search': 'Tabu', 'HHO': 'HHO',
    'Kempe Chain': 'Kempe', 'Simulated Annealing': 'SA',
    'ALNS': 'ALNS', 'Great Deluge': 'GD',
    'ABC': 'ABC', 'Genetic Algorithm': 'GA', 'LAHC': 'LAHC',
    'Natural Selection': 'NS', 'IP': 'IP',
}
ALGO_ORDER = ['Greedy', 'Tabu Search', 'HHO', 'Kempe Chain',
              'Simulated Annealing', 'ALNS', 'Great Deluge',
              'ABC', 'Genetic Algorithm', 'LAHC', 'Natural Selection', 'IP']

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
