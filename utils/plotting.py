"""
Reads from ResultsLogger DataFrames and produces plots for:
  1. Algorithm comparison bars  2. Soft/hard breakdown
  3. Runtime vs quality scatter 4. Multi-dataset heatmap
  5. Parameter sensitivity      6. Scaling analysis
  7. Feasibility rates          8. Summary dashboard
"""
import json, os
from collections import defaultdict
 
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
 
ALGO_COLORS = {'Greedy':'#2196F3','IP':'#F44336','Tabu Search':'#4CAF50','HHO':'#FF9800'}
ALGO_MARKERS = {'Greedy':'o','IP':'s','Tabu Search':'^','HHO':'D'}
SOFT_KEYS   = ['two_in_a_row','two_in_a_day','period_spread','non_mixed_durations','front_load','period_penalty','room_penalty']
SOFT_LABELS = ['2-in-a-Row','2-in-a-Day','Period Spread','Mixed Duration','Front Load','Period Pen.','Room Pen.']
SOFT_COLORS = ['#E53935','#FB8C00','#FDD835','#43A047','#1E88E5','#8E24AA','#6D4C41']
 
def _c(algo): return ALGO_COLORS.get(algo,'#9E9E9E')
 
def _style():
    plt.rcParams.update({'figure.figsize':(10,6),'figure.dpi':120,'font.size':11,
        'axes.titlesize':14,'axes.labelsize':12,'axes.grid':True,'grid.alpha':0.3,
        'legend.fontsize':10,'figure.facecolor':'white'})
 
def plot_algorithm_comparison(df, dataset=None, metric='soft_penalty', title=None, save_path=None, ax=None):
    if not HAS_MPL: return
    _style()
    data = df[df['dataset']==dataset] if dataset else df.copy()
    grouped = data.groupby('algorithm')[metric].agg(['mean','std','min','count']).reset_index().sort_values('mean')
    if ax is None: fig, ax = plt.subplots(figsize=(8,5))
    else: fig = ax.figure
    colors = [_c(a) for a in grouped['algorithm']]
    bars = ax.bar(grouped['algorithm'], grouped['mean'], yerr=grouped['std'], color=colors, alpha=0.85, capsize=5)
    for bar, (_, row) in zip(bars, grouped.iterrows()):
        lbl = f"{row['mean']:.0f}" if row['mean']>10 else f"{row['mean']:.2f}"
        if row['count']>1: lbl += f"\n(n={int(row['count'])})"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), lbl, ha='center', va='bottom', fontsize=9, fontweight='bold')
    labels = {'soft_penalty':'Soft Penalty','runtime':'Runtime (s)','hard_violations':'Hard Violations','memory_peak_mb':'Peak Memory (MB)'}
    ax.set_ylabel(labels.get(metric, metric))
    ds = f" — {dataset}" if dataset else ""
    ax.set_title(title or f"Algorithm Comparison: {labels.get(metric, metric)}{ds}", fontweight='bold')
    if save_path: fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
 
def plot_soft_breakdown(df_or_dict, dataset=None, title=None, save_path=None):
    if not HAS_MPL: return
    _style()
    import pandas as pd
    if isinstance(df_or_dict, pd.DataFrame):
        data = df_or_dict[df_or_dict['dataset']==dataset] if dataset else df_or_dict
        breakdown = {a: {k: g[k].mean() for k in SOFT_KEYS} for a, g in data.groupby('algorithm')}
    else:
        breakdown = df_or_dict
    algos = list(breakdown.keys()); n = len(algos)
    fig, ax = plt.subplots(figsize=(max(8, n*2.2), 6))
    x = range(n); bottoms = [0.0]*n
    for key, label, color in zip(SOFT_KEYS, SOFT_LABELS, SOFT_COLORS):
        vals = [breakdown[a].get(key, 0) for a in algos]
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.88, width=0.6)
        for j, v in enumerate(vals):
            total = sum(breakdown[algos[j]].get(k,0) for k in SOFT_KEYS)
            if v > 0 and total > 0 and v/total > 0.05:
                ax.text(j, bottoms[j]+v/2, f"{v:.0f}", ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        bottoms = [b+v for b, v in zip(bottoms, vals)]
    for j, a in enumerate(algos):
        ax.text(j, bottoms[j]+max(bottoms)*0.01, f"{bottoms[j]:.0f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(algos, fontsize=12)
    ax.set_ylabel('Soft Penalty')
    ax.set_title(title or f"Soft Constraint Breakdown{' — '+dataset if dataset else ''}", fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
 
def plot_runtime_vs_quality(df, dataset=None, title=None, save_path=None):
    if not HAS_MPL: return
    _style()
    data = df[df['feasible']==True].copy()
    if dataset: data = data[data['dataset']==dataset]
    fig, ax = plt.subplots(figsize=(10,6))
    for algo in sorted(data['algorithm'].unique()):
        adf = data[data['algorithm']==algo]
        ax.scatter(adf['runtime'], adf['soft_penalty'], label=algo, color=_c(algo),
                   marker=ALGO_MARKERS.get(algo,'o'), s=80, alpha=0.7, edgecolors='white')
    ax.set_xlabel('Runtime (s)'); ax.set_ylabel('Soft Penalty')
    ax.set_title(title or f"Quality vs Time{' — '+dataset if dataset else ''}", fontweight='bold')
    ax.legend(); fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
 
def plot_multi_dataset_heatmap(df, metric='soft_penalty', title=None, save_path=None):
    if not HAS_MPL: return
    _style()
    pivot = df.pivot_table(values=metric, index='algorithm', columns='dataset', aggfunc='mean')
    if pivot.empty: return
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*2.5), max(4, len(pivot.index)*1.2)))
    im = ax.imshow(pivot.values, cmap='YlOrRd_r', aspect='auto')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i,j]
            if not np.isnan(v):
                c = 'white' if v > np.nanmean(pivot.values) else 'black'
                ax.text(j, i, f"{v:.0f}", ha='center', va='center', fontsize=10, color=c, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8, label=metric.replace('_',' ').title())
    ax.set_title(title or f"Cross-Dataset: {metric.replace('_',' ').title()}", fontweight='bold')
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
 
def plot_feasibility_rates(df, title=None, save_path=None):
    if not HAS_MPL: return
    _style()
    rates = df.groupby('algorithm')['feasible'].mean().sort_values(ascending=False)*100
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(rates.index, rates.values, color=[_c(a) for a in rates.index], alpha=0.85)
    for bar, v in zip(bars, rates.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{v:.0f}%", ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feasibility Rate (%)'); ax.set_ylim(0,115)
    ax.set_title(title or 'Feasibility Rate', fontweight='bold')
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
 
def plot_scaling(df, x_col='num_exams', y_col='runtime', title=None, save_path=None):
    if not HAS_MPL: return
    _style()
    fig, ax = plt.subplots(figsize=(10,6))
    for algo in sorted(df['algorithm'].unique()):
        adf = df[df['algorithm']==algo]
        g = adf.groupby(x_col)[y_col].agg(['mean','std']).reset_index().sort_values(x_col)
        ax.errorbar(g[x_col], g['mean'], yerr=g['std'], label=algo, color=_c(algo),
                     marker=ALGO_MARKERS.get(algo,'o'), linewidth=2, markersize=8, capsize=4)
    ax.set_xlabel(x_col.replace('_',' ').title()); ax.set_ylabel(y_col.replace('_',' ').title())
    ax.set_title(title or f"{y_col.replace('_',' ').title()} vs {x_col.replace('_',' ').title()}", fontweight='bold')
    ax.legend(); fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
 
def plot_parameter_sensitivity(df, param_col, metric='soft_penalty', algorithm=None, title=None, save_path=None):
    if not HAS_MPL: return
    _style()
    data = df[df['algorithm']==algorithm] if algorithm else df.copy()
    g = data.groupby(param_col).agg(soft_mean=(metric,'mean'), soft_std=(metric,'std'),
        rt_mean=('runtime','mean'), rt_std=('runtime','std')).reset_index().sort_values(param_col)
    fig, ax1 = plt.subplots(figsize=(10,5))
    c1, c2 = '#4CAF50', '#E91E63'
    ax1.errorbar(g[param_col], g['soft_mean'], yerr=g['soft_std'], color=c1, marker='^', linewidth=2, capsize=4, label=metric.replace('_',' ').title())
    ax1.set_xlabel(param_col.replace('_',' ').title()); ax1.set_ylabel(metric.replace('_',' ').title(), color=c1)
    ax2 = ax1.twinx()
    ax2.errorbar(g[param_col], g['rt_mean'], yerr=g['rt_std'], color=c2, marker='s', linewidth=2, capsize=4, linestyle='--', label='Runtime')
    ax2.set_ylabel('Runtime (s)', color=c2)
    al = f" ({algorithm})" if algorithm else ""
    ax1.set_title(title or f"Parameter Sensitivity{al}", fontweight='bold')
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, la1+la2, loc='upper right')
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
 
def plot_summary_dashboard(df, dataset=None, save_path=None):
    if not HAS_MPL: return
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plot_algorithm_comparison(df, dataset=dataset, metric='soft_penalty', ax=axes[0,0])
    plot_algorithm_comparison(df, dataset=dataset, metric='runtime', ax=axes[0,1])
    # Soft breakdown
    data = df[df['dataset']==dataset] if dataset else df
    breakdown = {a: {k: g[k].mean() for k in SOFT_KEYS} for a, g in data.groupby('algorithm')}
    ax = axes[1,0]; algos = list(breakdown.keys()); x = range(len(algos)); bottoms = [0.0]*len(algos)
    for key, label, color in zip(SOFT_KEYS, SOFT_LABELS, SOFT_COLORS):
        vals = [breakdown[a].get(key,0) for a in algos]
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.88, width=0.6)
        bottoms = [b+v for b, v in zip(bottoms, vals)]
    ax.set_xticks(x); ax.set_xticklabels(algos); ax.set_ylabel('Soft Penalty')
    ax.set_title('Soft Constraint Breakdown', fontweight='bold'); ax.legend(fontsize=8)
    # Feasibility
    ax = axes[1,1]; rates = data.groupby('algorithm')['feasible'].mean()*100
    ax.bar(rates.index, rates.values, color=[_c(a) for a in rates.index], alpha=0.85)
    for i, (a, v) in enumerate(rates.items()):
        ax.text(i, v+1, f"{v:.0f}%", ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feasibility (%)'); ax.set_ylim(0,115); ax.set_title('Feasibility Rate', fontweight='bold')
    fig.suptitle(f"Results Dashboard{' — '+dataset if dataset else ''}", fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
 
def generate_all_plots(logger_or_df, output_dir="results"):
    if not HAS_MPL: print("[Plot] matplotlib not available"); return
    import pandas as pd
    df = logger_or_df.to_dataframe() if hasattr(logger_or_df, 'to_dataframe') else logger_or_df
    if df.empty: print("[Plot] No data"); return
    os.makedirs(output_dir, exist_ok=True)
    for ds in df['dataset'].unique():
        s = ds.replace(' ','_')
        plot_algorithm_comparison(df, dataset=ds, metric='soft_penalty', save_path=f"{output_dir}/{s}_soft_comparison.png")
        plot_algorithm_comparison(df, dataset=ds, metric='runtime', save_path=f"{output_dir}/{s}_runtime_comparison.png")
        plot_soft_breakdown(df, dataset=ds, save_path=f"{output_dir}/{s}_soft_breakdown.png")
        plot_runtime_vs_quality(df, dataset=ds, save_path=f"{output_dir}/{s}_quality_vs_time.png")
    if len(df['dataset'].unique()) > 1:
        plot_multi_dataset_heatmap(df, save_path=f"{output_dir}/heatmap_soft.png")
        plot_feasibility_rates(df, save_path=f"{output_dir}/feasibility_rates.png")
    plot_summary_dashboard(df, save_path=f"{output_dir}/dashboard.png")
    print(f"[Plot] All plots saved to {output_dir}/")
 
def plot_soft_constraint_breakdown(breakdown: dict, output_dir: str = "results"):
    """Legacy interface for main.py."""
    plot_soft_breakdown(breakdown, save_path=os.path.join(output_dir, "soft_constraint_breakdown.png"))