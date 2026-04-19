"""Paper-figure package for batch_018 results."""
from utils.plots.paper.fig1_pareto import make_fig1
from utils.plots.paper.fig2_family_heatmap import make_fig2
from utils.plots.paper.fig3_chain_methodology import make_fig3
from utils.plots.paper.fig4_scaling import make_fig4
from utils.plots.paper.fig5_sensitivity import make_fig5
from utils.plots.paper.fig6_ip_vs_heuristic import make_fig6
from utils.plots.paper.fig7_gap_heatmap import make_fig7
from utils.plots.paper.fig8_gap_leaderboard import make_fig8

__all__ = ["make_fig1", "make_fig2", "make_fig3", "make_fig4",
           "make_fig5", "make_fig6", "make_fig7", "make_fig8"]
