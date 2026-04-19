"""Paper-table package for batch_018 results."""
from utils.tables.paper.t1_leaderboard import make_t1
from utils.tables.paper.t2_chain_top5 import make_t2
from utils.tables.paper.t3_partial_adopt import make_t3
from utils.tables.paper.t4_family_comparison import make_t4

__all__ = ["make_t1", "make_t2", "make_t3", "make_t4"]
