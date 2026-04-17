"""Shared plotting primitives — palette, label helpers, paper-style rcParams.

Every figure in this project pulls its colors, markers, and axis styling from
here. Change one entry and it cascades to every plot in the paper — that's
the whole point. If you find yourself hard-coding an algorithm color inside
a specific plot function, stop and add it to :data:`ALGO_COLORS` instead.

The public API is :func:`algo_color`, :func:`algo_marker`, :func:`algo_short`,
:func:`algo_order`, :func:`apply_paper_style`. The underscore-prefixed aliases
(:func:`_c`, :func:`_m`, ...) are kept for compatibility with existing call
sites inside ``utils/plotting.py`` so the split is invisible to callers.
"""
from __future__ import annotations

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:  # pragma: no cover — hard dep in practice; kept soft for Colab bootstrap.
    HAS_MPL = False


# ── Canonical palette / metadata ─────────────────────────────────────────
# Okabe-Ito-inspired, colorblind-friendly. We duplicate some colors on purpose
# for aliased algorithm names (e.g. "Simulated Annealing" and "Multi-Neighbourhood SA"
# are the same algorithm, and WOA/HHO share the exploration-heavy red because
# they sit in the same family of population-based exploration methods).
ALGO_COLORS = {
    "Greedy":                 "#4E79A7",
    "Feasibility":            "#7B7B7B",
    "Seeder":                 "#2F5D8A",
    "Tabu Search":            "#F28E2B",
    "Kempe Chain":            "#76B7B2",
    "Simulated Annealing":    "#59A14F",
    "Multi-Neighbourhood SA": "#59A14F",
    "ALNS":                   "#EDC948",
    "Great Deluge":           "#B07AA1",
    "ABC":                    "#FF9DA7",
    "Genetic Algorithm":      "#9C755F",
    "GA":                     "#9C755F",
    "LAHC":                   "#BAB0AC",
    "WOA":                    "#E15759",
    "HHO":                    "#E15759",
    "HHO+":                   "#E15759",
    "CP-SAT":                 "#D37295",
    "CP-SAT B&B":             "#D37295",
    "GVNS":                   "#4B0082",
}

ALGO_MARKERS = {
    "Greedy": "o", "Feasibility": "o", "Seeder": "o",
    "Tabu Search": "^", "Kempe Chain": "v",
    "Simulated Annealing": "s", "Multi-Neighbourhood SA": "s",
    "ALNS": "P", "Great Deluge": "X",
    "ABC": "h", "Genetic Algorithm": "p", "GA": "p", "LAHC": "H",
    "WOA": "D", "HHO": "D", "HHO+": "D",
    "CP-SAT": "d", "CP-SAT B&B": "d", "GVNS": "*",
}

ALGO_SHORT = {
    "Greedy": "Greedy", "Feasibility": "Feas", "Seeder": "Seed",
    "Tabu Search": "Tabu", "Kempe Chain": "Kempe",
    "Simulated Annealing": "SA", "Multi-Neighbourhood SA": "SA",
    "ALNS": "ALNS", "Great Deluge": "GD",
    "ABC": "ABC", "Genetic Algorithm": "GA", "GA": "GA", "LAHC": "LAHC",
    "WOA": "WOA", "HHO": "HHO+", "HHO+": "HHO+",
    "CP-SAT": "CP-SAT", "CP-SAT B&B": "CP-SAT", "GVNS": "GVNS",
}

# Canonical display order — keep constructive methods left, metaheuristics
# middle, exact methods right. Anything outside this list sorts to the end.
ALGO_ORDER = [
    "Greedy", "Feasibility", "Seeder",
    "Tabu Search", "Kempe Chain",
    "Simulated Annealing", "Multi-Neighbourhood SA",
    "ALNS", "Great Deluge",
    "ABC", "Genetic Algorithm", "GA", "LAHC",
    "WOA", "HHO", "HHO+",
    "CP-SAT", "CP-SAT B&B", "GVNS",
]

# ── Search-paradigm taxonomy ────────────────────────────────────────────
# Four families for the paper's faceted figures. The mapping is canonical:
# if a new algorithm ships, add it here (and in ALGO_ORDER) so every
# family-aware plot picks it up automatically.
ALGO_FAMILY = {
    "Greedy":                  "Construction",
    "Feasibility":             "Construction",
    "Seeder":                  "Construction",
    "Tabu Search":             "Trajectory",
    "Kempe Chain":             "Trajectory",
    "Simulated Annealing":     "Trajectory",
    "Multi-Neighbourhood SA":  "Trajectory",
    "LAHC":                    "Trajectory",
    "Great Deluge":            "Trajectory",
    "GVNS":                    "Trajectory",
    "ALNS":                    "Trajectory",
    "Genetic Algorithm":       "Population",
    "GA":                      "Population",
    "ABC":                     "Population",
    "WOA":                     "Population",
    "HHO":                     "Population",
    "HHO+":                    "Population",
    "CP-SAT":                  "Exact / Hybrid",
    "CP-SAT B&B":              "Exact / Hybrid",
}

FAMILY_ORDER = ["Construction", "Trajectory", "Population", "Exact / Hybrid"]

# Faint family-tint backgrounds for facet titles — subtle; readers see it as
# a visual cue, not a color story competing with the algorithm palette.
FAMILY_COLORS = {
    "Construction":    "#4E79A7",
    "Trajectory":      "#F28E2B",
    "Population":      "#E15759",
    "Exact / Hybrid":  "#8E24AA",
}

# ── Soft-constraint breakdown axis metadata ──────────────────────────────
SOFT_KEYS = [
    "two_in_a_row", "two_in_a_day", "period_spread",
    "non_mixed_durations", "front_load", "period_penalty", "room_penalty",
]
SOFT_LABELS = [
    "2-in-a-Row", "2-in-a-Day", "Period Spread",
    "Mixed Dur.", "Front Load", "Period Pen.", "Room Pen.",
]
SOFT_COLORS = [
    "#E53935", "#FB8C00", "#FDD835", "#43A047",
    "#1E88E5", "#8E24AA", "#6D4C41",
]

METRIC_LABELS = {
    "soft_penalty":    "Soft Penalty",
    "runtime":         "Runtime (s)",
    "hard_violations": "Hard Violations",
    "memory_peak_mb":  "Peak Memory (MB)",
}


# ── Public accessors ─────────────────────────────────────────────────────

def algo_color(algo: str) -> str:
    """Canonical color for *algo* with a neutral gray fallback."""
    return ALGO_COLORS.get(algo, "#888888")


def algo_marker(algo: str) -> str:
    """Canonical marker glyph for *algo* with a circle fallback."""
    return ALGO_MARKERS.get(algo, "o")


def algo_short(algo: str) -> str:
    """Short display label for *algo* (falls back to *algo* unchanged)."""
    return ALGO_SHORT.get(algo, algo)


def algo_order(algos):
    """Sort *algos* into canonical display order — unknowns to the end."""
    idx = {a: i for i, a in enumerate(ALGO_ORDER)}
    return sorted(algos, key=lambda a: idx.get(a, 99))


def algo_family(algo: str) -> str:
    """Search-paradigm family for *algo*. Unknown names get ``"Other"``."""
    return ALGO_FAMILY.get(algo, "Other")


def group_by_family(algos):
    """Return ``[(family, [algos])]`` in canonical family order.

    Empty families are skipped so a two-family subset (say Construction +
    Exact) renders 1x2 instead of a 2x2 with blank panels.
    """
    buckets = {}
    for a in algos:
        buckets.setdefault(algo_family(a), []).append(a)
    ordered = [(f, buckets[f]) for f in FAMILY_ORDER if f in buckets]
    if "Other" in buckets:
        ordered.append(("Other", buckets["Other"]))
    return ordered


def apply_paper_style() -> None:
    """Apply the paper-wide matplotlib rcParams.

    Safe to call more than once (idempotent). A no-op if matplotlib is missing,
    so Colab bootstrap cells that import the package before installing matplotlib
    don't blow up.
    """
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "figure.dpi":        130,
        "font.size":         11,
        "font.family":       "sans-serif",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "axes.grid":         True,
        "grid.alpha":        0.22,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.fontsize":   9,
        "legend.framealpha": 0.85,
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
    })


# ── Legacy-compat helpers (used inside utils/plotting.py) ────────────────
# Kept under the old underscore names so the split is a no-op for callers.

def _c(algo):
    return algo_color(algo)


def _m(algo):
    return algo_marker(algo)


def _short(algo):
    return algo_short(algo)


def _order(algos):
    return algo_order(algos)


def _style():
    apply_paper_style()


def _kfmt(ax, axis="y"):
    """Thousands/millions formatter — only fires for values >= 1000."""
    if not HAS_MPL:
        return
    def _fmt(x, _):
        if abs(x) >= 1e6: return f"{x/1e6:,.1f}M"
        if abs(x) >= 1e3: return f"{x/1e3:,.1f}k"
        if abs(x) >= 10:  return f"{x:,.0f}"
        return f"{x:.2f}"
    fmt = ticker.FuncFormatter(_fmt)
    if axis in ("y", "both"): ax.yaxis.set_major_formatter(fmt)
    if axis in ("x", "both"): ax.xaxis.set_major_formatter(fmt)


def _apply_xlabels(ax, names, *, short=True):
    """Set x-tick labels with auto-rotation when 5+ labels would collide."""
    labels = [algo_short(n) for n in names] if short else list(names)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    if len(labels) >= 5:
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")


def _save(fig, path):
    """Persist *fig* to *path* with consistent DPI/bounding. No-op if path is falsy."""
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
