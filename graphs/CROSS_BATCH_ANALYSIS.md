# Cross-batch analysis

Run `scripts/make_batch_comparison.py` to refresh these tables.

```
pooled 848 rows · 18 algos · 2 batches

Table 1 — Coverage (run counts per algo × batch)
========================================================
algo           family    tier  018 (paper)  019 (cached)
-------------  ------  ------  -----------  ------------
tabu             tabu    base           56             0
tabu_cached      tabu  cached            0            24
sa                 sa    base           56             0
sa_cached          sa  cached            0            24
gd                 gd    base           56             0
gd_cached          gd  cached            0            24
lahc             lahc    base           56             0
lahc_cached      lahc  cached            0            24
alns             alns    base           56             0
alns_thompson    alns  cached            0            24
abc               abc    base           56             0
ga                 ga    base           56             0
hho               hho    base           56             0
woa               woa    base           56             0
kempe           kempe    base           56             0
vns               vns    base           56             0
greedy         greedy    base           56             0
cpsat           cpsat    base           56             0

Table 2 — Global mean normalized soft (lower = better; 1.00 = best-in-pool)
===========================================================================
algo           family    tier  018 (paper)  019 (cached)   best  n_batches
-------------  ------  ------  -----------  ------------  -----  ---------
tabu_cached      tabu  cached            –          1.17   1.17          1
vns               vns    base         1.19             –   1.19          1
lahc_cached      lahc  cached            –          1.22   1.22          1
tabu             tabu    base         1.24             –   1.24          1
sa_cached          sa  cached            –          1.32   1.32          1
abc               abc    base         1.33             –   1.33          1
kempe           kempe    base         1.41             –   1.41          1
cpsat           cpsat    base         1.56             –   1.56          1
woa               woa    base         1.62             –   1.62          1
lahc             lahc    base         1.77             –   1.77          1
alns_thompson    alns  cached            –          2.33   2.33          1
ga                 ga    base         2.41             –   2.41          1
sa                 sa    base         2.64             –   2.64          1
alns             alns    base         3.51             –   3.51          1
hho               hho    base        23.14             –  23.14          1
gd_cached          gd  cached            –         24.83  24.83          1
gd                 gd    base        26.66             –  26.66          1
greedy         greedy    base        36.64             –  36.64          1

Table 3 — Tier progression by family (mean norm. soft; Δ < 0 = improvement)
===========================================================================
family   base  cached  base→cached Δ
------  -----  ------  -------------
tabu     1.24    1.17          -0.07
sa       2.64    1.32          -1.33
gd      26.66   24.83          -1.83
lahc     1.77    1.22          -0.55
alns     3.51    2.33          -1.18
abc      1.33       –              –
ga       2.41       –              –
hho     23.14       –              –
woa      1.62       –              –
kempe    1.41       –              –
vns      1.19       –              –
greedy  36.64       –              –
cpsat    1.56       –              –

Table 4 — Per-instance winner (minimum raw soft across every run)
=================================================================
instance               algo         batch    soft  seed
--------------  -----------  ------------  ------  ----
exam_comp_set1         tabu   018 (paper)   8,365    48
exam_comp_set2        kempe   018 (paper)   1,976    47
exam_comp_set3  lahc_cached  019 (cached)  19,084    43
exam_comp_set4          abc   018 (paper)  19,001    46
exam_comp_set5  lahc_cached  019 (cached)   4,509    43
exam_comp_set6        kempe   018 (paper)  29,130    47
exam_comp_set7  tabu_cached  019 (cached)   7,757    44
exam_comp_set8  tabu_cached  019 (cached)  12,554    42

Table 5 — No same-batch variant pairs available.

Table 6 — Mean runtime (s) per algo × batch (batch_018 not recorded)
====================================================================
algo           018 (paper)  019 (cached)
-------------  -----------  ------------
tabu_cached              –          4.64
sa_cached                –          7.81
gd_cached                –         18.57
lahc_cached              –         10.30
alns_thompson            –         18.76

Table 7 — PERF_ROADMAP smoke-test reference (docs/PERF_ROADMAP.md)
==================================================================
Scope: 5000-iter single-seed smoke runs used during Phase-2 development.
These are NOT pooled with the batches above — shown for cross-reference.

algo             instance   scalar ms   cached ms   cached/scalar   soft Δ
-------------    --------   ---------   ---------   -------------   ---------------
tabu             set4            9451         787           12.0×   identical (36587)
tabu             set7            6261        1720           3.64×   identical (8653)
sa               set4             (*)         (*)           1.49×   −857 (better)
sa               set7             (*)         (*)           1.05×   −3352 (better)
gd               set4             (*)         (*)           0.82×   identical
gd               set7             (*)         (*)           1.57×   −98k (56% better)
lahc             set4             (*)         (*)           1.05×   −1826 (better)
lahc             set7             (*)         (*)           1.19×   +5% worse
alns             set4             (*)         (*)           0.75×   slight better
alns             set7             (*)         (*)           0.65×   identical
alns_thompson    set7 10k        133.3s       52.4s         2.54×   −107 (1% better)

```
