# Colab Runbook — `notebooks/colab_runner.ipynb`

Short, opinionated guide to running the full paper batch on Google Colab without
breaking your local checkout. Follow it top-to-bottom the first time; afterwards
you only need step 3–6.

---

## TL;DR

1. Open https://colab.research.google.com/github/Dialovos/exam-scheduling-algos-analyze/blob/master/notebooks/colab_runner.ipynb
2. Runtime → **Change runtime type** → pick one of:
   - **A100** (recommended): ~12 vCPUs, 83 GB RAM — roughly 10× faster than free tier.
   - **T4 High-RAM**: ~4 vCPUs, 25 GB — ~4× faster than free tier.
   - **CPU** (free): 2 vCPUs, 12 GB — works, just slow.
   Don't pay for GPU thinking it'll help the solver — it won't. You're paying for the CPU/RAM profile that comes with the GPU tier.
3. (Optional but recommended) Run the **Drive-mount** cell near the top. It auto-copies every zip (`batch_colab.zip`, `batch_colab_ip.zip`, `batch_colab_scaling_sweep.zip`, `batch_colab_chain.zip`) into `/content/drive/MyDrive/exam_scheduling_batches/` so you can't lose them to a runtime disconnect.
4. Set `NUM_SEEDS` in the batch cell to taste (default `3`). Heuristics run in parallel across cores.
5. Runtime → **Run all**. The heuristic batch streams ETA as it runs.
6. When the main zip (`batch_colab.zip`) finishes downloading, the **post-export IP cell** kicks in automatically and runs CP-SAT with a Tabu warm-start on every instance ≤ 900 exams.
7. Second zip (`batch_colab_ip.zip`) downloads, then the **scaling ladder** (100 → 1000 exams, every algo × 2 seeds) and the **parameter sweep** (1-D sensitivity per algo on the 1000-exam instance) run back-to-back.
8. Third zip (`batch_colab_scaling_sweep.zip`) downloads, then the **chain-discovery tune** (cell #26) fires — screens every algo, evolves 3-step chains via successive halving, deep-tunes the survivors, rescored against tuned params. Winner lands in `tuned_params.json` as `best_chain`. Top-5 log gets written by cell #28.
9. Fourth zip (`batch_colab_chain.zip`) downloads at the end. Unzip all four locally and re-render figures with `make reproduce`.

---

## Zero-impact rules (keeps your laptop clean)

- **Colab runs on a remote VM.** Nothing it does touches your local clone.
  It `git clone`s a fresh copy into the Colab sandbox at `/content/exam-scheduling/`.
- **Never `git push` from Colab.** The clone is `--depth 1`, there are no
  authentication credentials, and you don't want Colab-generated commits in
  your history anyway.
- **Colab VMs are ephemeral.** They evaporate when idle for ~20 min or when
  you close the tab. Always run the zip/download cell **before** walking away.
- **The batch writes only to `results/colab_batch/`.** If you want a named
  batch dir, rename it after unzipping locally (e.g. `batch_018_colab`).

---

## Step-by-step

### 1 — Open the notebook

Easiest: use the "Open in Colab" URL above. That link reads the notebook
straight from GitHub, so you always get the latest version.

If you've **forked** the repo, edit cell #4 before running:

```python
GH_USER = 'YourFork'   # ← change this
REPO    = 'exam-scheduling'
```

### 2 — Pick runtime

The batch cell now runs heuristics in parallel via `ProcessPoolExecutor`, so the wall-clock drops linearly with vCPU count. Rough ETAs for 13 algos × 8 sets × 3 seeds = 312 runs:

| Runtime         | vCPUs | Free? | Main batch | IP sweep   | Scaling + param sweep | Chain tune  |
|-----------------|:-----:|:-----:|:----------:|:----------:|:---------------------:|:-----------:|
| Free CPU        | 2     | yes   | 60–90 min  | 25–40 min  | 120–200 min           | 300–400 min |
| CPU High-RAM    | 8     | Pro   | 20–35 min  | 20–30 min  | 90–150 min            | 150–180 min |
| T4 High-RAM     | 4     | yes   | 20–35 min  | 20–30 min  | 90–150 min            | 220–280 min |
| A100            | 12    | Pro   | 8–15 min   | 15–25 min  | 40–70 min             |  90–120 min |

Notes:
- The solver is **CPU-only** — the GPU sits idle. A100/T4 is worth picking anyway, because those tiers ship with way more vCPUs and RAM than the free tier. That's what speeds things up.
- Peak memory per heuristic run is well under 1 GB, so "High-RAM" is never a bottleneck on its own.
- IP (post-export) runs **sequentially per instance** because CP-SAT parallelises internally (`--ip-workers 0` = all cores). Stacking outer parallelism would thrash.

### 3 — Trim scope if you want a quick smoke

In the batch cell (`## 4. Full batch`), any of these is fine:

```python
ALGOS     = ['tabu', 'sa']        # just 2 algos
DATASETS  = DATASETS[:2]          # first 2 ITC sets only
NUM_SEEDS = 1                     # single seed
```

That drops the run to 2–5 min on any runtime.

### 3b — Mount Drive (optional but smart)

The **Drive-mount cell** at the top of the notebook is the recovery net. If you run it:

- You get an auth prompt once, grant "All files" access, and you're done.
- Both zips (`batch_colab.zip` and `batch_colab_ip.zip`) get copied into `/content/drive/MyDrive/exam_scheduling_batches/` as soon as they're built.
- If the Colab runtime disconnects mid-run, the zip files that already finished are safely in Drive.

If you *don't* run the mount cell, `files.download(...)` still works — it just sends to your computer's Downloads folder, not Drive.

### 4 — Run all

`Runtime → Run all` (or `⌘/Ctrl + F9`). Keep the tab visible — Colab
throttles background tabs and will disconnect eventually.

The loop prints an ETA every run. If you see repeated failures in one
algo, stop the run (`Runtime → Interrupt execution`), inspect
`results/colab_batch/failures.log`, and decide whether to continue.

### 5 — Download results (main batch)

The zip/download cell zips `results/colab_batch/` and triggers a browser download (`batch_colab.zip`). If Drive is mounted it also saves a copy there. If the download doesn't start in your browser, manually run:

```python
from google.colab import files
files.download('batch_colab.zip')
```

**Without Drive mounted:** if you miss the download and the runtime times out, everything is gone. Download right away.
**With Drive mounted:** the zip is safe in Drive even if the runtime dies — you can re-download it from `/content/drive/MyDrive/exam_scheduling_batches/` anytime.

### 5b — Post-export IP (automatic)

The IP cell runs **after** the main batch is zipped and downloaded. It:

1. Picks a Tabu `.sln` from each `results/colab_batch/tabu_<set>_seed*/` as a CP-SAT warm-start hint.
2. Invokes `main.py --algo ip --ip-warmstart ... --ip-workers 0 --ip-time 300` per instance ≤ 900 exams.
3. Writes to `results/colab_batch_ip/ip_<set>/`.
4. Zips as `batch_colab_ip.zip` and triggers a second browser download + Drive copy.

If you want to skip IP entirely, just stop after cell #6 — the main batch is already exported.

### 5c — Scaling + parameter sweep (automatic)

After the IP zip, two more cells fire automatically:

1. **Scaling batch (cell #10)** — generates `instances/synthetic_scaling_{50..1000}.exam` in step-50 increments (seed 42, idempotent) and runs every algorithm across the 20-step ladder × 3 seeds in parallel → 780 runs. Writes `results/colab_batch_scaling/aggregated.csv` keyed by `num_exams` — ready for `plot_scaling`. Edit `SCALING_STEP = 25` at the cell top for an even denser 40-size ladder.
2. **Parameter sweep (cell #12)** — invokes `python -m tooling.param_sweep` on the 1000-exam instance. For every algorithm listed in `tooling/tuner/search_spaces.py` it varies one knob at a time (5 values, log-spaced) with the others pinned at tuned defaults, 3 seeds each → ~270 runs. Writes `results/colab_batch_sweep/sensitivity.csv` — ready for `plot_parameter_sensitivity`.

Both outputs get zipped together as `batch_colab_scaling_sweep.zip` and downloaded (+ Drive copy). Cost on A100: ~40–70 min for the pair; on CPU High-RAM (50 GB / 8 vCPU): ~90–150 min.

Skip either by clearing its cell before **Run all**. If sweep aborts mid-run, the partial CSV is still flushed row-by-row.

### 5d — Chain discovery + top-5 (automatic)

After the scaling/sweep zip, the last three cells run the heaviest phase:

1. **Chain-discovery tune (cell #26)** — `python main.py --mode tune` across all ITC sets with `chain-pop 14 × chain-rounds 5 × param-trials 5 × eval-datasets 3 × max-time 9600s`. Phases run in order: screen → chain discovery (successive halving) → core-algo extraction → deep param tuning → top-5 chain rescore. Checkpoint is flushed every few evals to `results/colab_batch_chain/tuning/checkpoint.json`; re-running the cell after a disconnect auto-resumes. Best chain lands in `tuned_params.json` as `best_chain` + `best_chain_score`.
2. **Top-5 extraction (cell #28)** — reads the tuner checkpoint's `chain_history`, dedupes by algorithm signature (e.g. `tabu → sa → gd`), keeps best score per structure, writes `top5_chains.json` + `top5_chains.txt`. The txt file is what you want to paste into the paper.
3. **Zip cell (cell #30)** — bundles `results/colab_batch_chain/` + `tuned_params.json` into `batch_colab_chain.zip`, Drive-backed if mounted.

Knobs in cell #26 (`CHAIN_POP`, `CHAIN_ROUNDS`, `PARAM_TRIALS`, `EVAL_DATASETS`, `MAX_TIME`) give you direct control over budget. If you only care about the chain finder (skip deep param tuning), drop `PARAM_TRIALS = 0` and shrink `MAX_TIME` accordingly.

**Chain-finder v2 (2026-04):** newer defaults kick in automatically — max chain length **10** (up from 5), **adaptive successive-halving eta** (aggressive rung-0 pruning at larger pops), **1-point crossover** between survivors (25 %), **partial-credit scoring** for chains whose last step times out (2.5 % penalty), **step-level early-stop** on hopeless partials (>2.5× dataset baseline), **no adjacent algo duplicates**, and an on-disk **prefix `.sln` cache** that warm-starts chains sharing a prefix with a previously-evaluated chain. All are on by default; rollback via `--no-chain-prefix-cache`, `--no-chain-partial-credit`, `--no-chain-early-stop`, `--chain-allow-duplicates`, or `--chain-max-len 5`.

Cost on A100 (12 vCPU): ~90–120 min; CPU High-RAM (8 vCPU): ~150–180 min; free CPU (2 vCPU): 5–7 h — consider dropping `EVAL_DATASETS=2` there.

### 6 — Bring it back to local

```bash
cd ~/Developer/cli/claude/personal_proj/exam-scheduling
mkdir -p results/batch_018_colab
unzip ~/Downloads/batch_colab.zip -d results/batch_018_colab/
# The zip contains results/colab_batch/* — flatten it:
mv results/batch_018_colab/results/colab_batch/* results/batch_018_colab/
rmdir results/batch_018_colab/results/colab_batch results/batch_018_colab/results

# If you also ran the post-export IP cell, pull the IP zip into the same batch dir:
unzip ~/Downloads/batch_colab_ip.zip -d results/batch_018_colab/
mv results/batch_018_colab/results/colab_batch_ip/* results/batch_018_colab/
rmdir results/batch_018_colab/results/colab_batch_ip results/batch_018_colab/results

# Scaling + parameter sweep (third zip, optional):
unzip ~/Downloads/batch_colab_scaling_sweep.zip -d results/batch_018_colab/
mv results/batch_018_colab/results/colab_batch_scaling results/batch_018_colab/
mv results/batch_018_colab/results/colab_batch_sweep results/batch_018_colab/
rmdir results/batch_018_colab/results

# Chain discovery (fourth zip, optional):
unzip ~/Downloads/batch_colab_chain.zip -d results/batch_018_colab/
mv results/batch_018_colab/results/colab_batch_chain results/batch_018_colab/
rmdir results/batch_018_colab/results
# If tuned_params.json came through, it unzips to repo root — merge or discard.

# Regenerate figures from the Colab batch:
make reproduce REPRO_BATCH=results/batch_018_colab
```

The figures land in `graphs/`. Commit only the figures you actually want to
keep — the batch CSVs are fine to leave untracked unless the paper cites
them directly.

---

## Gotchas

| Symptom                                        | Fix                                                                  |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| "Disk quota exceeded" mid-batch                | Colab free tier has ~70 GB. Shouldn't happen. Check `!df -h`.       |
| One algo always fails (`rc=139`)               | Segfault in the C++ binary. Try `!make clean && make` to rebuild.   |
| CP-SAT takes forever                           | Edit `--cpsat-time 60` default if you lower scope; it's hardcoded.   |
| `generate_all_plots` crashes in cell #10       | Happens if `aggregated.csv` has missing columns. Skip that cell — `make reproduce` does the same job locally and is easier to debug. |
| Runtime disconnected before the main download  | If Drive was mounted, grab the zip from there. Otherwise, re-run from step 4. Mount Drive next time. |
| IP cell skips every instance                   | No Tabu solutions on disk — the main batch didn't include `'tabu'` or failed on every set. Re-run the batch with Tabu. |
| IP cell dies mid-run                           | Main batch is already safe on disk/Drive. Re-running just the IP cell is fine; it overwrites `results/colab_batch_ip/`. |
| Sweep errors on `--tabu-tenure`                | You're on a stale checkout. Pull the latest — `--tabu-tenure` was added to `main.py` alongside the sweep. |
| Scaling cell "No module named 'core.generator'" | `%cd` into the repo wasn't run — re-execute cell #2 (clone + cd). |
| You forked → `git clone` 404s in cell #4      | Make sure the fork is public, or use `https://<token>@github.com/...`.|
| Top-5 cell: `chain_history is empty`          | Chain-discovery phase never ran — `MAX_TIME` got spent on screen/param phases. Lower `PARAM_TRIALS` or raise `MAX_TIME`. |
| Top-5 cell: `No tuner checkpoint`             | Section 12 didn't produce a checkpoint — check the tune log for the reason (missing binary, dataset glob empty, etc). |
| Chain tune seems stuck                        | Each chain eval is 2–3 algo runs × `EVAL_DATASETS × n_seeds` — a full-fidelity rung can take 10+ min on free CPU. Watch for the `Round N/5` progress line. |

---

## What NOT to do

- **Don't** run Colab cells locally. They use `!apt-get install`, `%cd`,
  and `google.colab.files` — none of which exist on your laptop.
- **Don't** edit `notebooks/exam_scheduling.ipynb` in Colab. That's your local
  experiment notebook; the Colab sandbox gets `notebooks/colab_runner.ipynb`.
- **Don't** put secrets (API keys, tokens) into the notebook.
