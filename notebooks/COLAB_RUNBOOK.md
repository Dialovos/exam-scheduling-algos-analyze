# Colab Runbook — `colab_runner.ipynb`

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
3. (Optional but recommended) Run the **Drive-mount** cell near the top. It auto-copies both result zips into `/content/drive/MyDrive/exam_scheduling_batches/` so you can't lose them to a runtime disconnect.
4. Set `NUM_SEEDS` in the batch cell to taste (default `3`). Heuristics run in parallel across cores.
5. Runtime → **Run all**. The heuristic batch streams ETA as it runs.
6. When the main zip (`batch_colab.zip`) finishes downloading, the **post-export IP cell** kicks in automatically and runs CP-SAT with a Tabu warm-start on every instance ≤ 900 exams.
7. Second zip (`batch_colab_ip.zip`) downloads at the end. Unzip both locally and re-render figures with `make reproduce`.

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

| Runtime     | vCPUs | Free? | ETA (main batch) | ETA (IP sweep) |
|-------------|:-----:|:-----:|:----------------:|:--------------:|
| Free CPU    | 2     | yes   | 60–90 min        | 25–40 min      |
| T4 High-RAM | 4     | yes   | 20–35 min        | 20–30 min      |
| A100        | 12    | Pro   | 8–15 min         | 15–25 min      |

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
| You forked → `git clone` 404s in cell #4      | Make sure the fork is public, or use `https://<token>@github.com/...`.|

---

## What NOT to do

- **Don't** run Colab cells locally. They use `!apt-get install`, `%cd`,
  and `google.colab.files` — none of which exist on your laptop.
- **Don't** edit `exam_scheduling.ipynb` in Colab. That's your local
  experiment notebook; the Colab sandbox gets `colab_runner.ipynb`.
- **Don't** put secrets (API keys, tokens) into the notebook. Colab
  notebooks are shareable by default and the repo is public.
