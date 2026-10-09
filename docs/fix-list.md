# Cleanup checklist

- [x] CI and the Colab link point at `main`.
- [x] Add `.gitignore` for environments, caches, build output, and editor files.
- [x] Stop tracking the Verilator `obj_dir/` build output and ignore it.
- [x] Remove GPU/CUDA Phase 3 code, build flags, notebook, script, results, and docs.
- [x] Remove executable bits from the paper and slides.
- [x] Open the README with a project, run, test, and status summary.
- [x] Rewrite the owner's old commits to use the GitHub noreply address, so the old school account no longer shows as a contributor. The teammate's two commits keep their original author email.

- [ ] Old commits still contain compiled `.pyc` caches and a Windows `.exe`; removing them needs a history rewrite and force-push.
- [ ] `results/` holds about 11.5k tracked files (~159 MB on disk), mostly `batch_018_colab/`; consider a compressed release asset.
- [ ] Update the GitHub repository description, which still mentions CUDA.
- [ ] Check whether the paper and slides describe CUDA variants and decide whether to leave the published versions as they are.
- [ ] Decide whether the HDL/FPGA cycle-sim (`cpp/src/hdl/`, `docs/FPGA_DESIGN.md`) stays in scope.
- [ ] Review `docs/PERF_ROADMAP.md` and `docs/POST_COMPACT_ROADMAP.md`; these planning notes may be stale.
- [ ] The protected `results/batch_019_colab/INDEX.md` still refers to the removed measurement batch; update it only if changes to paper data are allowed.
- [ ] Check the README's OR-Tools description of the C++ CP-SAT solver against its pure C++ branch-and-bound implementation.
