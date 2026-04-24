#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
#  batch 19 — validate Phase-2 cached / Thompson / SIMD algorithms
#  on the full ITC 2007 suite.
#
#  Usage:  ./scripts/run_batch19.sh <out_dir> "<seeds>" "<algos>" "<sets>"
#  Typical:
#     make batch19                      (uses default 3-seed spec)
#     ./scripts/run_batch19.sh results/batch_019_validation \
#                              "42 43 44" \
#                              "tabu_cached sa_cached gd_cached lahc_cached alns_thompson" \
#                              "exam_comp_set1 exam_comp_set2 exam_comp_set3 \
#                               exam_comp_set4 exam_comp_set5 exam_comp_set6 \
#                               exam_comp_set7 exam_comp_set8"
#
#  Design notes:
#   - Runs locally on a fresh clone (just needs `make` to have been run).
#   - Same script is used from the Colab notebook (notebooks/batch19_colab.ipynb)
#     so local and cloud pipelines produce identical outputs.
#   - Writes CSV summary + per-run JSON under $out_dir.
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

OUT_DIR="${1:-results/batch_019_validation}"
SEEDS="${2:-42 43 44}"
ALGOS="${3:-tabu_cached sa_cached gd_cached lahc_cached alns_thompson}"
SETS="${4:-exam_comp_set1 exam_comp_set2 exam_comp_set3 exam_comp_set4 \
          exam_comp_set5 exam_comp_set6 exam_comp_set7 exam_comp_set8}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT_DIR}/cpp/build/exam_solver"
INSTANCE_DIR="${ROOT_DIR}/instances"

if [[ ! -x "${BIN}" ]]; then
  echo "error: ${BIN} not found. Run 'make' first." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/summary.csv"
echo "algo,instance,seed,feasible,hard,soft,runtime_sec,iterations" > "${CSV}"

total=0
for s in ${SETS}; do
  for a in ${ALGOS}; do
    for seed in ${SEEDS}; do
      total=$((total + 1))
    done
  done
done

i=0
for s in ${SETS}; do
  inst="${INSTANCE_DIR}/${s}.exam"
  if [[ ! -f "${inst}" ]]; then
    echo "warn: ${inst} missing, skipping" >&2
    continue
  fi
  for a in ${ALGOS}; do
    for seed in ${SEEDS}; do
      i=$((i + 1))
      printf "[%d/%d] %-18s %-18s seed=%s  " "${i}" "${total}" "${a}" "${s}" "${seed}"
      run_dir="${OUT_DIR}/${a}_${s}_seed${seed}"
      mkdir -p "${run_dir}"

      # Algo-specific iteration budgets (paper-grade, not smoke)
      case "${a}" in
        tabu|tabu_simd|tabu_cached) iter_flags="--tabu-iters 3000 --tabu-patience 800" ;;
        sa|sa_cached)               iter_flags="--sa-iters 80000" ;;
        gd|gd_cached)               iter_flags="--gd-iters 80000" ;;
        lahc|lahc_cached)           iter_flags="--lahc-iters 80000" ;;
        alns|alns_cached|alns_thompson) iter_flags="--alns-iters 3000" ;;
        vns|vns_cached)             iter_flags="--vns-iters 5000" ;;
        kempe)                      iter_flags="--kempe-iters 5000" ;;
        *)                          iter_flags="" ;;
      esac

      # Run solver, capture JSON output
      t0=$(date +%s.%N)
      "${BIN}" "${inst}" --algo "${a}" --seed "${seed}" ${iter_flags} \
        --output-dir "${run_dir}" \
        > "${run_dir}/result.json" 2> "${run_dir}/stderr.log" || true
      t1=$(date +%s.%N)
      wall=$(echo "${t1} - ${t0}" | bc -l)

      # Parse JSON (expect fields: feasible, hard, soft, runtime_sec, iterations)
      if [[ -s "${run_dir}/result.json" ]]; then
        feas=$(grep -oP '"feasible":\s*\K(true|false)' "${run_dir}/result.json" | head -1 || echo "?")
        hard=$(grep -oP '"hard_violations":\s*\K[0-9]+' "${run_dir}/result.json" | head -1 || echo "?")
        soft=$(grep -oP '"soft_penalty":\s*\K[0-9]+' "${run_dir}/result.json" | head -1 || echo "?")
        rt=$(grep -oP '"runtime_sec":\s*\K[0-9.]+' "${run_dir}/result.json" | head -1 || echo "${wall}")
        its=$(grep -oP '"iterations":\s*\K[0-9]+' "${run_dir}/result.json" | head -1 || echo "0")
      else
        feas="?" ; hard="?" ; soft="?" ; rt="${wall}" ; its="0"
      fi
      echo "${a},${s},${seed},${feas},${hard},${soft},${rt},${its}" >> "${CSV}"
      printf "hard=%s soft=%s  (%.1fs)\n" "${hard}" "${soft}" "${wall}"
    done
  done
done

echo ""
echo "Done. Summary: ${CSV}"
echo "Run 'python3 scripts/summarize_batch19.py ${OUT_DIR}' to see best-per-instance."
