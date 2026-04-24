/*
 * CudaEvaluator — host-side wrapper for batched full move_delta scoring.
 *
 * This file owns:
 *   • Persistent flat arrays mirroring FastEvaluator/CachedEvaluator state
 *     (laid out for cudaMemcpy'ing straight to device).
 *   • `score_delta_cpu_ref(...)` — a bit-exact CPU twin of what the CUDA
 *     kernel computes. Used on no-GPU builds as the CPU fallback AND as
 *     the validator for the kernel (see `bench_eval.cpp`).
 *   • `score_batch(...)` — dispatches to GPU kernel (HAVE_CUDA) or the
 *     CPU twin.
 *
 * The CPU twin covers every term in CachedEvaluator::move_delta:
 *   • hard: adj-conflicts (cached contrib diffs recomputed here), duration,
 *           room-capacity overflow, PHC (EXAM_COINCIDENCE/EXCLUSION/AFTER),
 *           RHC (at-most-one-other-room)
 *   • soft: period_spread, two_in_row, two_in_day, period_pen, room_pen,
 *           front_load (large exams in last periods)
 *
 * State-sync contract:
 *   • Build once via ctor (static tables: adj, period_day, etc.).
 *   • Call `sync_state(sol)` before each batch — pushes period_of,
 *     room_of, pr_enroll, pr_count.
 *   • On GPU, upload happens once per call (cheap — O(ne + np*nr)).
 *   • Kernel derives old_pid/old_rid from synced period_of/room_of
 *     using mv_eid, so score_batch only takes (mv_eid, mv_new_pid,
 *     mv_new_rid).
 */

#pragma once

#include "../models.h"
#include "../evaluator.h"
#include "../evaluator_cached.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_CUDA
// Forward-declared C hooks into delta_kernel.cu — defined only when compiled
// with nvcc and linked against libdelta_cuda.so.
extern "C" {
    struct CudaStaticParams {
        int ne, np, nr;
        int adj_stride, phc_total;
        int w_2row, w_2day, w_spread;
        int w_mixed;
        int fl_penalty;
        int n_rhc;
        int n_students;
        int max_student_degree;  // sizing bound for per-student local scratch
    };

    void* cuda_state_create(
        const CudaStaticParams* params,
        const int32_t* h_adj_other, const int32_t* h_adj_cnt, const int32_t* h_adj_len,
        const int32_t* h_exam_dur, const int32_t* h_exam_enroll,
        const int32_t* h_period_dur, const int32_t* h_period_day,
        const int32_t* h_period_daypos, const int32_t* h_period_pen,
        const int32_t* h_room_cap, const int32_t* h_room_pen,
        const uint8_t* h_is_large, const uint8_t* h_is_last_period,
        const uint8_t* h_is_rhc_exam, const int32_t* h_rhc_exam_ids,
        const int32_t* h_phc_starts, const int32_t* h_phc_others, const int32_t* h_phc_tcodes,
        const int32_t* h_student_starts, const int32_t* h_student_flat);

    void cuda_state_sync_dynamic(
        void* state,
        const int32_t* h_period_of, const int32_t* h_room_of,
        const int32_t* h_pr_enroll, const int32_t* h_pr_count);

    void cuda_state_score_batch(
        void* state,
        const int32_t* h_mv_eid,
        const int32_t* h_mv_new_pid,
        const int32_t* h_mv_new_rid,
        int n_moves,
        int64_t* h_out_deltas_fixed);   // dh*100000 + ds in int64 fixed-point

    void cuda_state_score_placement_batch(
        void* state,
        const int32_t* h_mv_eid,
        const int32_t* h_mv_new_pid,
        const int32_t* h_mv_new_rid,
        int n_moves,
        int64_t* h_out_costs);

    // Batch-score N full solutions. Input SoA arrays:
    //   pop_period_of: [N × ne]
    //   pop_room_of:   [N × ne]
    //   pop_pr_enroll: [N × np × nr]
    //   pop_pr_count:  [N × np × nr]
    // Output: [N] int64 fixed-point (hard*100000 + soft).
    void cuda_state_score_full_batch(
        void* state,
        const int32_t* h_pop_period_of,
        const int32_t* h_pop_room_of,
        const int32_t* h_pop_pr_enroll,
        const int32_t* h_pop_pr_count,
        int N,
        int64_t* h_out_fitness);

    // Parallel SA portfolio — N_seeds × n_iters in one launch.
    void cuda_parallel_sa_run(
        void* state,
        int n_seeds, int n_iters,
        double init_temp, double cooling,
        const int32_t* h_pop_po, const int32_t* h_pop_ro,
        const int32_t* h_pop_pe, const int32_t* h_pop_pc,
        const int64_t* h_current_fitness,
        const uint64_t* h_rng_seeds,
        int32_t* h_best_po, int32_t* h_best_ro,
        int64_t* h_best_fitness);

    void cuda_state_destroy(void* state);
    int cuda_runtime_available();
}
#endif

class CudaEvaluator {
public:
    const ProblemInstance& P;
    const CachedEvaluator& Ecach;
    int ne, np, nr;
    bool gpu_active;

    // Static tables (uploaded once)
    int adj_stride;
    std::vector<int32_t> adj_other, adj_cnt, adj_len;
    std::vector<int32_t> exam_dur, exam_enroll;
    std::vector<int32_t> period_dur, period_day, period_daypos, period_pen;
    std::vector<int32_t> room_cap, room_pen;
    std::vector<uint8_t> is_large, is_last_period, is_rhc_exam;
    std::vector<int32_t> rhc_exam_ids;
    std::vector<int32_t> phc_starts, phc_others, phc_tcodes;
    // Per-student CSR for exact fe.full_eval semantics (counts n-1 duplicates
    // per cluster, not C(n,2) pairs). Matches ITC 2007 conflict definition.
    std::vector<int32_t> student_starts;  // [n_students + 1]
    std::vector<int32_t> student_flat;    // total enrollments
    int n_students;
    int max_student_degree;  // longest single student's exam list (kernel scratch sizing)
    int w_2row, w_2day, w_spread, fl_penalty;

    // Dynamic mirrors (synced per score_batch)
    std::vector<int32_t> period_of_mirror, room_of_mirror;
    std::vector<int32_t> pr_enroll_mirror, pr_count_mirror;

    // Non-owning pointer to the Solution that was last sync'd — lets the
    // CPU fallback read pr_enroll/pr_count directly from sol (O(1) per
    // lookup) instead of copying the full np×nr tables. GPU path still
    // needs the mirrors for cudaMemcpy.
    const Solution* sol_last_sync = nullptr;

    // Lazy GPU-upload flag: set by sync_state, cleared by lazy_sync_gpu_if_needed
    // after upload. Avoids per-iter PCIe traffic when threshold routes batches
    // to CPU.
    mutable bool gpu_dirty = false;

    void* d_state = nullptr;

    explicit CudaEvaluator(const CachedEvaluator& e)
        : P(e.P), Ecach(e), ne(e.ne), np(e.np), nr(e.nr),
          gpu_active(false)
    {
        // Static adjacency (dense stride)
        int max_deg = 0;
        for (int i = 0; i < ne; i++)
            max_deg = std::max(max_deg, (int)P.adj[i].size());
        adj_stride = std::max(1, max_deg);
        adj_other.assign((size_t)ne * adj_stride, 0);
        adj_cnt  .assign((size_t)ne * adj_stride, 0);
        adj_len  .assign(ne, 0);
        for (int i = 0; i < ne; i++) {
            adj_len[i] = (int)P.adj[i].size();
            for (int k = 0; k < adj_len[i]; k++) {
                adj_other[(size_t)i * adj_stride + k] = P.adj[i][k].first;
                adj_cnt  [(size_t)i * adj_stride + k] = P.adj[i][k].second;
            }
        }

        // Static exam/period/room tables
        exam_dur.assign(ne, 0); exam_enroll.assign(ne, 0);
        for (int i = 0; i < ne; i++) {
            exam_dur[i]    = Ecach.exam_dur[i];
            exam_enroll[i] = Ecach.exam_enroll[i];
        }
        period_dur.assign(np, 0); period_day.assign(np, 0);
        period_daypos.assign(np, 0); period_pen.assign(np, 0);
        for (int p = 0; p < np; p++) {
            period_dur[p]    = Ecach.period_dur[p];
            period_day[p]    = Ecach.period_day[p];
            period_daypos[p] = Ecach.period_daypos[p];
            period_pen[p]    = Ecach.period_pen[p];
        }
        room_cap.assign(nr, 0); room_pen.assign(nr, 0);
        for (int r = 0; r < nr; r++) {
            room_cap[r] = Ecach.room_cap[r];
            room_pen[r] = Ecach.room_pen[r];
        }

        // Bitset-like uint8 arrays
        is_large.assign(ne, 0);
        for (int id : Ecach.large_exams) if (id >= 0 && id < ne) is_large[id] = 1;
        is_last_period.assign(np, 0);
        for (int id : Ecach.last_periods) if (id >= 0 && id < np) is_last_period[id] = 1;
        is_rhc_exam.assign(ne, 0);
        rhc_exam_ids.clear();
        for (int id : Ecach.rhc_exams) {
            if (id >= 0 && id < ne) { is_rhc_exam[id] = 1; rhc_exam_ids.push_back(id); }
        }

        // Flatten phc_by_exam into CSR (phc_starts, phc_others, phc_tcodes)
        phc_starts.assign(ne + 1, 0);
        for (int i = 0; i < ne; i++)
            phc_starts[i + 1] = phc_starts[i] + (int)Ecach.phc_by_exam[i].size();
        phc_others.assign(phc_starts[ne], 0);
        phc_tcodes.assign(phc_starts[ne], 0);
        for (int i = 0; i < ne; i++) {
            int base = phc_starts[i];
            for (size_t k = 0; k < Ecach.phc_by_exam[i].size(); k++) {
                phc_others[base + (int)k] = Ecach.phc_by_exam[i][k].first;
                phc_tcodes[base + (int)k] = Ecach.phc_by_exam[i][k].second;
            }
        }

        w_2row = Ecach.w_2row;  w_2day = Ecach.w_2day;  w_spread = Ecach.w_spread;
        fl_penalty = Ecach.fl_penalty;

        // Student-exams CSR
        n_students = (int)P.student_exams.size();
        student_starts.assign(n_students + 1, 0);
        max_student_degree = 0;
        for (int s = 0; s < n_students; s++) {
            int sz = (int)P.student_exams[s].size();
            student_starts[s + 1] = student_starts[s] + sz;
            if (sz > max_student_degree) max_student_degree = sz;
        }
        student_flat.assign(student_starts[n_students], 0);
        for (int s = 0; s < n_students; s++) {
            int base = student_starts[s];
            for (int i = 0; i < (int)P.student_exams[s].size(); i++)
                student_flat[base + i] = P.student_exams[s][i];
        }

        // Dynamic mirrors
        period_of_mirror.assign(ne, -1);
        room_of_mirror.assign(ne, -1);
        pr_enroll_mirror.assign((size_t)np * nr, 0);
        pr_count_mirror.assign((size_t)np * nr, 0);

#ifdef HAVE_CUDA
        if (cuda_runtime_available()) {
            CudaStaticParams params{ne, np, nr, adj_stride, (int)phc_others.size(),
                                     w_2row, w_2day, w_spread,
                                     Ecach.E.w_mixed,
                                     fl_penalty,
                                     (int)rhc_exam_ids.size(),
                                     n_students, max_student_degree};
            d_state = cuda_state_create(
                &params,
                adj_other.data(), adj_cnt.data(), adj_len.data(),
                exam_dur.data(), exam_enroll.data(),
                period_dur.data(), period_day.data(), period_daypos.data(), period_pen.data(),
                room_cap.data(), room_pen.data(),
                is_large.data(), is_last_period.data(),
                is_rhc_exam.data(), rhc_exam_ids.data(),
                phc_starts.data(), phc_others.data(), phc_tcodes.data(),
                student_starts.data(), student_flat.data());
            gpu_active = (d_state != nullptr);
        }
#endif
    }

    ~CudaEvaluator() {
#ifdef HAVE_CUDA
        if (d_state) cuda_state_destroy(d_state);
#endif
    }

    CudaEvaluator(const CudaEvaluator&) = delete;
    CudaEvaluator& operator=(const CudaEvaluator&) = delete;

    // Sync dynamic state from Solution. Call once per score_batch.
    //
    // Cheap CPU-side part always runs (period_of/room_of mirrors + sol ptr).
    // GPU upload is DEFERRED: marks gpu_dirty=true; actual upload happens
    // only when score_batch is about to fire the kernel (via
    // lazy_sync_gpu_if_needed). Saves ~4 cudaMemcpy calls per iter when
    // threshold routes the batch to CPU.
    void sync_state(const Solution& sol) {
        sol_last_sync = &sol;
        for (int i = 0; i < ne; i++) {
            period_of_mirror[i] = sol.period_of[i];
            room_of_mirror[i]   = sol.room_of[i];
        }
#ifdef HAVE_CUDA
        gpu_dirty = gpu_active;
#endif
    }

#ifdef HAVE_CUDA
    // Upload device-side dynamic state if dirty. Idempotent within one sync.
    void lazy_sync_gpu_if_needed() const {
        if (!gpu_active || !gpu_dirty || !sol_last_sync) return;
        for (int p = 0; p < np; p++) {
            for (int r = 0; r < nr; r++) {
                const_cast<std::vector<int32_t>&>(pr_enroll_mirror)[(size_t)p * nr + r]
                    = sol_last_sync->get_pr_enroll(p, r);
                const_cast<std::vector<int32_t>&>(pr_count_mirror)[(size_t)p * nr + r]
                    = sol_last_sync->get_pr_count(p, r);
            }
        }
        cuda_state_sync_dynamic(d_state,
                                 period_of_mirror.data(), room_of_mirror.data(),
                                 pr_enroll_mirror.data(), pr_count_mirror.data());
        gpu_dirty = false;
    }
#endif

    // ── CPU twin of the GPU kernel ──
    // Computes the same fitness delta CachedEvaluator::move_delta does,
    // but from flat arrays only — no unordered_set / nested vector access.
    // This is what the CUDA kernel will compute on device; on CPU builds,
    // score_batch falls back to calling this in a loop.
    inline double score_delta_cpu_ref(int eid, int new_pid, int new_rid) const {
        int old_pid = period_of_mirror[eid];
        // Unplaced-exam path isn't covered by this scorer — caller must
        // ensure all candidates are placed (tabu/VNS always are). Returning
        // +inf marks the move as unpickable without misleading callers.
        if (old_pid < 0) return 1e18;
        int old_rid = room_of_mirror[eid];
        if (old_pid == new_pid && old_rid == new_rid) return 0.0;

        double dh = 0.0, ds = 0.0;

        // ── adj-conflict + proximity + spread (the "cached contrib" part) ──
        // We compute it directly from adj rather than reading soft/hard_contrib
        // because the GPU kernel won't have those tables. Result is identical.
        int new_day  = period_day[new_pid];
        int new_dpos = period_daypos[new_pid];
        int old_day  = period_day[old_pid];
        int old_dpos = period_daypos[old_pid];

        int len = adj_len[eid];
        const int32_t* adj_o = adj_other.data() + (size_t)eid * adj_stride;
        const int32_t* adj_c = adj_cnt.data()   + (size_t)eid * adj_stride;
        for (int i = 0; i < len; i++) {
            int other = adj_o[i];
            int cnt   = adj_c[i];
            if (cnt == 0) continue;
            int o_pid = period_of_mirror[other];
            if (o_pid < 0) continue;
            int o_day  = period_day[o_pid];
            int o_dpos = period_daypos[o_pid];

            // new-period contribution (added)
            if (new_pid == o_pid) dh += cnt;
            if (new_day == o_day) {
                int g = std::abs(new_dpos - o_dpos);
                if (g == 1)      ds += (double)w_2row * cnt;
                else if (g > 1)  ds += (double)w_2day * cnt;
            }
            int og_new = std::abs(new_pid - o_pid);
            if (og_new > 0 && og_new <= w_spread) ds += cnt;

            // old-period contribution (subtracted)
            if (old_pid == o_pid) dh -= cnt;
            if (old_day == o_day) {
                int g = std::abs(old_dpos - o_dpos);
                if (g == 1)      ds -= (double)w_2row * cnt;
                else if (g > 1)  ds -= (double)w_2day * cnt;
            }
            int og_old = std::abs(old_pid - o_pid);
            if (og_old > 0 && og_old <= w_spread) ds -= cnt;
        }

        // ── Duration (exam doesn't fit in the period) ──
        int dur = exam_dur[eid];
        if (dur > period_dur[old_pid]) dh -= 1;
        if (dur > period_dur[new_pid]) dh += 1;

        // ── Room-capacity overflow delta ──
        // Read from sol directly on CPU path (skips O(np×nr) mirror copy).
        int enr = exam_enroll[eid];
        int old_total = sol_last_sync->get_pr_enroll(old_pid, old_rid);
        int new_total = sol_last_sync->get_pr_enroll(new_pid, new_rid);
        dh -= ((old_total > room_cap[old_rid]) ? 1 : 0) -
              (((old_total - enr) > room_cap[old_rid]) ? 1 : 0);
        dh += (((new_total + enr) > room_cap[new_rid]) ? 1 : 0) -
              ((new_total > room_cap[new_rid]) ? 1 : 0);

        // ── Period / room penalty ──
        ds += period_pen[new_pid] - period_pen[old_pid];
        ds += room_pen[new_rid] - room_pen[old_rid];

        // ── Front-load ──
        if (is_large[eid] && fl_penalty > 0) {
            int was  = is_last_period[old_pid];
            int will = is_last_period[new_pid];
            if (was && !will)      ds -= fl_penalty;
            else if (!was && will) ds += fl_penalty;
        }

        // ── Period hard constraints (PHC) — CSR iteration ──
        int phc_s = phc_starts[eid];
        int phc_e = phc_starts[eid + 1];
        for (int i = phc_s; i < phc_e; i++) {
            int other = phc_others[i];
            int tcode = phc_tcodes[i];
            int o_pid = period_of_mirror[other];
            if (o_pid < 0) continue;
            if      (tcode == 0) { if (old_pid != o_pid) dh -= 1; if (new_pid != o_pid) dh += 1; }
            else if (tcode == 1) { if (old_pid == o_pid) dh -= 1; if (new_pid == o_pid) dh += 1; }
            else if (tcode == 2) { if (old_pid <= o_pid) dh -= 1; if (new_pid <= o_pid) dh += 1; }
            else if (tcode == 3) { if (old_pid >= o_pid) dh -= 1; if (new_pid >= o_pid) dh += 1; }
        }

        // ── Room hard constraints (RHC) — "exclusive-room" semantics ──
        if (!rhc_exam_ids.empty()) {
            int oc = sol_last_sync->get_pr_count(old_pid, old_rid);
            int nc = sol_last_sync->get_pr_count(new_pid, new_rid);
            if (is_rhc_exam[eid]) {
                if (oc > 1) dh -= 1;
                if (nc > 0) dh += 1;
            }
            for (int re : rhc_exam_ids) {
                if (re == eid || re >= ne) continue;
                int rp = period_of_mirror[re]; if (rp < 0) continue;
                int rr = room_of_mirror[re];
                if (rp == old_pid && rr == old_rid && oc == 2) dh -= 1;
                if (rp == new_pid && rr == new_rid && nc == 1) dh += 1;
            }
        }

        return dh * 100000.0 + ds;
    }

    // ── Placement scorer ──
    // Cost of placing an UNPLACED exam at (new_pid, new_rid). Mirrors
    // repair_greedy in cpp/src/alns.h exactly: adj-conflict (100000 per
    // conflicting neighbor, NOT weighted by shared-students), proximity
    // (spread/2-in-row/2-in-day per neighbor), period_pen, frontload,
    // room capacity-overflow (100000), room_pen. Does NOT include
    // PHC/RHC/duration — repair_greedy doesn't check those either; the
    // evaluator will score the full fitness on the next full_eval.
    //
    // This is the kernel's CPU twin for the placement kernel. Used both
    // as the CPU fallback and as the validator target.
    inline long long score_placement_cpu_ref(int eid, int new_pid, int new_rid) const {
        int new_day  = period_day[new_pid];
        int new_dpos = period_daypos[new_pid];
        long long pcost = 0;

        int len = adj_len[eid];
        const int32_t* adj_o = adj_other.data() + (size_t)eid * adj_stride;
        for (int i = 0; i < len; i++) {
            int other = adj_o[i];
            int o_pid = period_of_mirror[other];
            if (o_pid < 0) continue;

            if (o_pid == new_pid) {
                pcost += 100000;
            } else {
                int gap = std::abs(new_pid - o_pid);
                if (gap > 0 && gap <= w_spread) pcost += 1;
                if (period_day[o_pid] == new_day) {
                    int g = std::abs(period_daypos[o_pid] - new_dpos);
                    if (g == 1)      pcost += w_2row;
                    else if (g > 1)  pcost += w_2day;
                }
            }
        }
        pcost += period_pen[new_pid];
        if (is_large[eid] && fl_penalty > 0 && is_last_period[new_pid])
            pcost += fl_penalty;

        int enr = exam_enroll[eid];
        int cur = sol_last_sync->get_pr_enroll(new_pid, new_rid);
        if (cur + enr > room_cap[new_rid]) pcost += 100000;
        pcost += room_pen[new_rid];

        return pcost;
    }

    // ── Population full-eval batch API ──
    // CPU path calls fe.full_eval per solution (bit-exact, matches parent
    // algos). GPU path is reserved for full_eval_kernel (scheduled — writes
    // one block per solution, uses same adj-based semantics as
    // score_full_cpu_ref_adj below).
    //
    // Usage pattern (in e.g. GA per-generation):
    //     Cuev.score_full_batch(offspring_vec, fitness_vec);
    void score_full_batch(const std::vector<Solution>& sols,
                          std::vector<double>& out_fitness) const
    {
        int N = (int)sols.size();
        out_fitness.resize(N);

#ifdef HAVE_CUDA
        // Full-eval has per-sol work that scales with ne + np*nr;
        // threshold N for batch-score is smaller than move-delta threshold.
        // Default: N >= 8 solutions makes the SoA packing + kernel worthwhile.
        static int pop_threshold = []() {
            const char* env = std::getenv("EXAM_CUDA_POP_THRESHOLD");
            return env ? std::atoi(env) : 8;
        }();
        if (gpu_active && N >= pop_threshold) {
            // Pack N solutions into flat SoA arrays, upload, dispatch kernel.
            std::vector<int32_t> pop_po((size_t)N * ne);
            std::vector<int32_t> pop_ro((size_t)N * ne);
            std::vector<int32_t> pop_pe((size_t)N * np * nr);
            std::vector<int32_t> pop_pc((size_t)N * np * nr);
            for (int k = 0; k < N; k++) {
                const Solution& s = sols[k];
                for (int i = 0; i < ne; i++) {
                    pop_po[(size_t)k * ne + i] = s.period_of[i];
                    pop_ro[(size_t)k * ne + i] = s.room_of[i];
                }
                for (int p = 0; p < np; p++)
                    for (int r = 0; r < nr; r++) {
                        size_t idx = (size_t)k * np * nr + (size_t)p * nr + r;
                        pop_pe[idx] = s.get_pr_enroll(p, r);
                        pop_pc[idx] = s.get_pr_count(p, r);
                    }
            }
            std::vector<int64_t> fixed(N);
            cuda_state_score_full_batch(d_state,
                pop_po.data(), pop_ro.data(), pop_pe.data(), pop_pc.data(),
                N, fixed.data());
            for (int k = 0; k < N; k++) out_fitness[k] = (double)fixed[k];
            return;
        }
#endif
        for (int k = 0; k < N; k++)
            out_fitness[k] = Ecach.E.full_eval(sols[k]).fitness();
    }

    double score_full_single(const Solution& sol) const {
        return Ecach.E.full_eval(sol).fitness();
    }

    // ── Full-eval CPU twin (per-student, matches fe.full_eval bit-exact) ──
    // Mirrors evaluator.h::full_eval: per-student sort-and-count for conflicts
    // (counts n-1 duplicates per cluster, matching the ITC 2007 semantic).
    // Bit-exact match to fe.full_eval on BOTH feasible and infeasible inputs.
    //
    // Sync contract: caller must have called sync_state(sol) first.
    double score_full_cpu_ref(const Solution& sol) const {
        long long hard_conf = 0, hard_dur = 0, hard_room = 0, hard_phc = 0, hard_rhc = 0;
        long long soft_2row = 0, soft_2day = 0, soft_spread = 0;
        long long soft_pp = 0, soft_rp = 0, soft_fl = 0, soft_mix = 0;
        const auto& po = sol.period_of;
        const auto& ro = sol.room_of;

        // ── Per-student: conflicts + proximity (matches fe.full_eval lines 180-225) ──
        for (int s = 0; s < n_students; s++) {
            int s_start = student_starts[s];
            int s_len = student_starts[s + 1] - s_start;
            if (s_len == 0) continue;

            std::vector<int> pids;
            pids.reserve(s_len);
            for (int i = 0; i < s_len; i++) {
                int eid = student_flat[s_start + i];
                if (eid < ne && po[eid] >= 0) pids.push_back(po[eid]);
            }
            if (pids.empty()) continue;
            std::sort(pids.begin(), pids.end());

            // Conflicts: adjacent duplicates in sorted list
            for (int i = 1; i < (int)pids.size(); i++)
                if (pids[i] == pids[i - 1]) hard_conf += 1;

            // Unique pids for proximity
            std::vector<int> upids;
            upids.reserve(pids.size());
            for (int p : pids) if (upids.empty() || upids.back() != p) upids.push_back(p);

            for (int i = 0; i < (int)upids.size(); i++) {
                int pi = upids[i];
                int di = period_day[pi], posi = period_daypos[pi];
                for (int j = i + 1; j < (int)upids.size(); j++) {
                    int pj = upids[j];
                    int dj = period_day[pj];
                    if (di == dj) {
                        int gpos = std::abs(posi - period_daypos[pj]);
                        if (gpos == 1)      soft_2row += w_2row;
                        else if (gpos > 1)  soft_2day += w_2day;
                    }
                    int gap = std::abs(pj - pi);
                    if (gap > 0 && gap <= w_spread) soft_spread += 1;
                }
            }
        }

        // ── Per-exam terms ──
        for (int e = 0; e < ne; e++) {
            int pe = po[e]; if (pe < 0) continue;
            int re = ro[e];
            if (exam_dur[e] > period_dur[pe]) hard_dur += 1;
            soft_pp += period_pen[pe];
            soft_rp += room_pen[re];
            if (is_large[e] && fl_penalty > 0 && is_last_period[pe])
                soft_fl += fl_penalty;
        }

        // ── Per-slot: room-cap + non-mixed durations ──
        std::vector<std::vector<int>> pr_exams((size_t)np * nr);
        for (int e = 0; e < ne; e++) {
            int pe = po[e]; if (pe < 0) continue;
            pr_exams[(size_t)pe * nr + ro[e]].push_back(e);
        }
        int w_mixed = Ecach.E.w_mixed;
        for (int p = 0; p < np; p++) {
            for (int r = 0; r < nr; r++) {
                int total = sol.get_pr_enroll(p, r);
                if (total > room_cap[r]) hard_room += 1;
                auto& eids = pr_exams[(size_t)p * nr + r];
                if (eids.size() > 1) {
                    int first_dur = exam_dur[eids[0]];
                    bool mixed = false;
                    for (size_t i = 1; i < eids.size(); i++)
                        if (exam_dur[eids[i]] != first_dur) { mixed = true; break; }
                    if (mixed) soft_mix += w_mixed;
                }
            }
        }

        // ── PHC (once-per-pair via e < other guard not needed; full_eval iterates phcs directly) ──
        for (int e = 0; e < ne; e++) {
            int pe = po[e]; if (pe < 0) continue;
            int phc_s = phc_starts[e];
            int phc_end = phc_starts[e + 1];
            for (int i = phc_s; i < phc_end; i++) {
                int other = phc_others[i];
                int tcode = phc_tcodes[i];
                if (other < e) continue;  // count each pair once
                int po2 = po[other]; if (po2 < 0) continue;
                if      (tcode == 0) { if (pe != po2) hard_phc += 1; }
                else if (tcode == 1) { if (pe == po2) hard_phc += 1; }
                else if (tcode == 2) { if (pe <= po2) hard_phc += 1; }
                else if (tcode == 3) { if (pe >= po2) hard_phc += 1; }
            }
        }

        // ── RHC ──
        for (int eid : rhc_exam_ids) {
            if (eid >= ne) continue;
            int pid = po[eid]; if (pid < 0) continue;
            int rid = ro[eid];
            int cnt = sol.get_pr_count(pid, rid);
            if (cnt > 1) hard_rhc += 1;
        }

        long long hard = hard_conf + hard_dur + hard_room + hard_phc + hard_rhc;
        long long soft = soft_2row + soft_2day + soft_spread + soft_pp + soft_rp + soft_fl + soft_mix;
        return (double)hard * 100000.0 + (double)soft;
    }

    // ── Full-eval CPU twin (adj-based — legacy, kept for perf comparison) ──
    // Computes fitness from flat arrays only, using adj-based aggregation.
    // Matches fe.full_eval() only on FEASIBLE solutions; on infeasible the
    // conflict-count diverges (adj counts C(n,2) per cluster vs the
    // per-student n-1 that fe.full_eval and score_full_cpu_ref use).
    //
    // Sync contract: caller must have called sync_state(sol) first; this
    // function reads period_of/room_of from the saved sol pointer.
    double score_full_cpu_ref_adj(const Solution& sol) const {
        long long hard = 0;
        long long soft = 0;
        const auto& po = sol.period_of;
        const auto& ro = sol.room_of;

        // ── Conflicts + proximity + spread (adj-based, halved to match pairs) ──
        // For each exam, walk adj; contribution from each (e, nb) pair is
        // cnt × (conflict_indicator | proximity × weight | spread). Halve
        // at the end since each pair is counted both directions.
        long long conf2 = 0, two_row2 = 0, two_day2 = 0, spread2 = 0;
        for (int e = 0; e < ne; e++) {
            int pe = po[e]; if (pe < 0) continue;
            int dpe = period_day[pe];
            int dposE = period_daypos[pe];
            int len = adj_len[e];
            const int32_t* adj_o = adj_other.data() + (size_t)e * adj_stride;
            const int32_t* adj_c = adj_cnt.data()   + (size_t)e * adj_stride;
            for (int i = 0; i < len; i++) {
                int nb = adj_o[i]; int cnt = adj_c[i];
                if (cnt == 0) continue;
                int pnb = po[nb]; if (pnb < 0) continue;

                if (pe == pnb) conf2 += cnt;
                if (dpe == period_day[pnb]) {
                    int g = std::abs(dposE - period_daypos[pnb]);
                    if (g == 1)      two_row2 += (long long)w_2row * cnt;
                    else if (g > 1)  two_day2 += (long long)w_2day * cnt;
                }
                int gap = std::abs(pe - pnb);
                if (gap > 0 && gap <= w_spread) spread2 += cnt;
            }
        }
        hard += conf2 / 2;
        soft += two_row2 / 2;
        soft += two_day2 / 2;
        soft += spread2 / 2;

        // ── Duration + room-cap + period/room pen + frontload ──
        for (int e = 0; e < ne; e++) {
            int pe = po[e]; if (pe < 0) continue;
            int re = ro[e];
            if (exam_dur[e] > period_dur[pe]) hard += 1;
            soft += period_pen[pe];
            soft += room_pen[re];
            if (is_large[e] && fl_penalty > 0 && is_last_period[pe])
                soft += fl_penalty;
        }

        // ── Room occupancy (via pr_enroll) + non-mixed durations ──
        // Build per-slot exam lists on the fly (could be cached but it's O(ne))
        std::vector<std::vector<int>> pr_exams((size_t)np * nr);
        for (int e = 0; e < ne; e++) {
            int pe = po[e]; if (pe < 0) continue;
            pr_exams[(size_t)pe * nr + ro[e]].push_back(e);
        }
        int w_mixed = Ecach.E.w_mixed;
        for (int p = 0; p < np; p++) {
            for (int r = 0; r < nr; r++) {
                int total = sol.get_pr_enroll(p, r);
                if (total > room_cap[r]) hard += 1;
                auto& eids = pr_exams[(size_t)p * nr + r];
                if (eids.size() > 1) {
                    int first_dur = exam_dur[eids[0]];
                    bool mixed = false;
                    for (size_t i = 1; i < eids.size(); i++) {
                        if (exam_dur[eids[i]] != first_dur) { mixed = true; break; }
                    }
                    if (mixed) soft += w_mixed;
                }
            }
        }

        // ── PHC ──
        for (int e = 0; e < ne; e++) {
            int pe = po[e]; if (pe < 0) continue;
            int phc_s = phc_starts[e];
            int phc_end = phc_starts[e + 1];
            for (int i = phc_s; i < phc_end; i++) {
                int other = phc_others[i];
                int tcode = phc_tcodes[i];
                if (other < e) continue;  // count each pair once
                int po2 = po[other]; if (po2 < 0) continue;
                if      (tcode == 0) { if (pe != po2) hard += 1; }
                else if (tcode == 1) { if (pe == po2) hard += 1; }
                else if (tcode == 2) { if (pe <= po2) hard += 1; }
                else if (tcode == 3) { if (pe >= po2) hard += 1; }
            }
        }

        // ── RHC ──
        for (int eid : rhc_exam_ids) {
            if (eid >= ne) continue;
            int pid = po[eid]; if (pid < 0) continue;
            int rid = ro[eid];
            int cnt = sol.get_pr_count(pid, rid);
            if (cnt > 1) hard += 1;
        }

        return (double)hard * 100000.0 + (double)soft;
    }

    // Batch-score placements. GPU path launches delta_kernel_placement;
    // CPU path loops score_placement_cpu_ref.
    void score_placement_batch(const std::vector<int32_t>& mv_eid,
                                const std::vector<int32_t>& mv_new_pid,
                                const std::vector<int32_t>& mv_new_rid,
                                std::vector<long long>& out_costs) const
    {
        int n = (int)mv_eid.size();
        out_costs.resize(n);

#ifdef HAVE_CUDA
        if (gpu_active && n >= cuda_batch_threshold()) {
            lazy_sync_gpu_if_needed();
            std::vector<int64_t> fixed(n);
            cuda_state_score_placement_batch(d_state,
                                              mv_eid.data(), mv_new_pid.data(), mv_new_rid.data(),
                                              n, fixed.data());
            for (int i = 0; i < n; i++) out_costs[i] = (long long)fixed[i];
            return;
        }
#endif
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            out_costs[i] = score_placement_cpu_ref(mv_eid[i], mv_new_pid[i], mv_new_rid[i]);
    }

    // Batch-size threshold below which GPU is strictly slower than CPU
    // (launch overhead + memcpy > per-candidate compute savings). Tuned
    // empirically on RTX 3050 Ti + CUDA 12.0: ~2000 candidates is the
    // crossover. Below → route to CPU. Above → use kernel.
    // Override at runtime via env var EXAM_CUDA_BATCH_THRESHOLD.
    static int cuda_batch_threshold() {
        static int cached = []() {
            const char* env = std::getenv("EXAM_CUDA_BATCH_THRESHOLD");
            if (env) return std::atoi(env);
            return 2000;
        }();
        return cached;
    }

    // Batch-score n moves. Dispatch:
    //   • GPU active AND batch ≥ threshold → CUDA kernel (int64 fixed-point)
    //   • otherwise → Ecach.move_delta (cache-hit path, ~30 ns/call)
    //
    // Guard rationale: on small batches, kernel launch + memcpy dominate,
    // so CPU wins. Threshold prevents GPU from ever being slower. CPU path
    // is already validated bit-exact with the kernel.
    void score_batch(const std::vector<int32_t>& mv_eid,
                     const std::vector<int32_t>& mv_new_pid,
                     const std::vector<int32_t>& mv_new_rid,
                     std::vector<double>& out_deltas) const
    {
        int n = (int)mv_eid.size();
        out_deltas.resize(n);

#ifdef HAVE_CUDA
        if (gpu_active && n >= cuda_batch_threshold()) {
            lazy_sync_gpu_if_needed();
            std::vector<int64_t> fixed(n);
            cuda_state_score_batch(d_state,
                                   mv_eid.data(), mv_new_pid.data(), mv_new_rid.data(),
                                   n, fixed.data());
            for (int i = 0; i < n; i++) out_deltas[i] = (double)fixed[i];
            return;
        }
#endif
        // CPU fallback — OMP-parallelize (matches tabu_cached's parallel scan)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            out_deltas[i] = Ecach.move_delta(*sol_last_sync, mv_eid[i], mv_new_pid[i], mv_new_rid[i]);
    }

};
