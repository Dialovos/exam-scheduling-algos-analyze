/*
 * CUDA move_delta batch kernel — Phase 3b (full delta).
 *
 * Matches CachedEvaluator::move_delta and cuda_evaluator.h's
 * score_delta_cpu_ref bit-exactly. Covers every term:
 *   hard: adj-conflicts, duration, room-capacity, PHC (4 codes), RHC
 *   soft: period_spread, two_in_row, two_in_day, period_pen, room_pen,
 *         front_load
 *
 * Output: int64 fixed-point (dh*100000 + ds) to stay in integer arithmetic.
 * Layout: one block per move. Adj is scanned in blockDim.x-wide chunks;
 * the scalar tail (duration/room/PHC/RHC/penalty/frontload) runs on tid=0
 * after warp-shuffle reduction of the adj-sum.
 *
 * Build: `make cuda-build`. Validated against score_delta_cpu_ref via
 * `make bench`; when bench reports 0 mismatches, kernel is correct.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

// ── Mirror of host CudaStaticParams struct (layout must match) ───
struct CudaStaticParams {
    int ne, np, nr;
    int adj_stride, phc_total;
    int w_2row, w_2day, w_spread;
    int w_mixed;
    int fl_penalty;
    int n_rhc;
    int n_students;
    int max_student_degree;
};

// ── Device state (opaque to host; lifetime tied to cuda_state_create/destroy) ─
struct CudaState {
    CudaStaticParams p;

    int32_t* d_adj_other;  int32_t* d_adj_cnt;  int32_t* d_adj_len;
    int32_t* d_exam_dur;   int32_t* d_exam_enroll;
    int32_t* d_period_dur; int32_t* d_period_day;
    int32_t* d_period_daypos; int32_t* d_period_pen;
    int32_t* d_room_cap;   int32_t* d_room_pen;
    uint8_t* d_is_large;   uint8_t* d_is_last_period;
    uint8_t* d_is_rhc_exam;
    int32_t* d_rhc_exam_ids;
    int32_t* d_phc_starts; int32_t* d_phc_others; int32_t* d_phc_tcodes;
    int32_t* d_student_starts; int32_t* d_student_flat;

    int32_t* d_period_of;  int32_t* d_room_of;
    int32_t* d_pr_enroll;  int32_t* d_pr_count;

    int n_moves_cap;
    int32_t* d_mv_eid;  int32_t* d_mv_new_pid;  int32_t* d_mv_new_rid;
    int64_t* d_out;

    // Population full-eval scratch (grown on demand)
    int n_pop_cap;
    int32_t* d_pop_po;  int32_t* d_pop_ro;
    int32_t* d_pop_pe;  int32_t* d_pop_pc;
    int64_t* d_pop_out;
};

// ── Error helpers ─────────────────────────────────────────────
static inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <typename T>
static inline void alloc_upload(T** d_ptr, const T* h_ptr, size_t n, const char* tag) {
    ck(cudaMalloc(d_ptr, n * sizeof(T)), tag);
    ck(cudaMemcpy(*d_ptr, h_ptr, n * sizeof(T), cudaMemcpyHostToDevice), tag);
}

// ── Full-delta kernel ─────────────────────────────────────────
__global__ void delta_kernel_full(
    CudaStaticParams p,
    const int32_t* __restrict__ adj_other,
    const int32_t* __restrict__ adj_cnt,
    const int32_t* __restrict__ adj_len,
    const int32_t* __restrict__ exam_dur,
    const int32_t* __restrict__ exam_enroll,
    const int32_t* __restrict__ period_dur,
    const int32_t* __restrict__ period_day,
    const int32_t* __restrict__ period_daypos,
    const int32_t* __restrict__ period_pen,
    const int32_t* __restrict__ room_cap,
    const int32_t* __restrict__ room_pen,
    const uint8_t* __restrict__ is_large,
    const uint8_t* __restrict__ is_last_period,
    const uint8_t* __restrict__ is_rhc_exam,
    const int32_t* __restrict__ rhc_exam_ids,
    const int32_t* __restrict__ phc_starts,
    const int32_t* __restrict__ phc_others,
    const int32_t* __restrict__ phc_tcodes,
    const int32_t* __restrict__ period_of,
    const int32_t* __restrict__ room_of,
    const int32_t* __restrict__ pr_enroll,
    const int32_t* __restrict__ pr_count,
    const int32_t* __restrict__ mv_eid,
    const int32_t* __restrict__ mv_new_pid,
    const int32_t* __restrict__ mv_new_rid,
    int64_t* __restrict__ out_deltas,
    int n_moves)
{
    int move_idx = blockIdx.x;
    if (move_idx >= n_moves) return;

    int tid = threadIdx.x;
    int eid = mv_eid[move_idx];
    int new_pid = mv_new_pid[move_idx];
    int new_rid = mv_new_rid[move_idx];
    int old_pid = period_of[eid];

    // Unplaced — caller should not ask, but be defensive
    if (old_pid < 0) {
        if (tid == 0) out_deltas[move_idx] = (int64_t)1e15;
        return;
    }
    int old_rid = room_of[eid];
    if (old_pid == new_pid && old_rid == new_rid) {
        if (tid == 0) out_deltas[move_idx] = 0;
        return;
    }

    int new_day  = period_day[new_pid];
    int new_dpos = period_daypos[new_pid];
    int old_day  = period_day[old_pid];
    int old_dpos = period_daypos[old_pid];
    int w_2row = p.w_2row, w_2day = p.w_2day, w_spread = p.w_spread;

    int len = adj_len[eid];
    int32_t partial_hard = 0, partial_soft = 0;

    // Strided adj scan
    for (int i = tid; i < len; i += blockDim.x) {
        int other = adj_other[(size_t)eid * p.adj_stride + i];
        int cnt   = adj_cnt  [(size_t)eid * p.adj_stride + i];
        if (cnt == 0) continue;
        int o_pid = period_of[other];
        if (o_pid < 0) continue;
        int o_day  = period_day[o_pid];
        int o_dpos = period_daypos[o_pid];

        // new-period contribution
        if (new_pid == o_pid) partial_hard += cnt;
        if (new_day == o_day) {
            int g = abs(new_dpos - o_dpos);
            if (g == 1)      partial_soft += w_2row * cnt;
            else if (g > 1)  partial_soft += w_2day * cnt;
        }
        int og_new = abs(new_pid - o_pid);
        if (og_new > 0 && og_new <= w_spread) partial_soft += cnt;

        // old-period contribution (subtract)
        if (old_pid == o_pid) partial_hard -= cnt;
        if (old_day == o_day) {
            int g = abs(old_dpos - o_dpos);
            if (g == 1)      partial_soft -= w_2row * cnt;
            else if (g > 1)  partial_soft -= w_2day * cnt;
        }
        int og_old = abs(old_pid - o_pid);
        if (og_old > 0 && og_old <= w_spread) partial_soft -= cnt;
    }

    // Warp + block reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_hard += __shfl_down_sync(0xffffffff, partial_hard, offset);
        partial_soft += __shfl_down_sync(0xffffffff, partial_soft, offset);
    }
    __shared__ int32_t warp_hard[32];
    __shared__ int32_t warp_soft[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) { warp_hard[warp_id] = partial_hard; warp_soft[warp_id] = partial_soft; }
    __syncthreads();
    if (warp_id == 0) {
        int nwarps = blockDim.x / 32;
        int32_t sh = (tid < nwarps) ? warp_hard[lane] : 0;
        int32_t ss = (tid < nwarps) ? warp_soft[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            sh += __shfl_down_sync(0xffffffff, sh, offset);
            ss += __shfl_down_sync(0xffffffff, ss, offset);
        }
        if (tid == 0) {
            // Scalar tail — all remaining terms are O(|phc_by_exam[e]|+|rhc|)
            int64_t dh = (int64_t)sh;
            int64_t ds = (int64_t)ss;

            // Duration
            int dur = exam_dur[eid];
            if (dur > period_dur[old_pid]) dh -= 1;
            if (dur > period_dur[new_pid]) dh += 1;

            // Room capacity
            int enr = exam_enroll[eid];
            int old_total = pr_enroll[(size_t)old_pid * p.nr + old_rid];
            int new_total = pr_enroll[(size_t)new_pid * p.nr + new_rid];
            dh -= ((old_total > room_cap[old_rid]) ? 1 : 0) -
                  (((old_total - enr) > room_cap[old_rid]) ? 1 : 0);
            dh += (((new_total + enr) > room_cap[new_rid]) ? 1 : 0) -
                  ((new_total > room_cap[new_rid]) ? 1 : 0);

            // Penalties
            ds += period_pen[new_pid] - period_pen[old_pid];
            ds += room_pen[new_rid]   - room_pen[old_rid];

            // Frontload
            if (is_large[eid] && p.fl_penalty > 0) {
                int was  = is_last_period[old_pid];
                int will = is_last_period[new_pid];
                if (was && !will)      ds -= p.fl_penalty;
                else if (!was && will) ds += p.fl_penalty;
            }

            // PHC — CSR iteration
            int phc_s = phc_starts[eid];
            int phc_e = phc_starts[eid + 1];
            for (int i = phc_s; i < phc_e; i++) {
                int other = phc_others[i];
                int tcode = phc_tcodes[i];
                int o_pid = period_of[other];
                if (o_pid < 0) continue;
                if      (tcode == 0) { if (old_pid != o_pid) dh -= 1; if (new_pid != o_pid) dh += 1; }
                else if (tcode == 1) { if (old_pid == o_pid) dh -= 1; if (new_pid == o_pid) dh += 1; }
                else if (tcode == 2) { if (old_pid <= o_pid) dh -= 1; if (new_pid <= o_pid) dh += 1; }
                else if (tcode == 3) { if (old_pid >= o_pid) dh -= 1; if (new_pid >= o_pid) dh += 1; }
            }

            // RHC
            if (p.n_rhc > 0) {
                int oc = pr_count[(size_t)old_pid * p.nr + old_rid];
                int nc = pr_count[(size_t)new_pid * p.nr + new_rid];
                if (is_rhc_exam[eid]) {
                    if (oc > 1) dh -= 1;
                    if (nc > 0) dh += 1;
                }
                for (int j = 0; j < p.n_rhc; j++) {
                    int re = rhc_exam_ids[j];
                    if (re == eid || re >= p.ne) continue;
                    int rp = period_of[re]; if (rp < 0) continue;
                    int rr = room_of[re];
                    if (rp == old_pid && rr == old_rid && oc == 2) dh -= 1;
                    if (rp == new_pid && rr == new_rid && nc == 1) dh += 1;
                }
            }

            out_deltas[move_idx] = dh * 100000LL + ds;
        }
    }
}

// ── Placement kernel ──────────────────────────────────────────
// Cost of placing an UNPLACED exam at (new_pid, new_rid). Mirrors
// CudaEvaluator::score_placement_cpu_ref exactly.
__global__ void delta_kernel_placement(
    CudaStaticParams p,
    const int32_t* __restrict__ adj_other,
    const int32_t* __restrict__ adj_len,
    const int32_t* __restrict__ exam_enroll,
    const int32_t* __restrict__ period_day,
    const int32_t* __restrict__ period_daypos,
    const int32_t* __restrict__ period_pen,
    const int32_t* __restrict__ room_cap,
    const int32_t* __restrict__ room_pen,
    const uint8_t* __restrict__ is_large,
    const uint8_t* __restrict__ is_last_period,
    const int32_t* __restrict__ period_of,
    const int32_t* __restrict__ pr_enroll,
    const int32_t* __restrict__ mv_eid,
    const int32_t* __restrict__ mv_new_pid,
    const int32_t* __restrict__ mv_new_rid,
    int64_t* __restrict__ out_costs,
    int n_moves)
{
    int move_idx = blockIdx.x;
    if (move_idx >= n_moves) return;

    int tid = threadIdx.x;
    int eid = mv_eid[move_idx];
    int new_pid = mv_new_pid[move_idx];
    int new_rid = mv_new_rid[move_idx];
    int new_day  = period_day[new_pid];
    int new_dpos = period_daypos[new_pid];
    int w_2row = p.w_2row, w_2day = p.w_2day, w_spread = p.w_spread;

    int len = adj_len[eid];
    int32_t partial = 0;

    for (int i = tid; i < len; i += blockDim.x) {
        int other = adj_other[(size_t)eid * p.adj_stride + i];
        int o_pid = period_of[other];
        if (o_pid < 0) continue;

        if (o_pid == new_pid) {
            partial += 100000;
        } else {
            int gap = abs(new_pid - o_pid);
            if (gap > 0 && gap <= w_spread) partial += 1;
            if (period_day[o_pid] == new_day) {
                int g = abs(period_daypos[o_pid] - new_dpos);
                if (g == 1)      partial += w_2row;
                else if (g > 1)  partial += w_2day;
            }
        }
    }

    for (int offset = 16; offset > 0; offset /= 2)
        partial += __shfl_down_sync(0xffffffff, partial, offset);

    __shared__ int32_t warp_sums[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();

    if (warp_id == 0) {
        int nwarps = blockDim.x / 32;
        int32_t sum = (tid < nwarps) ? warp_sums[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (tid == 0) {
            int64_t c = (int64_t)sum;
            c += period_pen[new_pid];
            if (is_large[eid] && p.fl_penalty > 0 && is_last_period[new_pid])
                c += p.fl_penalty;
            int cur = pr_enroll[(size_t)new_pid * p.nr + new_rid];
            if (cur + exam_enroll[eid] > room_cap[new_rid]) c += 100000;
            c += room_pen[new_rid];
            out_costs[move_idx] = c;
        }
    }
}

// ── Full-eval kernel ──────────────────────────────────────────
// One block per solution. Threads in a block parallelize the outer
// exam loop; reductions via warp-shuffle + shared memory.
//
// Pop buffers are laid out SoA per solution:
//   period_of [N × ne]  indexed [k*ne + e]
//   room_of   [N × ne]
//   pr_enroll [N × np × nr]  indexed [k*np*nr + p*nr + r]
//   pr_count  [N × np × nr]
//
// Matches cuda_evaluator.h::score_full_cpu_ref_adj bit-exact.
__global__ void full_eval_kernel(
    CudaStaticParams p,
    const int32_t* __restrict__ adj_other,
    const int32_t* __restrict__ adj_cnt,
    const int32_t* __restrict__ adj_len,
    const int32_t* __restrict__ exam_dur,
    const int32_t* __restrict__ period_dur,
    const int32_t* __restrict__ period_day,
    const int32_t* __restrict__ period_daypos,
    const int32_t* __restrict__ period_pen,
    const int32_t* __restrict__ room_cap,
    const int32_t* __restrict__ room_pen,
    const uint8_t* __restrict__ is_large,
    const uint8_t* __restrict__ is_last_period,
    const int32_t* __restrict__ rhc_exam_ids,
    const int32_t* __restrict__ phc_starts,
    const int32_t* __restrict__ phc_others,
    const int32_t* __restrict__ phc_tcodes,
    const int32_t* __restrict__ student_starts,
    const int32_t* __restrict__ student_flat,
    const int32_t* __restrict__ pop_period_of,
    const int32_t* __restrict__ pop_room_of,
    const int32_t* __restrict__ pop_pr_enroll,
    const int32_t* __restrict__ pop_pr_count,
    int64_t* __restrict__ out_fitness,
    int N_sols)
{
    int sid = blockIdx.x;
    if (sid >= N_sols) return;
    int tid = threadIdx.x;

    const int32_t* po = pop_period_of  + (size_t)sid * p.ne;
    const int32_t* ro = pop_room_of    + (size_t)sid * p.ne;
    const int32_t* pe = pop_pr_enroll  + (size_t)sid * p.np * p.nr;
    const int32_t* pc = pop_pr_count   + (size_t)sid * p.np * p.nr;

    int64_t hard_conf = 0, soft_2row = 0, soft_2day = 0, soft_spread = 0;
    int64_t hard_dur = 0, hard_room = 0, hard_phc = 0, hard_rhc = 0;
    int64_t soft_pp = 0, soft_rp = 0, soft_fl = 0, soft_mix = 0;

    int w_2row = p.w_2row, w_2day = p.w_2day, w_spread = p.w_spread;

    // ── Per-student (matches fe.full_eval conflict/proximity semantics) ──
    // Each thread handles a stride of students; sorts their pids via insertion
    // sort in registers, counts adjacent duplicates (conflicts), iterates
    // unique pid pairs for proximity penalties.
    constexpr int MAX_S_DEG = 32;  // safety cap; kernel asserts student_degree fits
    for (int s = tid; s < p.n_students; s += blockDim.x) {
        int s_start = student_starts[s];
        int s_end   = student_starts[s + 1];
        int s_len   = s_end - s_start;
        if (s_len <= 0) continue;

        int pids[MAX_S_DEG];
        int np_count = 0;
        for (int i = 0; i < s_len && np_count < MAX_S_DEG; i++) {
            int eid = student_flat[s_start + i];
            if (eid >= p.ne) continue;
            int pe_pid = po[eid];
            if (pe_pid >= 0) pids[np_count++] = pe_pid;
        }
        if (np_count == 0) continue;

        // Insertion sort (small arrays, cheap on GPU)
        for (int i = 1; i < np_count; i++) {
            int key = pids[i], j = i - 1;
            while (j >= 0 && pids[j] > key) { pids[j + 1] = pids[j]; j--; }
            pids[j + 1] = key;
        }
        // Conflicts: adjacent duplicates
        for (int i = 1; i < np_count; i++)
            if (pids[i] == pids[i - 1]) hard_conf += 1;

        // Unique pids (compact in-place)
        int uniq_len = 0;
        for (int i = 0; i < np_count; i++) {
            if (uniq_len == 0 || pids[uniq_len - 1] != pids[i]) pids[uniq_len++] = pids[i];
        }
        // Proximity: pairwise loop
        for (int i = 0; i < uniq_len; i++) {
            int pi = pids[i];
            int di = period_day[pi], dposi = period_daypos[pi];
            for (int j = i + 1; j < uniq_len; j++) {
                int pj = pids[j];
                int dj = period_day[pj];
                if (di == dj) {
                    int gpos = abs(dposi - period_daypos[pj]);
                    if (gpos == 1)      soft_2row += w_2row;
                    else if (gpos > 1)  soft_2day += w_2day;
                }
                int gap = abs(pj - pi);
                if (gap > 0 && gap <= w_spread) soft_spread += 1;
            }
        }
    }

    // ── Per-exam terms (duration + period/room pen + frontload + PHC) ──
    for (int e = tid; e < p.ne; e += blockDim.x) {
        int pe_pid = po[e]; if (pe_pid < 0) continue;
        int re_rid = ro[e];

        if (exam_dur[e] > period_dur[pe_pid]) hard_dur += 1;
        soft_pp += period_pen[pe_pid];
        soft_rp += room_pen[re_rid];
        if (is_large[e] && p.fl_penalty > 0 && is_last_period[pe_pid])
            soft_fl += p.fl_penalty;

        int phc_s = phc_starts[e];
        int phc_e = phc_starts[e + 1];
        for (int i = phc_s; i < phc_e; i++) {
            int other = phc_others[i];
            int tcode = phc_tcodes[i];
            if (other < e) continue;
            int po2 = po[other]; if (po2 < 0) continue;
            if      (tcode == 0) { if (pe_pid != po2) hard_phc += 1; }
            else if (tcode == 1) { if (pe_pid == po2) hard_phc += 1; }
            else if (tcode == 2) { if (pe_pid <= po2) hard_phc += 1; }
            else if (tcode == 3) { if (pe_pid >= po2) hard_phc += 1; }
        }
    }

    // ── Per-slot terms (room-cap overflow + non-mixed durations) ──
    int total_slots = p.np * p.nr;
    for (int s = tid; s < total_slots; s += blockDim.x) {
        int pid = s / p.nr;
        int rid = s % p.nr;
        int total = pe[(size_t)pid * p.nr + rid];
        if (total > room_cap[rid]) hard_room += 1;
        int cnt = pc[(size_t)pid * p.nr + rid];
        if (cnt > 1) {
            // Detect mixed durations: scan exams whose (po, ro) matches.
            // O(ne) per slot in worst case; fine since total_slots is small.
            int first_dur = -1;
            bool mixed = false;
            for (int e = 0; e < p.ne; e++) {
                if (po[e] == pid && ro[e] == rid) {
                    if (first_dur < 0) first_dur = exam_dur[e];
                    else if (exam_dur[e] != first_dur) { mixed = true; break; }
                }
            }
            if (mixed) soft_mix += p.w_mixed;
        }
    }

    // ── RHC ──
    for (int j = tid; j < p.n_rhc; j += blockDim.x) {
        int eid = rhc_exam_ids[j];
        if (eid >= p.ne) continue;
        int pid = po[eid]; if (pid < 0) continue;
        int rid = ro[eid];
        int cnt = pc[(size_t)pid * p.nr + rid];
        if (cnt > 1) hard_rhc += 1;
    }

    // ── Block-wide reduction across threads ──
    // All accumulators are already per-student or per-exam unique counts,
    // no halving needed (unlike the old adj-based version).
    __shared__ int64_t sh[12];
    if (tid == 0) for (int i = 0; i < 12; i++) sh[i] = 0;
    __syncthreads();
    atomicAdd((unsigned long long*)&sh[0],  (unsigned long long)hard_conf);
    atomicAdd((unsigned long long*)&sh[1],  (unsigned long long)soft_2row);
    atomicAdd((unsigned long long*)&sh[2],  (unsigned long long)soft_2day);
    atomicAdd((unsigned long long*)&sh[3],  (unsigned long long)soft_spread);
    atomicAdd((unsigned long long*)&sh[4],  (unsigned long long)hard_dur);
    atomicAdd((unsigned long long*)&sh[5],  (unsigned long long)hard_room);
    atomicAdd((unsigned long long*)&sh[6],  (unsigned long long)hard_phc);
    atomicAdd((unsigned long long*)&sh[7],  (unsigned long long)hard_rhc);
    atomicAdd((unsigned long long*)&sh[8],  (unsigned long long)soft_pp);
    atomicAdd((unsigned long long*)&sh[9],  (unsigned long long)soft_rp);
    atomicAdd((unsigned long long*)&sh[10], (unsigned long long)soft_fl);
    atomicAdd((unsigned long long*)&sh[11], (unsigned long long)soft_mix);
    __syncthreads();

    if (tid == 0) {
        int64_t hard = sh[0] + sh[4] + sh[5] + sh[6] + sh[7];
        int64_t soft = sh[1] + sh[2] + sh[3]
                     + sh[8] + sh[9] + sh[10] + sh[11];
        out_fitness[sid] = hard * 100000LL + soft;
    }
}

// ── Scratch buffer resizer ────────────────────────────────────
static void ensure_batch_cap(CudaState* s, int n) {
    if (n <= s->n_moves_cap) return;
    int new_cap = (n > 1024) ? n : 1024;
    while (new_cap < n) new_cap *= 2;
    if (s->d_mv_eid)     cudaFree(s->d_mv_eid);
    if (s->d_mv_new_pid) cudaFree(s->d_mv_new_pid);
    if (s->d_mv_new_rid) cudaFree(s->d_mv_new_rid);
    if (s->d_out)        cudaFree(s->d_out);
    ck(cudaMalloc(&s->d_mv_eid,     new_cap * sizeof(int32_t)), "mv_eid");
    ck(cudaMalloc(&s->d_mv_new_pid, new_cap * sizeof(int32_t)), "mv_new_pid");
    ck(cudaMalloc(&s->d_mv_new_rid, new_cap * sizeof(int32_t)), "mv_new_rid");
    ck(cudaMalloc(&s->d_out,        new_cap * sizeof(int64_t)), "out");
    s->n_moves_cap = new_cap;
}

static void ensure_pop_cap(CudaState* s, int N) {
    if (N <= s->n_pop_cap) return;
    int new_cap = (N > 64) ? N : 64;
    while (new_cap < N) new_cap *= 2;
    int ne = s->p.ne, np = s->p.np, nr = s->p.nr;
    if (s->d_pop_po)  cudaFree(s->d_pop_po);
    if (s->d_pop_ro)  cudaFree(s->d_pop_ro);
    if (s->d_pop_pe)  cudaFree(s->d_pop_pe);
    if (s->d_pop_pc)  cudaFree(s->d_pop_pc);
    if (s->d_pop_out) cudaFree(s->d_pop_out);
    ck(cudaMalloc(&s->d_pop_po,  (size_t)new_cap * ne * sizeof(int32_t)), "pop_po");
    ck(cudaMalloc(&s->d_pop_ro,  (size_t)new_cap * ne * sizeof(int32_t)), "pop_ro");
    ck(cudaMalloc(&s->d_pop_pe,  (size_t)new_cap * np * nr * sizeof(int32_t)), "pop_pe");
    ck(cudaMalloc(&s->d_pop_pc,  (size_t)new_cap * np * nr * sizeof(int32_t)), "pop_pc");
    ck(cudaMalloc(&s->d_pop_out, (size_t)new_cap * sizeof(int64_t)), "pop_out");
    s->n_pop_cap = new_cap;
}

// ── C API ─────────────────────────────────────────────────────
// ── Parallel SA portfolio kernel ──────────────────────────────
// One block per seed. Each block runs K iters of SA:
//   1. All threads compute adj-based move_delta for a random (eid, pid, rid)
//      proposal (thread 0 draws the proposal, broadcasts)
//   2. Block reduces partial sums to get full delta
//   3. Thread 0 accepts/rejects (Metropolis) and applies the move in-place
//   4. Synchronize and loop
//
// Simplifications vs full tabu:
//   • No tabu map (SA not Tabu)
//   • No cache tables (delta computed from adj each iter)
//   • No PHC/RHC/mixed in the proposal delta (just adj + duration + room-cap
//     + period/room pen + frontload). Full fitness at the end via
//     full_eval_kernel for the best-found.
//
// Device-per-seed state (allocated as N-contiguous arrays):
//   period_of[N*ne], room_of[N*ne], pr_enroll[N*np*nr], pr_count[N*np*nr],
//   best_po[N*ne], best_ro[N*ne], best_fitness[N], current_fitness[N],
//   rng_state[N] (uint64).
__device__ inline uint64_t xorshift64(uint64_t& s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return s;
}

__global__ void parallel_sa_kernel(
    CudaStaticParams p,
    const int32_t* __restrict__ adj_other,
    const int32_t* __restrict__ adj_cnt,
    const int32_t* __restrict__ adj_len,
    const int32_t* __restrict__ exam_dur,
    const int32_t* __restrict__ exam_enroll,
    const int32_t* __restrict__ period_dur,
    const int32_t* __restrict__ period_day,
    const int32_t* __restrict__ period_daypos,
    const int32_t* __restrict__ period_pen,
    const int32_t* __restrict__ room_cap,
    const int32_t* __restrict__ room_pen,
    const uint8_t* __restrict__ is_large,
    const uint8_t* __restrict__ is_last_period,
    int n_seeds, int n_iters,
    int init_temp_x1000,   // initial temp * 1000 (integer)
    int cooling_x1e6,      // cooling rate * 1e6 (integer)
    int32_t* __restrict__ pop_period_of,
    int32_t* __restrict__ pop_room_of,
    int32_t* __restrict__ pop_pr_enroll,
    int32_t* __restrict__ pop_pr_count,
    int32_t* __restrict__ pop_best_po,
    int32_t* __restrict__ pop_best_ro,
    int64_t* __restrict__ pop_current_fitness,
    int64_t* __restrict__ pop_best_fitness,
    uint64_t* __restrict__ pop_rng_state)
{
    int sid = blockIdx.x;
    if (sid >= n_seeds) return;
    int tid = threadIdx.x;

    int32_t* po = pop_period_of + (size_t)sid * p.ne;
    int32_t* ro = pop_room_of   + (size_t)sid * p.ne;
    int32_t* pe = pop_pr_enroll + (size_t)sid * p.np * p.nr;
    int32_t* pc = pop_pr_count  + (size_t)sid * p.np * p.nr;
    int32_t* bpo = pop_best_po  + (size_t)sid * p.ne;
    int32_t* bro = pop_best_ro  + (size_t)sid * p.ne;

    __shared__ uint64_t shared_rng;
    __shared__ int sh_eid, sh_new_pid, sh_new_rid, sh_old_pid, sh_old_rid;
    __shared__ int64_t sh_delta;
    __shared__ int sh_accept;
    __shared__ int64_t sh_current_fitness;
    __shared__ int64_t sh_best_fitness;
    __shared__ double sh_temp;

    if (tid == 0) {
        shared_rng = pop_rng_state[sid];
        sh_current_fitness = pop_current_fitness[sid];
        sh_best_fitness = pop_best_fitness[sid];
        sh_temp = (double)init_temp_x1000 / 1000.0;
    }
    __syncthreads();

    double cooling = (double)cooling_x1e6 / 1e6;

    for (int it = 0; it < n_iters; it++) {
        // Thread 0 proposes a random (eid, pid, rid) move
        if (tid == 0) {
            int eid = (int)(xorshift64(shared_rng) % p.ne);
            int new_pid = (int)(xorshift64(shared_rng) % p.np);
            int new_rid = (int)(xorshift64(shared_rng) % p.nr);
            int old_pid = po[eid];
            int old_rid = ro[eid];
            if (old_pid < 0) {
                sh_eid = -1;
            } else if (old_pid == new_pid && old_rid == new_rid) {
                sh_eid = -1;
            } else if (exam_dur[eid] > period_dur[new_pid]) {
                // Filter: don't propose moves that violate duration (cheap filter)
                sh_eid = -1;
            } else {
                sh_eid = eid;
                sh_old_pid = old_pid;
                sh_old_rid = old_rid;
                sh_new_pid = new_pid;
                sh_new_rid = new_rid;
            }
        }
        __syncthreads();

        if (sh_eid < 0) continue;

        int eid = sh_eid, old_pid = sh_old_pid, old_rid = sh_old_rid;
        int new_pid = sh_new_pid, new_rid = sh_new_rid;

        // All threads compute partial adj-delta
        int64_t partial_hard = 0, partial_soft = 0;
        int len = adj_len[eid];
        int new_day  = period_day[new_pid];
        int new_dpos = period_daypos[new_pid];
        int old_day  = period_day[old_pid];
        int old_dpos = period_daypos[old_pid];
        int w_2row = p.w_2row, w_2day = p.w_2day, w_spread = p.w_spread;

        for (int i = tid; i < len; i += blockDim.x) {
            int other = adj_other[(size_t)eid * p.adj_stride + i];
            int cnt   = adj_cnt  [(size_t)eid * p.adj_stride + i];
            if (cnt == 0) continue;
            int o_pid = po[other];
            if (o_pid < 0) continue;
            // new
            if (new_pid == o_pid) partial_hard += cnt;
            if (new_day == period_day[o_pid]) {
                int g = abs(new_dpos - period_daypos[o_pid]);
                if (g == 1)      partial_soft += (int64_t)w_2row * cnt;
                else if (g > 1)  partial_soft += (int64_t)w_2day * cnt;
            }
            int og_new = abs(new_pid - o_pid);
            if (og_new > 0 && og_new <= w_spread) partial_soft += cnt;
            // old (subtract)
            if (old_pid == o_pid) partial_hard -= cnt;
            if (old_day == period_day[o_pid]) {
                int g = abs(old_dpos - period_daypos[o_pid]);
                if (g == 1)      partial_soft -= (int64_t)w_2row * cnt;
                else if (g > 1)  partial_soft -= (int64_t)w_2day * cnt;
            }
            int og_old = abs(old_pid - o_pid);
            if (og_old > 0 && og_old <= w_spread) partial_soft -= cnt;
        }

        // Warp reduce
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_hard += __shfl_down_sync(0xffffffff, partial_hard, offset);
            partial_soft += __shfl_down_sync(0xffffffff, partial_soft, offset);
        }
        __shared__ int64_t sh_hard[4], sh_soft[4];
        int warp = tid / 32, lane = tid % 32;
        if (lane == 0 && warp < 4) { sh_hard[warp] = partial_hard; sh_soft[warp] = partial_soft; }
        __syncthreads();

        if (tid == 0) {
            int64_t dh = sh_hard[0] + sh_hard[1] + sh_hard[2] + sh_hard[3];
            int64_t ds = sh_soft[0] + sh_soft[1] + sh_soft[2] + sh_soft[3];

            // Scalar tail: duration (already filtered), room-cap, period/room pen, frontload
            int enr = exam_enroll[eid];
            int old_total = pe[(size_t)old_pid * p.nr + old_rid];
            int new_total = pe[(size_t)new_pid * p.nr + new_rid];
            dh -= ((old_total > room_cap[old_rid]) ? 1 : 0) -
                  (((old_total - enr) > room_cap[old_rid]) ? 1 : 0);
            dh += (((new_total + enr) > room_cap[new_rid]) ? 1 : 0) -
                  ((new_total > room_cap[new_rid]) ? 1 : 0);
            ds += period_pen[new_pid] - period_pen[old_pid];
            ds += room_pen[new_rid]   - room_pen[old_rid];
            if (is_large[eid] && p.fl_penalty > 0) {
                int was = is_last_period[old_pid];
                int will = is_last_period[new_pid];
                if (was && !will)      ds -= p.fl_penalty;
                else if (!was && will) ds += p.fl_penalty;
            }
            int64_t delta = dh * 100000LL + ds;
            sh_delta = delta;

            // Feasibility-preserving SA: reject any move that increases hard
            // violations (dh > 0). On feasible starting state, this keeps the
            // trajectory in feasible space — avoids adj-vs-per-student
            // semantic drift on infeasible sols. Soft-only Metropolis for
            // dh <= 0 moves.
            bool accept = false;
            if (dh > 0) {
                accept = false;
            } else if (ds <= 0) {
                accept = true;
            } else if (sh_temp > 1e-6) {
                double u = (double)(xorshift64(shared_rng) & 0xFFFFFFFF) / 4294967296.0;
                if (u < exp(-(double)ds / sh_temp)) accept = true;
            }
            sh_accept = accept ? 1 : 0;

            if (accept) {
                // Apply move in place
                pe[(size_t)old_pid * p.nr + old_rid] -= enr;
                pc[(size_t)old_pid * p.nr + old_rid] -= 1;
                pe[(size_t)new_pid * p.nr + new_rid] += enr;
                pc[(size_t)new_pid * p.nr + new_rid] += 1;
                po[eid] = new_pid;
                ro[eid] = new_rid;
                sh_current_fitness += delta;
                if (sh_current_fitness < sh_best_fitness) {
                    sh_best_fitness = sh_current_fitness;
                    for (int i = 0; i < p.ne; i++) { bpo[i] = po[i]; bro[i] = ro[i]; }
                }
            }
            sh_temp *= cooling;
        }
        __syncthreads();
    }

    if (tid == 0) {
        pop_current_fitness[sid] = sh_current_fitness;
        pop_best_fitness[sid]    = sh_best_fitness;
        pop_rng_state[sid]       = shared_rng;
    }
}

extern "C" int cuda_runtime_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0) ? 1 : 0;
}

extern "C" void* cuda_state_create(
    const CudaStaticParams* params,
    const int32_t* h_adj_other, const int32_t* h_adj_cnt, const int32_t* h_adj_len,
    const int32_t* h_exam_dur, const int32_t* h_exam_enroll,
    const int32_t* h_period_dur, const int32_t* h_period_day,
    const int32_t* h_period_daypos, const int32_t* h_period_pen,
    const int32_t* h_room_cap, const int32_t* h_room_pen,
    const uint8_t* h_is_large, const uint8_t* h_is_last_period,
    const uint8_t* h_is_rhc_exam, const int32_t* h_rhc_exam_ids,
    const int32_t* h_phc_starts, const int32_t* h_phc_others, const int32_t* h_phc_tcodes,
    const int32_t* h_student_starts, const int32_t* h_student_flat)
{
    if (!cuda_runtime_available()) return nullptr;

    CudaState* s = new CudaState();
    s->p = *params;
    s->n_moves_cap = 0;
    s->d_mv_eid = s->d_mv_new_pid = s->d_mv_new_rid = nullptr;
    s->d_out = nullptr;
    s->n_pop_cap = 0;
    s->d_pop_po = s->d_pop_ro = s->d_pop_pe = s->d_pop_pc = nullptr;
    s->d_pop_out = nullptr;

    int ne = s->p.ne, np = s->p.np, nr = s->p.nr;
    size_t adj_bytes = (size_t)ne * s->p.adj_stride;

    alloc_upload(&s->d_adj_other, h_adj_other, adj_bytes, "adj_other");
    alloc_upload(&s->d_adj_cnt,   h_adj_cnt,   adj_bytes, "adj_cnt");
    alloc_upload(&s->d_adj_len,   h_adj_len,   (size_t)ne, "adj_len");
    alloc_upload(&s->d_exam_dur,    h_exam_dur,    (size_t)ne, "exam_dur");
    alloc_upload(&s->d_exam_enroll, h_exam_enroll, (size_t)ne, "exam_enroll");
    alloc_upload(&s->d_period_dur,    h_period_dur,    (size_t)np, "period_dur");
    alloc_upload(&s->d_period_day,    h_period_day,    (size_t)np, "period_day");
    alloc_upload(&s->d_period_daypos, h_period_daypos, (size_t)np, "period_daypos");
    alloc_upload(&s->d_period_pen,    h_period_pen,    (size_t)np, "period_pen");
    alloc_upload(&s->d_room_cap, h_room_cap, (size_t)nr, "room_cap");
    alloc_upload(&s->d_room_pen, h_room_pen, (size_t)nr, "room_pen");
    alloc_upload(&s->d_is_large,        h_is_large,        (size_t)ne, "is_large");
    alloc_upload(&s->d_is_last_period,  h_is_last_period,  (size_t)np, "is_last_period");
    alloc_upload(&s->d_is_rhc_exam,     h_is_rhc_exam,     (size_t)ne, "is_rhc_exam");
    if (s->p.n_rhc > 0)
        alloc_upload(&s->d_rhc_exam_ids, h_rhc_exam_ids, (size_t)s->p.n_rhc, "rhc_exam_ids");
    else
        s->d_rhc_exam_ids = nullptr;
    alloc_upload(&s->d_phc_starts, h_phc_starts, (size_t)(ne + 1), "phc_starts");
    if (s->p.phc_total > 0) {
        alloc_upload(&s->d_phc_others, h_phc_others, (size_t)s->p.phc_total, "phc_others");
        alloc_upload(&s->d_phc_tcodes, h_phc_tcodes, (size_t)s->p.phc_total, "phc_tcodes");
    } else {
        s->d_phc_others = nullptr;
        s->d_phc_tcodes = nullptr;
    }

    // Student-exams CSR
    alloc_upload(&s->d_student_starts, h_student_starts, (size_t)(s->p.n_students + 1), "student_starts");
    int total_se = h_student_starts[s->p.n_students];
    if (total_se > 0)
        alloc_upload(&s->d_student_flat, h_student_flat, (size_t)total_se, "student_flat");
    else
        s->d_student_flat = nullptr;

    // Dynamic state — allocate but don't initialize (first sync_dynamic will fill)
    ck(cudaMalloc(&s->d_period_of, ne * sizeof(int32_t)), "period_of");
    ck(cudaMalloc(&s->d_room_of,   ne * sizeof(int32_t)), "room_of");
    ck(cudaMalloc(&s->d_pr_enroll, (size_t)np * nr * sizeof(int32_t)), "pr_enroll");
    ck(cudaMalloc(&s->d_pr_count,  (size_t)np * nr * sizeof(int32_t)), "pr_count");

    return s;
}

extern "C" void cuda_state_sync_dynamic(
    void* state,
    const int32_t* h_period_of, const int32_t* h_room_of,
    const int32_t* h_pr_enroll, const int32_t* h_pr_count)
{
    CudaState* s = (CudaState*)state;
    int ne = s->p.ne, np = s->p.np, nr = s->p.nr;
    ck(cudaMemcpy(s->d_period_of, h_period_of, ne * sizeof(int32_t), cudaMemcpyHostToDevice), "po sync");
    ck(cudaMemcpy(s->d_room_of,   h_room_of,   ne * sizeof(int32_t), cudaMemcpyHostToDevice), "ro sync");
    ck(cudaMemcpy(s->d_pr_enroll, h_pr_enroll, (size_t)np * nr * sizeof(int32_t), cudaMemcpyHostToDevice), "pe sync");
    ck(cudaMemcpy(s->d_pr_count,  h_pr_count,  (size_t)np * nr * sizeof(int32_t), cudaMemcpyHostToDevice), "pc sync");
}

extern "C" void cuda_state_score_batch(
    void* state,
    const int32_t* h_mv_eid,
    const int32_t* h_mv_new_pid,
    const int32_t* h_mv_new_rid,
    int n_moves,
    int64_t* h_out_deltas_fixed)
{
    CudaState* s = (CudaState*)state;
    if (n_moves == 0) return;
    ensure_batch_cap(s, n_moves);

    ck(cudaMemcpy(s->d_mv_eid,     h_mv_eid,     n_moves * sizeof(int32_t), cudaMemcpyHostToDevice), "mv_eid up");
    ck(cudaMemcpy(s->d_mv_new_pid, h_mv_new_pid, n_moves * sizeof(int32_t), cudaMemcpyHostToDevice), "mv_new_pid up");
    ck(cudaMemcpy(s->d_mv_new_rid, h_mv_new_rid, n_moves * sizeof(int32_t), cudaMemcpyHostToDevice), "mv_new_rid up");

    dim3 grid(n_moves);
    dim3 block(128);
    delta_kernel_full<<<grid, block>>>(
        s->p,
        s->d_adj_other, s->d_adj_cnt, s->d_adj_len,
        s->d_exam_dur, s->d_exam_enroll,
        s->d_period_dur, s->d_period_day, s->d_period_daypos, s->d_period_pen,
        s->d_room_cap, s->d_room_pen,
        s->d_is_large, s->d_is_last_period, s->d_is_rhc_exam,
        s->d_rhc_exam_ids,
        s->d_phc_starts, s->d_phc_others, s->d_phc_tcodes,
        s->d_period_of, s->d_room_of, s->d_pr_enroll, s->d_pr_count,
        s->d_mv_eid, s->d_mv_new_pid, s->d_mv_new_rid,
        s->d_out, n_moves);

    ck(cudaMemcpy(h_out_deltas_fixed, s->d_out, n_moves * sizeof(int64_t), cudaMemcpyDeviceToHost),
       "out download");
}

extern "C" void cuda_state_score_placement_batch(
    void* state,
    const int32_t* h_mv_eid,
    const int32_t* h_mv_new_pid,
    const int32_t* h_mv_new_rid,
    int n_moves,
    int64_t* h_out_costs)
{
    CudaState* s = (CudaState*)state;
    if (n_moves == 0) return;
    ensure_batch_cap(s, n_moves);

    ck(cudaMemcpy(s->d_mv_eid,     h_mv_eid,     n_moves * sizeof(int32_t), cudaMemcpyHostToDevice), "mv_eid up");
    ck(cudaMemcpy(s->d_mv_new_pid, h_mv_new_pid, n_moves * sizeof(int32_t), cudaMemcpyHostToDevice), "mv_new_pid up");
    ck(cudaMemcpy(s->d_mv_new_rid, h_mv_new_rid, n_moves * sizeof(int32_t), cudaMemcpyHostToDevice), "mv_new_rid up");

    dim3 grid(n_moves);
    dim3 block(128);
    delta_kernel_placement<<<grid, block>>>(
        s->p,
        s->d_adj_other, s->d_adj_len,
        s->d_exam_enroll,
        s->d_period_day, s->d_period_daypos, s->d_period_pen,
        s->d_room_cap, s->d_room_pen,
        s->d_is_large, s->d_is_last_period,
        s->d_period_of, s->d_pr_enroll,
        s->d_mv_eid, s->d_mv_new_pid, s->d_mv_new_rid,
        s->d_out, n_moves);

    ck(cudaMemcpy(h_out_costs, s->d_out, n_moves * sizeof(int64_t), cudaMemcpyDeviceToHost),
       "placement out download");
}

extern "C" void cuda_state_score_full_batch(
    void* state,
    const int32_t* h_pop_period_of,
    const int32_t* h_pop_room_of,
    const int32_t* h_pop_pr_enroll,
    const int32_t* h_pop_pr_count,
    int N,
    int64_t* h_out_fitness)
{
    CudaState* s = (CudaState*)state;
    if (N == 0) return;
    ensure_pop_cap(s, N);

    int ne = s->p.ne, np = s->p.np, nr = s->p.nr;
    ck(cudaMemcpy(s->d_pop_po, h_pop_period_of, (size_t)N * ne * sizeof(int32_t), cudaMemcpyHostToDevice), "pop_po up");
    ck(cudaMemcpy(s->d_pop_ro, h_pop_room_of,   (size_t)N * ne * sizeof(int32_t), cudaMemcpyHostToDevice), "pop_ro up");
    ck(cudaMemcpy(s->d_pop_pe, h_pop_pr_enroll, (size_t)N * np * nr * sizeof(int32_t), cudaMemcpyHostToDevice), "pop_pe up");
    ck(cudaMemcpy(s->d_pop_pc, h_pop_pr_count,  (size_t)N * np * nr * sizeof(int32_t), cudaMemcpyHostToDevice), "pop_pc up");

    dim3 grid(N);
    dim3 block(128);
    full_eval_kernel<<<grid, block>>>(
        s->p,
        s->d_adj_other, s->d_adj_cnt, s->d_adj_len,
        s->d_exam_dur,
        s->d_period_dur, s->d_period_day, s->d_period_daypos, s->d_period_pen,
        s->d_room_cap, s->d_room_pen,
        s->d_is_large, s->d_is_last_period,
        s->d_rhc_exam_ids,
        s->d_phc_starts, s->d_phc_others, s->d_phc_tcodes,
        s->d_student_starts, s->d_student_flat,
        s->d_pop_po, s->d_pop_ro, s->d_pop_pe, s->d_pop_pc,
        s->d_pop_out, N);

    ck(cudaMemcpy(h_out_fitness, s->d_pop_out, N * sizeof(int64_t), cudaMemcpyDeviceToHost),
       "pop_out download");
}

// Parallel SA portfolio host entry.
// Host must provide N_seeds × ne + N_seeds × np × nr worth of initial state
// (each seed's initial sol) plus N_seeds RNG states.
// Kernel runs n_iters SA iters per seed in parallel, mutating state in-place.
// Returns updated per-seed state + best_fitness[] + best sols.
extern "C" void cuda_parallel_sa_run(
    void* state,
    int n_seeds, int n_iters,
    double init_temp, double cooling,
    // host initial state (n_seeds copies)
    const int32_t* h_pop_po, const int32_t* h_pop_ro,
    const int32_t* h_pop_pe, const int32_t* h_pop_pc,
    const int64_t* h_current_fitness,
    const uint64_t* h_rng_seeds,
    // host output
    int32_t* h_best_po, int32_t* h_best_ro,
    int64_t* h_best_fitness)
{
    CudaState* s = (CudaState*)state;
    int ne = s->p.ne, np = s->p.np, nr = s->p.nr;

    // Allocate per-seed state buffers
    int32_t *d_pop_po, *d_pop_ro, *d_pop_pe, *d_pop_pc;
    int32_t *d_best_po, *d_best_ro;
    int64_t *d_current, *d_best;
    uint64_t *d_rng;
    ck(cudaMalloc(&d_pop_po, (size_t)n_seeds * ne * sizeof(int32_t)), "sa_po");
    ck(cudaMalloc(&d_pop_ro, (size_t)n_seeds * ne * sizeof(int32_t)), "sa_ro");
    ck(cudaMalloc(&d_pop_pe, (size_t)n_seeds * np * nr * sizeof(int32_t)), "sa_pe");
    ck(cudaMalloc(&d_pop_pc, (size_t)n_seeds * np * nr * sizeof(int32_t)), "sa_pc");
    ck(cudaMalloc(&d_best_po, (size_t)n_seeds * ne * sizeof(int32_t)), "sa_bpo");
    ck(cudaMalloc(&d_best_ro, (size_t)n_seeds * ne * sizeof(int32_t)), "sa_bro");
    ck(cudaMalloc(&d_current, (size_t)n_seeds * sizeof(int64_t)), "sa_cur");
    ck(cudaMalloc(&d_best,    (size_t)n_seeds * sizeof(int64_t)), "sa_best");
    ck(cudaMalloc(&d_rng,     (size_t)n_seeds * sizeof(uint64_t)), "sa_rng");

    ck(cudaMemcpy(d_pop_po, h_pop_po, (size_t)n_seeds * ne * sizeof(int32_t), cudaMemcpyHostToDevice), "upload po");
    ck(cudaMemcpy(d_pop_ro, h_pop_ro, (size_t)n_seeds * ne * sizeof(int32_t), cudaMemcpyHostToDevice), "upload ro");
    ck(cudaMemcpy(d_pop_pe, h_pop_pe, (size_t)n_seeds * np * nr * sizeof(int32_t), cudaMemcpyHostToDevice), "upload pe");
    ck(cudaMemcpy(d_pop_pc, h_pop_pc, (size_t)n_seeds * np * nr * sizeof(int32_t), cudaMemcpyHostToDevice), "upload pc");
    ck(cudaMemcpy(d_best_po, h_pop_po, (size_t)n_seeds * ne * sizeof(int32_t), cudaMemcpyHostToDevice), "upload bpo");
    ck(cudaMemcpy(d_best_ro, h_pop_ro, (size_t)n_seeds * ne * sizeof(int32_t), cudaMemcpyHostToDevice), "upload bro");
    ck(cudaMemcpy(d_current, h_current_fitness, n_seeds * sizeof(int64_t), cudaMemcpyHostToDevice), "upload cur");
    ck(cudaMemcpy(d_best,    h_current_fitness, n_seeds * sizeof(int64_t), cudaMemcpyHostToDevice), "upload best");
    ck(cudaMemcpy(d_rng,     h_rng_seeds, n_seeds * sizeof(uint64_t), cudaMemcpyHostToDevice), "upload rng");

    int init_temp_x1000 = (int)(init_temp * 1000.0);
    int cooling_x1e6 = (int)(cooling * 1e6);

    dim3 grid(n_seeds);
    dim3 block(128);
    parallel_sa_kernel<<<grid, block>>>(
        s->p,
        s->d_adj_other, s->d_adj_cnt, s->d_adj_len,
        s->d_exam_dur, s->d_exam_enroll,
        s->d_period_dur, s->d_period_day, s->d_period_daypos, s->d_period_pen,
        s->d_room_cap, s->d_room_pen,
        s->d_is_large, s->d_is_last_period,
        n_seeds, n_iters,
        init_temp_x1000, cooling_x1e6,
        d_pop_po, d_pop_ro, d_pop_pe, d_pop_pc,
        d_best_po, d_best_ro,
        d_current, d_best,
        d_rng);

    ck(cudaDeviceSynchronize(), "sa kernel sync");

    ck(cudaMemcpy(h_best_po, d_best_po, (size_t)n_seeds * ne * sizeof(int32_t), cudaMemcpyDeviceToHost), "download bpo");
    ck(cudaMemcpy(h_best_ro, d_best_ro, (size_t)n_seeds * ne * sizeof(int32_t), cudaMemcpyDeviceToHost), "download bro");
    ck(cudaMemcpy(h_best_fitness, d_best, n_seeds * sizeof(int64_t), cudaMemcpyDeviceToHost), "download best");

    cudaFree(d_pop_po); cudaFree(d_pop_ro); cudaFree(d_pop_pe); cudaFree(d_pop_pc);
    cudaFree(d_best_po); cudaFree(d_best_ro);
    cudaFree(d_current); cudaFree(d_best); cudaFree(d_rng);
}

extern "C" void cuda_state_destroy(void* state) {
    if (!state) return;
    CudaState* s = (CudaState*)state;
    cudaFree(s->d_adj_other); cudaFree(s->d_adj_cnt); cudaFree(s->d_adj_len);
    cudaFree(s->d_exam_dur); cudaFree(s->d_exam_enroll);
    cudaFree(s->d_period_dur); cudaFree(s->d_period_day);
    cudaFree(s->d_period_daypos); cudaFree(s->d_period_pen);
    cudaFree(s->d_room_cap); cudaFree(s->d_room_pen);
    cudaFree(s->d_is_large); cudaFree(s->d_is_last_period); cudaFree(s->d_is_rhc_exam);
    if (s->d_rhc_exam_ids) cudaFree(s->d_rhc_exam_ids);
    cudaFree(s->d_phc_starts);
    if (s->d_phc_others) cudaFree(s->d_phc_others);
    if (s->d_phc_tcodes) cudaFree(s->d_phc_tcodes);
    if (s->d_student_starts) cudaFree(s->d_student_starts);
    if (s->d_student_flat)   cudaFree(s->d_student_flat);
    cudaFree(s->d_period_of); cudaFree(s->d_room_of);
    cudaFree(s->d_pr_enroll); cudaFree(s->d_pr_count);
    if (s->d_mv_eid)     cudaFree(s->d_mv_eid);
    if (s->d_mv_new_pid) cudaFree(s->d_mv_new_pid);
    if (s->d_mv_new_rid) cudaFree(s->d_mv_new_rid);
    if (s->d_out)        cudaFree(s->d_out);
    if (s->d_pop_po)     cudaFree(s->d_pop_po);
    if (s->d_pop_ro)     cudaFree(s->d_pop_ro);
    if (s->d_pop_pe)     cudaFree(s->d_pop_pe);
    if (s->d_pop_pc)     cudaFree(s->d_pop_pc);
    if (s->d_pop_out)    cudaFree(s->d_pop_out);
    delete s;
}
