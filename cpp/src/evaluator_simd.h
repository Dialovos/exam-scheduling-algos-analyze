/*
 * Alt-route: SIMD / inline-asm variants of FastEvaluator::move_delta.
 * Does NOT modify FastEvaluator — wraps it, reuses its public state.
 *
 * Three variants, all strictly equivalent in result to the scalar baseline:
 *   move_delta_adj   — scalar, uses adj[eid] instead of student_exams double
 *                      loop. O(|adj[eid]|) vs O(Σ|student_exams[s]|).
 *   move_delta_simd  — AVX2 intrinsics on the inner neighbour loop. Gathers
 *                      po/day/daypos in 8-wide batches, masked for padding.
 *   move_delta_asm   — inline asm kernel for the hot conflict-count core.
 *                      Full move_delta with the inner loop hand-written in
 *                      AVX2 asm (vpgatherdd / vpcmpeqd / vpand).
 *
 * Correctness check: (other, shared_students) pair-based multiplication
 * yields the same total as per-student accumulation when both students
 * take both exams. See README or adj construction in models.h.
 */

#pragma once

#include "evaluator.h"

#include <cstdint>
#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#define EVAL_SIMD_AVX2 1
#endif

class FastEvaluatorSIMD {
public:
    const FastEvaluator& E;

    // Padded SoA adjacency for SIMD gather.
    // adj_other[eid][i] / adj_cnt[eid][i] — size is a multiple of 8, padding
    // has cnt=0 so padded lanes contribute nothing to any accumulator.
    std::vector<std::vector<int32_t>> adj_other;
    std::vector<std::vector<int32_t>> adj_cnt;
    std::vector<int32_t> adj_len;  // real (unpadded) length

    explicit FastEvaluatorSIMD(const FastEvaluator& e) : E(e) {
        int ne = E.ne;
        adj_other.resize(ne);
        adj_cnt.resize(ne);
        adj_len.resize(ne);
        for (int eid = 0; eid < ne; eid++) {
            adj_len[eid] = (int32_t)E.P.adj[eid].size();
            adj_other[eid].reserve(((adj_len[eid] + 7) / 8) * 8);
            adj_cnt[eid].reserve(((adj_len[eid] + 7) / 8) * 8);
            for (auto& pr : E.P.adj[eid]) {
                adj_other[eid].push_back(pr.first);
                adj_cnt[eid].push_back(pr.second);
            }
            while (adj_other[eid].size() % 8 != 0) {
                adj_other[eid].push_back(0);  // valid index, cnt=0 nullifies
                adj_cnt[eid].push_back(0);
            }
        }
    }

    // ── Shared non-student tail ────────────────────────────
    // Everything that isn't the student/adjacency loop. Identical code in
    // all three variants — factored out for clarity, not to save work.
    inline void add_non_student_parts(
        const Solution& sol, int eid, int old_pid, int old_rid,
        int new_pid, int new_rid, double& dh, double& ds) const
    {
        const auto& po = sol.period_of;
        int dur = E.exam_dur[eid];
        if (dur > E.period_dur[old_pid]) dh -= 1;
        if (dur > E.period_dur[new_pid]) dh += 1;

        int enr = E.exam_enroll[eid];
        int old_total = sol.get_pr_enroll(old_pid, old_rid);
        int new_total = sol.get_pr_enroll(new_pid, new_rid);
        dh -= ((old_total > E.room_cap[old_rid]) ? 1 : 0) -
              (((old_total - enr) > E.room_cap[old_rid]) ? 1 : 0);
        dh += (((new_total + enr) > E.room_cap[new_rid]) ? 1 : 0) -
              ((new_total > E.room_cap[new_rid]) ? 1 : 0);

        ds += E.period_pen[new_pid] - E.period_pen[old_pid];
        ds += E.room_pen[new_rid] - E.room_pen[old_rid];

        if (E.large_exams.count(eid) && E.fl_penalty > 0) {
            bool was_late = E.last_periods.count(old_pid) > 0;
            bool will_late = E.last_periods.count(new_pid) > 0;
            if (was_late && !will_late)      ds -= E.fl_penalty;
            else if (!was_late && will_late)  ds += E.fl_penalty;
        }

        for (auto& pc : E.phc_by_exam[eid]) {
            int other = pc.first, tcode = pc.second;
            int o_pid = po[other]; if (o_pid < 0) continue;
            if (tcode == 0)      { if (old_pid != o_pid) dh -= 1; if (new_pid != o_pid) dh += 1; }
            else if (tcode == 1) { if (old_pid == o_pid) dh -= 1; if (new_pid == o_pid) dh += 1; }
            else if (tcode == 2) { if (old_pid <= o_pid) dh -= 1; if (new_pid <= o_pid) dh += 1; }
            else if (tcode == 3) { if (old_pid >= o_pid) dh -= 1; if (new_pid >= o_pid) dh += 1; }
        }

        if (!E.rhc_exams.empty()) {
            int oc = sol.get_pr_count(old_pid, old_rid);
            int nc = sol.get_pr_count(new_pid, new_rid);
            bool eid_rhc = E.rhc_exams.count(eid) > 0;
            if (eid_rhc) {
                if (oc > 1) dh -= 1;
                if (nc > 0) dh += 1;
            }
            for (int re : E.rhc_exams) {
                if (re == eid || re >= E.ne) continue;
                int rp = po[re]; if (rp < 0) continue;
                int rr = sol.room_of[re];
                if (rp == old_pid && rr == old_rid && oc == 2) dh -= 1;
                if (rp == new_pid && rr == new_rid && nc == 1) dh += 1;
            }
        }
    }

    // ── Variant 1: scalar, adj-based student loop ───────────
    double move_delta_adj(const Solution& sol, int eid, int new_pid, int new_rid) const {
        int old_pid = sol.period_of[eid];
        if (old_pid < 0) return E.move_delta(sol, eid, new_pid, new_rid);
        int old_rid = sol.room_of[eid];
        if (old_pid == new_pid && old_rid == new_rid) return 0.0;

        const auto& po = sol.period_of;
        double dh = 0, ds = 0;

        int old_day = E.period_day[old_pid], old_dpos = E.period_daypos[old_pid];
        int new_day = E.period_day[new_pid], new_dpos = E.period_daypos[new_pid];
        int w_2row = E.w_2row, w_2day = E.w_2day, w_spread = E.w_spread;

        const int32_t* other_p = adj_other[eid].data();
        const int32_t* cnt_p   = adj_cnt[eid].data();
        int32_t len            = adj_len[eid];

        for (int i = 0; i < len; i++) {
            int other = other_p[i], cnt = cnt_p[i];
            int o_pid = po[other];
            if (o_pid < 0) continue;

            if (o_pid == old_pid) dh -= cnt;
            if (o_pid == new_pid) dh += cnt;

            int o_day  = E.period_day[o_pid];
            int o_dpos = E.period_daypos[o_pid];

            if (old_day == o_day) {
                int g = std::abs(old_dpos - o_dpos);
                if (g == 1)      ds -= (double)w_2row * cnt;
                else if (g > 1)  ds -= (double)w_2day * cnt;
            }
            int og = std::abs(old_pid - o_pid);
            if (og > 0 && og <= w_spread) ds -= cnt;

            if (new_day == o_day) {
                int g = std::abs(new_dpos - o_dpos);
                if (g == 1)      ds += (double)w_2row * cnt;
                else if (g > 1)  ds += (double)w_2day * cnt;
            }
            int ng = std::abs(new_pid - o_pid);
            if (ng > 0 && ng <= w_spread) ds += cnt;
        }

        add_non_student_parts(sol, eid, old_pid, old_rid, new_pid, new_rid, dh, ds);
        return dh * 100000.0 + ds;
    }

#ifdef EVAL_SIMD_AVX2

    // ── Variant 2: AVX2 intrinsics ─────────────────────────
    double move_delta_simd(const Solution& sol, int eid, int new_pid, int new_rid) const {
        int old_pid = sol.period_of[eid];
        if (old_pid < 0) return E.move_delta(sol, eid, new_pid, new_rid);
        int old_rid = sol.room_of[eid];
        if (old_pid == new_pid && old_rid == new_rid) return 0.0;

        const int* po_ptr   = sol.period_of.data();
        const int* day_ptr  = E.period_day.data();
        const int* dpos_ptr = E.period_daypos.data();

        const __m256i old_pid_v  = _mm256_set1_epi32(old_pid);
        const __m256i new_pid_v  = _mm256_set1_epi32(new_pid);
        const __m256i old_day_v  = _mm256_set1_epi32(E.period_day[old_pid]);
        const __m256i new_day_v  = _mm256_set1_epi32(E.period_day[new_pid]);
        const __m256i old_dpos_v = _mm256_set1_epi32(E.period_daypos[old_pid]);
        const __m256i new_dpos_v = _mm256_set1_epi32(E.period_daypos[new_pid]);
        const __m256i w2row_v    = _mm256_set1_epi32(E.w_2row);
        const __m256i w2day_v    = _mm256_set1_epi32(E.w_2day);
        const __m256i wspread_v  = _mm256_set1_epi32(E.w_spread);
        const __m256i one_v      = _mm256_set1_epi32(1);
        const __m256i zero_v     = _mm256_setzero_si256();
        const __m256i negone_v   = _mm256_set1_epi32(-1);

        __m256i dh_acc = zero_v;
        __m256i ds_acc = zero_v;

        const int32_t* other_p = adj_other[eid].data();
        const int32_t* cnt_p   = adj_cnt[eid].data();
        int padded_len = (int)adj_other[eid].size();   // multiple of 8

        for (int i = 0; i < padded_len; i += 8) {
            __m256i other_v = _mm256_loadu_si256((const __m256i*)(other_p + i));
            __m256i cnt_v   = _mm256_loadu_si256((const __m256i*)(cnt_p   + i));

            // o_pid = po[other]  (safe: padding uses other=0, always in range)
            __m256i o_pid_v = _mm256_i32gather_epi32(po_ptr, other_v, 4);

            // valid = (o_pid > -1) AND (cnt != 0)   — kills padding + unassigned
            __m256i valid = _mm256_andnot_si256(
                _mm256_cmpeq_epi32(cnt_v, zero_v),
                _mm256_cmpgt_epi32(o_pid_v, negone_v));

            // Conflict: dh += cnt*(eq_new - eq_old)
            __m256i eq_old = _mm256_and_si256(_mm256_cmpeq_epi32(o_pid_v, old_pid_v), valid);
            __m256i eq_new = _mm256_and_si256(_mm256_cmpeq_epi32(o_pid_v, new_pid_v), valid);
            __m256i dh_add = _mm256_and_si256(cnt_v, eq_new);
            __m256i dh_sub = _mm256_and_si256(cnt_v, eq_old);
            dh_acc = _mm256_add_epi32(dh_acc, _mm256_sub_epi32(dh_add, dh_sub));

            // Masked gather day[o_pid], daypos[o_pid]
            __m256i o_day_v  = _mm256_mask_i32gather_epi32(zero_v, day_ptr,  o_pid_v, valid, 4);
            __m256i o_dpos_v = _mm256_mask_i32gather_epi32(zero_v, dpos_ptr, o_pid_v, valid, 4);

            // ── old-side proximity ──
            __m256i same_old = _mm256_and_si256(_mm256_cmpeq_epi32(old_day_v, o_day_v), valid);
            __m256i d_old    = _mm256_abs_epi32(_mm256_sub_epi32(old_dpos_v, o_dpos_v));
            __m256i is2row_o = _mm256_and_si256(same_old, _mm256_cmpeq_epi32(d_old, one_v));
            __m256i is2day_o = _mm256_and_si256(same_old, _mm256_cmpgt_epi32(d_old, one_v));
            __m256i t2row_o  = _mm256_and_si256(_mm256_mullo_epi32(cnt_v, w2row_v), is2row_o);
            __m256i t2day_o  = _mm256_and_si256(_mm256_mullo_epi32(cnt_v, w2day_v), is2day_o);
            ds_acc = _mm256_sub_epi32(ds_acc, _mm256_add_epi32(t2row_o, t2day_o));

            __m256i gap_old  = _mm256_abs_epi32(_mm256_sub_epi32(old_pid_v, o_pid_v));
            __m256i g_gt0_o  = _mm256_cmpgt_epi32(gap_old, zero_v);
            __m256i g_le_o   = _mm256_andnot_si256(_mm256_cmpgt_epi32(gap_old, wspread_v),
                                                   negone_v);
            __m256i spread_o = _mm256_and_si256(_mm256_and_si256(g_gt0_o, g_le_o), valid);
            ds_acc = _mm256_sub_epi32(ds_acc, _mm256_and_si256(cnt_v, spread_o));

            // ── new-side proximity ──
            __m256i same_new = _mm256_and_si256(_mm256_cmpeq_epi32(new_day_v, o_day_v), valid);
            __m256i d_new    = _mm256_abs_epi32(_mm256_sub_epi32(new_dpos_v, o_dpos_v));
            __m256i is2row_n = _mm256_and_si256(same_new, _mm256_cmpeq_epi32(d_new, one_v));
            __m256i is2day_n = _mm256_and_si256(same_new, _mm256_cmpgt_epi32(d_new, one_v));
            __m256i t2row_n  = _mm256_and_si256(_mm256_mullo_epi32(cnt_v, w2row_v), is2row_n);
            __m256i t2day_n  = _mm256_and_si256(_mm256_mullo_epi32(cnt_v, w2day_v), is2day_n);
            ds_acc = _mm256_add_epi32(ds_acc, _mm256_add_epi32(t2row_n, t2day_n));

            __m256i gap_new  = _mm256_abs_epi32(_mm256_sub_epi32(new_pid_v, o_pid_v));
            __m256i g_gt0_n  = _mm256_cmpgt_epi32(gap_new, zero_v);
            __m256i g_le_n   = _mm256_andnot_si256(_mm256_cmpgt_epi32(gap_new, wspread_v),
                                                   negone_v);
            __m256i spread_n = _mm256_and_si256(_mm256_and_si256(g_gt0_n, g_le_n), valid);
            ds_acc = _mm256_add_epi32(ds_acc, _mm256_and_si256(cnt_v, spread_n));
        }

        // Horizontal sum
        auto hsum = [](__m256i v) -> int {
            __m128i lo = _mm256_castsi256_si128(v);
            __m128i hi = _mm256_extracti128_si256(v, 1);
            __m128i s  = _mm_add_epi32(lo, hi);
            s = _mm_hadd_epi32(s, s);
            s = _mm_hadd_epi32(s, s);
            return _mm_cvtsi128_si32(s);
        };

        double dh = (double)hsum(dh_acc);
        double ds = (double)hsum(ds_acc);

        add_non_student_parts(sol, eid, old_pid, old_rid, new_pid, new_rid, dh, ds);
        return dh * 100000.0 + ds;
    }

    // ── Variant 3: inline-asm conflict kernel ──────────────
    // AT&T syntax note: "insn src2, src1, dst" — operand order is REVERSED
    // from Intel manuals. vpcmpgtd src2, src1, dst  ==> dst = (src1 > src2).
    //
    // This batch kernel does the conflict-delta for one 8-wide group entirely
    // in inline asm (once o_pid is gathered via intrinsic for portability).
    // Returns the per-lane (cnt * (eq_new - eq_old)) vector.
    static inline __m256i conflict_batch_asm(
        __m256i o_pid, __m256i cnt_v,
        __m256i old_v, __m256i new_v,
        __m256i negone, __m256i zero)
    {
        __m256i valid, tmp, eq_old, eq_new, delta;
        asm volatile (
            // valid = (o_pid > -1)
            "vpcmpgtd %[negone], %[opid], %[valid]\n\t"
            // tmp   = (cnt == 0)
            "vpcmpeqd %[zero],   %[cnt],  %[tmp]\n\t"
            // valid = (NOT tmp) AND valid     -- vpandn src2, src1, dst ==> dst = ~src1 & src2
            "vpandn   %[valid],  %[tmp],  %[valid]\n\t"
            // eq_old = (o_pid == old_v) AND valid
            "vpcmpeqd %[oldv],   %[opid], %[eqo]\n\t"
            "vpand    %[valid],  %[eqo],  %[eqo]\n\t"
            // eq_new = (o_pid == new_v) AND valid
            "vpcmpeqd %[newv],   %[opid], %[eqn]\n\t"
            "vpand    %[valid],  %[eqn],  %[eqn]\n\t"
            // eqn := cnt AND eqn ;  eqo := cnt AND eqo
            "vpand    %[cnt],    %[eqn],  %[eqn]\n\t"
            "vpand    %[cnt],    %[eqo],  %[eqo]\n\t"
            // delta = eqn - eqo
            "vpsubd   %[eqo],    %[eqn],  %[delta]\n\t"
            : [valid] "=&x"(valid),
              [tmp]   "=&x"(tmp),
              [eqo]   "=&x"(eq_old),
              [eqn]   "=&x"(eq_new),
              [delta] "=&x"(delta)
            : [opid]   "x"(o_pid),
              [cnt]    "x"(cnt_v),
              [oldv]   "x"(old_v),
              [newv]   "x"(new_v),
              [negone] "x"(negone),
              [zero]   "x"(zero)
        );
        return delta;
    }

    // Full move_delta with inline-asm conflict kernel.
    // Soft proximity part uses scalar adj loop — asm demo is scoped to the
    // single hottest op (conflict count), where it's cleanly isolable.
    double move_delta_asm(const Solution& sol, int eid, int new_pid, int new_rid) const {
        int old_pid = sol.period_of[eid];
        if (old_pid < 0) return E.move_delta(sol, eid, new_pid, new_rid);
        int old_rid = sol.room_of[eid];
        if (old_pid == new_pid && old_rid == new_rid) return 0.0;

        const auto& po = sol.period_of;
        const int* po_ptr = po.data();

        const __m256i old_v  = _mm256_set1_epi32(old_pid);
        const __m256i new_v  = _mm256_set1_epi32(new_pid);
        const __m256i negone = _mm256_set1_epi32(-1);
        const __m256i zero   = _mm256_setzero_si256();

        __m256i acc = zero;
        const int32_t* other_p = adj_other[eid].data();
        const int32_t* cnt_p   = adj_cnt[eid].data();
        int padded_len = (int)adj_other[eid].size();

        for (int i = 0; i < padded_len; i += 8) {
            __m256i other_v = _mm256_loadu_si256((const __m256i*)(other_p + i));
            __m256i cnt_v   = _mm256_loadu_si256((const __m256i*)(cnt_p   + i));
            __m256i o_pid_v = _mm256_i32gather_epi32(po_ptr, other_v, 4);
            __m256i delta = conflict_batch_asm(o_pid_v, cnt_v, old_v, new_v, negone, zero);
            acc = _mm256_add_epi32(acc, delta);
        }

        auto hsum = [](__m256i v) -> int {
            __m128i lo = _mm256_castsi256_si128(v);
            __m128i hi = _mm256_extracti128_si256(v, 1);
            __m128i s  = _mm_add_epi32(lo, hi);
            s = _mm_hadd_epi32(s, s);
            s = _mm_hadd_epi32(s, s);
            return _mm_cvtsi128_si32(s);
        };

        double dh = (double)hsum(acc);
        double ds = 0;

        // ── Soft proximity via scalar adj loop ──
        int old_day = E.period_day[old_pid], old_dpos = E.period_daypos[old_pid];
        int new_day = E.period_day[new_pid], new_dpos = E.period_daypos[new_pid];
        int w_2row = E.w_2row, w_2day = E.w_2day, w_spread = E.w_spread;
        int32_t len = adj_len[eid];

        for (int i = 0; i < len; i++) {
            int other = other_p[i], cnt = cnt_p[i];
            int o_pid = po[other]; if (o_pid < 0) continue;
            int o_day  = E.period_day[o_pid];
            int o_dpos = E.period_daypos[o_pid];
            if (old_day == o_day) {
                int g = std::abs(old_dpos - o_dpos);
                if (g == 1) ds -= (double)w_2row * cnt;
                else if (g > 1) ds -= (double)w_2day * cnt;
            }
            int og = std::abs(old_pid - o_pid);
            if (og > 0 && og <= w_spread) ds -= cnt;
            if (new_day == o_day) {
                int g = std::abs(new_dpos - o_dpos);
                if (g == 1) ds += (double)w_2row * cnt;
                else if (g > 1) ds += (double)w_2day * cnt;
            }
            int ng = std::abs(new_pid - o_pid);
            if (ng > 0 && ng <= w_spread) ds += cnt;
        }

        add_non_student_parts(sol, eid, old_pid, old_rid, new_pid, new_rid, dh, ds);
        return dh * 100000.0 + ds;
    }

#endif  // EVAL_SIMD_AVX2
};
