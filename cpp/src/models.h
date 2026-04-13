/*
 * Exam, Period, Room, ProblemInstance, Solution, constraint, and weighting
 * structs. ProblemInstance::build_derived() builds adjacency + conflict data.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

// ============================================================
//  BASIC TYPES
// ============================================================

struct Exam {
    int id;
    int duration;
    std::vector<int> students; // sorted
    int enrollment() const { return (int)students.size(); }
};

struct Period {
    int id;
    std::string date;
    std::string time_str;
    int duration;
    int penalty;
    int day;
};

struct Room {
    int id;
    int capacity;
    int penalty;
};

struct PeriodHardConstraint {
    int exam1;
    std::string type; // EXAM_COINCIDENCE, EXCLUSION, AFTER
    int exam2;
};

struct RoomHardConstraint {
    int exam_id;
    std::string type; // ROOM_EXCLUSIVE
};

struct Weightings {
    int two_in_a_row = 1;
    int two_in_a_day = 1;
    int period_spread = 1;
    int non_mixed_durations = 1;
    int fl_n_largest = 0;
    int fl_n_last = 0;
    int fl_penalty = 0;
};

// ============================================================
//  PROBLEM INSTANCE
// ============================================================

struct ProblemInstance {
    std::vector<Exam> exams;
    std::vector<Period> periods;
    std::vector<Room> rooms;
    std::vector<PeriodHardConstraint> phcs;
    std::vector<RoomHardConstraint> rhcs;
    Weightings w;

    // Derived (call build_derived() after loading)
    std::vector<std::vector<std::pair<int, int>>> adj; // adj[e] = {(neighbor, shared_students)}
    std::vector<std::vector<int>> student_exams;       // student_exams[sid] = {exam_ids}
    std::map<int, std::vector<int>> periods_per_day;   // day -> sorted period ids
    std::vector<int> period_daypos;                    // period -> position within its day

    int n_e() const { return (int)exams.size(); }
    int n_p() const { return (int)periods.size(); }
    int n_r() const { return (int)rooms.size(); }

    void build_derived() {
        int ne = n_e();
        int np = n_p();

        // student -> exams
        int max_sid = 0;
        for (auto& e : exams)
            for (int s : e.students)
                max_sid = std::max(max_sid, s);
        student_exams.assign(max_sid + 1, {});
        for (auto& e : exams)
            for (int s : e.students)
                student_exams[s].push_back(e.id);

        // conflict matrix
        std::map<std::pair<int, int>, int> conflicts;
        for (int s = 0; s <= max_sid; s++) {
            auto& se = student_exams[s];
            for (int i = 0; i < (int)se.size(); i++)
                for (int j = i + 1; j < (int)se.size(); j++) {
                    int a = std::min(se[i], se[j]);
                    int b = std::max(se[i], se[j]);
                    conflicts[{a, b}]++;
                }
        }

        adj.assign(ne, {});
        for (auto& [pr, cnt] : conflicts) {
            adj[pr.first].push_back({pr.second, cnt});
            adj[pr.second].push_back({pr.first, cnt});
        }

        // periods per day
        periods_per_day.clear();
        for (auto& p : periods)
            periods_per_day[p.day].push_back(p.id);
        for (auto& [d, pids] : periods_per_day)
            std::sort(pids.begin(), pids.end());

        period_daypos.assign(np, 0);
        for (auto& [d, pids] : periods_per_day)
            for (int i = 0; i < (int)pids.size(); i++)
                period_daypos[pids[i]] = i;
    }
};

// ============================================================
//  SOLUTION
// ============================================================

struct Solution {
    int n_e = 0, n_p = 0, n_r = 0;
    std::vector<int> period_of;  // -1 = unassigned
    std::vector<int> room_of;    // -1 = unassigned
    std::vector<int> pr_enroll;  // flat [period * n_r + room] -> total enrollment
    std::vector<int> pr_count;   // flat [period * n_r + room] -> exam count (for room_exclusive)
    std::vector<int> enroll_cache;
    const ProblemInstance* prob = nullptr;

    Solution() = default;

    void init(const ProblemInstance& p) {
        prob = &p;
        n_e = p.n_e(); n_p = p.n_p(); n_r = p.n_r();
        period_of.assign(n_e, -1);
        room_of.assign(n_e, -1);
        pr_enroll.assign(n_p * n_r, 0);
        pr_count.assign(n_p * n_r, 0);
        enroll_cache.resize(n_e);
        for (int i = 0; i < n_e; i++)
            enroll_cache[i] = p.exams[i].enrollment();
    }

    int pr_key(int pid, int rid) const {
        return pid * n_r + rid;
    }

    int get_pr_enroll(int pid, int rid) const {
        return pr_enroll[pr_key(pid, rid)];
    }

    int get_pr_count(int pid, int rid) const {
        return pr_count[pr_key(pid, rid)];
    }

    void assign(int eid, int pid, int rid) {
        int enr = enroll_cache[eid];
        int old_pid = period_of[eid];
        if (old_pid >= 0) {
            int old_rid = room_of[eid];
            int key = pr_key(old_pid, old_rid);
            pr_enroll[key] -= enr;
            pr_count[key]--;
        }
        period_of[eid] = pid;
        room_of[eid] = rid;
        int key = pr_key(pid, rid);
        pr_enroll[key] += enr;
        pr_count[key]++;
    }

    Solution copy() const {
        Solution s;
        s.n_e = n_e; s.n_p = n_p; s.n_r = n_r;
        s.period_of = period_of;
        s.room_of = room_of;
        s.pr_enroll = pr_enroll;
        s.pr_count = pr_count;
        s.prob = prob;
        s.enroll_cache = enroll_cache;
        return s;
    }
};

// ============================================================
//  ALGORITHM RESULT (shared return type)
// ============================================================

struct EvalResult {
    int conflicts = 0;
    int room_occupancy = 0;
    int period_utilisation = 0;
    int period_related = 0;
    int room_related = 0;
    int two_in_a_row = 0;
    int two_in_a_day = 0;
    int period_spread = 0;
    int non_mixed_durations = 0;
    int front_load = 0;
    int period_penalty = 0;
    int room_penalty = 0;

    int hard() const {
        return conflicts + room_occupancy + period_utilisation +
               period_related + room_related;
    }
    int soft() const {
        return two_in_a_row + two_in_a_day + period_spread +
               non_mixed_durations + front_load +
               period_penalty + room_penalty;
    }
    bool feasible() const { return hard() == 0; }
    double fitness() const { return (double)hard() * 100000.0 + soft(); }
};

struct AlgoResult {
    Solution sol;
    EvalResult eval;
    double runtime_sec;
    int iterations;
    std::string algorithm;
};