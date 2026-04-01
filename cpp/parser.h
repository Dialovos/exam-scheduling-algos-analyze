/*
 * parser.h — ITC 2007 .exam file parser
 *
 * Reads [Exams], [Periods], [Rooms], [PeriodHardConstraints],
 * [RoomHardConstraints], and [InstitutionalWeightings] sections.
 */

#pragma once

#include "models.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace parser {

static inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static inline std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ','))
        parts.push_back(trim(tok));
    return parts;
}

inline ProblemInstance parse_exam_file(const std::string& filepath, int limit = 0) {
    ProblemInstance prob;
    std::ifstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "ERROR: Cannot open " << filepath << std::endl;
        std::exit(1);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line))
        lines.push_back(trim(line));

    int idx = 0;
    int n_lines = (int)lines.size();

    while (idx < n_lines) {
        const std::string& ln = lines[idx];

        // ── [Exams:N] ──
        if (ln.substr(0, 7) == "[Exams:") {
            int n = std::stoi(ln.substr(7, ln.size() - 8));
            int actual_n = (limit > 0) ? std::min(n, limit) : n;
            idx++;
            for (int eid = 0; eid < n && idx < n_lines; eid++, idx++) {
                auto parts = split_csv(lines[idx]);
                if (eid < actual_n) {
                    Exam e;
                    e.id = eid;
                    e.duration = std::stoi(parts[0]);
                    for (int i = 1; i < (int)parts.size(); i++) {
                        std::string s = trim(parts[i]);
                        if (!s.empty())
                            e.students.push_back(std::stoi(s));
                    }
                    std::sort(e.students.begin(), e.students.end());
                    prob.exams.push_back(std::move(e));
                }
            }
            continue;
        }
        // ── [Periods:N] ──
        else if (ln.substr(0, 9) == "[Periods:") {
            int n = std::stoi(ln.substr(9, ln.size() - 10));
            idx++;
            std::string cur_date;
            int cur_day = -1;
            for (int pid = 0; pid < n && idx < n_lines; pid++, idx++) {
                auto parts = split_csv(lines[idx]);
                Period p;
                p.id = pid;
                p.date = parts[0];
                p.time_str = parts.size() > 1 ? parts[1] : "";
                p.duration = parts.size() > 2 ? std::stoi(parts[2]) : 0;
                p.penalty = parts.size() > 3 ? std::stoi(parts[3]) : 0;
                if (p.date != cur_date) { cur_day++; cur_date = p.date; }
                p.day = cur_day;
                prob.periods.push_back(std::move(p));
            }
            continue;
        }
        // ── [Rooms:N] ──
        else if (ln.substr(0, 7) == "[Rooms:") {
            int n = std::stoi(ln.substr(7, ln.size() - 8));
            idx++;
            for (int rid = 0; rid < n && idx < n_lines; rid++, idx++) {
                auto parts = split_csv(lines[idx]);
                Room r;
                r.id = rid;
                r.capacity = std::stoi(parts[0]);
                r.penalty = parts.size() > 1 ? std::stoi(parts[1]) : 0;
                prob.rooms.push_back(std::move(r));
            }
            continue;
        }
        // ── [PeriodHardConstraints] ──
        else if (ln == "[PeriodHardConstraints]") {
            idx++;
            int max_eid = (int)prob.exams.size();
            while (idx < n_lines && !lines[idx].empty() && lines[idx][0] != '[') {
                auto parts = split_csv(lines[idx]);
                if (parts.size() >= 3) {
                    int e1 = std::stoi(parts[0]);
                    std::string ct = parts[1];
                    int e2 = std::stoi(parts[2]);
                    if (e1 < max_eid && e2 < max_eid)
                        prob.phcs.push_back({e1, ct, e2});
                }
                idx++;
            }
            continue;
        }
        // ── [RoomHardConstraints] ──
        else if (ln == "[RoomHardConstraints]") {
            idx++;
            int max_eid = (int)prob.exams.size();
            while (idx < n_lines && !lines[idx].empty() && lines[idx][0] != '[') {
                auto parts = split_csv(lines[idx]);
                if (parts.size() >= 2) {
                    int eid = std::stoi(parts[0]);
                    std::string ct = parts[1];
                    if (eid < max_eid)
                        prob.rhcs.push_back({eid, ct});
                }
                idx++;
            }
            continue;
        }
        // ── [InstitutionalWeightings] ──
        else if (ln == "[InstitutionalWeightings]") {
            idx++;
            while (idx < n_lines && !lines[idx].empty() && lines[idx][0] != '[') {
                std::string raw = lines[idx];
                auto parse_val = [&](const std::string& raw) -> int {
                    size_t pos = raw.find(':');
                    if (pos != std::string::npos)
                        return std::stoi(trim(raw.substr(pos + 1)));
                    pos = raw.find(',');
                    if (pos != std::string::npos)
                        return std::stoi(trim(raw.substr(pos + 1)));
                    return 0;
                };

                if (raw.substr(0, 9) == "TWOINAROW")
                    prob.w.two_in_a_row = parse_val(raw);
                else if (raw.substr(0, 9) == "TWOINADAY")
                    prob.w.two_in_a_day = parse_val(raw);
                else if (raw.substr(0, 12) == "PERIODSPREAD")
                    prob.w.period_spread = parse_val(raw);
                else if (raw.substr(0, 17) == "NONMIXEDDURATIONS")
                    prob.w.non_mixed_durations = parse_val(raw);
                else if (raw.substr(0, 9) == "FRONTLOAD") {
                    auto parts = split_csv(raw);
                    if (parts.size() >= 4) {
                        prob.w.fl_n_largest = std::stoi(parts[1]);
                        prob.w.fl_n_last = std::stoi(parts[2]);
                        prob.w.fl_penalty = std::stoi(parts[3]);
                    }
                }
                idx++;
            }
            continue;
        }
        else {
            idx++;
        }
    }

    prob.build_derived();
    return prob;
}

}