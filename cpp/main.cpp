/*
 * main.cpp — Entry point for the C++ exam solver
 *
 * Compile:  g++ -O3 -std=c++20 -o exam_solver main.cpp
 * Usage:    ./exam_solver <file.exam> [options]
 *
 * All algorithm logic lives in the header files:
 *   models.h     — data structures
 *   parser.h     — .exam file parser
 *   evaluator.h  — full + delta evaluation
 *   greedy.h     — greedy heuristic
 *   tabu.h       — tabu search
 *   hho.h        — Harris Hawks Optimization
 */

#include "models.h"
#include "parser.h"
#include "evaluator.h"
#include "greedy.h"
#include "tabu.h"
#include "hho.h"
#include "kempe.h"
#include "sa.h"
#include "alns.h"
#include "gd.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// ── JSON output ─────────────────────────────────────────────

static void write_result_json(ostream& out, const AlgoResult& r) {
    const auto& ev = r.eval;
    out << "  {\n"
        << "    \"algorithm\": \"" << r.algorithm << "\",\n"
        << "    \"runtime\": " << fixed << setprecision(6) << r.runtime_sec << ",\n"
        << "    \"iterations\": " << r.iterations << ",\n"
        << "    \"feasible\": " << (ev.feasible() ? "true" : "false") << ",\n"
        << "    \"hard_violations\": " << ev.hard() << ",\n"
        << "    \"soft_penalty\": " << ev.soft() << ",\n"
        << "    \"conflicts\": " << ev.conflicts << ",\n"
        << "    \"room_occupancy\": " << ev.room_occupancy << ",\n"
        << "    \"period_utilisation\": " << ev.period_utilisation << ",\n"
        << "    \"period_related\": " << ev.period_related << ",\n"
        << "    \"room_related\": " << ev.room_related << ",\n"
        << "    \"two_in_a_row\": " << ev.two_in_a_row << ",\n"
        << "    \"two_in_a_day\": " << ev.two_in_a_day << ",\n"
        << "    \"period_spread\": " << ev.period_spread << ",\n"
        << "    \"non_mixed_durations\": " << ev.non_mixed_durations << ",\n"
        << "    \"front_load\": " << ev.front_load << ",\n"
        << "    \"period_penalty\": " << ev.period_penalty << ",\n"
        << "    \"room_penalty\": " << ev.room_penalty << "\n"
        << "  }";
}

static void write_solution_file(const Solution& sol, const string& filepath) {
    ofstream f(filepath);
    for (int e = 0; e < sol.n_e; e++) {
        if (sol.period_of[e] >= 0)
            f << sol.period_of[e] << ", " << sol.room_of[e] << "\n";
        else
            f << "-1, -1\n";
    }
}

// ── Main ────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    string filepath;
    string algo         = "all";
    int limit           = 0;
    int tabu_iters      = 200;
    int tabu_tenure     = 15;
    int tabu_patience   = 50;
    int hho_pop         = 30;
    int hho_iters       = 100;
    int sa_iters        = 5000;
    int kempe_iters     = 3000;
    int alns_iters      = 2000;
    int gd_iters        = 5000;
    int seed            = 42;
    string output_dir   = "results";
    bool verbose        = false;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if      (arg == "--algo"           && i+1 < argc) algo          = argv[++i];
        else if (arg == "--limit"          && i+1 < argc) limit         = stoi(argv[++i]);
        else if (arg == "--tabu-iters"     && i+1 < argc) tabu_iters    = stoi(argv[++i]);
        else if (arg == "--tabu-tenure"    && i+1 < argc) tabu_tenure   = stoi(argv[++i]);
        else if (arg == "--tabu-patience"  && i+1 < argc) tabu_patience = stoi(argv[++i]);
        else if (arg == "--hho-pop"        && i+1 < argc) hho_pop       = stoi(argv[++i]);
        else if (arg == "--hho-iters"      && i+1 < argc) hho_iters     = stoi(argv[++i]);
        else if (arg == "--sa-iters"       && i+1 < argc) sa_iters      = stoi(argv[++i]);
        else if (arg == "--kempe-iters"    && i+1 < argc) kempe_iters   = stoi(argv[++i]);
        else if (arg == "--alns-iters"     && i+1 < argc) alns_iters    = stoi(argv[++i]);
        else if (arg == "--gd-iters"       && i+1 < argc) gd_iters      = stoi(argv[++i]);
        else if (arg == "--seed"           && i+1 < argc) seed          = stoi(argv[++i]);
        else if (arg == "--output-dir"     && i+1 < argc) output_dir    = argv[++i];
        else if (arg == "--verbose" || arg == "-v")        verbose       = true;
        else if (arg[0] != '-')                            filepath      = arg;
    }

    if (filepath.empty()) {
        cerr << "Usage: exam_solver <file.exam> [options]\n"
             << "\nOptions:\n"
             << "  --algo greedy|tabu|hho|kempe|sa|alns|gd|all  Algorithm (default: all)\n"
             << "  --limit N                    Load only first N exams (0=all)\n"
             << "  --tabu-iters N               Tabu max iterations (default: 200)\n"
             << "  --tabu-tenure N              Tabu tenure (default: 15)\n"
             << "  --tabu-patience N            Tabu patience (default: 50)\n"
             << "  --hho-pop N                  HHO population size (default: 30)\n"
             << "  --hho-iters N                HHO max iterations (default: 100)\n"
             << "  --sa-iters N                 SA iterations (default: 5000)\n"
             << "  --kempe-iters N              Kempe Chain iterations (default: 3000)\n"
             << "  --alns-iters N               ALNS iterations (default: 2000)\n"
             << "  --gd-iters N                 Great Deluge iterations (default: 5000)\n"
             << "  --seed N                     Random seed (default: 42)\n"
             << "  --output-dir DIR             Output directory (default: results)\n"
             << "  -v, --verbose                Print progress to stderr\n";
        return 1;
    }

    // ── Parse ──
    auto prob = parser::parse_exam_file(filepath, limit);

    if (verbose) {
        cerr << "Problem: " << prob.n_e() << " exams, "
             << prob.n_p() << " periods, " << prob.n_r() << " rooms\n"
             << "Weights: 2row=" << prob.w.two_in_a_row
             << " 2day=" << prob.w.two_in_a_day
             << " spread=" << prob.w.period_spread
             << " mixed=" << prob.w.non_mixed_durations
             << " front=(" << prob.w.fl_n_largest << ","
             << prob.w.fl_n_last << "," << prob.w.fl_penalty << ")\n";
    }

    // ── Run algorithms ──
    vector<AlgoResult> results;
    string ne_str = to_string(prob.n_e());

    if (algo == "all" || algo == "greedy") {
        auto r = solve_greedy(prob, verbose);
        write_solution_file(r.sol, output_dir + "/solution_greedy_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (algo == "all" || algo == "tabu") {
        auto r = solve_tabu(prob, tabu_iters, tabu_tenure, tabu_patience, seed, verbose);
        write_solution_file(r.sol, output_dir + "/solution_tabu_search_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (algo == "all" || algo == "hho") {
        auto r = solve_hho(prob, hho_pop, hho_iters, seed, verbose);
        write_solution_file(r.sol, output_dir + "/solution_hho_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (algo == "all" || algo == "kempe") {
        auto r = solve_kempe(prob, kempe_iters, seed, verbose);
        write_solution_file(r.sol, output_dir + "/solution_kempe_chain_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (algo == "all" || algo == "sa") {
        auto r = solve_sa(prob, sa_iters, 0.0, 0.9995, seed, verbose);
        write_solution_file(r.sol, output_dir + "/solution_simulated_annealing_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (algo == "all" || algo == "alns") {
        auto r = solve_alns(prob, alns_iters, 0.15, seed, verbose);
        write_solution_file(r.sol, output_dir + "/solution_alns_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (algo == "all" || algo == "gd") {
        auto r = solve_great_deluge(prob, gd_iters, 0.0, seed, verbose);
        write_solution_file(r.sol, output_dir + "/solution_great_deluge_" + ne_str + ".sln");
        results.push_back(move(r));
    }

    // ── JSON to stdout ──
    cout << "[\n";
    for (int i = 0; i < (int)results.size(); i++) {
        write_result_json(cout, results[i]);
        if (i + 1 < (int)results.size()) cout << ",";
        cout << "\n";
    }
    cout << "]\n";

    return 0;
}