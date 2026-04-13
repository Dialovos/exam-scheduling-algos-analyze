/*
 * Compile:  g++ -O3 -std=c++20 -o exam_solver main.cpp
 * Usage:    ./exam_solver <file.exam> [options]
 *
 * Parses CLI args, runs selected algorithms, writes JSON results to stdout
 * and .sln files to output-dir/solutions/.
 */

#include "models.h"
#include "parser.h"
#include "evaluator.h"
#include "greedy.h"
#include "tabu.h"
#include "kempe.h"
#include "sa.h"
#include "alns.h"
#include "gd.h"
#include "abc.h"
#include "ga.h"
#include "lahc.h"
#include "woa.h"
#include "cpsat.h"
#include "vns.h"
#include "feasibility.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
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
    // HHO archived — replaced by WOA (pending)
    int sa_iters        = 5000;
    int kempe_iters     = 3000;
    int alns_iters      = 2000;
    int gd_iters        = 5000;
    int abc_pop         = 30;
    int abc_iters       = 3000;
    int ga_pop          = 50;
    int ga_iters        = 500;
    int lahc_iters      = 5000;
    int lahc_list       = 0;
    int woa_pop         = 25;
    int woa_iters       = 3000;
    double cpsat_time   = 60.0;
    int vns_iters       = 5000;
    int vns_budget      = 0;
    int seed            = 42;
    string output_dir   = "results";
    string init_solution_path;
    bool verbose        = false;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if      (arg == "--algo"           && i+1 < argc) algo          = argv[++i];
        else if (arg == "--limit"          && i+1 < argc) limit         = stoi(argv[++i]);
        else if (arg == "--tabu-iters"     && i+1 < argc) tabu_iters    = stoi(argv[++i]);
        else if (arg == "--tabu-tenure"    && i+1 < argc) tabu_tenure   = stoi(argv[++i]);
        else if (arg == "--tabu-patience"  && i+1 < argc) tabu_patience = stoi(argv[++i]);
        // HHO args removed (archived)
        else if (arg == "--sa-iters"       && i+1 < argc) sa_iters      = stoi(argv[++i]);
        else if (arg == "--kempe-iters"    && i+1 < argc) kempe_iters   = stoi(argv[++i]);
        else if (arg == "--alns-iters"     && i+1 < argc) alns_iters    = stoi(argv[++i]);
        else if (arg == "--gd-iters"       && i+1 < argc) gd_iters      = stoi(argv[++i]);
        else if (arg == "--abc-pop"        && i+1 < argc) abc_pop       = stoi(argv[++i]);
        else if (arg == "--abc-iters"      && i+1 < argc) abc_iters     = stoi(argv[++i]);
        else if (arg == "--ga-pop"         && i+1 < argc) ga_pop        = stoi(argv[++i]);
        else if (arg == "--ga-iters"       && i+1 < argc) ga_iters      = stoi(argv[++i]);
        else if (arg == "--lahc-iters"     && i+1 < argc) lahc_iters    = stoi(argv[++i]);
        else if (arg == "--lahc-list"      && i+1 < argc) lahc_list     = stoi(argv[++i]);
        else if (arg == "--woa-pop"       && i+1 < argc) woa_pop       = stoi(argv[++i]);
        else if (arg == "--woa-iters"     && i+1 < argc) woa_iters     = stoi(argv[++i]);
        else if (arg == "--cpsat-time"    && i+1 < argc) cpsat_time    = stod(argv[++i]);
        else if (arg == "--vns-iters"    && i+1 < argc) vns_iters     = stoi(argv[++i]);
        else if (arg == "--vns-budget"   && i+1 < argc) vns_budget    = stoi(argv[++i]);
        else if (arg == "--seed"           && i+1 < argc) seed          = stoi(argv[++i]);
        else if (arg == "--output-dir"     && i+1 < argc) output_dir    = argv[++i];
        else if (arg == "--init-solution"  && i+1 < argc) init_solution_path = argv[++i];
        else if (arg == "--verbose" || arg == "-v")        verbose       = true;
        else if (arg[0] != '-')                            filepath      = arg;
    }

    if (filepath.empty()) {
        cerr << "Usage: exam_solver <file.exam> [options]\n"
             << "\nOptions:\n"
             << "  --algo ALGO[,ALGO,...]       Algorithm(s): greedy,feasibility,tabu,kempe,sa,alns,gd,abc,ga,lahc,woa,cpsat,vns,all (default: all)\n"
             << "  --limit N                    Load only first N exams (0=all)\n"
             << "  --tabu-iters N               Tabu max iterations (default: 200)\n"
             << "  --tabu-tenure N              Tabu tenure (default: 15)\n"
             << "  --tabu-patience N            Tabu patience (default: 50)\n"
             // HHO help removed
             << "  --sa-iters N                 SA iterations (default: 5000)\n"
             << "  --kempe-iters N              Kempe Chain iterations (default: 3000)\n"
             << "  --alns-iters N               ALNS iterations (default: 2000)\n"
             << "  --gd-iters N                 Great Deluge iterations (default: 5000)\n"
             << "  --abc-pop N                  ABC colony size (default: 30)\n"
             << "  --abc-iters N                ABC iterations (default: 3000)\n"
             << "  --ga-pop N                   GA population size (default: 50)\n"
             << "  --ga-iters N                 GA generations (default: 500)\n"
             << "  --lahc-iters N               LAHC iterations (default: 5000)\n"
             << "  --lahc-list N                LAHC list length (0=auto) (default: 0)\n"
             << "  --woa-pop N                  WOA population size (default: 25)\n"
             << "  --woa-iters N                WOA iterations (default: 3000)\n"
             << "  --cpsat-time SEC             CP-SAT time limit in seconds (default: 60)\n"
             << "  --vns-iters N                VNS iterations (default: 5000)\n"
             << "  --vns-budget N               VNS scan budget per VND level (0=auto)\n"
             << "  --seed N                     Random seed (default: 42)\n"
             << "  --output-dir DIR             Output directory (default: results)\n"
             << "  --init-solution PATH         Warm-start from .sln file\n"
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

    // ── Load initial solution (warm-start) ──
    Solution init_sol;
    Solution* init_sol_ptr = nullptr;
    if (!init_solution_path.empty()) {
        init_sol.init(prob);
        ifstream f(init_solution_path);
        string line; int eid = 0;
        while (getline(f, line) && eid < prob.n_e()) {
            auto comma = line.find(',');
            if (comma != string::npos) {
                int pid = stoi(line.substr(0, comma));
                int rid = stoi(line.substr(comma + 1));
                if (pid >= 0 && rid >= 0) init_sol.assign(eid, pid, rid);
            }
            eid++;
        }
        init_sol_ptr = &init_sol;
        if (verbose) cerr << "Loaded init solution from " << init_solution_path << endl;
    }

    // ── Parse algo string into a set (supports "all", "sa", "sa,gd,tabu", etc.) ──
    set<string> algos;
    {
        istringstream ss(algo);
        string token;
        while (getline(ss, token, ','))
            if (!token.empty()) algos.insert(token);
    }
    bool run_all = algos.count("all") > 0;
    auto want = [&](const string& a) { return run_all || algos.count(a) > 0; };

    // ── Run algorithms ──
    vector<AlgoResult> results;
    string ne_str = to_string(prob.n_e());
    string sln_dir = output_dir + "/solutions";
    filesystem::create_directories(sln_dir);

    // When running multiple algos, compute greedy once and share as init solution
    // instead of each algorithm rebuilding it independently.
    const Solution* shared_init = init_sol_ptr;
    AlgoResult greedy_result;
    if (algos.size() > 1 || run_all || want("greedy")) {
        greedy_result = solve_greedy(prob, verbose);
        write_solution_file(greedy_result.sol, sln_dir + "/solution_greedy_" + ne_str + ".sln");
        if (want("greedy"))
            results.push_back(greedy_result);
        if (!init_sol_ptr)
            shared_init = &greedy_result.sol;
    }
    // ── Feasibility pre-processor: when greedy is infeasible, run targeted
    //    feasibility solver and use its output as shared_init for all algorithms.
    AlgoResult feasibility_result;
    if (shared_init && !shared_init->period_of.empty()) {
        FastEvaluator fe_check(prob);
        int greedy_hard = fe_check.count_hard_fast(*shared_init);
        if (greedy_hard > 0) {
            if (verbose)
                cerr << "Greedy infeasible (hard=" << greedy_hard
                     << "), running feasibility solver...\n";
            feasibility_result = solve_feasibility(prob, seed, verbose, shared_init);
            write_solution_file(feasibility_result.sol,
                                sln_dir + "/solution_feasibility_" + ne_str + ".sln");
            if (want("feasibility"))
                results.push_back(feasibility_result);
            // Use feasibility output as shared init for all subsequent algorithms
            if (feasibility_result.eval.hard() < greedy_hard)
                shared_init = &feasibility_result.sol;
        }
    }

    if (want("tabu")) {
        auto r = solve_tabu(prob, tabu_iters, tabu_tenure, tabu_patience, seed, verbose, shared_init);
        write_solution_file(r.sol, sln_dir + "/solution_tabu_search_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    // HHO block removed (archived to cpp/src/archive/hho.h)
    if (want("kempe")) {
        auto r = solve_kempe(prob, kempe_iters, seed, verbose, shared_init);
        write_solution_file(r.sol, sln_dir + "/solution_kempe_chain_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (want("sa")) {
        auto r = solve_sa(prob, sa_iters, 0.0, 0.9995, seed, verbose, shared_init);
        write_solution_file(r.sol, sln_dir + "/solution_simulated_annealing_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (want("alns")) {
        auto r = solve_alns(prob, alns_iters, 0.15, seed, verbose, shared_init);
        write_solution_file(r.sol, sln_dir + "/solution_alns_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (want("gd")) {
        auto r = solve_great_deluge(prob, gd_iters, 0.0, seed, verbose, shared_init);
        write_solution_file(r.sol, sln_dir + "/solution_great_deluge_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (want("abc")) {
        auto r = solve_abc(prob, abc_pop, abc_iters, 0, seed, verbose, shared_init);
        write_solution_file(r.sol, sln_dir + "/solution_abc_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (want("ga")) {
        auto r = solve_ga(prob, ga_pop, ga_iters, 0.8, 0.15, seed, verbose, shared_init);
        write_solution_file(r.sol, sln_dir + "/solution_genetic_algorithm_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (want("lahc")) {
        auto r = solve_lahc(prob, lahc_iters, lahc_list, seed, verbose, shared_init);
        write_solution_file(r.sol, sln_dir + "/solution_lahc_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (want("woa")) {
        auto r = solve_woa(prob, woa_pop, woa_iters, seed, verbose, shared_init);
        write_solution_file(r.sol, sln_dir + "/solution_woa_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (want("cpsat")) {
        auto r = solve_cpsat(prob, cpsat_time, seed, verbose, shared_init);
        write_solution_file(r.sol, sln_dir + "/solution_cpsat_" + ne_str + ".sln");
        results.push_back(move(r));
    }
    if (want("vns")) {
        auto r = solve_vns(prob, vns_iters, vns_budget, seed, verbose, shared_init);
        write_solution_file(r.sol, sln_dir + "/solution_vns_" + ne_str + ".sln");
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