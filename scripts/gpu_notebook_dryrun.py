#!/usr/bin/env python3
"""
Dry-run of the Colab notebook's benchmark logic against the local binary.
Catches issues (bad flags, missing variants, parse errors) before Colab.

Run from repo root:
    python3 scripts/gpu_notebook_dryrun.py
"""
import json
import os
import re
import subprocess
import sys
import time

REPO_DIR = os.path.abspath(os.path.dirname(__file__) + '/..')
os.chdir(REPO_DIR)

# Mirrors the notebook's VARIANT_PAIRS
VARIANT_PAIRS = [
    ('tabu_cached',    'tabu_cuda',        'tabu-iters',    100),
    ('sa_cached',      'sa_parallel_cuda', 'sa-iters',      500),
    ('alns_thompson',  'alns_cuda',        'alns-iters',    50),
    ('ga',             'ga_cuda',          'ga-iters',      30),
    ('abc',            'abc_cuda',         'abc-iters',     50),
    ('hho',            'hho_cuda',         'hho-iters',     30),
    ('woa',            'woa_cuda',         'woa-iters',     50),
]

INSTANCES = [4, 7]  # subset
SEEDS = [42]

def run(cmd, timeout=120):
    r = subprocess.run(cmd, shell=True, timeout=timeout, capture_output=True, text=True)
    return r

def check_binary():
    if not os.path.exists('cpp/build/exam_solver'):
        print('FAIL: cpp/build/exam_solver missing — run `make all HAVE_CUDA=1`')
        sys.exit(1)
    # Check if HAVE_CUDA linked
    r = run('ldd cpp/build/exam_solver | grep -E "cudart|delta_cuda"')
    has_cuda = bool(r.stdout.strip())
    print(f'  binary linked against CUDA libs: {has_cuda}')
    return has_cuda

def check_gpu_status(have_cuda):
    if not have_cuda:
        print('  (binary is CPU-only — skipping gpu=on check)')
        return
    checks = [
        ('tabu_cuda',        'tabu-iters'),
        ('sa_parallel_cuda', 'sa-iters'),
        ('alns_cuda',        'alns-iters'),
        ('ga_cuda',          'ga-iters'),
        ('abc_cuda',         'abc-iters'),
        ('hho_cuda',         'hho-iters'),
        ('woa_cuda',         'woa-iters'),
    ]
    all_on = True
    for algo, flag in checks:
        r = run(f'./cpp/build/exam_solver instances/exam_comp_set4.exam '
                f'--algo {algo} --seed 42 --{flag} 5 -v 2>&1 | grep -oE "gpu=(on|off)" | head -1')
        status = r.stdout.strip()
        marker = 'PASS' if 'on' in status else 'CHECK'
        print(f'    {marker}  {algo:22s} {status or "no gpu= line"}')
        if 'on' not in status: all_on = False
    return all_on

def solve(instance, algo, seed, flag, iters):
    cmd = (f'./cpp/build/exam_solver instances/exam_comp_set{instance}.exam '
           f'--algo {algo} --seed {seed} --{flag} {iters}')
    r = run(cmd, timeout=300)
    if r.returncode != 0:
        return {'error': f'exit {r.returncode}: {r.stderr[:100]}'}
    try:
        data = json.loads(r.stdout)
        out = data[0] if isinstance(data, list) and data else data
        return {
            'runtime_s': float(out.get('runtime', 0)),
            'soft': int(out.get('soft_penalty', -1)),
            'hard': int(out.get('hard_violations', -1)),
            'feasible': bool(out.get('feasible', False)),
        }
    except Exception as e:
        return {'error': f'parse: {e} | stdout_head={r.stdout[:200]}'}

def main():
    print('=== Local dry-run of notebook logic ===')

    print('\n[1] Binary check')
    have_cuda = check_binary()

    print('\n[2] gpu=on check')
    check_gpu_status(have_cuda)

    print('\n[3] Benchmark sweep (subset)')
    rows = []
    for inst in INSTANCES:
        for cpu, gpu, flag, iters in VARIANT_PAIRS:
            for seed in SEEDS:
                for variant in (cpu, gpu):
                    res = solve(inst, variant, seed, flag, iters)
                    err = res.get('error', '')
                    if err:
                        print(f'    ERROR set{inst} {variant} seed={seed}: {err}')
                        rows.append({'set': inst, 'algo': variant, 'seed': seed, 'error': err})
                    else:
                        print(f'    set{inst} {variant:22s} seed={seed} '
                              f'rt={res["runtime_s"]:.2f} soft={res["soft"]} feas={res["feasible"]}')
                        rows.append({'set': inst, 'algo': variant, 'seed': seed, **res})

    print('\n[4] Parity check (same-seed CPU vs GPU)')
    mismatches = 0
    matches = 0
    for inst in INSTANCES:
        for cpu, gpu, _, _ in VARIANT_PAIRS:
            for seed in SEEDS:
                cpu_row = next((r for r in rows if r['set'] == inst and r['algo'] == cpu and r['seed'] == seed and 'soft' in r), None)
                gpu_row = next((r for r in rows if r['set'] == inst and r['algo'] == gpu and r['seed'] == seed and 'soft' in r), None)
                if not cpu_row or not gpu_row: continue
                if cpu_row['soft'] == gpu_row['soft']:
                    matches += 1
                    print(f'    MATCH    set{inst} {cpu} vs {gpu} soft={cpu_row["soft"]}')
                else:
                    mismatches += 1
                    expected_drift = cpu in ['sa_cached']  # sa_parallel_cuda is experimental
                    tag = 'DRIFT-EXPECTED' if expected_drift else 'MISMATCH'
                    print(f'    {tag:15s} set{inst} {cpu}={cpu_row["soft"]} vs {gpu}={gpu_row["soft"]} diff={gpu_row["soft"]-cpu_row["soft"]}')

    print(f'\n=== Summary: {matches} matches, {mismatches} mismatches (some expected for sa_parallel_cuda) ===')

if __name__ == '__main__':
    main()
