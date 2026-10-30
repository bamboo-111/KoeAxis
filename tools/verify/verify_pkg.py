#!/usr/bin/env python
"""Single entry for per-package verification (docs/40 12, docs/76).

Runs, in parallel, only what a ticket needs:
  1. replay of the new plan(s) via replay_par.py
  2. the new/affected test files via tests_par.py
and lists stray `unit-host-*` byproduct directories (reports only; never deletes).

Full replay / all test files are for the final closeout: use --full.
Usage:
  python tools/verify/verify_pkg.py --prefix VER-B21- --plans b21-s1 --tests test_b21_s1
  python tools/verify/verify_pkg.py --prefix VER-FINAL- --full
"""
import argparse
import pathlib
import json
import os
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--prefix', required=True, help='output id prefix for replay, e.g. VER-B21-')
ap.add_argument('--plans', nargs='*', default=[], help='plan names to replay')
ap.add_argument('--tests', nargs='*', default=[], help='test modules, e.g. test_b21_s1')
ap.add_argument('--full', action='store_true', help='all plans and all test files (closeout only)')
ap.add_argument('--workers', type=int, default=min(os.cpu_count() or 8, 16))
ap.add_argument('--failed-from', default=None, help='previous --out-dir: rerun only the plans/tests that failed there')
ap.add_argument('--concurrent', action='store_true', help='EXPERIMENTAL: run replay and tests at once (they share runs/ and interfere; measured 17 spurious failures)')
ap.add_argument('--out-dir', default=None, help='where to write manifests (default: scratch-like ./verify-out)')
a = ap.parse_args()

prev = pathlib.Path(a.failed_from) if a.failed_from else None
if prev:
    try:
        a.plans = [x['plan'] for x in json.load(open(prev / 'replay.json', encoding='utf-8'))['plans'] if x.get('problem', True)] if (prev / 'replay.json').is_file() else []
        a.tests = [r[0].split('.')[-1] for r in json.load(open(prev / 'tests.json', encoding='utf-8')) if r[1] != 0] if (prev / 'tests.json').is_file() else []
    except Exception as e:
        ap.error('cannot read previous manifests in %s: %s' % (prev, e))
    a.full = False
    print('rerun failed only: plans=%s tests=%s' % (a.plans, a.tests), flush=True)
    if not (a.plans or a.tests):
        print('RESULT: OK (nothing failed in previous run)'); sys.exit(0)
if not a.full and not (a.plans or a.tests):
    ap.error('give --plans/--tests, or --full for closeout')

out = pathlib.Path(a.out_dir) if a.out_dir else ROOT / 'verify-out'
out.mkdir(parents=True, exist_ok=True)
rc = 0


def build():
    jobs = []
    if a.full or a.plans:
        cmd = [sys.executable, str(HERE / 'replay_par.py'), '--prefix', a.prefix,
               '--out', str(out / 'replay.json'), '--workers', str(a.workers)]
        if (out / 'replay.json').is_file():
            cmd += ['--order-from', str(out / 'replay.json')]
        if a.plans and not a.full:
            cmd += ['--only', *a.plans]
        jobs.append(('replay', cmd))
    if a.full or a.tests:
        cmd = [sys.executable, str(HERE / 'tests_par.py'), '--workers', str(a.workers),
               '--out', str(out / 'tests.json')]
        if (out / 'tests.json').is_file():
            cmd += ['--order-from', str(out / 'tests.json')]
        if a.tests and not a.full:
            cmd += ['--only', *a.tests]
        jobs.append(('tests', cmd))
    return jobs


jobs = build()
if a.failed_from and pathlib.Path(a.failed_from).resolve() != out.resolve():
    pass  # manifests of the rerun go to --out-dir; the previous ones stay untouched
if not a.concurrent or len(jobs) < 2:
    for name, cmd in jobs:
        print('+', ' '.join(cmd), flush=True)
        rc |= subprocess.call(cmd, cwd=ROOT) != 0
else:
    procs = []
    for name, cmd in jobs:
        print('+', ' '.join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, cwd=ROOT))
    for p in procs:
        rc |= p.wait() != 0

stray = sorted(p for p in (ROOT / 'runs').rglob('unit-host-*') if p.is_dir())
if stray:
    print('NOTE: %d stray unit-host-* byproduct dir(s); not evidence, remove before commit '
          '(skip dirs of still-running sessions):' % len(stray))
    for p in stray[:20]:
        print('  ', p.relative_to(ROOT))
if rc:
    print('RERUN ONLY FAILURES: python tools/verify/verify_pkg.py --prefix %sR- --failed-from %s --out-dir %s' % (a.prefix, out, out.parent / (out.name + '-rerun')))
print('RESULT:', 'FAIL' if rc else 'OK')
sys.exit(1 if rc else 0)
