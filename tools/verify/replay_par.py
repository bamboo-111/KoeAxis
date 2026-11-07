import json, os, subprocess, hashlib, pathlib, sys, platform, re, shutil
ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
dirty = subprocess.check_output(['git', 'status', '--short'], text=True).strip()
pass
import argparse
ap = argparse.ArgumentParser(description='Parallel replay of committed plans against committed evidence')
ap.add_argument('--prefix', required=True, help='new output id prefix, e.g. VER-001-')
ap.add_argument('--out', required=True, help='manifest json path')
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--only', nargs='*', help='restrict to these plan names')
ap.add_argument('--order-from', help='previous replay.json; run slowest plans first')
ap.add_argument('--refs', default=str(pathlib.Path(__file__).with_name('plan-refs.json')))
ARGS = ap.parse_args()
PLANS = [(x['plan'], x['reference']) for x in json.load(open(ARGS.refs, encoding='utf-8')) if not ARGS.only or x['plan'] in ARGS.only]
import time
_prev = {}
if ARGS.order_from and pathlib.Path(ARGS.order_from).is_file():
    try:
        _prev = {x['plan']: x.get('seconds', 0) for x in json.load(open(ARGS.order_from, encoding='utf-8'))['plans']}
    except Exception:
        _prev = {}
PLANS.sort(key=lambda it: -_prev.get(it[0], 0))
ENV_FILES = {'snapshot.json', 'report.json', 'resource-validation.json', 'reference-execution.json', 'reference-executions.json', 'resource-pack.json'}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def walk(a, b, p, out):
    if type(a) is not type(b):
        out.append(p); return
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b: out.append(p + '/' + k)
            else: walk(a[k], b[k], p + '/' + k, out)
    elif isinstance(a, list):
        if len(a) != len(b): out.append(p + '[len]'); return
        for x, y in zip(a, b): walk(x, y, p + '[]', out)
    elif a != b:
        out.append(p)


# ffmpeg stderr embeds the process pointer (`@` + 16 hex, or `@ 0x...`). ASLR changes it every run.
_FFMPEG_PTR = re.compile(br'@ (?:0x[0-9a-fA-F]+|[0-9a-fA-F]{16})(?![0-9a-fA-F])')


def comparable(name, data):
    if name == 'real-tool-executions.json':
        return _FFMPEG_PTR.sub(b'@ <addr>', data)
    return data

env = dict(os.environ)
for k in ('PYTHONIOENCODING', 'PYTHONPYCACHEPREFIX'):
    env.pop(k, None)
import concurrent.futures as cf
WORKERS=ARGS.workers
def run1(item):
    try:
        return run1_inner(item)
    except Exception as e:  # one broken plan must not discard the other results
        print(item[0], 'ERROR', type(e).__name__, e, flush=True)
        return {'plan': item[0], 'exit_code': -1, 'stderr_bytes': 0, 'failed': 1,
                'business_files': [], 'error': '%s: %s' % (type(e).__name__, e)}


def run1_inner(item):
    t0 = time.time()
    plan, ref = item
    outid = ARGS.prefix + plan
    outdir = ROOT / 'runs' / 'experiments' / outid
    if outdir.exists():  # stale leftover of a crashed earlier run with the same prefix
        shutil.rmtree(outdir)
    r = subprocess.run([sys.executable, '-B', '-m', 'host', '--plan', 'experiments/specifications/%s.json' % plan,
                        '--output', 'runs/experiments/' + outid], env=env, capture_output=True, timeout=1800)
    ev = outdir / 'evidence'
    refdir = ROOT / 'runs' / 'experiments' / ref
    if not (refdir / 'report.json').exists() and (refdir / 'evidence' / 'report.json').exists():
        refdir = refdir / 'evidence'
    if (refdir / 'evidence').is_dir() and not (refdir / 'report.json').exists():
        refdir = refdir / 'evidence'
    rep = json.loads((ev / 'report.json').read_text('utf-8'))
    rref = json.loads((refdir / 'report.json').read_text('utf-8'))
    business = []
    env_diffs = {}
    for f in sorted(os.listdir(ev)):
        fp = ev / f
        if not fp.is_file():
            continue
        rp = refdir / f
        if not rp.is_file():
            business.append({'file': f, 'result': 'no-committed-counterpart'}); continue
        raw_new, raw_old = fp.read_bytes(), rp.read_bytes()
        same = comparable(f, raw_new) == comparable(f, raw_old)
        if f in ENV_FILES:
            if not same:
                try:
                    o = []
                    walk(json.loads(fp.read_text('utf-8')), json.loads(rp.read_text('utf-8')), '', o)
                    env_diffs[f] = sorted(set(re.sub(r'\[\d+\]', '[]', x) for x in o))
                except Exception as e:
                    env_diffs[f] = ['unparsable:' + type(e).__name__]
        else:
            row = {'file': f, 'sha256': sha(fp), 'identical_to_committed': same}
            if f == 'real-tool-executions.json' and raw_new != raw_old:
                row['normalized'] = 'ffmpeg-pointer-address'
            business.append(row)
    res = ({
        'plan': plan, 'committed_reference': str(refdir.relative_to(ROOT)).replace('\\', '/'),
        'exit_code': r.returncode, 'stderr_bytes': len(r.stderr),
        'status': rep.get('status'), 'passed': rep.get('passed'), 'failed': rep.get('failed'),
        'case_count': rep.get('case_count'),
        'committed_status': rref.get('status'), 'committed_passed': rref.get('passed'), 'committed_failed': rref.get('failed'),
        'models_run': rep.get('models_run'),
        'business_files': business,
        'identity_or_environment_diffs': env_diffs,
        'seconds': round(time.time() - t0),
    })
    print(plan, r.returncode, rep.get('passed'), rep.get('failed'), res['seconds'], flush=True)
    shutil.rmtree(outdir)
    return res
with cf.ThreadPoolExecutor(WORKERS) as ex:
    results = list(ex.map(run1, PLANS))
manifest = {
    'id': 'B01-REL-001', 'kind': 'controlled-replay', 'head': head, 'python': platform.python_version(),
    'host_version': 'see host.runtime.HOST_VERSION', 'plans': results,
    'rule': 'business files must be byte-identical to committed evidence, except real-tool-executions.json may differ only by ffmpeg pointer addresses (@ + 16 hex, or @ 0x...), replaced with <addr> before compare; snapshot/report/resource files may differ only in environment/host identity fields (listed); status/passed/failed must equal committed',
}
(pathlib.Path(ARGS.out)).write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
bad = [x['plan'] for x in results if x['exit_code'] != 0 or x['stderr_bytes'] or x['failed'] or any(b.get('identical_to_committed') is False for b in x['business_files'])]
for x in results:
    x['problem'] = x['plan'] in bad
(pathlib.Path(ARGS.out)).write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
print('PROBLEMS', bad)
