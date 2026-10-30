import subprocess, sys, os, re, time, concurrent.futures as cf, json
import pathlib, argparse
ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--out')
ap.add_argument('--only', nargs='*', help='module names like test_b12_s1')
ap.add_argument('--order-from', help='previous tests.json; run slowest modules first')
ARGS = ap.parse_args()
files = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--','tests/test_*.py'], text=True).split()
mods = sorted(f[:-3].replace('/','.') for f in files if not ARGS.only or f.split('/')[-1][:-3] in ARGS.only)
if ARGS.only:
    missing = [n for n in ARGS.only if not any(f.split('/')[-1][:-3] == n for f in files)]
    if missing:
        print('ERROR: no such test module(s):', missing, flush=True); sys.exit(2)
if not mods:
    print('ERROR: empty test list', flush=True); sys.exit(2)
_prev = {}
if ARGS.order_from and os.path.isfile(ARGS.order_from):
    try:
        _prev = {r[0]: r[3] for r in json.load(open(ARGS.order_from, encoding='utf-8'))}
    except Exception:
        _prev = {}
mods.sort(key=lambda m: -_prev.get(m, 0))
env = dict(os.environ); env.pop('PYTHONIOENCODING', None); env.pop('PYTHONPYCACHEPREFIX', None)
def run(m):
    t=time.time()
    r = subprocess.run([sys.executable,'-B','-m','unittest',m], env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=2800)
    if ARGS.out:
        logs = pathlib.Path(ARGS.out).parent / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        (logs / (m + '.log')).write_text(r.stdout + '\n--- STDERR ---\n' + r.stderr, encoding='utf-8', newline='\n')
    tail = r.stderr.strip().splitlines()[-6:]
    ran = re.search(r'Ran (\d+) test', r.stderr)
    out = (m, r.returncode, int(ran.group(1)) if ran else None, round(time.time()-t), tail[-1] if tail else '')
    print(m.split('.')[-1], out[1], out[2], out[3], out[4], flush=True)
    if r.returncode != 0:
        print('---- %s stderr tail ----\n%s' % (m, '\n'.join(r.stderr.strip().splitlines()[-40:])), flush=True)
    return out
W=ARGS.workers
with cf.ThreadPoolExecutor(W) as ex:
    res=list(ex.map(run, mods))
print('FAILED:', [r[0] for r in res if r[1]!=0])
if ARGS.out: json.dump(res, open(ARGS.out,'w'))
sys.exit(1 if any(r[1]!=0 for r in res) else 0)
