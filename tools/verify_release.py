"""Explicit offline release smoke suite, using the existing package verifier.

No model downloads, live APIs, full test discovery or historical episode replay.
Use --failed-from to retain the first run and rerun only failing modules.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    'test_real_chain_root_contract', 'test_real_chain_runner', 'test_real_chain_align',
    'test_real_chain_finish', 'test_real_chain_correct', 'test_real_llm_stages',
    'test_real_mimo', 'test_real_chain_llm', 'test_real_chain_mimo_full',
    'test_real_chain_bind', 'test_real_chain_host312', 'test_real_chain_launch312',
    'test_opt54_content', 'test_opt54_integration', 'test_opt54_performance',
    'test_opt54_memory', 'test_simplify_state', 'test_simplify_llm_content',
    'test_simplify_resource_scope', 'test_release_layout',
]
HOOK = '''import socket
def blocked(*args, **kwargs):
    raise RuntimeError("release verification socket blocked")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--failed-from', default=None)
    args = parser.parse_args()
    output = Path(args.out_dir).resolve() if args.out_dir else ROOT / 'verify-out' / uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for name in list(env):
        if (name.startswith(('REAL_', 'OPT54_')) or
                any(word in name.upper() for word in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD'))):
            env.pop(name, None)
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OPT54_EXPECT_BLOCKED='1')
    lock = os.environ.get('KOEAXIS_VERIFY_LOCK', str(ROOT / 'verify-out' / 'release.lock'))
    Path(lock).parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock, timeout=600), tempfile.TemporaryDirectory(prefix='koeaxis-offline-') as temporary:
        hook = Path(temporary) / 'sitecustomize.py'
        hook.write_text(HOOK, encoding='utf-8', newline='\n')
        env['PI_SCRATCH_DIR'] = temporary
        env['PYTHONPATH'] = os.pathsep.join([temporary, str(ROOT), env.get('PYTHONPATH', '')])
        commands = [[sys.executable, '-B', str(ROOT / 'tools/build_runtime.py')]]
        verify = [sys.executable, '-B', str(ROOT / 'tools/verify/verify_pkg.py'),
                  '--prefix', 'RELEASE-', '--workers', '1', '--out-dir', str(output)]
        if args.failed_from:
            verify += ['--failed-from', str(Path(args.failed_from).resolve())]
        else:
            verify += ['--tests', *TESTS]
        commands.append(verify)
        for label, command in zip(('build', 'tests'), commands):
            with (output / (label + '.log')).open('w', encoding='utf-8', newline='\n') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps({'step': label, 'exit_code': result.returncode, 'log': str(output / (label + '.log'))}), flush=True)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
