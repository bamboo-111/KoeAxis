"""Source-checkout portability checks; no audit repository or Git objects needed."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseLayout(unittest.TestCase):
    def test_frozen_baseline_identities(self):
        rows = json.loads((ROOT / 'tests/fixtures/baselines/index.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest(), row['sha256'])

    def test_portable_templates(self):
        profiles = list((ROOT / 'runtime/registry-templates').rglob('contracts-*.json'))
        self.assertTrue(profiles)
        for path in profiles:
            data = json.loads(path.read_text(encoding='utf-8'))
            if 'legacy_sources' not in data:
                continue
            self.assertEqual(data['legacy_sources']['root'], 'compat/legacy')
            self.assertEqual(data['resource_pack'], [])
            for item in data['legacy_sources']['files']:
                self.assertEqual(hashlib.sha256((ROOT / 'compat/legacy' / item['path']).read_bytes()).hexdigest(), item['sha256'])

    def test_legacy_import_is_repository_local(self):
        code = 'import sys; sys.path.insert(0, sys.argv[1]); import qwen_asr; print(qwen_asr.__file__)'
        result = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT / 'compat/legacy')],
                                cwd=ROOT.parent, text=True, encoding='utf-8', capture_output=True, check=True)
        self.assertEqual(Path(result.stdout.strip()).resolve(), (ROOT / 'compat/legacy/qwen_asr/__init__.py').resolve())

    def test_generated_registry_and_sources_verify(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/check_runtime.py')],
                                cwd=ROOT.parent, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'ok')

    def test_offline_guard_is_inherited(self):
        if os.environ.get('OPT54_EXPECT_BLOCKED') != '1':
            self.skipTest('run via tools/verify_release.py for network guard')
        for fn, args in ((socket.socket.connect, (None, ('127.0.0.1', 9))),
                         (socket.socket.connect_ex, (None, ('127.0.0.1', 9))),
                         (socket.create_connection, (('127.0.0.1', 9),))):
            with self.assertRaisesRegex(RuntimeError, 'socket blocked'):
                fn(*args)
        code = 'import socket; socket.create_connection(("127.0.0.1",9))'
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, encoding='utf-8')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('release verification socket blocked', result.stderr)


if __name__ == '__main__':
    unittest.main()
