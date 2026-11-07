"""Launcher checks: execute only a synthetic early failure, never heavy boundaries.

Run under the project's shared verification lock and socket audit guard.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / 'modules/temporary/real_chain_runner/launch.py'


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix='launch312-', dir=os.environ['PI_SCRATCH_DIR']))
        self.addCleanup(shutil.rmtree, self.temp, True)
        self.output = 'runs/experiments/CHAIN-READY-LAUNCH-TEST-' + uuid.uuid4().hex
        self.addCleanup(shutil.rmtree, ROOT / self.output, True)
        for name in ('model', 'cache', 'align'):
            (self.temp / name).mkdir()
        for name in ('input.wav', 'weights', 'enrollment.json', 'memory.json', 'mfa.exe', 'dictionary', 'acoustic'):
            (self.temp / name).write_bytes(b'synthetic')
        self.config = {'plan_id': 'CHAIN-READY-001', 'config': {
            'media': str(self.temp/'input.wav'), 'model': str(self.temp/'model'),
            'model_cache': str(self.temp/'cache'), 'enrollment': str(self.temp/'enrollment.json'),
            'weights': str(self.temp/'weights'), 'memory': str(self.temp/'memory.json'), 'device': 'cpu',
            'prepare_options': ['--unsupported-synthetic-option'], 'transcribe_options': [],
            'align_resources': {'cache': str(self.temp/'cache'), 'asr_model': str(self.temp/'model'),
                'align_model': str(self.temp/'align'), 'mfa_exe': str(self.temp/'mfa.exe'),
                'mfa_dictionary': str(self.temp/'dictionary'), 'mfa_acoustic': str(self.temp/'acoustic')}}}
        self.path = self.temp / 'config.json'
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.config), encoding='utf-8', newline='\n')

    def run_launcher(self, *args, online=False):
        env = dict(os.environ)
        # Retain the inherited socket guard and installed dependency search path.
        env['PYTHONPATH'] = os.environ.get('PYTHONPATH', os.environ['PI_SCRATCH_DIR'])
        for name in ('GEMINI_API_KEY', 'MIMO_API_KEY', 'REAL_LLM_STAGES_ONLINE', 'REAL_MIMO_ONLINE'):
            env.pop(name, None)
        if online:
            env.update({'GEMINI_API_KEY': 'synthetic', 'MIMO_API_KEY': 'synthetic',
                        'REAL_LLM_STAGES_ONLINE': '1', 'REAL_MIMO_ONLINE': '1'})
        return subprocess.run([sys.executable, '-B', str(LAUNCH), '--config', str(self.path), *args],
                              cwd=ROOT, env=env, text=True, encoding='utf-8', capture_output=True, timeout=180)

    def test_default_check_never_creates_output(self):
        result = self.run_launcher('--output', self.output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['mode'], 'check')
        self.assertFalse(json.loads(result.stdout)['models_or_network_tested'])
        self.assertFalse((ROOT/self.output).exists())

    def test_reject_invalid_arguments_without_creating_output(self):
        existing = self.output + '-existing'
        (ROOT / existing).mkdir(parents=True)
        self.addCleanup(shutil.rmtree, ROOT / existing, True)
        for args in [('--execute',), ('--output', '../bad'), ('--output', existing),
                     ('--execute', '--output', self.output), ('--unknown',)]:
            result = self.run_launcher(*args)
            self.assertNotEqual(result.returncode, 0, args)
            self.assertFalse((ROOT/self.output).exists())
        self.config['config']['retry'] = True
        self.save()
        result = self.run_launcher('--output', self.output)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((ROOT/self.output).exists())

    def test_real_host_business_failure_is_nonzero(self):
        # prepare rejects the unsupported option before importing legacy stages or using tools/models.
        result = self.run_launcher('--execute', '--output', self.output, online=True)
        self.assertEqual(result.returncode, 1, (result.stdout, result.stderr))
        summary = json.loads(result.stdout)
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['category'], 'MediaStageFailed')
        state = json.loads((ROOT/self.output/'fixed-chain-state.json').read_text(encoding='utf-8'))
        self.assertEqual(state['attempts'][0]['status'], 'failed')
        self.assertEqual(state['attempts'][0]['stages'], [])
        self.assertFalse((ROOT/self.output/'work/export/subtitles.srt').exists())
        print('REAL_HOST_EARLY_FAILURE ' + json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
