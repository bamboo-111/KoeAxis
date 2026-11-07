import json
import shutil
import uuid
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.temporary.real_chain_runner.runner import STAGES, run, sha_file


class FakeHost:
    def __init__(self, root, fail=None):
        self.root, self.fail, self.calls = root, fail, []

    def call(self, target, arguments):
        self.calls.append(target)
        if target == self.fail:
            raise RuntimeError('injected failure')
        (self.root / (target + '.json')).write_text('{}', encoding='utf-8')
        return {'outcome': 'returned', 'returned': {'report': {'status': 'completed'}}}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root = base / 'run'
        self.input = base / 'input'
        self.input.write_text('input', encoding='utf-8')
        self.inputs = {str(self.input): sha_file(self.input)}
        self.stages = [{'name': n, 'target': n, 'arguments': {}, 'outputs': [n + '.json']} for n in STAGES]
        self.socket_patch = patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden'))
        self.socket_patch.start()
        self.addCleanup(self.socket_patch.stop)

    def test_success_and_resume(self):
        host = FakeHost(self.root)
        self.assertEqual(run(host, self.stages, self.inputs, self.root)['stages'], len(STAGES))
        self.assertEqual(host.calls, list(STAGES))
        host.calls.clear()
        run(host, self.stages, self.inputs, self.root, resume=True)
        self.assertEqual(host.calls, [])

    def test_failure_stops_then_resumes(self):
        host = FakeHost(self.root, 'align')
        with self.assertRaises(RuntimeError):
            run(host, self.stages, self.inputs, self.root)
        self.assertEqual(host.calls, list(STAGES[:4]))
        host.fail, host.calls = None, []
        run(host, self.stages, self.inputs, self.root, resume=True)
        self.assertEqual(host.calls, list(STAGES[3:]))

    def test_changed_input_and_output_rejected(self):
        host = FakeHost(self.root)
        run(host, self.stages, self.inputs, self.root)
        self.input.write_text('changed', encoding='utf-8')
        with self.assertRaises(ValueError):
            run(host, self.stages, self.inputs, self.root, resume=True)
        self.input.write_text('input', encoding='utf-8')
        (self.root / 'prepare.json').write_text('tampered', encoding='utf-8')
        with self.assertRaises(ValueError):
            run(host, self.stages, self.inputs, self.root, resume=True)

    def test_incomplete_order_rejected(self):
        with self.assertRaises(ValueError):
            run(FakeHost(self.root), self.stages[:-1], self.inputs, self.root)


class AssemblyTests(unittest.TestCase):
    def test_real_host_fixed_assembly_rejects_missing_config(self):
        from host.runtime import Host
        project = Path(__file__).resolve().parents[1]
        entry = 'plan-entry-312.json' if sys.version_info[:2] == (3, 12) else 'plan-entry.json'
        registration = json.loads((project / 'module-registry/real_chain_runner' / entry).read_text(encoding='utf-8'))
        output = 'runs/experiments/CHAIN-READY-UNIT-' + uuid.uuid4().hex
        self.addCleanup(shutil.rmtree, project / output, True)
        host = Host(project, output, registration)
        try:
            task = {'run_id': 'bootstrap-run', 'binding_id': 'bootstrap'}
            result = host.invoke(host.envelope(task, registration['entry'], {'plan_id': 'CHAIN-READY-001', 'config': {}}))
            self.assertEqual(result['outcome'], 'raised')
            self.assertEqual(result['raised']['category'], 'InvalidConfig')
            self.assertIn('media', result['raised']['message'])
        finally:
            host.end_runs()


if __name__ == '__main__':
    unittest.main()
