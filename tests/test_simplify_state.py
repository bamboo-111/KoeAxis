"""Fixed-root state projections and failure semantics, using synthetic calls only."""
import json
import shutil
import subprocess
import types
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from host.contracts import ModuleError
from modules.temporary.real_chain_runner import operations as ops
from tests.test_real_chain_root_contract import RecordingApi, STAGES, make_config


class FrozenDateTime:
    @classmethod
    def now(cls, tz):
        return datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


class StateTests(unittest.TestCase):
    def test_projection_and_partial_failure(self):
        for count in (0, 1):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                attempt = {'stages': [{'name': 'prepare', 'artifacts': {}}] * count}
                with patch.object(ops, 'datetime', FrozenDateTime):
                    ops._progress(root, attempt, 'failed')
                expected = {'stage': 'fixed-chain', 'status': 'failed', 'done': count,
                            'total': 12, 'current': 'prepare' if count else None,
                            'updated_at': '2020-01-02T03:04:05+00:00', 'summary': 'fixed chain failed'}
                data = (json.dumps(expected, ensure_ascii=True, indent=2, allow_nan=False) + '\n').encode()
                self.assertEqual((root / 'work/progress.json').read_bytes(), data)
                self.assertEqual((root / 'progress.json').read_bytes(), data)
                original = ops._save_state
                calls = []
                def fail_second(path, value):
                    calls.append(path.relative_to(root).as_posix())
                    if len(calls) == 2:
                        raise OSError('second write')
                    original(path, value)
                with patch.object(ops, 'datetime', FrozenDateTime), patch.object(ops, '_save_state', fail_second):
                    with self.assertRaisesRegex(OSError, 'second write'):
                        ops._progress(root, attempt, 'completed')
                self.assertEqual(calls, ['work/progress.json', 'progress.json'])
                self.assertEqual(json.loads((root / 'work/progress.json').read_bytes())['status'], 'completed')
                self.assertEqual((root / 'progress.json').read_bytes(), data)

    def test_checkpoint_failure_prevents_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            attempt = {'status': 'running', 'stages': []}
            state = {'attempts': [attempt]}
            owner = ops._RootState(root, root, root / 'fixed-chain-state.json', state, attempt)
            with patch.object(ops, '_save_state', side_effect=OSError('checkpoint')):
                with self.assertRaisesRegex(OSError, 'checkpoint'):
                    owner.finish('failed')
            self.assertEqual(attempt['status'], 'failed')
            self.assertFalse((root / 'work/progress.json').exists())

    def test_terminal_and_retry_from_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(tmp)
            config['performance_enabled'] = False
            root = Path(tmp) / 'out'
            args = {'plan_id': 'CHAIN-READY-001', 'config': config}
            api = RecordingApi(root, config, 'stage.transcribe')
            with patch.object(ops, 'datetime', FrozenDateTime):
                with self.assertRaises(ModuleError):
                    ops.run_root(args, api)
                state = json.loads((root / 'fixed-chain-state.json').read_text(encoding='utf-8'))
                self.assertEqual(state['attempts'][-1]['status'], 'failed')
                self.assertEqual(json.loads((root / 'progress.json').read_bytes())['done'], 1)
                self.assertEqual((root / 'progress.json').read_bytes(), (root / 'work/progress.json').read_bytes())
                retry = dict(config, retry=True)
                second = RecordingApi(root / 'attempts/0002', retry)
                second.snapshot = lambda: {'environment': {'output': str(root)}}
                ops.run_root({'plan_id': 'CHAIN-READY-001', 'config': retry}, second)
                self.assertEqual(tuple(row[0] for row in second.calls), STAGES)
                progress = root / 'attempts/0002/progress.json'
                self.assertEqual(json.loads(progress.read_bytes())['status'], 'completed')
                self.assertEqual(progress.read_bytes(), (root / 'attempts/0002/work/progress.json').read_bytes())
                with self.assertRaises(ModuleError):
                    ops.run_root({'plan_id': 'CHAIN-READY-001', 'config': retry}, second)

    def test_baseline_bytes_order_and_errors(self):
        """Same paths, fixed clock; compare full bytes without dropping fields."""
        source = (ops.ROOT / 'tests/fixtures/baselines/813e9d2/modules/temporary/real_chain_runner/operations.py').read_bytes()
        old = types.ModuleType('_fixed_root_baseline')
        old.__file__ = ops.__file__
        exec(compile(source, ops.__file__, 'exec'), old.__dict__)

        def exercise(module, root, config, fail_stage, fail_write, retry, mutate):
            events, errors, snapshots = [], [], []
            original = module._save_state

            def record(path, value):
                events.append((path.relative_to(root).as_posix(),
                               json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)))
                if len(events) == fail_write:
                    raise OSError('injected write ' + str(fail_write))
                original(path, value)

            def run(api, cfg):
                try:
                    module.run_root({'plan_id': 'CHAIN-READY-001', 'config': cfg}, api)
                except (ModuleError, OSError) as exc:
                    errors.append((type(exc).__name__, str(exc)))
                snapshots.append({p.relative_to(root).as_posix(): p.read_bytes()
                                  for p in sorted(root.rglob('*')) if p.is_file()})

            with patch.object(module, 'datetime', FrozenDateTime), patch.object(module, '_save_state', record):
                first = RecordingApi(root, config, fail_stage)
                run(first, config)
                calls = [tuple(row[0] for row in first.calls)]
                if retry:
                    cfg = dict(config, retry=True)
                    if mutate == 'input':
                        cfg['device'] = 'changed'
                    elif mutate == 'artifact':
                        (root / 'work/audio.wav').write_bytes(b'changed')
                    second = RecordingApi(root / 'attempts/0002', cfg)
                    second.snapshot = lambda: {'environment': {'output': str(root)}}
                    run(second, cfg)
                    calls.append(tuple(row[0] for row in second.calls))
            return events, errors, snapshots, calls

        cases = [(None, None, False, None)]
        cases += [(stage, None, False, None) for stage in STAGES]
        cases += [(None, n, False, None) for n in (1, 2, 3, 4, 38, 39, 40)]
        cases += [('stage.transcribe', None, True, mutation)
                  for mutation in (None, 'input', 'artifact')]
        cases += [(None, None, True, None)]  # completed attempts cannot retry
        for scenario in cases:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                config = make_config(tmp)
                config['performance_enabled'] = False
                root = (Path(tmp) / 'out').resolve()
                before = exercise(old, root, config, *scenario)
                shutil.rmtree(root)
                after = exercise(ops, root, config, *scenario)
                self.assertEqual(after, before)
