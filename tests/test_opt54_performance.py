"""Focused regression checks for OPT54 input digest reuse."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.temporary.real_chain_runner import operations
from tests.test_real_chain_root_contract import make_config, RecordingApi
from modules.temporary.real_chain_runner.performance import summarize_calls
from host.contracts import ModuleError


class InputDigestTests(unittest.TestCase):
    def test_reuse_is_local_to_one_content_check(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(directory)
            item = Path(config['model']) / 'weights.bin'
            item.write_bytes(b'first')
            original = operations._sha_file
            with patch.object(operations, '_sha_file', wraps=original) as digest:
                first = operations._input_signature(config)
                self.assertEqual(sum(Path(call.args[0]).resolve() == item.resolve()
                                     for call in digest.call_args_list), 1)
            item.write_bytes(b'second')
            self.assertNotEqual(first, operations._input_signature(config))


class IntegrationTests(unittest.TestCase):
    class Api(RecordingApi):
        def call(self, target, args, artifacts=None):
            self.write(self.work.parent / 'progress.json',
                       '{"stage":"audio-openai","status":"partial","done":147,"total":146}')
            if self.fail == 'exception' and target['module_id'] == 'stage.translate':
                raise RuntimeError('synthetic failure')
            response = super().call(target, args, artifacts)
            if target['module_id'] == 'stage.export':
                report = response['returned']['report']
                report['post_export_quality'] = {'status': 'WARN'}
                self.write(self.work / 'final_quality.json', '{"status":"WARN"}')
            return response

    def test_root_success_and_failure_flush(self):
        for fail in (None, 'stage.translate', 'exception', 'stage.export'):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                config = make_config(directory)
                api = self.Api(directory, config, fail)
                invoke = lambda: operations.run_root({'config': config}, api)
                if fail:
                    with self.assertRaises((ModuleError, RuntimeError)):
                        invoke()
                else:
                    self.assertEqual(invoke()['status'], 'completed')
                progress = json.loads((Path(directory) / 'work/progress.json').read_text())
                root_progress = json.loads((Path(directory) / 'progress.json').read_text())
                self.assertEqual(root_progress, progress)
                self.assertEqual(root_progress['stage'], 'fixed-chain')
                perf = json.loads((Path(directory) / 'performance.json').read_text())
                self.assertEqual(progress['status'], 'failed' if fail else 'completed')
                self.assertEqual(progress['total'], 12)
                self.assertEqual(progress['done'], 11 if fail == 'stage.export' else 6 if fail else 12)
                self.assertEqual(perf['status'], progress['status'])
                self.assertTrue(any(x['name'] == ('stage.translate' if fail in ('stage.translate', 'exception') else 'stage.export') for x in perf['stages']))
                self.assertNotIn('prompt_sha256', json.dumps(perf))

    def test_measurement_failure_does_not_change_business_or_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(directory)
            api = self.Api(directory, config, None)
            module = operations._performance_module()
            def broken(*args, **kwargs):
                raise RuntimeError('synthetic measurement failure')
            with patch.object(module.Performance, 'timed_stage', broken), \
                 patch.object(module.Performance, 'stage', broken), \
                 patch.object(operations, '_performance_module', return_value=module):
                self.assertEqual(operations.run_root({'config': config}, api)['status'], 'completed')
            root = Path(directory)
            self.assertEqual(json.loads((root / 'work/progress.json').read_text())['status'], 'completed')
            self.assertEqual(json.loads((root / 'performance.json').read_text())['status'], 'completed')

    def test_call_coverage_does_not_deduplicate_same_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = Path(directory) / 'calls/split-a'
            calls.mkdir(parents=True)
            for i, record in enumerate([
                {'network': True, 'ok': True, 'prompt_sha256': 'same', 'usage_known': True,
                 'usage': {'prompt_tokens': 0, 'completion_tokens': 2}, 'request_duration_s': 1},
                {'network': True, 'ok': False, 'prompt_sha256': 'same'},
                {'reused': True, 'network': False}]):
                (calls / (str(i) + '.json')).write_text(json.dumps(record), encoding='utf-8')
            summary = summarize_calls(directory)
            self.assertEqual((summary['network_requests'], summary['success'], summary['failure'], summary['reuse']), (2, 1, 1, 1))
            self.assertEqual((summary['duration_known'], summary['usage_known']), (1, 1))
            self.assertEqual(summary['usage']['prompt_tokens'], 0)
            self.assertIsNone(summary['retry_count'])


class FinishAdapterTests(unittest.TestCase):
    def test_export_calls_reference_then_post_gate(self):
        import contextlib
        from types import SimpleNamespace
        from modules.temporary.real_chain_finish import legacy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / 'work'
            (work / 'export').mkdir(parents=True)
            reference = root / 'reference.ass'
            reference.write_text('Dialogue: 0,0:00:00.00,0:00:01.00,1.理名,,0,0,0,,テスト\n', encoding='utf-8')
            calls = []
            paths = SimpleNamespace(final_quality_report=work / 'final_quality.json')
            legacy._legacy_imports()
            from qwen_asr import commands, ass_quality, models
            def run(args, _paths):
                calls.append(args)
                if args.command == 'export':
                    (work / 'export/subtitles.srt').write_text('srt', encoding='utf-8')
                if args.command == 'quality-gate':
                    paths.final_quality_report.write_text(json.dumps({'status': 'WARN', 'checks': []}), encoding='utf-8')
                return 0
            with patch.object(legacy, 'PROJECT', root), patch.object(models.WorkPaths, 'from_workdir', return_value=paths), patch.object(legacy, 'isolated_environment', return_value=contextlib.nullcontext()), patch.object(commands, 'cmd_export', run), patch.object(commands, 'cmd_quality_gate', run), patch.object(ass_quality, 'cmd_ass_quality', run):
                result = legacy.run_stage('export', work, ass_reference=str(reference))
            self.assertEqual([args.command for args in calls], ['export', 'ass-quality', 'quality-gate'])
            self.assertEqual(calls[1].source, 'aligned')
            self.assertEqual(calls[1].include_styles, '1.理名')
            self.assertTrue(calls[2].include_export)
            self.assertTrue(calls[2].require_srt)
            self.assertEqual(result['post_export_quality']['status'], 'WARN')
