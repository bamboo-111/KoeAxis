"""OPT54 composed Host checks; heavy providers are replaced, sockets forbidden."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.temporary.real_chain_runner import operations, performance


class Integration(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {'GEMINI_BASE_URL': 'https://example.invalid/v1beta',
                                               'MIMO_BASE_URL': 'https://example.invalid/v1'})
        environment.start()
        self.addCleanup(environment.stop)

    def test_context_restore_and_signature(self):
        from tests.test_real_chain_root_contract import make_config
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'OPT54_CONTENT_CONTEXT': ''}):
            config = make_config(folder)
            original = operations._input_signature(config)
            config['content_context'] = 'togeradi-v1'
            self.assertNotEqual(original, operations._input_signature(config))
            with self.assertRaises(RuntimeError):
                with operations._content_environment(config):
                    self.assertEqual(os.environ['OPT54_CONTENT_CONTEXT'], 'togeradi-v1')
                    raise RuntimeError('synthetic')
            self.assertEqual(os.environ['OPT54_CONTENT_CONTEXT'], '')

    def test_nested_calls_and_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = Path(folder) / 'stage/calls/full'
            calls.mkdir(parents=True)
            for i, (start, end) in enumerate(((1, 3), (2, 4), (4, 5))):
                (calls / f'{i}.json').write_text(json.dumps({'network': True, 'ok': True,
                    'request_started_unix_s': start, 'request_finished_unix_s': end}), encoding='utf-8')
            result = performance.summarize_calls(folder)
            self.assertEqual(result['network_requests'], 3)
            self.assertEqual(result['observed_peak_concurrency'], 2)
            self.assertEqual(result['interval_known'], 3)
            self.assertIsNone(result['retry_count'])

    def test_composed_host_noninterference_and_quality(self):
        from tests import test_real_chain_host312 as fixture
        outputs = []
        for enabled in (False, True):
            case = fixture.HostFullPath('test_success')
            try:
                case.setUp()
                case.config.update(performance_enabled=enabled, content_context='togeradi-v1')
                result = case.invoke()
                self.assertEqual(result['outcome'], 'returned', result)
                root = fixture.ROOT / case.output
                self.assertEqual((root / 'performance.json').is_file(), enabled)
                progress = json.loads((root / 'work/progress.json').read_text(encoding='utf-8'))
                self.assertEqual((progress['status'], progress['done'], progress['total']), ('completed', 12, 12))
                quality = result['returned']['report']['stages'][-1]['report']['post_export_quality']
                self.assertNotEqual(quality['status'], 'FAIL')
                outputs.append(((root / 'work/export/subtitles.srt').read_bytes(),
                                (root / 'work/export/subtitles.ass').read_bytes()))
                if enabled:
                    measured = json.loads((root / 'performance.json').read_text(encoding='utf-8'))
                    self.assertEqual(measured['status'], 'completed')
                    self.assertEqual(len([s for s in measured['stages'] if s['name'].startswith('stage.') or s['name'] == 'quality']), 12)
                    self.assertGreater(measured['calls']['network_requests'], 0)
                    self.assertIsNotNone(measured['alignment_timing'])
                    self.assertIn('cpu_scope', measured['resources'])
                    self.assertIn('cpu_process_seconds', measured['resources'])
                    self.assertIsNone(measured['measurement_coverage']['model_inference_seconds']['value'])
                    self.assertGreaterEqual(measured['measurement_coverage']['stage_wall']['count'], 12)
            finally:
                case.doCleanups()
        self.assertEqual(outputs[0], outputs[1])

    def test_composed_host_independent_reference_diagnostic(self):
        from tests import test_real_chain_host312 as fixture
        case = fixture.HostFullPath('test_success')
        try:
            case.setUp()
            reference = case.temp / 'human.ass'
            reference.write_text(
                '[Script Info]\nScriptType: v4.00+\n[Events]\n'
                'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'
                'Dialogue: 0,0:00:00.10,0:00:02.50,1.理名,,0,0,0,,こんにちは世界です\n',
                encoding='utf-8', newline='\n')
            case.config.update(ass_reference=str(reference), performance_enabled=True)
            result = case.invoke()
            self.assertEqual(result['outcome'], 'returned', result)
            report = result['returned']['report']['stages'][-1]['report']
            self.assertEqual(report['legacy_ass_similarity']['source'], 'aligned')
            self.assertEqual(report['legacy_ass_similarity']['status'], 'preserved')
            diagnostic = report['human_reference_diagnostic']
            self.assertEqual(diagnostic['schema'], 'gold54-human-reference-diagnostic/v2')
            self.assertTrue(diagnostic['verdict'].startswith('NOT_ASSESSED'))
            root = fixture.ROOT / case.output
            progress = json.loads((root / 'work/progress.json').read_text(encoding='utf-8'))
            self.assertEqual((progress['status'], progress['done'], progress['total']), ('completed', 12, 12))
            self.assertTrue((root / 'work/export/subtitles.ass').is_file())
        finally:
            case.doCleanups()

    def test_composed_host_versioned_memory(self):
        from tests import test_real_chain_host312 as fixture
        case = fixture.HostFullPath('test_success')
        try:
            case.setUp()
            case.config['memory'] = str(fixture.ROOT / 'experiments/references/togeradi-memory/togeradi-v2.json')
            result = case.invoke()
            self.assertEqual(result['outcome'], 'returned', result)
            root = fixture.ROOT / case.output
            memory_report = json.loads((root / 'work/reports/program_memory.json').read_text(encoding='utf-8'))
            self.assertIn('protected_cross_cue_addresses', memory_report)
            progress = json.loads((root / 'work/progress.json').read_text(encoding='utf-8'))
            self.assertEqual((progress['status'], progress['done'], progress['total']), ('completed', 12, 12))
            self.assertTrue((root / 'work/export/subtitles.srt').is_file())
            self.assertTrue((root / 'work/export/subtitles.ass').is_file())
        finally:
            case.doCleanups()

    def test_composed_host_failure_flush(self):
        from tests import test_real_chain_host312 as fixture
        case = fixture.HostFullPath('test_failure_stops_downstream')
        try:
            case.setUp()
            case.config.update(performance_enabled=True, content_context='togeradi-v1')
            case.fail_asr = True
            result = case.invoke()
            self.assertEqual(result['outcome'], 'raised', result)
            root = fixture.ROOT / case.output
            progress = json.loads((root / 'work/progress.json').read_text(encoding='utf-8'))
            measured = json.loads((root / 'performance.json').read_text(encoding='utf-8'))
            self.assertEqual(progress['status'], 'failed')
            self.assertEqual(measured['status'], 'failed')
            self.assertFalse((root / 'work/export/subtitles.srt').exists())
        finally:
            case.doCleanups()
