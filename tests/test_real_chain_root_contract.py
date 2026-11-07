"""Fixed-root call contract tests: a recording API, not a real Host or stage test."""
import tempfile
import unittest
from pathlib import Path

from host.contracts import ModuleError
from modules.temporary.real_chain_runner.operations import Implementation

STAGES = ('stage.prepare', 'stage.transcribe', 'stage.correct', 'stage.align', 'stage.speaker',
          'stage.split', 'stage.translate', 'stage.audio_review', 'stage.proofread_realign',
          'quality', 'stage.normalize', 'stage.export')
def make_config(directory):
    root = Path(directory) / 'inputs'
    root.mkdir()
    for name in ('media', 'enrollment', 'weights', 'memory', 'mfa', 'dict', 'acoustic'):
        (root / name).write_text(name, encoding='utf-8')
    for name in ('model', 'cache', 'align'):
        (root / name).mkdir()
    return {'media': str(root/'media'), 'model': str(root/'model'), 'model_cache': str(root/'cache'),
            'enrollment': str(root/'enrollment'), 'weights': str(root/'weights'), 'device': 'cpu',
            'memory': str(root/'memory'), 'align_resources': {'cache': str(root/'cache'),
            'asr_model': str(root/'model'), 'align_model': str(root/'align'), 'mfa_exe': str(root/'mfa'),
            'mfa_dictionary': str(root/'dict'), 'mfa_acoustic': str(root/'acoustic')}}


class RecordingApi:
    def __init__(self, output, config, fail=None):
        self.output = Path(output)
        self.work = self.output / 'work'
        self.config = config
        self.fail = fail
        self.calls = []

    def snapshot(self):
        return {'environment': {'output': str(self.output)}}

    def write(self, path, text):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return str(path)

    def call(self, target, args, artifacts=None):
        module = target['module_id']
        self.calls.append((module, args, artifacts))
        w = self.work
        if module == 'stage.prepare':
            self.write(w / 'segments.json', 'segments')
            self.write(w / 'audio.wav', 'audio')
            report = {'status': 'completed', 'artifact': str(w / 'segments.json'), 'audio': str(w / 'audio.wav')}
        elif module == 'stage.transcribe':
            self.write(w / 'transcript_segments.json', 'raw')
            report = {'status': 'completed', 'artifact': self.write(w / 'transcript_segments.raw.json', 'raw')}
        elif module == 'stage.correct':
            assert Path(args['raw_source']).read_text() == 'raw'
            for item in artifacts:
                self.write(Path(self.snapshot()['environment']['output']) / item['locator'], 'corrected')
            report = {'failed': 0, 'load_transcripts': True, 'input_preserved': True, 'corrected_count': 1}
        elif module == 'stage.align':
            assert Path(args['transcript_source']).read_text() == 'corrected'
            assert args['resources'] == self.config['align_resources']
            report = {'failed': 0, 'count': 1, 'aligned_path': self.write(w / 'alignment/aligned_segments.json', 'aligned')}
        elif module == 'stage.speaker':
            report = {'status': 'completed', 'artifact': self.write(w / 'speaker_scores.npz', 'speaker')}
        elif module == 'stage.split':
            assert Path(args['upstream'], 'transcript_segments.json').read_text() == 'corrected'
            assert Path(args['speaker_resources']['npz']).resolve() == (w / 'speaker_scores.npz').resolve()
            assert Path(args['speaker_resources']['audio']).resolve() == (w / 'audio.wav').resolve()
            assert Path(args['upstream'], 'aligned_segments.json').read_text() == 'aligned'
            report = {'status': 'partial', 'business_complete': True,
                      'artifact': self.write(self.output / 'stages/split-a/split_segments.json', 'split')}
        elif module == 'stage.translate':
            assert Path(args['upstream'], 'split_segments.json').read_text() == 'split'
            report = {'status': 'partial', 'business_complete': True,
                      'artifact': self.write(self.output / 'stages/translate-a/translated_segments.json', 'translated')}
        elif module == 'stage.audio_review':
            assert args['phase'] == 'full'
            assert Path(args['upstream'], 'translated_segments.json').read_text() == 'translated'
            report = {'status': 'partial', 'phase': 'full', 'business_ok': True,
                      'artifact': self.write(self.output / 'stages/audio-openai/experiments/mimo-proofread/mimo_proofread_segments.json', 'mimo')}
            for name in ('mimo_proofread_report.json', 'mimo_stage1_text_suspect_report.json',
                         'mimo_stage2_nearby_audio_report.json'):
                self.write(self.output / 'stages/audio-openai/experiments/mimo-proofread' / name, 'report')
        else:
            assert (w / 'experiments/mimo-proofread/mimo_proofread_segments.json').read_text() == 'mimo'
            report = {'returncode': 0, 'business': {'status': 'PASS'},
                      'artifact': self.write(w / ('artifact-' + module), module)}
            if module == 'stage.export':
                self.write(w / 'export/subtitles.ass', 'ass')
                self.write(w / 'final_quality.json', 'quality')
                report['post_export_quality'] = {'status': 'PASS'}
        if module == self.fail:
            if module in ('stage.prepare', 'stage.transcribe', 'stage.speaker'):
                report['status'] = 'failed'
            elif module == 'stage.correct':
                report['failed'] = 1
            elif module == 'stage.align':
                report['failed'] = 1
            elif module in ('stage.split', 'stage.translate'):
                report['business_complete'] = False
            elif module == 'stage.audio_review':
                report['business_ok'] = False
            else:
                report['returncode'] = 1
        return {'outcome': 'returned', 'returned': {'report': report}}


class FixedRootTests(unittest.TestCase):
    def test_order_and_artifact_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(directory)
            api = RecordingApi(directory, config)
            result, _ = Implementation({'role': 'root'}).invoke('run', {'plan_id': 'CHAIN-READY-001', 'config': config}, api)
            self.assertEqual(result['report']['status'], 'completed')
            self.assertEqual(tuple(row[0] for row in api.calls), STAGES)
            correct = api.calls[2]
            self.assertEqual(len(correct[2]), 3)
            self.assertTrue(all(r['writer_operation']['module_id'] == 'stage.correct' for r in correct[2]))

    def test_every_failure_stops_downstream(self):
        for index, stage in enumerate(STAGES):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                config = make_config(directory)
                api = RecordingApi(directory, config, stage)
                with self.assertRaises(ModuleError):
                    Implementation({'role': 'root'}).invoke('run', {'plan_id': 'CHAIN-READY-001', 'config': config}, api)
                self.assertEqual(tuple(row[0] for row in api.calls), STAGES[:index + 1])


if __name__ == '__main__':
    unittest.main()
