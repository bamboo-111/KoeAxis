import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.temporary.real_chain_finish.handoff import InvalidHandoff, validate_handoff
from modules.temporary.real_chain_finish.operations import synthetic, manifest_resources, prepare_current
from modules.temporary.real_chain_finish.legacy import run_stage, _legacy_imports

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / 'runs/experiments/REAL-CHAIN-FINISH-001/continuation-1'


class FinishTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.socket_connect = patch('socket.socket.connect', side_effect=RuntimeError('network disabled in offline tests'))
        cls.socket_connect.start()
        cls.addClassCleanup(cls.socket_connect.stop)

    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def handoff(self):
        names = ['aligned_segments.json', 'transcript_segments.json', 'segments.json',
                 'split_segments.json', 'translated_segments.json', 'mimo_proofread_segments.json',
                 'mimo_proofread_report.json']
        sha = hashlib.sha256(b'current').hexdigest()
        for name in names:
            (self.root / name).write_bytes(b'current')
        (self.root / 'audio').write_bytes(b'audio')
        manifest = {'schema': 'real-chain-finish-handoff-v1', 'status': 'ready', 'workdir': str(self.root),
                    'files': {name: sha for name in names}, 'lineage': {'llm': 'current', 'mimo': 'current'},
                    'source_audio': {'path': str(self.root / 'audio'), 'sha256': hashlib.sha256(b'audio').hexdigest()}}
        path = self.root / 'handoff.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        return path, manifest

    def test_handoff_and_corrupt(self):
        path, manifest = self.handoff()
        self.assertEqual(validate_handoff(path), manifest)
        (self.root / 'segments.json').write_bytes(b'changed')
        with self.assertRaises(InvalidHandoff):
            validate_handoff(path)

    def test_missing_handoff(self):
        with self.assertRaises(InvalidHandoff):
            validate_handoff(self.root / 'missing')

    def test_resources_use_manifest_purpose_and_check_copied_sha(self):
        _, manifest = self.handoff()
        cache = self.root / 'run-2.npz'
        enrollment = self.root / 'enrollment.json'
        cache.write_bytes(b'run-2')
        enrollment.write_bytes(b'enrollment')
        def resource(purpose, path):
            return {'purpose': purpose, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest['resources'] = [resource('source-audio', self.root / 'audio'),
                                 resource('speaker-score-cache', cache), resource('speaker-enrollment', enrollment)]
        _, speaker, resources = prepare_current(manifest, self.root / 'prepared')
        self.assertEqual(Path(speaker), enrollment)
        self.assertEqual((self.root / 'prepared/speaker_scores.npz').read_bytes(), b'run-2')
        self.assertEqual(resources['speaker-score-cache']['path'], str(cache))
        manifest['resources'].append(resource('speaker-score-cache', cache))
        with self.assertRaises(ValueError):
            manifest_resources(manifest)
        manifest['resources'].pop()
        cache.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            prepare_current(manifest, self.root / 'corrupt')

    def test_real_cpu_normalize_memory_export(self):
        work = self.root / 'work'
        memory = synthetic(work)
        self.assertEqual(run_stage('realign', work)['business']['candidate_count'], 0)
        self.assertEqual(run_stage('quality', work)['returncode'], 0)
        self.assertEqual(run_stage('normalize', work, memory)['returncode'], 0)
        self.assertGreater(json.loads((work / 'reports/program_memory.json').read_text(encoding='utf-8'))['changed_cue_count'], 0)
        self.assertEqual(run_stage('export', work)['returncode'], 0)
        self.assertTrue((work / 'export/subtitles.srt').stat().st_size)

    def test_quality_failure_blocks_normalize(self):
        work = self.root / 'work'
        memory = synthetic(work)
        (work / 'translated_segments.json').write_text('{"1":{"original_subtitle":"x"}}', encoding='utf-8')
        self.assertEqual(run_stage('quality', work)['business']['status'], 'FAIL')
        with self.assertRaises(RuntimeError):
            run_stage('normalize', work, memory)
        self.assertFalse((work / 'normalized_segments.json').exists())

    def test_nonzero_candidate_legacy_model_boundary(self):
        work = self.root / 'work'
        synthetic(work)
        _legacy_imports()
        from qwen_asr.models import WorkPaths
        from qwen_asr.cli import build_parser
        from qwen_asr import proofread_realign as pr
        manifest_path = work / 'experiments/mimo-proofread/mimo_proofread_segments.json'
        data = json.loads(manifest_path.read_text(encoding='utf-8'))
        data['1']['needs_realign'] = True
        manifest_path.write_text(json.dumps(data), encoding='utf-8')
        (work / 'audio.wav').write_bytes(b'boundary replaced')
        args = build_parser().parse_args(['proofread-realign', '--workdir', str(work)])
        class Model:
            def __init__(self, *args, **kwargs): pass
            def load(self): pass
            def close(self): pass
        def aligned(model, audio, output, subtitle_id, item, **kwargs):
            item['realign_status'] = 'completed'
            item['needs_realign'] = False
            return {'status': 'completed', 'method': 'synthetic-model'}
        with patch.object(pr, '_realign_one', aligned):
            report = pr.run_proofread_realign_stage(args, WorkPaths.from_workdir(work), aligner_factory=Model)
        self.assertEqual(report['candidate_count'], 1)
        self.assertEqual(report['completed_count'], 1)
        self.assertEqual(report['failed_count'], 0)


if __name__ == '__main__':
    unittest.main()
