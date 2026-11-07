"""Fixed-root integration on the real Host; external model/HTTP/tools are faked.

Run with the verified 3.12 isolated resource profile, shared lock and socket guard.
"""
import base64
import hashlib
import io
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
import uuid
import wave
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OLD = pathlib.Path(__file__).resolve().parents[1] / "compat" / "legacy"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('_host312_legacy_guard', ROOT/'modules/temporary/real_align_recovery/chain.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
guard.protect_legacy(OLD)
sys.path.insert(0, str(OLD))
import tests.test_real_chain_bind as bind
from tests.test_real_chain_bind import Binder, completion


class HostFullPath(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from qwen_asr.commands import stages
        from qwen_asr.models import SpeechRegion, TranscriptSegment, AlignedSegment, AlignedToken
        from qwen_asr.speaker import voiceprint
        import qwen_asr.mimo_requests as mimo
        self.temp = pathlib.Path(tempfile.mkdtemp(prefix='host312-input-', dir=os.environ['PI_SCRATCH_DIR']))
        self.output = 'runs/experiments/CHAIN-READY-HOST312-TEST-' + uuid.uuid4().hex
        self.addCleanup(shutil.rmtree, ROOT / self.output, True)
        self.addCleanup(shutil.rmtree, self.temp, True)
        self.binder = Binder()
        self.binder.install()
        self.addCleanup(self.binder.restore)
        self.stack = []
        self.addCleanup(lambda: [p.stop() for p in reversed(self.stack)])
        self.model_calls = []
        self.fail_asr = False
        self.suspect = False
        self.fail_mimo = False
        self.mimo_calls = []
        def replace(obj, name, value):
            p = patch.object(obj, name, value); p.start(); self.stack.append(p)
        def extract(source, target, **kwargs):
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target); return target
        def export(source, segment):
            target = pathlib.Path(segment.audio_path); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        class Vad:
            def detect(self, audio): return [SpeechRegion(0.1, 3.8)]
        owner = self
        class ASR:
            def __init__(self, **kwargs): pass
            def load(self): owner.model_calls.append('asr')
            def close(self): pass
            def run_batch(self, segments):
                if owner.fail_asr: raise RuntimeError('synthetic ASR failure')
                return [TranscriptSegment(s.segment_id, s.audio_path, s.global_start_time, s.global_end_time,
                                          'こんにちは世界です', 'Japanese') for s in segments]
        class Align:
            def __init__(self, **kwargs): pass
            def load(self): owner.model_calls.append('align')
            def close(self): pass
            def run_segment(self, s, cleanup=False):
                return AlignedSegment(s.segment_id, s.audio_path, s.global_start_time, s.global_end_time,
                    s.text, s.language, [AlignedToken(s.text, 0.1, 2.5)])
        def load_speaker_model(weights, device):
            owner.model_calls.append('speaker')
            return object()
        def embeddings(samples, model, **kwargs):
            return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (20, 1))
        replace(stages, 'extract_audio', extract)
        replace(stages, 'create_vad_adapter', lambda *a, **kw: Vad())
        replace(stages, 'export_segment_audio', export)
        replace(stages, 'QwenASRTranscriber', ASR)
        replace(stages, 'QwenForcedAligner', Align)
        replace(voiceprint, 'load_redimnet', load_speaker_model)
        replace(voiceprint, 'window_embeddings', embeddings)
        default_gemini = bind.fixed_gemini
        def gemini_response(text):
            import re
            match = re.search(r'<input_subtitle>(\{.*?\})</input_subtitle>', text, re.S)
            if owner.suspect and match:
                data = json.loads(match.group(1))
                return json.dumps({key: {'translation': '合成简体甲', 'asr_suspect': True,
                    'needs_audio_review': True, 'suspect_types': ['name'], 'reason': 'synthetic',
                    'confidence': 0.2} for key in data}, ensure_ascii=False)
            return default_gemini(text)
        def mimo_http(**kwargs):
            owner.mimo_calls.append(kwargs.get('messages'))
            if owner.fail_mimo:
                raise RuntimeError('HTTP 401 Invalid API Key')
            return completion('[]')
        replace(bind, 'fixed_gemini', gemini_response)
        replace(mimo, 'chat_completion_create', mimo_http)
        for key, value in {'REAL_LLM_STAGES_ONLINE':'1', 'GEMINI_API_KEY':'synthetic-key',
                           'REAL_MIMO_ONLINE':'1', 'MIMO_API_KEY':'synthetic-key'}.items():
            p = patch.dict(os.environ, {key:value}); p.start(); self.stack.append(p)
        audio = self.temp / 'input.wav'
        with wave.open(str(audio), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(b'\0\0' * 64000)
        for name in ('model', 'cache', 'align'):
            (self.temp / name).mkdir()
        for name in ('weights', 'mfa.exe', 'dictionary', 'acoustic'):
            (self.temp / name).write_bytes(b'synthetic')
        (self.temp / 'enrollment.json').write_text(json.dumps({'members': {'A': [1.0, 0.0]}}), encoding='utf-8')
        (self.temp / 'memory.json').write_text(json.dumps({'rules': []}), encoding='utf-8')
        self.config = {'media':str(audio), 'model':str(self.temp/'model'), 'model_cache':str(self.temp/'cache'),
            'enrollment':str(self.temp/'enrollment.json'), 'weights':str(self.temp/'weights'), 'device':'cpu',
            'memory':str(self.temp/'memory.json'), 'prepare_options':['--eager-segment-export'],
            'transcribe_options':['--language','Japanese','--device','cpu'],
            'align_resources': {'cache':str(self.temp/'cache'), 'asr_model':str(self.temp/'model'),
                'align_model':str(self.temp/'align'), 'mfa_exe':str(self.temp/'mfa.exe'),
                'mfa_dictionary':str(self.temp/'dictionary'), 'mfa_acoustic':str(self.temp/'acoustic')}}

    def invoke(self):
        from host.runtime import Host
        registration = json.loads((ROOT/'module-registry/real_chain_runner/plan-entry-312.json').read_text(encoding='utf-8'))
        if not hasattr(self, 'host'):
            self.host = Host(ROOT, self.output, registration)
            self.addCleanup(self.host.end_runs)
        return self.host.invoke(self.host.envelope({'run_id':'bootstrap-run','binding_id':'bootstrap'}, registration['entry'],
            {'plan_id':'CHAIN-READY-001','config':self.config}))

    def test_success(self):
        result = self.invoke()
        self.assertEqual(result['outcome'], 'returned', result)
        report = result['returned']['report']
        self.assertEqual(report['status'], 'completed')
        self.assertEqual([entry['stage'] for entry in report['stages']], [
            'stage.prepare', 'stage.transcribe', 'stage.correct', 'stage.align', 'stage.speaker',
            'stage.split', 'stage.translate', 'stage.audio_review', 'stage.proofread_realign',
            'quality', 'stage.normalize', 'stage.export'])
        for name in ('subtitles.srt', 'subtitles.ass'):
            path = ROOT/self.output/'work/export'/name
            self.assertTrue(path.is_file() and path.stat().st_size > 0, str(path))
        self.assertTrue((ROOT/self.output/'work/transcript_segments.raw.json').is_file())
        self.assertTrue((ROOT/self.output/'work/experiments/mimo-proofread/mimo_proofread_segments.json').is_file())
        raw = json.loads((ROOT/self.output/'work/transcript_segments.raw.json').read_text(encoding='utf-8'))
        aligned = json.loads((ROOT/self.output/'work/aligned_segments.json').read_text(encoding='utf-8'))
        self.assertEqual(raw[0]['text'], 'こんにちは世界です')
        self.assertEqual(aligned[0]['segment_id'], raw[0]['segment_id'])
        self.assertIn('こんにちは', (ROOT/self.output/'work/export/subtitles.srt').read_text(encoding='utf-8-sig'))
        self.assertGreater(self.binder.calls, 0)
        print('HOST312_EVIDENCE ' + json.dumps({'stages': [x['stage'] for x in report['stages']],
            'model_calls': self.model_calls, 'srt_bytes': (ROOT/self.output/'work/export/subtitles.srt').stat().st_size,
            'ass_bytes': (ROOT/self.output/'work/export/subtitles.ass').stat().st_size,
            'mimo_report': report['stages'][7]['report']}, ensure_ascii=True, sort_keys=True))
        self.assertEqual(self.model_calls, ['asr', 'align', 'speaker'])

    def test_failure_stops_downstream(self):
        self.fail_asr = True
        result = self.invoke()
        self.assertEqual(result['outcome'], 'raised', result)
        self.assertEqual(result['raised']['category'], 'MediaStageFailed')
        self.assertIn('synthetic ASR failure', result['raised']['message'])
        self.assertEqual(self.model_calls, ['asr'])
        self.assertFalse((ROOT/self.output/'work/alignment').exists())
        self.assertFalse((ROOT/self.output/'work/export/subtitles.srt').exists())

    def test_suspect_audio_success(self):
        self.suspect = True
        result = self.invoke()
        self.assertEqual(result['outcome'], 'returned', result)
        report = result['returned']['report']['stages'][7]['report']
        self.assertTrue(report['business_ok'], report)
        self.assertGreater(report['suspect_count'], 0, report)
        self.assertGreater(report['batch_count'], 0, report)
        self.assertGreater(len(self.mimo_calls), 0)
        self.assertGreater(report['successful_requests'], 0)
        parts = self.mimo_calls[0][0]['content']
        self.assertTrue(any(part.get('type') == 'input_audio' and part.get('input_audio', {}).get('data')
                            for part in parts if isinstance(part, dict)))
        stage2 = json.loads((ROOT/self.output/'work/experiments/mimo-proofread/mimo_stage2_nearby_audio_report.json').read_text(encoding='utf-8'))
        audio_parts = [part['input_audio'] for part in parts if isinstance(part, dict) and part.get('type') == 'input_audio']
        with wave.open(io.BytesIO(base64.b64decode(audio_parts[0]['data'])), 'rb') as clip:
            frames = clip.readframes(clip.getnframes())
            self.assertTrue(frames and not any(frames))
        for name in ('segments.json', 'split_segments.json', 'translated_segments.json'):
            actual = hashlib.sha256((ROOT/self.output/'work'/name).read_bytes()).hexdigest()
            self.assertEqual(report['inputs'][name], actual)
        self.assertTrue(stage2 and all(item.get('status') == 'completed' for item in stage2))
        self.assertIn('こんにちは', str(self.mimo_calls))
        print('MIMO_NONZERO ' + json.dumps(report, ensure_ascii=True, sort_keys=True))

    def test_suspect_audio_failure_stops(self):
        self.suspect = True
        self.fail_mimo = True
        result = self.invoke()
        self.assertEqual(result['outcome'], 'raised', result)
        self.assertEqual(result['raised']['category'], 'StageIncomplete', result)
        self.assertEqual(result['raised']['message'], 'audio_review')
        self.assertGreater(len(self.mimo_calls), 0)
        self.assertFalse((ROOT/self.output/'work/export/subtitles.srt').exists())
        self.fail_mimo = False
        self.config['retry'] = True
        retried = self.invoke()
        self.assertEqual(retried['outcome'], 'returned', retried)
        self.assertEqual(retried['returned']['report']['attempt'], 2)
        self.assertGreater(len(self.mimo_calls), 1)
        self.assertTrue((ROOT/self.output/'attempts/0002/work/export/subtitles.srt').is_file())

    def test_failed_attempt_retry_and_digest_guards(self):
        self.fail_asr = True
        first = self.invoke()
        self.assertEqual(first['outcome'], 'raised', first)
        state_path = ROOT/self.output/'fixed-chain-state.json'
        state = json.loads(state_path.read_text(encoding='utf-8'))
        self.assertEqual(state['attempts'][0]['status'], 'failed')
        self.assertEqual([x['name'] for x in state['attempts'][0]['stages']], ['stage.prepare'])
        self.config['retry'] = True
        self.fail_asr = False
        self.config['device'] = 'changed'
        changed_config = self.invoke()
        self.assertEqual(changed_config['raised']['category'], 'InputChanged', changed_config)
        self.config['device'] = 'cpu'
        with patch.dict(os.environ, {'GEMINI_MODEL': 'changed-model'}):
            changed_env = self.invoke()
            self.assertEqual(changed_env['raised']['category'], 'InputChanged', changed_env)
        audio = pathlib.Path(self.config['media'])
        original = audio.read_bytes()
        audio.write_bytes(original + b'x')
        changed_input = self.invoke()
        self.assertEqual(changed_input['raised']['category'], 'InputChanged', changed_input)
        audio.write_bytes(original)
        artifact = ROOT/self.output/'work/segments.json'
        old = artifact.read_bytes()
        artifact.write_bytes(old + b' ')
        changed_artifact = self.invoke()
        self.assertEqual(changed_artifact['raised']['category'], 'ArtifactChanged', changed_artifact)
        artifact.write_bytes(old)
        result = self.invoke()
        self.assertEqual(result['outcome'], 'returned', result)
        self.assertEqual(result['returned']['report']['attempt'], 2)
        state = json.loads(state_path.read_text(encoding='utf-8'))
        self.assertEqual([x['status'] for x in state['attempts']], ['failed', 'completed'])
        self.assertEqual(len(state['attempts'][1]['stages']), 12)
        self.assertTrue((ROOT/self.output/'attempts/0002/work/export/subtitles.ass').is_file())
        print('RETRY_EVIDENCE ' + json.dumps(state, ensure_ascii=True, sort_keys=True))

if __name__ == '__main__': unittest.main(verbosity=2)
