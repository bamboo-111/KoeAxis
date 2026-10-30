"""Lineage-bound MiMo scheduling with a provider-boundary fake; no sockets."""
import hashlib
import json
import os
import pathlib
import shutil
import socket
import tempfile
import threading
import unittest
import wave

from tests.test_real_chain_bind import completion, write_json

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRATCH = pathlib.Path(os.environ['PI_SCRATCH_DIR'])
NAMES = ('aligned_segments.json', 'transcript_segments.json', 'segments.json', 'split_segments.json', 'translated_segments.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lineage(root):
    hashes = {name: sha(root / name) for name in NAMES}
    write_json(root / 'lineage.json', {'source': 'synthetic', 'producer': 'fixture', 'files': hashes,
                                     'inputs': {name: hashes[name] for name in NAMES[:2]}})


def fixture(root, marker):
    root.mkdir(parents=True, exist_ok=True)
    wav = root / 'audio.wav'
    with wave.open(str(wav), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(b'\0\0' * 16000 * 25)
    base = {'segment_id': 'seg1', 'global_start_time': 0.0, 'global_end_time': 25.0, 'text': marker, 'status': 'completed'}
    aligned = dict(base, tokens=[{'text': marker, 'start_time': 0.1, 'end_time': 0.5}])
    segment = dict(base, audio_path=str(wav), source_audio_path=str(wav), language='Japanese')
    split, translated = {}, {}
    for i in range(1, 13):
        cue = {'start_time': i * 1000, 'end_time': i * 1000 + 700,
               'original_subtitle': marker + str(i), 'translated_subtitle': ''}
        split[str(i)] = dict(cue, segment_id='seg1')
        translated[str(i)] = dict(cue, translated_subtitle='译' + marker + str(i), needs_audio_review=True,
                                  suspect_confidence=0.2, suspect_types=['proper_noun'])
    for name, payload in zip(NAMES, ([aligned], [base], [segment], split, translated)):
        write_json(root / name, payload)
    lineage(root)
    return root


class ChainTest(unittest.TestCase):
    def setUp(self):
        self.temp = pathlib.Path(tempfile.mkdtemp(prefix='mimo-chain-', dir=SCRATCH))
        self.outputs = []
        self.saved = {k: os.environ.get(k) for k in ('REAL_MIMO_ONLINE', 'MIMO_API_KEY', 'MIMO_BASE_URL', 'REAL_MIMO_CHAIN_UPSTREAM', 'REAL_MIMO_CHAIN_RESUME_OUTPUT', 'REAL_CHAIN_BIND_UPSTREAM')}
        for key in self.saved: os.environ.pop(key, None)
        os.environ['REAL_MIMO_ONLINE'] = '1'; os.environ['MIMO_API_KEY'] = 'synthetic-key'
        import sys
        old = str(ROOT / 'compat' / 'legacy')
        if old not in sys.path: sys.path.insert(0, old)
        import qwen_asr.mimo_requests as req
        self.req = req; self.original = req.chat_completion_create
        self.connect = socket.socket.connect; self.create = socket.create_connection
        self.count = 0; self.fail_at = None; self.prompts = []; self.lock = threading.Lock()
        def fake(**kwargs):
            with self.lock:
                self.count += 1
                n = self.count
                self.prompts.append(str(kwargs.get('messages')))
            if self.fail_at is not None and n >= self.fail_at:
                if self.fail_at > 1:
                    import time; time.sleep(1)
                raise RuntimeError('HTTP 401 Invalid API Key')
            return completion('[]')
        def blocked(*args, **kwargs): raise AssertionError('socket forbidden')
        req.chat_completion_create = fake
        socket.socket.connect = blocked; socket.create_connection = blocked

    def tearDown(self):
        self.req.chat_completion_create = self.original
        socket.socket.connect = self.connect; socket.create_connection = self.create
        for key, val in self.saved.items():
            if val is None: os.environ.pop(key, None)
            else: os.environ[key] = val
        for output in self.outputs: shutil.rmtree(ROOT / output, ignore_errors=True)
        shutil.rmtree(self.temp, ignore_errors=True)

    def run_host(self, source, suffix):
        import host
        from host.runtime import Host
        work_output = 'runs/experiments/REAL-CHAIN-MIMO-FULL-TEST-' + suffix
        output = work_output
        if work_output in self.outputs:
            output = work_output + '-RETRY' + str(sum(x.startswith(work_output) for x in self.outputs))
            os.environ['REAL_MIMO_CHAIN_RESUME_OUTPUT'] = work_output
        else:
            os.environ.pop('REAL_MIMO_CHAIN_RESUME_OUTPUT', None)
        self.outputs.append(output)
        os.environ['REAL_MIMO_CHAIN_UPSTREAM'] = str(source)
        registration = next(x for x in host._plan_registry['plans'] if x['plan'].endswith('/real-chain-mimo-full.json'))
        obj = Host(ROOT, output, registration)
        try:
            result = obj.invoke(obj.envelope({'run_id': 'bootstrap-run', 'binding_id': 'bootstrap'}, registration['entry'], {'plan_id': 'REAL-CHAIN-MIMO-FULL-001'}))
        finally: obj.end_runs()
        self.assertEqual(result['outcome'], 'returned', result)
        return result['returned']['report'], ROOT / work_output
    def test_success_and_input_change(self):
        source = fixture(self.temp / 'a', 'CHAINALPHA')
        meta = json.loads((source / 'lineage.json').read_text(encoding='utf-8'))
        meta.update(source='current-run', producer='real-chain-llm')
        write_json(source / 'lineage.json', meta)
        report, out = self.run_host(source, 'SUCCESS')
        self.assertEqual(report['status'], 'partial', report)
        self.assertTrue(report['full_success']); self.assertFalse(report['chain_passed'])
        self.assertEqual([x['report']['phase'] for x in report['calls']], ['first-batch', 'full'])
        self.assertGreaterEqual(report['calls'][0]['report']['batch_count'], 2)
        evidence = {'inputs': report['inputs'], 'input_sha256': report['input_sha256'], 'phases': {}}
        for phase in ('first-batch', 'full'):
            records = list((out / 'calls/audio-openai' / phase).glob('*.json'))
            self.assertTrue(records, phase)
            self.assertTrue(any('CHAINALPHA' in json.loads(p.read_text(encoding='utf-8'))['prompt'] for p in records))
            parsed = [json.loads(p.read_text(encoding='utf-8')) for p in records]
            self.assertTrue(all(hashlib.sha256(x['prompt'].encode('utf-8')).hexdigest() == x['prompt_sha256'] for x in parsed))
            evidence['phases'][phase] = {'attempts': len(parsed), 'successful': sum(x['ok'] for x in parsed),
                                         'unique_prompt_sha256': sorted({x['prompt_sha256'] for x in parsed})}
        self.assertTrue((out / report['calls'][-1]['report']['artifact']).is_file())
        artifact = out / report['calls'][-1]['report']['artifact']
        evidence['artifact_sha256'] = sha(artifact)
        stage2 = out / 'stages/audio-openai/experiments/mimo-proofread/mimo_stage2_nearby_audio_report.json'
        completed = json.loads(stage2.read_text(encoding='utf-8'))
        self.assertEqual({str(x['id']) for x in completed if x.get('status') == 'completed'}, {str(x) for x in range(1, 13)})
        evidence['stage2_sha256'] = sha(stage2)
        print('CHAIN_EVIDENCE ' + json.dumps(evidence, sort_keys=True), flush=True)
        count = self.count
        changed = fixture(self.temp / 'b', 'CHAINBETA')
        rejected, _ = self.run_host(changed, 'SUCCESS')
        self.assertEqual(rejected['status'], 'failed'); self.assertIn('digest differs', rejected['detail'])
        self.assertEqual(count, self.count)
        audio = source / 'audio.wav'
        with audio.open('ab') as stream: stream.write(b'changed')
        rejected_audio, _ = self.run_host(source, 'SUCCESS')
        self.assertEqual(rejected_audio['status'], 'failed'); self.assertIn('digest differs', rejected_audio['detail'])
        self.assertEqual(count, self.count)

    def test_first_failure_stops(self):
        source = fixture(self.temp / 'first', 'FAILFIRST')
        self.fail_at = 1
        report, out = self.run_host(source, 'FIRSTFAIL')
        self.assertEqual(report['status'], 'failed'); self.assertEqual(self.count, 1)
        self.assertEqual(len(report['calls']), 1)
        self.assertFalse(json.loads((out / 'chain-state.json').read_text(encoding='utf-8'))['first_success'])
        self.assertFalse((out / 'calls/audio-openai/full').exists())

    def test_full_failure_then_resume(self):
        source = fixture(self.temp / 'resume', 'RESUMEMARK')
        self.fail_at = 3
        report, out = self.run_host(source, 'RESUME')
        self.assertEqual(report['status'], 'failed', report)
        state = json.loads((out / 'chain-state.json').read_text(encoding='utf-8'))
        self.assertTrue(state['first_success']); self.assertFalse(state['full_success'])
        stage2 = out / 'stages/audio-openai/experiments/mimo-proofread/mimo_stage2_nearby_audio_report.json'
        completed_before = json.loads(stage2.read_text(encoding='utf-8'))
        self.assertGreater(len([x for x in completed_before if x.get('status') == 'completed']), 3)
        self.assertLess(len([x for x in completed_before if x.get('status') == 'completed']), 12)
        before = self.count
        first_records = {p.name: p.read_bytes() for p in (out / 'calls/audio-openai/first-batch').glob('*.json')}
        self.fail_at = None
        result, out = self.run_host(source, 'RESUME')
        self.assertEqual(result['status'], 'partial', result)
        self.assertEqual([c['report']['phase'] for c in result['calls']], ['full'])
        self.assertGreater(self.count, before)
        self.assertEqual(first_records, {p.name: p.read_bytes() for p in (out / 'calls/audio-openai/first-batch').glob('*.json')})
        self.assertTrue(json.loads((out / 'chain-state.json').read_text(encoding='utf-8'))['full_success'])
        completed_after = json.loads(stage2.read_text(encoding='utf-8'))
        self.assertEqual({str(x['id']) for x in completed_after if x.get('status') == 'completed'}, {str(x) for x in range(1, 13)})
        print('RESUME_EVIDENCE ' + json.dumps({'completed_before': len(completed_before), 'completed_after': len(completed_after), 'stage2_sha256': sha(stage2), 'requests_before': before, 'requests_after': self.count}, sort_keys=True), flush=True)

    def test_invalid_inputs_zero_requests(self):
        for case in ('missing', 'json', 'hash', 'dependency'):
            source = fixture(self.temp / case, 'INVALID')
            if case == 'missing': (source / 'aligned_segments.json').unlink()
            elif case == 'json': (source / 'aligned_segments.json').write_text('{', encoding='utf-8')
            elif case == 'hash': write_json(source / 'lineage.json', {'source': 'synthetic', 'files': {}})
            else:
                data = json.loads((source / 'segments.json').read_text(encoding='utf-8'))
                data[0]['global_end_time'] = 24.0
                write_json(source / 'segments.json', data); lineage(source)
            before = self.count
            report, _ = self.run_host(source, 'INVALID-' + case)
            self.assertEqual(report['status'], 'failed', report); self.assertEqual(before, self.count)

    def test_crossing_cue_passes_and_wrong_parent_fails(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('mimo_ops_timing', ROOT / 'modules/temporary/real_mimo/operations.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        root = self.temp / 'cross'
        root.mkdir()
        wav = root / 'audio.wav'
        with wave.open(str(wav), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(b'\0\0' * 8000)
        def write_case(split_id, start, end=2500):
            segments = [
                {'segment_id': 'a', 'global_start_time': 0.0, 'global_end_time': 2.0, 'text': 'a', 'status': 'completed', 'audio_path': str(wav)},
                {'segment_id': 'b', 'global_start_time': 2.0, 'global_end_time': 4.0, 'text': 'b', 'status': 'completed', 'audio_path': str(wav)},
            ]
            aligned = [dict(item) for item in segments]
            transcript = [dict(item) for item in segments]
            split = {'1': {'segment_id': split_id, 'start_time': start, 'end_time': end, 'original_subtitle': '跨段'}}
            translated = {'1': dict(split['1'], translated_subtitle='跨段译')}
            for name, payload in zip(NAMES, (aligned, transcript, segments, split, translated)):
                write_json(root / name, payload)
            lineage(root)
        write_case('a', 1500)
        self.assertIn('lineage.json', mod._validate_chain(root))
        write_case('b', 1500)
        with self.assertRaisesRegex(ValueError, 'start outside segment'):
            mod._validate_chain(root)
        write_case('a', 5000, 5600)
        with self.assertRaisesRegex(ValueError, 'not covered'):
            mod._validate_chain(root)

    def test_mimo_reuse_requires_exact_prompt_and_drops_ambiguous(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('mimo_ops_reuse', ROOT / 'modules/temporary/real_mimo/operations.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cache = self.temp / 'reuse'
        cache.mkdir()
        prompt = 'exact public prompt'
        digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        write_json(cache / '0001.json', {
            'ok': True, 'failure_class': 'none', 'model': 'mimo-v2.6-pro',
            'prompt': prompt, 'prompt_sha256': digest, 'response': '[]',
        })
        write_json(cache / '0002.json', {
            'ok': False, 'failure_class': 'api_request_rejected', 'model': 'mimo-v2.6-pro',
            'prompt': 'empty', 'prompt_sha256': hashlib.sha256(b'empty').hexdigest(), 'response': '',
        })
        self.assertEqual(mod.load_mimo_reuse(cache), {digest: '[]'})
        write_json(cache / '0003.json', {
            'ok': True, 'failure_class': 'none', 'model': 'mimo-v2.6-pro',
            'prompt': prompt, 'prompt_sha256': digest, 'response': '[{}]',
        })
        self.assertEqual(mod.load_mimo_reuse(cache), {})



if __name__ == '__main__': unittest.main()
