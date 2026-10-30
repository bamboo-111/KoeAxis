import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from modules.temporary.real_llm_stages import operations as llm
from modules.temporary.real_mimo import operations as mimo


class API:
    def __init__(self, value=''):
        self.value = value
    def environment(self, name, default=''):
        return self.value if name == 'OPT54_CONTENT_CONTEXT' else default


def response(usage):
    return NS(usage=usage, choices=[NS(message=NS(content='ok'))])


class ContentTests(unittest.TestCase):
    def setUp(self):
        self.socket_patch = patch.object(socket.socket, 'connect', side_effect=AssertionError('socket blocked'))
        self.connect_patch = patch.object(socket, 'create_connection', side_effect=AssertionError('socket blocked'))
        self.socket_patch.start()
        self.connect_patch.start()
        self.addCleanup(self.socket_patch.stop)
        self.addCleanup(self.connect_patch.stop)

    def test_optional_context_and_translate_public_parameter(self):
        self.assertEqual(llm._content_context(API()), '')
        self.assertEqual(mimo._content_glossary(API()), [])
        context = llm._content_context(API('togeradi-v1'))
        entries = mimo._content_glossary(API('togeradi-v1'))
        self.assertIn('トゲナシトゲアリ', context)
        self.assertIn('メールや記号', context)
        for word in ('ミニオン', 'ミリオン', 'togeradi@interfm.jp'):
            self.assertNotIn(word, context)
            self.assertNotIn(word, [e['source'] for e in entries])
        self.assertIn('トゲナシトゲアリ', [e['source'] for e in entries])
        captured = []
        modules = {'models': NS(WorkPaths=NS(from_workdir=lambda x: x)),
                   'stages': NS(run_translate_stage=lambda **kw: captured.append(kw))}
        llm._run_translate(modules, Path('.'), {'base': 'test', 'key': 'fake'}, context)
        llm._run_translate(modules, Path('.'), {'base': 'test', 'key': 'fake'})
        self.assertEqual(captured[0]['program_context'], context)
        self.assertEqual(captured[1]['program_context'], '')
        self.assertNotEqual(llm._sha_text(context), llm._sha_text(''))

    def test_text_and_mimo_public_arguments(self):
        context = llm._content_context(API('togeradi-v1'))
        entries = mimo._content_glossary(API('togeradi-v1'))
        observed = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'translated.json').write_text('{}', encoding='utf-8')
            (root / 'segments.json').write_text('{}', encoding='utf-8')
            paths = NS(translated_manifest=root / 'translated.json', segments_manifest=root / 'segments.json')
            proof = NS(TextProofreadConfig=lambda **kw: NS(**kw), load_glossary=lambda p: [],
                       build_requests=lambda a, b, g, config: observed.append(config.program_context) or [],
                       run_text_proofread=lambda p, config, **kw: observed.append(config.program_context) or {})
            pool = NS(GeminiEndpoint=lambda **kw: NS(**kw), GeminiRelayPool=lambda *a, **kw: NS(close=lambda: None))
            modules = {'models': NS(WorkPaths=NS(from_workdir=lambda w: paths)), 'gemini_pool': pool, 'text_proofread': proof}
            llm._run_text(modules, root, [{'name': 'test', 'base': '', 'key': ''}], NS(set_total=lambda n: None), context)
            self.assertEqual(observed, [context, context])
            seen = []
            proof = NS(_load_existing_branch=lambda *a: {}, _load_existing_report=lambda *a: {},
                       _process_stage2_nearby_audio_batch_task=lambda **kw: seen.append(kw['glossary']) or [],
                       _write_two_stage_outputs=lambda *a, **kw: None)
            args = NS(nearby_context_subtitles=1, nearby_padding_s=1, max_glossary_entries=80,
                      max_retries=1, retry_base_delay=0, retry_max_delay=0, stage2_apply_threshold=.9)
            mimo._run_one_batch({'proof': proof}, root, None, None, args, [], {}, entries, {}, ['1'])
            self.assertEqual(seen, [entries])

    def test_capture_success_failure_usage_and_reuse(self):
        for module in (llm, mimo):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with patch.object(module, '_output_dir', return_value=root), patch.object(module, 'update_progress'):
                    cap = (llm._Capture('x', 'translate', 'r', []) if module is llm
                           else mimo._Capture('x', 'audio-openai', 'full', [], archive=False))
                    kwargs = ({'model': 'test', 'payload': {'contents': [{'parts': [{'text': 'メール a@b.jp ! ミニオン ミリオン'}]}]}}
                              if module is llm else {'config': NS(model=mimo.MODEL, base_url=''), 'messages': [{'content': 'メール a@b.jp ! ミニオン ミリオン'}]})
                    for use in (NS(prompt_tokens=0, completion_tokens=0), None, NS(prompt_tokens=1)):
                        cap._wrap(lambda **kw: response(use))(**kwargs)
                    def fail(**kw):
                        raise RuntimeError('failure')
                    with self.assertRaises(RuntimeError):
                        cap._wrap(fail)(**kwargs)
                    rows = sorted(root.glob('calls/translate-r/*.json') if module is llm else root.glob('calls/audio-openai/full/*.json'))
                    self.assertEqual(len(rows), 4)
                    data = [json.loads(p.read_text(encoding='utf-8')) for p in rows]
                    self.assertEqual([x['usage_known'] for x in data], [True, False, False, False])
                    self.assertTrue(all(x['network'] and not x['reused'] and x['request_duration_s'] >= 0 for x in data))
                    self.assertFalse(data[-1]['ok'])
                    self.assertIn('a@b.jp ! ミニオン ミリオン', data[0]['prompt'])
                    self.assertNotIn('http://', json.dumps(data))
                    prompt = data[0]['prompt']
                    if module is llm:
                        cap.set_reuse({module._sha_text(prompt): {'model': 'test', 'response': 'ok'}})
                    else:
                        cap.set_reuse({module._sha_text(prompt): 'ok'})
                    cap._wrap(lambda **kw: self.fail('unexpected provider call'))(**kwargs)
                    reused = list(root.glob('calls/**/reused/*.json'))
                    self.assertEqual(len(reused), 1)
                    row = json.loads(reused[0].read_text(encoding='utf-8'))
                    self.assertTrue(row['reused'])
                    self.assertFalse(row['network'])
                    self.assertIsNone(row['request_duration_s'])
                    self.assertEqual(cap.completed, 4)


class EffectDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        import sys
        path = Path('runs/experiments/OPT54-CONTENT-002/diagnose.py')
        if path.is_file():
            spec = importlib.util.spec_from_file_location('opt54_diagnose', path)
            cls.diag = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.diag)
        cls.legacy_path = str(Path(__file__).resolve().parents[1] / 'compat/legacy')
        sys.path.insert(0, cls.legacy_path)
        from qwen_asr.mimo_application import evaluate_stage2_suggestion
        cls.evaluate = staticmethod(evaluate_stage2_suggestion)

    @classmethod
    def tearDownClass(cls):
        import sys
        sys.path.remove(cls.legacy_path)

    @unittest.skipUnless(Path('runs/experiments/OPT54-CONTENT-002/diagnose.py').is_file(),
                         'local historical request archive is not part of the release')
    def test_recorded_requests_and_stage_attribution(self):
        data = self.diag.build()
        self.assertEqual([len(data['requests'][x]) for x in ('translate-a', 'audio-openai')], [34, 146])
        for rows in data['requests'].values():
            self.assertTrue(all(x['sha_matches'] and x['digest_prefix_present'] for x in rows))
        self.assertTrue(all(x['digest_full_present'] for x in data['requests']['translate-a']))
        self.assertTrue(all(not x['digest_full_present'] for x in data['requests']['audio-openai']))
        self.assertEqual(len(data['cases']), 21)
        case = next(x for x in data['cases'] if x['reference_line'] == 25)
        self.assertEqual(case['stages']['split'][0]['original'], case['stages']['translate'][0]['original'])
        self.assertEqual(case['reviews'][0]['rejections'][0]['reason'], 'original-content-dropped-without-ass-reference')
        self.assertNotEqual(case['stages']['raw'][0]['original'], case['stages']['correct'][0]['original'])

    @unittest.skipUnless(Path('runs/experiments/OPT54-CONTENT-002/diagnose.py').is_file(),
                         'local historical request archive is not part of the release')
    def test_candidate_and_guard_diagnostics_distinguish_unknown(self):
        reports = [{'id': '1', 'status': 'completed', 'suggestions': [{'suggested_original': 'x'}],
                    'rejections': [{'reason': 'guard'}]}]
        rows = self.diag.select_reviews(['1', '2', '3'], ['1', '3'], reports)
        self.assertEqual([x['interpretation'] for x in rows], ['rejected_requires_independent_evidence',
                         'not_selected_not_a_proven_defect', 'selected_no_report'])
        self.assertEqual(rows[0]['rejections'][0]['reason'], 'guard')
        self.assertEqual(rows[0]['suggestions'][0]['suggested_original'], 'x')

    def test_correct_words_addresses_and_ambiguous_words_not_forced(self):
        for text in ('トゲナシトゲアリ', 'トゲラジアットインタFMドットJP',
                     'togeradi@interfm.jp', 'other@example.org', 'ミニオン', 'ミリオン'):
            with self.subTest(text=text):
                item = {'original_subtitle': text, 'translated_subtitle': '原来的译文'}
                suggestion = {'id': '1', 'asr_suspect': False, 'needs_audio_review': False,
                              'suggested_original': '', 'suggested_translation': '', 'confidence': 0.99}
                result = self.evaluate(subtitle_id='1', normalized=suggestion, current_item=item,
                                       apply_confidence_threshold=0.9)
                self.assertNotIn('original_subtitle', result.updates)
                self.assertNotIn('translated_subtitle', result.updates)

    def test_original_guard_rejects_unsubstantiated_replacement(self):
        item = {'original_subtitle': 'interfm.jpトゲなしトゲありの', 'translated_subtitle': '原来的译文'}
        suggestion = {'id': '1', 'asr_suspect': True, 'needs_audio_review': False,
                      'suggested_original': 'interfm.jp トゲナシトゲアリの',
                      'suggested_translation': '原来的译文', 'confidence': 0.95}
        result = self.evaluate(subtitle_id='1', normalized=suggestion, current_item=item,
                               apply_confidence_threshold=0.9)
        self.assertEqual(result.rejection['reason'], 'original-content-dropped-without-ass-reference')
        self.assertFalse(result.updates)

    def test_offline_socket_hook_is_active(self):
        import os
        if os.environ.get('OPT54_EXPECT_BLOCKED') != '1':
            self.skipTest('requires offline verification startup hook')
        for fn, args in ((socket.socket.connect, (None, ('127.0.0.1', 9))),
                         (socket.socket.connect_ex, (None, ('127.0.0.1', 9))),
                         (socket.create_connection, (('127.0.0.1', 9),))):
            with self.assertRaisesRegex(RuntimeError, 'socket blocked'):
                fn(*args)

    def test_ambiguous_replacements_need_evidence(self):
        for before, after in [('ミニオン', 'ミリオン'), ('ミリオン', 'ミニオン'),
                              ('トゲナシトゲアリ', '別の名前')]:
            item = {'original_subtitle': before, 'translated_subtitle': '原来的译文'}
            suggestion = {'id': '1', 'asr_suspect': True, 'needs_audio_review': False,
                          'suggested_original': after, 'suggested_translation': '原来的译文',
                          'confidence': 0.95}
            result = self.evaluate(subtitle_id='1', normalized=suggestion, current_item=item,
                                   apply_confidence_threshold=0.9)
            self.assertFalse(result.updates)
            self.assertIsNotNone(result.rejection)

    def test_changed_context_digest_invalidates_real_capture_reuse(self):
        for module in (llm, mimo):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                with patch.object(module, '_output_dir', return_value=Path(tmp)), patch.object(module, 'update_progress'):
                    cap = (llm._Capture('x', 'translate', 'r', []) if module is llm
                           else mimo._Capture('x', 'audio-openai', 'full', [], archive=False))
                    old, new = 'context sha256=111', 'context sha256=222'
                    cached = {'model': 'test', 'response': 'cached'} if module is llm else 'cached'
                    cap.set_reuse({module._sha_text(old): cached})
                    def args(text):
                        return ({'model': 'test', 'payload': {'contents': [{'parts': [{'text': text}]}]}}
                                if module is llm else {'config': NS(model=mimo.MODEL, base_url=''),
                                                       'messages': [{'content': text}]})
                    calls = []
                    def provider(**kw):
                        calls.append(kw)
                        return response(NS(prompt_tokens=1, completion_tokens=1))
                    wrapped = cap._wrap(provider)
                    wrapped(**args(old))
                    self.assertEqual(len(calls), 0)
                    wrapped(**args(new))
                    self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
