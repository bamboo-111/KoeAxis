import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from modules.temporary.real_chain_finish.legacy import (
    OLD, _legacy_imports, apply_address_boundary_protection, run_stage,
)
from modules.temporary.real_chain_finish.operations import synthetic

ROOT = Path(__file__).resolve().parents[1]
MEMORY_V2 = ROOT / 'experiments/references/togeradi-memory/togeradi-v2.json'
OUT = ROOT / 'runs/experiments/OPT54-MEMORY-003'
OLD_MEMORY = OLD / 'samples/program-memory-togeradi.json'


def row(source, translation):
    return {'original_subtitle': source, 'translated_subtitle': translation,
            'start_time': 100, 'end_time': 900, 'speaker': 'A', 'unrelated': {'x': 1}}


class MemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _legacy_imports()
        from qwen_asr import program_memory as pm
        cls.pm = pm
        cls.old = pm.load_program_memory(OLD_MEMORY)
        cls.new = pm.load_program_memory(MEMORY_V2)

    def test_old_reproduction_and_v2_cue(self):
        for text in ('other.togeradi@internet.jp', 'togeradi@internet.jp.example.org'):
            old = self.pm.apply_to_cue(text, text, self.old)
            new = self.pm.apply_to_cue(text, text, self.new)
            self.assertNotEqual(old.source, text)
            self.assertNotEqual(old.translation, text)
            self.assertEqual((new.source, new.translation), (text, text))
        for text in ('togeradi@interfm.jp', 'other.togeradi@internet.jp',
                     'togeradi@internet.jp.example.org'):
            result = self.pm.apply_to_cue(text, text, self.new)
            self.assertEqual((result.source, result.translation), (text, text))
        result = self.pm.apply_to_cue('togeradi@internet.jp', 'togeradio@internet.jp', self.new)
        self.assertEqual((result.source, result.translation), ('togeradi@interfm.jp',)*2)
        self.assertEqual(self.pm.apply_to_cue('インターFM', '', self.new).source, 'InterFM')

    def test_actual_manifest_adjacent_and_nontext(self):
        for a, b in [('other.', 'togeradi@internet.jp'),
                     ('togeradi@internet.jp', '.example.org'),
                     ('other.togeradi@', 'internet.jp')]:
            manifest = {'1': row(a, a), '2': row(b, b), '3': row('インターFM', '')}
            before = copy.deepcopy(manifest)
            report = apply_address_boundary_protection(manifest, self.new, self.pm.apply_to_manifest)
            self.assertEqual(manifest['1'], before['1'])
            self.assertEqual(manifest['2'], before['2'])
            self.assertEqual(manifest['3']['original_subtitle'], 'InterFM')
            self.assertTrue(report['protected_cross_cue_addresses'])
            for key in manifest:
                for field in ('start_time', 'end_time', 'speaker', 'unrelated'):
                    self.assertEqual(manifest[key][field], before[key][field])
        manifest = {'1': row('赤色の', ''), '2': row('共済を', '')}
        apply_address_boundary_protection(manifest, self.new, self.pm.apply_to_manifest)
        self.assertIn('赤色の響祭', manifest['1']['original_subtitle'] + manifest['2']['original_subtitle'])
    def test_other_program_resource_and_unselected_compatibility(self):
        other = {'rules': [{'id': 'other-show', 'field': 'source',
                            'pattern': 'hello@old\\.example', 'replace': 'hello@new.example'}]}
        memory = self.pm.parse_program_memory(other)
        manifest = {'1': row('prefix.', ''), '2': row('hello@old.example', '')}
        before = copy.deepcopy(manifest)
        report = apply_address_boundary_protection(manifest, memory, self.pm.apply_to_manifest)
        self.assertEqual(manifest, before)
        self.assertTrue(report['protected_cross_cue_addresses'])
        direct = copy.deepcopy(before)
        self.pm.apply_to_manifest(direct, memory)
        self.assertNotEqual(direct, before)

        OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            for enabled in (False, True):
                work = Path(tmp) / str(enabled)
                synthetic(work)
                self.assertEqual(run_stage('quality', work)['returncode'], 0)
                resource = Path(tmp) / (str(enabled) + '.json')
                resource.write_text(json.dumps({**other, 'protect_cross_cue_addresses': enabled}), encoding='utf-8')
                self.assertEqual(run_stage('normalize', work, str(resource))['returncode'], 0)
                report = json.loads((work / 'reports/program_memory.json').read_text(encoding='utf-8'))
                self.assertEqual('protected_cross_cue_addresses' in report, enabled)

    def test_real_normalize_cross_cue(self):
        OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            work = Path(tmp) / 'work'
            synthetic(work)
            path = work / 'experiments/mimo-proofread/mimo_proofread_segments.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            first = data['1']
            second = copy.deepcopy(first)
            first.update(original_subtitle='other.', translated_subtitle='other.', start_time=0, end_time=1000)
            second.update(original_subtitle='togeradi@internet.jp', translated_subtitle='togeradi@internet.jp', start_time=1100, end_time=2000)
            data['2'] = second
            path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            with patch('qwen_asr.commands.stages._require_quality_gate_before_formal_output', return_value=0):
                self.assertEqual(run_stage('normalize', work, str(MEMORY_V2))['returncode'], 0)
            before = json.loads((work / 'reports/normalized-before-program-memory.json').read_text(encoding='utf-8'))
            after = json.loads((work / 'normalized_segments.json').read_text(encoding='utf-8'))
            self.assertEqual(before, after)
            report = json.loads((work / 'reports/program_memory.json').read_text(encoding='utf-8'))
            self.assertTrue(report['protected_cross_cue_addresses'])

    def test_real_finish_normalize(self):
        OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            work = Path(tmp) / 'work'
            synthetic(work)
            self.assertEqual(run_stage('realign', work)['business']['candidate_count'], 0)
            self.assertEqual(run_stage('quality', work)['returncode'], 0)
            path = work / 'experiments/mimo-proofread/mimo_proofread_segments.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            data['1']['original_subtitle'] = 'other.togeradi@internet.jp と togeradi@internet.jp'
            data['1']['translated_subtitle'] = 'other.togeradi@internet.jp と togeradio@internet.jp'
            path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            with patch('qwen_asr.commands.stages._require_quality_gate_before_formal_output', return_value=0):
                result = run_stage('normalize', work, str(MEMORY_V2))
            self.assertEqual(result['returncode'], 0)
            before = json.loads((work / 'reports/normalized-before-program-memory.json').read_text(encoding='utf-8'))
            after = json.loads((work / 'normalized_segments.json').read_text(encoding='utf-8'))
            self.assertEqual(list(before), list(after))
            output = ' '.join(x['original_subtitle'] + ' ' + x['translated_subtitle'] for x in after.values())
            self.assertIn('other.togeradi@internet.jp', output)
            self.assertIn('togeradi@interfm.jp', output)
            for key in before:
                for field in ('start_time', 'end_time', 'speaker'):
                    self.assertEqual(after[key].get(field), before[key].get(field))
            report = json.loads((work / 'reports/program_memory.json').read_text(encoding='utf-8'))
            self.assertIn('email-source', report['rule_hit_counts'])


if __name__ == '__main__':
    unittest.main()
