"""Host entry and stage adapters for the finish chain."""
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from host.contracts import ModuleError, ref, returned

VERSION = '0.1.0-real-chain-finish'
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STAGES = [('stage.proofread_realign', 'realign'), ('quality', 'quality'),
          ('stage.normalize', 'normalize'), ('stage.export', 'export')]


def load(name):
    spec = importlib.util.spec_from_file_location('_finish_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def synthetic(work):
    work.mkdir(parents=True, exist_ok=False)
    row = {'start_time': 0, 'end_time': 2000, 'original_subtitle': 'こんにちは', 'translated_subtitle': '你好',
           'speaker': '', 'asr_suspect': False, 'needs_audio_review': False, 'suspect_types': [],
           'suspect_reason': '', 'suspect_confidence': 0.95}
    transcript = [{'segment_id': 's1', 'audio_path': 'segment.wav', 'global_start_time': 0.0,
                   'global_end_time': 2.0, 'text': 'こんにちは', 'status': 'completed'}]
    aligned = [{'segment_id': 's1', 'global_start_time': 0.0, 'global_end_time': 2.0, 'status': 'completed',
                'tokens': [{'text': 'こんにちは', 'start_time': 0.0, 'end_time': 2.0}]}]
    for name, value in [('transcript_segments.json', transcript), ('aligned_segments.json', aligned),
                        ('split_segments.json', {'1': row}), ('translated_segments.json', {'1': row}),
                        ('experiments/mimo-proofread/mimo_proofread_segments.json', {'1': row})]:
        write_json(work / name, value)
    (work / 'reports').mkdir(exist_ok=True)
    (work / 'export').mkdir(exist_ok=True)
    memory = work / 'memory.json'
    write_json(memory, {'rules': [{'id': 'synthetic', 'field': 'translation', 'pattern': '你好', 'replace': '您好'}]})
    return str(memory)


def manifest_resources(manifest):
    resources = {}
    for item in manifest.get('resources', []):
        purpose = item['purpose']
        if purpose in resources:
            raise ValueError('duplicate resource purpose: ' + purpose)
        resources[purpose] = item
    for purpose in ('source-audio', 'speaker-enrollment', 'speaker-score-cache'):
        if purpose not in resources:
            raise ValueError('missing resource purpose: ' + purpose)
    audio = resources['source-audio']
    if (Path(audio['path']).resolve() != Path(manifest['source_audio']['path']).resolve()
            or audio['sha256'] != manifest['source_audio']['sha256']):
        raise ValueError('source audio resource does not match handoff')
    return resources


def prepare_current(manifest, work):
    resources = manifest_resources(manifest)
    source = Path(manifest['workdir']).resolve()
    work.mkdir(parents=True, exist_ok=False)
    names = {}
    for name in manifest['files']:
        relative = Path(name)
        basename = relative.name
        if basename in names:
            raise ValueError('duplicate handoff basename: ' + basename)
        names[basename] = True
        if basename.startswith('mimo_proofread_'):
            target = work / 'experiments/mimo-proofread' / basename
        else:
            target = work / basename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        load('handoff')._check(target, manifest['files'][name])
    (work / 'reports').mkdir(exist_ok=True)
    (work / 'export').mkdir(exist_ok=True)
    shutil.copy2(resources['source-audio']['path'], work / 'audio.wav')
    load('handoff')._check(work / 'audio.wav', resources['source-audio']['sha256'])
    shutil.copyfile(resources['speaker-score-cache']['path'], work / 'speaker_scores.npz')
    load('handoff')._check(work / 'speaker_scores.npz', resources['speaker-score-cache']['sha256'])
    memory = load('legacy').OLD / 'samples/program-memory-togeradi.json'
    return str(memory), resources['speaker-enrollment']['path'], resources


def comparison(work):
    current = work / 'normalized_segments.json'
    baseline = load('legacy').OLD / 'workspaces/radio-54-togeradi-20261018/manifests/normalized_segments.json'
    if not current.is_file() or not baseline.is_file():
        return {'status': 'NOT_ASSESSED'}
    a = json.loads(current.read_text(encoding='utf-8'))
    b = json.loads(baseline.read_text(encoding='utf-8'))
    fields = ['start_time', 'end_time', 'original_subtitle', 'translated_subtitle', 'speaker']
    common = set(a) & set(b)
    return {'status': 'compared_by_cue_id', 'current_count': len(a), 'baseline_count': len(b),
            'shared_ids': len(common), 'current_only': len(set(a) - set(b)), 'baseline_only': len(set(b) - set(a)),
            'field_differences': {field: sum(a[k].get(field) != b[k].get(field) for k in common) for field in fields},
            'current_sha256': hashlib.sha256(current.read_bytes()).hexdigest(),
            'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest()}


class Implementation:
    def __init__(self, settings):
        self.role = settings['role']

    def invoke(self, operation, arguments, api):
        if self.role != 'root':
            if operation != 'run':
                raise ModuleError('UnknownOperation', operation)
            result = load('legacy').run_stage(self.role, arguments['work'], arguments['memory'] or None,
                                               arguments['speaker'] or None, arguments.get('ass_reference') or None)
            return {'report': result}, []
        if operation != 'run' or arguments['plan_id'] != 'REAL-CHAIN-FINISH-001':
            raise ModuleError('InvalidPlan', str(arguments))
        mode = api.environment('REAL_CHAIN_FINISH_MODE', '') or 'offline'
        if mode not in ('offline', 'real'):
            raise ModuleError('InvalidMode', mode)
        output = (ROOT / api.snapshot()['environment']['output']).resolve()
        work = output / 'work'
        if mode == 'real':
            handoff = api.environment('REAL_CHAIN_FINISH_HANDOFF', '') or ''
            expected_sha = api.environment('REAL_CHAIN_FINISH_HANDOFF_SHA256', '') or ''
            if not handoff or len(expected_sha) != 64 or any(c not in '0123456789abcdef' for c in expected_sha):
                raise ModuleError('InvalidHandoff', 'explicit handoff and SHA256 required')
            load('handoff')._check(Path(handoff), expected_sha)
            manifest = load('handoff').validate_handoff(handoff, current=True)
            memory, speaker, resources = prepare_current(manifest, work)
        else:
            memory = synthetic(work)
            speaker = None
        reports = []
        for module, name in STAGES:
            child = returned(api.call(ref(module, 'run', VERSION), {'work': str(work), 'memory': memory, 'speaker': speaker, 'ass_reference': None}))['report']
            reports.append(child)
            if child['returncode'] != 0:
                break
            if name in ('realign', 'quality') and child['business']['status'] == 'FAIL':
                break
        passed = len(reports) == 4 and all(r['returncode'] == 0 for r in reports)
        checks = [{'id': 'four_stages', 'passed': passed},
                  {'id': 'output_nonempty', 'passed': (work / 'export/subtitles.srt').is_file()},
                  {'id': 'memory_report', 'passed': (work / 'reports/program_memory.json').is_file()}]
        stable = {'mode': mode, 'checks': checks, 'stage_order': [r['stage'] for r in reports],
                  'chain_passed': False, 'migration_equivalence': 'NOT_ASSESSED'}
        write_json(output / 'evidence/cases.json', stable)
        write_json(output / 'evidence/report.json', {'passed': sum(c['passed'] for c in checks),
                   'failed': sum(not c['passed'] for c in checks), 'case_count': len(checks), **stable})
        write_json(output / 'evidence/criteria.json', {'chain_passed': False, 'migration_equivalence': 'NOT_ASSESSED'})
        if mode == 'real':
            artifacts = {}
            for name in ('normalized_segments.json', 'reports/normalized-before-program-memory.json',
                         'reports/program_memory.json', 'reports/proofread_realign.json',
                         'subtitles.srt', 'export/subtitles.srt', 'export/subtitles.ass', 'speaker_scores.npz'):
                path = work / name
                if path.is_file():
                    artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(output / 'evidence/stage_reports.json', {'stages': reports, 'business_complete': passed,
                       'handoff_path': handoff, 'handoff_sha256': expected_sha, 'resources': resources,
                       'program_memory': {'path': memory, 'sha256': hashlib.sha256(Path(memory).read_bytes()).hexdigest()},
                       'input_files': manifest['files'], 'audio_sha256': manifest['source_audio']['sha256'],
                       'artifact_sha256': artifacts, 'baseline_comparison': comparison(work)})
        if not passed or not all(c['passed'] for c in checks):
            raise ModuleError('FinishFailed', 'stage failed or did not produce output', partial={'reports': reports})
        return {'report': stable}, []
