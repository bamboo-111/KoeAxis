"""Fixed Host root and explicit media stage bindings; not yet production verified."""
import contextlib
import hashlib
import importlib.util
import os
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from host.contracts import ModuleError, artifact, ref, returned

VERSION = '0.1.0-real-chain-runner'
ROOT = Path(__file__).resolve().parents[3]


def _copy(source, destination, output):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if output not in source.parents or output not in destination.parents:
        raise ModuleError('InvalidArtifact', 'stage artifact is outside Host output')
    if not source.is_file() or not source.stat().st_size:
        raise ModuleError('MissingArtifact', str(source))
    if source == destination:
        return
    if destination.exists():
        raise ModuleError('ArtifactExists', str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _reported_path(report, output):
    value = report.get('artifact')
    if not isinstance(value, str) or not value:
        raise ModuleError('MissingArtifact', 'stage report has no artifact')
    path = Path(value)
    return (path if path.is_absolute() else output / path).resolve()


def _write_ref(locator):
    value = artifact(locator, owner='workflow.pipeline')
    value['writer_operation'] = ref('stage.correct', 'run_bound', VERSION)
    return value


@contextlib.contextmanager
def _suspend_ordinary_legacy_imports():
    """Temporarily free captured package names for Host's verified closure.

    Media adapters import the legacy packages normally; the correction adapter
    instead constructs a captured-byte closure that rejects occupied names.
    Restore the exact ordinary modules after the synchronous Host call.
    """
    from host.runtime import _REGISTRY_LOCK
    with _REGISTRY_LOCK:
        names = {name: module for name, module in sys.modules.items()
                 if name.split('.')[0] in ('qwen_asr', 'optimizer')}
        for name in names:
            del sys.modules[name]
        try:
            yield
        finally:
            conflicts = [name for name in names if name in sys.modules]
            sys.modules.update(names)
            if conflicts:
                raise RuntimeError('legacy closure left occupied names: ' + ', '.join(conflicts))


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _input_signature(config, stats=None):
    clean = {key: value for key, value in config.items() if key != 'retry'}
    resources = clean['align_resources']
    paths = {name: clean[name] for name in ('media', 'model', 'model_cache', 'enrollment', 'weights', 'memory')}
    if clean.get('ass_reference'):
        paths['ass_reference'] = clean['ass_reference']
    paths.update({'align.' + name: value for name, value in resources.items()})
    options = clean.get('transcribe_options', [])
    if '--asr-context-file' in options:
        index = options.index('--asr-context-file')
        if index + 1 >= len(options):
            raise ModuleError('InvalidConfig', 'missing asr context file')
        paths['asr-context-file'] = options[index + 1]
    entries = {}
    # Scope this cache to this single content verification, never across runs.
    digests = {}
    counts = {'reads': 0, 'reuse': 0, 'read_seconds': 0.0}

    def digest_file(path):
        path = path.resolve()
        if path not in digests:
            started = time.perf_counter()
            try:
                digests[path] = _sha_file(path)
            finally:
                counts['reads'] += 1
                counts['read_seconds'] += time.perf_counter() - started
        else:
            counts['reuse'] += 1
        return digests[path]
    for name, value in sorted(paths.items()):
        if not isinstance(value, str):
            raise ModuleError('InvalidConfig', 'invalid input path: ' + name)
        path = Path(value).resolve()
        if path.is_file():
            entries[name] = {'path': str(path), 'sha256': digest_file(path)}
        elif path.is_dir():
            files = {}
            for item in sorted(path.rglob('*')):
                if item.is_file():
                    files[item.relative_to(path).as_posix()] = digest_file(item)
            entries[name] = {'path': str(path), 'files': files}
        else:
            raise ModuleError('InvalidConfig', 'missing input path: ' + name)
    env_names = sorted(name for name in os.environ if name.startswith(('GEMINI_', 'MIMO_'))
                       or name in ('REAL_LLM_STAGES_ONLINE', 'REAL_MIMO_ONLINE', 'OPT54_CONTENT_CONTEXT'))
    environment = {name: os.environ[name] for name in env_names}
    encoded = json.dumps({'config': clean, 'inputs': entries, 'environment': environment}, ensure_ascii=True,
                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if stats is not None:
        stats.update(counts)
    return hashlib.sha256(encoded).hexdigest()


def _save_state(path, state):
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(state, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _verify_attempts(output, state):
    for attempt in state.get('attempts', []):
        for stage in attempt.get('stages', []):
            for relative, expected in stage['artifacts'].items():
                path = (output / relative).resolve()
                if not path.is_relative_to(output) or not path.is_file() or _sha_file(path) != expected:
                    raise ModuleError('ArtifactChanged', 'completed artifact changed: ' + relative)


def _alignment_timing(output):
    path = output / 'work/alignment/alignment_timing.json'
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _performance_module():
    spec = importlib.util.spec_from_file_location('_chain_performance', Path(__file__).with_name('performance.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _progress(output, attempt, status):
    work = output / 'work'
    work.mkdir(parents=True, exist_ok=True)
    done = len(attempt['stages'])
    progress = {'stage': 'fixed-chain', 'status': status,
                'done': done, 'total': 12, 'current': attempt['stages'][-1]['name'] if done else None,
                'updated_at': datetime.now(timezone.utc).isoformat(), 'summary': 'fixed chain ' + status}
    _save_state(work / 'progress.json', progress)
    _save_state(output / 'progress.json', progress)

@contextlib.contextmanager
def _content_environment(config):
    selected = config.get('content_context', os.environ.get('OPT54_CONTENT_CONTEXT', ''))
    if selected not in ('', 'togeradi-v1'):
        raise ModuleError('InvalidConfig', 'unknown content_context')
    previous = os.environ.get('OPT54_CONTENT_CONTEXT')
    if selected:
        os.environ['OPT54_CONTENT_CONTEXT'] = selected
    else:
        os.environ.pop('OPT54_CONTENT_CONTEXT', None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('OPT54_CONTENT_CONTEXT', None)
        else:
            os.environ['OPT54_CONTENT_CONTEXT'] = previous


def run_root(arguments, api):
    # This candidate cannot support concurrent roots in one process: ordinary
    # legacy imports coexist with Host's captured-byte legacy closure.
    from host.runtime import _REGISTRY_LOCK
    perfmod = _performance_module()
    enabled = arguments.get('config', {}).get('performance_enabled', True)
    if type(enabled) is not bool:
        raise ModuleError('InvalidConfig', 'performance_enabled must be boolean')
    perf = perfmod.Performance(sampler=perfmod.sample_resources, enabled=enabled)
    try:
        perf.start_sampling()
    except Exception:
        pass  # Measurement must not prevent business execution.
    if not _REGISTRY_LOCK.acquire(blocking=False):
        try:
            perf.stop_sampling()
        except Exception:
            pass
        raise ModuleError('ConcurrentRoot', 'another legacy operation is active in this process')
    host_output = None
    output = None
    hash_stats = {}
    outcome = 'failed'
    try:
        config = arguments['config']
        if not isinstance(config, dict):
            raise ModuleError('InvalidConfig', 'config must be an object')
        for name in ('media', 'model', 'model_cache', 'enrollment', 'weights', 'device', 'memory'):
            if not isinstance(config.get(name), str) or not config[name]:
                raise ModuleError('InvalidConfig', 'missing ' + name)
        if not isinstance(config.get('align_resources'), dict):
            raise ModuleError('InvalidConfig', 'missing align_resources')
        retry = config.get('retry', False)
        if not isinstance(retry, bool):
            raise ModuleError('InvalidConfig', 'retry must be boolean')
        host_output_name = api.snapshot()['environment']['output']
        host_output = (ROOT / host_output_name).resolve()
        host_output.mkdir(parents=True, exist_ok=True)
        checkpoint = host_output / 'fixed-chain-state.json'
        started = time.perf_counter()
        try:
            signature = _input_signature(config, hash_stats)
        finally:
            try:
                perf.stage('input_signature', time.perf_counter() - started)
            except Exception:
                pass
        if retry:
            if not checkpoint.is_file():
                raise ModuleError('InvalidResume', 'no previous attempt')
            state = json.loads(checkpoint.read_text(encoding='utf-8'))
            if state.get('signature') != signature:
                raise ModuleError('InputChanged', 'input or configuration digest differs')
            _verify_attempts(host_output, state)
            if not state['attempts'] or state['attempts'][-1]['status'] != 'failed':
                raise ModuleError('InvalidResume', 'only a failed attempt may be retried')
        else:
            if checkpoint.exists():
                raise ModuleError('InvalidResume', 'existing attempt; explicit retry required')
            state = {'schema': 1, 'signature': signature, 'attempts': []}
        number = len(state['attempts']) + 1
        relative = '' if number == 1 else 'attempts/%04d' % number
        output_name = host_output_name if not relative else str(Path(host_output_name) / relative)
        output = (ROOT / output_name).resolve()
        attempt = {'number': number, 'directory': relative or '.', 'status': 'running', 'stages': []}
        state['attempts'].append(attempt)
        _save_state(checkpoint, state)
        try:
            with _content_environment(config):
                result = _run_attempt(arguments, api, output_name, host_output, checkpoint, state, attempt, perf)
            _verify_attempts(host_output, state)
        except BaseException:
            attempt['status'] = 'failed'
            _save_state(checkpoint, state)
            _progress(output, attempt, 'failed')
            raise
        attempt['status'] = 'completed'
        _save_state(checkpoint, state)
        _progress(output, attempt, 'completed')
        outcome = 'completed'
        result.update(attempt=number, input_signature=signature)
        return result
    finally:
        try:
            perf.stop_sampling()
        except Exception:
            pass
        if enabled and output is not None:
            try:
                target = output
                target.mkdir(parents=True, exist_ok=True)
                report = perf.report()
                alignment = _alignment_timing(target)
                calls = perfmod.summarize_calls(target)
                report.update(schema=1, status=outcome, input_hash=hash_stats,
                              calls=calls, alignment_timing=alignment,
                              model_measurement='partial: Host stage and alignment boundaries; model internals unknown',
                              measurement_coverage={
                                  'stage_wall': {'source': 'Host api.call monotonic duration', 'count': sum(s['name'] != 'input_signature' for s in report['stages']), 'scope': 'whole stage including adapter overhead'},
                                  'alignment': {'source': 'alignment_timing.json', 'count': int(alignment is not None), 'missing_reason': None if alignment is not None else 'alignment timing file unavailable'},
                                  'model_load_seconds': {'value': None, 'count': 0, 'missing_reason': 'legacy model lifecycle not exposed at Host boundary'},
                                  'model_inference_seconds': {'value': None, 'count': 0, 'missing_reason': 'Host stage duration includes non-inference work'},
                                  'model_release_seconds': {'value': None, 'count': 0, 'missing_reason': 'legacy release boundary not exposed'},
                                  'sdk_retry_count': {'value': None, 'count': 0, 'missing_reason': 'SDK internal HTTP attempts not observable'},
                                  'recorded_requests': {'source': 'stored call records', 'count': calls['network_requests'], 'scope': 'wrapper requests, not SDK HTTP attempts'},
                                  'concurrency': {'source': 'recorded wrapper request intervals', 'count': calls['interval_known'], 'scope': 'not server concurrency'},
                                  'resources': {'source': report['resources']['sampling_method'], 'count': report['resources']['sample_count'], 'scope': 'RSS current process; GPU sampled allocated bytes when already initialized'}})
                _save_state(target / 'performance.json', report)
            except Exception:
                pass
        _REGISTRY_LOCK.release()


def _run_attempt(arguments, api, output_name, host_output, checkpoint, state, attempt, perf):
    config = arguments['config']
    output = (ROOT / output_name).resolve()
    work = output / 'work'
    reports = []

    def mark(stage, paths):
        artifacts = {}
        for path in paths:
            path = Path(path).resolve()
            if not path.is_relative_to(output) or not path.is_file() or not path.stat().st_size:
                raise ModuleError('MissingArtifact', str(path))
            artifacts[path.relative_to(host_output).as_posix()] = _sha_file(path)
        attempt['stages'].append({'name': stage, 'artifacts': artifacts})
        _save_state(checkpoint, state)
        _progress(output, attempt, 'running')

    def call(module, operation, args, artifacts=None):
        target = ref(module, operation, VERSION)
        started = time.perf_counter()
        started_unix = time.time()
        try:
            response = api.call(target, args, artifacts) if artifacts is not None else api.call(target, args)
        finally:
            try:
                perf.timed_stage(module, started_unix, time.time(), time.perf_counter() - started)
            except Exception:
                pass
        report = returned(response).get('report')
        if not isinstance(report, dict):
            raise ModuleError('InvalidReport', module)
        reports.append({'stage': module, 'report': report})
        return report

    def completed(report, stage):
        if report.get('status') != 'completed':
            raise ModuleError('StageIncomplete', stage)

    media_config = dict(config, work=str(work))
    prepare = call('stage.prepare', 'run_bound', {'config': media_config})
    completed(prepare, 'prepare')
    if _reported_path(prepare, output) != work / 'segments.json' or Path(prepare.get('audio', '')).resolve() != work / 'audio.wav':
        raise ModuleError('InvalidArtifact', 'prepare paths')
    _copy(work / 'segments.json', work / 'segments.json', output)
    _copy(work / 'audio.wav', work / 'audio.wav', output)
    mark('stage.prepare', [work / 'segments.json', work / 'audio.wav'])
    transcribe = call('stage.transcribe', 'run_bound', {'config': media_config})
    completed(transcribe, 'transcribe')
    raw = _reported_path(transcribe, output)
    if raw != work / 'transcript_segments.raw.json':
        raise ModuleError('InvalidArtifact', 'raw transcript path')
    _copy(raw, raw, output)
    mark('stage.transcribe', [raw])

    prefix = output.relative_to(host_output).as_posix()
    refs = [_write_ref((prefix + '/' if prefix != '.' else '') + 'correct/' + name)
            for name in ('raw-backup.json', 'transcript_segments.json', 'corrected_segments.json')]
    with _suspend_ordinary_legacy_imports():
        correct = call('stage.correct', 'run_bound', {'mode': 'rules', 'raw_source': str(raw),
            'raw_backup': refs[0], 'transcript_manifest': refs[1], 'report_manifest': refs[2]}, refs)
    if (correct.get('failed') != 0 or correct.get('load_transcripts') is not True or
            correct.get('input_preserved') is not True or not isinstance(correct.get('corrected_count'), int) or
            correct['corrected_count'] <= 0):
        raise ModuleError('StageIncomplete', 'correct')
    transcript = output / 'correct/transcript_segments.json'
    _copy(transcript, work / 'transcript_segments.corrected.json', output)
    mark('stage.correct', [output / 'correct' / name for name in
                           ('raw-backup.json', 'transcript_segments.json', 'corrected_segments.json')])
    align = call('stage.align', 'run_bound', {'mode': 'real', 'transcript_source': str(transcript),
        'raw_source': str(raw), 'work_dir': str(work / 'alignment'), 'source_audio': str(work / 'audio.wav'),
        'resources': config['align_resources']})
    if align.get('failed') != 0 or align.get('count') != correct['corrected_count']:
        raise ModuleError('StageIncomplete', 'align')
    aligned = Path(align.get('aligned_path', '')).resolve()
    if aligned != work / 'alignment/aligned_segments.json':
        raise ModuleError('InvalidArtifact', 'alignment path')
    _copy(aligned, work / 'aligned_segments.json', output)
    mark('stage.align', [aligned])
    speaker = call('stage.speaker', 'run_bound', {'config': media_config})
    completed(speaker, 'speaker')
    if _reported_path(speaker, output) != work / 'speaker_scores.npz':
        raise ModuleError('InvalidArtifact', 'speaker score path')
    _copy(work / 'speaker_scores.npz', work / 'speaker_scores.npz', output)
    mark('stage.speaker', [work / 'speaker_scores.npz'])
    os.replace(work / 'transcript_segments.corrected.json', work / 'transcript_segments.json')
    split = call('stage.split', 'run_bound', {'mode': 'online', 'repeat': 'a', 'output': output_name,
        'upstream': str(work), 'speaker_resources': {'enroll': str(Path(config['enrollment']).resolve()),
        'npz': str(work / 'speaker_scores.npz'), 'audio': str(work / 'audio.wav')}})
    if split.get('business_complete') is not True:
        raise ModuleError('StageIncomplete', 'split', partial={'report': split})
    _copy(_reported_path(split, output), work / 'split_segments.json', output)
    mark('stage.split', [_reported_path(split, output), work / 'split_segments.json'])
    translate = call('stage.translate', 'run_bound', {'mode': 'online', 'repeat': 'a', 'output': output_name, 'upstream': str(work)})
    if translate.get('business_complete') is not True:
        raise ModuleError('StageIncomplete', 'translate')
    _copy(_reported_path(translate, output), work / 'translated_segments.json', output)
    mark('stage.translate', [_reported_path(translate, output), work / 'translated_segments.json'])
    mimo = call('stage.audio_review', 'run_bound', {'mode': 'online', 'phase': 'full', 'output': output_name, 'upstream': str(work)})
    if mimo.get('phase') != 'full' or mimo.get('business_ok') is not True:
        raise ModuleError('StageIncomplete', 'audio_review')
    _copy(_reported_path(mimo, output), work / 'experiments/mimo-proofread/mimo_proofread_segments.json', output)
    mimo_dir = _reported_path(mimo, output).parent
    for name in ('mimo_proofread_report.json', 'mimo_stage1_text_suspect_report.json',
                 'mimo_stage2_nearby_audio_report.json'):
        _copy(mimo_dir / name, work / 'experiments/mimo-proofread' / name, output)
    mark('stage.audio_review', [mimo_dir / name for name in
         ('mimo_proofread_segments.json', 'mimo_proofread_report.json',
          'mimo_stage1_text_suspect_report.json', 'mimo_stage2_nearby_audio_report.json')])

    for module in ('stage.proofread_realign', 'quality', 'stage.normalize', 'stage.export'):
        report = call(module, 'run', {'work': str(work), 'memory': config['memory'], 'speaker': config['enrollment'], 'ass_reference': config.get('ass_reference')})
        if report.get('returncode') != 0 or (module in ('stage.proofread_realign', 'quality') and
                (not isinstance(report.get('business'), dict) or report['business'].get('status') == 'FAIL')):
            raise ModuleError('StageIncomplete', module, partial={'report': report})
        if module == 'stage.export' and (not isinstance(report.get('post_export_quality'), dict)
                                         or report['post_export_quality'].get('status') == 'FAIL'):
            raise ModuleError('StageIncomplete', 'post-export quality', partial={'report': report})
        _copy(_reported_path(report, output), _reported_path(report, output), output)
        paths = [_reported_path(report, output)]
        if module == 'quality':
            quality_snapshot = output / 'checkpoint-artifacts/quality.json'
            _copy(paths[0], quality_snapshot, output)
            paths = [quality_snapshot]
        if module == 'stage.export':
            paths.append(work / 'export/subtitles.ass')
            paths.append(work / 'final_quality.json')
        mark(module, paths)
    return {'status': 'completed', 'stages': reports}


class Implementation:
    def __init__(self, settings):
        self.role = settings['role']

    def invoke(self, operation, arguments, api):
        if self.role != 'root':
            if operation != 'run_bound':
                raise ModuleError('UnknownOperation', operation)
            try:
                config = arguments['config']
                output = (ROOT / api.snapshot()['environment']['output']).resolve()
                work = Path(config['work']).resolve()
            except (KeyError, TypeError, ValueError) as exc:
                raise ModuleError('InvalidConfig', 'explicit work directory required') from exc
            if work == output or output not in work.parents:
                raise ModuleError('InvalidWorkDirectory', 'work must be inside this Host output')
            try:
                path = Path(__file__).resolve().parent / 'media.py'
                spec = importlib.util.spec_from_file_location('_chain_runner_media', path)
                media = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(media)
                return {'report': media.run(self.role, config)}, []
            except Exception as exc:
                raise ModuleError('MediaStageFailed', type(exc).__name__ + ': ' + str(exc)) from exc
        if operation != 'run' or arguments.get('plan_id') != 'CHAIN-READY-001':
            raise ModuleError('InvalidPlan', 'invalid fixed chain entry')
        return {'report': run_root(arguments, api)}, []
