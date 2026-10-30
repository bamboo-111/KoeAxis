"""Explicit bindings for the legacy prepare, ASR, and speaker operations.

This module is a candidate adapter; it does not supply a workflow entry by itself.
"""
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[3] / "compat" / "legacy"


@contextlib.contextmanager
def isolated(work):
    work = Path(work).resolve()
    if work == LEGACY.resolve() or LEGACY.resolve() in work.parents:
        raise ValueError('work directory must not be in legacy repository')
    work.mkdir(parents=True, exist_ok=True)
    names = {'HOME': 'home', 'USERPROFILE': 'home', 'TEMP': 'tmp', 'TMP': 'tmp',
             'APPDATA': 'appdata', 'LOCALAPPDATA': 'localappdata', 'HF_HOME': 'hf-home',
             'HF_HUB_CACHE': 'hf-cache', 'HUGGINGFACE_HUB_CACHE': 'hf-cache',
             'NUMBA_CACHE_DIR': 'numba-cache', 'MPLCONFIGDIR': 'mpl-cache',
             'TORCH_HOME': 'torch-home', 'XDG_CACHE_HOME': 'xdg-cache',
             'XDG_CONFIG_HOME': 'xdg-config', 'XDG_DATA_HOME': 'xdg-data'}
    updates = {key: str(work / value) for key, value in names.items()}
    updates.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                    'HF_DATASETS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
    for name in names.values():
        (work / name).mkdir(parents=True, exist_ok=True)
    before = {key: os.environ.get(key) for key in updates}
    cwd = Path.cwd()
    old_path = list(sys.path)
    os.environ.update(updates)
    sys.path.insert(0, str(LEGACY))
    os.chdir(work)
    try:
        yield
    finally:
        os.chdir(cwd)
        sys.path[:] = old_path
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _existing(value, name, directory=False):
    path = Path(value).resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(name + ' missing')
    return path


def _rows(path):
    rows = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('empty or invalid manifest: ' + str(path))
    ids = [row.get('segment_id') for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or not all(isinstance(x, str) and x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('invalid or duplicate segment IDs')
    return rows


def _options(values, allowed):
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError('options must be a list of strings')
    index = 0
    while index < len(values):
        key = values[index]
        if key not in allowed:
            raise ValueError('unsupported stage option: ' + key)
        index += 1
        if allowed[key]:
            if index == len(values) or values[index].startswith('--'):
                raise ValueError('missing option value: ' + key)
            index += 1
    return values




def run(role, config):
    work = Path(config['work']).resolve()
    if role == 'prepare':
        media = _existing(config['media'], 'media')
        argv = ['prepare', '--workdir', str(work), '--media', str(media), '--no-resume', '--skip-preflight']
        argv.extend(_options(config.get('prepare_options', []), {
            '--vad-backend': True, '--vad-threshold': True, '--max-segment-seconds': True,
            '--min-segment-seconds': True, '--preferred-silence-ms': True,
            '--min-silence-ms': True, '--padding-ms': True, '--overlap-ms': True,
            '--eager-segment-export': False}))
    elif role == 'transcribe':
        _existing(config['model'], 'ASR model', directory=True)
        _existing(config['model_cache'], 'model cache', directory=True)
        argv = ['transcribe', '--workdir', str(work), '--model', str(Path(config['model']).resolve()),
                '--model-cache-dir', str(Path(config['model_cache']).resolve()), '--local-files-only',
                '--no-resume', '--skip-preflight']
        argv.extend(_options(config.get('transcribe_options', []), {
            '--dtype': True, '--device': True, '--batch-size': True, '--batch-mode': True,
            '--batch-order': True, '--max-new-tokens': True, '--language': True,
            '--asr-context-file': True, '--asr-context-mode': True}))
    elif role != 'speaker':
        raise ValueError('unknown media binding')
    with isolated(work):
        from qwen_asr.models import WorkPaths
        paths = WorkPaths.from_workdir(work)
        if role == 'speaker':
            audio = _existing(paths.audio_path, 'prepared audio')
            enrollment = _existing(config['enrollment'], 'speaker enrollment')
            weights = _existing(config['weights'], 'speaker weights')
            from qwen_asr.speaker.voiceprint import ensure_speaker_track
            track = ensure_speaker_track(audio, paths.segments_manifest.parent, enrollment,
                                         speakers=None, weights=weights, device=config['device'])
            artifact = paths.segments_manifest.parent / 'speaker_scores.npz'
            _existing(artifact, 'speaker score cache')
            return {'status': 'completed', 'members': list(track.members), 'artifact': str(artifact)}
        from qwen_asr.cli import build_parser
        args = build_parser().parse_args(argv)
        code = args.func(args, paths)
        if code != 0:
            raise RuntimeError(role + ' returned nonzero')
        if role == 'prepare':
            rows = _rows(paths.segments_manifest)
            _existing(paths.audio_path, 'prepared audio')
            return {'status': 'completed', 'count': len(rows), 'artifact': str(paths.segments_manifest),
                    'audio': str(paths.audio_path)}
        segments = _rows(paths.segments_manifest)
        rows = _rows(paths.transcript_manifest)
        expected = [row['segment_id'] for row in segments]
        if [row['segment_id'] for row in rows] != expected or any(row.get('status') != 'completed' for row in rows):
            raise RuntimeError('ASR output incomplete or reordered')
        raw = paths.raw_transcript_manifest
        if raw.exists():
            raise RuntimeError('raw transcript already exists')
        with raw.open('xb') as target, paths.transcript_manifest.open('rb') as source:
            shutil.copyfileobj(source, target)
        return {'status': 'completed', 'count': len(rows), 'artifact': str(raw)}
