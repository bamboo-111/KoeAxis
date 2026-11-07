"""Explicit bindings for the legacy prepare, ASR, and speaker operations.

This module is a candidate adapter; it does not supply a workflow entry by itself.
"""
import contextlib
import json
import os
import shutil
import sys
import math
import subprocess
from fractions import Fraction
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
    def restore_bindings():
        sys.path[:] = old_path
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    # Register before mutation, including chdir: entry failures must also unwind.
    # ExitStack keeps restoring bindings even if restoring cwd itself fails.
    with contextlib.ExitStack() as scope:
        scope.callback(restore_bindings)
        scope.callback(os.chdir, cwd)
        os.environ.update(updates)
        sys.path.insert(0, str(LEGACY))
        os.chdir(work)
        yield


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




def source_metadata(media, device):
    """Probe once before prepare; retain original video geometry across audio stages."""
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration:stream=codec_type,width,height,sample_aspect_ratio:stream_disposition=attached_pic:stream_tags=rotate:stream_side_data=rotation',
        '-of', 'json', str(media)], capture_output=True, text=True, encoding='utf-8',
        timeout=30, check=True)
    data = json.loads(result.stdout)
    try:
        duration = float(data.get('format', {}).get('duration'))
    except (TypeError, ValueError):
        duration = 0
    duration = duration if math.isfinite(duration) and duration > 0 else None
    video = None
    for stream in data.get('streams', []):
        if stream.get('codec_type') != 'video' or stream.get('disposition', {}).get('attached_pic'):
            continue
        width, height = int(stream['width']), int(stream['height'])
        if width <= 0 or height <= 0:
            raise ValueError('invalid source video dimensions')
        sar = stream.get('sample_aspect_ratio') or '1:1'
        ratio = Fraction(sar.replace(':', '/')) if sar not in ('N/A', '0:1', '') else Fraction(1)
        if ratio <= 0:
            raise ValueError('invalid source sample aspect ratio')
        display_width = max(1, round(width * ratio))
        rotation = float(stream.get('tags', {}).get('rotate', 0))
        for side in stream.get('side_data_list', []):
            if 'rotation' in side:
                rotation = float(side['rotation'])
        if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 0.01:
            raise ValueError('unsupported source video rotation')
        rotated = int(round(rotation / 90)) % 2
        video = {'width': height if rotated else display_width,
                 'height': display_width if rotated else height,
                 'coded_width': width, 'coded_height': height,
                 'sample_aspect_ratio': sar, 'rotation': rotation}
        break
    return {'schema': 1, 'media_path': str(Path(media).resolve()), 'duration_s': duration,
            'device': device, 'video': video}


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
        if role == 'prepare':
            metadata = source_metadata(media, config.get('device', 'cpu'))
            temporary = work / 'media-source.json.tmp'
            temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
                                 encoding='utf-8', newline='\n')
            os.replace(temporary, work / 'media-source.json')
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
