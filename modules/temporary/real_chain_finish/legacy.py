"""Bindings to the legacy stage commands; no subtitle algorithm is reimplemented here."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import re
from pathlib import Path

OLD = Path(__file__).resolve().parents[3] / "compat" / "legacy"
PROJECT = Path(__file__).resolve().parents[3]
GPU_LOCK = PROJECT / 'runs/experiments/REAL-CHAIN-ALIGN-ON/gpu.lock'
ALIGN_MODEL = PROJECT / 'models/aligner'


def _legacy_imports():
    if str(OLD) not in sys.path:
        sys.path.insert(0, str(OLD))
    from qwen_asr.cli import build_parser
    from qwen_asr.models import WorkPaths
    return build_parser, WorkPaths


@contextlib.contextmanager
def isolated_environment(work):
    work = Path(work)
    dirs = {'HOME': 'home', 'USERPROFILE': 'home', 'TEMP': 'tmp', 'TMP': 'tmp',
            'APPDATA': 'appdata', 'LOCALAPPDATA': 'localappdata', 'HF_HOME': 'hf-home',
            'HF_HUB_CACHE': 'hf-cache', 'HUGGINGFACE_HUB_CACHE': 'hf-cache',
            'NUMBA_CACHE_DIR': 'numba-cache', 'MPLCONFIGDIR': 'mpl-cache', 'TORCH_HOME': 'torch-home',
            'XDG_CACHE_HOME': 'xdg-cache', 'XDG_CONFIG_HOME': 'xdg-config', 'XDG_DATA_HOME': 'xdg-data'}
    updates = {key: str(work / value) for key, value in dirs.items()}
    updates.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                    'PYTHONDONTWRITEBYTECODE': '1'})
    for value in dirs.values():
        (work / value).mkdir(parents=True, exist_ok=True)
    before = {key: os.environ.get(key) for key in updates}
    cwd = Path.cwd()
    os.environ.update(updates)
    os.chdir(work)
    try:
        yield
    finally:
        os.chdir(cwd)
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def mfa_externalized(work):
    from qwen_asr import mfa_environment
    from qwen_asr.mfa_experiment import _project_mfa_root
    del _project_mfa_root
    target = Path(work) / 'mfa-root'
    for name in ('dictionary/japanese_mfa.dict', 'acoustic/japanese_mfa.zip'):
        source = OLD / 'tools/mfa-root/pretrained_models' / name
        destination = target / 'pretrained_models' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    exe = OLD / 'tools/mfa-env/Scripts/mfa.exe'
    if not exe.is_file():
        raise RuntimeError('MFA executable not available')
    original_root, original_command = mfa_environment.find_project_mfa_root, mfa_environment.build_project_mfa_command
    old_path = os.environ.get('PATH', '')
    old_mfa = os.environ.get('MFA_ROOT_DIR')
    envroot = OLD / 'tools/mfa-env'
    os.environ['PATH'] = os.pathsep.join([str(envroot / p) for p in ('Library/bin', 'Library/usr/bin', 'Library/mingw-w64/bin', 'Scripts', 'bin', '.')] + [old_path])
    os.environ['MFA_ROOT_DIR'] = str(target)
    mfa_environment.find_project_mfa_root = lambda *args: target
    mfa_environment.build_project_mfa_command = lambda *args: ([str(exe)], str(exe), 'direct')
    try:
        yield
    finally:
        mfa_environment.find_project_mfa_root, mfa_environment.build_project_mfa_command = original_root, original_command
        os.environ['PATH'] = old_path
        if old_mfa is None:
            os.environ.pop('MFA_ROOT_DIR', None)
        else:
            os.environ['MFA_ROOT_DIR'] = old_mfa


def human_reference_diagnostic(work, reference):
    """Read-only cue diagnostic, independent of legacy CLI progress and gate."""
    import importlib.util
    paths = [Path(reference), work / 'split_segments.json', work / 'normalized_segments.json', work / 'export/subtitles.ass']
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        return {'schema': 'gold54-human-reference-diagnostic/v2', 'verdict': 'NOT_ASSESSED', 'missing': missing}
    source = PROJECT / 'modules/temporary/verification_gold54/evaluate.py'
    spec = importlib.util.spec_from_file_location('_finish_gold54_diagnostic', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_v2(*paths)
_ADDRESS = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+\s*@\s*[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_MASK = {i: 0xF0000 + i for i in range(128)}
_UNMASK = {value: key for key, value in _MASK.items()}
def apply_address_boundary_protection(manifest, memory, legacy_apply):
    """Protect addresses crossing an original adjacent-cue boundary before legacy processing.

    Ambiguous cross-cue addresses are left unchanged; the legacy algorithm remains
    responsible for all other rules and for rebuilding its ordinary change report.
    """
    items = [(str(k), v) for k, v in manifest.items() if isinstance(v, dict)]
    fields = ('original_subtitle', 'translated_subtitle')
    if any(any(0xF0000 <= ord(c) < 0xF0080 for c in str(item.get(field) or ''))
           for _, item in items for field in fields):
        raise ValueError('reserved program-memory mask characters in input')
    protected = {}
    for (lk, left), (rk, right) in zip(items, items[1:]):
        for field in fields:
            a, b = str(left.get(field) or ''), str(right.get(field) or '')
            for match in _ADDRESS.finditer(a + b):
                if match.start() < len(a) < match.end():
                    protected.setdefault((lk, field), set()).update(range(match.start(), len(a)))
                    protected.setdefault((rk, field), set()).update(range(0, match.end() - len(a)))
    for key, item in items:
        for field in fields:
            indexes = protected.get((key, field), set())
            if indexes:
                value = str(item.get(field) or '')
                item[field] = ''.join(chr(_MASK[ord(c)]) if i in indexes and ord(c) < 128 else c
                                      for i, c in enumerate(value))
    def unmask(value):
        if isinstance(value, str):
            return value.translate(_UNMASK)
        if isinstance(value, list):
            return [unmask(v) for v in value]
        if isinstance(value, dict):
            return {k: unmask(v) for k, v in value.items()}
        return value
    try:
        report = legacy_apply(manifest, memory)
    finally:
        for _, item in items:
            for field in fields:
                if field in item and isinstance(item[field], str):
                    item[field] = unmask(item[field])
    report = unmask(report)
    report['protected_cross_cue_addresses'] = [
        {'id': key, 'field': field} for key, field in sorted(protected)]
    return report


def deterministic_args(stage, work, memory_file=None, speaker_enroll=None, ass=None, styles=None):
    """Explicit legacy command settings for the fixed finish-chain operations."""
    from types import SimpleNamespace
    from qwen_asr.optimizer_bridge import DEFAULT_OPTIMIZER_ROOT
    common = dict(workdir=str(work), log_level='INFO', resume=False, force=False,
                  skip_preflight=False, performance_profile=False, performance_output='',
                  performance_sample_interval=0.2, dry_run_check=False)
    optimizer = str(DEFAULT_OPTIMIZER_ROOT)
    options = {
        'quality': dict(command='quality-gate', include_export=False, require_srt=False),
        'post_quality': dict(command='quality-gate', include_export=True, require_srt=True),
        'normalize': dict(command='normalize', source='mimo', optimizer_root=optimizer,
                          extend_ms=350, snap_gap_ms=200, min_blank_ms=300,
                          join_fragments=False, program_memory_file=str(memory_file) if memory_file else None),
        'export': dict(command='export', format='srt', source='normalized', export_mode='custom',
                       export_path=str(work / 'export/subtitles.srt'), media_path=str(work / 'audio.wav'),
                       max_subtitle_duration=6.0, min_subtitle_duration=1.0, max_chars_per_line_zh=18,
                       max_chars_per_line_en=42, max_lines=2, pause_split_seconds=0.8,
                       coarse_subtitles=False, optimizer_root=optimizer,
                       speaker_enroll=str(speaker_enroll) if speaker_enroll else None, speakers=None,
                       speaker_min_margin=0.1, speaker_model_path=None, speaker_device='auto'),
        'ass': dict(command='ass-quality', ass=str(ass), source='aligned', optimizer_root=optimizer,
                    include_styles=styles, exclude_style_prefixes='OP,ED', offset_ms=None,
                    window_ms=1200, diagnostic_window_ms=8000, low_score_threshold=0.45,
                    fail_score_threshold=0.2, max_cases=30, output=None, markdown_output=None),
    }
    return SimpleNamespace(**common, **options[stage])


def run_stage(stage, work, memory_file=None, speaker_enroll=None, ass_reference=None):
    """Execute an old command and report its return plus independently inspected artifacts."""
    work = Path(work).resolve()
    if PROJECT.resolve() not in work.parents:
        raise ValueError('work directory must be outside old repository and inside project')
    if str(OLD) not in sys.path:
        sys.path.insert(0, str(OLD))
    from qwen_asr.models import WorkPaths
    paths = WorkPaths.from_workdir(work)
    if stage == 'realign':
        parser, _ = _legacy_imports()
        args = parser().parse_args(['proofread-realign', '--proofread-realign-mfa-fallback', 'local',
                    '--proofread-realign-model', str(ALIGN_MODEL), '--model-cache-dir', str(work / 'model-cache'),
                    '--workdir', str(work), '--no-resume'])
        handler = args.func
    else:
        from qwen_asr.commands import cmd_quality_gate, cmd_normalize, cmd_export
        handler = {'quality': cmd_quality_gate, 'normalize': cmd_normalize, 'export': cmd_export}[stage]
        args = deterministic_args(stage, work, memory_file, speaker_enroll)
    with isolated_environment(work):
        if stage == 'realign':
            from qwen_asr.proofread_realign import _pending_items, _clear_punctuation_only_realign_flags
            manifest = json.loads(paths.mimo_proofread_manifest.read_text(encoding='utf-8'))
            _clear_punctuation_only_realign_flags(manifest)
            needs_gpu = bool(_pending_items(manifest))
            if needs_gpu:
                from filelock import FileLock
                lock = FileLock(str(GPU_LOCK), timeout=600)
                lock_record = work / 'reports/gpu-lock.json'
                try:
                    with lock:
                        lock_record.write_text(json.dumps({'pid': os.getpid(), 'path': str(GPU_LOCK), 'acquired': True, 'released': False}), encoding='utf-8')
                        with mfa_externalized(work):
                            code = args.func(args, paths)
                finally:
                    if lock_record.is_file():
                        lock_record.write_text(json.dumps({'pid': os.getpid(), 'path': str(GPU_LOCK), 'acquired': True, 'released': not lock.is_locked}), encoding='utf-8')
            else:
                code = args.func(args, paths)
            result_path = work / 'reports/proofread_realign.json'
        elif stage == 'normalize':
            import qwen_asr.commands.normalize as normalizer
            original = normalizer.apply_program_memory_if_requested
            def capture(args_, paths_):
                shutil.copyfile(paths_.normalized_manifest, work / 'reports/normalized-before-program-memory.json')
                if not memory_file or not json.loads(Path(memory_file).read_text(encoding='utf-8')).get('protect_cross_cue_addresses', False):
                    return original(args_, paths_)
                import qwen_asr.program_memory as pm
                legacy_apply = pm.apply_to_manifest
                pm.apply_to_manifest = lambda manifest, memory: apply_address_boundary_protection(manifest, memory, legacy_apply)
                try:
                    return original(args_, paths_)
                finally:
                    pm.apply_to_manifest = legacy_apply
            normalizer.apply_program_memory_if_requested = capture
            try:
                code = handler(args, paths)
            finally:
                normalizer.apply_program_memory_if_requested = original
            result_path = paths.normalized_manifest
        else:
            if stage == 'export' and speaker_enroll:
                from qwen_asr.speaker.voiceprint import SpeakerTrack, cache_key, load_enrollment
                key = cache_key(paths.audio_path, load_enrollment(speaker_enroll), None)
                if SpeakerTrack.load(paths.aligned_manifest.parent / 'speaker_scores.npz', key) is None:
                    raise RuntimeError('speaker cache mismatch; refusing unguarded model load')
            code = handler(args, paths)
            result_path = paths.final_quality_report if stage == 'quality' else work / 'export/subtitles.srt'
            if stage == 'export' and speaker_enroll and not (work / 'export/subtitles.ass').is_file():
                raise RuntimeError('speaker ASS not produced')
            if stage == 'export' and code == 0:
                # A reference must be supplied independently; never self-score the export.
                if ass_reference:
                    reference = Path(ass_reference).resolve()
                    if not reference.is_file():
                        raise ValueError('ASS reference missing')
                    styles = sorted({parts[3].strip() for line in reference.read_text(encoding='utf-8-sig').splitlines()
                                     if line.startswith('Dialogue:')
                                     for parts in [line.split(',', 9)] if len(parts) == 10 and parts[3].strip()})
                    if not styles:
                        raise ValueError('ASS reference has no dialogue styles')
                    from qwen_asr.ass_quality import cmd_ass_quality
                    ass_args = deterministic_args('ass', work, ass=reference, styles=','.join(styles))
                    ass_code = cmd_ass_quality(ass_args, paths)
                    if ass_code not in (0, 1):
                        raise RuntimeError('ASS quality command failed')
                from qwen_asr.commands import cmd_quality_gate
                quality_args = deterministic_args('post_quality', work)
                quality_code = cmd_quality_gate(quality_args, paths)
                if quality_code != 0:
                    code = quality_code
    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise RuntimeError(f'{stage} produced no business artifact')
    report = {'stage': stage, 'returncode': code, 'artifact': str(result_path)}
    if stage == 'export':
        quality_path = paths.final_quality_report
        if not quality_path.is_file():
            raise RuntimeError('post-export quality report missing')
        report['post_export_quality'] = json.loads(quality_path.read_text(encoding='utf-8'))
        report['ass_reference'] = 'provided' if ass_reference else 'not provided; ASS check may be skipped'
        report['legacy_ass_similarity'] = {'source': 'aligned', 'granularity': 'reference dialogue versus aligned segment',
                                           'purpose': 'existing production check; not final subtitle quality',
                                           'status': 'preserved' if ass_reference else 'NOT_ASSESSED'}
        if ass_reference:
            report['human_reference_diagnostic'] = human_reference_diagnostic(work, Path(ass_reference).resolve())
        else:
            report['human_reference_diagnostic'] = {'verdict': 'NOT_ASSESSED: no reference'}
    if stage in ('quality', 'realign'):
        report['business'] = json.loads(result_path.read_text(encoding='utf-8'))
    return report
