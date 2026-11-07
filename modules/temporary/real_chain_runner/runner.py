"""Candidate workflow core. Not a registered or production-ready host entry.

Stage adapters must be supplied through host calls; no legacy artifacts are inputs.
"""
import hashlib
import json
import os
import time
from pathlib import Path

from host.contracts import returned

STAGES = ('prepare', 'transcribe', 'correct', 'align', 'speaker', 'split',
          'translate', 'mimo', 'proofread_realign', 'quality', 'normalize', 'export')


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _save(path, data):
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def run(api, stages, inputs, output, resume=False, observer=None):
    """Run explicitly bound stages; adapters must return report.status=completed.

    inputs maps explicit file paths to their expected SHA-256. Stage definitions
    contain target, arguments and required output files. Only digests are saved;
    arguments and input paths are not copied into the checkpoint.
    """
    if tuple(s.get('name') for s in stages) != STAGES:
        raise ValueError('incomplete or reordered stage bindings')
    for path, expected in inputs.items():
        if not Path(path).is_file() or sha_file(path) != expected:
            raise ValueError('input digest mismatch')
    root = Path(output).resolve()
    checkpoint = root / 'checkpoint.json'
    signature = identity({'stages': stages, 'inputs': inputs})
    if resume:
        if not checkpoint.is_file():
            raise ValueError('resume checkpoint missing')
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        if state.get('signature') != signature:
            raise ValueError('resume identity mismatch')
        completed = state.get('completed')
        if not isinstance(completed, list) or len(completed) > len(STAGES):
            raise ValueError('invalid checkpoint')
        for i, item in enumerate(completed):
            if item.get('name') != STAGES[i]:
                raise ValueError('checkpoint stage order mismatch')
            declared = stages[i]['outputs']
            if set(item.get('outputs', {})) != set(declared):
                raise ValueError('checkpoint outputs mismatch')
            for rel, expected in item['outputs'].items():
                path = (root / rel).resolve()
                if not path.is_relative_to(root) or not path.is_file() or sha_file(path) != expected:
                    raise ValueError('checkpoint artifact mismatch')
    else:
        root.mkdir(parents=True, exist_ok=False)
        state = {'schema': 1, 'signature': signature, 'completed': []}
        _save(checkpoint, state)
    for stage in stages[len(state['completed']):]:
        name = stage['name']
        outputs = stage.get('outputs')
        if not isinstance(outputs, list) or not outputs or len(set(outputs)) != len(outputs):
            raise ValueError('stage requires distinct outputs')
        for rel in outputs:
            path = (root / rel).resolve()
            if not path.is_relative_to(root) or path == root:
                raise ValueError('output outside run directory')
        started = time.monotonic()
        try:
            report = returned(api.call(stage['target'], stage['arguments']))['report']
            if not isinstance(report, dict) or report.get('status') != 'completed':
                raise RuntimeError('stage did not complete: ' + name)
            hashes = {}
            for rel in outputs:
                path = (root / rel).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise RuntimeError('missing stage artifact: ' + name)
                hashes[rel] = sha_file(path)
            state['completed'].append({'name': name, 'outputs': hashes})
            _save(checkpoint, state)
        finally:
            if observer is not None:
                observer.stage(name, time.monotonic() - started)
    return {'status': 'completed', 'stages': len(state['completed'])}
