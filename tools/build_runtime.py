"""Generate verified local catalogs from portable templates and installed wheels.

No downloading or installation. Run after installing requirements-runtime.txt.
Generated module-registry files are machine-local and intentionally untracked.
"""
import hashlib
import importlib.metadata as metadata
import importlib.util
import platform
import subprocess
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Build-time use of the pure contract module; importing Host before catalogs exist
# would fail its mandatory bootstrap identity checks.
_spec = importlib.util.spec_from_file_location('_build_contracts', ROOT / 'host/contracts.py')
_contracts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contracts)
digest = _contracts.digest


def marker_environment():
    return {"python_version": ".".join(map(str, sys.version_info[:2])),
            "python_full_version": platform.python_version(),
            "sys_platform": sys.platform, "platform_system": platform.system(),
            "platform_python_implementation": platform.python_implementation(),
            "implementation_name": sys.implementation.name, "extra": ""}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    temporary.replace(path)


def build():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('The fixed-chain candidate requires Python 3.12')
    pack = []
    marker = marker_environment()
    for item in read(ROOT / 'runtime/dependencies.json'):
        dist = metadata.distribution(item['name'])
        if dist.version != item['version']:
            raise RuntimeError('Version mismatch for ' + item['name'] + ': expected ' + item['version'])
        files = [{'path': str(p).replace('\\', '/'), 'sha256': sha(Path(dist.locate_file(p)))}
                 for p in dist.files or []]
        if not files:
            raise RuntimeError('Empty distribution RECORD: ' + item['name'])
        pack.append({**item, 'files': files, 'marker_profile': marker})
    template_root = ROOT / 'runtime/registry-templates'
    templates = [(p.relative_to(template_root), read(p)) for p in sorted(template_root.rglob('*.json'))]
    # First validate every compatibility source before writing any output.
    for _, data in templates:
        spec = data.get('legacy_sources')
        if spec:
            if spec['root'] != 'compat/legacy':
                raise ValueError('Unexpected compatibility source root')
            for item in spec['files']:
                path = (ROOT / spec['root'] / item['path']).resolve()
                if not path.is_relative_to((ROOT / spec['root']).resolve()) or sha(path) != item['sha256']:
                    raise ValueError('Compatibility source identity mismatch: ' + item['path'])
    for relative, data in templates:
        if relative.name.startswith('contracts-'):
            if 'resource_pack' in data:
                data['resource_pack'] = pack
            write(ROOT / 'module-registry' / relative, data)
    for relative, data in templates:
        if relative.name.startswith('contracts-'):
            continue
        if 'implementations' in data:
            for record in data['implementations']:
                for item in record['content_files']:
                    path = (ROOT / item['path']).resolve()
                    if not path.is_relative_to(ROOT):
                        raise ValueError('Content escapes repository: ' + item['path'])
                    item['sha256'] = sha(path)
                record['content_digest'] = digest({k: v for k, v in record.items() if k != 'content_digest'})
            data['trusted_bootstrap']['sha256'] = sha(ROOT / data['trusted_bootstrap']['path'])
        write(ROOT / 'module-registry' / relative, data)
    subprocess.run([sys.executable, '-B', str(ROOT / 'tools/check_runtime.py')], cwd=ROOT, check=True)
    (ROOT / 'runs/experiments').mkdir(parents=True, exist_ok=True)
    print(json.dumps({'status': 'ok', 'distributions': len(pack), 'registry_files': len(templates),
                      'network_or_models_used': False}))


if __name__ == '__main__':
    build()
