"""Check every published catalog and the installed fixed-chain resource pack."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from host.contracts import digest
from host.runtime import capture_legacy_sources, verify_resource_pack


def check():
    records = files = 0
    for path in sorted((ROOT / 'module-registry').rglob('catalog-*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        bootstrap = data['trusted_bootstrap']
        assert hashlib.sha256((ROOT / bootstrap['path']).read_bytes()).hexdigest() == bootstrap['sha256'], path
        for record in data['implementations']:
            assert digest({k: v for k, v in record.items() if k != 'content_digest'}) == record['content_digest'], path
            records += 1
            for item in record['content_files']:
                assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
                files += 1
    profile = json.loads((ROOT / 'module-registry/real_chain_runner/contracts-real-chain-runner-312.json').read_text(encoding='utf-8'))
    assert profile.get('resource_pack'), 'resource pack is required'
    _, evidence = verify_resource_pack(profile['resource_pack'])
    spec = profile['legacy_sources']
    capture_legacy_sources({**spec, 'root': str(ROOT / spec['root'])})
    print(json.dumps({'catalog_records': records, 'content_file_records': files,
                      'verified_distributions': len(evidence), 'status': 'ok'}))


if __name__ == '__main__':
    check()
